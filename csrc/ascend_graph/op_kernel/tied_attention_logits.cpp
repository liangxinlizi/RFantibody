// Native AscendC CUDA-order FP32 contraction for the tied attention of the
// RFdiffusion reverse path (``BiasedAxialAttention``):
//
//   out[b, h, i, j] = SUM_{k = 0..K-1} SUM_{n = 0..N-1}
//                         fma_f32(query[b, n, h, k, i], key[b, n, h, k, j], acc)
//
// with ``k`` as the outer and ``n`` as the inner loop variable (ascending) and
// every accumulation step a single correctly rounded float32 FMA, starting from
// ``+0.0f``.  That is the reduction order and rounding the recording CUDA
// reference exhibits.  The arithmetic primitives live
// in ``compensated_fma_ordered.h`` (compensated product + two TwoSums +
// vectorized round-to-odd).  There is no ``MulAddDst``/``FusedMulAdd`` on the
// path (both were measured to round the product before the add on this SoC) and
// no Torch / NumPy / CPU computation or fallback anywhere: the contraction is
// computed only here.
//
// Layouts (contiguous float32, produced by the Python adapter):
//   queryT [B, N, H, K, I]   keyT [B, N, H, K, J]   out [B, H, I, J]
//
// Work partition: the output rows ``(b, h, iTileIndex)`` form a flat unit list;
// block ``b`` owns one contiguous range of that list, so the split is disjoint,
// deterministic and free of atomics and of cross-core reductions.  ``iTile`` is
// the fixed number of output rows one unit accumulates at once: the key operand
// of a ``(k, n)`` step is then split once and reused by all ``iTile`` rows, and
// each row keeps its own accumulator so the per-output ``(k, n)`` order is
// exactly the reference order.
//
// Tiling: the output column axis is walked in tiles of at most ``kJTileLimit``
// lanes and the ``n`` axis of the query staging in chunks of at most
// ``kQChunkRows`` rows, so an arbitrary (including non-32-byte-aligned) ``I``,
// ``J``, ``K`` and ``N`` is safe: every global transfer uses ``DataCopyPad`` with
// the exact byte length and every vector call uses the same element count.  The
// UB buffers are sized from those two runtime minima, never from a shape of a
// particular model, case or example.
//
// Alignment invariant of every GM -> UB transfer of this kernel: the installed
// ``DataCopyPad`` GM -> UB contract (``DataCopyPadGm2UBImpl`` in
// ``impl/dav_c220/kernel_operator_data_copy_impl.h``) requires the destination
// UB address to be 32-byte aligned (``ONE_BLK_SIZE == 32``).  Every such
// destination of this kernel is therefore 32-byte aligned by construction: each
// staged query row starts at a multiple of ``kStageAlignElements`` float32
// (``qStageStride``, a whole 32-byte block, not the raw ``kITile`` row length),
// the query staging buffer itself is allocated in whole blocks from that stride,
// and the key-row queue buffers are sized from ``laneStride``, which is a
// multiple of ``kAlignElements`` float32.  The UB -> GM store side is equally
// block aligned (``acc`` rows at ``row * laneStride``).
//
// Every GlobalTensor the kernel touches is bound with ``SetGlobalBuffer`` in
// ``Init`` before first use (an unbound GlobalTensor has neither a DDR base nor
// a size).  The scalar reads of the staged query block are ordered against the
// MTE2 stage with the installed SetFlag/WaitFlag contract, and the vector ->
// MTE3 store ordering is explicit so an accumulator row is stored bit-exactly
// (no ``+0.0f`` copy that would rewrite a ``-0.0`` result).

#include "kernel_operator.h"

#include "compensated_fma_ordered.h"

namespace {

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

using ascend_graph_ordered::kFloatScratchArrays;
using ascend_graph_ordered::kIntConstantArrays;
using ascend_graph_ordered::kIntScratchArrays;
using ascend_graph_ordered::OrderedFma;

// float32 elements per 32-byte vector block.
constexpr int64_t kAlignElements = 8;
// Output rows accumulated at once by one work unit (mirrored by the host bridge
// when it counts the units).
constexpr int32_t kITile = 4;
// One staged query row occupies a whole 32-byte vector block.  The installed
// DataCopyPad GM -> UB contract requires a 32-byte aligned destination
// (ONE_BLK_SIZE == 32), and a stride of kITile == 4 float32 (16 bytes) would
// leave every odd staged row unaligned.
constexpr int32_t kStageAlignElements = 8;   // 8 float32 == 32 bytes
// Maximum output columns (lanes) of one UB tile.
constexpr int32_t kJTileLimit = 512;
// Maximum number of `n` rows of the query operand staged per MTE2 batch.
constexpr int32_t kQChunkRows = 64;
// Queue depth of the key-row stage.
constexpr int32_t kBufferNum = 2;

class KernelTiedAttentionLogits {
public:
    __aicore__ inline KernelTiedAttentionLogits() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR out,
                                int64_t B, int64_t N, int64_t I, int64_t J,
                                int64_t H, int64_t K, int64_t numBlocks)
    {
        this->numN = N;
        this->dimI = I;
        this->dimJ = J;
        this->numH = H;
        this->numK = K;

        const int64_t iTiles = (I + kITile - 1) / kITile;
        const int64_t totalUnits = B * H * iTiles;
        if (numBlocks <= 0) {
            numBlocks = 1;
        }
        const int64_t perCore = (totalUnits + numBlocks - 1) / numBlocks;
        this->u0 = static_cast<int64_t>(AscendC::GetBlockIdx()) * perCore;
        int64_t u1 = this->u0 + perCore;
        if (u1 > totalUnits) {
            u1 = totalUnits;
        }
        this->u1 = u1;
        this->iTiles = iTiles;
        if (this->u0 >= this->u1 || I <= 0 || J <= 0 || N <= 0 || K <= 0) {
            return;
        }

        this->jTile = J < static_cast<int64_t>(kJTileLimit)
                          ? static_cast<int32_t>(J)
                          : kJTileLimit;
        this->laneStride = static_cast<int32_t>(RoundUp(this->jTile));
        this->qChunkRows = N < static_cast<int64_t>(kQChunkRows)
                               ? static_cast<int32_t>(N)
                               : kQChunkRows;
        this->qStageStride = RoundUp(kITile) < kStageAlignElements
                                 ? kStageAlignElements
                                 : static_cast<int32_t>(RoundUp(kITile));

        queryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(query),
            static_cast<uint64_t>(B) * static_cast<uint64_t>(N) *
                static_cast<uint64_t>(H) * static_cast<uint64_t>(K) *
                static_cast<uint64_t>(I));
        keyGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(key),
            static_cast<uint64_t>(B) * static_cast<uint64_t>(N) *
                static_cast<uint64_t>(H) * static_cast<uint64_t>(K) *
                static_cast<uint64_t>(J));
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(B) * static_cast<uint64_t>(H) *
                static_cast<uint64_t>(I) * static_cast<uint64_t>(J));

        const uint32_t laneBytes = static_cast<uint32_t>(
            this->laneStride * static_cast<int32_t>(sizeof(float)));
        const uint32_t floatScratchBytes =
            static_cast<uint32_t>(kFloatScratchArrays) * laneBytes;
        const uint32_t intScratchBytes =
            static_cast<uint32_t>(kIntScratchArrays) * laneBytes;
        const uint32_t intConstantBytes =
            static_cast<uint32_t>(kIntConstantArrays) * laneBytes;
        const uint32_t accBytes =
            static_cast<uint32_t>(kITile) * laneBytes;
        const uint32_t qStageBytes = static_cast<uint32_t>(
            this->qChunkRows * this->qStageStride *
            static_cast<int32_t>(sizeof(float)));

        pipe.InitBuffer(accBuf, accBytes);
        pipe.InitBuffer(floatScratchBuf, floatScratchBytes);
        pipe.InitBuffer(intScratchBuf, intScratchBytes);
        pipe.InitBuffer(intConstantBuf, intConstantBytes);
        pipe.InitBuffer(qStageBuf, qStageBytes);
        pipe.InitBuffer(keyInQueue, kBufferNum, laneBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->u0 >= this->u1) {
            return;
        }
        LocalTensor<float> acc = accBuf.Get<float>();
        LocalTensor<float> qStage = qStageBuf.Get<float>();
        this->fma.Init(floatScratchBuf.Get<float>(),
                       intScratchBuf.Get<int32_t>(),
                       intConstantBuf.Get<int32_t>(), this->laneStride);

        for (int64_t unit = this->u0; unit < this->u1; ++unit) {
            const int64_t perBatch = this->numH * this->iTiles;
            const int64_t batch = unit / perBatch;
            const int64_t remainder = unit - batch * perBatch;
            const int64_t head = remainder / this->iTiles;
            const int64_t iTileIndex = remainder - head * this->iTiles;
            const int64_t i0 = iTileIndex * kITile;
            const int64_t iRows = Minimal(static_cast<int64_t>(kITile),
                                          this->dimI - i0);

            for (int64_t j0 = 0; j0 < this->dimJ; j0 += this->jTile) {
                const int64_t jCount =
                    Minimal(static_cast<int64_t>(this->jTile), this->dimJ - j0);
                const int32_t count = static_cast<int32_t>(jCount);
                this->AccumulateTile(batch, head, i0, iRows, j0, count, acc,
                                     qStage);
                this->StoreTile(batch, head, i0, iRows, j0, count, acc);
            }
        }
    }

private:
    __aicore__ inline static int64_t Minimal(int64_t lhs, int64_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline static int64_t RoundUp(int64_t value)
    {
        int64_t aligned = ((value + kAlignElements - 1) / kAlignElements) *
                          kAlignElements;
        if (aligned < kAlignElements) {
            aligned = kAlignElements;
        }
        return aligned;
    }

    // Ordered accumulation of one (b, h, i-tile, j-tile) output block:
    //   k outer, n inner, ascending, one correctly rounded FMA per step.
    __aicore__ inline void AccumulateTile(int64_t batch, int64_t head,
                                          int64_t i0, int64_t iRows, int64_t j0,
                                          int32_t count, LocalTensor<float> &acc,
                                          LocalTensor<float> &qStage)
    {
        float aValues[kITile];
        for (int64_t row = 0; row < iRows; ++row) {
            AscendC::Duplicate(acc[static_cast<uint32_t>(row * this->laneStride)],
                               0.0f, count);
        }

        for (int64_t k = 0; k < this->numK; ++k) {
            for (int64_t nStart = 0; nStart < this->numN;
                 nStart += this->qChunkRows) {
                const int64_t nEnd = Minimal(
                    nStart + static_cast<int64_t>(this->qChunkRows), this->numN);
                this->StageQuery(batch, head, k, i0, iRows, nStart, nEnd,
                                 qStage);
                // The staged query block is read by the scalar unit.
                SyncMte2ToScalar();
                for (int64_t n = nStart; n < nEnd; ++n) {
                    LocalTensor<float> bRow =
                        this->LoadKeyRow(batch, head, n, k, j0, count);
                    this->fma.SplitMultiplicand(bRow, count);
                    // Read every staged query scalar of this ``(k, n)`` step
                    // first; the single scalar -> vector flag below then orders
                    // all of those reads against the vector steps that consume
                    // them.
                    const int64_t staged =
                        (n - nStart) * static_cast<int64_t>(this->qStageStride);
                    for (int64_t row = 0; row < iRows; ++row) {
                        aValues[row] = qStage.GetValue(
                            static_cast<uint32_t>(staged + row));
                    }
                    SyncScalarToVector();
                    for (int64_t row = 0; row < iRows; ++row) {
                        this->fma.Step(
                            aValues[row], bRow,
                            acc[static_cast<uint32_t>(row * this->laneStride)],
                            count);
                    }
                    this->keyInQueue.FreeTensor(bRow);
                }
                // The next MTE2 batch must not overwrite the staged query block
                // while the scalar unit is still reading it.
                SyncScalarToMte2();
            }
        }
    }

    // Stage `query[b, n, h, k, i0 .. i0+iRows)` for one `k` and one `n` chunk.
    __aicore__ inline void StageQuery(int64_t batch, int64_t head, int64_t k,
                                      int64_t i0, int64_t iRows, int64_t nStart,
                                      int64_t nEnd, LocalTensor<float> &qStage)
    {
        const int64_t headStride = this->numK * this->dimI;
        for (int64_t n = nStart; n < nEnd; ++n) {
            const int64_t offset =
                ((batch * this->numN + n) * this->numH + head) * headStride +
                k * this->dimI + i0;
            DataCopyExtParams params{
                1, static_cast<uint32_t>(iRows * static_cast<int64_t>(
                                                sizeof(float))),
                0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(qStage[static_cast<uint32_t>((n - nStart) *
                                                     this->qStageStride)],
                        queryGm[static_cast<uint64_t>(offset)], params,
                        padParams);
        }
    }

    __aicore__ inline LocalTensor<float> LoadKeyRow(int64_t batch, int64_t head,
                                                    int64_t n, int64_t k,
                                                    int64_t j0, int32_t count)
    {
        LocalTensor<float> row = this->keyInQueue.AllocTensor<float>();
        const int64_t offset =
            ((batch * this->numN + n) * this->numH + head) *
                (this->numK * this->dimJ) +
            k * this->dimJ + j0;
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(row, keyGm[static_cast<uint64_t>(offset)], params,
                    padParams);
        this->keyInQueue.EnQue(row);
        return this->keyInQueue.DeQue<float>();
    }

    // Store the tiled accumulator rows bit-exactly (no arithmetic on the way
    // out, so a `-0.0` result keeps its sign).
    __aicore__ inline void StoreTile(int64_t batch, int64_t head, int64_t i0,
                                     int64_t iRows, int64_t j0, int32_t count,
                                     LocalTensor<float> &acc)
    {
        SyncVectorToMte3();
        for (int64_t row = 0; row < iRows; ++row) {
            const int64_t outRow =
                (batch * this->numH + head) * this->dimI + i0 + row;
            const int64_t offset = outRow * this->dimJ + j0;
            DataCopyExtParams params{
                1,
                static_cast<uint32_t>(count *
                                      static_cast<int32_t>(sizeof(float))),
                0, 0, 0};
            DataCopyPad(outGm[static_cast<uint64_t>(offset)],
                        acc[static_cast<uint32_t>(row * this->laneStride)],
                        params);
        }
        // The next tile rewrites the accumulator rows in place.
        SyncMte3ToVector();
    }

    // Installed CANN SetFlag/WaitFlag contracts (`GetTPipePtr()` is the global
    // free function of this toolkit).
    __aicore__ inline void SyncMte2ToScalar()
    {
        auto id = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
        AscendC::SetFlag<HardEvent::MTE2_S>(id);
        AscendC::WaitFlag<HardEvent::MTE2_S>(id);
    }
    __aicore__ inline void SyncScalarToMte2()
    {
        auto id = GetTPipePtr()->FetchEventID(HardEvent::S_MTE2);
        AscendC::SetFlag<HardEvent::S_MTE2>(id);
        AscendC::WaitFlag<HardEvent::S_MTE2>(id);
    }
    __aicore__ inline void SyncScalarToVector()
    {
        auto id = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        AscendC::SetFlag<HardEvent::S_V>(id);
        AscendC::WaitFlag<HardEvent::S_V>(id);
    }
    __aicore__ inline void SyncVectorToMte3()
    {
        auto id = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        AscendC::SetFlag<HardEvent::V_MTE3>(id);
        AscendC::WaitFlag<HardEvent::V_MTE3>(id);
    }
    __aicore__ inline void SyncMte3ToVector()
    {
        auto id = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
        AscendC::SetFlag<HardEvent::MTE3_V>(id);
        AscendC::WaitFlag<HardEvent::MTE3_V>(id);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, kBufferNum> keyInQueue;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> floatScratchBuf;
    TBuf<TPosition::VECCALC> intScratchBuf;
    TBuf<TPosition::VECCALC> intConstantBuf;
    // The staging buffer carries the position the installed GM -> UB
    // ``DataCopyPad`` contract names (``VECIN``), like every accepted kernel's
    // queue-based MTE2 destination.
    TBuf<TPosition::VECIN> qStageBuf;
    GlobalTensor<float> queryGm;
    GlobalTensor<float> keyGm;
    GlobalTensor<float> outGm;
    OrderedFma fma;
    int64_t numN = 0;
    int64_t dimI = 0;
    int64_t dimJ = 0;
    int64_t numH = 0;
    int64_t numK = 0;
    int64_t iTiles = 0;
    int64_t u0 = 0;
    int64_t u1 = 0;
    int32_t jTile = 0;
    int32_t laneStride = 0;
    int32_t qChunkRows = 0;
    int32_t qStageStride = 0;
};

}  // namespace

// Entry point consumed by the host bridge (`aclrtlaunch_tied_attention_logits`).
extern "C" __global__ __aicore__ void tied_attention_logits(
    GM_ADDR queryT, GM_ADDR keyT, GM_ADDR out, int64_t B, int64_t N, int64_t I,
    int64_t J, int64_t H, int64_t K, int64_t numBlocks)
{
    KernelTiedAttentionLogits op;
    op.Init(queryT, keyT, out, B, N, I, J, H, K, numBlocks);
    op.Process();
}
