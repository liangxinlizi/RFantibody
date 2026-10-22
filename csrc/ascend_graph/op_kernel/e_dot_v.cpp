// Native AscendC implementation of the DGL `e_dot_v` graph operator used by the
// SE3-Transformer attention path of RFdiffusion/RF2:
//
//   e_dot_v(g, key, query)[e, h, 0]
//       = sum_{j < numFeat} key[e, h, j] * query[dst[e], h, j]
//
// The host bridge flattens the trailing head/feature dimensions, so the kernel
// receives `key` as contiguous `[E, H*numFeat]` float32, `query` as
// `[N, H*numFeat]` float32, `dst` as `int64 [E]` and writes `out` as
// `[E, H]` float32.  The Python wrapper restores DGL's output shape
// `key.shape[:-1] + (1,)`.
//
// There is no destination aggregation here: every edge writes exactly its own
// output row, so no adjacency table, no atomics and no cross-core reduction are
// needed.  Each AIV core owns a contiguous edge range and the result is
// deterministic.
//
// Every global tensor the kernel touches (`key`, `query`, `dst`, `out`) is bound
// in `Init` with `SetGlobalBuffer` before first use: an unbound GlobalTensor has
// neither a DDR base nor a size, so any DMA on it addresses memory outside the
// valid range.
//
// All GM traffic goes through `DataCopyPad`, so an arbitrary (un-aligned)
// feature width is safe.  The scalar reads of the DMA-staged destination index
// and of the vector-produced product row are ordered against the other pipes
// with the installed CANN SetFlag/WaitFlag contract.  An out-of-range `dst`
// publishes a loud NaN row (never a clamp) so an invalid graph index fails the
// parity gate instead of silently changing the reduction.  There is no Torch /
// NumPy / CPU fallback and no per-shape or per-example branch.

#include "kernel_operator.h"

namespace {

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

// Loud poison value for an invalid destination index.
constexpr float kNaN = __builtin_nanf("");
// float32 elements per 32-byte vector block.
constexpr int64_t kAlignElements = 8;
constexpr int32_t kBufferNum = 2;

class KernelEdgeDotV {
public:
    __aicore__ inline KernelEdgeDotV() {}

    __aicore__ inline void Init(GM_ADDR key, GM_ADDR query, GM_ADDR dst,
                                GM_ADDR out, int64_t numEdges, int64_t numHeads,
                                int64_t numFeat, int64_t numNodes,
                                int64_t numBlocks)
    {
        this->numHeads = numHeads;
        this->numFeat = numFeat;
        this->numNodes = numNodes;
        this->width = numHeads * numFeat;

        // UB row widths rounded up to whole 32-byte float32 blocks.
        this->featAlign = RoundUp(this->width);
        this->outAlign = RoundUp(numHeads);

        if (numBlocks <= 0) {
            numBlocks = 1;
        }
        const int64_t perCore = (numEdges + numBlocks - 1) / numBlocks;
        this->e0 = static_cast<int64_t>(AscendC::GetBlockIdx()) * perCore;
        int64_t e1 = this->e0 + perCore;
        if (e1 > numEdges) {
            e1 = numEdges;
        }
        this->e1 = e1;
        if (this->e0 >= this->e1) {
            return;
        }

        keyGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(key),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(this->width));
        queryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(query),
            static_cast<uint64_t>(numNodes) * static_cast<uint64_t>(this->width));
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(dst),
                              static_cast<uint64_t>(numEdges));
        // The MTE3 store target must be bound here: an unbound GlobalTensor
        // makes StoreOut address outside DDR (observed as AIV retCode 0x31,
        // "The DDR address of the MTE instruction is out of range").
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numHeads));

        const uint32_t rowBufferBytes =
            static_cast<uint32_t>(this->featAlign * static_cast<int64_t>(sizeof(float)));
        const uint32_t outBufferBytes =
            static_cast<uint32_t>(this->outAlign * static_cast<int64_t>(sizeof(float)));
        pipe.InitBuffer(keyInQueue, kBufferNum, rowBufferBytes);
        pipe.InitBuffer(qInQueue, kBufferNum, rowBufferBytes);
        pipe.InitBuffer(prodBuf, rowBufferBytes);
        pipe.InitBuffer(outBuf, outBufferBytes);
        // One int64 destination index (8 bytes) staged into a 32-byte block.
        pipe.InitBuffer(dstBuf, 32);
    }

    __aicore__ inline void Process()
    {
        if (this->e0 >= this->e1) {
            return;
        }
        LocalTensor<float> prod = prodBuf.Get<float>();
        LocalTensor<float> outLocal = outBuf.Get<float>();
        LocalTensor<int64_t> dstLocal = dstBuf.Get<int64_t>();
        const int32_t rowWidth = static_cast<int32_t>(this->width);
        const int32_t outWidth = static_cast<int32_t>(this->numHeads);

        for (int64_t e = this->e0; e < this->e1; ++e) {
            // Stage dst[e] and read it as a UB scalar.  The scalar read must
            // complete before the next edge's DMA overwrites `dstLocal`.
            DataCopyExtParams dstParams{1,
                                        static_cast<uint32_t>(sizeof(int64_t)),
                                        0, 0, 0};
            DataCopyPadExtParams<int64_t> dstPad{false, 0, 0, 0};
            DataCopyPad(dstLocal, dstGm[static_cast<uint64_t>(e)], dstParams,
                        dstPad);
            SyncMte2toS();
            const int64_t d = dstLocal.GetValue(0);
            SyncStoMte2();

            if (d < 0 || d >= this->numNodes) {
                // Invalid graph index: publish a loud NaN row so the parity
                // gate fails instead of reading out of bounds.
                for (int64_t h = 0; h < this->numHeads; ++h) {
                    outLocal.SetValue(static_cast<uint32_t>(h), kNaN);
                }
                StoreOut(e, outLocal, outWidth);
                continue;
            }

            LocalTensor<float> k = LoadRow(keyGm, keyInQueue, e);
            LocalTensor<float> q = LoadRow(queryGm, qInQueue, d);
            AscendC::Mul(prod, k, q, rowWidth);
            keyInQueue.FreeTensor(k);
            qInQueue.FreeTensor(q);

            // Vector product -> scalar reads.
            SyncVtoS();
            for (int64_t h = 0; h < this->numHeads; ++h) {
                float acc = 0.0f;
                const int64_t base = h * this->numFeat;
                for (int64_t j = 0; j < this->numFeat; ++j) {
                    acc += prod.GetValue(static_cast<uint32_t>(base + j));
                }
                outLocal.SetValue(static_cast<uint32_t>(h), acc);
            }
            // Both the scalar reads of `prod` and the scalar writes of
            // `outLocal` must be complete before the vector unit reuses them.
            SyncStoV();

            StoreOut(e, outLocal, outWidth);
        }
    }

private:
    __aicore__ inline static int64_t RoundUp(int64_t value)
    {
        int64_t aligned = ((value + kAlignElements - 1) / kAlignElements) *
                          kAlignElements;
        if (aligned < kAlignElements) {
            aligned = kAlignElements;
        }
        return aligned;
    }

    __aicore__ inline LocalTensor<float> LoadRow(
        GlobalTensor<float> &gm, TQue<TPosition::VECIN, kBufferNum> &queue,
        int64_t row)
    {
        LocalTensor<float> x = queue.AllocTensor<float>();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(this->width *
                                  static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(
            x, gm[static_cast<uint64_t>(row) * static_cast<uint64_t>(this->width)],
            params, padParams);
        queue.EnQue(x);
        return queue.DeQue<float>();
    }

    __aicore__ inline void StoreOut(int64_t edge, LocalTensor<float> &y,
                                    int32_t outWidth)
    {
        // Scalar writes to `y` must be visible before the MTE3 read.
        SyncStoMte3();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(outWidth * static_cast<int32_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPad(outGm[static_cast<uint64_t>(edge) *
                          static_cast<uint64_t>(this->numHeads)],
                    y, params);
        // The DMA read of `y` must finish before the next edge's scalar writes
        // reuse the buffer.
        SyncMte3toS();
    }

    // Installed CANN SetFlag/WaitFlag contracts (`GetTPipePtr()` is the global
    // free function of this toolkit).
    __aicore__ inline void SyncMte2toS()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(id);
    }
    __aicore__ inline void SyncStoMte2()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(id);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(id);
    }
    __aicore__ inline void SyncVtoS()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(id);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(id);
    }
    __aicore__ inline void SyncStoV()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(id);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(id);
    }
    __aicore__ inline void SyncStoMte3()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(id);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(id);
    }
    __aicore__ inline void SyncMte3toS()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(id);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, kBufferNum> keyInQueue;
    TQue<TPosition::VECIN, kBufferNum> qInQueue;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> outBuf;
    TBuf<TPosition::VECCALC> dstBuf;
    GlobalTensor<float> keyGm;
    GlobalTensor<float> queryGm;
    GlobalTensor<int64_t> dstGm;
    GlobalTensor<float> outGm;
    int64_t numHeads = 0;
    int64_t numFeat = 0;
    int64_t numNodes = 0;
    int64_t width = 0;
    int64_t featAlign = 0;
    int64_t outAlign = 0;
    int64_t e0 = 0;
    int64_t e1 = 0;
};

}  // namespace

// Entry point consumed by the host bridge.  `numHeads` is the number of
// trailing feature groups (one dot product per group) and `numFeat` is the
// per-group reduced width, so the flattened row width is `numHeads * numFeat`.
extern "C" __global__ __aicore__ void edge_dot_v(
    GM_ADDR key, GM_ADDR query, GM_ADDR dst, GM_ADDR out, int64_t numEdges,
    int64_t numHeads, int64_t numFeat, int64_t numNodes, int64_t numBlocks)
{
    KernelEdgeDotV op;
    op.Init(key, query, dst, out, numEdges, numHeads, numFeat, numNodes,
            numBlocks);
    op.Process();
}
