// Native AscendC implementation of DGL's destination-normalized edge softmax,
// i.e. `dgl.ops.edge_softmax(g, logits, norm_by='dst')`, consumed by the
// SE3-Transformer attention path of RFdiffusion/RF2
// (`include/SE3Transformer/se3_transformer/model/layers/attention.py`).
//
// For every destination node d and every trailing feature/head column f the
// kernel computes exactly DGL 1.1.3's formulation:
//
//     m_d,f = max_{e : dst[e] == d} logits[e, f]
//     s_d,f = sum_{e : dst[e] == d} exp(logits[e, f] - m_d,f)
//     out[e, f] = exp(logits[e, f] - m_d,f) / s_d,f
//
// The stabilising maximum is taken per destination and per column - never
// graph-wide - so a destination whose logits are a thousand below another
// destination's still receives weights that sum to one.  There is no epsilon /
// tiny salting of the denominator and no clamping: for a non-empty destination
// at least one edge attains the per-column maximum, so the denominator is
// always >= exp(0) = 1.  Duplicate (parallel) edges are distinct CSR entries and
// are counted separately.  A destination with in-degree zero owns no edge rows
// and therefore produces no output.
//
// Input contract (all produced by the accepted native `edge_topology` kernel and
// bound with `SetGlobalBuffer` before first use, because an unbound
// GlobalTensor has neither a DDR base nor a size):
//
//     logits  float32 [E, W]  flattened edge logits (W = heads * channels)
//     indices int32   [E]     CSR edge ids, ascending inside one destination
//     offsets int64   [N+1]   CSR range of every destination
//     out     float32 [E, W]  per-edge normalized weights
//
// Parallelism / determinism: each AIV block owns a contiguous destination range
// (`numBlocks = min(vector_core_count, num_nodes)` on the host) and writes only
// the rows of its own destinations, so no atomics and no cross-core reduction
// are involved and the result is reproducible.  Inside one destination the
// edges are visited in the CSR's stable ascending edge-id order.
//
// Tiling: the feature width is walked in tiles of at most 2048 float32 elements;
// every GM transfer uses `DataCopyPad` with the exact byte length and the vector
// ops use that same element count, so an arbitrary - including
// non-32-byte-aligned - row width is safe.  UB buffers are sized to the
// 32-byte-rounded tile.
//
// The kernel is a three-pass per-destination algorithm (max, sum of exp, divide)
// because UB cannot hold a whole destination's edge rows.  Correctness first;
// this is not a performance claim.  Scalar GM reads of the CSR table are ordered
// against the following MTE2 row loads with the installed CANN `S_MTE2`
// SetFlag/WaitFlag contract, the vector -> MTE3 ordering is supplied by the
// VECOUT queue, exactly as in the accepted `segment_reduce_common.h` kernel.
//
// An inconsistent CSR index (negative or >= E) is not silently dropped and not
// clamped: its load is skipped (never an out-of-range DMA), the destination's
// exp-sum is poisoned with NaN and every valid row of that destination is
// written as NaN, so an invalid table fails the parity gate loudly.
//
// Only device vector intrinsics are used (`Adds`, `Max`, `Sub`, `Exp`, `Add`,
// `Div`, `Duplicate`, `DataCopyPad`).  There is no Torch/DGL/NumPy composition,
// no `scatter`/`index_add`, no CPU arithmetic and no per-shape or per-example
// branch.

#include "kernel_operator.h"

namespace ascend_graph_softmax {

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;

// float32 elements per 32-byte vector block.
constexpr int64_t kAlignElements = 8;
// Queue depth of the input/output UB stages.
constexpr int32_t kBufferNum = 2;
// Largest feature-width tile kept in UB (float32 elements).
constexpr int64_t kMaxTileElements = 2048;
// Loud poison value for an inconsistent CSR index.
constexpr float kNaN = __builtin_nanf("");

class KernelEdgeSoftmaxDst {
public:
    __aicore__ inline KernelEdgeSoftmaxDst() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR indices, GM_ADDR offsets,
                                GM_ADDR out, int64_t numEdges, int64_t numNodes,
                                int64_t width, int64_t numBlocks)
    {
        this->numEdges = numEdges;
        this->width = width;

        if (numBlocks <= 0) {
            numBlocks = 1;
        }
        const int64_t perSeg = (numNodes + numBlocks - 1) / numBlocks;
        this->s0 = static_cast<int64_t>(AscendC::GetBlockIdx()) * perSeg;
        int64_t s1 = this->s0 + perSeg;
        if (s1 > numNodes) {
            s1 = numNodes;
        }
        this->s1 = s1;
        if (this->s0 >= this->s1 || width <= 0) {
            return;
        }

        this->tile = width < kMaxTileElements ? width : kMaxTileElements;
        const uint32_t tileBytes = static_cast<uint32_t>(
            RoundUp(this->tile) * static_cast<int64_t>(sizeof(float)));

        logitsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(logits),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(width));
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices),
                                  static_cast<uint64_t>(numEdges));
        offsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(offsets),
                                  static_cast<uint64_t>(numNodes) + 1U);
        // The MTE3 store target must be bound here as well (an unbound
        // GlobalTensor addresses DDR outside the valid range).
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(width));

        pipe.InitBuffer(inQueue, kBufferNum, tileBytes);
        pipe.InitBuffer(outQueue, kBufferNum, tileBytes);
        pipe.InitBuffer(maxBuf, tileBytes);
        pipe.InitBuffer(sumBuf, tileBytes);
        pipe.InitBuffer(expBuf, tileBytes);
        pipe.InitBuffer(storeBuf, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->s0 >= this->s1) {
            return;
        }
        LocalTensor<float> maxAcc = maxBuf.Get<float>();
        LocalTensor<float> sumAcc = sumBuf.Get<float>();
        LocalTensor<float> expTile = expBuf.Get<float>();
        LocalTensor<float> storeTile = storeBuf.Get<float>();

        for (int64_t s = this->s0; s < this->s1; ++s) {
            const int64_t begin = offsetsGm.GetValue(static_cast<uint64_t>(s));
            const int64_t end =
                offsetsGm.GetValue(static_cast<uint64_t>(s) + 1U);
            if (end <= begin) {
                // Zero-in-degree destination: it owns no edge row, so there is
                // no output to produce (identical to DGL).
                continue;
            }

            for (int64_t tileStart = 0; tileStart < this->width;
                 tileStart += this->tile) {
                int64_t len = this->width - tileStart;
                if (len > this->tile) {
                    len = this->tile;
                }
                const int32_t count = static_cast<int32_t>(len);

                bool valid = true;

                // Pass 1: per-destination, per-column maximum.
                bool haveMaximum = false;
                for (int64_t p = begin; p < end; ++p) {
                    const int64_t e = ReadIndex(p);
                    if (!ValidEdge(e)) {
                        valid = false;
                        continue;
                    }
                    SyncStoMte2();
                    LocalTensor<float> row =
                        LoadRow(e * this->width + tileStart, len);
                    if (!haveMaximum) {
                        AscendC::Adds(maxAcc, row, 0.0f, count);
                        haveMaximum = true;
                    } else {
                        AscendC::Max(maxAcc, maxAcc, row, count);
                    }
                    inQueue.FreeTensor(row);
                }

                // Pass 2: per-destination, per-column sum of exp(logit - max).
                AscendC::Duplicate(sumAcc, 0.0f, count);
                for (int64_t p = begin; p < end; ++p) {
                    const int64_t e = ReadIndex(p);
                    if (!ValidEdge(e)) {
                        continue;
                    }
                    SyncStoMte2();
                    LocalTensor<float> row =
                        LoadRow(e * this->width + tileStart, len);
                    AscendC::Sub(expTile, row, maxAcc, count);
                    AscendC::Exp(expTile, expTile, count);
                    AscendC::Add(sumAcc, sumAcc, expTile, count);
                    inQueue.FreeTensor(row);
                }
                if (!valid) {
                    // Loud poison, never a clamp or a substituted denominator.
                    AscendC::Duplicate(sumAcc, kNaN, count);
                }

                // Pass 3: out[e, :] = exp(logit - max) / sum.  With a poisoned
                // denominator the valid rows of this destination become NaN.
                for (int64_t p = begin; p < end; ++p) {
                    const int64_t e = ReadIndex(p);
                    if (!ValidEdge(e)) {
                        continue;
                    }
                    SyncStoMte2();
                    LocalTensor<float> row =
                        LoadRow(e * this->width + tileStart, len);
                    AscendC::Sub(expTile, row, maxAcc, count);
                    AscendC::Exp(expTile, expTile, count);
                    AscendC::Div(storeTile, expTile, sumAcc, count);
                    inQueue.FreeTensor(row);
                    StoreOut(e * this->width + tileStart, storeTile, count);
                }
            }
        }
    }

private:
    __aicore__ inline static int64_t RoundUp(int64_t value)
    {
        int64_t aligned =
            ((value + kAlignElements - 1) / kAlignElements) * kAlignElements;
        if (aligned < kAlignElements) {
            aligned = kAlignElements;
        }
        return aligned;
    }

    __aicore__ inline int64_t ReadIndex(int64_t position) const
    {
        return static_cast<int64_t>(
            indicesGm.GetValue(static_cast<uint64_t>(position)));
    }

    __aicore__ inline bool ValidEdge(int64_t e) const
    {
        return e >= 0 && e < this->numEdges;
    }

    __aicore__ inline LocalTensor<float> LoadRow(int64_t gmOffset, int64_t len)
    {
        LocalTensor<float> row = inQueue.AllocTensor<float>();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(row, logitsGm[static_cast<uint64_t>(gmOffset)], params,
                    padParams);
        inQueue.EnQue(row);
        return inQueue.DeQue<float>();
    }

    __aicore__ inline void StoreOut(int64_t gmOffset,
                                    LocalTensor<float> &value, int32_t count)
    {
        // The vector-produced tile is staged through the VECOUT queue, whose
        // EnQue/DeQue supplies the V -> MTE3 ordering.
        LocalTensor<float> y = outQueue.AllocTensor<float>();
        AscendC::Adds(y, value, 0.0f, count);
        outQueue.EnQue(y);
        y = outQueue.DeQue<float>();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(count * static_cast<int32_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPad(outGm[static_cast<uint64_t>(gmOffset)], y, params);
        outQueue.FreeTensor(y);
    }

    // Installed CANN SetFlag/WaitFlag contract: a scalar GM read must be ordered
    // against the MTE2 row load that follows it.
    __aicore__ inline void SyncStoMte2()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(id);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(id);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, kBufferNum> inQueue;
    TQue<TPosition::VECOUT, kBufferNum> outQueue;
    TBuf<TPosition::VECCALC> maxBuf;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> expBuf;
    TBuf<TPosition::VECCALC> storeBuf;
    GlobalTensor<float> logitsGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<int64_t> offsetsGm;
    GlobalTensor<float> outGm;
    int64_t numEdges = 0;
    int64_t width = 0;
    int64_t tile = 0;
    int64_t s0 = 0;
    int64_t s1 = 0;
};

}  // namespace ascend_graph_softmax

// Entry point consumed by the host bridge; the generated host stub exports
// `aclrtlaunch_edge_softmax_dst`.
extern "C" __global__ __aicore__ void edge_softmax_dst(
    GM_ADDR logits, GM_ADDR indices, GM_ADDR offsets, GM_ADDR out,
    int64_t numEdges, int64_t numNodes, int64_t width, int64_t numBlocks)
{
    ascend_graph_softmax::KernelEdgeSoftmaxDst op;
    op.Init(logits, indices, offsets, out, numEdges, numNodes, width, numBlocks);
    op.Process();
}
