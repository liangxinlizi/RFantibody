// Native AscendC segment reduction over a native CSR edge table.
//
// This header owns the single reduction kernel body shared by the two DGL
// destination-edge operators consumed by the RFantibody model path:
//
//   copy_e_sum(g, feat)[s, :]  = sum_{e in in_edges(s)} feat[e, :]
//   copy_e_mean(g, feat)[s, :] = copy_e_sum[s, :] / max(in_degree(s), 1)
//
// The CSR table (`offsets` int64 [N+1], `indices` int32 [E]) is produced by the
// native `edge_topology` kernel (`op_kernel/edge_topology.cpp`); the order of
// the edges inside one destination is the stable ascending original edge id, so
// the float32 accumulation order is deterministic and independent of AIV
// scheduling.  There are no atomics, no Torch/DGL primitives and no CPU path.
//
// DGL 1.1.3 semantics (`dgl/ops/spmm.py`): `copy_e_mean` divides the per
// destination sum by `clamp(in_degrees, 1, max(num_edges, 1))`, which for a
// zero-in-degree destination yields exactly zero.  The kernel therefore uses a
// denominator of `max(deg, 1)` and writes zeros for empty destinations.
//
// Tiling: each AIV block owns a contiguous destination range
// (`numBlocks = min(vector_core_count, num_nodes)` on the host).  Within a
// destination the feature width is walked in tiles, so an arbitrary width -
// including a non-32-byte-aligned one - is safe: every GM transfer uses
// `DataCopyPad` with the exact byte length and the vector ops use that same
// element count.  The UB buffers are sized to the 32-byte-rounded tile so the
// pad-capable copy never overflows the allocation.
//
// Every GlobalTensor touched by the kernel is bound with `SetGlobalBuffer` in
// `Init` before first use (an unbound GlobalTensor has neither a DDR base nor a
// size).  Scalar GM reads (CSR offsets and edge indices) are ordered against the
// following MTE2 row loads with the same SetFlag/WaitFlag contract used by the
// accepted `edge_dot_v` kernel.  An out-of-range edge index poisons the whole
// destination row with NaN instead of silently dropping or clamping an edge.

#ifndef ASCEND_GRAPH_OP_KERNEL_SEGMENT_REDUCE_COMMON_H
#define ASCEND_GRAPH_OP_KERNEL_SEGMENT_REDUCE_COMMON_H

#include "kernel_operator.h"

namespace ascend_graph_segment {

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

// `kMean == false` -> copy_e_sum, `kMean == true` -> copy_e_mean.
template <bool kMean>
class KernelEdgeSegmentReduce {
public:
    __aicore__ inline KernelEdgeSegmentReduce() {}

    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR indices, GM_ADDR offsets,
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

        featGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(feat),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(width));
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices),
                                  static_cast<uint64_t>(numEdges));
        offsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(offsets),
                                  static_cast<uint64_t>(numNodes) + 1U);
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numNodes) * static_cast<uint64_t>(width));

        pipe.InitBuffer(inQueue, kBufferNum, tileBytes);
        pipe.InitBuffer(outQueue, kBufferNum, tileBytes);
        pipe.InitBuffer(accBuf, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->s0 >= this->s1) {
            return;
        }
        LocalTensor<float> acc = accBuf.Get<float>();

        for (int64_t s = this->s0; s < this->s1; ++s) {
            const int64_t begin = offsetsGm.GetValue(static_cast<uint64_t>(s));
            const int64_t end =
                offsetsGm.GetValue(static_cast<uint64_t>(s) + 1U);
            const int64_t deg = end - begin;

            for (int64_t tileStart = 0; tileStart < this->width;
                 tileStart += this->tile) {
                int64_t len = this->width - tileStart;
                if (len > this->tile) {
                    len = this->tile;
                }
                const int32_t count = static_cast<int32_t>(len);

                AscendC::Duplicate(acc, 0.0f, count);
                bool valid = true;
                for (int64_t p = begin; p < end; ++p) {
                    const int64_t e = static_cast<int64_t>(
                        indicesGm.GetValue(static_cast<uint64_t>(p)));
                    if (e < 0 || e >= this->numEdges) {
                        // Inconsistent CSR: fail loudly instead of reading
                        // outside the feature buffer or dropping an edge.
                        valid = false;
                        break;
                    }
                    // The scalar index read must be ordered against the MTE2
                    // row load (same contract as the accepted edge_dot_v).
                    SyncStoMte2();
                    LocalTensor<float> row =
                        LoadRow(e * this->width + tileStart, len);
                    AscendC::Add(acc, acc, row, count);
                    inQueue.FreeTensor(row);
                }

                if (!valid) {
                    AscendC::Duplicate(acc, kNaN, count);
                } else if (kMean) {
                    // DGL: sum / clamp(in_degree, 1, max(num_edges, 1)).
                    const float denom =
                        deg > 0 ? static_cast<float>(deg) : 1.0f;
                    AscendC::Muls(acc, acc, 1.0f / denom, count);
                }

                StoreOut(s * this->width + tileStart, acc, count);
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

    __aicore__ inline LocalTensor<float> LoadRow(int64_t gmOffset,
                                                 int64_t len)
    {
        LocalTensor<float> row = inQueue.AllocTensor<float>();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(len * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(row, featGm[static_cast<uint64_t>(gmOffset)], params,
                    padParams);
        inQueue.EnQue(row);
        return inQueue.DeQue<float>();
    }

    __aicore__ inline void StoreOut(int64_t gmOffset, LocalTensor<float> &value,
                                    int32_t count)
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

    // Installed CANN SetFlag/WaitFlag contracts.
    __aicore__ inline void SyncStoMte2()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(id);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(id);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, kBufferNum> inQueue;
    TQue<TPosition::VECOUT, kBufferNum> outQueue;
    TBuf<TPosition::VECCALC> accBuf;
    GlobalTensor<float> featGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<int64_t> offsetsGm;
    GlobalTensor<float> outGm;
    int64_t numEdges = 0;
    int64_t width = 0;
    int64_t tile = 0;
    int64_t s0 = 0;
    int64_t s1 = 0;
};

}  // namespace ascend_graph_segment

#endif  // ASCEND_GRAPH_OP_KERNEL_SEGMENT_REDUCE_COMMON_H
