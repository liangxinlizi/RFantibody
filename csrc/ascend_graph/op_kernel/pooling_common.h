// Native AscendC global pooling over contiguous node-feature rows.
//
// This header owns the single kernel body shared by the two DGL global-pooling
// operators consumed by the SE3-Transformer graph surface:
//
//   avg_pooling (dgl.nn.AvgPooling)  -> dgl.readout_nodes(op='mean')
//       out[b, :] = (sum_{i in segment b} feat[i, :]) / clamp(len_b, 1, N)
//   max_pooling (dgl.nn.MaxPooling)  -> dgl.readout_nodes(op='max')
//       out[b, :] = max_{i in segment b} feat[i, :]
//
// Exactly like `dgl.ops.segment.segment_reduce` (which both poolings reach
// through `dgl.readout_nodes`), a zero-length segment produces exactly zero:
// the mean denominator is `clamp(seglen, 1, num_rows)` (so `0 / 1 == 0`) and the
// max of an empty segment is DGL's `replace_inf_with_zero` result, i.e. zero -
// never `-inf`.  One graph yields `[1, ...]`, a `dgl.batch` of B graphs yields
// `[B, ...]`; the trailing feature dimensions are flattened by the host adapter
// and restored afterwards (the production `GPooling` passes `[N, F, 1]`).
//
// The segment table (`seglen` int64 [numSegs]) is graph metadata that the kernel
// itself turns into row offsets: every AIV block performs the sequential GM scan
// of `seglen` that yields (a) the global row offset of its own first segment and
// (b) the scanned total used to validate the table against `numRows`.  No host
// `cumsum`, no Torch index computation and no CPU arithmetic produces the
// segment offsets.
//
// Tiling / partition: `numBlocks = min(vector_core_count, numSegs)` on the host
// and each block owns a contiguous segment range, so there are no atomics and no
// cross-core reduction.  Inside a segment the feature width is walked in tiles
// of at most `kMaxTileElements` float32 elements; every GM transfer uses
// `DataCopyPad` with the exact byte length and the vector ops use the same
// element count, so a non-32-byte-aligned width (`C=1`, `C=3`, `C=13`, ...) is
// safe.  The UB buffers are sized to the 32-byte-rounded tile.
//
// Every GlobalTensor is bound with `SetGlobalBuffer` in `Init` before first use,
// and scalar GM reads of the segment table are ordered against the following
// MTE2 row loads with the same SetFlag/WaitFlag contract as the accepted
// `edge_dot_v`/`segment_reduce_common.h` kernels.
//
// Loud failure: a negative segment length or a scanned total that does not match
// `numRows` poisons the affected output rows with NaN instead of silently
// dropping or clamping rows.  There is no Torch/DGL/NumPy/CPU fallback, no
// `scatter`/`index_add`, no per-shape or per-example branch.

#ifndef ASCEND_GRAPH_OP_KERNEL_POOLING_COMMON_H
#define ASCEND_GRAPH_OP_KERNEL_POOLING_COMMON_H

#include "kernel_operator.h"

namespace ascend_graph_pool {

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
// Loud poison value for an inconsistent segment table.
constexpr float kNaN = __builtin_nanf("");

// `kMax == false` -> avg_pooling, `kMax == true` -> max_pooling.
template <bool kMax>
class KernelGlobalPool {
public:
    __aicore__ inline KernelGlobalPool() {}

    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR seglen, GM_ADDR out,
                                int64_t numRows, int64_t numSegs, int64_t width,
                                int64_t numBlocks)
    {
        this->numRows = numRows;
        this->numSegs = numSegs;
        this->width = width;

        if (numBlocks <= 0) {
            numBlocks = 1;
        }
        const int64_t perSeg = (numSegs + numBlocks - 1) / numBlocks;
        this->s0 = static_cast<int64_t>(AscendC::GetBlockIdx()) * perSeg;
        int64_t s1 = this->s0 + perSeg;
        if (s1 > numSegs) {
            s1 = numSegs;
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
            static_cast<uint64_t>(numRows) * static_cast<uint64_t>(width));
        seglenGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(seglen),
                                 static_cast<uint64_t>(numSegs));
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numSegs) * static_cast<uint64_t>(width));

        pipe.InitBuffer(inQueue, kBufferNum, tileBytes);
        pipe.InitBuffer(outQueue, kBufferNum, tileBytes);
        pipe.InitBuffer(accBuf, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->s0 >= this->s1 || this->width <= 0) {
            return;
        }

        // Native prefix scan of the segment table: the global row offset of this
        // block's first segment plus the scanned total used to validate the
        // table against the number of feature rows.
        int64_t total = 0;
        int64_t rowStart = 0;
        bool valid = true;
        for (int64_t s = 0; s < this->numSegs; ++s) {
            const int64_t len = ReadSegLen(s);
            if (s == this->s0) {
                rowStart = total;
            }
            if (len < 0) {
                valid = false;
            }
            total += len;
        }
        if (total != this->numRows) {
            valid = false;
        }

        if (!valid) {
            // Inconsistent segment table: fail loudly on the rows this block
            // owns instead of reading outside the feature buffer or silently
            // dropping/clamping rows.
            for (int64_t s = this->s0; s < this->s1; ++s) {
                PoisonSegment(s);
            }
            return;
        }

        LocalTensor<float> acc = accBuf.Get<float>();
        int64_t row = rowStart;
        for (int64_t s = this->s0; s < this->s1; ++s) {
            const int64_t segLen = ReadSegLen(s);
            for (int64_t tileStart = 0; tileStart < this->width;
                 tileStart += this->tile) {
                int64_t tileLen = this->width - tileStart;
                if (tileLen > this->tile) {
                    tileLen = this->tile;
                }
                const int32_t count = static_cast<int32_t>(tileLen);

                if (segLen > 0) {
                    PoolSegment(acc, row, tileStart, tileLen, count, segLen);
                } else {
                    // DGL semantics for a zero-length segment: mean is
                    // 0 / clamp(0, 1, N) == 0 and the empty max is
                    // replace_inf_with_zero(-inf) == 0.
                    AscendC::Duplicate(acc, 0.0f, count);
                }

                StoreOut(s * this->width + tileStart, acc, count);
            }
            row += segLen;
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

    // Scalar GM read of one segment length.  The read is ordered against the
    // following MTE2 row load (same contract as the accepted edge_dot_v kernel).
    __aicore__ inline int64_t ReadSegLen(int64_t index)
    {
        const int64_t value = seglenGm.GetValue(static_cast<uint64_t>(index));
        SyncStoMte2();
        return value;
    }

    // DGL 1.1.3: the mean denominator is
    // `clamp(seglen, 1, len(value))` cast to the value dtype (float32).
    __aicore__ inline float MeanDenominator(int64_t segLen) const
    {
        int64_t clamped = segLen;
        if (clamped < 1) {
            clamped = 1;
        }
        if (clamped > this->numRows) {
            clamped = this->numRows;
        }
        if (clamped < 1) {
            clamped = 1;
        }
        return static_cast<float>(clamped);
    }

    // Reduce one tile of one non-empty segment starting at feature row `row`.
    __aicore__ inline void PoolSegment(LocalTensor<float> &acc, int64_t row,
                                       int64_t tileStart, int64_t tileLen,
                                       int32_t count, int64_t segLen)
    {
        for (int64_t r = 0; r < segLen; ++r) {
            const int64_t gmOffset = (row + r) * this->width + tileStart;
            // The tile offset is scalar-GM derived; order the scalar reads
            // against the MTE2 row load exactly like the accepted kernels.
            SyncStoMte2();
            LocalTensor<float> src = LoadRow(gmOffset, tileLen);
            if (r == 0) {
                if (kMax) {
                    // Bit-exact first-row copy (`x * 1.0f` preserves -0.0 and
                    // every finite mantissa, unlike `x + 0.0f`).
                    AscendC::Muls(acc, src, 1.0f, count);
                } else {
                    AscendC::Duplicate(acc, 0.0f, count);
                    AscendC::Add(acc, acc, src, count);
                }
            } else if (kMax) {
                AscendC::Max(acc, acc, src, count);
            } else {
                AscendC::Add(acc, acc, src, count);
            }
            inQueue.FreeTensor(src);
        }
        if (!kMax) {
            const float denom = MeanDenominator(segLen);
            AscendC::Muls(acc, acc, 1.0f / denom, count);
        }
    }

    __aicore__ inline void PoisonSegment(int64_t segment)
    {
        LocalTensor<float> acc = accBuf.Get<float>();
        for (int64_t tileStart = 0; tileStart < this->width;
             tileStart += this->tile) {
            int64_t tileLen = this->width - tileStart;
            if (tileLen > this->tile) {
                tileLen = this->tile;
            }
            const int32_t count = static_cast<int32_t>(tileLen);
            AscendC::Duplicate(acc, kNaN, count);
            StoreOut(segment * this->width + tileStart, acc, count);
        }
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
    GlobalTensor<int64_t> seglenGm;
    GlobalTensor<float> outGm;
    int64_t numRows = 0;
    int64_t numSegs = 0;
    int64_t width = 0;
    int64_t tile = 0;
    int64_t s0 = 0;
    int64_t s1 = 0;
};

}  // namespace ascend_graph_pool

#endif  // ASCEND_GRAPH_OP_KERNEL_POOLING_COMMON_H
