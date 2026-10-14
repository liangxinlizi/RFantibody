// Native general per-destination scatter reduction for the DGL ``copy_e``
// surface consumed by the SE3-Transformer path:
//
//   copy_e_sum(g, msg)  -> out[d] = sum_{e : dst[e] == d} msg[e]
//   copy_e_mean(g, msg) -> out[d] = out_sum[d] / in_degree[d]   (0 if no edge)
//
// The destination grouping is the already-proven native stable counting-sort
// segment table (``start``, produced by ``edge_softmax_prepare``), so the
// messages enter this kernel row-sorted: rows ``[start[d], start[d+1])`` all
// belong to destination ``d``.  Each AIV core owns a contiguous destination
// range and accumulates its destinations' rows sequentially in UB, so the
// result is deterministic and needs no atomics and no Torch graph operator.
//
// Zero-degree destinations store an exact ``0`` row (DGL semantics -- never
// NaN, never salted).  Invalid segment offsets, which only the prepare kernel's
// loud poison path can produce, store a NaN row instead of being clamped or
// silently altered, so a corrupt table fails the parity gate loudly.
//
// Only DataCopyPad moves data between GM and UB; every address is derived from
// UB scalars.  The destination slice of ``start`` is staged once per core with
// a single bounded DataCopyPad ordered by the installed CANN MTE2_S
// SetFlag/WaitFlag contract.  There is no fallback, no atomics, no clamp and no
// per-example / per-width / per-core-count branch.

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

// Loud poison value for an invalid segment range (see edge_softmax_prepare).
constexpr float kNaN = __builtin_nanf("");

class KernelGraphScatterSum {
public:
    __aicore__ inline KernelGraphScatterSum() {}

    __aicore__ inline void Init(GM_ADDR msg_sorted, GM_ADDR start, GM_ADDR out,
                                int64_t numEdges, int64_t numFeat, int64_t numNodes,
                                int64_t numBlocks, int64_t meanMode)
    {
        this->numEdges = numEdges;
        this->numFeat = numFeat;
        this->numNodes = numNodes;
        this->numBlocks = numBlocks;
        this->meanMode = (meanMode != 0);
        // One 32-byte block covers 8 float32 lanes; round the feature width up
        // so every UB row buffer is a whole number of blocks.
        this->featAlign = ((numFeat + 7) / 8) * 8;
        if (this->featAlign < 8) {
            this->featAlign = 8;
        }

        msgGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(msg_sorted),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numFeat));
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numNodes) * static_cast<uint64_t>(numFeat));
        startGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(start),
                                static_cast<uint64_t>(numNodes) + 1);

        const uint32_t rowBytes = static_cast<uint32_t>(featAlign * sizeof(float));
        pipe.InitBuffer(rowInQueue, 2, rowBytes);
        pipe.InitBuffer(rowOutQueue, 2, rowBytes);
        pipe.InitBuffer(accBuf, rowBytes);

        if (numBlocks <= 0) {
            numBlocks = 1;
        }
        const int64_t perCore = (numNodes + numBlocks - 1) / numBlocks;
        this->perCore = perCore;
        this->d0 = static_cast<int64_t>(AscendC::GetBlockIdx()) * perCore;
        int64_t d1 = d0 + perCore;
        if (d1 > numNodes) {
            d1 = numNodes;
        }
        this->d1 = d1;
        int64_t cnt = d1 - d0 + 1;
        if (cnt < 1) {
            cnt = 1;
        }
        // Plain VECCALC TBuf for the offset slice: the scalar reads below are
        // ordered only by the installed MTE2_S SetFlag/WaitFlag sequence, not
        // by a queue's MTE2->V dependency.
        const uint32_t startBytes = static_cast<uint32_t>(
            ((cnt * static_cast<int64_t>(sizeof(int32_t))) + 31) / 32 * 32);
        pipe.InitBuffer(startBuf, startBytes);
    }

    __aicore__ inline void Process()
    {
        if (d0 >= d1) {
            return;
        }
        const int32_t feat = static_cast<int32_t>(numFeat);
        const int32_t featAligned = static_cast<int32_t>(featAlign);
        LocalTensor<float> acc = accBuf.Get<float>();

        // Stage start[d0 .. d1] into UB so the per-destination row ranges come
        // from UB instead of a scalar GM read.
        const int64_t cnt = d1 - d0 + 1;
        LocalTensor<int32_t> starts = startBuf.Get<int32_t>();
        DataCopyExtParams startParams{
            1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> startPad{false, 0, 0, 0};
        DataCopyPad(starts, startGm[static_cast<uint64_t>(d0)], startParams, startPad);

        // Installed CANN 8.1.RC1 MTE2 -> scalar contract.
        auto evtMte2ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2ToS);

        for (int64_t d = d0; d < d1; ++d) {
            int64_t s = static_cast<int64_t>(
                starts.GetValue(static_cast<uint32_t>(d - d0)));
            int64_t e = static_cast<int64_t>(
                starts.GetValue(static_cast<uint32_t>(d + 1 - d0)));

            if (s < 0 || e < s || e > numEdges) {
                // Only the prepare kernel's loud poison table reaches here.
                // Publish a NaN row rather than clamping: a corrupt segment
                // table must fail the parity gate, never silently change the
                // reduction.
                LocalTensor<float> y = rowOutQueue.AllocTensor<float>();
                AscendC::Duplicate(y, kNaN, featAligned);
                rowOutQueue.EnQue(y);
                LocalTensor<float> yOut = rowOutQueue.DeQue<float>();
                StoreRow(d, yOut);
                rowOutQueue.FreeTensor(yOut);
                continue;
            }

            // Sequential, deterministic accumulation of this destination's
            // contiguous sorted rows.  A zero-degree destination leaves the
            // accumulator at exactly 0 (mean and sum agree there).
            AscendC::Duplicate(acc, 0.0f, featAligned);
            for (int64_t i = s; i < e; ++i) {
                LocalTensor<float> x = LoadRow(i);
                AscendC::Add(acc, acc, x, feat);
                rowInQueue.FreeTensor(x);
            }
            if (meanMode && e > s) {
                AscendC::Muls(acc, acc, 1.0f / static_cast<float>(e - s), feat);
            }
            LocalTensor<float> y = rowOutQueue.AllocTensor<float>();
            AscendC::Adds(y, acc, 0.0f, feat);
            rowOutQueue.EnQue(y);
            LocalTensor<float> yOut = rowOutQueue.DeQue<float>();
            StoreRow(d, yOut);
            rowOutQueue.FreeTensor(yOut);
        }
    }

private:
    __aicore__ inline LocalTensor<float> LoadRow(int64_t row)
    {
        LocalTensor<float> x = rowInQueue.AllocTensor<float>();
        DataCopyExtParams params{
            1, static_cast<uint32_t>(numFeat * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(
            x,
            msgGm[static_cast<uint64_t>(row) * static_cast<uint64_t>(numFeat)],
            params, padParams);
        rowInQueue.EnQue(x);
        return rowInQueue.DeQue<float>();
    }

    __aicore__ inline void StoreRow(int64_t row, LocalTensor<float> &y)
    {
        DataCopyExtParams params{
            1, static_cast<uint32_t>(numFeat * sizeof(float)), 0, 0, 0};
        DataCopyPad(
            outGm[static_cast<uint64_t>(row) * static_cast<uint64_t>(numFeat)],
            y, params);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 2> rowInQueue;
    TQue<TPosition::VECOUT, 2> rowOutQueue;
    TBuf<TPosition::VECCALC> accBuf;
    TBuf<TPosition::VECCALC> startBuf;
    GlobalTensor<float> msgGm;
    GlobalTensor<float> outGm;
    GlobalTensor<int32_t> startGm;
    bool meanMode = false;
    int64_t numEdges = 0;
    int64_t numFeat = 0;
    int64_t numNodes = 0;
    int64_t numBlocks = 0;
    int64_t featAlign = 0;
    int64_t perCore = 0;
    int64_t d0 = 0;
    int64_t d1 = 0;
};

}  // namespace

// Entry point consumed by the host bridge.  ``meanMode == 0`` sums, otherwise
// each destination's sum is divided by its in-degree.
extern "C" __global__ __aicore__ void graph_scatter_sum(
    GM_ADDR msg_sorted, GM_ADDR start, GM_ADDR out, int64_t numEdges,
    int64_t numFeat, int64_t numNodes, int64_t numBlocks, int64_t meanMode)
{
    KernelGraphScatterSum op;
    op.Init(msg_sorted, start, out, numEdges, numFeat, numNodes, numBlocks,
            meanMode);
    op.Process();
}
