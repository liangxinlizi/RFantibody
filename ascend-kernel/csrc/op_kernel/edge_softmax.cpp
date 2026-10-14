// Native general per-destination edge softmax for DGL's
// `edge_softmax(g, logits, norm_by='dst')` semantics, plus the native
// adjacency preparation and un-permutation that make the whole operator
// self-contained on the NPU.
//
// This kernel computes, for every destination node independently and for every
// trailing feature/head column independently:
//
//     m_d,f = max_{e : dst[e] == d} logits[e, f]
//     s_d,f = sum_{e : dst[e] == d} exp(logits[e, f] - m_d,f)
//     out[e, f] = exp(logits[e, f] - m_d,f) / s_d,f
//
// Destinations with in-degree 0 own no edge rows and therefore produce no
// output.  Duplicate/parallel edges are counted separately.  The stabilising
// maximum is per destination and per feature column -- never a graph-wide
// maximum -- so a destination whose logits are thousands below another
// destination's still receives weights that sum to one.
//
// Iteration 17 removes every torch_npu operator from the `edge_softmax` host
// path.  The adjacency preprocessing (stable counting sort of the destination
// indices) and the final un-permutation are now native AscendC kernels:
//
//   * `edge_softmax_prepare` (single AIV core) reads `dst` (int64) and
//     `logits`, computes the deterministic stable counting-sort segment table
//     `start[d] = #{e : dst[e] < d}` (int32, length `numNodes + 1`), the
//     original-edge position `pos[e]` of each edge (int32, length `numEdges`,
//     `pos[e]` is the sorted row of edge `e`) and the gathered, row-sorted
//     `logits_sorted` (float, `[numEdges, numFeat]`).  It is deterministic:
//     edges are scanned in original order, so equal destinations are ordered
//     exactly like `torch.argsort(dst, stable=True)`.
//   * `edge_softmax` (multi-core AIV) computes the per-destination softmax in
//     row-sorted space.
//   * `edge_softmax_unpermute` (multi-core AIV) restores the product output to
//     the original edge order: `out[e] = out_sorted[pos[e]]`.
//
// Only DataCopyPad DMA moves data between GM and UB for every result; scalar GM
// reads (`GlobalTensor::GetValue`) are never used.  All three launches are raw
// `aclrtlaunch_*` calls on the caller's stream, so the producer->consumer order
// is the stream FIFO and there is no torch_npu task-queue reordering.
//
// The segment table is staged into UB once per core with a single bounded
// `DataCopyPad` and read back with UB scalar reads (`LocalTensor::GetValue`)
// ordered by the installed CANN 8.1.RC1 `MTE2_S` SetFlag/WaitFlag contract.
// The product kernel does not clamp or otherwise sanitise the segment offsets:
// the public boundary validates the graph indices and the offset table is
// monotone by construction, so a corrupt table fails loudly instead of silently
// altering the weights.  The test-only `edge_softmax_probe` entry shares this
// exact body and additionally reports the segment start each destination read
// (with a loud poison sentinel on any invalid range) without changing the
// product arithmetic or its launch signature.

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

// -FLT_MAX.  A finite lower bound keeps the running maximum free of infinities;
// unused/padded lanes are dominated by any real logit.  For a non-empty segment
// at least one edge attains the per-column maximum, so the denominator is always
// >= 1 and no `tiny`/`eps` salting is needed.
constexpr float kNegInf = -3.4028234663852886e+38f;

// 32-byte aligned byte size.  UB buffers used by DataCopyPad must be block
// aligned; the destination table is sized from the graph node count.
__aicore__ inline uint32_t AlignUpBytes(int64_t bytes)
{
    if (bytes <= 0) {
        return 32;
    }
    return static_cast<uint32_t>(((bytes + 31) / 32) * 32);
}

// Maximum UB bytes reserved for the destination counting table.  The realistic
// RFdiffusion/RF2 and DGL-parity graphs use a few thousand nodes at most, so the
// table is always UB-resident.  A larger graph is rejected loudly (poison
// `start[0] = -1`) instead of being silently truncated or clamped; the buffer is
// capped only so `InitBuffer` can never overrun UB.
constexpr int64_t kMaxNodeTableBytes = 128 * 1024;

// ---------------------------------------------------------------------------
// Per-destination softmax over row-sorted logits (product and probe entries).
// ---------------------------------------------------------------------------
class KernelEdgeSoftmax {
public:
    __aicore__ inline KernelEdgeSoftmax() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR start, GM_ADDR out,
                                GM_ADDR segOut, int64_t numEdges, int64_t numFeat,
                                int64_t numNodes, int64_t numBlocks)
    {
        this->numEdges = numEdges;
        this->numFeat = numFeat;
        this->numNodes = numNodes;
        this->numBlocks = numBlocks;
        // One 32-byte block covers 8 float32 lanes; round the feature width up
        // so every UB row buffer is a whole number of blocks.
        this->featAlign = ((numFeat + 7) / 8) * 8;
        if (this->featAlign < 8) {
            this->featAlign = 8;
        }

        logitsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(logits),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numFeat));
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numFeat));
        startGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(start),
                                static_cast<uint64_t>(numNodes) + 1);
        // Optional test-only segment write-back buffer (null for the product
        // entry).  When present the probe records the offset each destination
        // actually read; the product kernel never dereferences it.
        this->writeSeg = (segOut != nullptr);
        if (this->writeSeg) {
            segGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(segOut),
                                  static_cast<uint64_t>(numNodes));
        }

        const uint32_t rowBytes = static_cast<uint32_t>(featAlign * sizeof(float));
        pipe.InitBuffer(rowInQueue, 2, rowBytes);
        pipe.InitBuffer(rowOutQueue, 2, rowBytes);
        pipe.InitBuffer(mrunBuf, rowBytes);
        pipe.InitBuffer(srunBuf, rowBytes);
        pipe.InitBuffer(tmpBuf, rowBytes);

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
        // Plain VECCALC TBuf for the offset slice: with a TBuf the scalar read
        // below is ordered only by the installed MTE2_S SetFlag/WaitFlag
        // sequence, not by a queue's MTE2->V dependency.
        const uint32_t startBytes = static_cast<uint32_t>(
            ((cnt * static_cast<int64_t>(sizeof(int32_t))) + 31) / 32 * 32);
        pipe.InitBuffer(startBuf, startBytes);

        // Test-only segment write-back buffer: one int32 per destination owned
        // by this core, written back to `segGm` with a single UB->GM
        // DataCopyPad.  Scalar GM stores from the AIV core are not reliably
        // visible on this CANN build, so the write-back is staged in UB and
        // copied out exactly like every other result.  The product path never
        // allocates or touches it (`writeSeg == false`).
        if (this->writeSeg) {
            const int64_t segCount = (d1 > d0) ? (d1 - d0) : 0;
            pipe.InitBuffer(
                segBuf,
                AlignUpBytes(segCount * static_cast<int64_t>(sizeof(int32_t))));
        }
    }

    __aicore__ inline void Process()
    {
        if (d0 >= d1) {
            return;
        }
        const int32_t feat = static_cast<int32_t>(numFeat);
        const int32_t featAligned = static_cast<int32_t>(featAlign);

        LocalTensor<float> mrun = mrunBuf.Get<float>();
        LocalTensor<float> srun = srunBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<int32_t> seg;
        if (writeSeg) {
            seg = segBuf.Get<int32_t>();
        }

        // Stage start[d0 .. d1] (one int32 per destination boundary) into UB so
        // the per-destination row ranges come from UB instead of a scalar GM
        // read.
        const int64_t cnt = d1 - d0 + 1;
        LocalTensor<int32_t> starts = startBuf.Get<int32_t>();
        DataCopyExtParams startParams{
            1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> startPad{false, 0, 0, 0};
        DataCopyPad(starts, startGm[static_cast<uint64_t>(d0)], startParams, startPad);

        // Installed CANN 8.1.RC1 MTE2 -> scalar contract
        // (kernel_operator_sync_impl.h:70-72); GetTPipePtr() is the global free
        // function.
        auto evtMte2ToS = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2ToS);

        for (int64_t d = d0; d < d1; ++d) {
            int64_t s =
                static_cast<int64_t>(starts.GetValue(static_cast<uint32_t>(d - d0)));
            int64_t e =
                static_cast<int64_t>(starts.GetValue(static_cast<uint32_t>(d + 1 - d0)));
            if (writeSeg) {
                // Test-only route localization: report the segment start this
                // destination actually read.  A range violation is loud and
                // poison-encoded, and the row pipeline is skipped so the probe
                // can never issue an out-of-range access.  Valid offsets satisfy
                // 0 <= s <= e <= numEdges, so -1 can never be a real offset.
                if (s < 0 || e < s || e > numEdges) {
                    seg.SetValue(static_cast<uint32_t>(d - d0),
                                 static_cast<int32_t>(-1));
                    continue;
                }
                seg.SetValue(static_cast<uint32_t>(d - d0),
                             static_cast<int32_t>(s));
            }
            if (e <= s) {
                continue;  // zero-degree destination
            }

            // Pass 1: per-destination, per-column stabilising maximum.
            AscendC::Duplicate(mrun, kNegInf, featAligned);
            for (int64_t i = s; i < e; ++i) {
                LocalTensor<float> x = LoadRow(i);
                AscendC::Max(mrun, mrun, x, feat);
                rowInQueue.FreeTensor(x);
            }

            // Pass 2: per-destination, per-column denominator.
            AscendC::Duplicate(srun, 0.0f, featAligned);
            for (int64_t i = s; i < e; ++i) {
                LocalTensor<float> x = LoadRow(i);
                AscendC::Sub(tmp, x, mrun, feat);
                AscendC::Exp(tmp, tmp, feat);
                AscendC::Add(srun, srun, tmp, feat);
                rowInQueue.FreeTensor(x);
            }

            // Pass 3: normalise and store this destination's rows in sorted
            // order; the un-permute entry restores the original edge order.
            for (int64_t i = s; i < e; ++i) {
                LocalTensor<float> x = LoadRow(i);
                AscendC::Sub(tmp, x, mrun, feat);
                AscendC::Exp(tmp, tmp, feat);
                LocalTensor<float> y = rowOutQueue.AllocTensor<float>();
                AscendC::Div(y, tmp, srun, feat);
                rowOutQueue.EnQue(y);
                LocalTensor<float> yOut = rowOutQueue.DeQue<float>();
                StoreRow(i, yOut);
                rowOutQueue.FreeTensor(yOut);
                rowInQueue.FreeTensor(x);
            }
        }

        // Publish the probe's per-destination segment offsets with the same
        // UB->GM DataCopyPad pattern the product store uses; the S->MTE3 sync
        // orders the scalar UB writes before the copy reads them.
        if (writeSeg) {
            SyncStoMte3();
            DataCopyExtParams segParams{
                1,
                static_cast<uint32_t>(static_cast<int64_t>(d1 - d0) *
                                      static_cast<int64_t>(sizeof(int32_t))),
                0, 0, 0};
            DataCopyPad(segGm[static_cast<uint64_t>(d0)], seg, segParams);
        }
    }

private:
    __aicore__ inline LocalTensor<float> LoadRow(int64_t row)
    {
        LocalTensor<float> x = rowInQueue.AllocTensor<float>();
        DataCopyExtParams params{
            1, static_cast<uint32_t>(numFeat * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(x, logitsGm[static_cast<uint64_t>(row) * static_cast<uint64_t>(numFeat)],
                    params, padParams);
        rowInQueue.EnQue(x);
        return rowInQueue.DeQue<float>();
    }

    __aicore__ inline void StoreRow(int64_t row, LocalTensor<float> &y)
    {
        DataCopyExtParams params{
            1, static_cast<uint32_t>(numFeat * sizeof(float)), 0, 0, 0};
        DataCopyPad(outGm[static_cast<uint64_t>(row) * static_cast<uint64_t>(numFeat)], y, params);
    }

    __aicore__ inline void SyncStoMte3()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(id);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(id);
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 2> rowInQueue;
    TQue<TPosition::VECOUT, 2> rowOutQueue;
    TBuf<TPosition::VECCALC> mrunBuf;
    TBuf<TPosition::VECCALC> srunBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> startBuf;
    TBuf<TPosition::VECCALC> segBuf;
    GlobalTensor<float> logitsGm;
    GlobalTensor<float> outGm;
    GlobalTensor<int32_t> startGm;
    GlobalTensor<int32_t> segGm;
    bool writeSeg = false;
    int64_t numEdges = 0;
    int64_t numFeat = 0;
    int64_t numNodes = 0;
    int64_t numBlocks = 0;
    int64_t featAlign = 0;
    int64_t perCore = 0;
    int64_t d0 = 0;
    int64_t d1 = 0;
};

// ---------------------------------------------------------------------------
// Native deterministic stable counting-sort adjacency preparation.
//
// Produces `start` (int32, N+1), `pos` (int32, E, pos[e] = sorted row of edge e)
// and `logits_sorted` (float, [E, F]) with a stable scan over the edges in
// original order.  All output is written with DataCopyPad; no scalar GM read is
// used anywhere.  Launched with one AIV core; other block indices return.
// ---------------------------------------------------------------------------
constexpr int64_t kPrepareEdgeTile = 2048;

class KernelEdgeSoftmaxPrepare {
public:
    __aicore__ inline KernelEdgeSoftmaxPrepare() {}

    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR logits, GM_ADDR start,
                                GM_ADDR pos, GM_ADDR logits_sorted,
                                int64_t numEdges, int64_t numFeat, int64_t numNodes)
    {
        this->numEdges = numEdges;
        this->numFeat = numFeat;
        this->numNodes = numNodes;

        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(dst),
                              static_cast<uint64_t>(numEdges));
        logitsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(logits),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numFeat));
        startGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(start),
                                static_cast<uint64_t>(numNodes) + 1);
        posGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(pos),
                              static_cast<uint64_t>(numEdges));
        logitsSortedGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(logits_sorted),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numFeat));

        const int64_t tableNeeded =
            (static_cast<int64_t>(numNodes) + 1) * static_cast<int64_t>(sizeof(int32_t));
        this->tableFits = tableNeeded <= kMaxNodeTableBytes;
        const int64_t tableBytes =
            tableNeeded < kMaxNodeTableBytes ? tableNeeded : kMaxNodeTableBytes;
        pipe.InitBuffer(nodeBuf, AlignUpBytes(tableBytes));
        pipe.InitBuffer(dstTileBuf,
                        kPrepareEdgeTile * static_cast<int64_t>(sizeof(int64_t)));
        pipe.InitBuffer(posTileBuf,
                        kPrepareEdgeTile * static_cast<int64_t>(sizeof(int32_t)));
        pipe.InitBuffer(rowBuf,
                        AlignUpBytes(static_cast<int64_t>(numFeat) *
                                     static_cast<int64_t>(sizeof(float))));
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        if (numEdges <= 0 || numNodes <= 0 || numFeat <= 0) {
            return;
        }

        LocalTensor<int32_t> nodes = nodeBuf.Get<int32_t>();
        LocalTensor<int64_t> dstTile = dstTileBuf.Get<int64_t>();

        // A graph too large for the UB-resident counting table is rejected
        // loudly.  Only `start[0]` is written (as -1) and the kernel returns;
        // the consumer then fails on a clearly invalid negative offset instead
        // of silently altering the graph computation.
        if (!tableFits) {
            nodes.SetValue(0, -1);
            SyncStoMte3();
            DataCopyExtParams oneParams{
                1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(startGm[0], nodes, oneParams);
            return;
        }

        // Zero the per-destination counts.
        for (int64_t d = 0; d <= numNodes; ++d) {
            nodes.SetValue(static_cast<uint32_t>(d), 0);
        }

        // Histogram + range validation.  Edges are scanned in original order.
        bool invalid = false;
        for (int64_t base = 0; base < numEdges; base += kPrepareEdgeTile) {
            int64_t cur = numEdges - base;
            if (cur > kPrepareEdgeTile) {
                cur = kPrepareEdgeTile;
            }
            CopyInDst(dstTile, base, cur);
            for (int64_t i = 0; i < cur; ++i) {
                int64_t d =
                    static_cast<int64_t>(dstTile.GetValue(static_cast<uint32_t>(i)));
                if (d < 0 || d >= numNodes) {
                    invalid = true;
                    break;
                }
                int32_t c = nodes.GetValue(static_cast<uint32_t>(d));
                nodes.SetValue(static_cast<uint32_t>(d), c + 1);
            }
            SyncStoMte2();
            if (invalid) {
                break;
            }
        }

        if (invalid) {
            // Loud poison: every published offset is -1 so a consumer can never
            // mistake the table for a valid monotone one.
            for (int64_t d = 0; d <= numNodes; ++d) {
                nodes.SetValue(static_cast<uint32_t>(d), -1);
            }
            WriteStart(nodes);
            return;
        }

        // Exclusive prefix sum in place: nodes[d] = start[d]; nodes[N] = E.
        int32_t running = 0;
        for (int64_t d = 0; d < numNodes; ++d) {
            int32_t c = nodes.GetValue(static_cast<uint32_t>(d));
            nodes.SetValue(static_cast<uint32_t>(d), running);
            running += c;
        }
        nodes.SetValue(static_cast<uint32_t>(numNodes), running);

        // Publish `start`, then wait for the copy to finish reading `nodes`
        // before the same buffer is reused as the placement cursor.
        WriteStart(nodes);
        SyncMte3toS();

        // Placement + gather in one stable pass over the original edges.
        LocalTensor<int32_t> posTile = posTileBuf.Get<int32_t>();
        LocalTensor<float> row = rowBuf.Get<float>();
        for (int64_t base = 0; base < numEdges; base += kPrepareEdgeTile) {
            int64_t cur = numEdges - base;
            if (cur > kPrepareEdgeTile) {
                cur = kPrepareEdgeTile;
            }
            CopyInDst(dstTile, base, cur);
            for (int64_t i = 0; i < cur; ++i) {
                int64_t d =
                    static_cast<int64_t>(dstTile.GetValue(static_cast<uint32_t>(i)));
                int32_t p = nodes.GetValue(static_cast<uint32_t>(d));
                nodes.SetValue(static_cast<uint32_t>(d), p + 1);
                posTile.SetValue(static_cast<uint32_t>(i), p);
                GatherRow(row, base + i, p);
            }
            // Scalar reads of dstTile are complete; the next tile may overwrite
            // it.  Scalar writes to posTile are complete before the DMA read.
            SyncStoMte2();
            SyncStoMte3();
            DataCopyExtParams posParams{
                1, static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(int32_t))),
                0, 0, 0};
            DataCopyPad(posGm[static_cast<uint64_t>(base)], posTile, posParams);
            // The DMA read of posTile must finish before the next tile's scalar
            // writes reuse the buffer.
            SyncMte3toS();
        }
    }

private:
    __aicore__ inline void CopyInDst(LocalTensor<int64_t> &dstTile, int64_t base,
                                     int64_t cur)
    {
        DataCopyExtParams params{
            1, static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(int64_t))),
            0, 0, 0};
        DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
        DataCopyPad(dstTile, dstGm[static_cast<uint64_t>(base)], params, pad);
        SyncMte2toS();
    }

    __aicore__ inline void GatherRow(LocalTensor<float> &row, int64_t edge, int32_t p)
    {
        DataCopyExtParams inParams{
            1, static_cast<uint32_t>(numFeat * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPadExtParams<float> inPad{false, 0, 0, 0.0f};
        DataCopyPad(
            row,
            logitsGm[static_cast<uint64_t>(edge) * static_cast<uint64_t>(numFeat)],
            inParams, inPad);
        SyncMte2toMte3();
        DataCopyExtParams outParams{
            1, static_cast<uint32_t>(numFeat * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPad(
            logitsSortedGm[static_cast<uint64_t>(p) * static_cast<uint64_t>(numFeat)],
            row, outParams);
        SyncMte3toMte2();
    }

    __aicore__ inline void WriteStart(LocalTensor<int32_t> &nodes)
    {
        SyncStoMte3();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>((static_cast<int64_t>(numNodes) + 1) *
                                  static_cast<int64_t>(sizeof(int32_t))),
            0, 0, 0};
        DataCopyPad(startGm[0], nodes, params);
    }

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
    __aicore__ inline void SyncMte2toMte3()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(id);
    }
    __aicore__ inline void SyncMte3toMte2()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(id);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> nodeBuf;
    TBuf<TPosition::VECCALC> dstTileBuf;
    TBuf<TPosition::VECCALC> posTileBuf;
    TBuf<TPosition::VECCALC> rowBuf;
    GlobalTensor<int64_t> dstGm;
    GlobalTensor<float> logitsGm;
    GlobalTensor<int32_t> startGm;
    GlobalTensor<int32_t> posGm;
    GlobalTensor<float> logitsSortedGm;
    int64_t numEdges = 0;
    int64_t numFeat = 0;
    int64_t numNodes = 0;
    bool tableFits = true;
};

// ---------------------------------------------------------------------------
// Native un-permutation: out[e] = out_sorted[pos[e]] for every edge.
// Each AIV core owns a contiguous edge range; the source row is scattered and
// the destination row is contiguous.
// ---------------------------------------------------------------------------
constexpr int64_t kUnpermuteEdgeTile = 2048;

class KernelEdgeSoftmaxUnpermute {
public:
    __aicore__ inline KernelEdgeSoftmaxUnpermute() {}

    __aicore__ inline void Init(GM_ADDR pos, GM_ADDR out_sorted, GM_ADDR out,
                                int64_t numEdges, int64_t numFeat, int64_t numBlocks)
    {
        this->numEdges = numEdges;
        this->numFeat = numFeat;
        posGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(pos),
                              static_cast<uint64_t>(numEdges));
        outSortedGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out_sorted),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numFeat));
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numFeat));

        pipe.InitBuffer(posTileBuf,
                        kUnpermuteEdgeTile * static_cast<int64_t>(sizeof(int32_t)));
        pipe.InitBuffer(rowBuf,
                        AlignUpBytes(static_cast<int64_t>(numFeat) *
                                     static_cast<int64_t>(sizeof(float))));

        if (numBlocks <= 0) {
            numBlocks = 1;
        }
        const int64_t perCore = (numEdges + numBlocks - 1) / numBlocks;
        this->e0 = static_cast<int64_t>(AscendC::GetBlockIdx()) * perCore;
        int64_t e1 = e0 + perCore;
        if (e1 > numEdges) {
            e1 = numEdges;
        }
        this->e1 = e1;
    }

    __aicore__ inline void Process()
    {
        if (e0 >= e1) {
            return;
        }
        LocalTensor<int32_t> posTile = posTileBuf.Get<int32_t>();
        LocalTensor<float> row = rowBuf.Get<float>();
        for (int64_t base = e0; base < e1; base += kUnpermuteEdgeTile) {
            int64_t cur = e1 - base;
            if (cur > kUnpermuteEdgeTile) {
                cur = kUnpermuteEdgeTile;
            }
            DataCopyExtParams posParams{
                1, static_cast<uint32_t>(cur * static_cast<int64_t>(sizeof(int32_t))),
                0, 0, 0};
            DataCopyPadExtParams<int32_t> posPad{false, 0, 0, 0};
            DataCopyPad(posTile, posGm[static_cast<uint64_t>(base)], posParams,
                        posPad);
            SyncMte2toS();
            for (int64_t i = 0; i < cur; ++i) {
                int32_t src = posTile.GetValue(static_cast<uint32_t>(i));
                DataCopyExtParams inParams{
                    1,
                    static_cast<uint32_t>(numFeat *
                                          static_cast<int64_t>(sizeof(float))),
                    0, 0, 0};
                DataCopyPadExtParams<float> inPad{false, 0, 0, 0.0f};
                DataCopyPad(
                    row,
                    outSortedGm[static_cast<uint64_t>(src) *
                                static_cast<uint64_t>(numFeat)],
                    inParams, inPad);
                SyncMte2toMte3();
                DataCopyExtParams outParams{
                    1,
                    static_cast<uint32_t>(numFeat *
                                          static_cast<int64_t>(sizeof(float))),
                    0, 0, 0};
                DataCopyPad(
                    outGm[static_cast<uint64_t>(base + i) *
                          static_cast<uint64_t>(numFeat)],
                    row, outParams);
                SyncMte3toMte2();
            }
            // All scalar reads of posTile are complete before the next tile
            // overwrites it with DMA.
            SyncStoMte2();
        }
    }

private:
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
    __aicore__ inline void SyncMte2toMte3()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(id);
    }
    __aicore__ inline void SyncMte3toMte2()
    {
        auto id = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(id);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> posTileBuf;
    TBuf<TPosition::VECCALC> rowBuf;
    GlobalTensor<int32_t> posGm;
    GlobalTensor<float> outSortedGm;
    GlobalTensor<float> outGm;
    int64_t numEdges = 0;
    int64_t numFeat = 0;
    int64_t e0 = 0;
    int64_t e1 = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void edge_softmax(GM_ADDR logits, GM_ADDR start,
                                                   GM_ADDR out, int64_t numEdges,
                                                   int64_t numFeat, int64_t numNodes,
                                                   int64_t numBlocks)
{
    KernelEdgeSoftmax op;
    op.Init(logits, start, out, (GM_ADDR)nullptr, numEdges, numFeat, numNodes,
            numBlocks);
    op.Process();
}

// Test-only companion entry: identical math and launch contract as
// `edge_softmax`, plus a write-back of the segment start each destination read
// (`segOut[d] = start[d]`, or -1 on an invalid range).  It lets the independent
// DGL parity/localization tests distinguish a stale segment table from a row
// pipeline defect.  It is never used by the product model path.
extern "C" __global__ __aicore__ void edge_softmax_probe(
    GM_ADDR logits, GM_ADDR start, GM_ADDR out, GM_ADDR segOut,
    int64_t numEdges, int64_t numFeat, int64_t numNodes, int64_t numBlocks)
{
    KernelEdgeSoftmax op;
    op.Init(logits, start, out, segOut, numEdges, numFeat, numNodes, numBlocks);
    op.Process();
}

// Native deterministic stable counting-sort adjacency preparation.  Launched
// with one AIV core; produces `start`, `pos` and `logits_sorted`.
extern "C" __global__ __aicore__ void edge_softmax_prepare(
    GM_ADDR dst, GM_ADDR logits, GM_ADDR start, GM_ADDR pos, GM_ADDR logits_sorted,
    int64_t numEdges, int64_t numFeat, int64_t numNodes)
{
    KernelEdgeSoftmaxPrepare op;
    op.Init(dst, logits, start, pos, logits_sorted, numEdges, numFeat, numNodes);
    op.Process();
}

// Native un-permutation back to the original edge order.
extern "C" __global__ __aicore__ void edge_softmax_unpermute(
    GM_ADDR pos, GM_ADDR out_sorted, GM_ADDR out, int64_t numEdges, int64_t numFeat,
    int64_t numBlocks)
{
    KernelEdgeSoftmaxUnpermute op;
    op.Init(pos, out_sorted, out, numEdges, numFeat, numBlocks);
    op.Process();
}
