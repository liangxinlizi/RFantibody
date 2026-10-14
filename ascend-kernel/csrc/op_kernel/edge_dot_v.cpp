// Native general per-edge `e_dot_v` for the DGL surface consumed by the
// SE3-Transformer / RF2 attention path:
//
//   e_dot_v(g, key, query)[e, h, 0]
//       = sum_{j < numFeat} key[e, h, j] * query[dst[e], h, j]
//
// The host wraps `key` as ``[E, H*numFeat]`` and ``query`` as
// ``[N, H*numFeat]`` (H = numHeads, numFeat = per-head width) and the kernel
// writes ``out[e, h] = dot(key[e, h, :], query[dst[e], h, :])`` into a
// ``[E, H]`` output, which the host re-forms to DGL's ``key.shape[:-1] + (1,)``.
// There is no destination aggregation: each edge writes only its own output row,
// so there is no adjacency table and no cross-core dependency.  Each AIV core
// owns a contiguous edge range and the result is deterministic with no atomics.
//
// All GM accesses use DataCopyPad; scalar UB reads of the DMA-staged ``dst``
// value and of the vector-produced product row are ordered by the installed
// CANN 8.1.RC1 SetFlag/WaitFlag contract, exactly like the already-accepted
// ``graph_reduce`` / ``edge_softmax_prepare`` kernels.  An out-of-range ``dst``
// publishes a loud NaN row (never a clamp, never a silently altered value).
// There is no fallback, no atomics, no clamp and no per-example / per-width /
// per-core-count branch.

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

// Loud poison value for an out-of-range destination, so an invalid graph index
// fails the parity gate instead of silently changing the reduction.
constexpr float kNaN = __builtin_nanf("");

class KernelEdgeDotV {
public:
    __aicore__ inline KernelEdgeDotV() {}

    __aicore__ inline void Init(GM_ADDR key, GM_ADDR query, GM_ADDR dst,
                                GM_ADDR out, int64_t numEdges, int64_t numHeads,
                                int64_t numFeat, int64_t numNodes,
                                int64_t numBlocks)
    {
        this->numEdges = numEdges;
        this->numHeads = numHeads;
        this->numFeat = numFeat;
        this->numNodes = numNodes;
        this->width = numHeads * numFeat;

        // 32-byte aligned UB buffers: round the flattened row width and the
        // per-edge output width up to whole float32 blocks.
        this->featAlign = ((this->width + 7) / 8) * 8;
        if (this->featAlign < 8) {
            this->featAlign = 8;
        }
        this->outAlign = ((numHeads + 7) / 8) * 8;
        if (this->outAlign < 8) {
            this->outAlign = 8;
        }

        keyGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(key),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(this->width));
        queryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(query),
            static_cast<uint64_t>(numNodes) * static_cast<uint64_t>(this->width));
        dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(dst),
                              static_cast<uint64_t>(numEdges));
        outGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(out),
            static_cast<uint64_t>(numEdges) * static_cast<uint64_t>(numHeads));

        const uint32_t rowBytes =
            static_cast<uint32_t>(this->featAlign * sizeof(float));
        const uint32_t outBytes =
            static_cast<uint32_t>(this->outAlign * sizeof(float));
        pipe.InitBuffer(keyInQueue, 2, rowBytes);
        pipe.InitBuffer(qInQueue, 2, rowBytes);
        pipe.InitBuffer(prodBuf, rowBytes);
        pipe.InitBuffer(outBuf, outBytes);
        // One int64 destination index (8 B) staged into a 32-byte block.
        pipe.InitBuffer(dstBuf, 32);

        if (numBlocks <= 0) {
            numBlocks = 1;
        }
        const int64_t perCore = (numEdges + numBlocks - 1) / numBlocks;
        this->perCore = perCore;
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
        LocalTensor<float> prod = prodBuf.Get<float>();
        LocalTensor<float> outLocal = outBuf.Get<float>();
        LocalTensor<int64_t> dstLocal = dstBuf.Get<int64_t>();
        const int32_t rowWidth = static_cast<int32_t>(this->width);
        const int32_t outWidth = static_cast<int32_t>(numHeads);

        for (int64_t e = e0; e < e1; ++e) {
            // Stage dst[e] and read it as a UB scalar.  The scalar read must
            // complete before the next edge's DMA overwrites dstLocal.
            DataCopyExtParams dstParams{
                1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
            DataCopyPadExtParams<int64_t> dstPad{false, 0, 0, 0};
            DataCopyPad(dstLocal, dstGm[static_cast<uint64_t>(e)], dstParams,
                        dstPad);
            SyncMte2toS();
            const int64_t d = dstLocal.GetValue(0);
            SyncStoMte2();

            if (d < 0 || d >= numNodes) {
                // Invalid graph index: publish a loud NaN row so the parity
                // gate fails instead of reading out of bounds.
                for (int64_t h = 0; h < numHeads; ++h) {
                    outLocal.SetValue(static_cast<uint32_t>(h), kNaN);
                }
                StoreOut(e, outLocal, outWidth);
                continue;
            }

            LocalTensor<float> k = LoadKeyRow(e);
            LocalTensor<float> q = LoadQueryRow(d);
            AscendC::Mul(prod, k, q, rowWidth);
            keyInQueue.FreeTensor(k);
            qInQueue.FreeTensor(q);

            // Vector product -> scalar reads.
            SyncVtoS();
            for (int64_t h = 0; h < numHeads; ++h) {
                float acc = 0.0f;
                const int64_t base = h * numFeat;
                for (int64_t j = 0; j < numFeat; ++j) {
                    acc += prod.GetValue(static_cast<uint32_t>(base + j));
                }
                outLocal.SetValue(static_cast<uint32_t>(h), acc);
            }
            // Scalar reads of `prod` and scalar writes of `outLocal` must both
            // be complete before the vector unit reuses `prod` next edge.
            SyncStoV();

            StoreOut(e, outLocal, outWidth);
        }
    }

private:
    __aicore__ inline LocalTensor<float> LoadKeyRow(int64_t edge)
    {
        LocalTensor<float> x = keyInQueue.AllocTensor<float>();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(width * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(
            x,
            keyGm[static_cast<uint64_t>(edge) * static_cast<uint64_t>(width)],
            params, padParams);
        keyInQueue.EnQue(x);
        return keyInQueue.DeQue<float>();
    }

    __aicore__ inline LocalTensor<float> LoadQueryRow(int64_t node)
    {
        LocalTensor<float> x = qInQueue.AllocTensor<float>();
        DataCopyExtParams params{
            1,
            static_cast<uint32_t>(width * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(
            x,
            queryGm[static_cast<uint64_t>(node) * static_cast<uint64_t>(width)],
            params, padParams);
        qInQueue.EnQue(x);
        return qInQueue.DeQue<float>();
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
        DataCopyPad(outGm[static_cast<uint64_t>(edge) * static_cast<uint64_t>(numHeads)],
                    y, params);
        // The DMA read of `y` must finish before the next edge's scalar writes
        // reuse the buffer.
        SyncMte3toS();
    }

    // Installed CANN 8.1.RC1 SetFlag/WaitFlag contracts.  `GetTPipePtr()` is the
    // global free function on this toolkit.
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
    TQue<TPosition::VECIN, 2> keyInQueue;
    TQue<TPosition::VECIN, 2> qInQueue;
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> outBuf;
    TBuf<TPosition::VECCALC> dstBuf;
    GlobalTensor<float> keyGm;
    GlobalTensor<float> queryGm;
    GlobalTensor<int64_t> dstGm;
    GlobalTensor<float> outGm;
    int64_t numEdges = 0;
    int64_t numHeads = 0;
    int64_t numFeat = 0;
    int64_t numNodes = 0;
    int64_t width = 0;
    int64_t featAlign = 0;
    int64_t outAlign = 0;
    int64_t perCore = 0;
    int64_t e0 = 0;
    int64_t e1 = 0;
};

}  // namespace

// Entry point consumed by the host bridge.  ``numHeads`` is the number of
// trailing feature groups (one dot product per group); ``numFeat`` is the
// per-group width, so the flattened row width is ``numHeads * numFeat``.
extern "C" __global__ __aicore__ void edge_dot_v(
    GM_ADDR key, GM_ADDR query, GM_ADDR dst, GM_ADDR out, int64_t numEdges,
    int64_t numHeads, int64_t numFeat, int64_t numNodes, int64_t numBlocks)
{
    KernelEdgeDotV op;
    op.Init(key, query, dst, out, numEdges, numHeads, numFeat, numNodes,
            numBlocks);
    op.Process();
}
