// Route-control kernel for the RFantibody Ascend migration.
//
// This is scaffolding, *not* part of the product graph path: it computes the
// deterministic `y = x + 1` over a bounded float32 tensor so that a build,
// registration, launch, synchronize and copy-back defect can be separated from
// an operator-semantics defect in `edge_dot_v` (same library, same generated
// host stub, same `PrivateUse1` registration and launch discipline).
//
// Tiling: each AIV core owns a contiguous element range (host computes
// `perCore`), and each range is processed in `tileLength`-element tiles moved
// with `DataCopyPad`, so an arbitrary (un-aligned) total length is safe.

#include "kernel_operator.h"

namespace {

constexpr int32_t kBufferNum = 2;

class KernelGraphProbe {
public:
    __aicore__ inline KernelGraphProbe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t totalLength,
                                int64_t perCore, int64_t tileLength)
    {
        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        this->offset = blockIdx * perCore;
        int64_t length = totalLength - this->offset;
        if (length > perCore) {
            length = perCore;
        }
        if (length < 0) {
            length = 0;
        }
        this->blockLength = length;
        if (length <= 0) {
            return;
        }
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x) + this->offset,
                            length);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y) + this->offset,
                            length);
        this->tileLength = tileLength;
        const uint32_t tileBytes =
            static_cast<uint32_t>(tileLength * static_cast<int64_t>(sizeof(float)));
        pipe.InitBuffer(inQueueX, kBufferNum, tileBytes);
        pipe.InitBuffer(outQueueY, kBufferNum, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength <= 0) {
            return;
        }
        const int64_t tileNum =
            (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (int64_t i = 0; i < tileNum; ++i) {
            const int64_t remaining = this->blockLength - i * this->tileLength;
            const int64_t curTileLength =
                (remaining < this->tileLength) ? remaining : this->tileLength;
            CopyIn(i, curTileLength);
            Compute(curTileLength);
            CopyOut(i, curTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(curTileLength * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], params,
                             padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t curTileLength)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Adds(yLocal, xLocal, 1.0f, curTileLength);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(curTileLength * static_cast<int64_t>(sizeof(float))),
            0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->tileLength], yLocal, params);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    int64_t offset = 0;
    int64_t blockLength = 0;
    int64_t tileLength = 0;
};

}  // namespace

// Entry point consumed by the host bridge; the generated host stub exports
// `aclrtlaunch_graph_probe`.
extern "C" __global__ __aicore__ void graph_probe(GM_ADDR x, GM_ADDR y,
                                                  int64_t totalLength,
                                                  int64_t perCore,
                                                  int64_t tileLength)
{
    KernelGraphProbe op;
    op.Init(x, y, totalLength, perCore, tileLength);
    op.Process();
}
