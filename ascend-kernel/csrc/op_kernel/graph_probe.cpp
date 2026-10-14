// Minimal native AscendC kernel used to prove the build -> register -> launch
// -> synchronize -> copy-back path for the RFantibody DGL precision migration.
//
// This iteration deliberately implements only a trivial, deterministic
// operation (`y = x + 1`) so that data movement and the official
// `ascendc_library` launch contract can be established before any DGL
// operator semantics are ported.  Two-level tiling:
//   * block split across AIV cores (formerLength / tailLength from the host),
//   * in-core UB tiles aligned to 32 bytes, moved with DataCopyPad.

#include "kernel_operator.h"

namespace {
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelGraphProbe {
public:
    __aicore__ inline KernelGraphProbe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int64_t formerNum, int64_t formerLength,
                                int64_t tailLength, int64_t tileLength)
    {
        const int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < formerNum) {
            this->blockLength = formerLength;
            const int64_t offset = formerLength * blockIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, formerLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, formerLength);
        } else {
            this->blockLength = tailLength;
            const int64_t tailIdx = blockIdx - formerNum;
            const int64_t offset = formerLength * formerNum + tailLength * tailIdx;
            xGm.SetGlobalBuffer((__gm__ T *)x + offset, tailLength);
            yGm.SetGlobalBuffer((__gm__ T *)y + offset, tailLength);
        }

        this->tileLength = tileLength;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength <= 0) {
            return;
        }
        const int64_t tileNum = (this->blockLength + this->tileLength - 1) / this->tileLength;
        for (int64_t i = 0; i < tileNum; ++i) {
            const int64_t remaining = this->blockLength - i * this->tileLength;
            const int64_t curTileLength =
                (remaining < this->tileLength) ? remaining : this->tileLength;
            CopyIn(i, curTileLength);
            Compute(i, curTileLength);
            CopyOut(i, curTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams copyInParams{
            1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], copyInParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t /*progress*/, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::Adds(yLocal, xLocal, static_cast<T>(1), curTileLength);
        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t curTileLength)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams copyOutParams{
            1, static_cast<uint32_t>(curTileLength * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->tileLength], yLocal, copyOutParams);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    int64_t blockLength = 0;
    int64_t tileLength = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void graph_probe(GM_ADDR x, GM_ADDR y,
                                                  int64_t formerNum,
                                                  int64_t formerLength,
                                                  int64_t tailLength,
                                                  int64_t tileLength)
{
    KernelGraphProbe<float> op;
    op.Init(x, y, formerNum, formerLength, tailLength, tileLength);
    op.Process();
}
