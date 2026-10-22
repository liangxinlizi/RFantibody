// Native AscendC implementation of the DGL `MaxPooling` graph operator used by
// the SE3-Transformer pooling surface (``GPooling(pool='max')`` ->
// ``dgl.nn.MaxPooling`` -> ``dgl.readout_nodes(op='max')`` ->
// ``dgl.ops.segment.segment_reduce``):
//
//   max_pooling(feat, seglen)[b, ...] = max_{i in segment b} feat[i, ...]
//
// with DGL's `replace_inf_with_zero` semantics for an empty segment (exactly 0,
// never -inf).  The kernel body is shared with `avg_pooling` through
// `op_kernel/pooling_common.h`; this translation unit only instantiates the max
// variant so the official `ascendc_library` target generates its own
// `aclrtlaunch_max_pooling` entry point.  No Torch/DGL/NumPy/CPU fallback.

#include "pooling_common.h"

extern "C" __global__ __aicore__ void max_pooling(
    GM_ADDR feat, GM_ADDR seglen, GM_ADDR out,
    int64_t numRows, int64_t numSegs, int64_t width, int64_t numBlocks)
{
    ascend_graph_pool::KernelGlobalPool<true> op;
    op.Init(feat, seglen, out, numRows, numSegs, width, numBlocks);
    op.Process();
}
