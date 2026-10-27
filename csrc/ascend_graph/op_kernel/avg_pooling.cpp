// Native AscendC implementation of the DGL `AvgPooling` graph operator used by
// the SE3-Transformer pooling surface (``GPooling(pool='avg')`` ->
// ``dgl.nn.AvgPooling`` -> ``dgl.readout_nodes(op='mean')`` ->
// ``dgl.ops.segment.segment_reduce``):
//
//   avg_pooling(feat, seglen)[b, ...]
//       = (sum_{i in segment b} feat[i, ...]) / clamp(len_b, 1, num_rows)
//
// The kernel body is shared with `max_pooling` through
// `op_kernel/pooling_common.h`; this translation unit only instantiates the
// average variant so the official `ascendc_library` target generates its own
// `aclrtlaunch_avg_pooling` entry point.  The segment table is scanned on the
// device by the same kernel; there is no Torch/DGL/NumPy/CPU fallback here.

#include "pooling_common.h"

extern "C" __global__ __aicore__ void avg_pooling(
    GM_ADDR feat, GM_ADDR seglen, GM_ADDR out,
    int64_t numRows, int64_t numSegs, int64_t width, int64_t numBlocks)
{
    ascend_graph_pool::KernelGlobalPool<false> op;
    op.Init(feat, seglen, out, numRows, numSegs, width, numBlocks);
    op.Process();
}
