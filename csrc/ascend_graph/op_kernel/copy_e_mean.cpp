// Native AscendC implementation of the DGL `copy_e_mean` graph operator used by
// the SE3-Transformer convolution path of RFdiffusion and RF2:
//
//   copy_e_mean(g, feat)[s, ...] =
//       copy_e_sum(g, feat)[s, ...] / clamp(in_degree(s), 1, max(E, 1))
//
// The kernel body is shared with `copy_e_sum` through
// `op_kernel/segment_reduce_common.h`; this translation unit only instantiates
// the mean variant so the official `ascendc_library` target generates its own
// `aclrtlaunch_copy_e_mean` entry point.  The CSR table and the in-degrees come
// from the native `edge_topology` kernel; there is no Torch/DGL/NumPy/CPU
// fallback here.

#include "segment_reduce_common.h"

extern "C" __global__ __aicore__ void copy_e_mean(
    GM_ADDR feat, GM_ADDR indices, GM_ADDR offsets, GM_ADDR out,
    int64_t numEdges, int64_t numNodes, int64_t width, int64_t numBlocks)
{
    ascend_graph_segment::KernelEdgeSegmentReduce<true> op;
    op.Init(feat, indices, offsets, out, numEdges, numNodes, width, numBlocks);
    op.Process();
}
