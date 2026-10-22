// Native AscendC implementation of the DGL `copy_e_sum` graph operator used by
// the SE3-Transformer attention/convolution paths of RFdiffusion and RF2:
//
//   copy_e_sum(g, feat)[s, ...] = sum_{e in in_edges(s)} feat[e, ...]
//
// The kernel body is shared with `copy_e_mean` through
// `op_kernel/segment_reduce_common.h`; this translation unit only instantiates
// the sum variant so the official `ascendc_library` target generates its own
// `aclrtlaunch_copy_e_sum` entry point.  The CSR table is produced by the native
// `edge_topology` kernel; there is no Torch/DGL/NumPy/CPU fallback here.

#include "segment_reduce_common.h"

extern "C" __global__ __aicore__ void copy_e_sum(
    GM_ADDR feat, GM_ADDR indices, GM_ADDR offsets, GM_ADDR out,
    int64_t numEdges, int64_t numNodes, int64_t width, int64_t numBlocks)
{
    ascend_graph_segment::KernelEdgeSegmentReduce<false> op;
    op.Init(feat, indices, offsets, out, numEdges, numNodes, width, numBlocks);
    op.Process();
}
