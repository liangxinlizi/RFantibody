// Host-side bridge between the public PyTorch dispatcher (`torch.ops.ascend_graph`)
// and the native AscendC kernels produced by the installed official CANN route
// (`csrc/ascend_graph/CMakeLists.txt` -> `ascendc_library(ascend_graph SHARED)`).
//
// The official target owns device compilation, the generated host stub and the
// `aclrtlaunch_graph_probe` / `aclrtlaunch_edge_dot_v` /
// `aclrtlaunch_edge_topology` / `aclrtlaunch_copy_e_sum` /
// `aclrtlaunch_copy_e_mean` / `aclrtlaunch_edge_softmax_dst` /
// `aclrtlaunch_avg_pooling` / `aclrtlaunch_max_pooling` entry points.  This
// file only validates tensor contracts, computes the core split, fetches the
// current NPU stream and calls those entry points; it contains no device
// computation and no Torch-primitive emulation of a graph operator (no
// `at::gather`, `query[dst]`, `scatter`, `index_add`, `sum`, `max` or `exp` on
// the operator path).
//
// Every operator is registered for dispatch key `PrivateUse1`, so a CPU/CUDA
// tensor cannot silently take this route and an NPU tensor cannot silently fall
// back to a CPU kernel.

#include <cstdint>
#include <limits>
#include <tuple>

#include <torch/extension.h>
#include <acl/acl.h>

#include "aclrtlaunch_triple_chevrons_func.h"

namespace ascend_graph_ops {

namespace {

// float32 elements per 32-byte vector block.
constexpr int64_t kElementAlign = 8;
// Element count per UB tile for the route-control kernel.  Kept small enough for
// two double-buffered float32 queues.
constexpr int64_t kProbeTileElements = 2048;

// Number of AIV cores usable on the tensor's device.  A failed capability query
// falls back to a *core count* of one; it is never a device fallback.
int64_t vector_core_count(int64_t deviceIndex) {
    int64_t coreNum = 0;
    const aclError capRet = aclGetDeviceCapability(
        static_cast<uint32_t>(deviceIndex), ACL_DEVICE_INFO_VECTOR_CORE_NUM,
        &coreNum);
    if (capRet != ACL_SUCCESS || coreNum <= 0) {
        coreNum = 1;
    }
    return coreNum;
}

void check_npu_float32(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat,
                name, ": only float32 is supported, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, ": tensor must be on an NPU, got ", tensor.device());
}

void check_npu_int64(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kLong,
                name, ": only int64 is supported, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, ": tensor must be on an NPU, got ", tensor.device());
}

void check_npu_int32(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt,
                name, ": only int32 is supported, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, ": tensor must be on an NPU, got ", tensor.device());
}

}  // namespace

// Route control: `y = x + 1` executed by the native `graph_probe` AscendC
// kernel.  Its only purpose is to prove the build -> register -> launch ->
// synchronize -> copy-back chain for the observed SoC.
at::Tensor graph_probe(const at::Tensor& self, int64_t stream) {
    check_npu_float32(self, "graph_probe");
    TORCH_CHECK(self.dim() >= 1, "graph_probe: input must have at least one dim");
    TORCH_CHECK(self.is_contiguous(), "graph_probe: input must be contiguous");

    at::Tensor output = at::empty_like(self);
    const int64_t totalLength = self.numel();
    if (totalLength == 0) {
        return output;
    }

    int64_t blockDim = vector_core_count(self.device().index());
    if (blockDim > totalLength) {
        blockDim = totalLength;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    const int64_t perCore = (totalLength + blockDim - 1) / blockDim;
    const int64_t launchBlocks = (totalLength + perCore - 1) / perCore;

    int64_t tileLength = perCore < kProbeTileElements ? perCore
                                                      : kProbeTileElements;
    tileLength = ((tileLength + kElementAlign - 1) / kElementAlign) * kElementAlign;
    if (tileLength < kElementAlign) {
        tileLength = kElementAlign;
    }

    aclrtStream aclStream =
        reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    void* xPtr = const_cast<void*>(self.const_data_ptr());
    void* yPtr = output.data_ptr();

    const uint32_t ret = aclrtlaunch_graph_probe(
        static_cast<uint32_t>(launchBlocks), aclStream, xPtr, yPtr, totalLength,
        perCore, tileLength);
    TORCH_CHECK(ret == 0, "graph_probe: AscendC kernel launch failed, ret=", ret);

    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS,
                "graph_probe: stream synchronize failed, ret=", syncRet);
    return output;
}

// General per-edge DGL `e_dot_v(g, key, query)`:
//
//   out[e, h] = sum_{j < numFeat} key[e, h, j] * query[dst[e], h, j]
//
// `key` is the flattened `[E, numHeads*numFeat]` edge feature, `query` is the
// flattened `[N, numHeads*numFeat]` node feature and `dst` is the `[E]`
// destination index of every edge.  The gather, multiply and reduction all run
// in the native `edge_dot_v` kernel; the Python wrapper restores DGL's output
// shape `key.shape[:-1] + (1,)`.  There is no Torch graph operator, no CPU
// reduction and no fallback on the path.
at::Tensor e_dot_v(const at::Tensor& key, const at::Tensor& query,
                   const at::Tensor& dst, int64_t num_heads, int64_t num_feat,
                   int64_t num_nodes, int64_t stream) {
    check_npu_float32(key, "e_dot_v: key");
    check_npu_float32(query, "e_dot_v: query");
    check_npu_int64(dst, "e_dot_v: dst");
    TORCH_CHECK(key.dim() == 2, "e_dot_v: key must be 2-D, got ", key.dim());
    TORCH_CHECK(query.dim() == 2, "e_dot_v: query must be 2-D, got ", query.dim());
    TORCH_CHECK(key.is_contiguous(), "e_dot_v: key must be contiguous");
    TORCH_CHECK(query.is_contiguous(), "e_dot_v: query must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "e_dot_v: dst must be contiguous");
    TORCH_CHECK(num_heads >= 1, "e_dot_v: num_heads must be >= 1, got ",
                num_heads);
    TORCH_CHECK(num_feat >= 1, "e_dot_v: num_feat must be >= 1, got ", num_feat);
    TORCH_CHECK(num_nodes >= 0, "e_dot_v: num_nodes must be non-negative");

    at::Tensor key_c = key.contiguous();
    at::Tensor query_c = query.contiguous();
    at::Tensor dst_c = dst.contiguous();

    const int64_t numEdges = key_c.size(0);
    const int64_t width = key_c.size(1);
    TORCH_CHECK(width == num_heads * num_feat,
                "e_dot_v: key.size(1) must equal num_heads*num_feat, got ", width,
                " and ", num_heads * num_feat);
    TORCH_CHECK(query_c.size(1) == width,
                "e_dot_v: key and query must share the flattened width, got ",
                width, " and ", query_c.size(1));
    TORCH_CHECK(query_c.size(0) == num_nodes,
                "e_dot_v: query must have one row per node, got ",
                query_c.size(0), " rows and num_nodes=", num_nodes);
    TORCH_CHECK(dst_c.dim() == 1 && dst_c.numel() == numEdges,
                "e_dot_v: dst must have exactly one entry per edge");

    at::Tensor output = at::empty({numEdges, num_heads}, key_c.options());
    if (numEdges == 0 || width == 0) {
        return output;
    }

    int64_t blockDim = vector_core_count(key_c.device().index());
    if (blockDim > numEdges) {
        blockDim = numEdges;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    aclrtStream aclStream =
        reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    const uint32_t ret = aclrtlaunch_edge_dot_v(
        static_cast<uint32_t>(blockDim), aclStream, key_c.data_ptr(),
        query_c.data_ptr(), dst_c.data_ptr(), output.data_ptr(), numEdges,
        num_heads, num_feat, num_nodes, blockDim);
    TORCH_CHECK(ret == 0, "e_dot_v: AscendC kernel launch failed, ret=", ret);

    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS,
                "e_dot_v: stream synchronize failed, ret=", syncRet);
    return output;
}

// Native edge topology: `offsets` int64 [N+1], `indices` int32 [E] and the exact
// in-degree vector `degrees` int64 [N], produced by the deterministic
// single-block `edge_topology` AscendC counting sort.  The host allocates the
// buffers only; it performs no initialization and no index computation.  This
// is the native source of `DGLGraph.in_degrees()` and of the CSR table consumed
// by `copy_e_sum`/`copy_e_mean`, so no Torch reduction stands in for it.
std::tuple<at::Tensor, at::Tensor, at::Tensor> edge_topology(
    const at::Tensor& dst, int64_t num_nodes, int64_t stream) {
    check_npu_int64(dst, "edge_topology: dst");
    TORCH_CHECK(dst.dim() == 1,
                "edge_topology: dst must be 1-D, got ", dst.dim());
    TORCH_CHECK(dst.is_contiguous(),
                "edge_topology: dst must be contiguous");
    TORCH_CHECK(num_nodes >= 0,
                "edge_topology: num_nodes must be non-negative, got ",
                num_nodes);

    at::Tensor dst_c = dst.contiguous();
    const int64_t numEdges = dst_c.numel();
    TORCH_CHECK(
        numEdges <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "edge_topology: numEdges must fit the int32 indices dtype, got ",
        numEdges);
    TORCH_CHECK(num_nodes > 0 || numEdges == 0,
                "edge_topology: a graph with no nodes cannot have edges, got ",
                numEdges, " edges and num_nodes=0");

    at::Tensor offsets = at::empty({num_nodes + 1}, dst_c.options());
    at::Tensor degrees_storage =
        at::empty({num_nodes > 0 ? num_nodes : 1}, dst_c.options());
    at::Tensor degrees = degrees_storage.narrow(0, 0, num_nodes);
    at::Tensor cursor = at::empty({num_nodes > 0 ? num_nodes : 1},
                                  dst_c.options());
    at::Tensor indices_storage = at::empty({numEdges > 0 ? numEdges : 1},
                                           dst_c.options().dtype(at::kInt));
    at::Tensor indices = indices_storage.narrow(0, 0, numEdges);

    aclrtStream aclStream =
        reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    // `dst`/`indices` can be empty; pass a valid non-null dummy region instead
    // of a null pointer.  The kernel never dereferences them when E == 0.
    void* dstPtr = numEdges > 0 ? dst_c.data_ptr() : cursor.data_ptr();
    const uint32_t ret = aclrtlaunch_edge_topology(
        1U, aclStream, dstPtr, offsets.data_ptr(), indices_storage.data_ptr(),
        degrees_storage.data_ptr(), cursor.data_ptr(), numEdges, num_nodes);
    TORCH_CHECK(ret == 0, "edge_topology: AscendC kernel launch failed, ret=",
                ret);

    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS,
                "edge_topology: stream synchronize failed, ret=", syncRet);
    return std::make_tuple(offsets, indices, degrees);
}

namespace {

enum class ReduceKind { kSum, kMean };

// Shared host side of `copy_e_sum`/`copy_e_mean`: validation, output allocation
// and launch of the corresponding native entry point over the native CSR table.
// The reduction itself (accumulation, mean denominator) runs entirely in the
// device kernel.
at::Tensor segment_reduce(ReduceKind kind, const at::Tensor& feat,
                          const at::Tensor& indices, const at::Tensor& offsets,
                          int64_t num_nodes, int64_t stream) {
    const char* opName =
        kind == ReduceKind::kSum ? "copy_e_sum" : "copy_e_mean";
    check_npu_float32(feat, opName);
    check_npu_int32(indices, opName);
    check_npu_int64(offsets, opName);
    TORCH_CHECK(feat.dim() == 2,
                opName, ": feat must be 2-D [E, W], got ", feat.dim());
    TORCH_CHECK(feat.is_contiguous(),
                opName, ": feat must be contiguous");
    TORCH_CHECK(indices.dim() == 1,
                opName, ": indices must be 1-D, got ", indices.dim());
    TORCH_CHECK(indices.is_contiguous(),
                opName, ": indices must be contiguous");
    TORCH_CHECK(offsets.dim() == 1,
                opName, ": offsets must be 1-D, got ", offsets.dim());
    TORCH_CHECK(offsets.is_contiguous(),
                opName, ": offsets must be contiguous");
    TORCH_CHECK(num_nodes >= 0,
                opName, ": num_nodes must be non-negative, got ", num_nodes);

    const int64_t numEdges = feat.size(0);
    const int64_t width = feat.size(1);
    TORCH_CHECK(indices.numel() == numEdges,
                opName, ": indices must have one entry per edge, got ",
                indices.numel(), " and ", numEdges);
    TORCH_CHECK(offsets.numel() == num_nodes + 1,
                opName, ": offsets must have num_nodes+1 entries, got ",
                offsets.numel(), " and num_nodes=", num_nodes);

    at::Tensor output = at::empty({num_nodes, width}, feat.options());
    if (num_nodes == 0 || width == 0) {
        return output;
    }

    int64_t blockDim = vector_core_count(feat.device().index());
    if (blockDim > num_nodes) {
        blockDim = num_nodes;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    aclrtStream aclStream =
        reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    // Empty edge set: the kernel performs no row load, but the launch arguments
    // must still be valid pointers.
    void* featPtr = feat.numel() > 0
                        ? const_cast<void*>(feat.const_data_ptr())
                        : output.data_ptr();
    void* indicesPtr = indices.numel() > 0
                           ? const_cast<void*>(indices.const_data_ptr())
                           : output.data_ptr();

    const uint32_t ret =
        kind == ReduceKind::kSum
            ? aclrtlaunch_copy_e_sum(static_cast<uint32_t>(blockDim), aclStream,
                                     featPtr, indicesPtr, offsets.data_ptr(),
                                     output.data_ptr(), numEdges, num_nodes,
                                     width, blockDim)
            : aclrtlaunch_copy_e_mean(
                  static_cast<uint32_t>(blockDim), aclStream, featPtr,
                  indicesPtr, offsets.data_ptr(), output.data_ptr(), numEdges,
                  num_nodes, width, blockDim);
    TORCH_CHECK(ret == 0, opName, ": AscendC kernel launch failed, ret=", ret);

    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS, opName,
                ": stream synchronize failed, ret=", syncRet);
    return output;
}

}  // namespace

// Native `dgl.ops.copy_e_sum(g, feat)` over the native CSR table.
at::Tensor copy_e_sum(const at::Tensor& feat, const at::Tensor& indices,
                      const at::Tensor& offsets, int64_t num_nodes,
                      int64_t stream) {
    return segment_reduce(ReduceKind::kSum, feat, indices, offsets, num_nodes,
                          stream);
}

// Native `dgl.ops.copy_e_mean(g, feat)` over the native CSR table: the
// per-destination sum divided by `clamp(in_degree, 1, max(E, 1))`, exactly as
// DGL 1.1.3 defines it.
at::Tensor copy_e_mean(const at::Tensor& feat, const at::Tensor& indices,
                       const at::Tensor& offsets, int64_t num_nodes,
                       int64_t stream) {
    return segment_reduce(ReduceKind::kMean, feat, indices, offsets, num_nodes,
                          stream);
}

// Native `dgl.ops.edge_softmax(g, logits, norm_by='dst')` over the native CSR
// table.
//
// `logits` is the flattened `[E, W]` edge weight (W = heads * channels),
// `indices` int32 `[E]` and `offsets` int64 `[N+1]` are the native CSR table of
// `edge_topology`, and the result is `[E, W]`.  The per-destination maximum, the
// per-destination sum of exponentials and the division all run inside the
// native `edge_softmax_dst` kernel; the host only validates, allocates and
// launches.  There is no Torch/DGL/CPU softmax on the path.
at::Tensor edge_softmax_dst(const at::Tensor& logits, const at::Tensor& indices,
                            const at::Tensor& offsets, int64_t num_nodes,
                            int64_t stream) {
    const char* opName = "edge_softmax_dst";
    check_npu_float32(logits, opName);
    check_npu_int32(indices, opName);
    check_npu_int64(offsets, opName);
    TORCH_CHECK(logits.dim() == 2,
                opName, ": logits must be 2-D [E, W], got ", logits.dim());
    TORCH_CHECK(logits.is_contiguous(), opName, ": logits must be contiguous");
    TORCH_CHECK(indices.dim() == 1,
                opName, ": indices must be 1-D, got ", indices.dim());
    TORCH_CHECK(indices.is_contiguous(),
                opName, ": indices must be contiguous");
    TORCH_CHECK(offsets.dim() == 1,
                opName, ": offsets must be 1-D, got ", offsets.dim());
    TORCH_CHECK(offsets.is_contiguous(),
                opName, ": offsets must be contiguous");
    TORCH_CHECK(num_nodes >= 0,
                opName, ": num_nodes must be non-negative, got ", num_nodes);

    const int64_t numEdges = logits.size(0);
    const int64_t width = logits.size(1);
    TORCH_CHECK(indices.numel() == numEdges,
                opName, ": indices must have one entry per edge, got ",
                indices.numel(), " and ", numEdges);
    TORCH_CHECK(offsets.numel() == num_nodes + 1,
                opName, ": offsets must have num_nodes+1 entries, got ",
                offsets.numel(), " and num_nodes=", num_nodes);

    at::Tensor output = at::empty({numEdges, width}, logits.options());
    if (numEdges == 0 || width == 0 || num_nodes == 0) {
        return output;
    }

    int64_t blockDim = vector_core_count(logits.device().index());
    if (blockDim > num_nodes) {
        blockDim = num_nodes;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    aclrtStream aclStream =
        reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    const uint32_t ret = aclrtlaunch_edge_softmax_dst(
        static_cast<uint32_t>(blockDim), aclStream,
        const_cast<void*>(logits.const_data_ptr()),
        const_cast<void*>(indices.const_data_ptr()),
        const_cast<void*>(offsets.const_data_ptr()), output.data_ptr(),
        numEdges, num_nodes, width, blockDim);
    TORCH_CHECK(ret == 0, opName, ": AscendC kernel launch failed, ret=", ret);

    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS, opName,
                ": stream synchronize failed, ret=", syncRet);
    return output;
}

// Shared host side of `avg_pooling`/`max_pooling` (DGL `AvgPooling` and
// `MaxPooling` over contiguous node features): validation, output allocation
// and launch of the corresponding native entry point.  `feat` is the flattened
// `[N, W]` node feature, `seglen` the int64 `[numSegments]` segment table whose
// sum must equal `N`.  The segment scan, every accumulation/max and the DGL-exact
// mean denominator run entirely inside the device kernel; the host performs no
// index computation and no reduction (no Torch `sum`/`max`/`scatter` on the
// path).  A CPU/CUDA tensor is rejected by `check_npu_*`, never moved.
at::Tensor global_pool(bool useMax, const at::Tensor& feat,
                       const at::Tensor& seglen, int64_t numSegments,
                       int64_t stream) {
    const char* opName = useMax ? "max_pooling" : "avg_pooling";
    check_npu_float32(feat, opName);
    check_npu_int64(seglen, opName);
    TORCH_CHECK(feat.dim() == 2,
                opName, ": feat must be 2-D [N, W], got ", feat.dim());
    TORCH_CHECK(feat.is_contiguous(), opName, ": feat must be contiguous");
    TORCH_CHECK(seglen.dim() == 1,
                opName, ": seglen must be 1-D, got ", seglen.dim());
    TORCH_CHECK(seglen.is_contiguous(), opName, ": seglen must be contiguous");
    TORCH_CHECK(numSegments >= 1,
                opName, ": num_segments must be >= 1, got ", numSegments);
    TORCH_CHECK(seglen.numel() == numSegments,
                opName, ": seglen must have one entry per segment, got ",
                seglen.numel(), " and num_segments=", numSegments);
    TORCH_CHECK(feat.device() == seglen.device(),
                opName, ": feat and seglen must live on the same device, got ",
                feat.device(), " and ", seglen.device());

    const int64_t numRows = feat.size(0);
    const int64_t width = feat.size(1);
    TORCH_CHECK(numRows >= 1,
                opName, ": feat must have at least one row, got ", numRows);

    at::Tensor output = at::empty({numSegments, width}, feat.options());
    if (width == 0) {
        return output;
    }

    int64_t blockDim = vector_core_count(feat.device().index());
    if (blockDim > numSegments) {
        blockDim = numSegments;
    }
    if (blockDim < 1) {
        blockDim = 1;
    }

    aclrtStream aclStream =
        reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    void* featPtr = const_cast<void*>(feat.const_data_ptr());
    void* seglenPtr = const_cast<void*>(seglen.const_data_ptr());

    uint32_t ret = 0;
    if (useMax) {
        ret = aclrtlaunch_max_pooling(static_cast<uint32_t>(blockDim), aclStream,
                                      featPtr, seglenPtr, output.data_ptr(),
                                      numRows, numSegments, width, blockDim);
    } else {
        ret = aclrtlaunch_avg_pooling(static_cast<uint32_t>(blockDim), aclStream,
                                      featPtr, seglenPtr, output.data_ptr(),
                                      numRows, numSegments, width, blockDim);
    }
    TORCH_CHECK(ret == 0, opName, ": AscendC kernel launch failed, ret=", ret);

    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS, opName,
                ": stream synchronize failed, ret=", syncRet);
    return output;
}

// Native `dgl.nn.AvgPooling()(graph, feat)` (DGL `readout_nodes(op='mean')`):
// one row per segment, the segment sum divided by `clamp(seglen, 1, N)` and an
// exact zero for a zero-length segment.
at::Tensor avg_pooling(const at::Tensor& feat, const at::Tensor& seglen,
                       int64_t num_segments, int64_t stream) {
    return global_pool(false, feat, seglen, num_segments, stream);
}

// Native `dgl.nn.MaxPooling()(graph, feat)` (DGL `readout_nodes(op='max')`):
// one row per segment with DGL's `replace_inf_with_zero` semantics, i.e. an
// exact zero (never -inf) for a zero-length segment.
at::Tensor max_pooling(const at::Tensor& feat, const at::Tensor& seglen,
                       int64_t num_segments, int64_t stream) {
    return global_pool(true, feat, seglen, num_segments, stream);
}

}  // namespace ascend_graph_ops

TORCH_LIBRARY_FRAGMENT(ascend_graph, m) {
    m.def("graph_probe(Tensor x, int stream) -> Tensor");
    m.def(
        "e_dot_v(Tensor key, Tensor query, Tensor dst, int num_heads, "
        "int num_feat, int num_nodes, int stream) -> Tensor");
    m.def("edge_topology(Tensor dst, int num_nodes, int stream) -> "
          "(Tensor, Tensor, Tensor)");
    m.def(
        "copy_e_sum(Tensor feat, Tensor indices, Tensor offsets, "
        "int num_nodes, int stream) -> Tensor");
    m.def(
        "copy_e_mean(Tensor feat, Tensor indices, Tensor offsets, "
        "int num_nodes, int stream) -> Tensor");
    m.def(
        "edge_softmax_dst(Tensor logits, Tensor indices, Tensor offsets, "
        "int num_nodes, int stream) -> Tensor");
    m.def(
        "avg_pooling(Tensor feat, Tensor seglen, int num_segments, "
        "int stream) -> Tensor");
    m.def(
        "max_pooling(Tensor feat, Tensor seglen, int num_segments, "
        "int stream) -> Tensor");
}

TORCH_LIBRARY_IMPL(ascend_graph, PrivateUse1, m) {
    m.impl("graph_probe", TORCH_FN(ascend_graph_ops::graph_probe));
    m.impl("e_dot_v", TORCH_FN(ascend_graph_ops::e_dot_v));
    m.impl("edge_topology", TORCH_FN(ascend_graph_ops::edge_topology));
    m.impl("copy_e_sum", TORCH_FN(ascend_graph_ops::copy_e_sum));
    m.impl("copy_e_mean", TORCH_FN(ascend_graph_ops::copy_e_mean));
    m.impl("edge_softmax_dst", TORCH_FN(ascend_graph_ops::edge_softmax_dst));
    m.impl("avg_pooling", TORCH_FN(ascend_graph_ops::avg_pooling));
    m.impl("max_pooling", TORCH_FN(ascend_graph_ops::max_pooling));
}

// CPython requires a module export function for `import _ascend_graph`.  Loading
// this shared object already runs the `TORCH_LIBRARY_*` static initializers that
// register the schemas and their `PrivateUse1` implementations.
PYBIND11_MODULE(_ascend_graph, m) {
    m.doc() = "Native AscendC graph operators for the RFantibody NPU migration; "
              "module init registers ascend_graph::* via TORCH_LIBRARY.";
}
