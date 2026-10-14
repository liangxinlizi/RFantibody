// Host-side bridge between the public PyTorch dispatcher and the native
// AscendC kernels produced by the official `ascendc_library` route.
//
// `graph_probe.cpp`, `edge_softmax.cpp` and `graph_reduce.cpp` are compiled for
// the device and the generated host stub
// (`aclrtlaunch_triple_chevrons_func.h`) exposes C entry points
// `aclrtlaunch_graph_probe(...)`, `aclrtlaunch_edge_softmax(...)`,
// `aclrtlaunch_edge_softmax_prepare(...)`,
// `aclrtlaunch_edge_softmax_unpermute(...)` and
// `aclrtlaunch_graph_scatter_sum(...)`.  This file only computes the
// tiling, fetches the current NPU stream and calls those entry points; it
// contains no device computation and no Torch-primitive emulation of any graph
// operator.
//
// The DGL `edge_softmax(..., norm_by='dst')` product path is fully native:
// `edge_softmax_prepare` builds the stable counting-sort segment table
// (`start`), the original-edge position (`pos`) and the row-sorted logits on the
// NPU, `edge_softmax` runs the per-destination softmax and `edge_softmax_unpermute`
// restores the original edge order.  There is no `at::sort`, `index_select`,
// `scatter_add_`, `cumsum`, `index_copy_` or any other Torch graph operator on
// the path, and no CPU/CUDA fallback.  Likewise `npu::e_dot_v` computes
// `(key * query[dst]).sum(-1, keepdim=True)` entirely inside `edge_dot_v.cpp`,
// with no `at::gather`/`query[dst]` composition on the product path.

#include <cstdint>
#include <vector>

#include <torch/extension.h>
#include <acl/acl.h>

#include "aclrtlaunch_triple_chevrons_func.h"

namespace ascend_kernel {

namespace {
// Element count per UB tile.  Kept well below the UB budget for two double
// buffered float32 queues (2 queues * 2 buffers * 2048 * 4 B = 32 KiB).
constexpr int64_t kTileElements = 2048;
// 32-byte vector alignment for float32.
constexpr int64_t kElementAlign = 8;
// Per-core byte split alignment (cache line).
constexpr int64_t kCacheLineBytes = 512;

// Number of AIV cores to partition the destination or edge range over.  The
// native adjacency prepare kernel is intentionally single-core (deterministic
// counting sort); the softmax and un-permute kernels are data parallel.
int64_t used_core_count(const at::Tensor& reference, int64_t work_items) {
    int64_t coreNum = 0;
    const aclError capRet = aclGetDeviceCapability(
        static_cast<uint32_t>(reference.device().index()),
        ACL_DEVICE_INFO_VECTOR_CORE_NUM, &coreNum);
    if (capRet != ACL_SUCCESS || coreNum <= 0) {
        coreNum = 1;
    }
    int64_t used = coreNum < work_items ? coreNum : work_items;
    if (used < 1) {
        used = 1;
    }
    return used;
}
}  // namespace

at::Tensor graph_probe(const at::Tensor& self, int64_t stream) {
    TORCH_CHECK(self.scalar_type() == at::kFloat,
                "graph_probe: only float32 is supported, got ", self.scalar_type());
    TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
                "graph_probe: input must be an NPU tensor, got ", self.device());
    TORCH_CHECK(self.is_contiguous(), "graph_probe: input must be contiguous");

    at::Tensor output = at::empty_like(self);
    const int64_t totalLength = self.numel();
    if (totalLength == 0) {
        return output;
    }

    int64_t coreNum = 0;
    const aclError capRet = aclGetDeviceCapability(
        static_cast<uint32_t>(self.device().index()),
        ACL_DEVICE_INFO_VECTOR_CORE_NUM, &coreNum);
    if (capRet != ACL_SUCCESS || coreNum <= 0) {
        coreNum = 1;
    }

    const int64_t perCore = (totalLength + coreNum - 1) / coreNum;
    const int64_t perCoreAlign =
        (perCore + kCacheLineBytes - 1) / kCacheLineBytes * kCacheLineBytes;
    const int64_t usedCoreNum = (totalLength + perCoreAlign - 1) / perCoreAlign;
    const int64_t formerNum = usedCoreNum - 1;
    const int64_t formerLength = perCoreAlign;
    const int64_t tailLength = totalLength - formerNum * formerLength;

    int64_t tileLength = kTileElements;
    if (tileLength > perCoreAlign) {
        tileLength = perCoreAlign;
    }
    if (tileLength < kElementAlign) {
        tileLength = kElementAlign;
    }

    aclrtStream aclStream = reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    void* xPtr = const_cast<void*>(self.const_data_ptr());
    void* yPtr = output.data_ptr();

    const uint32_t ret = aclrtlaunch_graph_probe(
        static_cast<uint32_t>(usedCoreNum), aclStream, xPtr, yPtr,
        formerNum, formerLength, tailLength, tileLength);
    TORCH_CHECK(ret == 0, "graph_probe: AscendC kernel launch failed, ret=", ret);
    return output;
}

// General per-destination softmax for DGL `edge_softmax(g, logits, norm_by='dst')`.
//
// The complete DGL `norm_by='dst'` computation -- stable destination grouping,
// per-destination maximum, exponentiation, denominator, division and the
// restoration of the original edge order -- is executed by native AscendC
// kernels on the input's NPU.  Destinations are normalised independently, so no
// graph-wide shift is applied and no `tiny`/`eps` denominator salting is needed.
at::Tensor edge_softmax(const at::Tensor& logits, const at::Tensor& dst,
                        int64_t num_nodes, int64_t stream) {
    TORCH_CHECK(logits.scalar_type() == at::kFloat,
                "edge_softmax: only float32 is supported, got ", logits.scalar_type());
    TORCH_CHECK(logits.device().type() == c10::DeviceType::PrivateUse1,
                "edge_softmax: logits must be an NPU tensor, got ", logits.device());
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
                "edge_softmax: dst must be an NPU tensor, got ", dst.device());
    TORCH_CHECK(logits.dim() == 2, "edge_softmax: logits must be 2-D, got ", logits.dim());
    TORCH_CHECK(logits.is_contiguous(), "edge_softmax: logits must be contiguous");
    TORCH_CHECK(dst.scalar_type() == at::kLong,
                "edge_softmax: dst must be int64, got ", dst.scalar_type());
    TORCH_CHECK(dst.is_contiguous(), "edge_softmax: dst must be contiguous");
    TORCH_CHECK(num_nodes >= 0, "edge_softmax: num_nodes must be non-negative");

    at::Tensor logits_c = logits.contiguous();
    at::Tensor dst_c = dst.contiguous();
    const int64_t numEdges = logits_c.size(0);
    const int64_t numFeat = logits_c.size(1);
    TORCH_CHECK(dst_c.dim() == 1 && dst_c.numel() == numEdges,
                "edge_softmax: dst must have exactly one entry per edge");

    at::Tensor output = at::empty_like(logits_c);
    if (numEdges == 0 || numFeat == 0 || num_nodes == 0) {
        return output;
    }

    const auto intOptions = logits_c.options().dtype(at::kInt);
    at::Tensor start = at::empty({num_nodes + 1}, intOptions);
    at::Tensor pos = at::empty({numEdges}, intOptions);
    at::Tensor logits_sorted = at::empty_like(logits_c);
    at::Tensor out_sorted = at::empty_like(logits_c);

    aclrtStream aclStream = reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));

    // 1. Native stable counting-sort adjacency preparation (one AIV core).
    uint32_t ret = aclrtlaunch_edge_softmax_prepare(
        /*blockDim=*/1, aclStream, dst_c.data_ptr(), logits_c.data_ptr(),
        start.data_ptr(), pos.data_ptr(), logits_sorted.data_ptr(),
        numEdges, numFeat, num_nodes);
    TORCH_CHECK(ret == 0,
                "edge_softmax: edge_softmax_prepare launch failed, ret=", ret);

    // 2. Native per-destination softmax over the row-sorted logits.
    const int64_t softmaxCores = used_core_count(logits_c, num_nodes);
    ret = aclrtlaunch_edge_softmax(
        static_cast<uint32_t>(softmaxCores), aclStream, logits_sorted.data_ptr(),
        start.data_ptr(), out_sorted.data_ptr(), numEdges, numFeat, num_nodes,
        softmaxCores);
    TORCH_CHECK(ret == 0, "edge_softmax: AscendC kernel launch failed, ret=", ret);

    // 3. Native un-permutation back to the original edge order.
    const int64_t unpermuteCores = used_core_count(logits_c, numEdges);
    ret = aclrtlaunch_edge_softmax_unpermute(
        static_cast<uint32_t>(unpermuteCores), aclStream, pos.data_ptr(),
        out_sorted.data_ptr(), output.data_ptr(), numEdges, numFeat,
        unpermuteCores);
    TORCH_CHECK(ret == 0,
                "edge_softmax: edge_softmax_unpermute launch failed, ret=", ret);

    // All three launches are raw stream-ordered `aclrtlaunch_*` calls, so the
    // producer->consumer order between them is the stream FIFO.  Draining the
    // stream here only releases the derived scratch buffers (`start`, `pos`,
    // `logits_sorted`, `out_sorted`) that are freed when this function returns;
    // it is not used to order any queued framework producer.
    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS,
                "edge_softmax: stream synchronize failed, ret=", syncRet);

    return output;
}

// General per-destination scatter reduction shared by copy_e_sum / copy_e_mean /
// avg_pool.  `meanMode=false` sums the incoming rows, `meanMode=true` divides
// each destination's sum by its in-degree (zero-degree destinations stay
// exactly 0).  The destination grouping is the native stable counting-sort
// table produced by the already-proven `edge_softmax_prepare` kernel, and the
// reduction itself is the native `graph_scatter_sum` AscendC kernel.  No Torch
// graph operator (index_add / scatter_add_ / gather) and no CPU/CUDA fallback
// is on the path.
at::Tensor scatter_reduce(const at::Tensor& msg, const at::Tensor& dst,
                          int64_t num_nodes, int64_t stream, bool meanMode) {
    TORCH_CHECK(msg.scalar_type() == at::kFloat,
                "scatter_reduce: only float32 is supported, got ", msg.scalar_type());
    TORCH_CHECK(msg.device().type() == c10::DeviceType::PrivateUse1,
                "scatter_reduce: msg must be an NPU tensor, got ", msg.device());
    TORCH_CHECK(msg.dim() == 2, "scatter_reduce: msg must be 2-D, got ", msg.dim());
    TORCH_CHECK(msg.is_contiguous(), "scatter_reduce: msg must be contiguous");
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
                "scatter_reduce: dst must be an NPU tensor, got ", dst.device());
    TORCH_CHECK(dst.scalar_type() == at::kLong,
                "scatter_reduce: dst must be int64, got ", dst.scalar_type());
    TORCH_CHECK(dst.is_contiguous(), "scatter_reduce: dst must be contiguous");
    TORCH_CHECK(num_nodes >= 0, "scatter_reduce: num_nodes must be non-negative");

    at::Tensor msg_c = msg.contiguous();
    at::Tensor dst_c = dst.contiguous();
    const int64_t numEdges = msg_c.size(0);
    const int64_t numFeat = msg_c.size(1);
    TORCH_CHECK(dst_c.dim() == 1 && dst_c.numel() == numEdges,
                "scatter_reduce: dst must have exactly one entry per edge");

    // An empty reduction domain is all zeros (sum and mean agree); otherwise
    // the native kernel writes every node row (including an exact 0 row for
    // zero-degree destinations), so `empty` cannot leak uninitialised values
    // and cannot mask a skipped write.
    at::Tensor output = at::empty({num_nodes, numFeat}, msg_c.options());
    if (numEdges == 0 || numFeat == 0 || num_nodes == 0) {
        return at::zeros({num_nodes, numFeat}, msg_c.options());
    }

    const auto intOptions = msg_c.options().dtype(at::kInt);
    at::Tensor start = at::empty({num_nodes + 1}, intOptions);
    at::Tensor pos = at::empty({numEdges}, intOptions);
    at::Tensor msg_sorted = at::empty_like(msg_c);

    aclrtStream aclStream = reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));

    // 1. Native stable counting-sort adjacency preparation (one AIV core).
    uint32_t ret = aclrtlaunch_edge_softmax_prepare(
        /*blockDim=*/1, aclStream, dst_c.data_ptr(), msg_c.data_ptr(),
        start.data_ptr(), pos.data_ptr(), msg_sorted.data_ptr(), numEdges,
        numFeat, num_nodes);
    TORCH_CHECK(ret == 0,
                "scatter_reduce: edge_softmax_prepare launch failed, ret=", ret);

    // 2. Native per-destination sum/mean over the row-sorted messages.
    const int64_t cores = used_core_count(msg_c, num_nodes);
    ret = aclrtlaunch_graph_scatter_sum(
        static_cast<uint32_t>(cores), aclStream, msg_sorted.data_ptr(),
        start.data_ptr(), output.data_ptr(), numEdges, numFeat, num_nodes, cores,
        meanMode ? 1 : 0);
    TORCH_CHECK(ret == 0,
                "scatter_reduce: graph_scatter_sum launch failed, ret=", ret);

    // Both launches are raw stream-ordered `aclrtlaunch_*` calls; draining the
    // stream only releases the derived scratch buffers (`start`, `pos`,
    // `msg_sorted`) that are freed when this function returns.
    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS,
                "scatter_reduce: stream synchronize failed, ret=", syncRet);

    return output;
}

at::Tensor copy_e_sum(const at::Tensor& msg, const at::Tensor& dst,
                      int64_t num_nodes, int64_t stream) {
    return scatter_reduce(msg, dst, num_nodes, stream, /*meanMode=*/false);
}

at::Tensor copy_e_mean(const at::Tensor& msg, const at::Tensor& dst,
                       int64_t num_nodes, int64_t stream) {
    return scatter_reduce(msg, dst, num_nodes, stream, /*meanMode=*/true);
}

at::Tensor avg_pool(const at::Tensor& msg, const at::Tensor& dst,
                    int64_t num_nodes, int64_t stream) {
    return scatter_reduce(msg, dst, num_nodes, stream, /*meanMode=*/true);
}

// General per-edge DGL `e_dot_v(g, key, query)`.
//
// `key` is the flattened `[E, H*feat]` edge feature, `query` is the flattened
// `[N, H*feat]` node feature and `dst` is the `[E]` destination index.  For every
// edge `e` and head `h` the native `edge_dot_v` kernel computes
//
//     out[e, h] = sum_{j < feat} key[e, h, j] * query[dst[e], h, j]
//
// and the result is reshaped by the Python wrapper to DGL's
// `key.shape[:-1] + (1,)`.  The whole gather/multiply/reduce runs on the NPU:
// there is no `at::gather`, no `query[dst]` index and no Torch reduction on the
// path, and no CPU/CUDA fallback.  `num_nodes` is taken from `query` (one row per
// node), exactly as DGL indexes `query` by `dst`.
at::Tensor edge_dot_v(const at::Tensor& key, const at::Tensor& query,
                      const at::Tensor& dst, int64_t feat, int64_t stream) {
    TORCH_CHECK(key.scalar_type() == at::kFloat,
                "e_dot_v: only float32 is supported, got ", key.scalar_type());
    TORCH_CHECK(key.device().type() == c10::DeviceType::PrivateUse1,
                "e_dot_v: key must be an NPU tensor, got ", key.device());
    TORCH_CHECK(key.dim() == 2, "e_dot_v: key must be 2-D, got ", key.dim());
    TORCH_CHECK(key.is_contiguous(), "e_dot_v: key must be contiguous");
    TORCH_CHECK(query.scalar_type() == at::kFloat,
                "e_dot_v: only float32 is supported, got ", query.scalar_type());
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
                "e_dot_v: query must be an NPU tensor, got ", query.device());
    TORCH_CHECK(query.dim() == 2, "e_dot_v: query must be 2-D, got ", query.dim());
    TORCH_CHECK(query.is_contiguous(), "e_dot_v: query must be contiguous");
    TORCH_CHECK(dst.scalar_type() == at::kLong,
                "e_dot_v: dst must be int64, got ", dst.scalar_type());
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
                "e_dot_v: dst must be an NPU tensor, got ", dst.device());
    TORCH_CHECK(dst.is_contiguous(), "e_dot_v: dst must be contiguous");
    TORCH_CHECK(feat >= 1, "e_dot_v: feat must be >= 1, got ", feat);

    at::Tensor key_c = key.contiguous();
    at::Tensor query_c = query.contiguous();
    at::Tensor dst_c = dst.contiguous();
    const int64_t numEdges = key_c.size(0);
    const int64_t width = key_c.size(1);
    const int64_t numNodes = query_c.size(0);
    TORCH_CHECK(query_c.size(1) == width,
                "e_dot_v: key and query must share the flattened width, got ",
                width, " and ", query_c.size(1));
    TORCH_CHECK(width % feat == 0,
                "e_dot_v: key.size(1) must be divisible by feat, got ", width,
                " and ", feat);
    TORCH_CHECK(dst_c.dim() == 1 && dst_c.numel() == numEdges,
                "e_dot_v: dst must have exactly one entry per edge");

    const int64_t numHeads = width / feat;
    at::Tensor output = at::empty({numEdges, numHeads}, key_c.options());
    if (numEdges == 0 || width == 0) {
        return output;
    }

    aclrtStream aclStream =
        reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));
    const int64_t cores = used_core_count(key_c, numEdges);

    const uint32_t ret = aclrtlaunch_edge_dot_v(
        static_cast<uint32_t>(cores), aclStream, key_c.data_ptr(),
        query_c.data_ptr(), dst_c.data_ptr(), output.data_ptr(), numEdges,
        numHeads, feat, numNodes, cores);
    TORCH_CHECK(ret == 0, "e_dot_v: AscendC kernel launch failed, ret=", ret);

    // The launch is a raw stream-ordered `aclrtlaunch_*` call; draining the
    // stream only releases `output` (and guards the caller's subsequent host
    // reads).  It is not used to order any queued framework producer.
    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS,
                "e_dot_v: stream synchronize failed, ret=", syncRet);

    return output;
}

// Test-only companion of `edge_softmax`: identical native preparation and the
// same device math, plus a write-back of the per-destination segment start the
// native softmax kernel actually read.  Returns
// `{out_sorted, seg_read, start, pos}` so the DGL parity/localization tests can
// compare the kernel's raw, pre-un-permute output, the segment table it used and
// the native permutation against the host and CPU references.  It is not part
// of the product model path.
std::vector<at::Tensor> edge_softmax_debug(const at::Tensor& logits,
                                           const at::Tensor& dst,
                                           int64_t num_nodes, int64_t stream) {
    TORCH_CHECK(logits.scalar_type() == at::kFloat,
                "edge_softmax_debug: only float32 is supported, got ", logits.scalar_type());
    TORCH_CHECK(logits.device().type() == c10::DeviceType::PrivateUse1,
                "edge_softmax_debug: logits must be an NPU tensor, got ", logits.device());
    TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
                "edge_softmax_debug: dst must be an NPU tensor, got ", dst.device());
    TORCH_CHECK(logits.dim() == 2, "edge_softmax_debug: logits must be 2-D, got ", logits.dim());
    TORCH_CHECK(logits.is_contiguous(), "edge_softmax_debug: logits must be contiguous");
    TORCH_CHECK(dst.scalar_type() == at::kLong,
                "edge_softmax_debug: dst must be int64, got ", dst.scalar_type());
    TORCH_CHECK(dst.is_contiguous(), "edge_softmax_debug: dst must be contiguous");
    TORCH_CHECK(num_nodes > 0, "edge_softmax_debug: num_nodes must be positive");

    at::Tensor logits_c = logits.contiguous();
    at::Tensor dst_c = dst.contiguous();
    const int64_t numEdges = logits_c.size(0);
    const int64_t numFeat = logits_c.size(1);
    TORCH_CHECK(dst_c.dim() == 1 && dst_c.numel() == numEdges,
                "edge_softmax_debug: dst must have exactly one entry per edge");
    TORCH_CHECK(numEdges > 0 && numFeat > 0,
                "edge_softmax_debug: requires a non-empty graph");

    const auto intOptions = logits_c.options().dtype(at::kInt);
    at::Tensor start = at::empty({num_nodes + 1}, intOptions);
    at::Tensor pos = at::empty({numEdges}, intOptions);
    at::Tensor logits_sorted = at::empty_like(logits_c);
    at::Tensor out_sorted = at::empty_like(logits_c);
    at::Tensor seg_read = at::empty({num_nodes}, intOptions);

    aclrtStream aclStream = reinterpret_cast<aclrtStream>(static_cast<intptr_t>(stream));

    uint32_t ret = aclrtlaunch_edge_softmax_prepare(
        /*blockDim=*/1, aclStream, dst_c.data_ptr(), logits_c.data_ptr(),
        start.data_ptr(), pos.data_ptr(), logits_sorted.data_ptr(),
        numEdges, numFeat, num_nodes);
    TORCH_CHECK(ret == 0,
                "edge_softmax_debug: edge_softmax_prepare launch failed, ret=", ret);

    const int64_t probeCores = used_core_count(logits_c, num_nodes);
    ret = aclrtlaunch_edge_softmax_probe(
        static_cast<uint32_t>(probeCores), aclStream, logits_sorted.data_ptr(),
        start.data_ptr(), out_sorted.data_ptr(), seg_read.data_ptr(), numEdges,
        numFeat, num_nodes, probeCores);
    TORCH_CHECK(ret == 0,
                "edge_softmax_debug: AscendC kernel launch failed, ret=", ret);

    const aclError syncRet = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(syncRet == ACL_SUCCESS,
                "edge_softmax_debug: stream synchronize failed, ret=", syncRet);

    return {out_sorted, seg_read, start, pos};
}

}  // namespace ascend_kernel

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("graph_probe(Tensor x, int stream) -> Tensor");
    m.def("edge_softmax(Tensor logits, Tensor dst, int num_nodes, int stream) -> Tensor");
    m.def("edge_softmax_debug(Tensor logits, Tensor dst, int num_nodes, int stream) -> Tensor[]");
    m.def("copy_e_sum(Tensor msg, Tensor dst, int num_nodes, int stream) -> Tensor");
    m.def("copy_e_mean(Tensor msg, Tensor dst, int num_nodes, int stream) -> Tensor");
    m.def("avg_pool(Tensor msg, Tensor dst, int num_nodes, int stream) -> Tensor");
    m.def("e_dot_v(Tensor key, Tensor query, Tensor dst, int feat, int stream) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("graph_probe", TORCH_FN(ascend_kernel::graph_probe));
    m.impl("edge_softmax", TORCH_FN(ascend_kernel::edge_softmax));
    m.impl("edge_softmax_debug", TORCH_FN(ascend_kernel::edge_softmax_debug));
    m.impl("copy_e_sum", TORCH_FN(ascend_kernel::copy_e_sum));
    m.impl("copy_e_mean", TORCH_FN(ascend_kernel::copy_e_mean));
    m.impl("avg_pool", TORCH_FN(ascend_kernel::avg_pool));
    m.impl("e_dot_v", TORCH_FN(ascend_kernel::edge_dot_v));
}

// CPython requires a module export function for `import _graph_probe_binding`.
// The body may be empty: loading this shared object already runs the
// `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL` static initializers that
// register `npu::graph_probe` for dispatch key `PrivateUse1`.
PYBIND11_MODULE(_graph_probe_binding, m) {
    m.doc() = "Native AscendC graph operators for the RFantibody NPU migration; "
              "module init registers npu::graph_probe via TORCH_LIBRARY.";
}
