"""No-argument public-API case for the unmodified ``tools/npu_kernel_probe.py``.

The probe imports this module (which registers the native schemas through the
compiled binding), runs ``run()`` under the torch_npu profiler and requires a
hardware ``KERNEL_*`` trace event.  ``run`` therefore only touches the public
``rfantibody.ascend_ops`` API on ``npu:0``.
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401  (registers torch.npu before the extension loads)

from rfantibody import ascend_ops

# 2 edges / 3 nodes / 2 heads / reduced width 3 (non-32-byte-aligned rows).
_KEY = [
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[-1.0, 0.5, 2.0], [3.0, -2.0, 1.0]],
]
_QUERY = [
    [[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.5, -1.0, 2.0], [1.0, 2.0, 3.0]],
]
_DST = [0, 2]


def run() -> torch.Tensor:
    """Execute ``ascend_graph::e_dot_v`` on ``npu:0`` and return its NPU output."""
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    key = torch.tensor(_KEY, dtype=torch.float32, device=device)
    query = torch.tensor(_QUERY, dtype=torch.float32, device=device)
    dst = torch.tensor(_DST, dtype=torch.int64, device=device)
    return ascend_ops.e_dot_v(key, query, dst, len(_QUERY))


def run_builtin() -> torch.Tensor:
    """Positive instrumentation control: a *builtin* CANN op on ``npu:0``.

    Measured with the identical harness; records whether this vendor's chrome
    trace names builtin kernels or only task types.
    """
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    x = torch.arange(8, dtype=torch.float32, device=device)
    return torch.add(x, 1.0)


def run_edge_reductions():
    """Native edge-destination reduction bundle on ``npu:0``.

    Issues exactly four native AscendC launches through the public
    ``rfantibody.ascend_ops`` API: ``edge_topology`` (via
    :func:`ascend_ops.build_edge_topology`), ``copy_e_sum``, ``copy_e_mean`` and
    ``edge_topology`` again (via :func:`ascend_ops.in_degrees`).  The tiny valid
    topology has 5 edges over 4 destinations (node 1 has zero in-degree) and a
    12-float (48-byte, non-32-byte-aligned) feature width.
    """
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    feat = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [0.5, -1.0, 2.0, -2.5, 3.0, -3.5, 4.0, -4.5, 5.0, -5.5, 6.0, -6.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0, 0.0],
            [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    dst = torch.tensor([0, 2, 0, 3, 2], dtype=torch.int64, device=device)
    topology = ascend_ops.build_edge_topology(dst, 4)
    total = ascend_ops.copy_e_sum(feat, topology)
    mean = ascend_ops.copy_e_mean(feat, topology)
    degrees = ascend_ops.in_degrees(dst, 4)
    return total, mean, degrees


def run_edge_softmax() -> torch.Tensor:
    """Native destination edge softmax bundle on ``npu:0``.

    Issues exactly two native AscendC launches through the public
    ``rfantibody.ascend_ops`` API: ``edge_topology`` (via
    :func:`ascend_ops.build_edge_topology`) and ``edge_softmax`` (the new
    ``edge_softmax_dst`` kernel).  The tiny valid topology has 5 edges over 4
    destinations (node 1 has zero in-degree) and a 3-float (12-byte,
    non-32-byte-aligned) weight row.
    """
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    logits = torch.tensor(
        [
            [0.5, -1.0, 2.0],
            [1.0, 1.0, 1.0],
            [-2.0, 0.25, 0.0],
            [3.0, -0.5, 1.5],
            [0.0, 2.0, -1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    dst = torch.tensor([0, 2, 0, 3, 2], dtype=torch.int64, device=device)
    topology = ascend_ops.build_edge_topology(dst, 4)
    return ascend_ops.edge_softmax(logits, topology)


def run_pooling():
    """Native DGL global-pooling bundle on ``npu:0``.

    Issues exactly two native AscendC launches through the public
    ``rfantibody.ascend_ops`` API: ``avg_pooling`` (``dgl.nn.AvgPooling``) and
    ``max_pooling`` (``dgl.nn.MaxPooling``).  The tiny valid bundle is ``feat``
    ``[5, 3]`` (12-byte, non-32-byte-aligned rows) with ``seglen = [2, 0, 3]``,
    i.e. it includes a zero-length segment whose pooled row must be exactly
    zero for both variants.
    """
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    feat = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [-1.0, 0.5, 2.5],
            [2.0, -2.0, 0.0],
            [4.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    seglen = torch.tensor([2, 0, 3], dtype=torch.int64, device=device)
    mean = ascend_ops.avg_pooling(feat, seglen)
    maximum = ascend_ops.max_pooling(feat, seglen)
    return mean, maximum


def run_graph_build() -> torch.Tensor:
    """Native graph-construction bundle on ``npu:0``.

    Issues exactly one native AscendC launch through the public
    ``rfantibody.ascend_ops`` API: ``edge_topology``, executed eagerly by
    ``AscendGraph.__init__`` (the ``in_degrees()`` call reuses that table and
    launches nothing).  The graph has 5 edges over 4 destinations (node 1 has
    zero in-degree) and a 3-float (12-byte, non-32-byte-aligned) ``rel_pos`` edge
    feature, i.e. the same surface the SE3 model stores on the graph.
    """
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    src = torch.tensor([0, 1, 2, 0, 3], dtype=torch.int64, device=device)
    dst = torch.tensor([0, 2, 0, 3, 2], dtype=torch.int64, device=device)

    from rfantibody.ascend_ops.graph import AscendGraph

    graph = AscendGraph(src, dst, 4)
    graph.edata["rel_pos"] = torch.tensor(
        [
            [0.5, -1.0, 2.0],
            [1.5, 0.25, -0.75],
            [-2.0, 1.0, 0.0],
            [0.125, 0.5, -0.25],
            [3.0, -3.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return graph.in_degrees()


def run_graph_probe() -> torch.Tensor:
    """Positive instrumentation control: a *separate* native AscendC kernel.

    ``ascend_graph::graph_probe`` (``x + 1``) has its own kernel body
    (``csrc/ascend_graph/op_kernel/graph_probe.cpp``) and its own generated
    ``aclrtlaunch_graph_probe`` entry point, so a hardware task event for this
    case proves that the measurement harness observes native AscendC launches in
    this environment, without using the subject operator (``e_dot_v``) as its
    own control.  Numerical correctness of this kernel is asserted separately by
    ``test_native_route_control``; this case only has to issue one launch.
    """
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    x = torch.arange(8, dtype=torch.float32, device=device)
    return ascend_ops.graph_probe(x)


def run_cpu_only() -> torch.Tensor:
    """Negative instrumentation control: a window with no device work."""
    x = torch.arange(8, dtype=torch.float32)
    return x * 2.0


def run_rf2_topk_graph():
    """Production RF2 graph construction on ``npu:0`` (public API only).

    Builds the graph through ``rfantibody.rf2.network.util_module.make_topk_graph``
    with NPU tensors, i.e. the same builder the RF2 product path uses
    (``rf2/network/Track_module.py``, which passes the C-alpha trace
    ``xyz[:, :, 1, :]``, i.e. ``[B, L, 3]``).  Issues exactly one native AscendC
    launch (``edge_topology``, eagerly inside ``AscendGraph.__init__``) and stores
    the ``rel_pos`` edge feature the SE3 model consumes.
    """
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    from rfantibody.rf2.network.util_module import make_topk_graph

    batch, length = 1, 48
    positions = torch.arange(length, dtype=torch.float32, device=device)
    xyz = torch.stack(
        [
            9.0 * torch.sin(0.31 * positions) + 0.05 * positions,
            9.0 * torch.cos(0.27 * positions) - 0.03 * positions,
            2.0 * torch.sin(0.13 * positions + 0.7),
        ],
        dim=-1,
    ).reshape(batch, length, 3)
    pair = torch.zeros(batch, length, length, 8, dtype=torch.float32, device=device)
    residue_index = torch.arange(length, dtype=torch.int64, device=device).reshape(batch, length)
    graph, edge_features = make_topk_graph(xyz, pair, residue_index)
    return graph.edata["rel_pos"].to(torch.float32), edge_features.to(torch.float32)
