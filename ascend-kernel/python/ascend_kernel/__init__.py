"""Native AscendC graph-operator bridge for the RFantibody NPU migration.

``graph_probe`` proves the official CANN ``ascendc_library`` build -> register ->
launch route on ``npu:0``.  ``edge_softmax`` is the native general
per-destination DGL ``edge_softmax(..., norm_by='dst')`` operator used by the
SE3-Transformer attention path.  ``copy_e_sum`` / ``copy_e_mean`` / ``avg_pool``
are the native per-destination scatter reductions used by the SE3 convolution,
attention and average-pooling paths.  ``e_dot_v`` is the native per-edge
``(key * query[dst]).sum(-1, keepdim=True)`` dot product used by the SE3
attention path.
"""

from __future__ import annotations

# Register the PrivateUse1 (NPU) backend before the extension is loaded so the
# generated ``RegisterAscendBinary`` constructor can query the device.  The
# backend is required: a missing/broken torch_npu raises the real error here
# instead of being swallowed and failing later with a confusing message.
import torch_npu  # noqa: F401

# Importing the compiled binding runs the ``TORCH_LIBRARY`` static initializers
# that register ``npu::graph_probe`` for dispatch key ``PrivateUse1``.
from . import _graph_probe_binding as _binding  # noqa: E402,F401
import torch  # noqa: E402


def graph_probe(x: "torch.Tensor", stream: int | None = None) -> "torch.Tensor":
    """Elementwise ``x + 1`` executed by the native AscendC kernel on ``x``'s NPU.

    ``stream`` defaults to the current NPU stream so the launch is ordered with
    respect to the surrounding PyTorch work.
    """
    if stream is None:
        stream = int(torch.npu.current_stream().npu_stream)
    return torch.ops.npu.graph_probe(x, int(stream))


def edge_softmax(
    logits: "torch.Tensor",
    dst: "torch.Tensor",
    num_nodes: int,
    stream: int | None = None,
) -> "torch.Tensor":
    """DGL ``edge_softmax(..., norm_by='dst')`` on the native AscendC kernel.

    ``logits`` is ``[E, F]`` float32 on the NPU and ``dst`` is the ``[E]``
    destination index of each edge.  The per-destination maximum, exponentiation,
    denominator and division all run in the native kernel; only the adjacency
    sort/permutation happens outside the kernel.  ``stream`` defaults to the
    current NPU stream.
    """
    if stream is None:
        stream = int(torch.npu.current_stream().npu_stream)
    return torch.ops.npu.edge_softmax(
        logits.contiguous(), dst, int(num_nodes), int(stream)
    )


def _scatter_reduce(
    op_name: str,
    msg: "torch.Tensor",
    dst: "torch.Tensor",
    num_nodes: int,
    stream: int | None,
) -> "torch.Tensor":
    """Flatten trailing message dims, run the native reduction, restore shape.

    DGL's ``copy_e_*`` reductions preserve the trailing feature shape, so the
    message is viewed as ``(E, F)`` for the kernel and the result is restored to
    ``(num_nodes,) + msg.shape[1:]``.  ``dst`` is the ``[E]`` destination index.
    """
    if stream is None:
        stream = int(torch.npu.current_stream().npu_stream)
    msg_shape = tuple(msg.shape[1:])
    flat = msg.reshape(msg.shape[0], -1).contiguous()
    out = getattr(torch.ops.npu, op_name)(
        flat, dst.contiguous(), int(num_nodes), int(stream)
    )
    return out.reshape((int(num_nodes),) + msg_shape)


def copy_e_sum(
    msg: "torch.Tensor",
    dst: "torch.Tensor",
    num_nodes: int,
    stream: int | None = None,
) -> "torch.Tensor":
    """DGL ``copy_e_sum(g, msg)`` on the native AscendC reduction kernel."""
    return _scatter_reduce("copy_e_sum", msg, dst, num_nodes, stream)


def copy_e_mean(
    msg: "torch.Tensor",
    dst: "torch.Tensor",
    num_nodes: int,
    stream: int | None = None,
) -> "torch.Tensor":
    """DGL ``copy_e_mean(g, msg)`` on the native AscendC reduction kernel.

    Zero-degree destinations (no incoming edge) are exactly ``0``.
    """
    return _scatter_reduce("copy_e_mean", msg, dst, num_nodes, stream)


def avg_pool(
    msg: "torch.Tensor",
    dst: "torch.Tensor",
    num_nodes: int,
    stream: int | None = None,
) -> "torch.Tensor":
    """DGL ``AvgPooling`` (per-destination mean) on the native reduction kernel."""
    return _scatter_reduce("avg_pool", msg, dst, num_nodes, stream)


def e_dot_v(
    key: "torch.Tensor",
    query: "torch.Tensor",
    dst: "torch.Tensor",
    feat: int | None = None,
    stream: int | None = None,
) -> "torch.Tensor":
    """DGL ``e_dot_v(g, key, query)`` on the native per-edge AscendC kernel.

    ``key`` is ``(E, ...)`` and ``query`` is ``(N, ...)`` where the trailing dims
    are a head/feature shape whose last dim is reduced.  The leading trailing
    dims are flattened so the kernel sees ``[E, H*feat]`` / ``[N, H*feat]``, and
    the result is restored to DGL's output shape ``key.shape[:-1] + (1,)``.
    ``feat`` defaults to ``key.shape[-1]``; ``dst`` is the ``[E]`` destination
    index.  The gather, multiply and reduction all run on the NPU, with no Torch
    graph operator and no CPU/CUDA fallback.
    """
    if feat is None:
        feat = int(key.shape[-1])
    feat = int(feat)
    if feat < 1:
        raise ValueError(f"e_dot_v: feat must be >= 1, got {feat}")
    trailing = tuple(int(s) for s in key.shape[1:-1])
    num_edges = int(key.shape[0])
    num_heads = 1
    for size in trailing:
        num_heads *= size
    if stream is None:
        stream = int(torch.npu.current_stream().npu_stream)
    flat_key = key.reshape(num_edges, num_heads * feat).contiguous()
    flat_query = query.reshape(int(query.shape[0]), num_heads * feat).contiguous()
    out = torch.ops.npu.e_dot_v(
        flat_key, flat_query, dst.contiguous(), feat, int(stream)
    )
    return out.reshape((num_edges,) + trailing + (1,))


__all__ = [
    "graph_probe",
    "edge_softmax",
    "copy_e_sum",
    "copy_e_mean",
    "avg_pool",
    "e_dot_v",
]
