"""Native AscendC graph-operator bridge for the RFantibody NPU migration.

The compiled extension ``_ascend_graph`` is produced by
``csrc/ascend_graph/build.sh`` through the installed official CANN route
(``ascendc_library(ascend_graph SHARED ...)``) and registered in the public
PyTorch dispatcher as namespace ``ascend_graph`` with ``PrivateUse1`` (NPU)
implementations.

Public surface of the native graph operators:

* :func:`graph_probe` -- route control ``y = x + 1`` executed by the native
  ``graph_probe`` AscendC kernel on the input's NPU.
* :func:`e_dot_v` -- native per-edge ``dgl.ops.e_dot_v`` used by the
  SE3-Transformer attention path.
* :class:`EdgeTopology`, :func:`build_edge_topology`, :func:`copy_e_sum`,
  :func:`copy_e_mean`, :func:`in_degrees` -- native edge-destination reduction
  capability: a deterministic CSR/edge-topology builder plus the two DGL
  destination-edge reductions used by the attention/convolution paths.  The CSR
  table, the in-degrees and the reductions all execute in native AscendC
  kernels.
* :func:`edge_softmax` -- native destination-normalized
  ``dgl.ops.edge_softmax(g, logits, norm_by='dst')`` over the same native CSR
  table: per-destination/per-column max, ``exp``, sum and division in the
  ``edge_softmax_dst`` AscendC kernel.
* :func:`avg_pooling`, :func:`max_pooling` -- native DGL global pooling
  (``dgl.nn.AvgPooling`` / ``dgl.nn.MaxPooling``, i.e.
  ``dgl.readout_nodes(op='mean'|'max')``) over an int64 segment table: the
  segment scan, the accumulation/selection and the DGL-exact mean denominator
  all run in the ``avg_pooling`` / ``max_pooling`` AscendC kernels.
* :func:`tied_attention_logits` -- native CUDA-order FP32 contraction for the
  tied attention of the RFdiffusion reverse path.  It implements the upstream
  ``opt_einsum.contract('bnihk,bnjhk->bijh')`` with ``k`` as the outer and ``n``
  as the inner reduction variable and every accumulation step a correctly
  rounded float32 FMA, executed by the ``tied_attention_logits`` AscendC kernel
  (compensated product, two TwoSums, vectorized round-to-odd).  It is a numeric
  backend for an upstream ``opt_einsum`` expression rather than a DGL-API
  operator, so the contracted DGL operator surface is unaffected by it.
* :class:`AscendGraph` -- the DGL-shaped graph object of the NPU product path.
  It stores the ``[E]`` source/destination id arrays on the NPU, builds the
  native edge topology eagerly at construction, and exposes ``edges()``,
  ``num_nodes()``, ``in_degrees()``, ``edata[...]``, ``batch_num_nodes()`` and
  ``to(device)`` plus the graph-operator methods above.  Its
  ``is_ascend_native_graph`` marker makes
  ``se3_transformer.model.graph_backend`` dispatch to the native adapters.
* :func:`is_available` -- genuine NPU availability check (``torch.npu``).

Every operator performs exactly one ``torch.ops.ascend_graph.*`` call (the
public wrappers below add only reshape/type validation).  There is no Torch /
NumPy / DGL composition of the operator math and no CPU/CUDA fallback: a
non-NPU tensor raises at the host bridge.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

# Register the PrivateUse1 (NPU) backend before the extension is loaded so its
# static initializers can bind the NPU implementations.  A missing/broken
# torch_npu must raise the real error here instead of failing later with a
# confusing dispatch message.
import torch_npu  # noqa: F401,E402

# Importing the compiled binding runs the TORCH_LIBRARY static initializers that
# register ``ascend_graph::graph_probe`` / ``ascend_graph::e_dot_v`` for dispatch
# key ``PrivateUse1``.
from . import _ascend_graph as _binding  # noqa: E402,F401


def is_available() -> bool:
    """Return whether a genuine Ascend NPU is usable through ``torch.npu``."""
    try:
        return bool(torch.npu.is_available())
    except Exception:  # pragma: no cover - defensive, torch_npu is imported above
        return False


def _current_stream(stream: "int | None") -> int:
    if stream is not None:
        return int(stream)
    return int(torch.npu.current_stream().npu_stream)


def graph_probe(x: "torch.Tensor", stream: "int | None" = None) -> "torch.Tensor":
    """Run the native ``graph_probe`` route control (``x + 1``) on ``x``'s NPU."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"graph_probe: expected a torch.Tensor, got {type(x)!r}")
    return torch.ops.ascend_graph.graph_probe(x, _current_stream(stream))


def e_dot_v(
    key: "torch.Tensor",
    query: "torch.Tensor",
    dst: "torch.Tensor",
    num_nodes: "int | None" = None,
    stream: "int | None" = None,
) -> "torch.Tensor":
    """Native ``dgl.ops.e_dot_v(g, key, query)`` on the NPU.

    ``key`` is ``(E, ..., F)`` and ``query`` is ``(N, ..., F)``; the trailing
    dimensions before the reduced last dimension are the head/feature shape.  The
    result has DGL's output shape ``key.shape[:-1] + (1,)``.

    ``num_nodes`` is the number of rows of ``query`` (defaults to
    ``query.shape[0]``) and ``dst`` is the ``[E]`` destination index of each
    edge.  The gather, multiply and reduction all execute inside the native
    ``edge_dot_v`` AscendC kernel.
    """
    if not isinstance(key, torch.Tensor) or not isinstance(query, torch.Tensor):
        raise TypeError("e_dot_v: key and query must be torch.Tensor instances")
    if not isinstance(dst, torch.Tensor):
        raise TypeError("e_dot_v: dst must be a torch.Tensor")

    key_shape = tuple(int(size) for size in key.shape)
    if key.dim() < 2:
        raise ValueError(f"e_dot_v: key must have at least 2 dims, got {key.shape}")
    num_edges = key_shape[0]
    num_feat = key_shape[-1]
    trailing = key_shape[1:-1]
    num_heads = 1
    for size in trailing:
        num_heads *= size

    if num_nodes is None:
        num_nodes = int(query.shape[0])
    num_nodes = int(num_nodes)

    flat_key = key.reshape(num_edges, num_heads * num_feat).contiguous()
    flat_query = query.reshape(int(query.shape[0]), num_heads * num_feat).contiguous()

    out = torch.ops.ascend_graph.e_dot_v(
        flat_key,
        flat_query,
        dst.contiguous(),
        num_heads,
        num_feat,
        num_nodes,
        _current_stream(stream),
    )
    return out.reshape((num_edges,) + trailing + (1,))


class EdgeTopology(NamedTuple):
    """Native CSR edge topology of one graph (produced on the NPU).

    ``offsets`` is ``int64 [N+1]`` (exclusive prefix sum of the in-degrees),
    ``indices`` is ``int32 [E]`` (original edge ids grouped by destination and
    ascending inside each destination) and ``degrees`` is ``int64 [N]`` (the
    exact in-degree of every destination, i.e. ``DGLGraph.in_degrees()``).
    """

    offsets: torch.Tensor
    indices: torch.Tensor
    degrees: torch.Tensor


def build_edge_topology(
    dst: "torch.Tensor | EdgeTopology", num_nodes: "int | None" = None
) -> EdgeTopology:
    """Build the native CSR edge topology for the ``[E]`` destination ids.

    Executes exactly one ``ascend_graph::edge_topology`` native AscendC kernel on
    the NPU.  ``num_nodes`` must be supplied (and must include every destination
    id); it cannot be inferred without a device-side reduction, which this route
    deliberately does not do.
    """
    if isinstance(dst, EdgeTopology):
        return dst
    if not isinstance(dst, torch.Tensor):
        raise TypeError(
            f"build_edge_topology: expected a torch.Tensor, got {type(dst)!r}"
        )
    if dst.dim() != 1:
        raise ValueError(
            f"build_edge_topology: dst must be 1-D, got {tuple(dst.shape)}"
        )
    if num_nodes is None:
        raise ValueError(
            "build_edge_topology: num_nodes is required and cannot be inferred "
            "from a device tensor"
        )
    num_nodes = int(num_nodes)
    if num_nodes < 0:
        raise ValueError(
            f"build_edge_topology: num_nodes must be non-negative, got {num_nodes}"
        )
    offsets, indices, degrees = torch.ops.ascend_graph.edge_topology(
        dst, num_nodes, _current_stream(None)
    )
    return EdgeTopology(offsets=offsets, indices=indices, degrees=degrees)


def _segment_reduce(
    op_name: str, feat: "torch.Tensor", topology: EdgeTopology
) -> "torch.Tensor":
    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"{op_name}: expected a torch.Tensor, got {type(feat)!r}")
    if not isinstance(topology, EdgeTopology):
        raise TypeError(
            f"{op_name}: expected an EdgeTopology, got {type(topology)!r}"
        )
    if feat.dim() < 1:
        raise ValueError(
            f"{op_name}: feat must have at least 1 dim, got {tuple(feat.shape)}"
        )

    num_edges = int(feat.shape[0])
    num_nodes = int(topology.offsets.numel()) - 1
    trailing = tuple(int(size) for size in feat.shape[1:])
    width = 1
    for size in trailing:
        width *= size
    if int(topology.indices.numel()) != num_edges:
        raise ValueError(
            f"{op_name}: topology has {int(topology.indices.numel())} edges but "
            f"feat has {num_edges} rows"
        )

    # Explicit (E, W) so the degenerate E == 0 case is reshapeable (a `-1`
    # dimension is ambiguous for zero-element tensors).
    flat = feat.reshape(num_edges, width).contiguous()
    out = getattr(torch.ops.ascend_graph, op_name)(
        flat, topology.indices, topology.offsets, num_nodes, _current_stream(None)
    )
    return out.reshape((num_nodes,) + trailing)


def copy_e_sum(feat: "torch.Tensor", topology: EdgeTopology) -> "torch.Tensor":
    """Native ``dgl.ops.copy_e_sum(g, feat)`` over the native edge topology.

    ``feat`` is ``[E, ...]`` edge features; the result is ``[N, ...]`` with one
    row per destination node (zero for a destination with no incoming edge).
    """
    return _segment_reduce("copy_e_sum", feat, topology)


def copy_e_mean(feat: "torch.Tensor", topology: EdgeTopology) -> "torch.Tensor":
    """Native ``dgl.ops.copy_e_mean(g, feat)`` over the native edge topology.

    Follows DGL 1.1.3 exactly: the per-destination sum divided by
    ``clamp(in_degree, 1, max(num_edges, 1))``, so a destination with no incoming
    edge yields zeros.
    """
    return _segment_reduce("copy_e_mean", feat, topology)


def edge_softmax(
    logits: "torch.Tensor", topology: EdgeTopology
) -> "torch.Tensor":
    """Native ``dgl.ops.edge_softmax(g, logits, norm_by='dst')`` on the NPU.

    ``logits`` is ``[E, ...]`` (DGL's ``(E, *, 1)`` edge weight shape, e.g.
    ``[E, num_heads]`` on the SE3-Transformer attention path) and the result has
    the same shape.  For every destination and every trailing column the native
    ``edge_softmax_dst`` kernel computes the per-destination maximum, the sum of
    ``exp(logit - max)`` over the destination's incoming edges and the
    normalized weight, exactly as DGL 1.1.3 defines ``norm_by='dst'``.  A
    destination with in-degree zero owns no edge rows; duplicate edges are
    counted separately.

    ``topology`` is the :class:`EdgeTopology` built by
    :func:`build_edge_topology` (``len(indices)`` must equal ``logits.shape[0]``)
    and must live on the same NPU as ``logits``.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            f"edge_softmax: expected a torch.Tensor, got {type(logits)!r}"
        )
    if not isinstance(topology, EdgeTopology):
        raise TypeError(
            f"edge_softmax: expected an EdgeTopology, got {type(topology)!r}"
        )
    if logits.dim() < 1:
        raise ValueError(
            f"edge_softmax: logits must have at least 1 dim, got {tuple(logits.shape)}"
        )

    num_edges = int(logits.shape[0])
    num_nodes = int(topology.offsets.numel()) - 1
    trailing = tuple(int(size) for size in logits.shape[1:])
    width = 1
    for size in trailing:
        width *= size
    if int(topology.indices.numel()) != num_edges:
        raise ValueError(
            f"edge_softmax: topology has {int(topology.indices.numel())} edges "
            f"but logits has {num_edges} rows"
        )

    # Explicit (E, W) so the degenerate E == 0 case is reshapeable (a `-1`
    # dimension is ambiguous for zero-element tensors).
    flat = logits.reshape(num_edges, width).contiguous()
    out = torch.ops.ascend_graph.edge_softmax_dst(
        flat, topology.indices, topology.offsets, num_nodes, _current_stream(None)
    )
    return out.reshape((num_edges,) + trailing)


def in_degrees(
    dst_or_topology: "torch.Tensor | EdgeTopology",
    num_nodes: "int | None" = None,
) -> "torch.Tensor":
    """Native in-degree vector (``DGLGraph.in_degrees()``) as ``int64 [N]``.

    The counts are produced by the native ``edge_topology`` AscendC kernel, not
    by a Torch reduction.  An existing :class:`EdgeTopology` is reused as-is.
    """
    if isinstance(dst_or_topology, EdgeTopology):
        return dst_or_topology.degrees
    return build_edge_topology(dst_or_topology, num_nodes).degrees


def _global_pool(
    op_name: str, feat: "torch.Tensor", seglen: "torch.Tensor"
) -> "torch.Tensor":
    """Shared adapter for :func:`avg_pooling` and :func:`max_pooling`.

    ``feat`` is ``[N, ...]``; ``seglen`` is the int64 ``[B]`` segment table of a
    single graph (``graph.batch_num_nodes()``) or of a ``dgl.batch`` of graphs,
    with ``sum(seglen) == N``.  Exactly one ``torch.ops.ascend_graph.<op_name>``
    call runs on the current NPU stream; the trailing feature dimensions are
    flattened before the launch and restored on the result, so ``GPooling``
    (``[N, F, 1]`` -> ``[B, F, 1]``) and plain ``[N, D]`` features both work.
    """
    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"{op_name}: expected a torch.Tensor for feat, got {type(feat)!r}")
    if not isinstance(seglen, torch.Tensor):
        raise TypeError(
            f"{op_name}: expected a torch.Tensor for seglen, got {type(seglen)!r}"
        )
    if feat.dim() < 1:
        raise ValueError(
            f"{op_name}: feat must have at least 1 dim, got {tuple(feat.shape)}"
        )
    if seglen.dim() != 1:
        raise ValueError(
            f"{op_name}: seglen must be 1-D, got {tuple(seglen.shape)}"
        )
    if seglen.dtype != torch.int64:
        raise ValueError(
            f"{op_name}: seglen must be int64, got {seglen.dtype}"
        )
    if feat.dtype != torch.float32:
        raise ValueError(
            f"{op_name}: feat must be float32, got {feat.dtype}"
        )

    num_rows = int(feat.shape[0])
    if num_rows == 0:
        # DGL's own segment_reduce divides by `clamp(seglen, 1, len(value))`,
        # which degenerates for a feature tensor with no rows; this case is
        # rejected here instead of silently producing a different result.
        raise ValueError(
            f"{op_name}: feat must have at least one row; a zero-row feature "
            "tensor has no DGL-defined pooling result"
        )
    num_segments = int(seglen.shape[0])
    if num_segments < 1:
        raise ValueError(
            f"{op_name}: seglen must have at least one segment, got {num_segments}"
        )

    trailing = tuple(int(size) for size in feat.shape[1:])
    width = 1
    for size in trailing:
        width *= size

    # Explicit (N, W) so a degenerate width is reshapeable (a `-1` dimension is
    # ambiguous for zero-element tensors).
    flat = feat.reshape(num_rows, width).contiguous()
    out = getattr(torch.ops.ascend_graph, op_name)(
        flat, seglen.contiguous(), num_segments, _current_stream(None)
    )
    return out.reshape((num_segments,) + trailing)


def avg_pooling(feat: "torch.Tensor", seglen: "torch.Tensor") -> "torch.Tensor":
    """Native ``dgl.nn.AvgPooling()(graph, feat)`` on the NPU.

    ``feat`` is ``[N, ...]`` node features and ``seglen`` the int64 ``[B]``
    segment table (``graph.batch_num_nodes()`` for one graph, or the batched
    node counts of ``dgl.batch``).  The result is ``[B, ...]``.

    DGL 1.1.3 semantics reproduced exactly: the per-segment sum is divided by
    ``clamp(seglen, 1, N)`` in float32, so a zero-length segment yields exactly
    ``0``.  The segment scan, the accumulation and the division all execute in
    the native ``avg_pooling`` AscendC kernel; there is no Torch/DGL/NumPy/CPU
    fallback and no ``scatter``/``index_add`` composition.
    """
    return _global_pool("avg_pooling", feat, seglen)


def max_pooling(feat: "torch.Tensor", seglen: "torch.Tensor") -> "torch.Tensor":
    """Native ``dgl.nn.MaxPooling()(graph, feat)`` on the NPU.

    ``feat`` is ``[N, ...]`` node features and ``seglen`` the int64 ``[B]``
    segment table.  The result is ``[B, ...]``.

    DGL 1.1.3 semantics reproduced exactly: a zero-length segment yields exactly
    ``0`` (DGL's ``replace_inf_with_zero``), never ``-inf``.  The elementwise
    selection executes in the native ``max_pooling`` AscendC kernel; there is no
    Torch/DGL/NumPy/CPU fallback.
    """
    return _global_pool("max_pooling", feat, seglen)


# Imported last: ``AscendGraph`` uses the adapters defined above, and keeping the
# import at the end avoids a circular import through this package.
from .attention import tied_attention_logits  # noqa: E402
from .graph import AscendGraph  # noqa: E402

__all__ = [
    "AscendGraph",
    "EdgeTopology",
    "avg_pooling",
    "build_edge_topology",
    "copy_e_mean",
    "copy_e_sum",
    "e_dot_v",
    "edge_softmax",
    "graph_probe",
    "in_degrees",
    "is_available",
    "max_pooling",
    "tied_attention_logits",
]
