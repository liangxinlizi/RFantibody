"""Native ``AscendGraph``: the DGL-shaped graph object of the NPU product path.

DGL 1.1.3 is a CPU/CUDA library and has no ``PrivateUse1`` backend, so an NPU
graph cannot be built or dispatched through ``dgl.graph(...)`` and
``dgl.ops.*``.  :class:`AscendGraph` therefore provides exactly the surface the
SE3-Transformer model uses -- ``edges()``, ``num_nodes()``, ``in_degrees()``,
``edata[...]``, ``batch_num_nodes()`` and ``to(device)`` -- and serves every
graph operator through the accepted public adapters of
:mod:`rfantibody.ascend_ops`, i.e. through the compiled native AscendC kernels.

Kernel route per call (exactly one native launch, no Torch/NumPy/DGL composition
and no CPU arithmetic):

============================  =================================================
``AscendGraph(...)``           ``edge_topology`` (eager, at construction)
``in_degrees()``               the ``degrees`` output of that same launch
``e_dot_v(key, query)``        ``edge_dot_v``
``edge_softmax(logits)``       ``edge_softmax_dst``
``copy_e_sum(feat)``           ``copy_e_sum``
``copy_e_mean(feat)``          ``copy_e_mean``
``avg_pooling(feat)``          ``avg_pooling``
``max_pooling(feat)``          ``max_pooling``
============================  =================================================

The graph methods mirror ``dgl.ops`` semantics exactly: ``copy_e_*`` reduce over
the destination index of each edge (the ``dst`` array of the graph, in the
original edge order), a destination with in-degree zero yields zeros, and
``edge_softmax`` normalizes per destination (``norm_by='dst'``) with per-column
max/exp/sum/divide in the kernel.

``is_ascend_native_graph`` is the duck-typed marker
``se3_transformer.model.graph_backend`` dispatches on, so the model code keeps a
single spelling for both graph kinds.
"""

from __future__ import annotations

import torch

#: Attribute name that selects the native dispatch branch of ``graph_backend``.
NATIVE_GRAPH_MARKER = "is_ascend_native_graph"


def _ops():
    """The accepted public adapters, imported lazily (avoids a circular import).

    ``rfantibody.ascend_ops`` imports ``torch_npu`` and the compiled binding, so
    it must only be touched on the NPU path.
    """
    from rfantibody import ascend_ops

    return ascend_ops


class _AscendEdgeData:
    """Minimal DGL-shaped ``graph.edata`` mapping over NPU tensors.

    Supports the spellings the model uses (``graph.edata['rel_pos'] = ...`` and
    ``graph.edata['rel_pos']``) plus membership/iteration, and validates the DGL
    invariant that an edge feature's first dimension is the edge count.
    """

    def __init__(self, owner: "AscendGraph"):
        self._owner = owner

    def __getitem__(self, key):
        try:
            return self._owner._edata[key]
        except KeyError:
            raise KeyError(
                f"edge feature {key!r} is not set on this AscendGraph"
            ) from None

    def __setitem__(self, key, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"edge feature {key!r} must be a torch.Tensor, got {type(value)!r}"
            )
        if value.dim() < 1:
            raise ValueError(
                f"edge feature {key!r} must have at least 1 dim, got "
                f"{tuple(value.shape)}"
            )
        if int(value.shape[0]) != self._owner.num_edges():
            raise ValueError(
                f"edge feature {key!r} has {int(value.shape[0])} rows but the graph "
                f"has {self._owner.num_edges()} edges"
            )
        self._owner._edata[key] = value

    def __delitem__(self, key):
        del self._owner._edata[key]

    def __contains__(self, key) -> bool:
        return key in self._owner._edata

    def __iter__(self):
        return iter(self._owner._edata)

    def __len__(self) -> int:
        return len(self._owner._edata)

    def keys(self):
        return self._owner._edata.keys()

    def values(self):
        return self._owner._edata.values()

    def items(self):
        return self._owner._edata.items()

    def __repr__(self) -> str:
        return f"_AscendEdgeData({sorted(self._owner._edata)!r})"


class AscendGraph:
    """A single Ascend NPU graph served by the native AscendC graph kernels."""

    #: Duck-typed marker consumed by ``se3_transformer.model.graph_backend``.
    is_ascend_native_graph = True

    def __init__(self, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
            raise TypeError("AscendGraph: src and dst must be torch.Tensor instances")
        if src.dim() != 1 or dst.dim() != 1:
            raise ValueError(
                "AscendGraph: src and dst must be 1-D edge arrays, got "
                f"{tuple(src.shape)} and {tuple(dst.shape)}"
            )
        if src.shape != dst.shape:
            raise ValueError(
                f"AscendGraph: src has {int(src.numel())} entries but dst has "
                f"{int(dst.numel())}"
            )
        if src.dtype != torch.int64 or dst.dtype != torch.int64:
            raise TypeError(
                "AscendGraph: src and dst must be int64 node ids, got "
                f"{src.dtype} and {dst.dtype}"
            )
        if src.device.type != "npu" or dst.device.type != "npu":
            raise ValueError(
                "AscendGraph is the NPU graph object; it requires NPU src/dst "
                f"tensors (got {src.device} and {dst.device}). Use "
                "rfantibody.graph_factory.make_graph for device dispatch."
            )
        if src.device != dst.device:
            raise ValueError(
                f"AscendGraph: src and dst are on different devices ({src.device} "
                f"and {dst.device})"
            )
        num_nodes = int(num_nodes)
        if num_nodes < 1:
            raise ValueError(
                f"AscendGraph: num_nodes must be at least 1, got {num_nodes}"
            )

        self._src = src.contiguous()
        self._dst = dst.contiguous()
        self._num_nodes = num_nodes
        self._edata: dict = {}
        # Eager native topology: graph construction itself is an NPU operation.
        self._topology = _ops().build_edge_topology(self._dst, self._num_nodes)

    # -- metadata ---------------------------------------------------------- #
    @property
    def device(self) -> torch.device:
        return self._src.device

    def num_nodes(self) -> int:
        """Number of destination nodes (``DGLGraph.num_nodes()``)."""
        return self._num_nodes

    def num_edges(self) -> int:
        """Number of edges (``DGLGraph.num_edges()``)."""
        return int(self._dst.numel())

    def edges(self):
        """``(src, dst)`` NPU int64 edge arrays in the original edge order."""
        return self._src, self._dst

    def in_degrees(self) -> torch.Tensor:
        """``DGLGraph.in_degrees()`` as an int64 ``[N]`` NPU tensor."""
        return self._topology.degrees

    def topology(self):
        """The native :class:`rfantibody.ascend_ops.EdgeTopology` of this graph."""
        return self._topology

    def batch_num_nodes(self) -> torch.Tensor:
        """int64 ``[1]`` NPU tensor holding this single graph's node count.

        Matches DGL's shape for an unbatched graph, so a ``GPooling`` call site
        works identically on both graph kinds.
        """
        return torch.tensor([self._num_nodes], dtype=torch.int64, device=self.device)

    @property
    def edata(self) -> _AscendEdgeData:
        """DGL-shaped edge-feature mapping (for example ``edata['rel_pos']``)."""
        return _AscendEdgeData(self)

    def to(self, device) -> "AscendGraph":
        """Move the graph (edge arrays and edge features) to an NPU device."""
        device = torch.device(device)
        if device.type != "npu":
            raise ValueError(
                "AscendGraph can only be moved to an NPU device; asked for "
                f"{device}. Use rfantibody.graph_factory.make_graph for device "
                "dispatch."
            )
        self._src = self._src.to(device)
        self._dst = self._dst.to(device)
        self._edata = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in self._edata.items()
        }
        self._topology = _ops().build_edge_topology(self._dst, self._num_nodes)
        return self

    # -- native graph operators -------------------------------------------- #
    def e_dot_v(self, key: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Native ``dgl.ops.e_dot_v(graph, key, query)``."""
        return _ops().e_dot_v(key, query, self._dst, self._num_nodes)

    def edge_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Native ``dgl.ops.edge_softmax(graph, logits, norm_by='dst')``."""
        return _ops().edge_softmax(logits, self._topology)

    def copy_e_sum(self, feat: torch.Tensor) -> torch.Tensor:
        """Native ``dgl.ops.copy_e_sum(graph, feat)``."""
        return _ops().copy_e_sum(feat, self._topology)

    def copy_e_mean(self, feat: torch.Tensor) -> torch.Tensor:
        """Native ``dgl.ops.copy_e_mean(graph, feat)``."""
        return _ops().copy_e_mean(feat, self._topology)

    def avg_pooling(self, feat: torch.Tensor) -> torch.Tensor:
        """Native ``dgl.nn.AvgPooling()(graph, feat)``."""
        return _ops().avg_pooling(feat, self.batch_num_nodes())

    def max_pooling(self, feat: torch.Tensor) -> torch.Tensor:
        """Native ``dgl.nn.MaxPooling()(graph, feat)``."""
        return _ops().max_pooling(feat, self.batch_num_nodes())

    def __repr__(self) -> str:
        return (
            f"AscendGraph(num_nodes={self._num_nodes}, "
            f"num_edges={self.num_edges()}, device={self.device})"
        )


__all__ = ["NATIVE_GRAPH_MARKER", "AscendGraph"]
