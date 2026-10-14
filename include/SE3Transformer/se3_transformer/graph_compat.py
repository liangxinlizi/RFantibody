"""Device-agnostic drop-in for the small DGL surface used by SE3-Transformer.

Implements only what ``include/SE3Transformer`` and ``rfantibody`` consume:

* ``graph((src, dst), num_nodes)`` -> :class:`DGLGraph`
* ``graph.edges()`` / ``graph.edata``
* ``ops.copy_e_sum`` / ``ops.copy_e_mean`` / ``ops.e_dot_v``
* ``edge_softmax`` (``norm_by='dst'``)

Semantics mirror DGL's documented per-destination behavior; this contract is
exercised by ``test/graph_parity/test_graph_parity.py``.  ``edge_softmax``, the
``copy_e_sum`` / ``copy_e_mean`` reductions, ``e_dot_v`` and ``in_degrees``
execute through native AscendC kernels in ``ascend-kernel``.  Non-NPU
tensors are rejected by the native operators instead of silently falling back to
a CPU computation.

This module intentionally does **not** replace the ``dgl`` package globally.
Only the SE(3) graph path is rewired to it.
"""

import torch


class DGLGraph:
    """Minimal graph container matching the DGL attributes used on this path."""

    def __init__(self, src, dst, num_nodes):
        self._src = src
        self._dst = dst
        self.num_nodes = int(num_nodes)
        self.edata = {}

    def to(self, device):
        self._src = self._src.to(device)
        self._dst = self._dst.to(device)
        self.edata = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in self.edata.items()
        }
        return self

    def edges(self):
        return self._src, self._dst

    def num_edges(self):
        return self._src.shape[0]

    def in_degrees(self):
        """DGL ``DGLGraph.in_degrees()``: incoming-edge count per node.

        Degrees are the native per-destination reduction of an all-ones edge
        message (``npu::copy_e_sum`` over the native adjacency preparation), so
        no Torch ``scatter_add_``/``index_add`` graph operator is used.  The
        result is ``int64`` of shape ``(num_nodes,)`` with an exact ``0`` for
        every zero-degree node.  Non-NPU tensors are rejected instead of
        silently falling back to a CPU computation.
        """
        if self._dst.device.type != "npu":
            raise RuntimeError(
                "in_degrees requires NPU tensors (no CPU fallback); got %s"
                % (self._dst.device,)
            )
        from se3_transformer import graph_native

        num_edges = int(self._dst.shape[0])
        ones = torch.ones(
            (num_edges, 1), dtype=torch.float32, device=self._dst.device
        )
        counts = graph_native.copy_e_sum(ones, self._dst, self.num_nodes)
        return counts.reshape(self.num_nodes).to(torch.int64)


def graph(edges, num_nodes):
    """``dgl.graph((src, dst), num_nodes)`` equivalent."""
    src, dst = edges
    return DGLGraph(src, dst, num_nodes)


def copy_e_sum(g, msg):
    """Sum edge features into destination nodes (``gspmm(..., 'copy_e', 'sum')``).

    Executes the native AscendC per-destination reduction on the NPU.  Non-NPU
    tensors are rejected instead of silently falling back to a CPU computation.
    """
    if msg.device.type != "npu":
        raise RuntimeError(
            "copy_e_sum requires NPU tensors (no CPU fallback); got %s"
            % (msg.device,)
        )
    from se3_transformer import graph_native

    return graph_native.copy_e_sum(msg, g._dst, g.num_nodes)


def copy_e_mean(g, msg):
    """Mean edge features over destination nodes (zero-degree nodes -> 0).

    Executes the native AscendC per-destination reduction on the NPU.  Non-NPU
    tensors are rejected instead of silently falling back to a CPU computation.
    """
    if msg.device.type != "npu":
        raise RuntimeError(
            "copy_e_mean requires NPU tensors (no CPU fallback); got %s"
            % (msg.device,)
        )
    from se3_transformer import graph_native

    return graph_native.copy_e_mean(msg, g._dst, g.num_nodes)


def e_dot_v(g, key, query):
    """``dot(edge_key, query[destination])`` reduced over the last dimension.

    Delegates to the native AscendC per-edge ``e_dot_v`` kernel, which gathers
    ``query[dst[e]]``, multiplies it with ``key[e]`` and reduces the last
    dimension entirely on the NPU.  The result has DGL's output shape
    ``key.shape[:-1] + (1,)``.  Non-NPU tensors are rejected instead of silently
    falling back to a CPU computation.
    """
    if key.device.type != "npu" or query.device.type != "npu":
        raise RuntimeError(
            "e_dot_v requires NPU tensors (no CPU fallback); got key=%s query=%s"
            % (key.device, query.device)
        )
    from se3_transformer import graph_native

    return graph_native.e_dot_v(key, query, g._dst)


def edge_softmax(g, logits, norm_by="dst"):
    """Softmax over incoming edges per destination node (DGL ``norm_by='dst'``).

    Delegates to the native AscendC ``edge_softmax`` kernel, which computes an
    independent, stable **per-destination** softmax (per trailing feature/head
    column) on the NPU.  There is no graph-wide shift and no denominator
    ``tiny``/``eps`` salting: a destination whose logits are thousands below
    another destination's still receives weights that sum to one, and
    destinations with in-degree 0 produce no edges.  Non-NPU tensors are
    rejected instead of silently falling back to a CPU computation.
    """
    if norm_by != "dst":
        raise NotImplementedError(
            "graph_compat.edge_softmax supports norm_by='dst' only"
        )
    if logits.device.type != "npu":
        raise RuntimeError(
            "edge_softmax requires NPU tensors (no CPU fallback); got %s"
            % (logits.device,)
        )
    from se3_transformer import graph_native

    return graph_native.edge_softmax(logits, g._dst, g.num_nodes)


class _Ops:
    pass


ops = _Ops()
ops.copy_e_sum = copy_e_sum
ops.copy_e_mean = copy_e_mean
ops.e_dot_v = e_dot_v


# Only used if GPooling is ever instantiated (not on the RFdiffusion/RF2 path).
class AvgPooling:
    def __call__(self, g, feats):
        # Native per-destination mean; zero-degree nodes are exactly 0.
        return copy_e_mean(g, feats)


class MaxPooling:
    def __call__(self, g, feats):
        raise NotImplementedError(
            "MaxPooling is not used by the RFantibody SE3 path"
        )
