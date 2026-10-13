"""Device-agnostic drop-in for the small DGL surface used by SE3-Transformer.

Implements only what ``include/SE3Transformer`` and ``rfantibody`` consume:

* ``graph((src, dst), num_nodes)`` -> :class:`DGLGraph`
* ``graph.edges()`` / ``graph.edata``
* ``ops.copy_e_sum`` / ``ops.copy_e_mean`` / ``ops.e_dot_v``
* ``edge_softmax`` (``norm_by='dst'``)

Semantics mirror DGL 2.x (see ``ae-temp/latest_proposal.md`` documentation
research).  All reductions use only NPU-registered ops (``index_add``,
``scatter_add_``, ``amax``, ``exp``); no ``scatter_reduce``/``cummax``/``argmax``
are required, because this stack has no registered PrivateUse1 kernel for them.

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
        deg = torch.zeros(self.num_nodes, dtype=torch.long, device=self._dst.device)
        deg.scatter_add_(0, self._dst, torch.ones_like(self._dst))
        return deg


def graph(edges, num_nodes):
    """``dgl.graph((src, dst), num_nodes)`` equivalent."""
    src, dst = edges
    return DGLGraph(src, dst, num_nodes)


def copy_e_sum(g, msg):
    """Sum edge features into destination nodes (``gspmm(..., 'copy_rhs', 'sum')``)."""
    out = msg.new_zeros((g.num_nodes,) + tuple(msg.shape[1:]))
    return out.index_add(0, g._dst, msg)


def copy_e_mean(g, msg):
    """Mean edge features over destination nodes (zero-degree nodes -> 0)."""
    out = copy_e_sum(g, msg)
    deg = g.in_degrees().to(msg.dtype).clamp(min=1)
    return out / deg.view((-1,) + (1,) * (msg.dim() - 1))


def e_dot_v(g, key, query):
    """``dot(edge_key, query[destination])`` reduced over the last dimension."""
    q = query[g._dst]
    return (key * q).sum(dim=-1, keepdim=True)


def edge_softmax(g, logits, norm_by="dst"):
    """Softmax over incoming edges per destination node (DGL ``norm_by='dst'``).

    Stabilised with a per-feature-dimension global shift over all edges.  Softmax
    is invariant to a constant shift *within each destination group*, so a global
    shift is mathematically exact while avoiding the unsupported per-destination
    ``scatter_reduce``/``cummax`` reductions.
    """
    if norm_by != "dst":
        raise NotImplementedError(
            "graph_compat.edge_softmax supports norm_by='dst' only"
        )
    z = logits.amax(dim=0, keepdim=True)
    e = torch.exp(logits - z)
    denom = e.new_zeros((g.num_nodes,) + tuple(e.shape[1:])).index_add(0, g._dst, e)
    return e / denom[g._dst].clamp_min(torch.finfo(e.dtype).tiny)


class _Ops:
    pass


ops = _Ops()
ops.copy_e_sum = copy_e_sum
ops.copy_e_mean = copy_e_mean
ops.e_dot_v = e_dot_v


# Only used if GPooling is ever instantiated (not on the RFdiffusion/RF2 path).
class AvgPooling:
    def __call__(self, g, feats):
        return copy_e_mean(g, feats)


class MaxPooling:
    def __call__(self, g, feats):
        raise NotImplementedError(
            "MaxPooling is not used by the RFantibody SE3 path"
        )
