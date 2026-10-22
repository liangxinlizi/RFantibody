# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Single dispatch point for the graph operators used by the SE3-Transformer.

Every graph operator the model calls has exactly one implementation per graph
kind:

* a graph object that reports ``is_ascend_native_graph`` (an
  :class:`rfantibody.ascend_ops.graph.AscendGraph`) is served by the compiled
  native AscendC kernels through :mod:`rfantibody.ascend_ops`;
* any other graph object (a real ``dgl`` graph on CPU or CUDA) is served by the
  original ``dgl.ops`` / ``dgl.nn`` implementation.

The dispatch is a duck-typed attribute check, so this module never imports
``torch_npu``, the native extension or any project package: a CUDA/CPU install
and the CPU-DGL reference tests keep their original code path, and DGL is
imported lazily only on that path.

``graph.edges()``, ``graph.edata[...]``, ``graph.num_nodes()`` and
``graph.batch_num_nodes()`` keep their DGL spelling in the model; the native
graph object serves them.
"""

from __future__ import annotations


def _is_native(graph) -> bool:
    """Whether ``graph`` is served by the native AscendC kernels."""
    return bool(getattr(graph, "is_ascend_native_graph", False))


def e_dot_v(graph, key, query):
    """``dgl.ops.e_dot_v(graph, key, query)`` on the graph's own backend."""
    if _is_native(graph):
        return graph.e_dot_v(key, query)
    import dgl

    return dgl.ops.e_dot_v(graph, key, query)


def edge_softmax(graph, logits, norm_by: str = "dst"):
    """``dgl.ops.edge_softmax(graph, logits, norm_by=...)``.

    The model only uses the DGL default ``norm_by='dst'``; the native kernel
    implements exactly that normalization (per destination and per trailing
    column).  A different ``norm_by`` is rejected instead of being silently
    rounded to the destination semantics.
    """
    if norm_by != "dst":
        raise ValueError(
            "graph_backend.edge_softmax only implements DGL's norm_by='dst' "
            f"(the only mode the model uses), got norm_by={norm_by!r}"
        )
    if _is_native(graph):
        return graph.edge_softmax(logits)
    import dgl

    return dgl.ops.edge_softmax(graph, logits, norm_by=norm_by)


def copy_e_sum(graph, feat):
    """``dgl.ops.copy_e_sum(graph, feat)`` on the graph's own backend."""
    if _is_native(graph):
        return graph.copy_e_sum(feat)
    import dgl

    return dgl.ops.copy_e_sum(graph, feat)


def copy_e_mean(graph, feat):
    """``dgl.ops.copy_e_mean(graph, feat)`` on the graph's own backend."""
    if _is_native(graph):
        return graph.copy_e_mean(feat)
    import dgl

    return dgl.ops.copy_e_mean(graph, feat)


def pool(graph, feat, mode: str):
    """``dgl.nn.{AvgPooling,MaxPooling}()(graph, feat)`` on the graph's backend."""
    if mode not in ("avg", "max"):
        raise ValueError(f"graph_backend.pool got an unknown mode {mode!r}")
    if _is_native(graph):
        return graph.avg_pooling(feat) if mode == "avg" else graph.max_pooling(feat)
    from dgl.nn.pytorch import AvgPooling, MaxPooling

    return (AvgPooling() if mode == "avg" else MaxPooling())(graph, feat)


__all__ = [
    "copy_e_mean",
    "copy_e_sum",
    "e_dot_v",
    "edge_softmax",
    "pool",
]
