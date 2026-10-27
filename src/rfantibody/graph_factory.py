"""Graph construction for the product path and for the reference path.

``make_graph`` is the single call site used by the RFdiffusion (and later RF2)
graph builders.  It keeps one spelling for both graph kinds:

* on an Ascend NPU device it returns the native
  :class:`rfantibody.ascend_ops.graph.AscendGraph`, whose construction executes
  the native ``edge_topology`` AscendC kernel and whose operators are the native
  AscendC kernels;
* on any other device it returns a real ``dgl`` graph, so the documented CUDA
  installation path and the CPU reference/tests keep their original behaviour.

``torch_npu`` and the compiled binding are imported only on the NPU branch (they
are imported by ``rfantibody.ascend_ops``), so a CPU/CUDA install never needs the
NPU vendor stack.
"""

from __future__ import annotations

import torch


def make_graph(
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    device,
):
    """Build the graph object for ``device`` from ``[E]`` ``src``/``dst`` ids.

    ``num_nodes`` is the total destination/node count (``B * L`` for the
    batched RFdiffusion/RF2 graphs) and must include every id in ``src``/``dst``.
    """
    device = torch.device(device)
    if device.type == "npu":
        from rfantibody.ascend_ops.graph import AscendGraph

        return AscendGraph(src.to(device), dst.to(device), int(num_nodes))

    import dgl

    return dgl.graph((src, dst), num_nodes=int(num_nodes)).to(device)


__all__ = ["make_graph"]
