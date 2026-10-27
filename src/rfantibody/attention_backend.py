"""Device dispatch for contraction helpers shared by the attention modules.

The RFdiffusion tied attention needs a contraction whose float32 reduction order
matches the CUDA reference on every device.  This module is the single dispatch
point the model code calls:

* on an Ascend NPU the contraction runs in the native AscendC kernel
  ``ascend_graph::tied_attention_logits`` (CUDA-order, correctly rounded FP32 FMA
  chain, see :mod:`rfantibody.ascend_ops.attention`);
* on CPU/CUDA the upstream ``opt_einsum`` expression is executed verbatim, so the
  documented CUDA installation keeps its current bit-identical behaviour.

``torch_npu`` and the compiled binding are imported lazily inside the NPU branch,
so a CPU or CUDA installation never imports the vendor stack or the native
library.

Note that this is a *numeric backend* for an upstream ``opt_einsum`` contraction
and not a DGL-API operator: the contracted DGL operator surface
(``graph construction``/``edges``/``edata``/``in_degrees``/``copy_e_sum``/
``copy_e_mean``/``e_dot_v``/``edge_softmax``/pooling) is unaffected and
unchanged by it.
"""

from __future__ import annotations

import torch

#: The upstream ``opt_einsum`` expression implemented by this dispatch point.
EINSUM_EQUATION = "bnihk,bnjhk->bijh"


def tied_attention_logits(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """``opt_einsum.contract('bnihk,bnjhk->bijh', query, key)`` on any device.

    ``query`` is ``[B, N, I, H, K]``, ``key`` is ``[B, N, J, H, K]`` and the
    result is ``[B, I, J, H]``.  On ``npu`` the sum executes in the native
    AscendC ``tied_attention_logits`` kernel; on CPU/CUDA it stays the upstream
    ``opt_einsum`` contraction.
    """
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
        raise TypeError(
            "tied_attention_logits: query and key must be torch.Tensor "
            f"instances, got {type(query)!r} and {type(key)!r}"
        )
    if query.device.type == "npu":
        from rfantibody.ascend_ops.attention import (
            tied_attention_logits as native_tied_attention_logits,
        )

        return native_tied_attention_logits(query, key)

    from opt_einsum import contract

    return contract(EINSUM_EQUATION, query, key)
