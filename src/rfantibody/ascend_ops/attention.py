"""CUDA-order FP32 contraction for the tied attention of the RFdiffusion path.

The tied attention of ``BiasedAxialAttention`` is the contraction

.. code-block:: python

    attn = opt_einsum.contract("bnihk,bnjhk->bijh", query, key)

``query`` is ``[B, N, I, H, K]`` and ``key`` is ``[B, N, J, H, K]``; the result is
``[B, I, J, H]``.  On the recording CUDA device this contraction is evaluated as a
sequential, correctly rounded float32 multiply-add chain with ``k`` as the outer
and ``n`` as the inner loop variable, which is *not* the accumulation the Ascend
NPU produces for the same operands (the first reverse row of a measured
``BiasedAxialAttention`` forward deviates in its last bits).

:func:`tied_attention_logits` therefore dispatches to the native AscendC kernel
``ascend_graph::tied_attention_logits`` on an NPU, which reproduces that ordered
correctly rounded FMA chain bit-exactly.  Off the NPU the function returns the
upstream ``opt_einsum`` expression verbatim, so CPU and CUDA keep their current
(bit-identical) behaviour.

The two ``permute(...).contiguous()`` calls below are pure data movement that
puts the operands into the kernel's contiguous ``[B, N, H, K, I]`` /
``[B, N, H, K, J]`` layout; they perform no arithmetic, and the contraction
itself executes only inside the native kernel (no Torch ``einsum``/``bmm``/
``matmul``/``scatter``/``index_add`` composite on the NPU path).
"""

from __future__ import annotations

import torch

#: The upstream ``opt_einsum`` expression this helper implements.
EINSUM_EQUATION = "bnihk,bnjhk->bijh"


def _current_stream(stream: "int | None") -> int:
    if stream is not None:
        return int(stream)
    from rfantibody.ascend_ops import _current_stream as current_stream

    return current_stream(None)


def tied_attention_logits(
    query: torch.Tensor, key: torch.Tensor, stream: "int | None" = None
) -> torch.Tensor:
    """``opt_einsum.contract('bnihk,bnjhk->bijh', query, key)``.

    ``query`` is ``[B, N, I, H, K]``, ``key`` is ``[B, N, J, H, K]`` and the
    result is ``[B, I, J, H]``.  On an NPU the sum is executed by the native
    ``tied_attention_logits`` AscendC kernel with the CUDA reduction order
    (``k`` outer, ``n`` inner, one correctly rounded float32 FMA per step);
    elsewhere the upstream ``opt_einsum`` contraction is returned unchanged.
    """
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
        raise TypeError(
            "tied_attention_logits: query and key must be torch.Tensor "
            f"instances, got {type(query)!r} and {type(key)!r}"
        )
    if query.dim() != 5 or key.dim() != 5:
        raise ValueError(
            "tied_attention_logits: query and key must be 5-D "
            f"[B, N, I|J, H, K], got {tuple(query.shape)} and {tuple(key.shape)}"
        )
    if query.device != key.device:
        raise ValueError(
            "tied_attention_logits: query and key must live on the same device, "
            f"got {query.device} and {key.device}"
        )
    shared = (0, 1, 3, 4)  # B, N, H, K
    for axis in shared:
        if int(query.shape[axis]) != int(key.shape[axis]):
            raise ValueError(
                "tied_attention_logits: query and key must agree on B/N/H/K, "
                f"got {tuple(query.shape)} and {tuple(key.shape)}"
            )

    if query.device.type != "npu":
        from opt_einsum import contract

        return contract(EINSUM_EQUATION, query, key)

    if query.dtype != torch.float32 or key.dtype != torch.float32:
        raise ValueError(
            "tied_attention_logits: the native NPU route supports float32 only, "
            f"got {query.dtype} and {key.dtype}"
        )

    # [B, N, H, K, I] / [B, N, H, K, J]: pure data movement for the kernel layout.
    query_t = query.permute(0, 1, 3, 4, 2).contiguous()
    key_t = key.permute(0, 1, 3, 4, 2).contiguous()
    out = torch.ops.ascend_graph.tied_attention_logits(
        query_t, key_t, _current_stream(stream)
    )
    return out.permute(0, 2, 3, 1)
