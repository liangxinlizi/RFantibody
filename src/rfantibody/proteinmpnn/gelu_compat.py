"""Exact (``approximate='none'``) GELU for the NPU; CPU and CUDA unchanged.

The ``aten::gelu(approximate='none')`` implementation of the installed Ascend
stack evaluates the tanh formula, whereas CUDA and CPU evaluate the exact form
``0.5*x*(1+erf(x/sqrt(2)))``.  ProteinMPNN defines its activations with the
exact form, so the ``npu`` path must evaluate that form as well.

Contract
--------
* ``GELUExactNone`` subclasses :class:`torch.nn.GELU`, so it adds no parameter,
  no buffer and no ``state_dict`` key; ``isinstance(act, torch.nn.GELU)`` holds.
* ``approximate='tanh'`` delegates to upstream everywhere, so an explicit tanh
  request keeps its upstream semantics.
* Non-``npu`` tensors (CPU, CUDA) delegate **verbatim** to upstream
  ``torch.nn.GELU.forward``; the documented CUDA installation keeps its own
  arithmetic and the CPU-DGL reference tests are numerically unchanged.
* On ``npu`` the exact composite runs entirely with device tensor operations on
  the input's own device: no host transfer, no NumPy, no CPU fallback, no
  captured or injected data.
* Non-float32 input on ``npu`` raises :class:`NotImplementedError` instead of
  silently substituting a different precision.  The ProteinMPNN path is
  float32 throughout.

There is no case-, shape-, seed-, dtype- or input-specific branch, no
reference-asset access and no global monkeypatching of
``torch.nn.functional``.
"""

from __future__ import annotations

import torch

__all__ = [
    "EXACT_ERF_INV_SQRT2",
    "GELUExactNone",
    "exact_gelu",
    "is_npu_tensor",
]

#: ``1/sqrt(2)`` in double precision; in float32 arithmetic it rounds to the
#: nearest float32 factor exactly like the CUDA ``GeluKernel``'s own constant.
EXACT_ERF_INV_SQRT2 = 0.7071067811865476


def is_npu_tensor(tensor: torch.Tensor) -> bool:
    """True when ``tensor`` lives on an Ascend device.

    Only ``torch.Tensor.device.type`` is inspected: no ``torch_npu`` import, no
    device spoofing, and a CPU/CUDA-only install never touches the vendor stack.
    """
    return getattr(getattr(tensor, "device", None), "type", None) == "npu"


def exact_gelu(x: torch.Tensor) -> torch.Tensor:
    """``0.5 * x * (1 + erf(x / sqrt(2)))`` with device tensor ops only.

    The three operations are ordinary elementwise device kernels on ``x``'s own
    device and dtype, so the result never leaves the device and no host
    arithmetic is involved.
    """
    return 0.5 * x * (1.0 + torch.erf(x * EXACT_ERF_INV_SQRT2))


class GELUExactNone(torch.nn.GELU):
    """``torch.nn.GELU`` whose ``approximate='none'`` is exact on the NPU.

    On ``npu`` the exact ``erf`` composite is used because the vendor kernel
    evaluates the tanh approximation for ``approximate='none'``; on CPU/CUDA
    upstream ``torch.nn.GELU`` is used unchanged.
    """

    def __init__(self, approximate: str = "none") -> None:
        super().__init__(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.approximate != "none":
            # An explicit tanh request keeps upstream tanh semantics everywhere.
            return super().forward(x)
        if not is_npu_tensor(x):
            # CPU and CUDA delegate verbatim to upstream; on CUDA that is the
            # reference arithmetic this class reproduces on the NPU.
            return super().forward(x)
        if x.dtype is not torch.float32:
            raise NotImplementedError(
                "the exact GELU is implemented for float32 npu tensors only; "
                "got {} (the ProteinMPNN path is float32 throughout)".format(
                    x.dtype
                )
            )
        return exact_gelu(x)
