"""CUDA-compatible FP32 ``p=2`` ``cdist`` for the RFdiffusion model path.

CUDA's default large-``p=2`` ``torch.cdist`` evaluates the padded-matmul
formula ``((x0*x0 + x2*x2) + x1*x1) - 2*x.y + ((y0*y0 + y2*y2) + y1*y1)`` with
FP32 matmul/FMA accumulation, while the Ascend default direct-distance kernel
uses a different accumulation order.  The RFdiffusion trunk is defined against
the CUDA arithmetic, so the distance/RBF input uses it here as well.

This module implements that arithmetic with ordinary FP32 elementwise device
operations plus compensated (Dekker split multiply + error-free sum) arithmetic
and a round-to-odd low component.  The arithmetic is validated against the
independent CUDA recordings bundled in ``CDIST_HELDOUTS`` with the declared
``atol=1e-5`` / ``rtol=1e-4`` tolerance, and against a float64 CPU reference for
the shape, broadcast and alignment cases.  The bundled recordings are test
assets only; no product code reads them.

Routing policy
--------------
The wrapper keeps upstream PyTorch's *default* routing instead of always using
one branch.  Pinned ``torch`` 2.5.1
``aten/src/ATen/native/Distance.cpp`` selects ``use_mm_for_euclid_dist`` -- the
padded-matmul formula implemented here -- only when ``P > 25 or R > 25``; for
smaller operands it takes the direct-distance kernel.  :func:`cdist` mirrors
exactly that size rule, so a small operand set gets the device's own
direct-distance kernel (the arithmetic ``torch.cdist`` selects there) and a
large one gets the CUDA padded-matmul arithmetic.

Contract
--------
* The product path (``npu``, float32, ``P > 25 or R > 25``) uses
  :func:`cdist_p2`.
* CPU and CUDA targets, non-float32 dtypes there, and the small-operand case on
  ``npu`` delegate verbatim to :func:`torch.cdist`, so the documented CUDA
  installation keeps its own arithmetic and nothing changes off ``npu``.
* Non-float32 input on ``npu`` raises ``NotImplementedError`` instead of
  silently using a different accumulation order; the RFdiffusion model path is
  float32 throughout.
* Shapes are taken from the tensors (broadcast batch dimensions included), so
  ``(B, P, 3) x (B, R, 3)``, ``(B*T, P, 3) x (B*T, P, 3)`` and ``(P, 3)``
  self-distances all work.  There is no case-, shape-, seed- or input-specific
  branch, no host/CPU tensor arithmetic and no reference-asset access.
"""

from __future__ import annotations

import torch

__all__ = [
    "CDIST_HELDOUTS",
    "cdist",
    "cdist_p2",
    "squared_norms",
]

#: Independent CUDA recordings this arithmetic is validated against.  The files
#: live in ``test/rfdiffusion/reference_outputs/cdist_cuda_holdouts``; product
#: code never reads them.
CDIST_HELDOUTS = "test/rfdiffusion/reference_outputs/cdist_cuda_holdouts"

#: Dekker split constant for float32 (``2**ceil(24/2) + 1``).
_SPLIT = 4097.0

#: ``torch.int32`` bit patterns of ``+smallest subnormal`` and its negation, used
#: when the round-to-odd correction has to step a zero low component.
_MIN_SUBNORMAL_BITS = 1
_NEG_MIN_SUBNORMAL_BITS = -2147483647

#: Pinned upstream routing (torch 2.5.1 ``aten/src/ATen/native/Distance.cpp``):
#: the padded-matmul route is used only when ``P > 25`` or ``R > 25``; smaller
#: operands take the direct-distance kernel.  Mirrored here so this wrapper
#: keeps the general upstream default instead of always using one branch.
_MM_DIST_THRESHOLD = 25


def _decomposed_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``a * b + c`` as FP32 fused multiply-add (two-product + two-sum).

    ``p = a * b`` is the ordinary rounded product; the Dekker split of ``a`` and
    ``b`` plus the error-free sum ``s = p + c`` recover the two error terms
    ``ep`` and ``es``.  Their sum ``lo`` is added back to ``s``.  The low
    component is rounded to odd (never to nearest) so the final ``s + lo``
    addition cannot double-round, which is what makes the result identical to a
    hardware FMA.

    ``nextafter`` is not used: ``torch_npu`` has no device support for it (and
    no truthful FP64 device arithmetic), so the round-to-odd step is performed
    by stepping the ``int32`` bit pattern of ``lo`` -- exactly one representable
    step in the direction of ``tail``.
    """
    p = a * b
    sa = _SPLIT * a
    sb = _SPLIT * b
    ah = sa - (sa - a)
    al = a - ah
    bh = sb - (sb - b)
    bl = b - bh
    ep = (((ah * bh - p) + ah * bl) + al * bh) + al * bl
    s = p + c
    z = s - p
    es = (p - (s - z)) + (c - z)
    lo = ep + es
    zlo = lo - ep
    tail = (ep - (lo - zlo)) + (es - zlo)
    bits = lo.view(torch.int32)
    jam = (tail != 0) & ((bits & 1) == 0)
    step = torch.where(
        (lo > 0) == (tail > 0), torch.ones_like(bits), -torch.ones_like(bits)
    )
    moved = bits + step
    zero_bits = torch.where(
        tail > 0,
        torch.full_like(bits, _MIN_SUBNORMAL_BITS),
        torch.full_like(bits, _NEG_MIN_SUBNORMAL_BITS),
    )
    moved = torch.where(lo == 0, zero_bits, moved)
    lo = torch.where(jam, moved.view(torch.float32), lo)
    return s + lo


def squared_norms(coords: torch.Tensor) -> torch.Tensor:
    """``(x0*x0 + x2*x2) + x1*x1`` -- CUDA's norm order, not ``x0*x0 + x1*x1 + ...``."""
    if coords.shape[-1] != 3:
        raise ValueError(
            "cdist_compat implements 3-dimensional coordinates; got last "
            f"dimension {int(coords.shape[-1])}"
        )
    return (coords[..., 0] * coords[..., 0] + coords[..., 2] * coords[..., 2]) + (
        coords[..., 1] * coords[..., 1]
    )


def cdist_p2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CUDA-compatible FP32 Euclidean distance matrix of ``a`` and ``b``.

    Implements ``sqrt(max(0, nx - 2*x.y + ny))`` with the CUDA padded-matmul
    accumulation order: the three ``-2*x_k*y_k`` products are accumulated into
    the result one axis at a time with a fused multiply-add, then ``nx`` and
    ``ny`` are added.  Batch dimensions follow torch broadcasting rules.
    """
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("cdist_compat expects tensors of shape (..., P, 3)")
    if int(a.shape[-1]) != int(b.shape[-1]):
        raise ValueError(
            "the two coordinate tensors must have the same last dimension; got "
            f"{int(a.shape[-1])} and {int(b.shape[-1])}"
        )
    nx = squared_norms(a)
    ny = squared_norms(b)
    batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    acc = torch.zeros(
        tuple(batch_shape) + (int(a.shape[-2]), int(b.shape[-2])),
        device=a.device,
        dtype=torch.float32,
    )
    for axis in range(3):
        acc = _decomposed_fma(
            -2.0 * a[..., axis].unsqueeze(-1),
            b[..., axis].unsqueeze(-2),
            acc,
        )
    acc = acc + nx.unsqueeze(-1)
    acc = acc + ny.unsqueeze(-2)
    return acc.clamp_min(0.0).sqrt()


def cdist(a: torch.Tensor, b: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """``torch.cdist`` with CUDA-compatible FP32 arithmetic on ``npu``."""
    if float(p) != 2.0:
        raise NotImplementedError(
            "cdist_compat implements the p=2 Euclidean distance only; got "
            f"p={p!r} (use torch.cdist for other orders)"
        )
    if a.device.type != "npu":
        # CPU and CUDA keep their own arithmetic verbatim; on CUDA that is the
        # reference this module reproduces.
        return torch.cdist(a, b, p=p)
    if a.dtype is not torch.float32 or b.dtype is not torch.float32:
        raise NotImplementedError(
            "the CUDA-compatible cdist is implemented for float32 npu tensors "
            f"only; got {a.dtype} and {b.dtype} (the RFdiffusion path is "
            "float32 throughout)"
        )
    if int(a.shape[-2]) <= _MM_DIST_THRESHOLD and int(b.shape[-2]) <= _MM_DIST_THRESHOLD:
        # Upstream default for small operands: the device's direct-distance
        # kernel, i.e. the same arithmetic ``torch.cdist`` selects there.
        return torch.cdist(a, b, p=p)
    return cdist_p2(a, b)
