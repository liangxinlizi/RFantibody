"""CUDA-compatible distribution random streams for the NPU product path.

RFantibody's stochastic inference decisions (ProteinMPNN decoding order and the
per-position residue sample, and later the RF2/RFdiffusion masks) were produced
against a CUDA device generator in the original pipeline.  On an Ascend NPU the
vendor generator is a different stream, so the *same* seed selects a different
residue / decoding order and the versioned CUDA references cannot be reproduced.

This module therefore implements the distribution algorithm of the pinned CUDA
contract (PyTorch ``v2.3.1`` ``aten/src/ATen/native/cuda/DistributionTemplates.h``
plus ``curand_philox4x32_x``) on the host: Philox4x32-10 counter-based
generation with the documented CUDA execution profile (grid/stride/unroll), the
float32 Box-Muller normal transform (``curand_normal4``), the float32 and
float64 uniform transforms and the ``Exponential(1)`` transform used by the
``torch.multinomial`` fast path.  Given the same seed, offset and shape it
produces the same variates as the CUDA device generator, and it advances the
Philox offset exactly like the CUDA kernel does.

Contract
--------
* ``CUDA_RNG_PROFILE`` is *source-stream metadata*: the CUDA device profile
  (multiprocessor count, threads per multiprocessor, block size) whose launch
  geometry defines the original stream.  It describes the stream that is being
  reproduced; it is not a claim about NPU hardware.  It can be overridden with
  ``RFANTIBODY_CUDA_RNG_PROFILE="<sm_count>,<max_threads_per_sm>,<block_size>"``.
* State is per process and per device type (``seed`` plus Philox ``offset``).
  ``manual_seed(seed)`` resets it; otherwise it initialises lazily from the
  vendor generator's ``initial_seed()`` so a preceding
  ``torch.npu.manual_seed_all(...)`` is honoured.
* Every helper delegates verbatim to the native torch generator when the target
  device is not ``npu``: CPU and CUDA behaviour (including the CUDA path's own
  RNG stream) is untouched, so the documented CUDA installation keeps working.
* On ``npu`` only float32 ``normal``, float32/float64 ``exponential`` and the
  float32/float64 dropout *mask* are contracted; any other dtype raises
  ``NotImplementedError`` instead of silently falling back to a different
  stream.  The device dropout *product* exists for float32 only (see
  ``DEVICE_DTYPE_CONTRACT``).
* Only the *random input* is generated on the host CPU (exactly like the
  original PCIe-side draws the model already receives).  Every neural
  computation and the sampling decision itself (``div``/``argmax`` for
  ``multinomial``) execute on the device tensor.
* ``RFANTIBODY_RNG_TRACE=<dir>`` appends one JSONL record per device draw
  (kind, shape, dtype, seed, offset before/after).  It is off by default and is
  used for evidence only; it never influences computation.

The CUDA execution profile reproduced here
------------------------------------------
``block = block_size`` (256);
``grid = min(ceil(numel / block), sm_count * (max_threads_per_sm // block))``;
``stride = block * grid``;
``unroll = 4`` for float32 and ``2`` for float64 draws;
the Philox counter is interpreted as ``(offset // 4, 0, thread_index, 0)`` with
key ``(seed_lo, seed_hi, 0, 0)``;
the offset advances by ``((numel - 1) // (stride * 4) + 1) * 4`` per call, which
is the CUDA ``calc_execution_policy`` rule (note: it always uses the 4-output
``curand4`` engine-call count, also for the 2-value float64 draws).

The CUDA-compatible dropout (``fused_dropout``) contract
--------------------------------------------------------
``torch.nn.functional.dropout`` on a CUDA tensor runs
``aten/src/ATen/native/cuda/Dropout.cu`` with the *keep* probability
``p1m = 1 - p``: the offset advances by the same 4-word engine-call rule, and
the mask words follow the two kernels of that file:

* ``VEC in (2, 4)``: only when the tensor is non-overlapping-and-dense, the
  data pointer is aligned to ``VEC * sizeof(scalar_t)`` and ``numel % VEC == 0``
  (``UNROLL == 4`` is the only width cap, so an aligned float64 buffer uses
  ``VEC == 4``).  One ``curand_uniform4`` per ``VEC`` elements per thread.
  ``VEC == 4`` uses ``(w0, w1, w2, w3)``; ``VEC == 2`` uses the fresh
  ``(w0, w1)`` of *every* engine call and discards ``(w2, w3)``, because the
  kernel's ``gridxvec_loop_state`` starts at 0 and is only XOR-ed inside the
  branch it guards, so the ``else`` (reuse ``w2, w3``) branch is dead;
* ``VEC == 1``: four words per engine call, element index
  ``it * stride * 4 + ii * stride + tid``;
* index order: the ``VEC > 1`` kernel writes ``data[linearIndex]`` in *memory*
  order, so the flat host mask is placed with the tensor's own strides; the
  ``VEC == 1`` kernel (also the kernel selected for every non-dense layout)
  addresses logical elements through ``IndexToOffset``, so the flat mask is
  reshaped.  Both agree on contiguous tensors;
* keep test ``word < cutoff`` for float32 (``accscalar_t == float``, so the
  comparison is a float32 comparison; ``_keep_cutoff`` derives the equivalent
  integer threshold from the monotone ``curand_uniform4`` mapping);
* float64 compares the float32 uniform value against the keep probability in
  double precision (``accscalar_t == double``), which is *not* the same test as
  the float32 one near the threshold;
* the kept mask value is ``accscalar(1) / accscalar(keep)`` and the output is
  ``src * mask * scale`` in ``accscalar`` precision.

The ``Exponential(1)`` transform carries the pinned near-one guard
(``TransformationHelper.h``): ``curand_uniform`` is in ``(0, 1]`` and the
largest mapped word rounds to exactly ``1.0``, so ``val >= 1 - eps / 2``
substitutes ``log = -eps / 2`` instead of returning ``-log(1.0) == 0``.

``p == 0``, ``p == 1``, ``train=False`` and empty tensors are the source
short-cuts and never consume the stream.  The masks are generated on the host
and only the elementwise multiply runs on the device, exactly like the host
side random input the model already receives; no neural computation leaves the
device.  ``DEVICE_DTYPE_CONTRACT`` records the one bounded limitation this
Ascend stack forces: the float64 mask bits and the Philox stream are exact
float64 CUDA values, but there is no FP64 elementwise arithmetic on the device
(``torch_npu`` maps a requested ``Double`` cast/elementwise target to ``Float``
for device kernels), so a float64 *device product* is **unsupported**: it raises
``NotImplementedError`` instead of returning a float32 product that pretends to
be float64.  Float32, the dtype every RF2/ProteinMPNN activation uses, is exact.
"""

from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "CUDA_RNG_PROFILE",
    "CudaCompatDropout",
    "CudaCompatGenerator",
    "DEVICE_DTYPE_CONTRACT",
    "DEVICE_PRODUCT_DTYPES",
    "bernoulli_mask",
    "device_dtype_contract",
    "dropout",
    "dropout_mask_like",
    "dropout_offset_advance",
    "dropout_vector_size",
    "exponential",
    "get_seed_offset",
    "manual_seed",
    "multinomial",
    "randn",
    "randn_like",
]

ShapeLike = Union[int, Sequence[int], torch.Size]

_SEED_MASK = (1 << 64) - 1
_PROFILE_ENV = "RFANTIBODY_CUDA_RNG_PROFILE"
_TRACE_ENV = "RFANTIBODY_RNG_TRACE"

_DEFAULT_PROFILE = {"sm_count": 132, "max_threads_per_sm": 2048, "block_size": 256}

_MASK32 = np.uint64(0xFFFFFFFF)
_PHILOX_M0 = np.uint64(0xD2511F53)
_PHILOX_M1 = np.uint64(0xCD9E8D57)
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85

_UNIFORM32_SCALE = np.float32(2.0**-32)
_UNIFORM32_HALF = np.float32(2.0**-33)
_NORMAL32_SCALE = _UNIFORM32_SCALE * np.float32(2.0 * math.pi)
_NORMAL32_HALF = _NORMAL32_SCALE / np.float32(2.0)
_TWO_POW_M53 = 2.0**-53
_TWO_POW_M54 = 2.0**-54

# ``const int UNROLL = 4;`` in ``Dropout.cu`` (both the VEC and the VEC==1
# kernels advance the Philox counter in groups of four engine outputs).
_DROPOUT_UNROLL = 4

_LOCK = threading.RLock()
_MANUAL_SEED: Optional[int] = None
_STATES: dict[str, "_StreamState"] = {}


def _load_profile() -> dict:
    """Return the CUDA stream profile, honouring the documented env override."""
    override = os.environ.get(_PROFILE_ENV)
    if not override:
        return dict(_DEFAULT_PROFILE)
    parts = [part.strip() for part in override.split(",")]
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError(
            f"{_PROFILE_ENV} must be '<sm_count>,<max_threads_per_sm>,<block_size>', "
            f"got {override!r}"
        )
    try:
        sm_count, max_threads_per_sm, block_size = (int(part) for part in parts)
    except ValueError as exc:  # non-integer component
        raise ValueError(
            f"{_PROFILE_ENV} components must be integers, got {override!r}"
        ) from exc
    if sm_count <= 0 or max_threads_per_sm <= 0 or block_size <= 0:
        raise ValueError(
            f"{_PROFILE_ENV} components must be positive, got {override!r}"
        )
    return {
        "sm_count": sm_count,
        "max_threads_per_sm": max_threads_per_sm,
        "block_size": block_size,
    }


CUDA_RNG_PROFILE = _load_profile()


class _StreamState:
    """Seed and Philox counter offset of one device-type stream."""

    __slots__ = ("seed", "offset")

    def __init__(self, seed: int, offset: int = 0) -> None:
        self.seed = int(seed) & _SEED_MASK
        self.offset = int(offset)


# --------------------------------------------------------------------------- #
# Shape / dtype helpers
# --------------------------------------------------------------------------- #
def _normalise_shape(shape: ShapeLike) -> Tuple[int, ...]:
    if isinstance(shape, torch.Size):
        dims = tuple(int(dim) for dim in shape)
    elif isinstance(shape, int):
        dims = (int(shape),)
    else:
        dims = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in dims):
        raise ValueError(f"negative dimension in shape {shape!r}")
    return dims


def _np_dtype(dtype: torch.dtype) -> np.dtype:
    if dtype is torch.float32:
        return np.dtype(np.float32)
    if dtype is torch.float64:
        return np.dtype(np.float64)
    raise NotImplementedError(
        "CUDA-compatible random streams are implemented for float32 and float64 "
        f"only; got dtype {dtype}"
    )


def _resolve_device(device) -> torch.device:
    if device is None:
        return torch.empty(0).device
    return torch.device(device)


def _initial_seed(device_type: str) -> int:
    """Vendor generator seed for ``device_type`` (honours manual_seed_all)."""
    try:
        if device_type == "npu":
            import torch_npu  # noqa: F401  (registers torch.npu)

            return int(torch.npu.initial_seed()) & _SEED_MASK
        if device_type == "cuda":
            return int(torch.cuda.initial_seed()) & _SEED_MASK
    except Exception:  # pragma: no cover - vendor stack absent / not initialised
        pass
    return int(torch.initial_seed()) & _SEED_MASK


def _state_for(device_type: str) -> _StreamState:
    with _LOCK:
        state = _STATES.get(device_type)
        if state is None:
            seed = _MANUAL_SEED if _MANUAL_SEED is not None else _initial_seed(device_type)
            state = _StreamState(seed)
            _STATES[device_type] = state
        return state


def manual_seed(seed: int) -> None:
    """Reset the compatibility stream for every device type to ``seed``."""
    global _MANUAL_SEED
    value = int(seed) & _SEED_MASK
    with _LOCK:
        _MANUAL_SEED = value
        _STATES.clear()


def get_seed_offset(device=None) -> Tuple[int, int]:
    """Return ``(seed, philox_offset)`` of the compatibility stream."""
    device_type = _resolve_device(device).type
    state = _state_for(device_type)
    with _LOCK:
        return state.seed, state.offset


def _trace(
    kind: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    seed: int,
    offset_before: int,
    offset_after: int,
    device_type: str,
    extra: Optional[dict] = None,
) -> None:
    target = os.environ.get(_TRACE_ENV)
    if not target:
        return
    record = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": kind,
        "shape": list(shape),
        "dtype": str(dtype),
        "device_type": device_type,
        "seed": int(seed),
        "offset_before": int(offset_before),
        "offset_after": int(offset_after),
    }
    if extra:
        record.update(extra)
    os.makedirs(target, exist_ok=True)
    with _LOCK:
        with open(
            os.path.join(target, "rng_trace.jsonl"), "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Philox4x32-10 and the CUDA distribution transforms
# --------------------------------------------------------------------------- #
def _philox4x32_10(seed: int, counter: int, count: int) -> np.ndarray:
    """``philox4x32_10`` for ``count`` sequences starting at ``counter``.

    Mirrors ``curand_init(seed, thread_index, 4 * counter)`` followed by one
    ``curand4`` call: counter word 0 is ``counter``, counter word 1 is its high
    half, sequence words are the thread index, key is ``(seed_lo, seed_hi)``.
    """
    sequences = np.arange(count, dtype=np.uint64)
    x = np.full(count, np.uint64(int(counter) & 0xFFFFFFFF), dtype=np.uint64)
    y = np.full(count, np.uint64((int(counter) >> 32) & 0xFFFFFFFF), dtype=np.uint64)
    z = sequences & _MASK32
    w = sequences >> np.uint64(32)
    # Philox round keys are kept as Python integers so their addition can wrap
    # explicitly on every NumPy version (scalar uint64 overflow semantics differ).
    k0 = int(seed) & 0xFFFFFFFF
    k1 = (int(seed) >> 32) & 0xFFFFFFFF
    for _ in range(10):
        prod_a = x * _PHILOX_M0
        prod_b = z * _PHILOX_M1
        x = ((prod_b >> np.uint64(32)) ^ y ^ np.uint64(k0)) & _MASK32
        y = prod_b & _MASK32
        z = ((prod_a >> np.uint64(32)) ^ w ^ np.uint64(k1)) & _MASK32
        w = prod_a & _MASK32
        k0 = (k0 + _PHILOX_W0) & 0xFFFFFFFF
        k1 = (k1 + _PHILOX_W1) & 0xFFFFFFFF
    return np.stack([x, y, z, w], axis=-1).astype(np.uint32)


def _uniform_float32(word: np.ndarray) -> np.ndarray:
    """``curand_uniform4`` per-word mapping, computed in float32."""
    return word.astype(np.float32) * _UNIFORM32_SCALE + _UNIFORM32_HALF


def _normal4(bits: np.ndarray) -> np.ndarray:
    """``curand_normal4`` Box-Muller transform (float32, four values)."""
    columns = []
    for word_index in (0, 2):
        uniform = _uniform_float32(bits[:, word_index])
        angle = bits[:, word_index + 1].astype(np.float32) * _NORMAL32_SCALE + _NORMAL32_HALF
        amplitude = np.sqrt(np.float32(-2.0) * np.log(uniform))
        columns.append((amplitude * np.sin(angle)).astype(np.float32))
        columns.append((amplitude * np.cos(angle)).astype(np.float32))
    return np.stack(columns, axis=-1)


def _uniform2_double(bits: np.ndarray) -> np.ndarray:
    """``curand_uniform2_double`` mapping (two float64 values, 53 random bits)."""
    columns = []
    for word_index in (0, 2):
        low = bits[:, word_index].astype(np.uint64)
        high = bits[:, word_index + 1].astype(np.uint64)
        combined = (low ^ (high << np.uint64(21))).astype(np.float64)
        columns.append(combined * _TWO_POW_M53 + _TWO_POW_M54)
    return np.stack(columns, axis=-1)


def _transform(kind: str, dtype: torch.dtype, bits: np.ndarray) -> np.ndarray:
    if kind == "normal":
        return _normal4(bits)
    if kind != "exponential":
        raise ValueError(f"unknown distribution kind {kind!r}")
    # ``transformation::exponential`` (pinned ``TransformationHelper.h``, CUDA
    # branch, lines 126-139) is ``-log(val)`` with the near-one guard
    # ``log = val >= 1 - eps / 2 ? -eps / 2 : at::log(val)``.
    #
    # The guard is required, not defensive: ``curand_uniform`` returns values in
    # ``(0, 1]`` and the largest mapped word rounds to exactly ``1.0`` in both
    # dtypes (float32 ``word * 2**-32 + 2**-33`` for word ``0xFFFFFFFF``, and the
    # 53-bit float64 combination ``(2**53 - 1)`` scaled by ``2**-53`` plus
    # ``2**-54``), where ``log(1.0) == 0`` would return a zero sample instead of
    # the pinned ``eps / 2``.  ``T`` is the accumulator scalar
    # (``accscalar_t``): float32 for float32 tensors and float64 for float64
    # tensors, so the guard follows the ``uniform`` array dtype.
    if dtype is torch.float64:
        uniform = _uniform2_double(bits)
    else:
        uniform = np.stack(
            [_uniform_float32(bits[:, index]) for index in range(bits.shape[1])],
            axis=-1,
        )
    tiny = np.finfo(uniform.dtype).eps
    half = uniform.dtype.type(tiny) / uniform.dtype.type(2)
    log_value = np.where(
        uniform >= uniform.dtype.type(1.0) - half,  # val >= 1 - eps / 2
        -half,
        np.log(uniform),
    )
    return -log_value


# --------------------------------------------------------------------------- #
# CUDA ``fused_dropout`` mask family and the CUDA-compatible dropout API
# --------------------------------------------------------------------------- #
def _resolve_profile(profile: Optional[dict]) -> dict:
    """Validate and copy a ``(sm_count, max_threads_per_sm, block_size)`` profile."""
    resolved = dict(CUDA_RNG_PROFILE if profile is None else profile)
    for key in ("sm_count", "max_threads_per_sm", "block_size"):
        value = int(resolved[key])
        if value <= 0:
            raise ValueError(f"profile[{key!r}] must be positive, got {value!r}")
        resolved[key] = value
    return resolved


def _dropout_launch(numel: int, profile: dict) -> Tuple[int, int]:
    """Return ``(stride, counter_advance)`` of one CUDA ``fused_dropout`` launch.

    ``dropout_cuda`` uses ``block_size = 256``,
    ``grid = min(ceil(numel / block_size), sm_count * (max_threads_per_sm // block_size))``
    and ``counter_offset = ((numel - 1) / (block_size * grid * UNROLL) + 1) * UNROLL``.
    """
    block_size = int(profile["block_size"])
    blocks_per_sm = max(1, int(profile["max_threads_per_sm"]) // block_size)
    grid = min(
        (numel + block_size - 1) // block_size,
        int(profile["sm_count"]) * blocks_per_sm,
    )
    stride = block_size * max(1, grid)
    advance = ((numel - 1) // (stride * _DROPOUT_UNROLL) + 1) * _DROPOUT_UNROLL
    return stride, advance


def _keep_cutoff(p_keep: float) -> int:
    """Smallest word whose float32 ``curand_uniform4`` value is not below ``p_keep``.

    ``word -> float32(word * 2**-32 + 2**-33)`` is monotone non-decreasing, so the
    float32 keep test ``uniform < p_keep`` is exactly ``word < cutoff`` for the
    integer ``cutoff`` returned here (the search mirrors the independent CUDA
    derivation in the host dropout study).  For a float32 tensor the kernel's
    ``accscalar_t`` is ``float``, so the keep probability itself is compared as a
    float32 value.
    """
    target = float(np.float32(p_keep))
    low, high = 0, 1 << 32
    while low < high:
        middle = (low + high) // 2
        uniform = float(np.float32(middle) * _UNIFORM32_SCALE + _UNIFORM32_HALF)
        if uniform < target:
            low = middle + 1
        else:
            high = middle
    return low


def _keep_flags(
    words: np.ndarray, dtype: torch.dtype, cutoff: Optional[int], p_keep: float
) -> np.ndarray:
    """``rand.x < p`` keep flags of one ``curand_uniform4`` word window."""
    if dtype is torch.float64:
        # accscalar_t == double: the float32 uniform is promoted and compared in
        # double precision, which differs from the float32 test near the cutoff.
        uniform = _uniform_float32(words).astype(np.float64)
        return uniform < np.float64(p_keep)
    return words < np.uint64(cutoff)


def _bernoulli_flags(
    seed: int,
    offset: int,
    numel: int,
    p_keep: float,
    dtype: torch.dtype,
    vec: Optional[int],
    profile: dict,
) -> Tuple[np.ndarray, int]:
    """``fused_dropout`` keep flags (0/1, ``np.uint8``) and the consumed offset."""
    _np_dtype(dtype)  # validates that the dtype is contracted
    numel = int(numel)
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel!r}")
    p_keep = float(p_keep)
    if not 0.0 <= p_keep <= 1.0:
        raise ValueError(f"keep probability must be in [0, 1], got {p_keep!r}")
    if vec is None:
        # Flat count only: assume the dense, maximally aligned case.  The pinned
        # ``can_vectorize_up_to`` checks alignment to ``vec * sizeof(scalar_t)``
        # for up to ``UNROLL == 4`` scalar elements; there is no 16-byte cap, so
        # an aligned float64 vector uses vec 4.
        vec = _DROPOUT_UNROLL
        while vec > 1 and numel % vec:
            vec //= 2
    vec = int(vec)
    if vec not in (1, 2, _DROPOUT_UNROLL):
        raise ValueError(f"vector size must be 1, 2 or 4, got {vec!r}")
    if vec > 1 and numel % vec:
        raise ValueError(
            f"vectorised dropout needs numel % {vec} == 0, got numel={numel}"
        )
    if numel == 0:
        return np.empty(0, dtype=np.uint8), offset
    if offset % 4:
        raise ValueError(f"philox offsets are 4-aligned, got {offset!r}")

    stride, advance = _dropout_launch(numel, profile)
    counter = offset // 4
    cutoff = _keep_cutoff(p_keep) if dtype is torch.float32 else None
    flags = np.empty(numel, dtype=np.uint8)

    if vec > 1:
        # ``fused_dropout_kernel_vec``: one thread handles ``vec`` contiguous
        # elements; thread ``tid`` of iteration ``it`` covers group
        # ``it * stride + tid``.  The kernel's ``gridxvec_loop_state`` starts at
        # 0 and is XOR-ed only inside the ``else`` branch it guards, so every
        # iteration takes the fresh ``curand_uniform4`` branch: for ``VEC == 2``
        # each iteration consumes one new 4-word engine call and uses only the
        # returned ``(x, y)`` words (``(z, w)`` are discarded).  The H100
        # long/misaligned recordings confirm this (reusing the previous call's
        # ``(z, w)`` shifts 10,771 mask bits starting at element 540,685).
        groups = numel // vec
        for iteration in range((groups - 1) // stride + 1):
            words = _philox4x32_10(seed, counter + iteration, stride)
            start = iteration * stride
            count = min(stride, groups - start)
            if vec == _DROPOUT_UNROLL:
                window = words[:count]
            else:  # vec == 2: fresh (x, y) of every iteration
                window = words[:count, 0:2]
            flags[start * vec : (start + count) * vec] = _keep_flags(
                window, dtype, cutoff, p_keep
            ).reshape(-1)
    else:
        # ``fused_dropout_kernel``: four words per engine call, elements
        # ``it * stride * 4 + ii * stride + tid``.
        rounds = (
            numel + stride * _DROPOUT_UNROLL - 1
        ) // (stride * _DROPOUT_UNROLL)
        for iteration in range(rounds):
            words = _philox4x32_10(seed, counter + iteration, stride)
            keep = _keep_flags(words, dtype, cutoff, p_keep)
            base = iteration * stride * _DROPOUT_UNROLL
            for inner in range(_DROPOUT_UNROLL):
                start = base + inner * stride
                count = max(0, min(stride, numel - start))
                if count:
                    flags[start : start + count] = keep[:count, inner]

    return flags, offset + advance


def bernoulli_mask(
    numel: int,
    p_keep: float,
    *,
    dtype: torch.dtype = torch.float32,
    vec: Optional[int] = None,
    generator: Optional["CudaCompatGenerator"] = None,
    device=None,
) -> np.ndarray:
    """CUDA-compatible ``fused_dropout`` keep mask (0/1) for ``numel`` elements.

    The mask is a host ``np.uint8`` array in the logical element order of a
    dense tensor; the draw is charged to ``generator`` when given, otherwise to
    the compatibility stream of ``device`` (default: the current torch default
    device).
    """
    if generator is not None:
        return generator.bernoulli_mask(numel, p_keep, dtype=dtype, vec=vec)
    dev = _resolve_device(device)
    state = _state_for(dev.type)
    with _LOCK:
        offset_before = state.offset
        compat = CudaCompatGenerator(seed=state.seed, offset=offset_before)
        flags = compat.bernoulli_mask(numel, p_keep, dtype=dtype, vec=vec)
        state.offset = compat.offset
        _trace(
            "dropout_mask",
            (int(numel),),
            dtype,
            state.seed,
            offset_before,
            state.offset,
            dev.type,
            extra={"p_keep": float(p_keep), "p": 1.0 - float(p_keep), "vec": vec},
        )
    return flags


def dropout_offset_advance(
    numel: int, *, profile: Optional[dict] = None
) -> int:
    """Philox words one CUDA ``fused_dropout`` launch over ``numel`` consumes."""
    numel = int(numel)
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel!r}")
    if numel == 0:
        return 0
    return _dropout_launch(numel, _resolve_profile(profile))[1]


def _is_non_overlapping_and_dense(tensor: torch.Tensor) -> bool:
    """ATen ``compute_non_overlapping_and_dense`` (public stride analysis).

    Same predicate as ``torch._prims_common.is_non_overlapping_and_dense`` and
    as the tensor property the pinned ``get_vector_size`` calls, implemented
    from the public shape/stride API so the module needs no private helper and
    no tensor method that the installed torch does not expose.
    """
    if tensor.layout != torch.strided:
        return False
    if tensor.dim() == 0:
        return True
    if tensor.dim() == 1:
        return tensor.stride(0) == 1
    expected = 1
    for length, stride in sorted(
        zip(tensor.shape, tensor.stride()), key=lambda item: item[1]
    ):
        if length == 1:
            continue
        if stride != expected:
            return False
        expected *= length
    return True


def dropout_vector_size(tensor: torch.Tensor) -> int:
    """Vector width ``get_vector_size`` in the pinned ``Dropout.cu`` selects.

    The pinned rule requires ``is_non_overlapping_and_dense`` on the input,
    output and mask tensors and then asks ``can_vectorize_up_to<scalar_t>(ptr)``:
    ``4`` when the pointer is aligned to ``4 * sizeof(scalar_t)`` (32 bytes for
    float64), ``2`` when aligned to ``2 * sizeof(scalar_t)``, otherwise ``1``.
    ``UNROLL == 4`` is the only cap -- there is no 16-byte cap, so an aligned
    float64 buffer uses vec 4.  The width is then halved while
    ``numel % vec != 0``.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("dropout_vector_size expects a torch.Tensor")
    if not _is_non_overlapping_and_dense(tensor):
        return 1  # the vec-1 kernel keeps the IndexToOffset (logical) order
    element_size = int(tensor.element_size())
    if element_size <= 0:
        return 1
    pointer = int(tensor.data_ptr())
    numel = int(tensor.numel())
    vec = _DROPOUT_UNROLL
    while vec > 1 and (
        pointer % (vec * element_size) != 0 or numel % vec != 0
    ):
        vec //= 2  # can_vectorize_up_to alignment + remainder reduction
    return vec


def _dropout_scale(p_keep: float, dtype: torch.dtype) -> float:
    """``accscalar(1) / accscalar(keep)`` of the pinned kernel."""
    if dtype is torch.float64:
        return 1.0 / float(p_keep)
    return float(np.float32(1.0) / np.float32(p_keep))


DEVICE_DTYPE_CONTRACT = {
    torch.float32: (
        "exact float32 device masked product; mask bits and the Philox stream "
        "are the CUDA fused_dropout values (src * (kept * scale) in float32)"
    ),
    torch.float64: (
        "exact float64 CUDA keep mask, uniform/normal transforms and Philox "
        "offsets, but the device masked product is unsupported on this stack: "
        "torch_npu maps a requested Double cast/elementwise target to Float "
        "(ToKernelNpu.cpp v2.5.1 lines 149-155), so Tensor.to(torch.float64) "
        "returns float32 and no truthful float64 device product exists; "
        "dropout and dropout_mask_like raise NotImplementedError for float64 "
        "NPU inputs"
    ),
}

#: Dtypes whose *device product* is implemented.  ``float64`` keeps its exact
#: generator/mask contract but has no truthful device arithmetic here.
DEVICE_PRODUCT_DTYPES = (torch.float32,)


def device_dtype_contract(dtype: torch.dtype) -> str:
    """Supported device dtypes and their documented arithmetic contract."""
    try:
        return DEVICE_DTYPE_CONTRACT[dtype]
    except KeyError:
        raise NotImplementedError(
            "the CUDA-compatible dropout mask is implemented for float32 and "
            f"float64 only; got dtype {dtype}"
        ) from None


def _require_device_product_dtype(dtype: torch.dtype, operation: str) -> None:
    """Reject a dtype that has no truthful device product, before any draw."""
    device_dtype_contract(dtype)  # rejects dtypes outside float32/float64
    if dtype not in DEVICE_PRODUCT_DTYPES:
        raise NotImplementedError(
            "{}: {}".format(operation, DEVICE_DTYPE_CONTRACT[torch.float64])
        )


def _dropout_mask_tensor(
    flags: np.ndarray, tensor: torch.Tensor, vec: int
) -> torch.Tensor:
    """Place the flat CUDA mask on ``tensor`` with the kernel's index order.

    ``fused_dropout_kernel_vec`` indexes ``a.data[linearIndex]`` in *memory*
    order, so for ``vec > 1`` the flat mask is viewed with the tensor's own
    strides.  The ``vec == 1`` kernel (which is also the kernel the source
    selects for every non-dense layout) indexes logical elements through
    ``IndexToOffset``, so the flat mask is reshaped.  Both are identical for
    contiguous tensors.
    """
    keep = torch.from_numpy(flags).to(device=tensor.device, dtype=tensor.dtype)
    shape = tuple(int(dim) for dim in tensor.shape)
    if vec > 1 and not tensor.is_contiguous():
        return keep.as_strided(shape, tuple(int(s) for s in tensor.stride()))
    return keep.reshape(shape)


def _device_dropout_mask(
    tensor: torch.Tensor, p_keep: float, vec: Optional[int] = None
) -> torch.Tensor:
    """Scaled (``0`` / ``1/keep``) CUDA-compatible mask on ``tensor.device``."""
    dtype = tensor.dtype
    _require_device_product_dtype(dtype, "CUDA-compatible device dropout")
    p_keep = float(p_keep)
    if not 0.0 <= p_keep <= 1.0:
        raise ValueError(f"keep probability must be in [0, 1], got {p_keep!r}")
    numel = int(tensor.numel())
    resolved_vec = dropout_vector_size(tensor) if vec is None else int(vec)
    if numel == 0:
        return torch.empty_like(tensor)
    device = tensor.device
    state = _state_for(device.type)
    with _LOCK:
        offset_before = state.offset
        compat = CudaCompatGenerator(seed=state.seed, offset=offset_before)
        flags = compat.bernoulli_mask(numel, p_keep, dtype=dtype, vec=resolved_vec)
        state.offset = compat.offset
        _trace(
            "dropout",
            tuple(int(dim) for dim in tensor.shape),
            dtype,
            state.seed,
            offset_before,
            state.offset,
            device.type,
            extra={"p": 1.0 - p_keep, "p_keep": p_keep, "vec": resolved_vec},
        )
    return _dropout_mask_tensor(flags, tensor, resolved_vec) * _dropout_scale(
        p_keep, dtype
    )


def dropout_mask_like(
    tensor: torch.Tensor, p_keep: float, vec: Optional[int] = None
) -> torch.Tensor:
    """CUDA-compatible dropout mask (``0`` / ``1/keep``) with ``tensor.shape``.

    Only ``npu`` tensors are supported: CPU and CUDA targets keep using their
    native generator (see :func:`dropout`, which delegates verbatim there).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("dropout_mask_like expects a torch.Tensor")
    if tensor.device.type != "npu":
        raise NotImplementedError(
            "the CUDA-compatible dropout mask is only generated for npu tensors; "
            f"got {tensor.device} (CPU/CUDA keep their native generator)"
        )
    return _device_dropout_mask(tensor, p_keep, vec=vec)


def dropout(
    input: torch.Tensor, p: float, train: bool = True, inplace: bool = False
) -> torch.Tensor:
    """CUDA-compatible ``torch.nn.functional.dropout`` for the NPU product path.

    ``train=False``, ``p == 0``, ``p == 1`` and empty tensors are the source
    short-cuts and do not consume the stream.  Non-NPU targets delegate
    verbatim to :func:`torch.nn.functional.dropout`.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("dropout expects a torch.Tensor")
    p = float(p)
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"dropout probability has to be between 0 and 1, but got {p}"
        )
    if input.device.type != "npu":
        return F.dropout(input, p, train, inplace)
    # The device product exists for float32 only (see DEVICE_DTYPE_CONTRACT);
    # reject any other dtype before the stream is consumed.
    _require_device_product_dtype(input.dtype, "CUDA-compatible device dropout")
    if p == 0.0 or not train or input.numel() == 0:
        return input
    if p == 1.0:
        if inplace:
            input.zero_()
            return input
        return torch.zeros_like(input)
    mask = _device_dropout_mask(input, 1.0 - p)
    result = input * mask
    if inplace:
        input.copy_(result)
        return input
    return result


class CudaCompatDropout(torch.nn.Module):
    """Drop-in replacement for ``torch.nn.Dropout`` on the product path.

    Carries no parameters or buffers, so every checkpoint state dict keeps the
    position it had with ``nn.Dropout``.  On ``npu`` the mask comes from the
    CUDA-compatible ``fused_dropout`` stream and the multiply runs on the
    device; on CPU/CUDA the module is ``nn.Dropout`` verbatim (same native
    generator, same RNG consumption).
    """

    def __init__(self, p: float = 0.1, inplace: bool = False) -> None:
        super().__init__()
        p = float(p)
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p
        self.inplace = bool(inplace)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return dropout(input, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


# --------------------------------------------------------------------------- #
# Generator class
# --------------------------------------------------------------------------- #
class CudaCompatGenerator:
    """Explicit CUDA-compatible generator (seed, offset, stream profile)."""

    def __init__(
        self,
        seed: int = 0,
        offset: int = 0,
        profile: Optional[dict] = None,
    ) -> None:
        self.seed = int(seed) & _SEED_MASK
        self.offset = int(offset)
        if self.offset < 0:
            raise ValueError(f"philox offset must be non-negative, got {offset!r}")
        if self.offset % 4:
            raise ValueError(
                "philox offsets produced by torch draws are 4-aligned; got "
                f"{offset!r}"
            )
        resolved = _resolve_profile(profile)
        self.profile = resolved

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"CudaCompatGenerator(seed={self.seed}, offset={self.offset}, "
            f"profile={self.profile!r})"
        )

    # -- public draws ------------------------------------------------------- #
    def normal(self, shape: ShapeLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if dtype is not torch.float32:
            raise NotImplementedError(
                "the CUDA-compatible normal transform is implemented for float32 "
                f"(curand_normal4); got dtype {dtype}"
            )
        return self._draw("normal", shape, dtype)

    def exponential(
        self, shape: ShapeLike, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if dtype is not torch.float32 and dtype is not torch.float64:
            raise NotImplementedError(
                "the CUDA-compatible exponential transform is implemented for "
                f"float32 and float64; got dtype {dtype}"
            )
        return self._draw("exponential", shape, dtype)

    def bernoulli_mask(
        self,
        numel: int,
        p_keep: float,
        dtype: torch.dtype = torch.float32,
        vec: Optional[int] = None,
    ) -> np.ndarray:
        """CUDA ``fused_dropout`` keep mask (0/1 ``np.uint8``) for ``numel``.

        Advances :attr:`offset` by the exact counter consumption of the pinned
        kernel (``vec=None`` falls back to the dense, maximally aligned rule).
        """
        flags, self.offset = _bernoulli_flags(
            self.seed, self.offset, numel, p_keep, dtype, vec, self.profile
        )
        return flags

    # -- generation --------------------------------------------------------- #
    def _draw(self, kind: str, shape: ShapeLike, dtype: torch.dtype) -> torch.Tensor:
        dims = _normalise_shape(shape)
        np_dtype = _np_dtype(dtype)
        numel = 1
        for dim in dims:
            numel *= dim
        if numel == 0:
            return torch.empty(dims, dtype=dtype)

        block_size = int(self.profile["block_size"])
        blocks_per_sm = max(1, int(self.profile["max_threads_per_sm"]) // block_size)
        grid = min(
            (numel + block_size - 1) // block_size,
            int(self.profile["sm_count"]) * blocks_per_sm,
        )
        grid = max(1, grid)
        stride = block_size * grid
        unroll = 4 if dtype is torch.float32 else 2

        offset_before = self.offset
        counter = offset_before // 4
        advance = ((numel - 1) // (stride * 4) + 1) * 4
        iterations = (numel - 1) // (stride * unroll) + 1

        flat = np.empty(numel, dtype=np_dtype)
        for iteration in range(iterations):
            bits = _philox4x32_10(self.seed, counter + iteration, stride)
            values = _transform(kind, dtype, bits)
            for inner in range(unroll):
                start = (iteration * unroll + inner) * stride
                remaining = numel - start
                if remaining <= 0:
                    break
                chunk = min(stride, remaining)
                flat[start : start + chunk] = values[:chunk, inner]

        self.offset = offset_before + advance
        return torch.from_numpy(flat.reshape(dims))


# --------------------------------------------------------------------------- #
# Module-level helpers used by the product path
# --------------------------------------------------------------------------- #
def _device_draw(kind: str, shape: ShapeLike, device, dtype: torch.dtype) -> torch.Tensor:
    dev = _resolve_device(device)
    dims = _normalise_shape(shape)
    state = _state_for(dev.type)
    with _LOCK:
        offset_before = state.offset
        generator = CudaCompatGenerator(seed=state.seed, offset=offset_before)
        if kind == "normal":
            result = generator.normal(dims, dtype=dtype)
        else:
            result = generator.exponential(dims, dtype=dtype)
        state.offset = generator.offset
        _trace(kind, dims, dtype, state.seed, offset_before, state.offset, dev.type)
    return result.to(dev)


def randn(
    shape: ShapeLike, *, device=None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """CUDA-compatible ``torch.randn`` (device streams keep native torch)."""
    if _resolve_device(device).type != "npu":
        return torch.randn(shape, device=device, dtype=dtype)
    return _device_draw("normal", shape, device, dtype)


def randn_like(tensor: torch.Tensor) -> torch.Tensor:
    """CUDA-compatible ``torch.randn_like``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("randn_like expects a torch.Tensor")
    if tensor.device.type != "npu":
        return torch.randn_like(tensor)
    return randn(tensor.shape, device=tensor.device, dtype=tensor.dtype)


def exponential(
    shape: ShapeLike, *, device=None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """CUDA-compatible ``torch.empty(shape).exponential_(1)``."""
    if _resolve_device(device).type != "npu":
        return torch.empty(shape, device=device, dtype=dtype).exponential_()
    return _device_draw("exponential", shape, device, dtype)


def multinomial(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """CUDA-compatible ``torch.multinomial(probs, num_samples)``.

    The CUDA fast path for ``num_samples == 1`` without replacement is
    ``argmax(probs / Exponential(1))``; the exponential variates come from the
    compatibility stream and the division/argmax execute on ``probs.device``.
    CPU and CUDA targets delegate verbatim to ``torch.multinomial``.
    """
    if not isinstance(probs, torch.Tensor):
        raise TypeError("multinomial expects a torch.Tensor of probabilities")
    if probs.device.type != "npu":
        return torch.multinomial(probs, num_samples)
    if int(num_samples) != 1:
        raise NotImplementedError(
            "the CUDA-compatible multinomial path covers the documented "
            f"num_samples=1 fast path; got num_samples={num_samples}"
        )
    noise = exponential(probs.shape, device=probs.device, dtype=probs.dtype)
    return torch.div(probs, noise).argmax(dim=-1, keepdim=True)
