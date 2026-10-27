"""CUDA-compatible dropout (``fused_dropout``) parity, plan and route tests.

RF2's ``nn.Dropout`` layers run with ``model.training=True`` (the checkpoint is
loaded without ``eval()``), so every RF2 forward consumes a *device* random
stream.  On the original CUDA device that stream is ``fused_dropout_kernel``
from ``aten/src/ATen/native/cuda/Dropout.cu`` (pinned to torch 2.3.1); the
Ascend vendor generator is a different stream, so the versioned CUDA references
cannot be reproduced without reproducing the mask family itself.

``rfantibody.rng`` therefore also implements the CUDA dropout mask family
(``VEC in (1, 2, 4)`` layouts, float32/float64 keep tests, ``src * mask * scale``
in ``accscalar`` precision, the exact 4-word Philox counter consumption).  This
module checks it against **independent CUDA references**, never against the
candidate itself:

* ``reference_outputs/cuda_dropout_holdouts`` -- 126 recorded H100 CUDA
  ``fused_dropout`` masks (3 seeds x 7 sizes x {float32, float64} x 3 drop
  probabilities x the vector width that size uses), each with the mask SHA-256,
  the recorded ``mismatch`` (0) and the Philox offset before/after the call;
* ``reference_outputs/cuda_rf2_dropout_plan`` -- the 902 ``F.dropout`` events of
  the original deterministic RF2 run (shape/stride/dtype/p and the exact
  offset pair each event consumed), i.e. the real product draw plan.

Requirements are declared here, before any candidate result is inspected, and
are never relaxed afterwards:

==========================  =========================================  ==============
check                       requirement                                reference
==========================  =========================================  ==============
126 dropout masks           mask bits bit-identical, mask SHA-256 equal, H100 CUDA
                            ``mismatch == 0``                          recording
126 counter advances        ``offset_after == post`` exactly           H100 CUDA
per-seed replay             every recorded ``offset`` and ``post``     H100 CUDA
                            reproduced in file order                   recording
902-event RF2 draw plan     every recorded offset pair reproduced      CUDA RF2 run
                            exactly (``0 -> 112`` ... ``-> 101024``)
vector width dispatch       the pinned ``get_vector_size`` rule:     pinned source
                            ``can_vectorize_up_to`` alignment to
                            ``vec * sizeof(scalar_t)`` with
                            ``UNROLL = 4`` as the only cap
source short-cuts           ``p=0``/``p=1``/``train=False``/empty      pinned source
                            consume no stream
device route                the mask multiply executes on ``npu:0``    runtime
                            with exact ``src * scale`` values
CPU/CUDA delegation         the wrapper equals ``F.dropout`` and       native torch
                            consumes the native generator identically
layout mask placement       every strided layout returns an input-     pinned source
                            shaped mask in the kernel's index order
                            and advances the stream exactly
layout predicate            equals the ATen / ``torch._prims_common``  pinned source
                            non-overlapping-and-dense predicate
exponential near-one guard  the largest mapped word yields ``eps/2``,  pinned source
                            ordinary words keep ``-log(u)``
layout holdouts             the four independent H100 long/            H100 CUDA
                            misaligned recordings reproduce bit-       recording
                            identically, with the width the pinned
                            source selects from their pointer metadata
device dtype contract       float32/float64 generator contracts          pinned source
                            documented, every other dtype rejected;     + runtime
                            the float64 *device product* is              + host
                            documented unsupported and its rejection     observation
                            is executed (``NotImplementedError``)
==========================  =========================================  ==============

On a host with a genuine Ascend NPU the dropout multiply is executed through
the product API on ``npu:0`` (device, values and stream offsets asserted).
Without an NPU the identical expectations are evaluated with the compatibility
generator directly; no case is skipped.

The four files under ``reference_outputs/cuda_dropout_layout_holdouts`` are
**diagnostic test oracles**: independent H100 80GB HBM3 ``fused_dropout``
recordings of long/misaligned contiguous tensors, kept byte-identical to the
host recording.  No product code imports them; the pinned source they exercise
is PyTorch ``v2.5.1`` ``Dropout.cu`` / ``MemoryAccess.cuh`` (``UNROLL = 4``,
``gridxvec_loop_state`` starting at 0, ``can_vectorize_up_to``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rfantibody import rng as antibody_rng
from rfantibody.device import npu_available

# --- Pre-declared requirements (fixed before candidate execution) -----------
ORACLE_SHA256 = "c8a9c07b36db6dfa19db6ef1810a9067f439ef1fb93ce98b41ec6fddec129ce8"
ORACLE_CASES = 126
PLAN_SHA256 = "e3c01008702e6145ce69528aea702c08537e0ae245e3469d80e38832c3a5bc1a"
PLAN_EVENTS = 902
PLAN_FINAL_OFFSET = 101024
PLAN_FIRST_OFFSET_PAIR = (0, 112)
DROPOUT_P = 0.1
MASK_VALUE_TOLERANCE = 0.0  # the multiply is bit-exact, so exact equality
THROUGHPUT_NUMEL = 58491904  # the (1, 478, 478, 256) product shape

# The four independent H100 long/misaligned layout recordings (torch 2.5.1+cu118,
# NVIDIA H100 80GB HBM3).  Declared here, before any candidate result is
# inspected: the manifest bytes, the per-file npz bytes, the recorded keep-mask
# bytes, the Philox offset pair, and the vector width the pinned source selects
# from the recorded pointer metadata (never from a profiler kernel label).
LAYOUT_MANIFEST_SHA256 = (
    "c888a92c64ae638dfcfcc8443e7af6116fcf7ab279c846637f822a686013389a"
)
LAYOUT_HOLDOUTS = (
    {
        "file": "0.npz",
        "dtype": torch.float32,
        "numel": 600002,
        "mod32": 0,
        "vec": 2,
        "offset_after": 4,
        "npz_sha256": (
            "c1330d472038e1b7a51e64fd5c6ebea16fa329b1cedc9dc08ef676ae59556f52"
        ),
        "mask_sha256": (
            "b93ca4bb88f1b798b7fb68b3927a6f7c0c887bbe01037f7b9485ce8037b83095"
        ),
    },
    {
        "file": "1.npz",
        "dtype": torch.float64,
        "numel": 600000,
        "mod32": 0,
        "vec": 4,
        "offset_after": 4,
        "npz_sha256": (
            "a732c3eff1cab3c69e9d6df8e709fb326b7232732f6e41da27d7d272570e49d1"
        ),
        "mask_sha256": (
            "d9f7bc8f923e926c26d64e91af7dbb15243d97af6f41e7a4a087d41608a06dc4"
        ),
    },
    {
        "file": "2.npz",
        "dtype": torch.float64,
        "numel": 600002,
        "mod32": 0,
        "vec": 2,
        "offset_after": 4,
        "npz_sha256": (
            "c1330d472038e1b7a51e64fd5c6ebea16fa329b1cedc9dc08ef676ae59556f52"
        ),
        "mask_sha256": (
            "b93ca4bb88f1b798b7fb68b3927a6f7c0c887bbe01037f7b9485ce8037b83095"
        ),
    },
    {
        "file": "3.npz",
        "dtype": torch.float32,
        "numel": 600000,
        "mod32": 8,
        "vec": 2,
        "offset_after": 4,
        "npz_sha256": (
            "8a76618be65c2b88e09cdd16865363aef6d63267f0f149135296e8555afe63b3"
        ),
        "mask_sha256": (
            "39e759c32ff11e7309c0f7d247681610d0a295e007cd1569254c838c0855f786"
        ),
    },
)
LAYOUT_SOURCE_SHA256 = {
    "aten/src/ATen/native/cuda/Dropout.cu": (
        "dd8802fd7214dbc8880c446198b385915dc34932ea0f04f419911377c9ad926e"
    ),
    "aten/src/ATen/native/cuda/MemoryAccess.cuh": (
        "73877e2ca8f81668397d23730fc0e2846abcca955920b1508f9014087cec2372"
    ),
}

# --- Asset locations --------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_HOLDOUT_DIR = _MODULE_DIR / "reference_outputs" / "cuda_dropout_holdouts"
_PLAN_DIR = _MODULE_DIR / "reference_outputs" / "cuda_rf2_dropout_plan"
_LAYOUT_DIR = _MODULE_DIR / "reference_outputs" / "cuda_dropout_layout_holdouts"
_ORACLE_PATH = _HOLDOUT_DIR / "dropout-heldout-oracles.json"
_PLAN_PATH = _PLAN_DIR / "rf2-dropout-plan.json"

_EVIDENCE_DIR = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "outputs" / "evidence")
) / "rng_parity"
_EVIDENCE_PATH = _EVIDENCE_DIR / "cuda_compat_dropout_parity.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise RuntimeError("{} asset is missing: {}".format(description, path))
    return json.loads(path.read_text())


_ORACLE = _load_json(_ORACLE_PATH, "CUDA dropout holdout oracle")
_CASES = _ORACLE["cases"]
_PLAN = _load_json(_PLAN_PATH, "CUDA RF2 dropout draw plan")


def _case_id(case: dict) -> str:
    return "s{seed}-n{n}-{dtype}-vec{vec}-p{p}".format(
        seed=case["seed"],
        n=case["n"],
        dtype=case["dtype"].split(".")[-1],
        vec=case["vec"],
        p=case["p"],
    )


def _torch_dtype(name: str) -> torch.dtype:
    return {"torch.float32": torch.float32, "torch.float64": torch.float64}[name]


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _keep_probability(p_drop: float) -> float:
    return 1.0 - float(p_drop)


def _host_mask(seed: int, offset: int, case: dict) -> tuple[np.ndarray, int]:
    generator = antibody_rng.CudaCompatGenerator(seed=seed, offset=offset)
    flags = generator.bernoulli_mask(
        case["n"],
        _keep_probability(case["p"]),
        dtype=_torch_dtype(case["dtype"]),
        vec=case["vec"],
    )
    return flags, generator.offset


def _accel_device():
    if npu_available():
        try:
            if int(torch.npu.device_count()) >= 1:
                return torch.device("npu:0")
        except Exception:  # pragma: no cover - vendor stack anomaly
            return None
    return None


# --- Layout construction (alignment and strides are controlled) -------------
def _alignment_offset(
    pointer: int, target_mod: int, element_size: int, alignment: int = 32
) -> int:
    """Element offset so ``pointer + offset * element_size`` hits ``target_mod``."""
    for offset in range(alignment // element_size):
        if (pointer + offset * element_size) % alignment == target_mod:
            return offset
    raise AssertionError(
        "cannot place a {}-byte element buffer at {} modulo {} bytes".format(
            element_size, target_mod, alignment
        )
    )


def _view_1d(
    numel,
    dtype=torch.float32,
    mod32=0,
    device=None,
    filled=False,
):
    """Contiguous 1-D view with a controlled ``data_ptr % 32``."""
    element_size = int(torch.empty(0, dtype=dtype).element_size())
    base_size = int(numel) + 16
    if filled:
        base = torch.arange(1, base_size + 1, dtype=dtype, device=device)
    else:
        base = torch.empty(base_size, dtype=dtype, device=device)
    offset = _alignment_offset(int(base.data_ptr()), mod32, element_size)
    view = base.narrow(0, offset, int(numel))
    assert int(view.data_ptr()) % 32 == mod32, (
        "alignment control failed: data_ptr % 32 == {} instead of {}".format(
            int(view.data_ptr()) % 32, mod32
        )
    )
    assert view.is_contiguous()
    return view


def _contiguous(shape, dtype=torch.float32, mod32=0, device=None, filled=False):
    return _view_1d(
        _numel(shape), dtype=dtype, mod32=mod32, device=device, filled=filled
    ).view(tuple(int(dim) for dim in shape))


def _primary_source_vector_size(pointer, element_size, numel):
    """``get_vector_size`` of the pinned ``Dropout.cu``, from first principles.

    ``can_vectorize_up_to<scalar_t>(ptr)`` in ``MemoryAccess.cuh`` returns 4 when
    the pointer is aligned to ``4 * sizeof(scalar_t)``, 2 when it is aligned to
    ``2 * sizeof(scalar_t)``, else 1; ``UNROLL == 4`` is the only cap.  The width
    is then halved while ``numel % vec != 0``.
    """
    vec = 4 if pointer % (4 * element_size) == 0 else (
        2 if pointer % (2 * element_size) == 0 else 1
    )
    while vec > 1 and int(numel) % vec:
        vec //= 2
    return vec


def _layout_cases(device=None):
    """The required layout matrix: ``(name, tensor, dense?, contract)``."""
    return [
        (
            "contiguous_2x3x5_float32",
            _contiguous((2, 3, 5), filled=True, device=device),
            True,
            "dense and 16-byte aligned -> logical order equals memory order",
        ),
        (
            "transposed_dense_4x3_float32",
            _contiguous((3, 4), filled=True, device=device).t(),
            True,
            "dense transposed view -> vec > 1, mask placed with the tensor "
            "strides (memory order)",
        ),
        (
            "transposed_vec1_3x5_float32",
            _contiguous((5, 3), filled=True, device=device).t(),
            True,
            "dense but numel 15 -> vec 1, logical (IndexToOffset) order",
        ),
        (
            "strided_slice_8x8_step2_float32",
            _contiguous((8, 8), filled=True, device=device)[:, ::2],
            False,
            "non-dense slice -> vec 1, logical order",
        ),
        (
            "expanded_1x3_to_4x3_float32",
            _contiguous((1, 3), filled=True, device=device).expand(4, 3),
            False,
            "stride 0 -> non-dense -> vec 1, logical order",
        ),
        (
            "zero_dim_float32",
            torch.tensor(2.5, dtype=torch.float32, device=device),
            True,
            "0-dim tensor -> vec 1, single element mask",
        ),
        (
            "empty_0x5_float32",
            torch.empty(0, 5, dtype=torch.float32, device=device),
            True,
            "empty tensor -> source early return, empty mask, no stream advance",
        ),
    ]


def _prims_dense_predicate():
    """``torch._prims_common.is_non_overlapping_and_dense`` when importable."""
    try:
        from torch._prims_common import is_non_overlapping_and_dense
    except Exception:  # pragma: no cover - very old torch
        return None
    return is_non_overlapping_and_dense


def _expected_mask(tensor, p_keep, vec, seed, offset):
    """Expected scaled mask and offset from the (oracle-proven) flat flags.

    The placement mirrors the documented kernel index order independently of
    the product helper: memory order (``as_strided`` with the tensor strides)
    for ``vec > 1``, logical order (``reshape``) otherwise.
    """
    generator = antibody_rng.CudaCompatGenerator(seed=seed, offset=offset)
    flags = generator.bernoulli_mask(
        int(tensor.numel()), p_keep, dtype=tensor.dtype, vec=vec
    )
    scale = antibody_rng._dropout_scale(p_keep, tensor.dtype)
    flat = torch.from_numpy(flags).to(device=tensor.device, dtype=tensor.dtype)
    shape = tuple(int(dim) for dim in tensor.shape)
    if vec > 1 and not tensor.is_contiguous():
        placed = flat.as_strided(shape, tuple(int(s) for s in tensor.stride()))
        order = "memory order (as_strided with the tensor strides)"
    else:
        placed = flat.reshape(shape)
        order = "logical order (reshape to the tensor shape)"
    return placed * scale, int(generator.offset), order


def _state_offset(state) -> int:
    payload = np.asarray(state, dtype=np.uint8).tobytes()
    return int.from_bytes(payload[8:16], "little")


def _state_seed(state) -> int:
    payload = np.asarray(state, dtype=np.uint8).tobytes()
    return int.from_bytes(payload[0:8], "little")


# --- Machine-readable evidence ---------------------------------------------
_EVIDENCE = {
    "module": "test.rng_parity.test_cuda_compat_dropout",
    "candidate": (
        "rfantibody.rng fused_dropout mask family (VEC 1/2/4 layouts, "
        "float32/float64 keep tests, exact 4-word counter consumption) and the "
        "CUDA-compatible dropout module used by the RF2 product path"
    ),
    "reference": (
        "independent PyTorch 2.3.1+cu118 CUDA fused_dropout recordings on an "
        "NVIDIA H100 80GB HBM3 (reference/dropout-study); never the candidate"
    ),
    "requirements": {
        "holdout_mask_bits": "bit identical (sha256 of the C-order uint8 0/1 mask)",
        "holdout_offset_after": "exact equality to the recorded post offset",
        "rf2_draw_plan": "every recorded offset pair reproduced exactly",
        "mask_multiply": "exact equality (tolerance {})".format(MASK_VALUE_TOLERANCE),
    },
    "stream_profile": dict(antibody_rng.CUDA_RNG_PROFILE),
    "oracle_integrity": {},
    "holdout_cases": [],
    "stream_continuity": [],
    "draw_plan": {},
    "vector_size_dispatch": {},
    "layout_mask_shape": {},
    "layout_predicate": {},
    "layout_holdouts": {},
    "device_dtype_contract": {},
    "exponential_guard": {},
    "source_shortcuts": {},
    "device_route": {},
    "native_generator_non_regression": {},
    "product_wiring": {},
    "throughput": {},
    "contract_errors": {},
}


@pytest.fixture(scope="module", autouse=True)
def _write_evidence():
    _EVIDENCE["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _EVIDENCE["torch_version"] = torch.__version__
    try:
        import torch_npu  # noqa: F401

        _EVIDENCE["torch_npu_version"] = getattr(torch_npu, "__version__", "unknown")
    except Exception:  # pragma: no cover - CPU/CUDA hosts
        _EVIDENCE["torch_npu_version"] = None
    device = _accel_device()
    _EVIDENCE["npu_available"] = device is not None
    if device is not None:
        try:
            _EVIDENCE["npu_device_name"] = str(torch.npu.get_device_name(0))
        except Exception:  # pragma: no cover - vendor stack anomaly
            _EVIDENCE["npu_device_name"] = None
    yield
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    _EVIDENCE_PATH.write_text(json.dumps(_EVIDENCE, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# 1. Oracle integrity
# --------------------------------------------------------------------------- #
def test_oracle_assets_integrity():
    """The copied CUDA assets still match their recorded sha256."""
    oracle_sha = _sha256(_ORACLE_PATH)
    assert oracle_sha == ORACLE_SHA256, "dropout oracle asset changed"
    plan_sha = _sha256(_PLAN_PATH)
    assert plan_sha == PLAN_SHA256, "RF2 dropout plan asset changed"

    assert _ORACLE["passed"] is True
    assert _ORACLE["count"] == ORACLE_CASES
    assert _ORACLE["failed"] == []
    assert len(_CASES) == ORACLE_CASES
    assert all(case["mismatch"] == 0 for case in _CASES)
    assert all(case["passed"] is True for case in _CASES)
    assert {case["seed"] for case in _CASES} == {
        17,
        987654321,
        9223372036854775809,
    }
    assert {case["n"] for case in _CASES} == {3, 6, 21, 478, 1025, 4097, 300000}
    assert {case["vec"] for case in _CASES} == {1, 2, 4}

    holdout_provenance = _load_json(
        _HOLDOUT_DIR / "provenance.json", "dropout holdout provenance"
    )
    plan_provenance = _load_json(
        _PLAN_DIR / "provenance.json", "dropout plan provenance"
    )
    assert (
        holdout_provenance["source"]["sha256"] == ORACLE_SHA256
    ), "holdout provenance does not describe the copied oracle"
    assert holdout_provenance["revalidation"]["sha256"] == ORACLE_SHA256
    assert plan_provenance["source"]["sha256"] == (
        "a7193294acef30e98dacd5d7556a2e9b5aec06ce269368e121478b85335d8e8f"
    )
    assert plan_provenance["copied_files"][0]["sha256"] == PLAN_SHA256

    _EVIDENCE["oracle_integrity"] = {
        "holdout_sha256": oracle_sha,
        "holdout_cases": len(_CASES),
        "plan_sha256": plan_sha,
        "plan_events": len(_PLAN["events"]),
        "revalidation_sha256": holdout_provenance["revalidation"]["sha256"],
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 2. Mask parity over the 126 independent CUDA holdouts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "case", [pytest.param(case, id=_case_id(case)) for case in _CASES]
)
def test_holdout_mask_and_counter_parity(case):
    flags, offset_after = _host_mask(case["seed"], case["offset"], case)
    mask_sha = _sha256_bytes(flags.tobytes())

    assert flags.shape == (case["n"],)
    assert flags.dtype == np.uint8
    assert set(np.unique(flags)).issubset({0, 1})
    assert mask_sha == case["mask_sha256"], (
        "mask bits differ from the CUDA recording for {}: {} != {}".format(
            _case_id(case), mask_sha, case["mask_sha256"]
        )
    )
    assert offset_after == case["post"], (
        "counter advance for {} is {}..{} but the CUDA recording is {}..{}".format(
            _case_id(case), case["offset"], offset_after, case["offset"], case["post"]
        )
    )

    _EVIDENCE["holdout_cases"].append(
        {
            "case": _case_id(case),
            "seed": case["seed"],
            "numel": case["n"],
            "dtype": case["dtype"],
            "drop_probability": case["p"],
            "keep_probability": _keep_probability(case["p"]),
            "vec": case["vec"],
            "device": "host-generated mask, compared with the H100 CUDA recording",
            "offset_before": case["offset"],
            "offset_after_expected": case["post"],
            "offset_after_actual": offset_after,
            "mask_sha256_expected": case["mask_sha256"],
            "mask_sha256_actual": mask_sha,
            "mismatch_expected": case["mismatch"],
            "verdict": (
                "passed"
                if mask_sha == case["mask_sha256"] and offset_after == case["post"]
                else "failed"
            ),
        }
    )


@pytest.mark.parametrize("seed", [17, 987654321, 9223372036854775809])
def test_holdout_stream_continuity(seed):
    """A single generator replays one seed's recorded cases in file order."""
    cases = [case for case in _CASES if case["seed"] == seed]
    generator = antibody_rng.CudaCompatGenerator(seed=seed, offset=0)
    offsets = []
    for case in cases:
        assert generator.offset == case["offset"], (
            "stream position for {} is {} but the CUDA recording starts at "
            "{}".format(_case_id(case), generator.offset, case["offset"])
        )
        flags = generator.bernoulli_mask(
            case["n"],
            _keep_probability(case["p"]),
            dtype=_torch_dtype(case["dtype"]),
            vec=case["vec"],
        )
        assert _sha256_bytes(flags.tobytes()) == case["mask_sha256"]
        assert generator.offset == case["post"]
        offsets.append((case["offset"], generator.offset))

    _EVIDENCE["stream_continuity"].append(
        {
            "seed": seed,
            "cases": len(cases),
            "first_offset_pair": list(offsets[0]),
            "last_offset_pair": list(offsets[-1]),
            "final_offset": generator.offset,
            "verdict": "passed",
        }
    )
    assert len(cases) == ORACLE_CASES // 3


# --------------------------------------------------------------------------- #
# 3. The real RF2 device-dropout draw plan (902 events)
# --------------------------------------------------------------------------- #
def _recorded_shapes() -> list[tuple[tuple[int, ...], float]]:
    shapes = []
    for event in _PLAN["events"]:
        shape = tuple(int(dim) for dim in event["shape"])
        pair = (shape, float(event["p"]))
        if pair not in shapes:
            shapes.append(pair)
    return shapes


def test_rf2_dropout_draw_plan():
    """The 902 recorded RF2 offsets must follow from shape/dtype/p alone."""
    events = _PLAN["events"]
    assert len(events) == PLAN_EVENTS

    offset = int(_PLAN["stream"]["initial_offset"])
    assert offset == 0
    assert int(_PLAN["stream"]["seed"]) == 0
    replayed = []
    for event in events:
        expected_advance = int(event["offset_after"]) - int(event["offset_before"])
        assert event["offset_before"] == offset, (
            "event {} starts at {} but the recording starts at {}".format(
                event["index"], offset, event["offset_before"]
            )
        )
        assert event["dtype"] == "torch.float32"
        assert event["p"] == DROPOUT_P
        advance = antibody_rng.dropout_offset_advance(_numel(event["shape"]))
        assert advance == expected_advance, (
            "event {} consumed {} words in the recording but the candidate "
            "model advances {}".format(event["index"], expected_advance, advance)
        )
        offset += advance
        assert offset == int(event["offset_after"])
        replayed.append((int(event["offset_before"]), offset))

    assert replayed[0] == PLAN_FIRST_OFFSET_PAIR
    assert offset == PLAN_FINAL_OFFSET
    assert _PLAN["totals"]["offset_pairs_monotone_and_gap_free"] is True

    # The cheap product shape is replayed with real masks, not just counters:
    # the draw must advance by exactly the recorded delta of those events.
    cheap = [event for event in events if _numel(event["shape"]) <= 4096]
    assert cheap, "the plan must contain a small-shape event to draw directly"
    sampled = cheap[:3]
    for event in sampled:
        generator = antibody_rng.CudaCompatGenerator(
            seed=int(_PLAN["stream"]["seed"]),
            offset=int(event["offset_before"]),
        )
        flags = generator.bernoulli_mask(
            _numel(event["shape"]),
            _keep_probability(event["p"]),
            dtype=torch.float32,
            vec=antibody_rng.dropout_vector_size(
                torch.empty(tuple(event["shape"]), dtype=torch.float32)
            ),
        )
        assert flags.shape == (_numel(event["shape"]),)
        assert generator.offset == int(event["offset_after"])
        assert set(np.unique(flags)).issubset({0, 1})

    shape_dispatch = {}
    for shape, p_drop in _recorded_shapes():
        tensor = torch.empty(shape, dtype=torch.float32)
        vec = antibody_rng.dropout_vector_size(tensor)
        shape_dispatch["x".join(str(dim) for dim in shape)] = {
            "numel": _numel(shape),
            "drop_probability": p_drop,
            "vector_size": vec,
            "offset_advance": antibody_rng.dropout_offset_advance(_numel(shape)),
        }
    assert all(entry["vector_size"] == 4 for entry in shape_dispatch.values())

    _EVIDENCE["draw_plan"] = {
        "reference": (
            "test/rng_parity/reference_outputs/cuda_rf2_dropout_plan/"
            "rf2-dropout-plan.json"
        ),
        "reference_sha256": PLAN_SHA256,
        "events": len(replayed),
        "dtype": "torch.float32",
        "drop_probability": DROPOUT_P,
        "first_offset_pair": list(replayed[0]),
        "last_offset_pair": list(replayed[-1]),
        "final_offset": offset,
        "expected_final_offset": PLAN_FINAL_OFFSET,
        "sampled_events_with_real_masks": [int(event["index"]) for event in sampled],
        "shapes": shape_dispatch,
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 4. Vector width dispatch (pinned ``get_vector_size``)
# --------------------------------------------------------------------------- #
def test_vector_size_dispatch():
    """The pinned ``get_vector_size`` rule: alignment to ``vec * sizeof``.

    ``can_vectorize_up_to<scalar_t>(ptr)`` (``MemoryAccess.cuh``) rewards
    alignment to ``vec * sizeof(scalar_t)`` for up to ``UNROLL == 4`` scalar
    elements; there is no 16-byte cap, so an aligned float64 buffer uses vec 4.
    ``get_vector_size`` additionally requires the layout to be
    non-overlapping-and-dense and halves the width while ``numel % vec != 0``.
    """
    cases = {}

    def record(name, tensor, expected, clause):
        actual = int(antibody_rng.dropout_vector_size(tensor))
        element_size = int(tensor.element_size())
        numel = int(tensor.numel())
        pointer = int(tensor.data_ptr())
        dense = bool(antibody_rng._is_non_overlapping_and_dense(tensor))
        primary = (
            _primary_source_vector_size(pointer, element_size, numel) if dense else 1
        )
        cases[name] = {
            "shape": list(tensor.shape),
            "strides": list(tensor.stride()),
            "numel": numel,
            "dtype": str(tensor.dtype),
            "non_overlapping_and_dense": dense,
            "data_ptr_mod_32": pointer % 32,
            "data_ptr_mod_vector_bytes": pointer % (int(expected) * element_size),
            "vector_bytes": int(expected) * element_size,
            "expected": int(expected),
            "actual": actual,
            "primary_source": primary,
            "source_clause": clause,
        }
        assert actual == int(expected), "{}: expected vec {} got {} ({})".format(
            name, expected, actual, clause
        )
        assert primary == int(expected), (
            "{}: the independent primary-source composition selects vec {} but "
            "the case declares {} ({})".format(name, primary, expected, clause)
        )

    # Dense, aligned pointers: the width starts at ``UNROLL == 4`` (there is no
    # 16-byte cap) and is halved while ``numel % vec != 0``.
    record(
        "product_478x478x128",
        _contiguous((1, 478, 478, 128), mod32=0),
        4,
        "non-overlapping-and-dense + can_vectorize_up_to: pointer aligned to "
        "4 * sizeof(float32) == 16 B + numel % 4 == 0",
    )
    record(
        "product_478x478x256",
        _contiguous((1, 478, 478, 256), mod32=0),
        4,
        "non-overlapping-and-dense + can_vectorize_up_to: pointer aligned to "
        "4 * sizeof(float32) == 16 B + numel % 4 == 0",
    )
    record(
        "product_1x1x478x8x1",
        _contiguous((1, 1, 478, 8, 1), mod32=0),
        4,
        "non-overlapping-and-dense + can_vectorize_up_to: pointer aligned to "
        "4 * sizeof(float32) == 16 B + numel % 4 == 0",
    )
    record(
        "numel_478_float32",
        _view_1d(478, mod32=0),
        2,
        "numel % 4 != 0 -> halve to 2 (numel % 2 == 0, 8-byte alignment holds)",
    )
    record(
        "numel_6_float32",
        _view_1d(6, mod32=0),
        2,
        "numel % 4 != 0 -> halve to 2 (numel % 2 == 0)",
    )
    record(
        "numel_3_float32",
        _view_1d(3, mod32=0),
        1,
        "odd numel -> halve to 1 even though can_vectorize_up_to returns 4",
    )
    record(
        "numel_21_float32",
        _view_1d(21, mod32=0),
        1,
        "odd numel -> halve to 1 even though can_vectorize_up_to returns 4",
    )
    record(
        "numel_1025_float32",
        _view_1d(1025, mod32=0),
        1,
        "odd numel -> halve to 1 even though can_vectorize_up_to returns 4",
    )
    record(
        "numel_4097_float32",
        _view_1d(4097, mod32=0),
        1,
        "odd numel -> halve to 1 even though can_vectorize_up_to returns 4",
    )
    record(
        "float64_dense_64",
        _view_1d(64, dtype=torch.float64, mod32=0),
        4,
        "can_vectorize_up_to: vec 4 needs 4 * sizeof(float64) == 32 B alignment "
        "(UNROLL == 4 is the only cap, there is no 16-byte cap)",
    )
    record(
        "float64_dense_64_mod32_16",
        _view_1d(64, dtype=torch.float64, mod32=16),
        2,
        "can_vectorize_up_to: 16 B is not 32 B, so vec 4 is unavailable but "
        "2 * sizeof(float64) == 16 B alignment selects vec 2",
    )
    record(
        "float64_dense_64_mod32_8",
        _view_1d(64, dtype=torch.float64, mod32=8),
        1,
        "can_vectorize_up_to: 8 B is neither 32 B nor 16 B aligned -> vec 1",
    )
    record(
        "float64_odd_21",
        _view_1d(21, dtype=torch.float64, mod32=0),
        1,
        "float64 vec 4 is alignment-eligible but odd numel halves to 1",
    )

    # ``is_non_overlapping_and_dense`` is about the memory layout, not about
    # being contiguous: a transposed view is still dense (vec 4 when the
    # pointer and the element count allow it), while a strided slice or an
    # expanded (stride 0) view is not (vec 1).
    record(
        "transposed_dense_16x16",
        _contiguous((16, 16), mod32=0).t(),
        4,
        "dense transposed view keeps the can_vectorize_up_to alignment rule",
    )
    record(
        "transposed_dense_3x5",
        _contiguous((5, 3), mod32=0).t(),
        1,
        "dense transposed view with numel 15 -> odd numel -> 1",
    )
    record(
        "strided_slice_step2",
        _contiguous((8, 8), mod32=0)[:, ::2],
        1,
        "not non-overlapping-and-dense -> vec 1 (no can_vectorize_up_to query)",
    )
    record(
        "expanded_stride_zero",
        _contiguous((1, 3), mod32=0).expand(4, 3),
        1,
        "overlapping (stride 0) is not non-overlapping-and-dense -> vec 1",
    )
    record(
        "zero_dim",
        torch.tensor(2.5),
        1,
        "one element -> numel % 2 != 0 -> 1",
    )

    empty = torch.empty(0, 5, dtype=torch.float32)
    empty_pointer = int(empty.data_ptr())
    empty_expected = 4 if empty_pointer % 16 == 0 else 2
    if empty_pointer % 8 != 0:
        empty_expected = 1
    record(
        "empty_0x5_float32",
        empty,
        empty_expected,
        "numel 0 is divisible by every width; the measured pointer alignment to "
        "vec * sizeof(float32) decides (advance 0)",
    )
    assert antibody_rng.dropout_offset_advance(0) == 0

    # Per-width alignment: an 8-byte-but-not-16-byte-aligned float32 buffer is
    # still eligible for vec 2 (it only fails the vec 4 requirement).
    record(
        "aligned_8_bytes_even_numel_float32",
        _view_1d(64, mod32=8),
        2,
        "can_vectorize_up_to: vec 4 needs 16 B, vec 2 needs 8 B alignment only",
    )
    record(
        "aligned_8_bytes_odd_numel_float32",
        _view_1d(63, mod32=8),
        1,
        "8-byte aligned but odd numel -> 1",
    )

    # The four independent H100 layout recordings use exactly these dtypes,
    # element counts and pointer alignments.
    record(
        "holdout_0_float32_600002_mod32_0",
        _view_1d(600002, mod32=0),
        2,
        "H100 case 0.npz: vec 4 alignment holds but numel % 4 != 0 -> 2",
    )
    record(
        "holdout_1_float64_600000_mod32_0",
        _view_1d(600000, dtype=torch.float64, mod32=0),
        4,
        "H100 case 1.npz: 32 B alignment + numel % 4 == 0 -> vec 4 (no 16 B cap)",
    )
    record(
        "holdout_2_float64_600002_mod32_0",
        _view_1d(600002, dtype=torch.float64, mod32=0),
        2,
        "H100 case 2.npz: vec 4 alignment holds but numel % 4 != 0 -> 2",
    )
    record(
        "holdout_3_float32_600000_mod32_8",
        _view_1d(600000, mod32=8),
        2,
        "H100 case 3.npz: 8 B but not 16 B aligned -> vec 2",
    )

    # An explicit width that the shape cannot satisfy is rejected, not quietly
    # downgraded (the source only reaches a VEC kernel when numel % VEC == 0).
    with pytest.raises(ValueError):
        antibody_rng.bernoulli_mask(7, 0.9, vec=4)
    with pytest.raises(ValueError):
        antibody_rng.bernoulli_mask(3, 0.9, vec=2)
    with pytest.raises(ValueError):
        antibody_rng.bernoulli_mask(8, 0.9, vec=3)

    _EVIDENCE["vector_size_dispatch"] = {**cases, "verdict": "passed"}


# --------------------------------------------------------------------------- #
# 4a. Independent H100 long/misaligned layout recordings
# --------------------------------------------------------------------------- #
def test_cuda_dropout_layout_holdouts():
    """The four independent H100 long/misaligned mask recordings replay exactly.

    Each case controls ``data_ptr % 32`` to the alignment the H100 recording
    observed, so the vector width the pinned source selects (vec 2 / vec 4 / vec
    2 / vec 2) is exercised on a long contiguous buffer.  The expected mask bits
    and the expected offset come from the H100 recording, never from the
    candidate.
    """
    assert _LAYOUT_DIR.is_dir(), (
        "layout holdout asset directory is missing: {}".format(_LAYOUT_DIR)
    )
    manifest_path = _LAYOUT_DIR / "manifest.json"
    provenance_path = _LAYOUT_DIR / "provenance.json"
    assert manifest_path.is_file(), "layout holdout manifest is missing"
    assert provenance_path.is_file(), "layout holdout provenance is missing"

    manifest_sha = _sha256(manifest_path)
    assert manifest_sha == LAYOUT_MANIFEST_SHA256, (
        "layout holdout manifest changed: {} != {}".format(
            manifest_sha, LAYOUT_MANIFEST_SHA256
        )
    )
    manifest = _load_json(manifest_path, "H100 dropout layout manifest")
    provenance = _load_json(provenance_path, "H100 dropout layout provenance")

    # Provenance: the assets must document the independent source they come
    # from, with the two pinned primary-source files and their hashes.
    assert "independent" in provenance["scope"]
    assert provenance["torch"] == "2.5.1+cu118"
    assert "H100" in provenance["device"]
    assert provenance["manifest_sha256"] == LAYOUT_MANIFEST_SHA256
    assert provenance["collector"] == (
        "reference/dropout-layout-study/collect_dropout_layout_holdouts.py"
    )
    assert "dropout" in provenance["statement"]
    assert "test/rng_parity/test_cuda_compat_dropout.py" in provenance["statement"]
    for source_path, source_sha in LAYOUT_SOURCE_SHA256.items():
        entries = [
            entry
            for entry in provenance["source_files"]
            if entry["path"] == source_path
        ]
        assert len(entries) == 1, (
            "provenance must document {} exactly once".format(source_path)
        )
        entry = entries[0]
        assert entry["sha256"] == source_sha, (
            "provenance records {} for {}".format(entry["sha256"], source_path)
        )
        assert entry["url"] == (
            "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/" + source_path
        )

    manifest_cases = {case["file"]: case for case in manifest["cases"]}
    provenance_cases = {case["file"]: case for case in provenance["cases"]}
    declared_files = {case["file"] for case in LAYOUT_HOLDOUTS}
    assert set(manifest_cases) == declared_files
    assert set(provenance_cases) == declared_files

    device = _accel_device()
    rows = []
    for case in LAYOUT_HOLDOUTS:
        numel = int(case["numel"])
        manifest_case = manifest_cases[case["file"]]
        provenance_case = provenance_cases[case["file"]]
        assert str(case["dtype"]) == manifest_case["dtype"]
        assert numel == int(manifest_case["numel"])
        assert int(case["mod32"]) == int(manifest_case["ptr_mod_32"])
        assert int(case["vec"]) == int(
            manifest_case["expected_vec_from_primary_source"]
        )
        assert int(manifest_case["offset_before"]) == 0
        assert int(case["offset_after"]) == int(manifest_case["offset_after"])
        assert case["npz_sha256"] == manifest_case["sha256"]
        assert case["mask_sha256"] == manifest_case["mask_sha256"]
        assert case["npz_sha256"] == provenance_case["npz_sha256"]
        assert case["mask_sha256"] == provenance_case["mask_sha256"]

        path = _LAYOUT_DIR / case["file"]
        npz_sha = _sha256(path)
        assert npz_sha == case["npz_sha256"], (
            "{} bytes differ from the H100 recording".format(case["file"])
        )
        asset = np.load(path)
        keep = asset["keep"]
        assert keep.shape == (numel,)
        assert keep.dtype == np.uint8
        keep_sha = _sha256_bytes(keep.tobytes())
        assert keep_sha == case["mask_sha256"], (
            "{} recorded keep mask changed".format(case["file"])
        )
        assert _state_offset(asset["before"]) == 0
        assert _state_offset(asset["after"]) == int(case["offset_after"])

        tensor = _view_1d(
            numel,
            dtype=case["dtype"],
            mod32=int(case["mod32"]),
            device=device,
            filled=True,
        )
        pointer = int(tensor.data_ptr())
        element_size = int(tensor.element_size())
        assert pointer % 32 == int(case["mod32"])
        vec = int(antibody_rng.dropout_vector_size(tensor))
        primary = _primary_source_vector_size(pointer, element_size, numel)
        assert vec == int(case["vec"]), (
            "{}: product selects vec {} but the pinned source selects {}".format(
                case["file"], vec, case["vec"]
            )
        )
        assert primary == vec, (
            "{}: independent primary-source composition selects vec {}, the "
            "product selects {}".format(case["file"], primary, vec)
        )

        generator = antibody_rng.CudaCompatGenerator(seed=17, offset=0)
        flags = generator.bernoulli_mask(
            numel, 1.0 - DROPOUT_P, dtype=case["dtype"], vec=vec
        )
        mask_sha = _sha256_bytes(flags.tobytes())
        assert mask_sha == case["mask_sha256"], (
            "{}: mask bits differ from the H100 recording ({} != {})".format(
                case["file"], mask_sha, case["mask_sha256"]
            )
        )
        assert int(generator.offset) == int(case["offset_after"])

        if device is not None:
            antibody_rng.manual_seed(17)
            if case["dtype"] is torch.float32:
                result = antibody_rng.dropout(tensor, DROPOUT_P)
                seed_after, offset_after = antibody_rng.get_seed_offset(device)
                assert result.device.type == "npu"
                assert result.dtype is case["dtype"]
                assert result.shape == tensor.shape
                device_keep = (
                    (result != 0).to(torch.uint8).cpu().contiguous().numpy()
                )
                device_mask_sha = _sha256_bytes(device_keep.tobytes())
                assert device_mask_sha == case["mask_sha256"], (
                    "{}: the device mask differs from the H100 recording".format(
                        case["file"]
                    )
                )
                assert int(seed_after) == 17
                assert int(offset_after) == int(case["offset_after"])
                route = (
                    "product API on {} (mask generated on host, multiply on "
                    "device)".format(device)
                )
                dtype_contract = "exact float32 device product"
            else:
                # This stack has no truthful float64 device product
                # (``torch_npu`` maps a requested Double cast/elementwise
                # target to Float), so the product is documented unsupported
                # and its rejection is executed rather than a float32 result
                # being passed off as float64.  The generator/mask/offset
                # expectations above stay exactly as recorded.
                with pytest.raises(NotImplementedError):
                    antibody_rng.dropout(tensor, DROPOUT_P)
                with pytest.raises(NotImplementedError):
                    antibody_rng.dropout_mask_like(tensor, 1.0 - DROPOUT_P)
                seed_after, offset_after = antibody_rng.get_seed_offset(device)
                assert int(seed_after) == 17
                assert int(offset_after) == 0, (
                    "{}: the float64 rejection must happen before the stream "
                    "is consumed".format(case["file"])
                )
                device_mask_sha = None
                route = (
                    "documented unsupported float64 device product "
                    "(NotImplementedError) on {}".format(device)
                )
                dtype_contract = "unsupported float64 device product"
                result = None
        else:
            device_mask_sha = None
            dtype_contract = (
                "exact float32 device product"
                if case["dtype"] is torch.float32
                else "unsupported float64 device product"
            )
            route = (
                "no accelerator on this host; the identical mask/counter "
                "expectations are evaluated with the generator and are not "
                "device evidence"
            )

        rows.append(
            {
                "asset": (
                    "test/rng_parity/reference_outputs/"
                    "cuda_dropout_layout_holdouts/" + case["file"]
                ),
                "dtype": str(case["dtype"]),
                "numel": numel,
                "data_ptr_mod_32": pointer % 32,
                "vec": vec,
                "primary_source_vec": primary,
                "mask_sha256_expected": case["mask_sha256"],
                "mask_sha256_actual": mask_sha,
                "device_mask_sha256": device_mask_sha,
                "dtype_contract": dtype_contract,
                "offset_before": 0,
                "offset_after": int(generator.offset),
                "route": route,
                "verdict": (
                    "passed" if mask_sha == case["mask_sha256"] else "failed"
                ),
            }
        )
        asset.close()
        del keep, tensor
        if device is not None:
            del result

    _EVIDENCE["layout_holdouts"] = {
        "reference": (
            "test/rng_parity/reference_outputs/cuda_dropout_layout_holdouts"
        ),
        "manifest_sha256": manifest_sha,
        "scope": provenance["scope"],
        "torch": provenance["torch"],
        "device": provenance["device"],
        "source_files": provenance["source_files"],
        "cases": rows,
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 4b. Truthful device dtype contract
# --------------------------------------------------------------------------- #
def test_device_dtype_contract():
    """The documented device dtype contract is executed, not just declared.

    ``torch_npu`` maps a requested ``Double`` cast/elementwise target to
    ``Float`` for device kernels (``ToKernelNpu.cpp`` v2.5.1 lines 149-155), so
    an NPU float64 dropout product cannot be truthful.  The contract therefore
    documents the float64 *generator* (mask bits, transforms, Philox offsets) as
    exact, documents the float64 *device product* as unsupported, and the
    rejection is executed here instead of widening a float32 product.
    Everything outside float32/float64 is rejected the same way.
    """
    documented = {}
    for dtype in (torch.float32, torch.float64):
        text = antibody_rng.device_dtype_contract(dtype)
        assert isinstance(text, str) and text.strip(), (
            "{} needs a documented device contract".format(dtype)
        )
        documented[str(dtype)] = text
    assert "unsupported" in documented[str(torch.float64)]

    rejected = []
    for dtype in (torch.float16, torch.bfloat16, torch.int64):
        with pytest.raises(NotImplementedError):
            antibody_rng.device_dtype_contract(dtype)
        rejected.append(str(dtype))

    p_keep = 0.5
    device = _accel_device()
    if device is None:
        _EVIDENCE["device_dtype_contract"] = {
            "documented_contract": documented,
            "dtype_rejections": rejected,
            "device": None,
            "float64_device_product_supported": False,
            "native_double_arithmetic": None,
            "note": (
                "no accelerator on this host; the dtype contract and its "
                "rejections are asserted, the device arithmetic is not "
                "measured and is not claimed"
            ),
            "verdict": "passed",
        }
        return

    with pytest.raises(NotImplementedError):
        antibody_rng.dropout(
            torch.zeros(8, dtype=torch.float16, device=device), DROPOUT_P
        )

    # Diagnostics only (never gated): what the device actually does with a
    # requested float64 result, recorded so the documented limitation stays
    # evidence-backed.
    double_input = torch.full(
        (16,), 1.0 + 2.0**-40, dtype=torch.float64, device=device
    )
    direct = double_input * torch.ones_like(double_input)
    widened = (
        double_input.float() * torch.ones_like(double_input).float()
    ).to(torch.float64)
    native_double_arithmetic = bool(torch.equal(direct, double_input))
    direct_dtype = str(direct.dtype)

    dtype_rejections = {}
    for dtype in (torch.float64, torch.float16, torch.bfloat16):
        probe = torch.ones(8, dtype=dtype, device=device)
        with pytest.raises(NotImplementedError):
            antibody_rng.dropout(probe, DROPOUT_P)
        with pytest.raises(NotImplementedError):
            antibody_rng.dropout_mask_like(probe, 1.0 - DROPOUT_P)
        dtype_rejections[str(dtype)] = "NotImplementedError"

    # The float32 device product is exact and its stream is the documented one.
    source = torch.arange(1, 33, dtype=torch.float32, device=device).reshape(4, 8)
    antibody_rng.manual_seed(101)
    mask = antibody_rng.dropout_mask_like(source, p_keep)
    assert antibody_rng.get_seed_offset(device) == (101, 4)
    antibody_rng.manual_seed(101)
    result = antibody_rng.dropout(source, 1.0 - p_keep)
    assert antibody_rng.get_seed_offset(device) == (101, 4)

    scale = float(np.float32(1.0) / np.float32(p_keep))
    assert result.dtype is torch.float32
    assert result.device.type == "npu"
    assert result.shape == source.shape
    assert bool(torch.isfinite(result).all())
    assert torch.equal(result, source * mask), (
        "the float32 device product must be the exact src * mask product"
    )
    assert torch.equal(result != 0, mask != 0)
    assert set(torch.unique(mask).tolist()).issubset({0.0, scale})
    assert float(scale) == 2.0

    _EVIDENCE["device_dtype_contract"] = {
        "documented_contract": documented,
        "dtype_rejections": rejected,
        "device": str(device),
        "device_name": str(torch.npu.get_device_name(0)),
        "float64_device_product_supported": False,
        "float64_rejection": "NotImplementedError",
        "dtype_rejections_executed": dtype_rejections,
        "float32_product_exact": True,
        "float32_mask_scale": scale,
        "float32_stream_after": [101, 4],
        "native_double_arithmetic": native_double_arithmetic,
        "measurement": {
            "float64_requested_product_dtype": direct_dtype,
            "float64_requested_equals_float32_product": bool(
                torch.equal(direct, widened)
            ),
        },
        "dtype_preserved": "torch.float32 (float64 device product unsupported)",
        "note": (
            "native_double_arithmetic and the float64 measurement are recorded "
            "diagnostics, not gates: the vendor op-plugin maps a requested "
            "Double elementwise target to Float, so no truthful float64 device "
            "product exists and the product API rejects float64 instead of "
            "widening a float32 result"
        ),
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 4c. Every strided layout gets an input-shaped mask in the kernel index order
# --------------------------------------------------------------------------- #
def test_dropout_layouts_and_mask_shape():
    """Dense/strided/expanded/0-dim/empty layouts keep shape and index order."""
    device = _accel_device()
    target = device if device is not None else torch.device("cpu")
    p_keep = 0.9
    p_drop = 1.0 - p_keep
    scale = antibody_rng._dropout_scale(p_keep, torch.float32)
    seed = 771
    recorded = {}

    for name, tensor, dense_expected, contract in _layout_cases(target):
        numel = int(tensor.numel())
        vec = int(antibody_rng.dropout_vector_size(tensor))
        assert bool(antibody_rng._is_non_overlapping_and_dense(tensor)) is (
            dense_expected
        ), "{}: layout predicate disagrees with the expected class".format(name)

        expected_mask, expected_advance, order = _expected_mask(
            tensor, p_keep, vec, seed, 0
        )

        if device is not None:
            antibody_rng.manual_seed(seed)
            before = antibody_rng.get_seed_offset(device)
            result = antibody_rng.dropout(tensor, p_drop)
            after = antibody_rng.get_seed_offset(device)
            antibody_rng.manual_seed(seed)
            mask = antibody_rng.dropout_mask_like(tensor, p_keep)
            mask_after = antibody_rng.get_seed_offset(device)
            assert result.device.type == "npu", (
                "{}: result must stay on the NPU, got {}".format(
                    name, result.device
                )
            )
            assert mask.device.type == "npu"
            advance = int(after[1]) - int(before[1])
            mask_advance = int(mask_after[1])  # the mask draw started at offset 0
            route = "product API on {}".format(device)
        else:
            antibody_rng.manual_seed(seed)
            before = antibody_rng.get_seed_offset(target)
            mask = antibody_rng._device_dropout_mask(tensor, p_keep)
            after = antibody_rng.get_seed_offset(target)
            result = tensor * mask
            advance = int(after[1]) - int(before[1])
            mask_advance = advance
            delegated = antibody_rng.dropout(tensor, p_drop)
            assert delegated.shape == tensor.shape
            route = (
                "no accelerator on this host; generator + product mask helper "
                "on CPU tensors (identical expectations, never a skip)"
            )

        expected_step = 4 if numel else 0
        assert expected_advance == expected_step, (
            "{}: the generator consumed {} words, the pinned launch consumes "
            "{}".format(name, expected_advance, expected_step)
        )
        assert advance == expected_step, (
            "{}: the product draw consumed {} words instead of {}".format(
                name, advance, expected_step
            )
        )
        assert mask_advance == expected_step

        assert result.shape == tensor.shape, (
            "{}: dropout returned shape {} for input shape {}".format(
                name, tuple(result.shape), tuple(tensor.shape)
            )
        )
        assert mask.shape == tensor.shape
        assert bool(torch.isfinite(result).all())
        assert torch.equal(mask, expected_mask), (
            "{}: mask does not follow the {} contract".format(name, order)
        )

        kept = result != 0
        recovered = kept.to(tensor.dtype) * scale
        assert torch.equal(result, tensor * recovered), (
            "{}: the dropout result is not the exact ``src * (kept * scale)`` "
            "product".format(name)
        )
        assert set(torch.unique(recovered.reshape(-1)).tolist()).issubset(
            {0.0, scale}
        )
        assert torch.equal(result, tensor * expected_mask)

        recorded[name] = {
            "shape": list(tensor.shape),
            "strides": list(tensor.stride()),
            "numel": numel,
            "dtype": str(tensor.dtype),
            "non_overlapping_and_dense": dense_expected,
            "vec": vec,
            "mask_order": order,
            "contract": contract,
            "offset_words": expected_step,
            "mask_scale": scale,
            "route": route,
            "shape_preserved": True,
            "exact_product": True,
            "verdict": "passed",
        }

    _EVIDENCE["layout_mask_shape"] = {
        "p_keep": p_keep,
        "seed": seed,
        "layouts": recorded,
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 4d. The layout predicate equals the ATen / torch._prims_common one
# --------------------------------------------------------------------------- #
def test_layout_predicate_matches_prims():
    helper = _prims_dense_predicate()
    rows = {}
    for name, tensor, dense_expected, contract in _layout_cases():
        actual = bool(antibody_rng._is_non_overlapping_and_dense(tensor))
        assert actual is dense_expected, (
            "{}: predicate returned {} but the layout is {}".format(
                name, actual, "dense" if dense_expected else "not dense"
            )
        )
        if helper is not None:
            expected = bool(helper(tensor))
            assert actual is expected, (
                "{}: predicate {} differs from "
                "torch._prims_common.is_non_overlapping_and_dense {}".format(
                    name, actual, expected
                )
            )
        rows[name] = {
            "shape": list(tensor.shape),
            "strides": list(tensor.stride()),
            "non_overlapping_and_dense": actual,
            "torch_prims_available": helper is not None,
            "contract": contract,
        }

    _EVIDENCE["layout_predicate"] = {
        "layouts": rows,
        "torch_prims_helper_available": helper is not None,
        "comparison": (
            "equal to torch._prims_common.is_non_overlapping_and_dense on every "
            "layout"
            if helper is not None
            else "hand-derived expectation on every layout (helper unavailable)"
        ),
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 4e. Exponential near-one guard (pinned TransformationHelper.h)
# --------------------------------------------------------------------------- #
_NEAR_ONE_TOLERANCES = {
    ("normal", "torch.float32"): (1e-5, 1e-6),
    ("exponential", "torch.float32"): (1e-5, 1e-6),
    ("exponential", "torch.float64"): (1e-15, 1e-15),
}


def test_exponential_near_one_guard():
    """The pinned ``val >= 1 - eps/2`` guard, including the 84 holdouts."""
    # (a) reachability: the largest float32 word maps to exactly 1.0.
    top = np.array([[0xFFFFFFFF] * 4], dtype=np.uint32)
    uniform_top = antibody_rng._uniform_float32(top[:, 0])
    assert uniform_top.dtype == np.float32
    assert np.all(uniform_top == np.float32(1.0)), (
        "the largest curand word no longer rounds to exactly 1.0: {}".format(
            uniform_top
        )
    )

    # (b) the guard substitutes eps/2 for that word (float32 accumulator).
    guard32 = antibody_rng._transform("exponential", torch.float32, top)
    expected32 = np.float32(np.finfo(np.float32).eps / 2)
    assert guard32.dtype == np.float32
    assert np.all(guard32 == expected32), (
        "float32 near-one guard must return {}, got {}".format(
            expected32, guard32
        )
    )

    # (c) float64 accumulates in double: the maximum 53-bit combination
    # ``low = 0x001FFFFF, high = 0xFFFFFFFF`` gives ``combined = 2**53 - 1``,
    # which maps to ``1 - 2**-54`` and rounds to exactly 1.0, so the guard fires.
    bits64 = np.array(
        [[0x001FFFFF, 0xFFFFFFFF, 0x001FFFFF, 0xFFFFFFFF]], dtype=np.uint32
    )
    uniform64 = antibody_rng._uniform2_double(bits64)
    assert np.all(uniform64 == 1.0), (
        "the maximum float64 combination must round to 1.0, got {}".format(
            uniform64
        )
    )
    guard64 = antibody_rng._transform("exponential", torch.float64, bits64)
    expected64 = np.float64(np.finfo(np.float64).eps / 2)
    assert guard64.dtype == np.float64
    assert np.all(guard64 == expected64), (
        "float64 near-one guard must return {}, got {}".format(
            expected64, guard64
        )
    )

    # (d) control words keep the ordinary ``-log(u)`` value: the guard is a
    # near-one rule, not a global clamp.
    control = np.array([[0xFFFF0000] * 4], dtype=np.uint32)
    control_uniform = antibody_rng._uniform_float32(control[:, 0])
    control_expected = -np.log(control_uniform)
    control_actual = antibody_rng._transform("exponential", torch.float32, control)
    assert np.asarray(control_actual).shape == (1, 4), (
        "the transform must preserve the word-array shape (1, 4); got {}".format(
            np.asarray(control_actual).shape
        )
    )
    assert np.all(control_expected > 0.0)
    assert np.all(control_actual == control_expected), (
        "a word below 1 - eps/2 must keep its ordinary -log(u) value: {} vs "
        "{}".format(control_actual, control_expected)
    )

    far = np.array([[0x7FFFFFFF] * 4], dtype=np.uint32)
    far_uniform = antibody_rng._uniform_float32(far[:, 0])
    far_actual = antibody_rng._transform("exponential", torch.float32, far)
    assert np.asarray(far_actual).shape == (1, 4), (
        "the transform must preserve the word-array shape (1, 4); got {}".format(
            np.asarray(far_actual).shape
        )
    )
    assert np.all(far_actual == -np.log(far_uniform))
    assert np.all(far_actual > 0.01), (
        "the guard must not clamp ordinary uniforms (u ~ 0.5): {}".format(
            far_actual
        )
    )

    # (e) the 84 accepted H100 holdouts still pass unchanged (same pre-declared
    # tolerances and exact offsets as test_cuda_compat_rng.py).
    holdout_dir = _MODULE_DIR / "reference_outputs" / "cuda_philox_holdouts"
    manifest = _load_json(
        holdout_dir / "manifest.json", "CUDA Philox holdout manifest"
    )
    rows = manifest["rows"]
    assert len(rows) == 84
    replayed = 0
    worst = {}
    for row in rows:
        asset = np.load(holdout_dir / row["file"])
        dtype = _torch_dtype(row["dtype"])
        np_dtype = np.float32 if dtype is torch.float32 else np.float64
        reference = asset["value"].astype(np_dtype)
        offset_before = _state_offset(asset["state_before"])
        offset_after = _state_offset(asset["state_after"])
        assert _state_seed(asset["state_before"]) == row["seed"]
        generator = antibody_rng.CudaCompatGenerator(
            seed=int(row["seed"]), offset=offset_before
        )
        shape = tuple(int(dim) for dim in reference.shape)
        if row["kind"] == "normal":
            candidate = generator.normal(shape, dtype=dtype).numpy()
        else:
            candidate = generator.exponential(shape, dtype=dtype).numpy()
        atol, rtol = _NEAR_ONE_TOLERANCES[(row["kind"], row["dtype"])]
        difference = np.abs(candidate - reference)
        relative = difference / np.maximum(np.abs(reference), np.finfo(np_dtype).eps)
        max_abs = float(difference.max()) if difference.size else 0.0
        max_rel = float(relative.max()) if relative.size else 0.0
        assert bool(np.isfinite(candidate).all()), row["file"]
        assert np.allclose(
            candidate, reference, atol=atol, rtol=rtol, equal_nan=False
        ), (
            "{} outside the pre-declared tolerance atol={} rtol={}: max_abs={} "
            "max_rel={}".format(row["file"], atol, rtol, max_abs, max_rel)
        )
        assert int(generator.offset) == offset_after
        key = "{} {}".format(row["kind"], row["dtype"])
        entry = worst.setdefault(key, {"max_abs_error": 0.0, "max_rel_error": 0.0})
        entry["max_abs_error"] = max(entry["max_abs_error"], max_abs)
        entry["max_rel_error"] = max(entry["max_rel_error"], max_rel)
        replayed += 1

    _EVIDENCE["exponential_guard"] = {
        "pinned_source": (
            "aten/src/ATen/native/cuda/TransformationHelper.h lines 126-139 "
            "(torch v2.3.1): log = val >= 1 - eps/2 ? -eps/2 : at::log(val)"
        ),
        "float32_max_word_uniform": float(uniform_top[0]),
        "float32_guard_value": float(expected32),
        "float64_max_combination_uniform": float(uniform64.max()),
        "float64_guard_value": float(expected64),
        "control_word": "0xFFFF0000 keeps the ordinary -log(u) value",
        "control_value": float(np.asarray(control_actual).reshape(-1)[0]),
        "far_control_word": "0x7FFFFFFF keeps a value above 0.01",
        "far_control_value": float(np.asarray(far_actual).reshape(-1)[0]),
        "holdouts_replayed": replayed,
        "holdout_tolerances": {
            "{} {}".format(kind, dtype_name): {"atol": atol, "rtol": rtol}
            for (kind, dtype_name), (atol, rtol) in _NEAR_ONE_TOLERANCES.items()
        },
        "holdout_worst_errors": worst,
        "note": (
            "analytic expectations derived from the pinned source; the holdouts "
            "are the same accepted H100 recordings test_cuda_compat_rng.py uses"
        ),
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 5. Source short-cuts never consume the stream
# --------------------------------------------------------------------------- #
def test_source_shortcuts():
    device = _accel_device()
    seed = 20240617
    p_drop = 0.25
    evidence = {"p": p_drop, "seed": seed}

    if device is not None:
        antibody_rng.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
        before = antibody_rng.get_seed_offset(device)
        source = torch.arange(1, 65, dtype=torch.float32, device=device).reshape(8, 8)

        no_dropout = antibody_rng.dropout(source, 0.0)
        eval_mode = antibody_rng.dropout(source, p_drop, train=False)
        emptied = antibody_rng.dropout(
            torch.empty(0, device=device), p_drop, train=True
        )
        dropped_all = antibody_rng.dropout(source, 1.0)
        after = antibody_rng.get_seed_offset(device)

        assert no_dropout is source
        assert eval_mode is source
        assert emptied.numel() == 0
        assert torch.equal(dropped_all, torch.zeros_like(source))
        assert before == after, (
            "p=0/p=1/train=False/empty must not consume the compatibility stream "
            "({} -> {})".format(before, after)
        )

        inplace_source = source.clone()
        returned = antibody_rng.dropout(inplace_source, p_drop, inplace=True)
        assert returned is inplace_source
        assert antibody_rng.get_seed_offset(device)[1] > after[1]
        evidence.update(
            route="product API on {}".format(device),
            stream_before=list(before),
            stream_after=list(after),
            zeros_for_p1=bool(torch.equal(dropped_all, torch.zeros_like(source))),
            inplace_advanced_stream=True,
        )
    else:
        # No accelerator: the same source short-cuts hold for the native
        # delegation, and the compatibility mask primitive still refuses to
        # consume anything for an empty draw.
        source = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8)
        assert antibody_rng.dropout(source, 0.0) is source
        assert antibody_rng.dropout(source, p_drop, train=False) is source
        assert antibody_rng.dropout(torch.empty(0), p_drop).numel() == 0
        assert torch.equal(
            antibody_rng.dropout(source, 1.0), torch.zeros_like(source)
        )
        generator = antibody_rng.CudaCompatGenerator(seed=seed, offset=8)
        assert generator.bernoulli_mask(0, 0.75).size == 0
        assert generator.offset == 8
        evidence.update(
            route="no accelerator on this host; native delegation and generator",
            stream_before=[seed, 8],
            stream_after=[seed, 8],
            zeros_for_p1=True,
            inplace_advanced_stream=False,
        )

    _EVIDENCE["source_shortcuts"] = {**evidence, "verdict": "passed"}


# --------------------------------------------------------------------------- #
# 6. Device route: the mask multiply executes on the NPU
# --------------------------------------------------------------------------- #
def test_device_route_and_mask_multiply():
    device = _accel_device()
    p_drop = DROPOUT_P
    shape = (1, 1, 478, 8, 1)
    numel = _numel(shape)
    p_keep = _keep_probability(p_drop)
    scale = float(np.float32(1.0) / np.float32(p_keep))

    if device is None:
        # No accelerator: evaluate the identical mask expectations with the
        # compatibility generator so the contract is still checked (this is
        # explicitly *not* presented as NPU kernel evidence).
        generator = antibody_rng.CudaCompatGenerator(seed=42, offset=0)
        flags = generator.bernoulli_mask(numel, p_keep, dtype=torch.float32, vec=4)
        mask = torch.from_numpy(flags).to(torch.float32) * scale
        source = torch.arange(1, numel + 1, dtype=torch.float32).reshape(shape)
        result = source * mask
        kept = mask != 0
        assert generator.offset == 4
        assert torch.equal(result, source * scale * kept.to(torch.float32))
        _EVIDENCE["device_route"] = {
            "route": "no accelerator on this host; compatibility generator",
            "kept_fraction": float(kept.to(torch.float32).mean()),
            "offset_after": generator.offset,
            "verdict": "passed",
        }
        return

    antibody_rng.manual_seed(42)
    source = torch.arange(1, numel + 1, dtype=torch.float32, device=device).reshape(
        shape
    )
    before = antibody_rng.get_seed_offset(device)
    result = antibody_rng.dropout(source, p_drop)
    after = antibody_rng.get_seed_offset(device)

    assert result.device.type == "npu"
    assert result.shape == source.shape
    assert bool(torch.isfinite(result).all())
    assert before == (42, 0)
    assert after == (42, 4), "one fused_dropout launch must consume 4 words"

    kept = result != 0
    kept_fraction = float(kept.to(torch.float32).mean().item())
    assert 0.85 <= kept_fraction <= 0.95, (
        "kept fraction {} is inconsistent with p={}".format(kept_fraction, p_drop)
    )
    expected = source * scale * kept.to(torch.float32)
    assert torch.equal(result, expected), (
        "the mask multiply must be the exact accscalar `src * scale` product"
    )
    dropped = result[~kept]
    assert dropped.numel() == 0 or bool(torch.equal(dropped, torch.zeros_like(dropped)))

    # Reproducibility of the mask for a reset stream.
    antibody_rng.manual_seed(42)
    repeated = antibody_rng.dropout(source, p_drop)
    assert torch.equal(result, repeated)

    module = antibody_rng.CudaCompatDropout(p_drop)
    antibody_rng.manual_seed(42)
    module_result = module(source)
    assert torch.equal(module_result, result), (
        "CudaCompatDropout must use the same stream and mask as the functional API"
    )
    assert bool(module.training) is True
    module.eval()
    assert module(source) is source, "eval mode must be the identity"

    mask = antibody_rng.dropout_mask_like(source, p_keep)
    assert mask.device.type == "npu"
    assert mask.shape == source.shape
    assert set(torch.unique(mask).tolist()).issubset({0.0, scale})
    assert antibody_rng.get_seed_offset(device) == (42, 8), (
        "a fresh mask draw must consume its own 4 words"
    )

    _EVIDENCE["device_route"] = {
        "route": "product API on {}".format(device),
        "device_name": str(torch.npu.get_device_name(0)),
        "shape": list(shape),
        "dtype": str(result.dtype),
        "stream_before": list(before),
        "stream_after": list(after),
        "kept_fraction": kept_fraction,
        "mask_scale": scale,
        "exact_mask_multiply": True,
        "module_matches_functional": True,
        "eval_mode_identity": True,
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 7. CPU/CUDA native generator delegation
# --------------------------------------------------------------------------- #
def test_native_generator_non_regression():
    seed = 20240618
    p_drop = 0.3
    source = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8)

    torch.manual_seed(seed)
    state_before = torch.get_rng_state()
    torch_result = F.dropout(source, p_drop, True)
    state_after_torch = torch.get_rng_state()

    torch.manual_seed(seed)
    state_before_compat = torch.get_rng_state()
    compat_result = antibody_rng.dropout(source, p_drop, True)
    state_after_compat = torch.get_rng_state()

    assert torch.equal(state_before, state_before_compat)
    assert torch.equal(torch_result, compat_result), (
        "the CPU target must delegate verbatim to torch.nn.functional.dropout"
    )
    assert torch.equal(state_after_torch, state_after_compat), (
        "the CPU target must consume the native CPU generator identically"
    )

    torch.manual_seed(seed)
    F.dropout(source, p_drop, True)  # the native CPU draw consumes the stream
    torch_next = torch.randn(7)
    torch.manual_seed(seed)
    antibody_rng.dropout(source, p_drop, True)
    compat_next = torch.randn(7)
    assert torch.equal(torch_next, compat_next)

    module = antibody_rng.CudaCompatDropout(p_drop)
    torch.manual_seed(seed)
    nn_module = torch.nn.Dropout(p_drop)
    assert list(module.state_dict()) == list(nn_module.state_dict()) == []
    torch.manual_seed(seed)
    module_result = module(source)
    torch.manual_seed(seed)
    nn_result = nn_module(source)
    assert torch.equal(module_result, nn_result), (
        "on CPU CudaCompatDropout must be nn.Dropout verbatim"
    )

    cuda_evidence = {"available": False}
    if torch.cuda.is_available():  # pragma: no cover - CUDA host only
        cuda_source = source.cuda()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda_torch = F.dropout(cuda_source, p_drop, True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda_compat = antibody_rng.dropout(cuda_source, p_drop, True)
        assert torch.equal(cuda_torch, cuda_compat)
        cuda_evidence = {"available": True, "delegation_equal": True}
    else:
        cuda_evidence = {
            "available": False,
            "note": (
                "delegation is selected by device type, not by availability: "
                "every non-npu tensor takes the torch.nn.functional.dropout path "
                "asserted above for CPU"
            ),
        }

    _EVIDENCE["native_generator_non_regression"] = {
        "cpu_dropout_equal": True,
        "cpu_rng_state_consumption_equal": True,
        "cpu_next_draw_equal": True,
        "module_state_dict_empty": True,
        "module_equals_nn_dropout_on_cpu": True,
        "cuda": cuda_evidence,
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 8. The RF2 product path actually uses the compatibility module
# --------------------------------------------------------------------------- #
def test_rf2_product_wiring():
    from rfantibody.rf2.network.Attention_module import FeedForwardLayer, SequenceWeight

    feed_forward = FeedForwardLayer(64, 2)
    sequence_weight = SequenceWeight(64, 8, 8)
    assert isinstance(feed_forward.dropout, antibody_rng.CudaCompatDropout)
    assert isinstance(sequence_weight.dropout, antibody_rng.CudaCompatDropout)
    assert feed_forward.dropout.p == pytest.approx(0.1)
    assert sequence_weight.dropout.p == pytest.approx(0.1)
    assert FeedForwardLayer(64, 2, p_drop=0.25).dropout.p == pytest.approx(0.25)
    assert SequenceWeight(64, 8, 8, p_drop=0.15).dropout.p == pytest.approx(0.15)

    network_source = (
        _PROJECT_ROOT / "src" / "rfantibody" / "rf2" / "network" / "Attention_module.py"
    ).read_text()
    assert "nn.Dropout(" not in network_source, (
        "the RF2 network must not keep a native nn.Dropout construction site"
    )
    assert network_source.count("CudaCompatDropout(") == 2

    entrypoint_source = (
        _PROJECT_ROOT / "scripts" / "rf2_predict.py"
    ).read_text()
    assert "antibody_rng.manual_seed(seed)" in entrypoint_source, (
        "the deterministic RF2 entrypoint must seed the compatibility stream"
    )

    # The custom CPU Bernoulli head is a different (already bit-exact) stream
    # and must not have been folded into the compatibility module.
    head_source = (
        _PROJECT_ROOT / "src" / "rfantibody" / "rf2" / "network" / "util_module.py"
    ).read_text()
    assert "class Dropout" in head_source
    assert "Bernoulli" in head_source
    assert "CudaCompatDropout" not in head_source

    # The mask family must stay general: no example-, shape-, seed- or
    # case-specific token may appear in the RNG module.
    rng_source = (_PROJECT_ROOT / "src" / "rfantibody" / "rng.py").read_text()
    forbidden_tokens = (
        "478",
        "58491904",
        "RSV",
        "T305",
        "ab_prediction",
        "json_prediction",
    )
    absent_tokens = [
        token for token in forbidden_tokens if token not in rng_source
    ]
    assert absent_tokens == list(forbidden_tokens), (
        "src/rfantibody/rng.py contains case-specific token(s): {}".format(
            sorted(set(forbidden_tokens) - set(absent_tokens))
        )
    )

    _EVIDENCE["product_wiring"] = {
        "feed_forward_layer_module": type(feed_forward.dropout).__name__,
        "sequence_weight_module": type(sequence_weight.dropout).__name__,
        "network_native_dropout_sites": network_source.count("nn.Dropout("),
        "network_compat_dropout_sites": network_source.count("CudaCompatDropout("),
        "entrypoint_seeds_compat_stream": True,
        "cpu_bernoulli_head_untouched": True,
        "rng_module_forbidden_tokens_absent": absent_tokens,
        "cpu_head_sha256": _sha256(
            _PROJECT_ROOT
            / "src"
            / "rfantibody"
            / "rf2"
            / "network"
            / "util_module.py"
        ),
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 9. Vectorised mask generation cost on the largest product shape
# --------------------------------------------------------------------------- #
def test_mask_generation_throughput():
    generator = antibody_rng.CudaCompatGenerator(seed=0, offset=0)
    started = time.perf_counter()
    flags = generator.bernoulli_mask(THROUGHPUT_NUMEL, 0.9, dtype=torch.float32, vec=4)
    elapsed = time.perf_counter() - started

    assert flags.shape == (THROUGHPUT_NUMEL,)
    assert flags.dtype == np.uint8
    kept_fraction = float(flags.mean())
    assert 0.89 <= kept_fraction <= 0.91, (
        "kept fraction {} is inconsistent with keep=0.9".format(kept_fraction)
    )
    assert generator.offset == antibody_rng.dropout_offset_advance(THROUGHPUT_NUMEL)

    _EVIDENCE["throughput"] = {
        "numel": THROUGHPUT_NUMEL,
        "shape": [1, 478, 478, 256],
        "dtype": "torch.float32",
        "keep_probability": 0.9,
        "vector_size": 4,
        "wall_seconds": round(elapsed, 4),
        "elements_per_second": round(THROUGHPUT_NUMEL / max(elapsed, 1e-9), 1),
        "kept_fraction": kept_fraction,
        "offset_after": generator.offset,
        "note": (
            "measured on this host for the largest shape the RF2 dropout stream "
            "consumes; the generator is fully vectorised (one Philox call per "
            "launch iteration, no per-element Python loop)"
        ),
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 10. Documented contract errors
# --------------------------------------------------------------------------- #
def test_contract_errors():
    with pytest.raises(ValueError):
        antibody_rng.dropout(torch.zeros(4), -0.1)
    with pytest.raises(ValueError):
        antibody_rng.dropout(torch.zeros(4), 1.5)
    with pytest.raises(TypeError):
        antibody_rng.dropout([0.0, 1.0], 0.5)
    with pytest.raises(ValueError):
        antibody_rng.CudaCompatDropout(1.5)
    with pytest.raises(NotImplementedError):
        antibody_rng.dropout_mask_like(torch.zeros(8), 0.9)
    with pytest.raises(NotImplementedError):
        antibody_rng.bernoulli_mask(11, 0.9, dtype=torch.float16)
    with pytest.raises(ValueError):
        antibody_rng.bernoulli_mask(8, 1.5)
    with pytest.raises(ValueError):
        antibody_rng.dropout_offset_advance(-1)

    device = _accel_device()
    if device is not None:
        half = torch.zeros(8, dtype=torch.float16, device=device)
        with pytest.raises(NotImplementedError):
            antibody_rng.dropout(half, 0.1)
        # Strided layouts are supported (the pinned vec-1 kernel handles them),
        # so a non-contiguous tensor must produce an input-shaped result rather
        # than raise; the full layout matrix is checked by
        # ``test_dropout_layouts_and_mask_shape``.
        non_contiguous = torch.arange(
            1, 33, dtype=torch.float32, device=device
        ).view(8, 4).t()
        antibody_rng.manual_seed(9)
        strided = antibody_rng.dropout(non_contiguous, 0.1)
        assert strided.shape == non_contiguous.shape
        assert strided.device.type == "npu"
        assert bool(torch.isfinite(strided).all())

    _EVIDENCE["contract_errors"] = {
        "probability_bounds_rejected": True,
        "non_tensor_rejected": True,
        "module_probability_bounds_rejected": True,
        "non_npu_mask_request_rejected": True,
        "unsupported_dtype_rejected": True,
        "bad_keep_probability_rejected": True,
        "negative_numel_rejected": True,
        "npu_float16_rejected": device is not None,
        "npu_non_contiguous_supported": device is not None,
        "verdict": "passed",
    }
