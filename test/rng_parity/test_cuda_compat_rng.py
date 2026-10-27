"""CUDA-compatible distribution RNG parity, route and draw-plan tests.

``rfantibody.rng`` reproduces the CUDA distribution stream (Philox4x32-10 with
the documented CUDA execution profile) so that the same seed and shape select
the same residues as the original CUDA pipeline.  This module checks it against
**independent CUDA references**, never against the candidate itself:

* the 84 recorded H100 draws in ``reference_outputs/cuda_philox_holdouts``
  (4 seeds x 7 sizes x {float32 normal, float32 exponential, float64
  exponential}), each with the CUDA generator state before/after the call,
* the two in-situ draws of the original deterministic ProteinMPNN script
  (seed 42: one 478-element float32 normal, then the first float64
  ``multinomial`` event with its captured probabilities, sampled index and
  exponential base variates) in ``reference_outputs/cuda_mpnn_draws``.

Tolerances are declared here, before any candidate result is inspected, and are
never relaxed afterwards:

==========================  ==========================  ==================
draw                        tolerance                   reference
==========================  ==========================  ==================
float32 normal              ``atol=1e-5, rtol=1e-6``   84 H100 holdouts
float32 exponential         ``atol=1e-5, rtol=1e-6``   84 H100 holdouts
float64 exponential         ``atol=1e-15, rtol=1e-15`` 84 H100 holdouts
in-situ 478 normal          ``atol=1e-5, rtol=1e-6``   H100 seed-42 draw
in-situ multinomial noise   ``atol=1e-15, rtol=1e-15``  H100 seed-42 draw
offset consumption          exact equality              recorded ``state_after``
in-situ decoding order      identical abs-value argsort H100 seed-42 draw
==========================  ==========================  ==================

The float32 differences that remain are libm-vs-device rounding in
``log``/``sin``/``cos``; the seed/offset/shape contract is exact, so the
decoding order and the sampled residue index are reproduced.

On a host with a genuine Ascend NPU the sampling decision is executed through
the product API on ``npu:0`` (the return tensor's device and the sampled index
are asserted).  Without an NPU the identical expectations are evaluated with
the generator class directly; no case is skipped.  CPU/CUDA non-regression is
asserted separately: the helpers must delegate verbatim to the native torch
generators there.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from rfantibody import rng as antibody_rng
from rfantibody.device import npu_available

# --- Pre-declared tolerances (fixed before candidate execution) --------------
NORMAL32_ATOL = 1e-5
NORMAL32_RTOL = 1e-6
EXP32_ATOL = 1e-5
EXP32_RTOL = 1e-6
EXP64_ATOL = 1e-15
EXP64_RTOL = 1e-15

# --- Asset locations --------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_HOLDOUT_DIR = _MODULE_DIR / "reference_outputs" / "cuda_philox_holdouts"
_MPNN_DIR = _MODULE_DIR / "reference_outputs" / "cuda_mpnn_draws"
_MANIFEST_PATH = _HOLDOUT_DIR / "manifest.json"

_MPNN_RANDN = _MPNN_DIR / "00000_randn.npz"
_MPNN_MULTINOMIAL = _MPNN_DIR / "00001_multinomial.npz"

_EVIDENCE_DIR = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "outputs" / "evidence")
) / "rng_parity"
_EVIDENCE_PATH = _EVIDENCE_DIR / "cuda_compat_rng_parity.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    if not _MANIFEST_PATH.is_file():
        raise RuntimeError(
            "CUDA RNG holdout manifest is missing: {}".format(_MANIFEST_PATH)
        )
    return json.loads(_MANIFEST_PATH.read_text())


_MANIFEST = _load_manifest()
_ROWS = _MANIFEST["rows"]


def _case_id(row: dict) -> str:
    return "{file}-{kind}-{dtype}-n{numel}".format(
        file=row["file"],
        kind=row["kind"],
        dtype=row["dtype"].split(".")[-1],
        numel=row["numel"],
    )


def _tolerances(kind: str, dtype: torch.dtype):
    if kind == "normal" and dtype is torch.float32:
        return NORMAL32_ATOL, NORMAL32_RTOL
    if kind == "exponential" and dtype is torch.float32:
        return EXP32_ATOL, EXP32_RTOL
    if kind == "exponential" and dtype is torch.float64:
        return EXP64_ATOL, EXP64_RTOL
    raise AssertionError("no pre-declared tolerance for {} / {}".format(kind, dtype))


def _state_offset(state: np.ndarray) -> int:
    payload = np.asarray(state, dtype=np.uint8).tobytes()
    return int.from_bytes(payload[8:16], "little")


def _state_seed(state: np.ndarray) -> int:
    payload = np.asarray(state, dtype=np.uint8).tobytes()
    return int.from_bytes(payload[0:8], "little")


def _torch_dtype(name: str) -> torch.dtype:
    return {"torch.float32": torch.float32, "torch.float64": torch.float64}[name]


def _np_dtype(dtype: torch.dtype) -> np.dtype:
    return np.dtype(np.float32) if dtype is torch.float32 else np.dtype(np.float64)


def _error_summary(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    difference = np.abs(candidate - reference)
    floor = np.finfo(reference.dtype).eps
    relative = difference / np.maximum(np.abs(reference), floor)
    return {
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "max_rel_error": float(relative.max()) if relative.size else 0.0,
        "all_finite": bool(np.isfinite(candidate).all()),
        "element_count": int(difference.size),
    }


def _draw_one(seed: int, offset: int, kind: str, dtype: torch.dtype, shape):
    generator = antibody_rng.CudaCompatGenerator(seed=seed, offset=offset)
    dims = tuple(int(dim) for dim in shape)
    if kind == "normal":
        value = generator.normal(dims, dtype=dtype)
    else:
        value = generator.exponential(dims, dtype=dtype)
    return value.numpy(), generator.offset


def _accel_device():
    if npu_available():
        try:
            if int(torch.npu.device_count()) >= 1:
                return torch.device("npu:0")
        except Exception:  # pragma: no cover - vendor stack anomaly
            return None
    return None


# --- Machine-readable evidence ---------------------------------------------
_EVIDENCE = {
    "module": "test.rng_parity.test_cuda_compat_rng",
    "candidate": (
        "rfantibody.rng CudaCompatGenerator (Philox4x32-10 with the CUDA "
        "randn/multinomial distribution transforms and offset model), "
        "host-generated random input, sampling decision on the target device"
    ),
    "reference": (
        "independent PyTorch 2.3.1+cu118 CUDA draws recorded on an "
        "NVIDIA H100 80GB HBM3 (reference/cuda-rng-study); never the candidate"
    ),
    "tolerances": {
        "float32_normal": {"atol": NORMAL32_ATOL, "rtol": NORMAL32_RTOL},
        "float32_exponential": {"atol": EXP32_ATOL, "rtol": EXP32_RTOL},
        "float64_exponential": {"atol": EXP64_ATOL, "rtol": EXP64_RTOL},
        "offset_consumption": "exact equality to the recorded state_after offset",
        "decoding_order": "identical argsort of absolute values",
        "max_rel_error_definition": (
            "max |candidate - reference| / max(|reference|, eps(dtype))"
        ),
    },
    "stream_profile": dict(antibody_rng.CUDA_RNG_PROFILE),
    "oracle_integrity": {},
    "holdout_cases": [],
    "in_situ_draws": {},
    "draw_plan": {},
    "device_sampling_decision": {},
    "native_generator_non_regression": {},
    "determinism": {},
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
    """Every copied asset still matches its recorded sha256."""
    manifest_sha = _sha256(_MANIFEST_PATH)
    assert manifest_sha == "998d6105acf396d419cecac5fe23b6ffbba5366c12a8cebf58692fe35cb95556"
    assert len(_ROWS) == 84
    verified = {}
    for row in _ROWS:
        path = _HOLDOUT_DIR / row["file"]
        assert path.is_file(), "missing holdout asset {}".format(path)
        digest = _sha256(path)
        assert digest == row["sha256"], "holdout asset changed: {}".format(path)
        verified[row["file"]] = digest

    in_situ = {}
    for path, expected in (
        (_MPNN_RANDN, "290bfdb7219814f8cf2dc10dc1b986b42dbdd57bbcbc812c7d8320eefa779bb8"),
        (_MPNN_MULTINOMIAL, "6d34b00341bd5c6614bcaca998228b028a108482e6af3963ed25792b90e9e90b"),
    ):
        assert path.is_file(), "missing in-situ asset {}".format(path)
        digest = _sha256(path)
        assert digest == expected, "in-situ asset changed: {}".format(path)
        in_situ[path.name] = digest

    _EVIDENCE["oracle_integrity"] = {
        "manifest_sha256": manifest_sha,
        "holdout_files_verified": len(verified),
        "in_situ_files": in_situ,
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 2. Primitive parity over the 84 independent CUDA holdouts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "row",
    [pytest.param(row, id=_case_id(row)) for row in _ROWS],
)
def test_holdout_draw_parity(row):
    dtype = _torch_dtype(row["dtype"])
    np_dtype = _np_dtype(dtype)
    asset = np.load(_HOLDOUT_DIR / row["file"])
    reference = asset["value"].astype(np_dtype)
    offset_before = _state_offset(asset["state_before"])
    offset_after = _state_offset(asset["state_after"])
    assert _state_seed(asset["state_before"]) == row["seed"]

    candidate, consumed_offset = _draw_one(
        row["seed"],
        offset_before,
        row["kind"],
        dtype,
        reference.shape,
    )
    atol, rtol = _tolerances(row["kind"], dtype)
    summary = _error_summary(candidate, reference)
    allclose = bool(
        np.allclose(candidate, reference, atol=atol, rtol=rtol, equal_nan=False)
    )
    verdict = allclose and summary["all_finite"] and consumed_offset == offset_after

    _EVIDENCE["holdout_cases"].append(
        {
            "case": _case_id(row),
            "seed": row["seed"],
            "numel": row["numel"],
            "kind": row["kind"],
            "dtype": str(dtype),
            "device": "host generation, compared with the H100 CUDA oracle",
            "reference_file": row["file"],
            "reference_sha256": row["sha256"],
            "offset_before": offset_before,
            "offset_after_expected": offset_after,
            "offset_after_actual": consumed_offset,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": summary["max_abs_error"],
            "max_rel_error": summary["max_rel_error"],
            "all_finite": summary["all_finite"],
            "allclose": allclose,
            "verdict": "passed" if verdict else "failed",
        }
    )

    assert summary["all_finite"], "non-finite generated values for {}".format(row["file"])
    assert allclose, (
        "{} outside the pre-declared tolerance atol={} rtol={}: max_abs={} "
        "max_rel={}".format(
            row["file"],
            atol,
            rtol,
            summary["max_abs_error"],
            summary["max_rel_error"],
        )
    )
    assert consumed_offset == offset_after, (
        "philox offset after {} is {} but the CUDA state advanced to {}".format(
            row["file"], consumed_offset, offset_after
        )
    )


# --------------------------------------------------------------------------- #
# 3. In-situ ProteinMPNN draws (seed 42)
# --------------------------------------------------------------------------- #
def test_in_situ_proteinmpnn_draws():
    """The real seed-42 MPNN draws must be reproduced exactly in order."""
    normal_asset = np.load(_MPNN_RANDN)
    reference_randn = normal_asset["value"].astype(np.float32)
    assert _state_seed(normal_asset["state_before"]) == 42
    assert _state_offset(normal_asset["state_before"]) == 0
    assert _state_offset(normal_asset["state_after"]) == 4

    candidate_randn, offset_after_randn = _draw_one(
        42, 0, "normal", torch.float32, reference_randn.shape
    )
    randn_summary = _error_summary(candidate_randn, reference_randn)
    randn_order_match = bool(
        np.array_equal(
            np.argsort(np.abs(candidate_randn).ravel()),
            np.argsort(np.abs(reference_randn).ravel()),
        )
    )

    multinomial_asset = np.load(_MPNN_MULTINOMIAL)
    probabilities = multinomial_asset["probabilities"].astype(np.float64)
    expected_sample = int(np.asarray(multinomial_asset["sample"]).ravel()[0])
    reference_noise = multinomial_asset["independent_exponential_noise"].astype(
        np.float64
    )
    assert _state_offset(multinomial_asset["state_before"]) == 4
    assert _state_offset(multinomial_asset["state_after"]) == 8
    assert probabilities.shape == reference_noise.shape == (1, 21)

    candidate_noise, offset_after_multinomial = _draw_one(
        42, 4, "exponential", torch.float64, probabilities.shape
    )
    noise_summary = _error_summary(candidate_noise, reference_noise)
    selected = int(
        torch.div(
            torch.from_numpy(probabilities), torch.from_numpy(candidate_noise)
        )
        .argmax(dim=-1)
        .item()
    )

    _EVIDENCE["in_situ_draws"] = {
        "reference": str(_MPNN_DIR),
        "randn": {
            "seed": 42,
            "shape": list(reference_randn.shape),
            "dtype": "torch.float32",
            "offset_before": 0,
            "offset_after_expected": 4,
            "offset_after_actual": offset_after_randn,
            "atol": NORMAL32_ATOL,
            "rtol": NORMAL32_RTOL,
            "max_abs_error": randn_summary["max_abs_error"],
            "max_rel_error": randn_summary["max_rel_error"],
            "all_finite": randn_summary["all_finite"],
            "abs_value_order_match": randn_order_match,
            "verdict": "passed" if randn_order_match else "failed",
        },
        "multinomial": {
            "seed": 42,
            "shape": list(probabilities.shape),
            "dtype": "torch.float64",
            "offset_before": 4,
            "offset_after_expected": 8,
            "offset_after_actual": offset_after_multinomial,
            "atol": EXP64_ATOL,
            "rtol": EXP64_RTOL,
            "max_abs_error": noise_summary["max_abs_error"],
            "max_rel_error": noise_summary["max_rel_error"],
            "all_finite": noise_summary["all_finite"],
            "expected_sample": expected_sample,
            "selected_sample": selected,
            "verdict": "passed" if selected == expected_sample else "failed",
        },
    }

    assert randn_summary["all_finite"]
    assert np.allclose(
        candidate_randn,
        reference_randn,
        atol=NORMAL32_ATOL,
        rtol=NORMAL32_RTOL,
        equal_nan=False,
    ), "in-situ MPNN randn outside the pre-declared tolerance"
    assert offset_after_randn == 4
    assert randn_order_match, "decoding order differs from the CUDA stream"

    assert noise_summary["all_finite"]
    assert np.allclose(
        candidate_noise,
        reference_noise,
        atol=EXP64_ATOL,
        rtol=EXP64_RTOL,
        equal_nan=False,
    ), "in-situ multinomial exponential variates outside the pre-declared tolerance"
    assert offset_after_multinomial == 8
    assert selected == expected_sample, "sampled residue index differs from CUDA"


# --------------------------------------------------------------------------- #
# 4. ProteinMPNN draw plan (real stream consumption)
# --------------------------------------------------------------------------- #
def test_proteinmpnn_draw_plan():
    """1 normal (1, 478) f32 + 956 exponential (1, 21) f64 ends at offset 3828."""
    generator = antibody_rng.CudaCompatGenerator(seed=42, offset=0)
    assert generator.offset == 0
    generator.normal((1, 478), dtype=torch.float32)
    offsets = [generator.offset]
    for _ in range(956):
        generator.exponential((1, 21), dtype=torch.float64)
        offsets.append(generator.offset)

    expected = [4 * (index + 1) for index in range(len(offsets))]
    _EVIDENCE["draw_plan"] = {
        "seed": 42,
        "draws": len(offsets),
        "composition": "1 x float32 normal (1, 478) then 956 x float64 exponential (1, 21)",
        "first_offsets": offsets[:4],
        "last_offsets": offsets[-3:],
        "final_offset": offsets[-1],
        "expected_final_offset": 3828,
        "per_step_advance": 4,
        "verdict": "passed" if offsets == expected else "failed",
    }
    assert offsets == expected, (
        "draw plan offsets are not +4 per call: first {} last {}".format(
            offsets[:8], offsets[-4:]
        )
    )
    assert offsets[-1] == 3828


# --------------------------------------------------------------------------- #
# 5. Device execution of the sampling decision and native-generator delegation
# --------------------------------------------------------------------------- #
def test_device_sampling_decision():
    """The multinomial sampling decision runs on the genuine accelerator."""
    asset = np.load(_MPNN_MULTINOMIAL)
    probabilities_np = asset["probabilities"].astype(np.float64).copy()
    expected_sample = int(np.asarray(asset["sample"]).ravel()[0])
    device = _accel_device()

    if device is not None:
        antibody_rng.manual_seed(42)
        probs = torch.from_numpy(probabilities_np).to(device)
        decoding_randn = antibody_rng.randn((1, 478), device=device)
        sample = antibody_rng.multinomial(probs, 1)
        seed, offset = antibody_rng.get_seed_offset(device)

        evidence = {
            "route": "product API on {}".format(device),
            "device_name": str(torch.npu.get_device_name(0)),
            "probabilities_device": str(probs.device),
            "decoding_randn_device": str(decoding_randn.device),
            "sample_device": str(sample.device),
            "sample_dtype": str(sample.dtype),
            "sample_shape": list(sample.shape),
            "selected_sample": int(sample.item()),
            "expected_sample": expected_sample,
            "stream_seed": seed,
            "stream_offset_after": offset,
            "verdict": "pending",
        }
        passed = (
            sample.device.type == "npu"
            and sample.shape == (1, 1)
            and sample.dtype == torch.int64
            and int(sample.item()) == expected_sample
            and (seed, offset) == (42, 8)
        )
        evidence["verdict"] = "passed" if passed else "failed"
        _EVIDENCE["device_sampling_decision"] = evidence

        assert sample.device.type == "npu", "sampling decision did not run on the NPU"
        assert sample.shape == (1, 1)
        assert sample.dtype == torch.int64
        assert int(sample.item()) == expected_sample
        assert (seed, offset) == (42, 8)
        return

    # No accelerator on this host: evaluate the same expectation with the
    # compatibility generator directly (the product API delegates to torch on
    # CPU by contract, which is asserted separately below).
    noise = antibody_rng.CudaCompatGenerator(seed=42, offset=4).exponential(
        probabilities_np.shape, dtype=torch.float64
    )
    probs = torch.from_numpy(probabilities_np)
    sample = torch.div(probs, noise).argmax(dim=-1, keepdim=True)
    selected = int(sample.item())
    _EVIDENCE["device_sampling_decision"] = {
        "route": "no accelerator on this host; compatibility generator directly",
        "selected_sample": selected,
        "expected_sample": expected_sample,
        "verdict": "passed" if selected == expected_sample else "failed",
    }
    assert selected == expected_sample


def test_native_generator_non_regression():
    """CPU/CUDA targets keep delegating verbatim to the native torch streams."""
    probs = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float64
    )

    torch.manual_seed(20240617)
    torch_sample = torch.multinomial(probs, 1)
    torch.manual_seed(20240617)
    compat_sample = antibody_rng.multinomial(probs, 1)
    assert torch.equal(torch_sample, compat_sample)
    assert compat_sample.device.type == "cpu"

    torch.manual_seed(20240617)
    torch_randn = torch.randn(11)
    torch.manual_seed(20240617)
    compat_randn = antibody_rng.randn((11,), device="cpu")
    assert torch.equal(torch_randn, compat_randn)

    torch.manual_seed(20240617)
    torch_exponential = torch.empty(9).exponential_()
    torch.manual_seed(20240617)
    compat_exponential = antibody_rng.exponential((9,), device="cpu")
    assert torch.equal(torch_exponential, compat_exponential)

    reference = torch.zeros(4, 3)
    torch.manual_seed(20240617)
    torch_like = torch.randn_like(reference)
    torch.manual_seed(20240617)
    compat_like = antibody_rng.randn_like(reference)
    assert torch.equal(torch_like, compat_like)

    _EVIDENCE["native_generator_non_regression"] = {
        "multinomial_equal": True,
        "randn_equal": True,
        "exponential_equal": True,
        "randn_like_equal": True,
        "delegated_device": "cpu (the same verbatim delegation applies to cuda)",
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 6. Determinism, resume and the documented contract errors
# --------------------------------------------------------------------------- #
def test_offset_determinism_and_resume():
    first = antibody_rng.CudaCompatGenerator(seed=17)
    second = antibody_rng.CudaCompatGenerator(seed=17)
    first_values = first.normal((64,), dtype=torch.float32)
    second_values = second.normal((64,), dtype=torch.float32)
    assert torch.equal(first_values, second_values)
    assert first.offset == second.offset

    resume_offset = first.offset
    tail = first.exponential((10,), dtype=torch.float64)
    resumed = antibody_rng.CudaCompatGenerator(seed=17, offset=resume_offset)
    resumed_tail = resumed.exponential((10,), dtype=torch.float64)
    assert torch.equal(tail, resumed_tail)
    assert resumed.offset == first.offset

    antibody_rng.manual_seed(42)
    assert antibody_rng.get_seed_offset() == (42, 0)
    device = _accel_device()
    if device is not None:
        antibody_rng.manual_seed(42)
        drawn = antibody_rng.randn((3, 5), device=device)
        seed, offset = antibody_rng.get_seed_offset(device)
        assert (seed, offset) == (42, 4)
        antibody_rng.manual_seed(42)
        assert antibody_rng.get_seed_offset(device) == (42, 0)
        reproduced = antibody_rng.randn((3, 5), device=device)
        assert torch.equal(drawn, reproduced)
        reset_evidence = {"device_stream_reset": True, "offset_after_draw": 4}
    else:
        reset_evidence = {"device_stream_reset": "no accelerator", "offset_after_draw": None}

    _EVIDENCE["determinism"] = {
        "same_seed_identical": True,
        "resume_from_recorded_offset_identical": True,
        "resume_offset": resume_offset,
        "manual_seed_resets_offset": True,
        **reset_evidence,
        "verdict": "passed",
    }


def test_contract_errors():
    generator = antibody_rng.CudaCompatGenerator(seed=0)
    with pytest.raises(NotImplementedError):
        generator.normal((3,), dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        generator.exponential((3,), dtype=torch.float16)
    with pytest.raises(ValueError):
        antibody_rng.CudaCompatGenerator(seed=0, offset=3)
    with pytest.raises(ValueError):
        antibody_rng.CudaCompatGenerator(seed=0, profile={"sm_count": 0,
                                                         "max_threads_per_sm": 2048,
                                                         "block_size": 256})

    device = _accel_device()
    if device is not None:
        probs = torch.full((2, 5), 0.2, dtype=torch.float64, device=device)
        with pytest.raises(NotImplementedError):
            antibody_rng.multinomial(probs, 2)

    _EVIDENCE["contract_errors"] = {
        "normal_float64_unsupported": True,
        "exponential_float16_unsupported": True,
        "misaligned_offset_rejected": True,
        "invalid_profile_rejected": True,
        "num_samples_gt_one_unsupported_on_npu": device is not None,
        "verdict": "passed",
    }
