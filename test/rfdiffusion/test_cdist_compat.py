#!/usr/bin/env python3

"""
CUDA-compatible FP32 ``cdist`` for the RFdiffusion geometry path.

The RFdiffusion trunk computes pairwise distance matrices with
``torch.cdist(x, x)`` and feeds them through the radial basis function.  On
CUDA, the default large-``p=2`` ``torch.cdist`` evaluates the padded-matmul
formula ``((x0*x0 + x2*x2) + x1*x1) - 2*x.y + ((y0*y0 + y2*y2) + y1*y1)`` with
FP32 FMA accumulation; the Ascend default kernel uses direct-distance arithmetic
instead, so the model input differs by up to ``1.1e-2`` even for identical
coordinates.
``rfantibody.rfdiffusion.cdist_compat`` reproduces the CUDA arithmetic with
ordinary FP32 device operations plus compensated (Dekker split, error-free sum)
arithmetic and a round-to-odd low component, and keeps upstream's size-based
routing (direct-distance kernel for ``P <= 25 and R <= 25``).

What this module checks
-----------------------
====================================  ===============================================
independent H100 recordings            six CUDA ``torch.cdist`` distance arrays plus
                                      their CUDA norm terms, bundled
                                      byte-identically under
                                      ``reference_outputs/cdist_cuda_holdouts``
                                      with recorded sha256; never produced by the
                                      candidate
replay on the device                  every distance within the pre-declared
                                      tolerance, every norm term byte-exact, all
                                      values finite, result on ``npu:0``
generality                            non-square batch, self-distance, broadcast
                                      batch, and a ``numel`` not 32-byte aligned,
                                      compared against a float64 CPU reference
wiring                                the five RFdiffusion call sites use
                                      ``cdist_compat.cdist`` and no ``torch.cdist``
                                      remains in the RFdiffusion model path
====================================  ===============================================

Tolerances are declared here, before any replay, and are never relaxed after
inspecting results: ``ATOL = 1e-5``, ``RTOL = 1e-4`` in float32.
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

from rfantibody.rfdiffusion.cdist_compat import cdist, cdist_p2, squared_norms
from test.util.device_selection import npu_device_name

# --- Pre-declared tolerances (fixed before any replay) ----------------------
ATOL = 1e-05
RTOL = 1e-04

# --- Tracked assets ---------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parents[1]
_HOLDOUT_DIR = _MODULE_DIR / "reference_outputs" / "cdist_cuda_holdouts"
_MANIFEST_PATH = _HOLDOUT_DIR / "manifest.json"
_PROVENANCE_PATH = _HOLDOUT_DIR / "provenance.json"
MANIFEST_SHA256 = "50cbd6e813675d5a1ac30ecbce85632318434c6b18eb6fc5024aa5adf5b706f8"

_EVIDENCE_DIR = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "outputs" / "evidence")
)
_EVIDENCE_PATH = _EVIDENCE_DIR / "rfd_geometry_cdist.json"

#: The five RFdiffusion call sites that must use the compatible arithmetic.
CALL_SITES = {
    "src/rfantibody/rfdiffusion/Embeddings.py": (
        "rbf(cdist(xyz_t[:,:,1], xyz_t[:,:,1]))",
        "rbf(cdist(Cb, Cb))",
    ),
    "src/rfantibody/rfdiffusion/Track_module.py": (
        "rbf(cdist(xyz[:,:,1], xyz[:,:,1]))",
        "rbf(cdist(xyz[:,:,1,:], xyz[:,:,1,:]))",
    ),
    "src/rfantibody/rfdiffusion/util_module.py": (
        "D = cdist(xyz, xyz)",
    ),
}
IMPORT_LINE = "from rfantibody.rfdiffusion.cdist_compat import cdist"


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


_MANIFEST = _load_json(_MANIFEST_PATH, "CUDA cdist holdout manifest")
_PROVENANCE = _load_json(_PROVENANCE_PATH, "CUDA cdist holdout provenance")
_CASES = _MANIFEST["cases"]

_EVIDENCE = {
    "module": "test.rfdiffusion.test_cdist_compat",
    "candidate": (
        "rfantibody.rfdiffusion.cdist_compat (CUDA padded-matmul FP32 arithmetic "
        "with compensated multiply/error-free sum and round-to-odd low component, "
        "upstream size routing for small operands)"
    ),
    "reference": {
        "scope": _PROVENANCE["scope"],
        "torch": _PROVENANCE["torch"],
        "device": _PROVENANCE["device"],
        "tf32": _PROVENANCE["tf32"],
        "seed": _PROVENANCE["seed"],
        "identity": _PROVENANCE["reference_identity"],
        "manifest": "test/rfdiffusion/reference_outputs/cdist_cuda_holdouts/manifest.json",
        "provenance": (
            "test/rfdiffusion/reference_outputs/cdist_cuda_holdouts/provenance.json"
        ),
        "collector": _PROVENANCE["collector"],
        "manifest_sha256": MANIFEST_SHA256,
    },
    "tolerances": {"atol": ATOL, "rtol": RTOL, "declared_before_execution": True},
    "cases": [],
    "generality": {},
    "wiring": {},
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
    _EVIDENCE["npu_device_name"] = npu_device_name()
    yield
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    cases_passed = all(
        row.get("verdict") == "passed" for row in _EVIDENCE["cases"]
    ) and len(_EVIDENCE["cases"]) == len(_CASES)
    _EVIDENCE["verdict"] = "passed" if cases_passed else "failed"
    _EVIDENCE_PATH.write_text(
        json.dumps(_EVIDENCE, indent=2, sort_keys=True) + "\n"
    )


def _npu_device():
    """``torch.device('npu:0')`` when a genuine NPU is usable, else ``None``."""
    if npu_device_name() is None:
        return None
    try:
        if int(torch.npu.device_count()) >= 1:
            return torch.device("npu:0")
    except Exception:  # pragma: no cover - vendor stack anomaly
        return None
    return None


def _norm_bytes(values, reference):
    """Byte equality of an FP32 norm term plus whether the layouts also match.

    The recorded ``norm_x``/``norm_y`` arrays use the H100 study's keep-dim
    ``(batch, n, 1)`` layout, while :func:`squared_norms` returns
    ``(batch, n)``.  The arithmetic contract is the *byte sequence*, so the
    comparison aligns the layout and reports ``shape_matches`` separately
    instead of treating a reshape as different arithmetic.
    """
    value = np.asarray(values)
    expected = np.asarray(reference)
    if value.size != expected.size:
        return False, False
    same = np.array_equal(
        np.ascontiguousarray(value).reshape(-1).view(np.uint32),
        np.ascontiguousarray(expected).reshape(-1).view(np.uint32),
    )
    return bool(same), tuple(value.shape) == tuple(expected.shape)


def _tolerance_stats(candidate: np.ndarray, reference: np.ndarray) -> dict:
    values = np.asarray(candidate)
    expected = np.asarray(reference)
    difference = np.abs(values.astype(np.float64) - expected.astype(np.float64))
    scale = np.abs(expected.astype(np.float64))
    allowed = ATOL + RTOL * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(scale > 0.0, difference / scale, 0.0)
    bitwise_equal = None
    value_mismatches = None
    if values.dtype == expected.dtype == np.float32:
        bitwise_equal = bool(
            np.array_equal(values.view(np.uint32), expected.view(np.uint32))
        )
        value_mismatches = int(np.count_nonzero(values != expected))
    return {
        "max_abs": float(difference.max()) if difference.size else 0.0,
        "max_rel": float(relative.max()) if relative.size else 0.0,
        "outside_tolerance": int(np.count_nonzero(difference > allowed)),
        "bitwise_equal": bitwise_equal,
        "value_mismatches": value_mismatches,
    }


def _case_id(case: dict) -> str:
    kind = "self" if case["self"] else "cross"
    return "case{case}-b{batch}-{n}x{m}-{kind}".format(kind=kind, **case)


# --------------------------------------------------------------------------- #
# 1. The recorded CUDA assets are byte-identical to their provenance
# --------------------------------------------------------------------------- #
def test_cdist_holdout_assets_integrity():
    """Manifest, provenance and every ``.npz`` byte still match the recording."""
    manifest_sha = _sha256(_MANIFEST_PATH)
    assert manifest_sha == MANIFEST_SHA256, (
        "the CUDA cdist holdout manifest changed: {} != {}".format(
            manifest_sha, MANIFEST_SHA256
        )
    )
    assert _PROVENANCE["manifest_sha256"] == MANIFEST_SHA256
    assert _PROVENANCE["torch"] == _MANIFEST["torch"]
    assert _PROVENANCE["device"] == _MANIFEST["device"]
    assert _PROVENANCE["tf32"] is _MANIFEST["tf32"] is False
    assert _PROVENANCE["seed"] == _MANIFEST["seed"]
    assert "independent" in _PROVENANCE["scope"]
    assert "H100" in _PROVENANCE["device"]
    assert "not neural model outputs" in _PROVENANCE["scope"]

    assert len(_CASES) == 6
    assert len(_PROVENANCE["cases"]) == len(_CASES)
    provenance_cases = {case["file"]: case for case in _PROVENANCE["cases"]}
    assert set(provenance_cases) == {case["file"] for case in _CASES}
    for case in _CASES:
        path = _HOLDOUT_DIR / case["file"]
        assert path.is_file(), "holdout asset is missing: {}".format(path)
        assert _sha256(path) == case["sha256"], (
            "{} bytes differ from the CUDA recording".format(case["file"])
        )
        entry = provenance_cases[case["file"]]
        for key in ("batch", "n", "m", "self", "scale", "offset"):
            assert entry[key] == case[key], (
                "{} provenance {} differs from the manifest".format(
                    case["file"], key
                )
            )
        assert entry["sha256"] == case["sha256"]
        asset = np.load(path)
        try:
            assert tuple(asset["distance"].shape) == (
                int(case["batch"]),
                int(case["n"]),
                int(case["m"]),
            )
            assert asset["distance"].dtype == np.float32
            assert asset["x"].dtype == np.float32
            assert asset["y"].dtype == np.float32
            assert tuple(asset["x"].shape) == (int(case["batch"]), int(case["n"]), 3)
            assert tuple(asset["y"].shape) == (int(case["batch"]), int(case["m"]), 3)
            assert tuple(asset["norm_x"].shape) == (
                int(case["batch"]),
                int(case["n"]),
                1,
            )
            assert tuple(asset["norm_y"].shape) == (
                int(case["batch"]),
                int(case["m"]),
                1,
            )
        finally:
            asset.close()

    _EVIDENCE["oracle_integrity"] = {
        "manifest_sha256": manifest_sha,
        "provenance_sha256": _sha256(_PROVENANCE_PATH),
        "cases": len(_CASES),
        "non_aligned_case": "case5 (1x533x271, 533*271 numel per batch)",
        "dtype": "float32",
        "verdict": "passed",
    }


# --------------------------------------------------------------------------- #
# 2. Per-case replay against the independent CUDA recording
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_cdist_matches_cuda_recording(case):
    """The recorded CUDA distances are reproduced inside the declared tolerance."""
    path = _HOLDOUT_DIR / case["file"]
    asset = np.load(path)
    try:
        x = np.ascontiguousarray(asset["x"])
        y = np.ascontiguousarray(asset["y"])
        reference = np.ascontiguousarray(asset["distance"])
        norm_x = np.ascontiguousarray(asset["norm_x"])
        norm_y = np.ascontiguousarray(asset["norm_y"])
    finally:
        asset.close()

    device = _npu_device()
    row = {
        "case": case["case"],
        "asset": "test/rfdiffusion/reference_outputs/cdist_cuda_holdouts/"
        + case["file"],
        "asset_sha256": case["sha256"],
        "shape": list(reference.shape),
        "input_shapes": [list(x.shape), list(y.shape)],
        "dtype": str(reference.dtype),
        "self_distance": bool(case["self"]),
        "coordinate_scale": float(case["scale"]),
        "coordinate_offset": float(case["offset"]),
    }

    # The arithmetic always runs on host tensors first, so the asset, the
    # tolerance and the compensated operator are exercised even without an
    # accelerator.
    cpu_x = torch.from_numpy(x)
    cpu_y = torch.from_numpy(y)
    host_norm_x_ok, host_norm_x_shape = _norm_bytes(squared_norms(cpu_x).numpy(), norm_x)
    host_norm_y_ok, host_norm_y_shape = _norm_bytes(squared_norms(cpu_y).numpy(), norm_y)
    assert host_norm_x_ok, (
        "{}: the norm term must use CUDA's (x0*x0 + x2*x2) + x1*x1 order".format(
            case["file"]
        )
    )
    assert host_norm_y_ok, (
        "{}: the norm term must use CUDA's (y0*y0 + y2*y2) + y1*y1 order".format(
            case["file"]
        )
    )
    cpu_result = cdist_p2(cpu_x, cpu_y)
    assert cpu_result.dtype is torch.float32
    assert cpu_result.shape == tuple(reference.shape)
    cpu_values = cpu_result.numpy()
    assert bool(np.isfinite(cpu_values).all())
    cpu_stats = _tolerance_stats(cpu_values, reference)
    row["host_replay"] = dict(
        cpu_stats,
        device="cpu",
        norm_x_byteexact=host_norm_x_ok,
        norm_y_byteexact=host_norm_y_ok,
        norm_x_shape_matches=host_norm_x_shape,
        norm_y_shape_matches=host_norm_y_shape,
    )
    assert cpu_stats["outside_tolerance"] == 0, (
        "{}: host compensated arithmetic is outside atol={} rtol={} "
        "(max_abs={}, max_rel={})".format(
            case["file"],
            ATOL,
            RTOL,
            cpu_stats["max_abs"],
            cpu_stats["max_rel"],
        )
    )

    if device is not None:
        dev_x = torch.from_numpy(x).to(device)
        dev_y = torch.from_numpy(y).to(device)
        dev_norm_x_ok, dev_norm_x_shape = _norm_bytes(
            squared_norms(dev_x).cpu().numpy(), norm_x
        )
        dev_norm_y_ok, dev_norm_y_shape = _norm_bytes(
            squared_norms(dev_y).cpu().numpy(), norm_y
        )
        assert dev_norm_x_ok, (
            "{}: the device norm term must use CUDA's order".format(case["file"])
        )
        assert dev_norm_y_ok, (
            "{}: the device norm term must use CUDA's order".format(case["file"])
        )

        result = cdist(dev_x, dev_y)
        assert result.device.type == "npu", (
            "{}: the distance must stay on the NPU, got {}".format(
                case["file"], result.device
            )
        )
        assert result.device == device
        assert result.dtype is torch.float32
        assert result.shape == tuple(reference.shape)
        values = result.cpu().numpy()
        assert bool(np.isfinite(values).all()), (
            "{}: non-finite distance values".format(case["file"])
        )
        stats = _tolerance_stats(values, reference)
        row["device"] = str(result.device)
        row["device_name"] = str(torch.npu.get_device_name(0))
        row["norm_x_byteexact"] = dev_norm_x_ok
        row["norm_y_byteexact"] = dev_norm_y_ok
        row["norm_x_shape_matches"] = dev_norm_x_shape
        row["norm_y_shape_matches"] = dev_norm_y_shape
        row.update(stats)
        assert stats["outside_tolerance"] == 0, (
            "{}: NPU distance is outside atol={} rtol={} (max_abs={}, "
            "max_rel={}, {} of {} values)".format(
                case["file"],
                ATOL,
                RTOL,
                stats["max_abs"],
                stats["max_rel"],
                stats["outside_tolerance"],
                values.size,
            )
        )
        # ``cdist_p2`` is the same operator the public helper dispatches to.
        assert torch.equal(cdist_p2(dev_x, dev_y), result)
    else:  # pragma: no cover - this host has an NPU; kept for standalone use
        row["device"] = None
        row["device_name"] = None
        row["norm_x_byteexact"] = host_norm_x_ok
        row["norm_y_byteexact"] = host_norm_y_ok
        row["norm_x_shape_matches"] = host_norm_x_shape
        row["norm_y_shape_matches"] = host_norm_y_shape
        row.update(cpu_stats)
        row["note"] = (
            "no accelerator on this host; the same asset, tolerance and "
            "operator were replayed on host tensors and no device evidence is "
            "claimed"
        )

    row["verdict"] = "passed"
    _EVIDENCE["cases"].append(row)


# --------------------------------------------------------------------------- #
# 3. Generality: shapes and alignments not present in the recordings
# --------------------------------------------------------------------------- #
def _float64_reference(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Independent float64 CPU reference (``torch.cdist`` in double precision)."""
    with torch.no_grad():
        return torch.cdist(
            x.detach().cpu().double(), y.detach().cpu().double(), p=2
        ).numpy()


def _generality_cases():
    """Deterministic tensors (no seeds, no weights) covering the generality set."""
    base = torch.arange(0, 2 * 37 * 3, dtype=torch.float32).reshape(2, 37, 3)
    cross_y = torch.linspace(-3.0, 7.5, 2 * 53 * 3, dtype=torch.float32).reshape(
        2, 53, 3
    )
    self_side = torch.arange(0, 64 * 3, dtype=torch.float32).reshape(1, 64, 3)
    broadcast_x = torch.linspace(-2.0, 2.0, 3 * 11 * 3, dtype=torch.float32).reshape(
        3, 11, 3
    )
    broadcast_y = torch.linspace(4.0, 6.0, 1 * 3, dtype=torch.float32).reshape(
        1, 1, 3
    ).expand(3, 1, 3).contiguous()
    flat = torch.arange(0, 4 * 17 * 3, dtype=torch.float32) * 0.25
    aligned_source = flat.reshape(4, 17, 3)
    unaligned = aligned_source[1:2]  # (1, 17, 3): 51 float32 = 204 bytes
    return [
        (
            "non_square_batch_2x37_x_2x53",
            base,
            cross_y,
            "batched non-square operands",
        ),
        ("self_distance_1x64", self_side, self_side, "self-distance diagonal"),
        (
            "broadcast_batch_3x11_x_3x1",
            broadcast_x,
            broadcast_y,
            "broadcast batch dimension",
        ),
        (
            "numel_not_32B_aligned_1x17",
            unaligned,
            unaligned,
            "numel 51 float32 = 204 bytes, not a multiple of 32",
        ),
    ]


@pytest.mark.parametrize(
    "name,x,y,contract", _generality_cases(), ids=[c[0] for c in _generality_cases()]
)
def test_cdist_generality(name, x, y, contract):
    """Shape-safe and alignment-safe behaviour against a float64 CPU reference."""
    assert x.dtype is torch.float32 and y.dtype is torch.float32
    numel = int(x.numel())
    if name == "numel_not_32B_aligned_1x17":
        assert (numel * 4) % 32 != 0, (
            "the unaligned case must not be a multiple of 32 bytes; got "
            "{} bytes".format(numel * 4)
        )
    reference = _float64_reference(x, y)
    row = {
        "case": name,
        "contract": contract,
        "input_shapes": [list(x.shape), list(y.shape)],
        "input_numel": [int(x.numel()), int(y.numel())],
        "input_bytes": [int(x.numel()) * 4, int(y.numel()) * 4],
        "expected_shape": list(reference.shape),
        "dtype": "torch.float32",
        "reference": "torch.cdist(float64 CPU)",
    }

    device = _npu_device()
    if device is None:  # pragma: no cover - this host has an NPU
        result = cdist_p2(x, y)
        row["device"] = None
        row["device_evidence"] = "not measured on this host"
    else:
        result = cdist(x.to(device), y.to(device))
        assert result.device.type == "npu", (
            "{}: the result must not leave the NPU, got {}".format(
                name, result.device
            )
        )
        assert result.device == device
        row["device"] = str(result.device)
        row["device_name"] = str(torch.npu.get_device_name(0))

    assert result.dtype is torch.float32
    assert tuple(result.shape) == tuple(reference.shape), (
        "{}: shape {} != expected {}".format(
            name, tuple(result.shape), tuple(reference.shape)
        )
    )
    values = result.detach().cpu().numpy()
    assert bool(np.isfinite(values).all()), "{}: non-finite values".format(name)
    stats = _tolerance_stats(values, reference)
    row.update(stats)
    assert stats["outside_tolerance"] == 0, (
        "{}: outside atol={} rtol={} against the float64 reference "
        "(max_abs={}, max_rel={})".format(
            name, ATOL, RTOL, stats["max_abs"], stats["max_rel"]
        )
    )
    if name == "self_distance_1x64":
        diagonal = np.diagonal(values[0])
        max_diagonal = float(np.abs(diagonal).max())
        row["max_abs_diagonal"] = max_diagonal
        assert max_diagonal <= ATOL, (
            "self-distance diagonal must be ~0; got {}".format(max_diagonal)
        )
    row["verdict"] = "passed"
    _EVIDENCE["generality"][name] = row


# --------------------------------------------------------------------------- #
# 4. The five RFdiffusion call sites use the compatible arithmetic
# --------------------------------------------------------------------------- #
def test_cdist_wiring():
    """No ``torch.cdist`` remains in the RFdiffusion model path; five sites wired."""
    wiring = {}
    total_sites = 0
    for relative, sites in CALL_SITES.items():
        path = _PROJECT_ROOT / relative
        assert path.is_file(), "wired source file is missing: {}".format(relative)
        text = path.read_text()
        assert IMPORT_LINE in text, (
            "{} must import the compatible cdist ({})".format(relative, IMPORT_LINE)
        )
        assert "torch.cdist(" not in text, (
            "{} still calls torch.cdist, which does not reproduce the CUDA "
            "arithmetic on npu".format(relative)
        )
        found = []
        for site in sites:
            count = text.count(site)
            assert count == 1, (
                "{}: expected exactly one '{}' call site, found {}".format(
                    relative, site, count
                )
            )
            found.append(site)
            total_sites += 1
        wiring[relative] = {
            "import": IMPORT_LINE,
            "call_sites": found,
            "torch_cdist_absent": True,
        }
    assert total_sites == 5, (
        "the RFdiffusion model path has five cdist call sites; found {}".format(
            total_sites
        )
    )
    # The module itself must keep the documented dispatch contract.
    cpu_a = torch.zeros(2, 4, 3, dtype=torch.float32)
    cpu_b = torch.ones(2, 5, 3, dtype=torch.float32)
    assert torch.equal(cdist(cpu_a, cpu_b), torch.cdist(cpu_a, cpu_b)), (
        "non-NPU tensors must delegate verbatim to torch.cdist"
    )
    with pytest.raises(NotImplementedError):
        cdist(cpu_a, cpu_b, p=1)
    device = _npu_device()
    if device is not None:
        # A float64 tensor must keep float64 metadata (the vendor stack casts
        # ``.double()`` on transfer, so the dtype is requested at creation), and
        # the wrapper must reject it instead of silently using float32.
        double = torch.ones(2, 4, 3, dtype=torch.float64, device=device)
        assert double.dtype is torch.float64, (
            "the vendor stack must keep float64 metadata for a tensor created "
            "as float64"
        )
        with pytest.raises(NotImplementedError):
            cdist(double, double)
    _EVIDENCE["wiring"] = {
        "files": wiring,
        "total_call_sites": total_sites,
        "host_delegation": "torch.cdist verbatim for non-npu tensors",
        "non_p2_rejected": True,
        "non_float32_npu_rejected": device is not None,
        "verdict": "passed",
    }
