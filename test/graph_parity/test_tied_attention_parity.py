"""CUDA-order tied-attention contraction on the Ascend NPU.

What is checked, against criteria fixed *before* any candidate result is seen:

* the seven vendored oracle assets are the recorded actual-CUDA tensors (the
  file sha256 must match the vendored manifest) and hold float32 ``query``
  ``[B, N, I, H, K]``, ``key`` ``[B, N, J, H, K]`` and ``logits`` ``[B, I, J, H]``;
* the production dispatch point (``rfantibody.attention_backend``) computes
  ``opt_einsum.contract('bnihk,bnjhk->bijh', query, key)`` on ``npu:0``
  uint32-bit-exactly like every recorded actual-CUDA oracle, and also within
  ``atol=1e-5`` / ``rtol=1e-4``;
* the *un-migrated* ``opt_einsum`` contraction on ``npu:0`` is **not** bit-exact
  for at least one case, so the bit-exact agreement above is attributable to the
  ordered native kernel and not to the asset reproducing itself;
* off the NPU the dispatch point is bitwise identical to the upstream
  ``opt_einsum`` expression (CPU/CUDA behaviour unchanged);
* the production call site ``BiasedAxialAttention.forward`` of RFdiffusion
  dispatches the native adapter on ``npu:0`` (static source contract plus a
  recorded runtime dispatch that executes the real call), and the returned
  contraction agrees with the independent CPU contraction of the same operands;
* the public API executes a *hardware* kernel: the probe-identical profiler
  channel of the call contains a ``KERNEL_*`` task event, the op-level vendor
  report contains exactly one ``AI_CORE`` row named ``tied_attention_logits``,
  the negative token ``__no_such_kernel__`` resolves nowhere, and the unchanged
  instrumentation controls (a separate native AscendC kernel as positive
  control, a CPU-only window as negative control) behave as accepted.

Tolerances ``PARITY_ATOL``/``PARITY_RTOL`` are the task's float32 operator
contract and are declared here, before execution; they are never relaxed after
inspecting a result.  No case is skipped, xfailed or deselected: the only skip is
the suite-wide genuine "this host has no Ascend NPU" case of the shared
``npu_device`` fixture.
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

from test.graph_parity import route_evidence

# --- Pre-declared tolerances (fixed before candidate execution) --------------
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4

_EQUATION = "bnihk,bnjhk->bijh"
#: Kernel identity that must appear in the vendor report channel.
EXPECTED_KERNEL = "tied_attention_logits"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HOLDOUT_DIR = Path(__file__).resolve().parent / "cuda_ordered_holdouts"
_MANIFEST = _HOLDOUT_DIR / "manifest.json"
_EVIDENCE_PATH = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "outputs" / "evidence")
) / "graph_parity"
_EVIDENCE_FILE = _EVIDENCE_PATH / "tied_attention_parity.json"

_ROUTE_CASE = "test.graph_parity.test_tied_attention_parity:route_case_run"

_EVIDENCE = {
    "operator": (
        "opt_einsum.contract('bnihk,bnjhk->bijh', query, key): the tied attention "
        "contraction of rfantibody.rfdiffusion.Attention_module."
        "BiasedAxialAttention"
    ),
    "candidate": (
        "rfantibody.attention_backend.tied_attention_logits -> "
        "ascend_graph::tied_attention_logits -> native AscendC kernel "
        "tied_attention_logits on npu:0 (CUDA-order correctly rounded FP32 FMA "
        "chain: compensated product, two TwoSums, vectorized round-to-odd)"
    ),
    "reference": (
        "the seven vendored actual-CUDA oracle tensors recorded with torch "
        "2.5.1+cu118 on an NVIDIA H100 80GB HBM3 with tf32 disabled "
        "(independent synthetic operator oracles, no model weights or "
        "activations)"
    ),
    "tolerances": {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "structural_criterion": (
            "uint32 bit-exact equality of the returned float32 logits with the "
            "recorded actual-CUDA logits"
        ),
        "max_abs_error_definition": "max |candidate - reference| over all elements",
        "max_rel_error_definition": (
            "max |candidate - reference| / max(|reference|, 1e-12) over all elements"
        ),
        "declared_before": "module import, before any case is executed",
    },
    "assets": {},
    "cases": [],
    "plain_path_control": {},
    "off_npu_delegation": {},
    "production_call_site": {},
    "route": {},
}


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture(scope="module", autouse=True)
def _flush_evidence():
    """Write the evidence table even when a case fails."""
    yield
    _EVIDENCE["generated_utc"] = _utc_now()
    _EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
    _EVIDENCE_FILE.write_text(
        json.dumps(_EVIDENCE, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _manifest():
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert cases, f"{_MANIFEST} lists no cases"
    return payload, cases


def _recording():
    """Recording environment of the vendored actual-CUDA oracles."""
    payload, _ = _manifest()
    return {
        "torch": payload.get("torch"),
        "gpu": payload.get("gpu"),
        "tf32": payload.get("tf32"),
        "equation": payload["cases"][0].get("equation"),
        "input_seed": payload["cases"][0].get("input_seed"),
    }


def _load_case(entry):
    """Load one vendored oracle case and check it against the manifest."""
    path = _HOLDOUT_DIR / entry["file"]
    assert path.is_file(), f"missing vendored oracle asset {path}"
    digest = _sha256_file(path)
    assert digest == entry["sha256"], (
        f"{entry['file']}: sha256 {digest} does not match the vendored manifest "
        f"value {entry['sha256']}"
    )
    with np.load(path) as payload:
        query = np.ascontiguousarray(payload["query"])
        key = np.ascontiguousarray(payload["key"])
        logits = np.ascontiguousarray(payload["logits"])
    batch, num_n, dim_i, dim_j, num_h, num_k = entry["dimensions"]
    assert query.shape == (batch, num_n, dim_i, num_h, num_k), (
        f"{entry['file']}: query shape {query.shape} does not match the manifest "
        f"dimensions {entry['dimensions']} ([B, N, I, J, H, K])"
    )
    assert key.shape == (batch, num_n, dim_j, num_h, num_k), (
        f"{entry['file']}: key shape {key.shape} does not match the manifest "
        f"dimensions {entry['dimensions']}"
    )
    assert logits.shape == (batch, dim_i, dim_j, num_h), (
        f"{entry['file']}: logits shape {logits.shape} is not [B, I, J, H]"
    )
    for name, array in (("query", query), ("key", key), ("logits", logits)):
        assert array.dtype == np.float32, (
            f"{entry['file']}: {name} dtype is {array.dtype}, expected float32"
        )
    return {
        "path": path,
        "sha256": digest,
        "dimensions": [int(value) for value in entry["dimensions"]],
        "query": query,
        "key": key,
        "logits": logits,
    }


def _tolerances(candidate: torch.Tensor, reference: torch.Tensor):
    """Exact error metrics and bit-exactness of the two float32 tensors."""
    cand = candidate.detach().cpu()
    ref = reference.detach().cpu()
    diff = (cand.double() - ref.double()).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    if ref.numel():
        rel = diff / ref.double().abs().clamp_min(1e-12)
        max_rel = float(rel.max())
        allowed = PARITY_ATOL + PARITY_RTOL * ref.double().abs()
        outside = int((diff > allowed).sum())
    else:  # pragma: no cover - the vendored cases are never empty
        max_rel = 0.0
        outside = 0
    mismatch = int(
        (cand.contiguous().view(torch.int32) != ref.contiguous().view(torch.int32))
        .sum()
        .item()
    )
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "elements_outside_atol_plus_rtol": outside,
        "uint32_bit_mismatches": mismatch,
        "bit_exact": mismatch == 0,
        "all_finite": bool(torch.isfinite(cand).all() and torch.isfinite(ref).all()),
    }


# --------------------------------------------------------------------------- #
# Route channel case (imported by a fresh interpreter, no fixtures available)
# --------------------------------------------------------------------------- #
def route_case_run() -> torch.Tensor:
    """One production-path contraction on ``npu:0`` for the route channels.

    ``route_evidence`` imports this function in a fresh interpreter for the
    probe-identical hardware channel and the op-level report channel.  The case
    is a small non-degenerate contraction with a 16-byte (``K=4``) reduced width,
    i.e. a non-32-byte-aligned row, executed through the public dispatch point.
    """
    from rfantibody.attention_backend import tied_attention_logits

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu is unavailable")
    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(20251023)
    query = torch.randn(1, 3, 4, 2, 4, generator=generator).to(device)
    key = torch.randn(1, 3, 5, 2, 4, generator=generator).to(device)
    return tied_attention_logits(query, key)


# --------------------------------------------------------------------------- #
# 1. Vendored oracle assets
# --------------------------------------------------------------------------- #
def test_vendored_cuda_oracles_are_intact():
    """The vendored actual-CUDA oracle assets match the vendored manifest."""
    manifest, cases = _manifest()
    assert manifest.get("tf32") is False, (
        f"{_MANIFEST}: the oracles must be recorded with tf32 disabled, got "
        f"{manifest.get('tf32')!r}"
    )
    assert "cu118" in str(manifest.get("torch", "")), (
        f"{_MANIFEST}: unexpected recording torch build {manifest.get('torch')!r}"
    )
    _EVIDENCE["assets"] = {
        "directory": str(_HOLDOUT_DIR),
        "manifest": str(_MANIFEST),
        "manifest_sha256": _sha256_file(_MANIFEST),
        "provenance": (
            "byte-identical copies of the actual-CUDA operator oracles recorded "
            "on the host H100 (torch 2.5.1+cu118, tf32 disabled); the source "
            "directory is not part of the migrated tree"
        ),
        "recording": _recording(),
        "cases": [
            {
                "case": int(entry["case"]),
                "file": entry["file"],
                "sha256": entry["sha256"],
                "dimensions_b_n_i_j_h_k": entry["dimensions"],
                "scope": entry.get("scope"),
            }
            for entry in cases
        ],
    }
    for entry in cases:
        loaded = _load_case(entry)
        assert loaded["sha256"] == entry["sha256"]
    shape_rows = [entry["dimensions"] for entry in cases]
    assert any(int(row[5]) * 4 % 32 != 0 for row in shape_rows), (
        "no vendored case has a non-32-byte-aligned reduced width: "
        f"{shape_rows}"
    )
    assert any(int(row[1]) > 64 for row in shape_rows), (
        "no vendored case reduces over a long axis (the real model reduces over "
        f"N=478): {shape_rows}"
    )
    assert any(int(row[4]) > 1 for row in shape_rows), (
        f"no vendored case has multiple heads: {shape_rows}"
    )


# --------------------------------------------------------------------------- #
# 2. Ordered contraction parity against the actual-CUDA oracles
# --------------------------------------------------------------------------- #
def _case_params():
    _, cases = _manifest()
    return [
        pytest.param(entry, id="case{}".format(entry["case"])) for entry in cases
    ]


@pytest.mark.parametrize("entry", _case_params())
def test_ordered_contraction_matches_cuda_logits(native_graph_ops, npu_device, entry):
    """The production dispatch point reproduces every actual-CUDA oracle."""
    from rfantibody.attention_backend import tied_attention_logits

    loaded = _load_case(entry)
    query = torch.from_numpy(loaded["query"]).to(npu_device)
    key = torch.from_numpy(loaded["key"]).to(npu_device)
    reference = torch.from_numpy(loaded["logits"])

    candidate = tied_attention_logits(query, key)
    torch.npu.synchronize()

    batch, num_n, dim_i, dim_j, num_h, num_k = loaded["dimensions"]
    assert candidate.device.type == "npu", (
        f"candidate device is {candidate.device}, expected an NPU tensor"
    )
    assert candidate.dtype == torch.float32, (
        f"candidate dtype is {candidate.dtype}, expected float32"
    )
    assert tuple(candidate.shape) == (batch, dim_i, dim_j, num_h), (
        f"candidate shape {tuple(candidate.shape)} != [B, I, J, H] = "
        f"{(batch, dim_i, dim_j, num_h)}"
    )
    assert reference.device.type == "cpu"

    metrics = _tolerances(candidate, reference)
    record = {
        "case": int(entry["case"]),
        "asset_file": entry["file"],
        "asset_sha256": loaded["sha256"],
        "reference_identity": (
            "actual CUDA (torch {} on {}, tf32=false) recorded logits array in "
            "{}; independent synthetic operator oracle, never the candidate"
        ).format(
            _recording()["torch"], _recording()["gpu"], entry["file"]
        ),
        "shapes": {
            "query": list(query.shape),
            "key": list(key.shape),
            "candidate": list(candidate.shape),
            "reference": list(reference.shape),
        },
        "dtypes": {
            "query": str(query.dtype),
            "key": str(key.dtype),
            "candidate": str(candidate.dtype),
            "reference": str(reference.dtype),
        },
        "devices": {
            "query": str(query.device),
            "key": str(key.device),
            "candidate": str(candidate.device),
            "reference": str(reference.device),
        },
        "num_heads": int(num_h),
        "reduced_width": int(num_k),
        "reduction_length": int(num_n),
        "non_32_byte_aligned_reduced_width": bool((int(num_k) * 4) % 32 != 0),
        "non_32_byte_aligned_output_row": bool((int(dim_j) * 4) % 32 != 0),
        "input_sha256": {
            "query": _tensor_digest(query),
            "key": _tensor_digest(key),
            "reference_logits": _tensor_digest(reference),
        },
        "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        "artifact_sha256": native_graph_ops["artifact_sha256"],
        **metrics,
    }
    _EVIDENCE["cases"].append(record)

    assert metrics["all_finite"], (
        f"case {entry['case']}: a non-finite value is present in the candidate or "
        "the reference"
    )
    assert metrics["bit_exact"], (
        f"case {entry['case']}: {metrics['uint32_bit_mismatches']} of "
        f"{reference.numel()} element(s) differ from the actual-CUDA logits at "
        f"the bit level (max |diff| = {metrics['max_abs_error']!r})"
    )
    assert metrics["elements_outside_atol_plus_rtol"] == 0, (
        f"case {entry['case']}: {metrics['elements_outside_atol_plus_rtol']} "
        f"element(s) outside atol={PARITY_ATOL} + rtol={PARITY_RTOL} "
        f"(max |diff| = {metrics['max_abs_error']!r}, max rel = "
        f"{metrics['max_rel_error']!r})"
    )


# --------------------------------------------------------------------------- #
# 3. Control: the un-migrated contraction is not the oracle
# --------------------------------------------------------------------------- #
def test_plain_contraction_is_not_the_reference_candidate(npu_device):
    """The plain ``opt_einsum`` contraction on ``npu:0`` is not bit-exact.

    This is the control that makes the bit-exact result above attributable to the
    ordered compensated kernel: the same operands contracted by the un-migrated
    expression on the same device differ from the recorded CUDA value.
    """
    from opt_einsum import contract

    _, cases = _manifest()
    rows = []
    for entry in cases:
        loaded = _load_case(entry)
        query = torch.from_numpy(loaded["query"]).to(npu_device)
        key = torch.from_numpy(loaded["key"]).to(npu_device)
        reference = torch.from_numpy(loaded["logits"])
        plain = contract(_EQUATION, query, key)
        torch.npu.synchronize()
        assert plain.device.type == "npu"
        metrics = _tolerances(plain, reference)
        rows.append({"case": int(entry["case"]), **metrics})
    differing = [row["case"] for row in rows if not row["bit_exact"]]
    _EVIDENCE["plain_path_control"] = {
        "case": (
            "opt_einsum.contract('bnihk,bnjhk->bijh', query, key) on npu:0, i.e. "
            "the un-migrated expression of the same call site"
        ),
        "device": str(npu_device),
        "expectation": (
            "at least one recorded case must differ at the bit level, otherwise "
            "the asset would be reproducing itself"
        ),
        "rows": rows,
        "cases_not_bit_exact": differing,
    }
    assert differing, (
        "the un-migrated opt_einsum contraction on npu:0 is uint32-bit-exact for "
        "every vendored case, so this control cannot attribute the bit-exact "
        "agreement of the native route to the ordered kernel; rows: "
        f"{json.dumps(rows, sort_keys=True)}"
    )


# --------------------------------------------------------------------------- #
# 4. Off-NPU delegation is unchanged
# --------------------------------------------------------------------------- #
def test_dispatch_delegates_unchanged_off_npu():
    """On CPU the dispatch point is bitwise equal to the upstream expression."""
    from opt_einsum import contract

    from rfantibody.attention_backend import tied_attention_logits

    _, cases = _manifest()
    checked = 0
    for entry in cases:
        loaded = _load_case(entry)
        query = torch.from_numpy(loaded["query"])
        key = torch.from_numpy(loaded["key"])
        expected = contract(_EQUATION, query, key)
        candidate = tied_attention_logits(query, key)
        assert candidate.device.type == "cpu"
        assert tuple(candidate.shape) == tuple(expected.shape)
        assert torch.equal(
            candidate.contiguous().view(torch.int32),
            expected.contiguous().view(torch.int32),
        ), (
            f"case {entry['case']}: the CPU path is not bitwise identical to "
            "opt_einsum.contract"
        )
        checked += 1
    _EVIDENCE["off_npu_delegation"] = {
        "case": (
            "rfantibody.attention_backend.tied_attention_logits on CPU tensors"
        ),
        "expectation": (
            "bitwise identical to opt_einsum.contract('bnihk,bnjhk->bijh') on CPU"
        ),
        "cases_checked": checked,
        "passed": True,
    }


# --------------------------------------------------------------------------- #
# 5. Production call site dispatches the native adapter
# --------------------------------------------------------------------------- #
def test_production_call_site_dispatches_native(npu_device):
    """``BiasedAxialAttention.forward`` reaches the native adapter on ``npu:0``."""
    from opt_einsum import contract

    import rfantibody.ascend_ops.attention as native_attention
    from rfantibody.rfdiffusion.Attention_module import BiasedAxialAttention

    source_path = _PROJECT_ROOT / "src" / "rfantibody" / "rfdiffusion" / "Attention_module.py"
    source = source_path.read_text(encoding="utf-8")
    assert "from rfantibody.attention_backend import tied_attention_logits" in source, (
        f"{source_path} does not import the dispatch point"
    )
    assert _EQUATION not in source, (
        f"{source_path} still contains the un-migrated {_EQUATION!r} expression"
    )
    assert "attn = tied_attention_logits(query, key)" in source, (
        f"{source_path}: BiasedAxialAttention.forward does not call the dispatch "
        "point for the tied attention"
    )

    # Kernel staging contract (regression guard): every ``DataCopyPad`` GM -> UB
    # destination of the native kernel must be 32-byte aligned by construction,
    # so a staged query row advances by a whole 32-byte block (8 float32) and
    # never by the raw ``kITile`` row length (4 float32 == 16 bytes).
    kernel_path = (
        _PROJECT_ROOT
        / "csrc"
        / "ascend_graph"
        / "op_kernel"
        / "tied_attention_logits.cpp"
    )
    kernel_source = kernel_path.read_text(encoding="utf-8")
    assert "kStageAlignElements = 8" in kernel_source, (
        f"{kernel_path}: the staged query row alignment (8 float32 == 32 bytes) "
        "is not declared"
    )
    assert "qStageStride = RoundUp(kITile)" in kernel_source, (
        f"{kernel_path}: the staged query row stride is not derived from "
        "RoundUp(kITile) rounded to a whole 32-byte block"
    )
    assert "TBuf<TPosition::VECIN> qStageBuf;" in kernel_source, (
        f"{kernel_path}: the query staging buffer does not carry the VECIN "
        "position of the GM -> UB DataCopyPad contract"
    )
    assert "(n - nStart) * kITile" not in kernel_source, (
        f"{kernel_path}: the staging index still advances by kITile (16 bytes) "
        "instead of the 32-byte aligned qStageStride"
    )
    assert kernel_source.count("this->qStageStride") >= 4, (
        f"{kernel_path}: fewer than four uses of qStageStride, so the staging "
        "allocation and both index computations are not all stride-based"
    )

    torch.manual_seed(4242)
    module = BiasedAxialAttention(
        d_pair=16, d_bias=16, n_head=2, d_hidden=4, p_drop=0.0, is_row=True
    )
    module.eval().to(npu_device)
    generator = torch.Generator().manual_seed(777)
    length = 9
    pair = torch.randn(1, length, length, 16, generator=generator).to(npu_device)
    bias = torch.randn(1, length, length, 16, generator=generator).to(npu_device)

    calls = []
    real = native_attention.tied_attention_logits

    def recording(query, key, stream=None):
        # Records the operands of the real call and returns its real result
        # (no output substitution, no extra computation of the candidate).
        result = real(query, key, stream)
        calls.append(
            {
                "query": query.detach().clone(),
                "key": key.detach().clone(),
                "npu_result": result.detach().clone(),
                "query_device": str(query.device),
                "key_device": str(key.device),
                "result_device": str(result.device),
                "query_shape": list(query.shape),
                "key_shape": list(key.shape),
                "result_shape": list(result.shape),
                "query_dtype": str(query.dtype),
            }
        )
        return result

    native_attention.tied_attention_logits = recording
    try:
        with torch.no_grad():
            module(pair, bias)
        torch.npu.synchronize()
    finally:
        native_attention.tied_attention_logits = real

    assert len(calls) == 1, (
        "BiasedAxialAttention.forward must execute exactly one tied-attention "
        f"contraction, the dispatcher was called {len(calls)} time(s)"
    )
    call = calls[0]
    assert call["query_device"].startswith("npu"), (
        f"the call site passed a non-NPU query ({call['query_device']})"
    )
    assert call["key_device"].startswith("npu") and call[
        "result_device"
    ].startswith("npu"), (
        "the call site did not execute on the NPU: "
        f"key={call['key_device']} result={call['result_device']}"
    )
    expected_shape = [1, length, length, 2, 4]  # [B, N, I|J, H, K]
    assert call["query_shape"] == expected_shape and call["key_shape"] == expected_shape, (
        f"unexpected contraction operands {call['query_shape']} / "
        f"{call['key_shape']}, expected {expected_shape}"
    )
    assert call["query_dtype"] == "torch.float32"

    # Independent reference: the upstream expression on CPU over the very
    # operands the call site handed to the native adapter.
    expected = contract(
        _EQUATION, call["query"].cpu(), call["key"].cpu()
    )
    metrics = _tolerances(call["npu_result"], expected)
    _EVIDENCE["production_call_site"] = {
        "call_site": "BiasedAxialAttention.forward (is_row=True)",
        "source": str(source_path),
        "dispatch_chain": (
            "rfantibody.attention_backend.tied_attention_logits -> "
            "rfantibody.ascend_ops.attention.tied_attention_logits -> "
            "torch.ops.ascend_graph.tied_attention_logits"
        ),
        "module": {
            "batch": 1,
            "length": length,
            "d_pair": 16,
            "d_bias": 16,
            "n_head": 2,
            "d_hidden": 4,
        },
        "recorded_calls": len(calls),
        "recorded_operands": {
            "query_shape": call["query_shape"],
            "key_shape": call["key_shape"],
            "result_shape": call["result_shape"],
            "dtype": call["query_dtype"],
            "query_device": call["query_device"],
            "result_device": call["result_device"],
        },
        "reference": (
            "opt_einsum.contract('bnihk,bnjhk->bijh') on CPU over the recorded "
            "operands (independent path, no candidate output involved)"
        ),
        "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        **metrics,
    }
    assert metrics["all_finite"]
    assert metrics["elements_outside_atol_plus_rtol"] == 0, (
        "the contraction executed by the production call site deviates from the "
        f"independent CPU contraction by {metrics['max_abs_error']!r} "
        f"({metrics['elements_outside_atol_plus_rtol']} element(s) outside "
        f"atol={PARITY_ATOL} + rtol={PARITY_RTOL})"
    )


# --------------------------------------------------------------------------- #
# 6. Hardware route evidence
# --------------------------------------------------------------------------- #
def test_tied_attention_native_route_evidence(native_graph_ops, npu_device, tmp_path):
    """The public API executed a hardware ``tied_attention_logits`` kernel."""
    del native_graph_ops, npu_device  # the route cases bring their own device

    hardware = route_evidence._run_measurement(
        "hardware", _ROUTE_CASE, tmp_path / "hardware"
    )
    report = route_evidence._run_measurement(
        "report", _ROUTE_CASE, tmp_path / "report"
    )
    positive = route_evidence._run_measurement(
        "hardware",
        route_evidence.POSITIVE_CONTROL_CASE,
        tmp_path / "positive_instrumentation_control",
    )
    negative = route_evidence._run_measurement(
        "control",
        route_evidence.NEGATIVE_CONTROL_CASE,
        tmp_path / "negative_instrumentation_control",
    )

    artifacts = report.get("profile_artifacts") or {}
    resolved = route_evidence.resolve_token(artifacts, EXPECTED_KERNEL)
    op_names = list(report.get("op_names") or [])
    kernel_names = list(report.get("kernel_names") or [])
    ai_core_rows = report.get("ai_core_rows") or {}
    op_rows = [
        row
        for row in (ai_core_rows.get("op_summary") or [])
        if str(row.get("Op Name", "")) == EXPECTED_KERNEL
    ]
    task_rows = [
        row
        for row in (ai_core_rows.get("task_time") or [])
        if str(row.get("kernel_name", "")) == EXPECTED_KERNEL
    ]

    record = {
        "case": _ROUTE_CASE,
        "candidate": (
            "rfantibody.attention_backend.tied_attention_logits on npu:0 -> "
            "ascend_graph::tied_attention_logits -> native AscendC kernel "
            "tied_attention_logits"
        ),
        "isolation": (
            "one fresh interpreter per measurement "
            "(route_evidence._run_measurement -> "
            "python -m test.graph_parity.route_evidence --mode hardware|report|control)"
        ),
        "hardware_channel": {
            "profiler_level": hardware.get("profiler_level"),
            "hardware_event_count": hardware.get("hardware_event_count"),
            "task_counts": hardware.get("task_counts"),
            "events": hardware.get("events"),
            "trace": hardware.get("trace"),
            "command": hardware.get("command"),
            "exit_code": hardware.get("exit_code"),
            "log": hardware.get("log"),
        },
        "report_channel": {
            "profiler_level": report.get("profiler_level"),
            "op_names": op_names,
            "kernel_names": kernel_names,
            "ai_core_op_summary_rows_for_kernel": len(op_rows),
            "ai_core_task_time_rows_for_kernel": len(task_rows),
            "ai_core_op_summary_rows": len(ai_core_rows.get("op_summary") or []),
            "ai_core_task_time_rows": len(ai_core_rows.get("task_time") or []),
            "resolved": {
                "names": resolved["names"],
                "channels": resolved["channels"],
                "rows": resolved["rows"],
            },
            "negative_token": report.get("negative_token"),
            "negative_resolved": report.get("negative_resolved"),
            "trace": report.get("trace"),
            "command": report.get("command"),
            "exit_code": report.get("exit_code"),
            "log": report.get("log"),
        },
        "positive_instrumentation_control": {
            "case": route_evidence.POSITIVE_CONTROL_CASE,
            "hardware_event_count": positive.get("hardware_event_count"),
            "task_counts": positive.get("task_counts"),
            "command": positive.get("command"),
            "exit_code": positive.get("exit_code"),
            "log": positive.get("log"),
        },
        "negative_instrumentation_control": {
            "case": route_evidence.NEGATIVE_CONTROL_CASE,
            "hardware_event_count": negative.get("hardware_event_count"),
            "task_counts": negative.get("task_counts"),
            "command": negative.get("command"),
            "exit_code": negative.get("exit_code"),
            "log": negative.get("log"),
        },
        "written_utc": _utc_now(),
    }
    _EVIDENCE["route"] = record
    _EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
    (_EVIDENCE_PATH / "tied_attention_route_evidence.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Channel 1: the public call window issues at least one hardware task.
    hardware_count = int(hardware.get("hardware_event_count") or 0)
    assert hardware_count >= 1, (
        "the probe-identical profiler channel of the public call recorded no "
        f"KERNEL_* hardware task event (task counts: "
        f"{json.dumps(hardware.get('task_counts'), sort_keys=True)}, trace: "
        f"{hardware.get('trace')}, log: {hardware.get('log')})"
    )
    assert all(
        str(key).split("|")[-1].startswith("KERNEL_")
        for key in (hardware.get("task_counts") or {})
    ), f"non-hardware task type counted: {hardware.get('task_counts')}"

    # Channel 2: the vendor report names the kernel of this call exactly once.
    assert EXPECTED_KERNEL in op_names, (
        f"op_summary AI_CORE names are {op_names}, expected {EXPECTED_KERNEL!r}"
    )
    assert EXPECTED_KERNEL in kernel_names, (
        f"task_time AI_CORE names are {kernel_names}, expected {EXPECTED_KERNEL!r}"
    )
    assert len(op_rows) == 1, (
        f"expected exactly one AI_CORE op_summary row named {EXPECTED_KERNEL!r} "
        f"for the single native launch, got {len(op_rows)}"
    )
    assert len(task_rows) == 1, (
        f"expected exactly one AI_CORE task_time row named {EXPECTED_KERNEL!r} "
        f"for the single native launch, got {len(task_rows)}"
    )
    assert resolved["names"], (
        f"kernel name {EXPECTED_KERNEL!r} did not resolve in the raw vendor "
        f"artefacts: {artifacts}"
    )
    assert report.get("negative_resolved") == [], (
        f"negative control token {report.get('negative_token')!r} resolved to "
        f"{report.get('negative_resolved')}"
    )

    # Instrumentation controls: the harness sees native AscendC launches and no
    # event for a window without device work.
    assert int(positive.get("hardware_event_count") or 0) >= 1, (
        "the positive instrumentation control (a separate native AscendC kernel) "
        "recorded no hardware event, so the harness itself is not measuring "
        f"native launches (log: {positive.get('log')})"
    )
    assert int(negative.get("hardware_event_count") or 0) == 0, (
        "the negative instrumentation control (a CPU-only window) recorded "
        f"{negative.get('hardware_event_count')} hardware event(s), so a counted "
        f"event is not attributable to device work (log: {negative.get('log')})"
    )
