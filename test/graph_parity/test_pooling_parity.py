"""Native AscendC global pooling (avg/max): route artifacts, tiny execution,
CPU-DGL parity and hardware/report route evidence.

Capabilities under test - the last two operators of the consumed DGL surface:

* ``ascend_graph::avg_pooling`` -> native AscendC kernel ``avg_pooling``, i.e.
  ``dgl.nn.AvgPooling()(graph, feat)`` (``dgl.readout_nodes(op='mean')`` ->
  ``dgl.ops.segment.segment_reduce``),
* ``ascend_graph::max_pooling`` -> native AscendC kernel ``max_pooling``, i.e.
  ``dgl.nn.MaxPooling()(graph, feat)`` (``dgl.readout_nodes(op='max')``).

Fixed, pre-declared criteria (declared here, before any candidate result is
inspected, and never relaxed afterwards):

* the compiled artifact for the observed SoC exists and exports the generated
  ``aclrtlaunch_avg_pooling`` / ``aclrtlaunch_max_pooling`` entry points,
  ``ascend_graph::avg_pooling`` and ``ascend_graph::max_pooling`` are registered
  for dispatch key ``PrivateUse1``, a CPU tensor is rejected (no silent device
  fallback) and a zero-row feature tensor raises ``ValueError``,
* a hand-derived tiny case (N=5, W=3, ``seglen=[2,0,3]`` with a zero-length
  segment) is executed by the native kernels on ``npu:0`` and compared with a
  closed-form pooling expectation evaluated in double precision (``math.fsum`` /
  ``max``) - independent of both DGL and the candidate - at ``TINY_ATOL = 1e-6``;
  a zero-length segment must yield exactly zero,
* ``ascend_ops.avg_pooling`` / ``ascend_ops.max_pooling`` on ``npu:0`` match
  **real DGL 1.1.3 on CPU** built from the same feature and topology arrays - an
  independent reference, never the candidate.  Single-graph and ``dgl.batch``
  cases use ``dgl.nn.AvgPooling``/``dgl.nn.MaxPooling``; the batched case that
  contains a zero-node graph uses ``dgl.ops.segment.segment_reduce`` (the exact
  function ``readout_nodes`` calls) because a zero-node graph is not
  representable as a ``DGLGraph``.  Cases cover ordinary widths, the production
  ``GPooling`` shape ``[N, 32, 1]``, non-32-byte-aligned widths (C=1, C=13),
  repeated values around +1000/-1000, a batched multi-graph input, a batched
  input with a zero-length segment, a single-node graph and the topology of the
  production ``make_topk_graph``.

Operator tolerances: ``atol=1e-5``, ``rtol=1e-4`` (float32) for both variants;
max pooling is additionally required to be bit-exact (``torch.equal``) because it
is a pure selection with no arithmetic.  All outputs must be finite.  Passing
this operator parity does not by itself establish full-model parity.

The route test uses the vendor's two measured channels for the same public-API
bundle (each measurement in a fresh interpreter through
``route_evidence._run_measurement``): exactly two ``KERNEL_*`` hardware task
events for the two native launches (``avg_pooling`` + ``max_pooling``), and
``AI_CORE`` report rows whose names are exactly those two kernels, each token
resolved from the raw vendor artefacts with an empty negative-token resolution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import dgl
import numpy as np
import pytest
import torch
from dgl.nn import AvgPooling, MaxPooling
from dgl.ops.segment import segment_reduce

from test.graph_parity import route_evidence

# --- Pre-declared tolerances (fixed before candidate execution) --------------
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4
# Max pooling is a pure selection: its result must be identical to DGL's.
MAX_MODE_EXACT_REQUIRED = True
# Tiny hand-derived case: double-precision closed form vs float32 kernel.
TINY_ATOL = 1e-6
ALL_FINITE_CHECK = "all outputs finite"

_MODES = ("avg", "max")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_PATH = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "migration_reports")
) / "graph_parity"

_ROUTE_CASE = "test.graph_parity.route_case:run_pooling"
_EXPECTED_LAUNCHES = 2
_EXPECTED_KERNEL_NAMES = ("avg_pooling", "max_pooling")

_EVIDENCE = {
    "capability": (
        "native DGL global pooling: dgl.nn.AvgPooling (readout_nodes op='mean') "
        "and dgl.nn.MaxPooling (readout_nodes op='max') over an int64 segment "
        "table"
    ),
    "candidate": (
        "ascend_graph::avg_pooling -> native avg_pooling kernel and "
        "ascend_graph::max_pooling -> native max_pooling kernel, both executed "
        "on npu:0 through the public rfantibody.ascend_ops API"
    ),
    "reference": (
        "dgl {} on CPU: dgl.graph((src, dst), num_nodes=N) + dgl.nn.AvgPooling()/"
        "dgl.nn.MaxPooling()(graph, feat), or dgl.batch([...]) + the same pooling "
        "modules; the graph topology and the features are built from the same "
        "arrays (independent reference, never the candidate)"
    ).format(dgl.__version__),
    "tolerances": {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "max_mode_exact_required": MAX_MODE_EXACT_REQUIRED,
        "tiny_closed_form_atol": TINY_ATOL,
        "max_abs_error_definition": "max |candidate - reference| over all elements",
        "max_rel_error_definition": (
            "max |candidate - reference| / max(|reference|, 1e-12) over all elements"
        ),
        "assertion": (
            "torch.testing.assert_close(candidate, reference, atol=1e-5, "
            "rtol=1e-4); max mode additionally torch.equal; all outputs finite"
        ),
        "declared_before_execution": True,
        "all_finite_check": ALL_FINITE_CHECK,
    },
    "semantics": {
        "avg": (
            "out[b] = sum_{i in segment b} feat[i] / clamp(len_b, 1, N) in "
            "float32 (dgl.ops.segment.segment_reduce reducer='mean')"
        ),
        "max": (
            "out[b] = max_{i in segment b} feat[i]; an empty segment is DGL's "
            "replace_inf_with_zero result, i.e. exactly 0 and never -inf"
        ),
        "zero_length_segment": "exactly 0.0 for both variants",
        "one_graph_or_batch": (
            "one graph -> [1, ...]; a dgl.batch of B graphs -> [B, ...]; trailing "
            "feature dims are preserved (GPooling passes [N, F, 1])"
        ),
    },
    "tiny_native_execution": {},
    "artifacts": {},
    "cases": [],
    "route": {},
}


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #
def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    return _sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _errors(candidate: torch.Tensor, reference: torch.Tensor):
    """Exact error metrics of the candidate/reference float32 tensors."""
    cand = candidate.detach().cpu()
    ref = reference.detach().cpu()
    diff = (cand.double() - ref.double()).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    if ref.numel():
        rel = diff / ref.double().abs().clamp_min(1e-12)
        max_rel = float(rel.max())
        allowed = PARITY_ATOL + PARITY_RTOL * ref.double().abs()
        outside = int((diff > allowed).sum())
    else:
        max_rel = 0.0
        outside = 0
    return cand, ref, max_abs, max_rel, outside


def _all_finite(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    return bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())


def _record_case(case, mode, candidate, reference, reference_api, outside):
    """Append one machine-readable parity row and persist its exact inputs."""
    out_dir = _EVIDENCE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = case["feat"]
    seglen = list(case["seglen"])
    cand, ref, max_abs, max_rel, _ = _errors(candidate, reference)
    exact = bool(torch.equal(cand, ref))
    width = int(np.prod(feat.shape[1:], dtype=np.int64)) if feat.dim() > 1 else 1

    npz_path = out_dir / "pooling_{}_{}.npz".format(case["name"], mode)
    np.savez(
        npz_path,
        feat=feat.numpy(),
        seglen=np.asarray(seglen, dtype=np.int64),
        candidate=cand.numpy(),
        reference=ref.numpy(),
    )

    zero_segments = [int(index) for index, length in enumerate(seglen) if length == 0]
    zero_rows_exact = all(
        bool(torch.count_nonzero(cand[index]) == 0) for index in zero_segments
    )
    passed = bool(
        _all_finite(cand, ref)
        and outside == 0
        and zero_rows_exact
        and (exact or not (mode == "max" and MAX_MODE_EXACT_REQUIRED))
    )

    _EVIDENCE["cases"].append(
        {
            "case": case["name"],
            "mode": mode,
            "operator": "avg_pooling" if mode == "avg" else "max_pooling",
            "dgl_operator": "AvgPooling" if mode == "avg" else "MaxPooling",
            "description": case["description"],
            "provenance": case.get("provenance", {}),
            "shapes": {
                "feat": list(feat.shape),
                "seglen": [len(seglen)],
                "candidate": list(candidate.shape),
                "reference": list(reference.shape),
                "num_rows": int(feat.shape[0]),
                "num_segments": len(seglen),
                "flattened_width": int(width),
            },
            "dtypes": {
                "feat": str(feat.dtype),
                "seglen": "torch.int64",
                "candidate": str(candidate.dtype),
                "reference": str(reference.dtype),
            },
            "devices": {
                "candidate": str(candidate.device),
                "reference": str(reference.device),
                "reference_identity": (
                    "dgl {} on CPU; real DGL, never the candidate".format(
                        dgl.__version__
                    )
                ),
            },
            "reference_api": reference_api,
            "seglen": seglen,
            "input_sha256": {
                "feat": _tensor_digest(feat),
                "seglen": _sha256_bytes(
                    np.asarray(seglen, dtype=np.int64).tobytes()
                ),
            },
            "non_32_byte_aligned": bool((width * 4) % 32 != 0),
            "zero_length_segment": bool(zero_segments),
            "zero_length_segments": zero_segments,
            "zero_length_rows_exactly_zero": bool(zero_rows_exact),
            "batched": bool(len(seglen) > 1),
            "model_like_shape": bool(case.get("model_like", False)),
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "elements_outside_atol_plus_rtol": outside,
            "exact_match": exact,
            "all_finite": _all_finite(cand, ref),
            "tolerances": {
                "atol": PARITY_ATOL,
                "rtol": PARITY_RTOL,
                "exact_required": bool(mode == "max" and MAX_MODE_EXACT_REQUIRED),
            },
            "passed": passed,
            "inputs_npz": str(npz_path),
        }
    )


@pytest.fixture(scope="module", autouse=True)
def _flush_evidence():
    """Write the operator-parity table even if a case fails."""
    yield
    if not _EVIDENCE["cases"] and not _EVIDENCE["route"]:
        return
    _EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
    _EVIDENCE["generated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    (_EVIDENCE_PATH / "pooling.json").write_text(
        json.dumps(_EVIDENCE, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Reference construction (real DGL 1.1.3 on CPU)
# --------------------------------------------------------------------------- #
def _reference_graph(src, dst, num_nodes):
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    assert int(graph.num_nodes()) == int(num_nodes), (
        f"reference graph has {int(graph.num_nodes())} nodes, expected "
        f"num_nodes={num_nodes}"
    )
    ref_src, ref_dst = graph.edges()
    assert torch.equal(ref_src, src), "DGL graph source edge order changed"
    assert torch.equal(ref_dst, dst), "DGL graph destination edge order changed"
    return graph


def _reference_pool(case, mode):
    """Real DGL 1.1.3 CPU reference for one case/mode.

    Returns ``(reference, seglen_from_reference, reference_api)``.  The segment
    table is taken from DGL's own graph metadata (``batch_num_nodes()``) whenever
    the case is representable as a graph; only the batched case that contains a
    zero-node graph falls back to ``dgl.ops.segment.segment_reduce``, the exact
    function ``readout_nodes`` dispatches to.
    """
    reducer = "mean" if mode == "avg" else "max"
    pool = AvgPooling() if mode == "avg" else MaxPooling()
    pool_name = "AvgPooling" if mode == "avg" else "MaxPooling"
    graphs = [_reference_graph(src, dst, nodes) for (src, dst, nodes) in case["graphs"]]

    if case["reference_kind"] == "graph" and len(graphs) == 1:
        graph = graphs[0]
        reference = pool(graph, case["feat"])
        seglen = graph.batch_num_nodes()
        api = "dgl {} dgl.nn.{}(graph, feat) on a single dgl.graph".format(
            dgl.__version__, pool_name
        )
    elif case["reference_kind"] == "graph":
        graph = dgl.batch(graphs)
        reference = pool(graph, case["feat"])
        seglen = graph.batch_num_nodes()
        api = (
            "dgl {} dgl.batch([...]) + dgl.nn.{}(batched_graph, feat)".format(
                dgl.__version__, pool_name
            )
        )
    else:
        seglen = torch.tensor(case["seglen"], dtype=torch.int64)
        reference = segment_reduce(seglen, case["feat"], reducer=reducer)
        api = (
            "dgl {} dgl.ops.segment.segment_reduce(seglen, feat, reducer='{}') - "
            "the function dgl.readout_nodes calls; used because a zero-node graph "
            "is not representable as a DGLGraph".format(dgl.__version__, reducer)
        )

    assert seglen.dtype == torch.int64, f"reference seglen dtype is {seglen.dtype}"
    assert list(seglen.tolist()) == list(case["seglen"]), (
        f"{case['name']}: DGL metadata segment table {seglen.tolist()} != declared "
        f"{case['seglen']}"
    )
    return reference, seglen, api


# --------------------------------------------------------------------------- #
# Deterministic case construction
# --------------------------------------------------------------------------- #
def _edges(num_edges, num_nodes, seed):
    gen = torch.Generator().manual_seed(seed)
    if num_edges == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty.clone()
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    return src, dst


def _quantized(num_rows, width, seed, scale=1.0, offset=0.0):
    """Deterministic float32 features with exact binary fractions (no -0.0)."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.randint(
        -2048, 2049, (num_rows, width), generator=gen, dtype=torch.int64
    )
    return (values.to(torch.float32) / 8.0) * float(scale) + float(offset)


def _case(name, description, feat, seglen, graphs, reference_kind="graph",
          model_like=False, provenance=None):
    assert feat.dtype == torch.float32, f"{name}: feat dtype {feat.dtype}"
    assert feat.dim() >= 1, f"{name}: feat must have at least 1 dim"
    assert len(seglen) >= 1, f"{name}: need at least one segment"
    assert int(sum(seglen)) == int(feat.shape[0]), (
        f"{name}: seglen sums to {sum(seglen)} but feat has {int(feat.shape[0])} rows"
    )
    for src, dst, nodes in graphs:
        assert src.dtype == torch.int64 and dst.dtype == torch.int64
        assert src.dim() == 1 and dst.dim() == 1 and src.shape == dst.shape
        if src.numel():
            assert 0 <= int(src.min()) and int(src.max()) < int(nodes), (
                f"{name}: source id outside [0, {nodes})"
            )
            assert 0 <= int(dst.min()) and int(dst.max()) < int(nodes), (
                f"{name}: destination id outside [0, {nodes})"
            )
    return {
        "name": name,
        "description": description,
        "feat": feat,
        "seglen": [int(length) for length in seglen],
        "graphs": graphs,
        "reference_kind": reference_kind,
        "model_like": model_like,
        "provenance": provenance or {},
    }


def _case_ordinary_c8():
    num_rows, width = 96, 8
    src, dst = _edges(192, num_rows, 9101)
    feat = _quantized(num_rows, width, 9201)
    return _case(
        "ordinary_c8",
        "ordinary pooling over one graph, ordinary W=8 features",
        feat,
        [num_rows],
        [(src, dst, num_rows)],
    )


def _case_model_like_c32():
    num_rows, width = 64, 32
    src, dst = _edges(128, num_rows, 9102)
    feat = _quantized(num_rows, width, 9202).reshape(num_rows, width, 1)
    return _case(
        "model_like_c32",
        "production GPooling shape: [N, 32, 1] node features pooled to [1, 32, 1]",
        feat,
        [num_rows],
        [(src, dst, num_rows)],
        model_like=True,
    )


def _case_unaligned_c13():
    num_rows, width = 50, 13
    src, dst = _edges(90, num_rows, 9103)
    feat = _quantized(num_rows, width, 9203)
    return _case(
        "unaligned_c13",
        "non-32-byte-aligned width W=13 (52-byte rows)",
        feat,
        [num_rows],
        [(src, dst, num_rows)],
    )


def _case_unaligned_c1():
    num_rows, width = 33, 1
    src, dst = _edges(40, num_rows, 9104)
    feat = _quantized(num_rows, width, 9204)
    return _case(
        "unaligned_c1",
        "non-32-byte-aligned width W=1 (4-byte rows)",
        feat,
        [num_rows],
        [(src, dst, num_rows)],
    )


def _case_signed_large_values():
    pattern = torch.tensor(
        [
            [1000.0, -1000.0, 0.5, -0.5],
            [1000.0, -1000.0, -0.5, 0.5],
            [999.5, -999.5, 2.0, -2.0],
            [-3.0, 3.0, 1000.0, -1000.0],
            [-4.0, 4.0, -1000.0, 1000.0],
            [1.0, -1.0, 0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    feat = pattern.repeat(12, 1)
    num_rows = int(feat.shape[0])
    src, dst = _edges(100, num_rows, 9105)
    return _case(
        "signed_large_values",
        "repeated constant rows, negatives and +/-1000 magnitudes in one "
        "segment, W=4",
        feat,
        [num_rows],
        [(src, dst, num_rows)],
    )


def _case_batched_three_graphs():
    seglen = [7, 5, 9]
    num_rows = sum(seglen)
    width = 6
    graphs = []
    for index, (nodes, seed) in enumerate(zip(seglen, (9106, 9107, 9108))):
        edges = [12, 8, 15][index]
        src, dst = _edges(edges, nodes, seed)
        graphs.append((src, dst, nodes))
    feat = _quantized(num_rows, width, 9206)
    return _case(
        "batched_three_graphs",
        "dgl.batch of three graphs with seglen [7, 5, 9] (W=6): one output row "
        "per graph in the batch",
        feat,
        seglen,
        graphs,
        provenance={"batch_num_nodes": seglen},
    )


def _case_batched_with_empty_graph():
    seglen = [4, 0, 6]
    num_rows = sum(seglen)
    width = 5
    first = _edges(8, 4, 9109)
    third = _edges(12, 6, 9110)
    feat = _quantized(num_rows, width, 9207)
    return _case(
        "batched_with_empty_graph",
        "dgl.batch of [4, 0, 6] nodes: the zero-length middle segment must pool "
        "to exactly 0 for both variants (W=5)",
        feat,
        seglen,
        [
            (first[0], first[1], 4),
            (third[0], third[1], 6),
        ],
        reference_kind="segment",
        provenance={
            "batch_num_nodes": seglen,
            "reference_note": (
                "a zero-node graph cannot be represented as a DGLGraph, so the "
                "reference is dgl.ops.segment.segment_reduce - the exact function "
                "AvgPooling/MaxPooling reach through dgl.readout_nodes"
            ),
        },
    )


def _case_single_node_graph():
    width = 4
    feat = torch.tensor([[1.5, -2.5, 0.25, 3.0]], dtype=torch.float32)
    src = torch.zeros(1, dtype=torch.int64)
    dst = torch.zeros(1, dtype=torch.int64)
    return _case(
        "single_node_graph",
        "degenerate single-node graph (N=1, W=4): the pooled row is the node "
        "itself for both variants",
        feat,
        [1],
        [(src, dst, 1)],
    )


def _case_real_model_topk():
    """Production topology and the production GPooling feature shape."""
    from rfantibody.rf2.network.util_module import make_topk_graph

    batch, length = 1, 120
    positions = torch.arange(length, dtype=torch.float32)
    xyz = torch.stack(
        [
            9.0 * torch.sin(0.31 * positions) + 0.05 * positions,
            9.0 * torch.cos(0.27 * positions) - 0.03 * positions,
            2.0 * torch.sin(0.13 * positions + 0.7),
        ],
        dim=-1,
    ).reshape(batch, length, 3)
    pair = torch.zeros(batch, length, length, 8, dtype=torch.float32)
    residue_index = torch.arange(length, dtype=torch.int64).reshape(batch, length)
    graph, _ = make_topk_graph(xyz, pair, residue_index)
    src, dst = graph.edges()
    num_rows = int(graph.num_nodes())
    feat = _quantized(num_rows, 32, 9208).reshape(num_rows, 32, 1)
    return _case(
        "real_model_topk",
        "topology from rfantibody.rf2.network.util_module.make_topk_graph "
        "(kmin=32, top_k=128, B=1, L=120) with the production GPooling feature "
        "shape [N, 32, 1]",
        feat,
        [num_rows],
        [(src, dst, num_rows)],
        model_like=True,
        provenance={
            "topology_source": (
                "rfantibody.rf2.network.util_module.make_topk_graph (kmin=32, "
                "top_k=128, B=1, L=120)"
            ),
            "graph_num_nodes": num_rows,
            "graph_num_edges": int(src.numel()),
        },
    )


_CASES = [
    pytest.param(_case_ordinary_c8, id="ordinary_c8"),
    pytest.param(_case_model_like_c32, id="model_like_c32"),
    pytest.param(_case_unaligned_c13, id="unaligned_c13"),
    pytest.param(_case_unaligned_c1, id="unaligned_c1"),
    pytest.param(_case_signed_large_values, id="signed_large_values"),
    pytest.param(_case_batched_three_graphs, id="batched_three_graphs"),
    pytest.param(_case_batched_with_empty_graph, id="batched_with_empty_graph"),
    pytest.param(_case_single_node_graph, id="single_node_graph"),
    pytest.param(_case_real_model_topk, id="real_model_topk"),
]


# --------------------------------------------------------------------------- #
# Hand-derived closed form (double precision, independent of DGL and candidate)
# --------------------------------------------------------------------------- #
def _closed_form_pool(mode, rows, seglen):
    """Double-precision pooling expectation from ``math.fsum`` / ``max``."""
    width = len(rows[0])
    out = []
    row = 0
    for length in seglen:
        if length == 0:
            out.append([0.0] * width)
            continue
        segment = rows[row:row + length]
        row += length
        pooled = []
        for column in range(width):
            values = [entry[column] for entry in segment]
            if mode == "avg":
                pooled.append(math.fsum(values) / float(length))
            else:
                pooled.append(max(values))
        out.append(pooled)
    assert row == len(rows), f"seglen consumed {row} rows of {len(rows)}"
    return out


_TINY_SEGLEN = [2, 0, 3]  # the middle segment is empty
_TINY_FEAT = [
    [1.0, 2.0, 3.0],
    [3.0, 4.0, 5.0],
    [-1.0, 0.5, 2.5],
    [2.0, -2.0, 0.0],
    [4.0, 1.0, 1.0],
]


# --------------------------------------------------------------------------- #
# Artifact / dispatch / tiny execution
# --------------------------------------------------------------------------- #
def _dynamic_symbols(artifact: Path) -> str:
    nm = shutil.which("nm")
    if nm:
        completed = subprocess.run(
            [nm, "-D", "--defined-only", str(artifact)],
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        completed = subprocess.run(
            ["readelf", "--dyn-syms", "--wide", str(artifact)],
            capture_output=True,
            text=True,
            check=True,
        )
    return completed.stdout


def test_pooling_native_artifacts(native_graph_ops, npu_device):
    """Compiled artifact symbols, ``PrivateUse1`` kernels and no CPU fallback."""
    artifact = Path(native_graph_ops["artifact"])
    assert artifact.is_file(), f"compiled artifact missing: {artifact}"
    symbols = _dynamic_symbols(artifact)
    entries = ("aclrtlaunch_avg_pooling", "aclrtlaunch_max_pooling")
    for entry in entries:
        assert entry in symbols, f"generated route entry point missing: {entry}"
    for schema in ("ascend_graph::avg_pooling", "ascend_graph::max_pooling"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(schema, "PrivateUse1"), (
            f"{schema} has no PrivateUse1 kernel"
        )

    ops = native_graph_ops["ops"]
    cpu_feat = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32
    )
    cpu_seglen = torch.tensor([3], dtype=torch.int64)
    for name in ("avg_pooling", "max_pooling"):
        op = getattr(ops, name)
        with pytest.raises(RuntimeError):
            op(cpu_feat, cpu_seglen)
        with pytest.raises(ValueError):
            op(torch.zeros(0, 2), torch.tensor([0, 0], dtype=torch.int64))
        with pytest.raises(ValueError):
            op(torch.zeros(6, 2), torch.tensor([3, 3], dtype=torch.int32))

    _EVIDENCE["artifacts"] = {
        "kernel_library": str(artifact),
        "kernel_library_sha256": native_graph_ops["artifact_sha256"],
        "python_binding": native_graph_ops["binding"],
        "python_binding_sha256": native_graph_ops["binding_sha256"],
        "exported_entry_points": [entry for entry in entries if entry in symbols],
        "dispatch": "PrivateUse1",
        "device": str(npu_device),
        "cpu_tensors_rejected": True,
        "zero_row_features_rejected": True,
    }


@pytest.mark.parametrize("mode", _MODES)
def test_pooling_tiny_native_execution(native_graph_ops, npu_device, tmp_path, mode):
    """Hand-derived tiny case (N=5, W=3, ``seglen=[2,0,3]``) on the native kernels."""
    ops = native_graph_ops["ops"]
    op = ops.avg_pooling if mode == "avg" else ops.max_pooling
    feat = torch.tensor(_TINY_FEAT, dtype=torch.float32, device=npu_device)
    seglen = torch.tensor(_TINY_SEGLEN, dtype=torch.int64, device=npu_device)
    out = op(feat, seglen)
    torch.npu.synchronize()

    assert out.device.type == "npu", f"output device is {out.device}"
    assert tuple(out.shape) == (3, 3), (
        f"{mode}: output shape {tuple(out.shape)} != {(3, 3)}"
    )
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all(), f"{mode}: non-finite tiny output"

    expected = _closed_form_pool(mode, _TINY_FEAT, _TINY_SEGLEN)
    observed = out.cpu().tolist()
    for segment in range(len(_TINY_SEGLEN)):
        for column in range(3):
            difference = abs(observed[segment][column] - expected[segment][column])
            assert difference <= TINY_ATOL, (
                f"{mode} segment {segment} column {column}: observed "
                f"{observed[segment][column]!r} differs from the double-precision "
                f"closed form {expected[segment][column]!r} by {difference!r} > "
                f"{TINY_ATOL}"
            )
    # The zero-length segment must pool to exactly zero (DGL semantics).
    assert observed[1] == [0.0, 0.0, 0.0], (
        f"{mode}: the zero-length segment pooled to {observed[1]!r}, not exactly 0"
    )
    if mode == "max":
        assert observed == expected, (
            "max pooling is a pure selection: the tiny case must be bit-exact "
            f"against the closed form ({observed!r} != {expected!r})"
        )

    trace = route_evidence.collect_level0(
        lambda: op(feat, seglen),
        tmp_path / f"pooling_tiny_{mode}_hardware.json",
    )
    torch.npu.synchronize()

    _EVIDENCE["tiny_native_execution"][mode] = {
        "feat": _TINY_FEAT,
        "seglen": _TINY_SEGLEN,
        "num_rows": 5,
        "num_segments": 3,
        "zero_length_segment": 1,
        "shape": [3, 3],
        "closed_form": (
            "double-precision math.fsum mean / max destination pooling"
        ),
        "tolerance": TINY_ATOL,
        "expected_double": expected,
        "observed": observed,
        "exact_match_with_closed_form": bool(
            all(
                observed[segment][column] == expected[segment][column]
                for segment in range(len(_TINY_SEGLEN))
                for column in range(3)
            )
        ),
        "hardware_trace": trace,
    }


# --------------------------------------------------------------------------- #
# Parity against real DGL on CPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("case_factory", _CASES)
def test_pooling_parity_vs_cpu_dgl(native_graph_ops, npu_device, case_factory, mode):
    """``avg_pooling`` / ``max_pooling`` NPU kernels vs real DGL 1.1.3 on CPU."""
    case = case_factory()
    op = (
        native_graph_ops["ops"].avg_pooling
        if mode == "avg"
        else native_graph_ops["ops"].max_pooling
    )

    reference, seglen_check, reference_api = _reference_pool(case, mode)
    seglen = torch.tensor(case["seglen"], dtype=torch.int64, device=npu_device)
    candidate = op(case["feat"].to(npu_device), seglen)
    torch.npu.synchronize()

    assert candidate.device.type == "npu", (
        f"{case['name']}/{mode}: candidate device is {candidate.device}"
    )
    assert reference.device.type == "cpu", (
        f"{case['name']}/{mode}: reference device is {reference.device}"
    )
    assert tuple(candidate.shape) == tuple(reference.shape), (
        f"{case['name']}/{mode}: candidate shape {tuple(candidate.shape)} != "
        f"reference shape {tuple(reference.shape)}"
    )
    assert candidate.dtype == reference.dtype == torch.float32
    assert tuple(candidate.shape) == (len(case["seglen"]),) + tuple(case["feat"].shape[1:]), (
        f"{case['name']}/{mode}: output shape must be (num_segments,) + trailing"
    )
    assert tuple(seglen_check.shape) == (len(case["seglen"]),)
    assert sum(case["seglen"]) == int(case["feat"].shape[0])

    cand_cpu = candidate.cpu()
    assert torch.isfinite(cand_cpu).all(), (
        f"{case['name']}/{mode}: non-finite candidate output"
    )
    assert torch.isfinite(reference).all(), (
        f"{case['name']}/{mode}: non-finite reference output"
    )
    torch.testing.assert_close(
        cand_cpu, reference, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )

    if mode == "max" and MAX_MODE_EXACT_REQUIRED:
        assert torch.equal(cand_cpu, reference), (
            f"{case['name']}/{mode}: max pooling must reproduce DGL's selection "
            "bit-exactly; max |difference| = "
            f"{float((cand_cpu.double() - reference.double()).abs().max())!r}"
        )

    for segment, length in enumerate(case["seglen"]):
        if length == 0:
            assert int(torch.count_nonzero(cand_cpu[segment])) == 0, (
                f"{case['name']}/{mode}: zero-length segment {segment} pooled to "
                f"{cand_cpu[segment]!r}, not exactly zero"
            )

    _, _, _, _, outside = _errors(cand_cpu, reference)
    assert outside == 0, (
        f"{case['name']}/{mode}: {outside} element(s) outside atol={PARITY_ATOL} + "
        f"rtol={PARITY_RTOL} * |reference|"
    )

    _record_case(case, mode, candidate, reference, reference_api, outside)


# --------------------------------------------------------------------------- #
# Hardware / report route evidence
# --------------------------------------------------------------------------- #
def test_pooling_route_evidence(native_graph_ops, npu_device, tmp_path):
    """Two native launches, two ``KERNEL_*`` tasks, two report kernel names."""
    del native_graph_ops, npu_device  # device requirement is explicit via the case

    # One fresh interpreter per measurement (the module's isolation contract).
    hardware = route_evidence._run_measurement(
        "hardware", _ROUTE_CASE, tmp_path / "hardware"
    )
    report = route_evidence._run_measurement(
        "report", _ROUTE_CASE, tmp_path / "report"
    )

    artifacts = report.get("profile_artifacts") or {}
    resolved = {
        name: route_evidence.resolve_token(artifacts, name)
        for name in _EXPECTED_KERNEL_NAMES
    }

    op_names = list(report.get("op_names") or [])
    kernel_names = list(report.get("kernel_names") or [])
    ai_core_rows = report.get("ai_core_rows") or {}

    route_record = {
        "case": _ROUTE_CASE,
        "expected_launches": _EXPECTED_LAUNCHES,
        "expected_kernel_names": list(_EXPECTED_KERNEL_NAMES),
        "isolation": (
            "one fresh interpreter per measurement "
            "(route_evidence._run_measurement -> "
            "python -m test.graph_parity.route_evidence --mode hardware|report)"
        ),
        "hardware_channel": {
            "profiler_level": hardware.get("profiler_level"),
            "hardware_event_count": hardware.get("hardware_event_count"),
            "task_counts": hardware.get("task_counts"),
            "events": hardware.get("events"),
            "trace": hardware.get("trace"),
            "profile_artifacts": hardware.get("profile_artifacts"),
            "command": hardware.get("command"),
            "exit_code": hardware.get("exit_code"),
            "log": hardware.get("log"),
            "versions": hardware.get("versions"),
        },
        "report_channel": {
            "profiler_level": report.get("profiler_level"),
            "op_names": op_names,
            "kernel_names": kernel_names,
            "ai_core_op_summary_rows": len(ai_core_rows.get("op_summary") or []),
            "ai_core_task_time_rows": len(ai_core_rows.get("task_time") or []),
            "resolved": {
                name: {
                    "names": resolved[name]["names"],
                    "channels": resolved[name]["channels"],
                    "rows": resolved[name]["rows"],
                }
                for name in _EXPECTED_KERNEL_NAMES
            },
            "negative_token": report.get("negative_token"),
            "negative_resolved": report.get("negative_resolved"),
            "trace": report.get("trace"),
            "profile_artifacts": artifacts,
            "command": report.get("command"),
            "exit_code": report.get("exit_code"),
            "log": report.get("log"),
            "versions": report.get("versions"),
        },
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _EVIDENCE["route"] = route_record
    _EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
    (_EVIDENCE_PATH / "pooling_route_evidence.json").write_text(
        json.dumps(route_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Channel 1: exactly one hardware task per native launch.
    assert hardware.get("hardware_event_count") == _EXPECTED_LAUNCHES, (
        "expected exactly "
        f"{_EXPECTED_LAUNCHES} KERNEL_* hardware task events for the two native "
        f"launches, got {hardware.get('hardware_event_count')}: "
        f"{json.dumps(hardware.get('task_counts'), sort_keys=True)} "
        f"(trace: {hardware.get('trace')})"
    )
    assert hardware.get("task_counts"), "hardware channel recorded no task counts"
    assert all(
        str(key).split("|")[-1].startswith("KERNEL_")
        for key in hardware["task_counts"]
    ), f"non-hardware task type counted: {hardware['task_counts']}"

    # Channel 2: the vendor report names the two kernels of this bundle.
    assert set(op_names) == set(_EXPECTED_KERNEL_NAMES), (
        f"op_summary AI_CORE names are {op_names}, expected "
        f"{list(_EXPECTED_KERNEL_NAMES)}"
    )
    assert set(kernel_names) == set(_EXPECTED_KERNEL_NAMES), (
        f"task_time AI_CORE names are {kernel_names}, expected "
        f"{list(_EXPECTED_KERNEL_NAMES)}"
    )
    assert ai_core_rows.get("op_summary"), "no AI_CORE op_summary rows recorded"
    assert ai_core_rows.get("task_time"), "no AI_CORE task_time rows recorded"
    for name in _EXPECTED_KERNEL_NAMES:
        assert resolved[name]["names"], (
            f"kernel name {name!r} did not resolve in the raw vendor artefacts: "
            f"{artifacts}"
        )
        assert resolved[name]["channels"], (
            f"kernel name {name!r} resolved without a channel: {resolved[name]}"
        )
    assert report.get("negative_resolved") == [], (
        f"negative control token {report.get('negative_token')!r} resolved to "
        f"{report.get('negative_resolved')}"
    )
