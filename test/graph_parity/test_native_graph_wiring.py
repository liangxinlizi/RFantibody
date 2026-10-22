"""Production graph wiring: the native ``AscendGraph`` and the dispatch point.

Capability under test: the SE3-Transformer model reaches the graph through one
graph object and one dispatch point
(``se3_transformer.model.graph_backend``).  On ``npu:0`` that object is
:class:`rfantibody.ascend_ops.graph.AscendGraph`, built from the ``[E]`` int64
source/destination id arrays, and every graph operator of the consumed surface is
the compiled native AscendC kernel of the accepted iterations: ``edge_topology``
(eagerly at graph construction), ``edge_dot_v``, ``edge_softmax_dst``,
``copy_e_sum``, ``copy_e_mean``, ``avg_pooling`` and ``max_pooling``.  On a real
``dgl`` graph the same dispatch point keeps the original ``dgl.ops``/``dgl.nn``
implementation, which is also the independent reference of this module.

Fixed, pre-declared criteria (declared here, before any candidate result is
inspected, and never relaxed afterwards):

* metadata parity -- for several topologies (ordinary multiple destinations,
  duplicate edges, zero-in-degree nodes and a topology produced by the production
  ``rfantibody.rf2.network.util_module.make_topk_graph``) ``edges()``,
  ``num_nodes()``, ``num_edges()`` and ``in_degrees()`` of the native graph must
  equal **real DGL 1.1.3 on CPU** built from the same ``src``/``dst`` arrays
  exactly (``torch.equal``), the ``edata['rel_pos']`` round-trip must preserve
  the stored values, and ``batch_num_nodes()`` must be the int64 single-graph
  segment table DGL reports;
* dispatch parity -- the production functions ``graph_backend.e_dot_v``,
  ``.edge_softmax``, ``.copy_e_sum``, ``.copy_e_mean`` and ``.pool`` called with
  the native graph on ``npu:0`` must match the same operators on a real ``dgl``
  CPU graph holding the same arrays, at ``atol=1e-5``/``rtol=1e-4`` (float32);
  max pooling must additionally be bit-exact; every output must be finite; and
  every per-destination/per-column ``edge_softmax`` weight vector must sum to 1
  within ``1e-5``.  The cases include multiple heads, duplicate edges,
  zero-in-degree nodes, ordinary logits, destination groups around ``+1000`` and
  ``-1000``, the production attention/reduction shapes and non-32-byte-aligned
  widths.  The candidate is never its own reference;
* route -- one public-API ``AscendGraph`` construction in a fresh interpreter
  must produce exactly one ``KERNEL_*`` hardware task event and one ``AI_CORE``
  report row named ``edge_topology``, with the negative control token
  ``__no_such_kernel__`` resolving nowhere.

Passing this wiring test does not by itself establish full-model parity.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import dgl
import pytest
import torch

from test.graph_parity import route_evidence

# --- Pre-declared tolerances (fixed before candidate execution) --------------
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4
# Per-destination softmax weights must sum to one within this tolerance.
SOFTMAX_SUM_ATOL = 1e-5
# Max pooling is a pure selection: its result must be identical to DGL's.
POOL_MAX_EXACT_REQUIRED = True
ALL_FINITE_CHECK = "all outputs finite"

_GRAPH_BUILD_ROUTE_CASE = "test.graph_parity.route_case:run_graph_build"
_GRAPH_BUILD_EXPECTED_KERNELS = ("edge_topology",)
_GRAPH_BUILD_EXPECTED_EVENTS = 1

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_PATH = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "migration_reports")
) / "graph_parity"

_EVIDENCE = {
    "capability": (
        "production graph wiring: one native AscendGraph serving graph "
        "construction, edges/num_nodes/num_edges/edata/in_degrees and the "
        "SE3-Transformer graph operators through one dispatch point"
    ),
    "candidate": (
        "rfantibody.ascend_ops.graph.AscendGraph on npu:0 -> native AscendC "
        "kernels edge_topology, edge_dot_v, edge_softmax_dst, copy_e_sum, "
        "copy_e_mean, avg_pooling, max_pooling, called through the production "
        "se3_transformer.model.graph_backend dispatch functions"
    ),
    "reference": (
        "dgl {} on CPU: dgl.graph((src, dst), num_nodes=N) from the same src/dst "
        "arrays; metadata via DGLGraph.edges()/num_nodes()/num_edges()/"
        "in_degrees()/batch_num_nodes(); operators via dgl.ops.e_dot_v/"
        "edge_softmax(norm_by='dst')/copy_e_sum/copy_e_mean and "
        "dgl.nn.AvgPooling/MaxPooling (independent reference, never the "
        "candidate)"
    ).format(dgl.__version__),
    "tolerances": {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "softmax_per_destination_sum_atol": SOFTMAX_SUM_ATOL,
        "pool_max_mode_exact_required": POOL_MAX_EXACT_REQUIRED,
        "metadata_equality": "torch.equal against dgl 1.1.3 CPU (exact)",
        "max_abs_error_definition": "max |candidate - reference| over all elements",
        "max_rel_error_definition": (
            "max |candidate - reference| / max(|reference|, 1e-12) over all elements"
        ),
        "assertion": (
            "torch.testing.assert_close(candidate, reference, atol=1e-5, "
            "rtol=1e-4); max pooling additionally torch.equal; all outputs finite"
        ),
        "declared_before_execution": True,
        "all_finite_check": ALL_FINITE_CHECK,
    },
    "semantics": {
        "graph_construction": (
            "AscendGraph.__init__ validates the int64 [E] src/dst NPU arrays and "
            "eagerly builds the native CSR topology with one edge_topology launch"
        ),
        "copy_e_sum": (
            "out[i] = sum of feat over the edges whose dst == i "
            "(dgl.ops.copy_e_sum)"
        ),
        "copy_e_mean": (
            "out[i] = that sum / clamp(in_degree[i], 1, num_edges), so a "
            "zero-in-degree node yields zeros (DGL 1.1.3 copy_e_mean)"
        ),
        "edge_softmax": (
            "per destination and per trailing column: exp(x - max) / sum over the "
            "destination's incoming edges (DGL norm_by='dst')"
        ),
        "pooling": (
            "avg = sum / clamp(seglen, 1, N); max = selection with DGL "
            "replace_inf_with_zero; batch_num_nodes() is [N] for one graph"
        ),
        "drop_in": (
            "graph_backend falls back to dgl.ops/dgl.nn for any graph without the "
            "is_ascend_native_graph marker, so the CPU/CUDA reference paths are "
            "unchanged"
        ),
    },
    "artifacts": {},
    "metadata_cases": [],
    "dispatch_cases": [],
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


def _quantized(rows, width, seed):
    """Deterministic float32 values with exact binary fractions (no -0.0)."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.randint(
        -2048, 2049, (rows, width), generator=gen, dtype=torch.int64
    )
    return values.to(torch.float32) / 8.0


@pytest.fixture(scope="module", autouse=True)
def _flush_evidence():
    """Write the wiring evidence table even if a case fails."""
    yield
    if not _EVIDENCE["metadata_cases"] and not _EVIDENCE["dispatch_cases"]:
        if not _EVIDENCE["route"]:
            return
    _EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
    _EVIDENCE["generated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    (_EVIDENCE_PATH / "graph_wiring.json").write_text(
        json.dumps(_EVIDENCE, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Deterministic topologies
# --------------------------------------------------------------------------- #
def _topology(name, description, src, dst, num_nodes, provenance=None):
    src = torch.as_tensor(src, dtype=torch.int64).contiguous()
    dst = torch.as_tensor(dst, dtype=torch.int64).contiguous()
    assert src.dim() == 1 and dst.dim() == 1 and src.shape == dst.shape
    assert int(num_nodes) >= 1
    if dst.numel():
        assert 0 <= int(dst.min()) and int(dst.max()) < int(num_nodes), (
            f"{name}: destination id outside [0, {num_nodes})"
        )
        assert 0 <= int(src.min()) and int(src.max()) < int(num_nodes), (
            f"{name}: source id outside [0, {num_nodes})"
        )
    return {
        "name": name,
        "description": description,
        "src": src,
        "dst": dst,
        "num_nodes": int(num_nodes),
        "provenance": provenance or {},
    }


def _topology_ordinary():
    """Ordinary graph: several destinations, no duplicates, no isolated nodes."""
    return _topology(
        "ordinary_multiple_destinations",
        "8 edges over 4 destinations with ordinary in-degrees 2/2/2/2",
        [0, 0, 1, 1, 2, 2, 3, 3],
        [1, 2, 0, 3, 0, 3, 1, 2],
        4,
    )


def _topology_duplicate_edges():
    """Repeated duplicate edges: one destination receives the same edge three times."""
    return _topology(
        "duplicate_edges",
        "triple duplicate edge 0->1 plus duplicated 1->2 edges over 3 destinations",
        [0, 0, 0, 1, 1, 2],
        [1, 1, 1, 2, 2, 0],
        3,
    )


def _topology_zero_in_degree():
    """Zero-in-degree nodes: node 1 has no incoming edge, node 4 is isolated."""
    return _topology(
        "zero_in_degree",
        "5 edges over 5 nodes: node 1 has zero in-degree and node 4 is isolated",
        [0, 1, 2, 0, 3],
        [0, 2, 0, 3, 2],
        5,
    )


def _topology_model_derived():
    """Topology produced by the production ``make_topk_graph`` (B=1, L=120)."""
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
    num_nodes = int(graph.num_nodes())
    return _topology(
        "real_model_topk",
        "production make_topk_graph topology (top_k=128, kmin=32, B=1, L=120)",
        src,
        dst,
        num_nodes,
        provenance={
            "topology_source": (
                "rfantibody.rf2.network.util_module.make_topk_graph (top_k=128, "
                "kmin=32, B=1, L=120)"
            ),
            "graph_num_nodes": num_nodes,
            "graph_num_edges": int(src.numel()),
        },
    )


_TOPOLOGIES = [
    pytest.param(_topology_ordinary, id="ordinary_multiple_destinations"),
    pytest.param(_topology_duplicate_edges, id="duplicate_edges"),
    pytest.param(_topology_zero_in_degree, id="zero_in_degree"),
    pytest.param(_topology_model_derived, id="real_model_topk"),
]


# --------------------------------------------------------------------------- #
# Shared construction helpers
# --------------------------------------------------------------------------- #
def _native_graph(case, npu_device):
    from rfantibody.ascend_ops.graph import AscendGraph

    return AscendGraph(
        case["src"].to(npu_device), case["dst"].to(npu_device), case["num_nodes"]
    )


def _cpu_graph(case):
    """Real DGL 1.1.3 CPU graph from the same arrays (independent reference)."""
    graph = dgl.graph((case["src"], case["dst"]), num_nodes=case["num_nodes"])
    ref_src, ref_dst = graph.edges()
    assert torch.equal(ref_src, case["src"]), "DGL reordered the source edge array"
    assert torch.equal(ref_dst, case["dst"]), "DGL reordered the destination ids"
    return graph


def _record_metadata(
    case, graph, native_degrees, reference, rel_pos, edata_roundtrip_ok
):
    src, dst = graph.edges()
    batch_nodes = graph.batch_num_nodes()
    _EVIDENCE["metadata_cases"].append(
        {
            "case": case["name"],
            "description": case["description"],
            "provenance": case["provenance"],
            "shapes": {
                "src": list(src.shape),
                "dst": list(dst.shape),
                "rel_pos": list(rel_pos.shape),
                "in_degrees": list(native_degrees.shape),
                "batch_num_nodes": list(batch_nodes.shape),
            },
            "dtypes": {
                "src": str(src.dtype),
                "dst": str(dst.dtype),
                "in_degrees": str(native_degrees.dtype),
                "batch_num_nodes": str(batch_nodes.dtype),
            },
            "devices": {
                "candidate_graph": str(graph.device),
                "candidate_in_degrees": str(native_degrees.device),
                "reference_graph": str(reference.device),
            },
            "num_nodes": {
                "candidate": int(graph.num_nodes()),
                "reference": int(reference.num_nodes()),
            },
            "num_edges": {
                "candidate": int(graph.num_edges()),
                "reference": int(reference.num_edges()),
            },
            "in_degrees": native_degrees.detach().cpu().tolist(),
            "zero_in_degree_nodes": [
                node
                for node in range(int(reference.num_nodes()))
                if int(reference.in_degrees()[node]) == 0
            ],
            "input_sha256": {
                "src": _tensor_digest(case["src"]),
                "dst": _tensor_digest(case["dst"]),
            },
            "output_sha256": {
                "edges_src": _tensor_digest(src),
                "edges_dst": _tensor_digest(dst),
                "in_degrees": _tensor_digest(native_degrees),
                "reference_in_degrees": _tensor_digest(reference.in_degrees()),
            },
            "edges_exact_equal": bool(
                torch.equal(src.cpu(), reference.edges()[0])
                and torch.equal(dst.cpu(), reference.edges()[1])
            ),
            "in_degrees_exact_equal": bool(
                torch.equal(native_degrees.cpu(), reference.in_degrees())
            ),
            "batch_num_nodes_exact_equal": bool(
                torch.equal(batch_nodes.cpu(), reference.batch_num_nodes())
            ),
            "edata_roundtrip_exact_equal": bool(edata_roundtrip_ok),
            "non_32_byte_aligned_rel_pos": bool((3 * 4) % 32 != 0),
            "passed": True,
        }
    )


def _record_dispatch(case, op, candidate, reference, extra=None):
    cand, ref, max_abs, max_rel, outside = _errors(candidate, reference)
    exact = bool(torch.equal(cand, ref))
    exact_required = bool(op == "pool(max)" and POOL_MAX_EXACT_REQUIRED)
    finite = _all_finite(candidate, reference)
    sums_ok = bool((extra or {}).get("softmax_sums_ok", True))
    record = {
        "case": case["name"],
        "description": case["description"],
        "operator": op,
        "shapes": {
            "candidate": list(candidate.shape),
            "reference": list(reference.shape),
            "src": list(case["src"].shape),
            "dst": list(case["dst"].shape),
            "num_nodes": int(case["num_nodes"]),
        },
        "dtypes": {
            "candidate": str(candidate.dtype),
            "reference": str(reference.dtype),
        },
        "devices": {
            "candidate": str(candidate.device),
            "reference": str(reference.device),
            "reference_identity": (
                "dgl {} on CPU; real DGL, never the candidate".format(dgl.__version__)
            ),
        },
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "elements_outside_atol_plus_rtol": outside,
        "exact_match": exact,
        "exact_required": exact_required,
        "all_finite": finite,
        "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        "passed": bool(
            finite and outside == 0 and sums_ok and (exact or not exact_required)
        ),
    }
    record.update(extra or {})
    _EVIDENCE["dispatch_cases"].append(record)
    return record


def _logits_with_destination_groups(case, num_heads, seed):
    """Ordinary logits plus destination groups shifted by +1000 and -1000.

    Destination ``d`` is assigned ``d % 3``: ``0`` -> all its logits get
    ``+1000``, ``1`` -> ``-1000``, ``2`` -> ordinary values.  Inside a destination
    the logits still differ, so a destination's maximum is not a degenerate tie.
    """
    num_edges = int(case["dst"].numel())
    base = _quantized(num_edges, num_heads, seed)
    group = case["dst"].remainder(3)
    offset = torch.zeros(num_edges, 1, dtype=torch.float32)
    offset[group == 0] = 1000.0
    offset[group == 1] = -1000.0
    return base + offset


def _softmax_destination_sums(weights, dst, num_nodes):
    """Per-destination, per-column sums of the softmax weights (CPU)."""
    weights = weights.detach().cpu()
    dst = dst.detach().cpu()
    degrees = torch.bincount(dst, minlength=int(num_nodes))
    checked = 0
    worst = 0.0
    for node in range(int(num_nodes)):
        if int(degrees[node]) == 0:
            continue
        sums = weights[dst == node].double().sum(dim=0)
        worst = max(worst, float((sums - 1.0).abs().max()))
        checked += 1
    return checked, worst


# --------------------------------------------------------------------------- #
# 1. Graph metadata parity against real DGL on CPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case_factory", _TOPOLOGIES)
def test_ascend_graph_metadata_matches_dgl(native_graph_ops, npu_device, case_factory):
    """``edges``/``num_nodes``/``num_edges``/``in_degrees``/``edata`` parity."""
    case = case_factory()
    reference = _cpu_graph(case)
    graph = _native_graph(case, npu_device)
    torch.npu.synchronize()

    assert bool(getattr(graph, "is_ascend_native_graph", False)), (
        "AscendGraph does not expose the is_ascend_native_graph dispatch marker"
    )
    assert graph.device.type == "npu", f"graph device is {graph.device}"

    src, dst = graph.edges()
    assert src.device.type == "npu" and dst.device.type == "npu", (
        f"edges() returned non-NPU tensors: {src.device}, {dst.device}"
    )
    assert src.dtype == dst.dtype == torch.int64
    assert torch.equal(src.cpu(), reference.edges()[0]), (
        f"{case['name']}: edges() src differs from the DGL reference"
    )
    assert torch.equal(dst.cpu(), reference.edges()[1]), (
        f"{case['name']}: edges() dst differs from the DGL reference"
    )

    assert int(graph.num_nodes()) == int(reference.num_nodes())
    assert int(graph.num_edges()) == int(reference.num_edges())

    degrees = graph.in_degrees()
    assert degrees.device.type == "npu", f"in_degrees device is {degrees.device}"
    assert degrees.dtype == torch.int64
    assert tuple(degrees.shape) == (int(reference.num_nodes()),)
    assert torch.isfinite(degrees.float()).all()
    assert torch.equal(degrees.cpu(), reference.in_degrees()), (
        f"{case['name']}: native in_degrees {degrees.cpu().tolist()} != DGL "
        f"{reference.in_degrees().tolist()}"
    )

    num_edges = int(case["dst"].numel())
    rel_pos = _quantized(num_edges, 3, 5000 + len(case["name"]))
    graph.edata["rel_pos"] = rel_pos.to(npu_device)
    reference.edata["rel_pos"] = rel_pos.clone()
    assert "rel_pos" in graph.edata, (
        "edata membership check failed after storing rel_pos"
    )
    stored = graph.edata["rel_pos"]
    assert stored.device.type == "npu", f"stored rel_pos device is {stored.device}"
    assert tuple(stored.shape) == (num_edges, 3)
    edata_roundtrip_ok = bool(torch.equal(stored.cpu(), rel_pos))
    assert edata_roundtrip_ok, (
        "edata['rel_pos'] round-trip differs from the value that was stored"
    )

    batch_nodes = graph.batch_num_nodes()
    assert batch_nodes.device.type == "npu"
    assert batch_nodes.dtype == reference.batch_num_nodes().dtype == torch.int64
    assert torch.equal(batch_nodes.cpu(), reference.batch_num_nodes()), (
        f"{case['name']}: batch_num_nodes {batch_nodes.cpu().tolist()} != DGL "
        f"{reference.batch_num_nodes().tolist()}"
    )

    _EVIDENCE["artifacts"] = {
        "kernel_library": native_graph_ops["artifact"],
        "kernel_library_sha256": native_graph_ops["artifact_sha256"],
        "python_binding": native_graph_ops["binding"],
        "python_binding_sha256": native_graph_ops["binding_sha256"],
        "dispatch": "PrivateUse1",
        "graph_object": "rfantibody.ascend_ops.graph.AscendGraph",
        "dispatch_point": "se3_transformer.model.graph_backend",
        "device": str(npu_device),
    }
    _record_metadata(
        case, graph, degrees, reference, rel_pos, edata_roundtrip_ok
    )


# --------------------------------------------------------------------------- #
# 2. Production dispatch parity against real DGL on CPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case_factory", _TOPOLOGIES)
def test_graph_backend_dispatch_matches_dgl(native_graph_ops, npu_device, case_factory):
    """The production dispatch functions on ``AscendGraph`` vs real DGL on CPU."""
    del native_graph_ops  # artifact identity is recorded by the metadata test
    from se3_transformer.model import graph_backend

    case = case_factory()
    reference = _cpu_graph(case)
    graph = _native_graph(case, npu_device)

    num_edges = int(case["dst"].numel())
    num_nodes = int(case["num_nodes"])
    num_heads, num_feat = 2, 3

    # --- e_dot_v ---------------------------------------------------------- #
    key = _quantized(num_edges, num_heads * num_feat, 7001).reshape(
        num_edges, num_heads, num_feat
    )
    query = _quantized(num_nodes, num_heads * num_feat, 7002).reshape(
        num_nodes, num_heads, num_feat
    )
    candidate = graph_backend.e_dot_v(
        graph, key.to(npu_device), query.to(npu_device)
    )
    torch.npu.synchronize()
    expected = dgl.ops.e_dot_v(reference, key, query)
    assert candidate.device.type == "npu", f"e_dot_v output device {candidate.device}"
    assert expected.device.type == "cpu"
    assert tuple(candidate.shape) == tuple(expected.shape) == (
        num_edges,
        num_heads,
        1,
    ), f"e_dot_v shape {tuple(candidate.shape)} != {tuple(expected.shape)}"
    torch.testing.assert_close(
        candidate.cpu(), expected, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )
    _, _, _, _, outside = _errors(candidate, expected)
    assert outside == 0 and _all_finite(candidate, expected), (
        f"{case['name']}/e_dot_v: {outside} element(s) outside atol={PARITY_ATOL} "
        f"+ rtol={PARITY_RTOL} or a non-finite value"
    )
    record = _record_dispatch(
        case,
        "e_dot_v",
        candidate,
        expected,
        {
            "input_sha256": {
                "key": _tensor_digest(key),
                "query": _tensor_digest(query),
                "dst": _tensor_digest(case["dst"]),
            },
            "reference_api": (
                "dgl {} dgl.ops.e_dot_v(graph, key, query) on a CPU dgl.graph"
            ).format(dgl.__version__),
            "num_heads": num_heads,
            "reduced_width": num_feat,
            "non_32_byte_aligned": bool((num_feat * 4) % 32 != 0),
        },
    )
    assert record["passed"], f"{case['name']}/e_dot_v parity failed: {record}"

    # --- edge_softmax (norm_by='dst') ------------------------------------- #
    logits = _logits_with_destination_groups(case, num_heads, 7005)
    candidate = graph_backend.edge_softmax(graph, logits.to(npu_device))
    torch.npu.synchronize()
    expected = dgl.ops.edge_softmax(reference, logits, norm_by="dst")
    assert candidate.device.type == "npu"
    assert tuple(candidate.shape) == tuple(expected.shape) == (num_edges, num_heads)
    torch.testing.assert_close(
        candidate.cpu(), expected, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )
    _, _, _, _, outside = _errors(candidate, expected)
    checked, worst = _softmax_destination_sums(candidate, case["dst"], num_nodes)
    _, reference_worst = _softmax_destination_sums(expected, case["dst"], num_nodes)
    assert outside == 0 and _all_finite(candidate, expected), (
        f"{case['name']}/edge_softmax: {outside} element(s) outside tolerance or a "
        "non-finite value"
    )
    assert checked > 0, f"{case['name']}: no destination had an incoming edge"
    assert worst <= SOFTMAX_SUM_ATOL, (
        f"{case['name']}: per-destination softmax sums deviate by {worst!r} > "
        f"{SOFTMAX_SUM_ATOL} ({checked} destinations checked)"
    )
    assert reference_worst <= SOFTMAX_SUM_ATOL, (
        f"reference softmax sums deviate by {reference_worst!r}: the reference is "
        "not destination-normalized"
    )
    group = case["dst"].remainder(3)
    record = _record_dispatch(
        case,
        "edge_softmax",
        candidate,
        expected,
        {
            "input_sha256": {
                "logits": _tensor_digest(logits),
                "dst": _tensor_digest(case["dst"]),
            },
            "reference_api": (
                "dgl {} dgl.ops.edge_softmax(graph, logits, norm_by='dst') on a CPU "
                "dgl.graph"
            ).format(dgl.__version__),
            "num_heads": num_heads,
            "softmax_destinations_checked": checked,
            "softmax_max_sum_deviation": worst,
            "softmax_reference_max_sum_deviation": reference_worst,
            "softmax_sums_ok": bool(worst <= SOFTMAX_SUM_ATOL),
            "destination_groups": {
                "+1000": sorted(set(case["dst"][group == 0].tolist())),
                "-1000": sorted(set(case["dst"][group == 1].tolist())),
                "ordinary": sorted(set(case["dst"][group == 2].tolist())),
            },
            "non_32_byte_aligned": bool((num_heads * 4) % 32 != 0),
        },
    )
    assert record["passed"], f"{case['name']}/edge_softmax parity failed: {record}"

    # --- copy_e_sum / copy_e_mean ---------------------------------------- #
    channels, degree_dim = 3, 1
    weights = _quantized(num_edges, num_heads * channels * degree_dim, 7003).reshape(
        num_edges, num_heads, channels, degree_dim
    )
    reductions = (
        (
            "copy_e_sum",
            graph_backend.copy_e_sum,
            lambda feat: dgl.ops.copy_e_sum(reference, feat),
            "dgl.ops.copy_e_sum(graph, feat)",
        ),
        (
            "copy_e_mean",
            graph_backend.copy_e_mean,
            lambda feat: dgl.ops.copy_e_mean(reference, feat),
            "dgl.ops.copy_e_mean(graph, feat)",
        ),
    )
    zero_nodes = [
        node for node in range(num_nodes) if int(reference.in_degrees()[node]) == 0
    ]
    for op_name, dispatch, reference_call, reference_api in reductions:
        candidate = dispatch(graph, weights.to(npu_device))
        torch.npu.synchronize()
        expected = reference_call(weights)
        assert candidate.device.type == "npu"
        assert tuple(candidate.shape) == tuple(expected.shape) == (
            num_nodes,
            num_heads,
            channels,
            degree_dim,
        ), f"{op_name} shape {tuple(candidate.shape)} != {tuple(expected.shape)}"
        torch.testing.assert_close(
            candidate.cpu(), expected, atol=PARITY_ATOL, rtol=PARITY_RTOL
        )
        cand_cpu, _, _, _, outside = _errors(candidate, expected)
        assert outside == 0 and _all_finite(candidate, expected)
        zero_rows_exact = all(
            bool(torch.count_nonzero(cand_cpu[node]) == 0) for node in zero_nodes
        )
        assert zero_rows_exact, (
            f"{case['name']}/{op_name}: a zero-in-degree node did not reduce to "
            "exactly zero"
        )
        record = _record_dispatch(
            case,
            op_name,
            candidate,
            expected,
            {
                "input_sha256": {
                    "feat": _tensor_digest(weights),
                    "dst": _tensor_digest(case["dst"]),
                },
                "reference_api": (
                    "dgl {} " + reference_api + " on a CPU dgl.graph"
                ).format(dgl.__version__),
                "zero_in_degree_nodes": zero_nodes,
                "zero_in_degree_rows_exactly_zero": zero_rows_exact,
            },
        )
        assert record["passed"], f"{case['name']}/{op_name} parity failed: {record}"

    # --- global pooling (the GPooling call site of the model) ------------- #
    width = 13  # 52-byte rows: non-32-byte-aligned
    node_feats = _quantized(num_nodes, width, 7004)
    for mode in ("avg", "max"):
        candidate = graph_backend.pool(graph, node_feats.to(npu_device), mode)
        torch.npu.synchronize()
        pool_module = dgl.nn.AvgPooling() if mode == "avg" else dgl.nn.MaxPooling()
        expected = pool_module(reference, node_feats)
        assert candidate.device.type == "npu"
        assert tuple(candidate.shape) == tuple(expected.shape) == (1, width)
        torch.testing.assert_close(
            candidate.cpu(), expected, atol=PARITY_ATOL, rtol=PARITY_RTOL
        )
        cand_cpu, ref_cpu, max_abs, _, outside = _errors(candidate, expected)
        exact = bool(torch.equal(cand_cpu, ref_cpu))
        assert outside == 0 and _all_finite(candidate, expected)
        if mode == "max" and POOL_MAX_EXACT_REQUIRED:
            assert exact, (
                f"{case['name']}/pool(max) must reproduce DGL's selection "
                f"bit-exactly; max |difference| = {max_abs!r}"
            )
        record = _record_dispatch(
            case,
            f"pool({mode})",
            candidate,
            expected,
            {
                "input_sha256": {
                    "feat": _tensor_digest(node_feats),
                    "dst": _tensor_digest(case["dst"]),
                },
                "reference_api": (
                    "dgl {} dgl.nn.{}Pooling()(graph, feat) on a CPU dgl.graph"
                ).format(dgl.__version__, "Avg" if mode == "avg" else "Max"),
                "non_32_byte_aligned": bool((width * 4) % 32 != 0),
                "segment_table": [num_nodes],
            },
        )
        assert record["passed"], (
            f"{case['name']}/pool({mode}) parity failed: {record}"
        )


# --------------------------------------------------------------------------- #
# 3. Hardware / report route evidence for graph construction
# --------------------------------------------------------------------------- #
def test_graph_build_route_evidence(native_graph_ops, npu_device, tmp_path):
    """One ``AscendGraph`` construction -> exactly one ``edge_topology`` launch."""
    del native_graph_ops, npu_device  # the route case requires its own device

    hardware = route_evidence._run_measurement(
        "hardware", _GRAPH_BUILD_ROUTE_CASE, tmp_path / "hardware"
    )
    report = route_evidence._run_measurement(
        "report", _GRAPH_BUILD_ROUTE_CASE, tmp_path / "report"
    )

    artifacts = report.get("profile_artifacts") or {}
    resolved = {
        name: route_evidence.resolve_token(artifacts, name)
        for name in _GRAPH_BUILD_EXPECTED_KERNELS
    }
    op_names = list(report.get("op_names") or [])
    kernel_names = list(report.get("kernel_names") or [])
    ai_core_rows = report.get("ai_core_rows") or {}

    route_record = {
        "case": _GRAPH_BUILD_ROUTE_CASE,
        "candidate": (
            "rfantibody.ascend_ops.graph.AscendGraph(src, dst, num_nodes) built "
            "through the public rfantibody.ascend_ops API on npu:0"
        ),
        "expected_launches": _GRAPH_BUILD_EXPECTED_EVENTS,
        "expected_kernel_names": list(_GRAPH_BUILD_EXPECTED_KERNELS),
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
                for name in _GRAPH_BUILD_EXPECTED_KERNELS
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
    (_EVIDENCE_PATH / "graph_build_route_evidence.json").write_text(
        json.dumps(route_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Channel 1: exactly one hardware task for the one native launch.
    assert hardware.get("hardware_event_count") == _GRAPH_BUILD_EXPECTED_EVENTS, (
        "expected exactly "
        f"{_GRAPH_BUILD_EXPECTED_EVENTS} KERNEL_* hardware task event for the "
        "graph-construction launch, got "
        f"{hardware.get('hardware_event_count')}: "
        f"{json.dumps(hardware.get('task_counts'), sort_keys=True)} "
        f"(trace: {hardware.get('trace')})"
    )
    assert hardware.get("task_counts"), "hardware channel recorded no task counts"
    assert all(
        str(key).split("|")[-1].startswith("KERNEL_")
        for key in hardware["task_counts"]
    ), f"non-hardware task type counted: {hardware['task_counts']}"

    # Channel 2: the vendor report names the kernel of this bundle.
    assert set(op_names) == set(_GRAPH_BUILD_EXPECTED_KERNELS), (
        f"op_summary AI_CORE names are {op_names}, expected "
        f"{list(_GRAPH_BUILD_EXPECTED_KERNELS)}"
    )
    assert set(kernel_names) == set(_GRAPH_BUILD_EXPECTED_KERNELS), (
        f"task_time AI_CORE names are {kernel_names}, expected "
        f"{list(_GRAPH_BUILD_EXPECTED_KERNELS)}"
    )
    assert ai_core_rows.get("op_summary"), "no AI_CORE op_summary rows recorded"
    assert ai_core_rows.get("task_time"), "no AI_CORE task_time rows recorded"
    for name in _GRAPH_BUILD_EXPECTED_KERNELS:
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
