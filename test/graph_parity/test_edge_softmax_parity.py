"""Native AscendC destination edge softmax: route artifacts, tiny execution,
CPU-DGL parity, per-destination normalization and hardware/report route evidence.

The capability under test is one native operator:

* ``ascend_graph::edge_softmax_dst`` -> native AscendC kernel
  ``edge_softmax_dst``, i.e. ``dgl.ops.edge_softmax(g, logits, norm_by='dst')``
  over the native CSR table produced by ``ascend_graph::edge_topology``.

Fixed, pre-declared criteria (declared here, before any candidate result is
inspected, and never relaxed afterwards):

* the compiled artifact for the observed SoC exists and exports the generated
  ``aclrtlaunch_edge_softmax_dst`` entry point, ``ascend_graph::edge_softmax_dst``
  is registered for dispatch key ``PrivateUse1``, and a CPU tensor is rejected
  (no silent device fallback),
* a hand-derived tiny case (E=5, N=3, W=2 and W=3, one zero-in-degree
  destination) is executed by the native kernel on ``npu:0`` and compared with a
  closed-form softmax evaluated in double precision with ``math.exp`` - an
  expectation that is independent of both DGL and the candidate - at
  ``TINY_ATOL = 1e-6``,
* ``ascend_ops.edge_softmax`` on ``npu:0`` matches **real DGL 1.1.3 on CPU**
  (``dgl.graph((src, dst), num_nodes=N)`` + ``dgl.ops.edge_softmax``) built from
  the same ``src``/``dst``/``logits`` arrays - an independent reference, never
  the candidate.  Cases cover multiple destinations/heads, repeated duplicate
  edges, zero-in-degree nodes, ordinary logits, destination groups around
  ``+1000``/``-1000``, destinations that mix both extremes, the production
  attention width, and non-32-byte-aligned widths,
* for every destination with in-degree > 0 and every column the returned weights
  sum to one within ``WEIGHT_SUM_ATOL = 1e-5``, and all candidate and reference
  outputs are finite.

Operator tolerances: ``atol=1e-5``, ``rtol=1e-4`` (float32).  Passing this
operator parity does not by itself establish full-model parity.

The route test uses the vendor's two measured channels for the same public-API
bundle (each measurement in a fresh interpreter through
``route_evidence._run_measurement``): exactly two ``KERNEL_*`` hardware task
events for the two native launches (``edge_topology`` + ``edge_softmax_dst``),
and ``AI_CORE`` report rows whose names are exactly those two kernels, each
token resolved from the raw vendor artefacts with an empty negative-token
resolution.
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

from test.graph_parity import route_evidence

# --- Pre-declared tolerances (fixed before candidate execution) --------------
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4
# Per-destination, per-column softmax weight sum (DGL semantics: exactly one).
WEIGHT_SUM_ATOL = 1e-5
# Tiny hand-derived case: double-precision closed form vs float32 kernel.
TINY_ATOL = 1e-6
HAF_POISON_CHECK = "all outputs finite"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_PATH = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "migration_reports")
) / "graph_parity"

_ROUTE_CASE = "test.graph_parity.route_case:run_edge_softmax"
_EXPECTED_LAUNCHES = 2
_EXPECTED_KERNEL_NAMES = ("edge_topology", "edge_softmax_dst")

_EVIDENCE = {
    "capability": (
        "native destination edge_softmax(norm_by='dst') over the native CSR "
        "edge topology"
    ),
    "candidate": (
        "ascend_graph::build_edge_topology -> native edge_topology kernel and "
        "ascend_graph::edge_softmax_dst -> native edge_softmax_dst kernel, both "
        "executed on npu:0 through the public rfantibody.ascend_ops API"
    ),
    "reference": (
        "dgl {} dgl.graph((src, dst), num_nodes=N) + dgl.ops.edge_softmax("
        "graph, logits) on CPU; the graph and logits are built from the same "
        "src/dst/logits arrays (independent reference, never the candidate)"
    ).format(dgl.__version__),
    "tolerances": {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "weight_sum_atol": WEIGHT_SUM_ATOL,
        "tiny_closed_form_atol": TINY_ATOL,
        "max_abs_error_definition": "max |candidate - reference| over all elements",
        "max_rel_error_definition": (
            "max |candidate - reference| / max(|reference|, 1e-12) over all elements"
        ),
        "assertion": (
            "torch.testing.assert_close(candidate, reference, atol=1e-5, "
            "rtol=1e-4); all outputs finite; every per-destination-per-column "
            "weight sum within 1e-5 of 1"
        ),
        "declared_before_execution": True,
        "all_finite_check": HAF_POISON_CHECK,
    },
    "semantics": {
        "formula": (
            "m_d,c = max_{e in in_edges(d)} logits[e, c]; "
            "s_d,c = sum_{e in in_edges(d)} exp(logits[e, c] - m_d,c); "
            "out[e, c] = exp(logits[e, c] - m_d,c) / s_d,c"
        ),
        "shift_scope": "per destination and per trailing column, never graph-wide",
        "denominator_salting": "none (no epsilon/tiny/clamp)",
        "zero_in_degree": "owns no edge rows, therefore produces no output rows",
        "duplicate_edges": "distinct CSR entries, counted separately",
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


def _weight_sums(values_cpu, offsets, indices, num_nodes, width):
    """Verification-side per-destination/per-column sums (never a product path).

    The destination grouping comes from the native CSR table copied to the host;
    the reduction is only used to check DGL's sum-to-one property of the returned
    weights, not to compute them.
    """
    flat = values_cpu.reshape(-1, width).numpy().astype(np.float64)
    idx = indices.detach().cpu().numpy().astype(np.int64)
    off = offsets.detach().cpu().numpy().astype(np.int64)
    sums = np.zeros((num_nodes, width), dtype=np.float64)
    degrees = np.zeros(num_nodes, dtype=np.int64)
    for node in range(num_nodes):
        begin = int(off[node])
        end = int(off[node + 1])
        degrees[node] = end - begin
        if end > begin:
            sums[node] = flat[idx[begin:end]].sum(axis=0)
    return sums, degrees


def _record_case(case, topology_cpu, candidate, reference, offsets, indices,
                 sums, degrees, outside_tolerance):
    """Append one machine-readable parity row and persist its exact inputs."""
    out_dir = _EVIDENCE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    src = case["src"]
    dst = case["dst"]
    logits = case["logits"]
    width = int(logits.shape[1])
    num_edges = int(logits.shape[0])
    num_nodes = int(offsets.numel()) - 1

    cand, ref, max_abs, max_rel, outside = _errors(candidate, reference)
    # `sums` is a NumPy array: index it with a NumPy boolean mask, never with a
    # torch tensor (a 1-element torch tensor is read as an integer index).
    degrees_np = degrees.detach().cpu().numpy()
    with_destination = degrees_np > 0
    if bool(with_destination.any()):
        deviation = float(np.abs(sums[with_destination] - 1.0).max())
    else:
        deviation = 0.0

    npz_path = out_dir / f"edge_softmax_{case['name']}.npz"
    np.savez(
        npz_path,
        src=src.numpy(),
        dst=dst.numpy(),
        logits=logits.numpy(),
        offsets=offsets.numpy(),
        indices=indices.numpy(),
        degrees=degrees.numpy(),
        candidate=cand.numpy(),
        reference=ref.numpy(),
    )

    duplicate_pairs = 0
    if num_edges:
        pairs = torch.stack([src, dst], dim=1)
        unique = torch.unique(pairs, dim=0)
        duplicate_pairs = int(num_edges - unique.size(0))

    _EVIDENCE["cases"].append(
        {
            "case": case["name"],
            "description": case["description"],
            "provenance": case.get("provenance", {}),
            "shapes": {
                "logits": list(logits.shape),
                "src": list(src.shape),
                "dst": list(dst.shape),
                "offsets": list(offsets.shape),
                "indices": list(indices.shape),
                "candidate": list(candidate.shape),
                "reference": list(reference.shape),
                "num_edges": num_edges,
                "num_nodes": num_nodes,
                "weight_width": width,
            },
            "dtypes": {
                "logits": str(logits.dtype),
                "src": str(src.dtype),
                "dst": str(dst.dtype),
                "offsets": str(offsets.dtype),
                "indices": str(indices.dtype),
                "candidate": str(candidate.dtype),
                "reference": str(reference.dtype),
            },
            "devices": {
                "candidate": str(candidate.device),
                "reference": str(reference.device),
                "reference_identity": (
                    "dgl {} dgl.ops.edge_softmax on CPU (norm_by='dst'); "
                    "real DGL, never the candidate".format(dgl.__version__)
                ),
            },
            "input_sha256": {
                "src": _tensor_digest(src),
                "dst": _tensor_digest(dst),
                "logits": _tensor_digest(logits),
            },
            "topology_sha256": {
                "offsets": _tensor_digest(topology_cpu["offsets"]),
                "indices": _tensor_digest(topology_cpu["indices"]),
                "degrees": _tensor_digest(topology_cpu["degrees"]),
            },
            "non_32_byte_aligned": bool((width * 4) % 32 != 0),
            "duplicate_edges": duplicate_pairs > 0,
            "duplicate_edge_pairs": duplicate_pairs,
            "zero_in_degree_nodes": int((degrees == 0).sum()),
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "elements_outside_atol_plus_rtol": outside,
            "weight_sum_max_deviation": deviation,
            "all_finite": bool(
                torch.isfinite(cand).all() and torch.isfinite(ref).all()
            ),
            "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
            "weight_sum_atol": WEIGHT_SUM_ATOL,
            "assert_close_passed": outside_tolerance == 0,
            "passed": bool(
                _all_finite(cand, ref)
                and outside_tolerance == 0
                and deviation <= WEIGHT_SUM_ATOL
            ),
            "inputs_npz": str(npz_path),
        }
    )


def _all_finite(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    return bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())


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
    (_EVIDENCE_PATH / "edge_softmax.json").write_text(
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


def _reference_softmax(src, dst, num_nodes, logits):
    graph = _reference_graph(src, dst, num_nodes)
    return dgl.ops.edge_softmax(graph, logits)


def _check_topology(case, topology):
    """Validate the native CSR table against the recorded ``dst`` on CPU."""
    offsets_cpu = topology.offsets.detach().cpu()
    indices_cpu = topology.indices.detach().cpu()
    degrees_cpu = topology.degrees.detach().cpu()
    offsets = offsets_cpu.to(torch.int64)
    indices = indices_cpu.to(torch.int64)
    degrees = degrees_cpu.to(torch.int64)
    dst = case["dst"].to(torch.int64)
    num_edges = int(dst.numel())
    num_nodes = int(offsets.numel()) - 1

    assert offsets_cpu.dtype == torch.int64
    assert indices_cpu.dtype == torch.int32
    assert degrees_cpu.dtype == torch.int64
    assert offsets.numel() == num_nodes + 1
    assert indices.numel() == num_edges
    assert degrees.numel() == num_nodes
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == num_edges, (
        f"CSR table closes at {int(offsets[-1])}, expected {num_edges}"
    )
    assert bool((offsets[1:] >= offsets[:-1]).all()), "offsets are not monotonic"
    assert torch.equal(offsets[1:] - offsets[:-1], degrees), (
        "degrees disagree with the CSR offsets"
    )

    dst_np = dst.numpy()
    offsets_np = offsets.numpy()
    indices_np = indices.numpy()
    for node in range(num_nodes):
        begin = int(offsets_np[node])
        end = int(offsets_np[node + 1])
        group = indices_np[begin:end]
        assert list(group) == sorted(int(e) for e in group), (
            f"destination {node}: CSR edge ids are not ascending"
        )
        for edge in group:
            assert 0 <= int(edge) < num_edges, (
                f"destination {node}: CSR edge id {int(edge)} out of range"
            )
            assert int(dst_np[int(edge)]) == node, (
                f"destination {node}: edge {int(edge)} has dst "
                f"{int(dst_np[int(edge)])}"
            )
    return {
        "offsets": offsets_cpu,
        "indices": indices_cpu,
        "degrees": degrees_cpu,
    }


# --------------------------------------------------------------------------- #
# Deterministic case construction
# --------------------------------------------------------------------------- #
def _random_graph(num_edges: int, num_nodes: int, seed: int, dst_choices=None):
    gen = torch.Generator().manual_seed(seed)
    if num_edges == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty.clone()
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    if dst_choices is None:
        dst = torch.randint(
            0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64
        )
    else:
        choices = torch.as_tensor(dst_choices, dtype=torch.int64)
        picks = torch.randint(
            0, choices.numel(), (num_edges,), generator=gen, dtype=torch.int64
        )
        dst = choices[picks]
    return src, dst


def _random_logits(num_edges: int, width: int, seed: int, scale: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn((num_edges, width), generator=gen) * scale).to(
        torch.float32
    )


def _case(name, description, num_nodes, src, dst, logits, provenance=None,
          low_group_from=None):
    assert src.dim() == 1 and dst.dim() == 1 and src.shape == dst.shape
    assert src.dtype == torch.int64 and dst.dtype == torch.int64
    assert logits.dim() == 2 and logits.shape[0] == dst.numel()
    assert logits.dtype == torch.float32
    return {
        "name": name,
        "description": description,
        "num_nodes": int(num_nodes),
        "src": src,
        "dst": dst,
        "logits": logits,
        "provenance": provenance or {},
        # When set, every edge whose destination id is >= this value belongs to
        # the low-logit (+/-1000) group and must keep non-degenerate weights.
        "low_group_from": low_group_from,
    }


def _case_ordinary_h4():
    num_edges, num_nodes, width = 2048, 256, 4
    src, dst = _random_graph(num_edges, num_nodes, 7001)
    logits = _random_logits(num_edges, width, 8001)
    return _case(
        "ordinary_h4",
        "ordinary logits, production attention head count W=4",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_duplicate_edges_h4():
    pairs = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 3), (0, 3), (2, 2)]
    src = torch.tensor([pair[0] for pair in pairs] * 32, dtype=torch.int64)
    dst = torch.tensor([pair[1] for pair in pairs] * 32, dtype=torch.int64)
    logits = _random_logits(src.numel(), 4, 8002)
    return _case(
        "duplicate_edges_h4",
        "repeated duplicate (src, dst) pairs counted as separate CSR entries, W=4",
        6,
        src,
        dst,
        logits,
    )


def _case_zero_in_degree_h3():
    num_nodes = 40
    src, dst = _random_graph(
        120, num_nodes, 7003, dst_choices=list(range(0, num_nodes, 2))
    )
    logits = _random_logits(src.numel(), 3, 8003)
    return _case(
        "zero_in_degree_h3",
        "20 of 40 destinations have zero in-degree; W=3 (12-byte, "
        "non-32-byte-aligned rows)",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_signed1000_groups_h4():
    # Destinations are grouped by argmax over a +/-1000 two-logit head, the same
    # grouping the distogram/probability head produces.  Logits stay within
    # +/-4 of their group base, so every float32 value is exact and the low
    # group's weights (exp(diff)/sum with diff in [-8, 0]) stay well above the
    # float32 underflow region - a graph-wide maximum would collapse them to 0.
    num_nodes, width, num_edges = 64, 4, 2000
    half = num_nodes // 2
    group_logits = torch.full((num_nodes, 2), -1000.0, dtype=torch.float32)
    group_logits[:half, 0] = 1000.0
    group_logits[half:, 1] = 1000.0
    group_of_node = torch.argmax(group_logits, dim=1)
    group0 = torch.nonzero(group_of_node == 0).flatten()
    group1 = torch.nonzero(group_of_node == 1).flatten()
    gen = torch.Generator().manual_seed(3010)
    picks = torch.randint(0, 2, (num_edges,), generator=gen, dtype=torch.int64)
    within = torch.randint(0, half, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.where(picks == 0, group0[within], group1[within])
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    jitter = torch.randint(
        -4, 5, (num_edges, width), generator=gen, dtype=torch.int64
    ).to(torch.float32)
    # `dst < half` is [E]; make the base [E] first and unsqueeze afterwards so the
    # where() cannot broadcast the condition against the [E, 1] values.
    base = torch.where(
        dst < half,
        torch.full((num_edges,), 1000.0, dtype=torch.float32),
        torch.full((num_edges,), -1000.0, dtype=torch.float32),
    ).unsqueeze(1)
    logits = base + jitter
    return _case(
        "signed1000_groups_h4",
        "destination groups whose logits are ~+1000 and ~-1000; both groups "
        "must keep non-degenerate, sum-to-one weights, W=4",
        num_nodes,
        src,
        dst,
        logits,
        provenance={
            "grouping_source": (
                "destinations grouped by argmax over +1000/-1000 logits, the "
                "distogram/probability grouping used by the model"
            )
        },
        low_group_from=half,
    )


def _case_mixed_extreme_within_destination_h2():
    # One destination holds both +1000 and -1000 logits: exp(-2000) underflows to
    # zero in float32 exactly as DGL's own float32 path does, so the parity
    # comparison must agree on both the surviving and the vanished weights.
    num_nodes, width, num_edges = 8, 2, 400
    gen = torch.Generator().manual_seed(7004)
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    extreme_dst = 3
    picks = torch.randint(
        0, num_nodes - 1, (num_edges,), generator=gen, dtype=torch.int64
    )
    dst = torch.where(dst == extreme_dst, (picks + 1) % num_nodes, dst)
    forced = torch.arange(0, num_edges, 4, dtype=torch.int64)
    dst[forced] = extreme_dst
    jitter = torch.randint(
        -4, 5, (num_edges, width), generator=gen, dtype=torch.int64
    ).to(torch.float32)
    high = torch.zeros(num_edges, dtype=torch.bool)
    high[forced[0::2]] = True
    extreme = (
        torch.where(
            high.reshape(-1, 1),
            torch.full((num_edges, 1), 1000.0, dtype=torch.float32),
            torch.full((num_edges, 1), -1000.0, dtype=torch.float32),
        )
        + jitter
    )
    ordinary = _random_logits(num_edges, width, 8004)
    logits = torch.where(
        (dst == extreme_dst).reshape(-1, 1), extreme, ordinary
    )
    group = logits[dst == extreme_dst]
    assert bool((group > 0).any()) and bool((group < 0).any()), (
        "the mixed case must put both +1000 and -1000 logits in one destination"
    )
    return _case(
        "mixed_extreme_within_destination_h2",
        "one destination mixes +1000 and -1000 logits; the underflowing weights "
        "must match DGL's float32 result exactly enough for the pre-declared "
        "tolerances, W=2",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_unaligned_w1():
    num_edges, num_nodes, width = 257, 33, 1
    src, dst = _random_graph(num_edges, num_nodes, 7005)
    logits = _random_logits(num_edges, width, 8005)
    return _case(
        "unaligned_w1",
        "W=1 (4-byte rows): DGL's canonical [E, 1] edge weight shape",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_unaligned_w7():
    num_edges, num_nodes, width = 333, 64, 7
    src, dst = _random_graph(num_edges, num_nodes, 7006)
    logits = _random_logits(num_edges, width, 8006)
    return _case(
        "unaligned_w7",
        "non-32-byte-aligned W=7 (28-byte rows)",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_wide_w20():
    num_edges, num_nodes, width = 1024, 128, 20
    src, dst = _random_graph(num_edges, num_nodes, 7007)
    logits = _random_logits(num_edges, width, 8007)
    return _case(
        "wide_w20",
        "W=20 (80-byte rows): 4 heads x 5 channels",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_single_destination():
    num_edges, width = 9, 3
    src = torch.zeros(num_edges, dtype=torch.int64)
    dst = torch.zeros(num_edges, dtype=torch.int64)
    logits = _random_logits(num_edges, width, 8008)
    return _case(
        "single_destination",
        "degenerate graph with a single node (N=1, all edges into it), W=3",
        1,
        src,
        dst,
        logits,
    )


def _case_empty_edges():
    num_nodes, width = 8, 8
    src = torch.empty(0, dtype=torch.int64)
    dst = torch.empty(0, dtype=torch.int64)
    logits = torch.empty(0, width, dtype=torch.float32)
    return _case(
        "empty_edges",
        "degenerate graph with no edges (E=0): empty result with DGL's shape",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_high_degree_long_segment_h4():
    # One destination receives 512 parallel edges (a long accumulation order);
    # four other destinations keep short segments.
    long_segment = 512
    num_edges = long_segment + 4 * 8
    num_nodes = 5
    gen = torch.Generator().manual_seed(7009)
    src = torch.randint(0, num_nodes, (num_edges,), generator=gen, dtype=torch.int64)
    dst = torch.cat(
        [
            torch.zeros(long_segment, dtype=torch.int64),
            (torch.arange(4 * 8, dtype=torch.int64) % 4) + 1,
        ]
    )
    logits = _random_logits(num_edges, 4, 8009)
    return _case(
        "high_degree_long_segment_h4",
        "one destination with 512 duplicate edges plus four short segments, W=4",
        num_nodes,
        src,
        dst,
        logits,
    )


def _case_real_model_topk_h4():
    """Production topology and production attention head count."""
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
    logits = _random_logits(int(src.numel()), 4, 8102)
    return _case(
        "real_model_topk_h4",
        "topology from rfantibody.rf2.network.util_module.make_topk_graph "
        "(kmin=32, top_k=128, B=1, L=120) with the production attention head "
        "count W=4",
        int(graph.num_nodes()),
        src,
        dst,
        logits,
        provenance={
            "topology_source": (
                "rfantibody.rf2.network.util_module.make_topk_graph (kmin=32, "
                "top_k=128, B=1, L=120)"
            ),
            "graph_num_nodes": int(graph.num_nodes()),
            "graph_num_edges": int(src.numel()),
        },
    )


_CASES = [
    pytest.param(_case_ordinary_h4, id="ordinary_h4"),
    pytest.param(_case_duplicate_edges_h4, id="duplicate_edges_h4"),
    pytest.param(_case_zero_in_degree_h3, id="zero_in_degree_h3"),
    pytest.param(_case_signed1000_groups_h4, id="signed1000_groups_h4"),
    pytest.param(
        _case_mixed_extreme_within_destination_h2,
        id="mixed_extreme_within_destination_h2",
    ),
    pytest.param(_case_unaligned_w1, id="unaligned_w1"),
    pytest.param(_case_unaligned_w7, id="unaligned_w7"),
    pytest.param(_case_wide_w20, id="wide_w20"),
    pytest.param(_case_single_destination, id="single_destination"),
    pytest.param(_case_empty_edges, id="empty_edges"),
    pytest.param(_case_high_degree_long_segment_h4, id="high_degree_long_segment_h4"),
    pytest.param(_case_real_model_topk_h4, id="real_model_topk_h4"),
]


# --------------------------------------------------------------------------- #
# Hand-derived closed form (double precision, independent of DGL and candidate)
# --------------------------------------------------------------------------- #
def _closed_form_softmax(rows, dst, num_nodes):
    """Destination softmax in double precision from ``math.exp``."""
    width = len(rows[0])
    out = [[0.0] * width for _ in rows]
    groups = {}
    for edge, destination in enumerate(dst):
        groups.setdefault(int(destination), []).append(edge)
    for edges in groups.values():
        for column in range(width):
            maximum = max(rows[edge][column] for edge in edges)
            exponentials = [math.exp(rows[edge][column] - maximum) for edge in edges]
            total = math.fsum(exponentials)
            for edge, value in zip(edges, exponentials):
                out[edge][column] = value / total
    return out


_TINY_DST = [0, 2, 0, 2, 2]  # node 1 has zero in-degree
_TINY_LOGITS_W2 = [
    [0.0, 1.0],
    [1.0, -1.0],
    [2.0, 0.5],
    [-1.0, 2.0],
    [0.5, 0.25],
]
_TINY_LOGITS_W3 = [
    [0.0, 1.0, 0.25],
    [1.0, -1.0, 1.5],
    [2.0, 0.5, -0.75],
    [-1.0, 2.0, 2.5],
    [0.5, 0.25, 0.0],
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


def test_edge_softmax_native_artifacts(native_graph_ops, npu_device):
    """Compiled artifact symbol, ``PrivateUse1`` kernel and no CPU fallback."""
    artifact = Path(native_graph_ops["artifact"])
    assert artifact.is_file(), f"compiled artifact missing: {artifact}"
    symbols = _dynamic_symbols(artifact)
    entry = "aclrtlaunch_edge_softmax_dst"
    assert entry in symbols, f"generated route entry point missing: {entry}"
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "ascend_graph::edge_softmax_dst", "PrivateUse1"
    ), "ascend_graph::edge_softmax_dst has no PrivateUse1 kernel"

    ops = native_graph_ops["ops"]
    cpu_logits = torch.tensor([[0.5, -1.0], [1.0, 1.0]], dtype=torch.float32)
    cpu_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    cpu_indices = torch.tensor([0, 1], dtype=torch.int32)
    cpu_degrees = torch.tensor([1, 1], dtype=torch.int64)
    cpu_topology = ops.EdgeTopology(cpu_offsets, cpu_indices, cpu_degrees)
    with pytest.raises(RuntimeError):
        ops.edge_softmax(cpu_logits, cpu_topology)
    with pytest.raises(ValueError):
        ops.edge_softmax(
            torch.zeros(3, 2), ops.EdgeTopology(cpu_offsets, cpu_indices, cpu_degrees)
        )

    _EVIDENCE["artifacts"] = {
        "kernel_library": str(artifact),
        "kernel_library_sha256": native_graph_ops["artifact_sha256"],
        "python_binding": native_graph_ops["binding"],
        "python_binding_sha256": native_graph_ops["binding_sha256"],
        "exported_entry_points": [entry] if entry in symbols else [],
        "dispatch": "PrivateUse1",
        "device": str(npu_device),
        "cpu_tensors_rejected": True,
    }


def test_edge_softmax_tiny_native_execution(native_graph_ops, npu_device, tmp_path):
    """Hand-derived tiny case (E=5, N=3, W=2 and W=3) on the native kernels."""
    ops = native_graph_ops["ops"]
    dst = torch.tensor(_TINY_DST, dtype=torch.int64, device=npu_device)
    topology = ops.build_edge_topology(dst, 3)
    degrees = ops.in_degrees(topology)
    assert int(degrees.cpu()[1]) == 0, (
        "the tiny case must contain a zero-in-degree node"
    )

    observed_cases = {}
    for width, rows in ((2, _TINY_LOGITS_W2), (3, _TINY_LOGITS_W3)):
        logits = torch.tensor(rows, dtype=torch.float32, device=npu_device)
        out = ops.edge_softmax(logits, topology)
        torch.npu.synchronize()

        assert out.device.type == "npu", f"output device is {out.device}"
        assert tuple(out.shape) == (5, width), (
            f"W={width}: output shape {tuple(out.shape)} != {(5, width)}"
        )
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()

        expected = _closed_form_softmax(rows, _TINY_DST, 3)
        observed = out.cpu().tolist()
        for edge in range(len(rows)):
            for column in range(width):
                difference = abs(observed[edge][column] - expected[edge][column])
                assert difference <= TINY_ATOL, (
                    f"W={width} edge {edge} column {column}: observed "
                    f"{observed[edge][column]!r} differs from the double-precision "
                    f"closed form {expected[edge][column]!r} by {difference!r} > "
                    f"{TINY_ATOL}"
                )
        observed_cases[str(width)] = {
            "logits": rows,
            "expected_double": expected,
            "observed": observed,
            "shape": [5, width],
        }

    trace = route_evidence.collect_level0(
        lambda: ops.edge_softmax(
            torch.tensor(_TINY_LOGITS_W3, dtype=torch.float32, device=npu_device),
            topology,
        ),
        tmp_path / "edge_softmax_tiny_hardware.json",
    )
    torch.npu.synchronize()

    _EVIDENCE["tiny_native_execution"] = {
        "dst": _TINY_DST,
        "num_edges": 5,
        "num_nodes": 3,
        "zero_in_degree_node": 1,
        "degrees": degrees.cpu().tolist(),
        "closed_form": "double-precision math.exp destination softmax",
        "tolerance": TINY_ATOL,
        "cases": observed_cases,
        "hardware_trace": trace,
    }


# --------------------------------------------------------------------------- #
# Parity against real DGL on CPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case_factory", _CASES)
def test_edge_softmax_parity_vs_cpu_dgl(native_graph_ops, npu_device, case_factory):
    """``edge_softmax`` NPU kernel vs real DGL 1.1.3 on CPU."""
    case = case_factory()
    src = case["src"]
    dst = case["dst"]
    logits = case["logits"]
    num_nodes = case["num_nodes"]
    width = int(logits.shape[1])

    reference = _reference_softmax(src, dst, num_nodes, logits)

    ops = native_graph_ops["ops"]
    topology = ops.build_edge_topology(dst.to(npu_device), num_nodes)
    candidate = ops.edge_softmax(logits.to(npu_device), topology)
    torch.npu.synchronize()

    topology_cpu = _check_topology(case, topology)
    offsets = topology_cpu["offsets"].to(torch.int64)
    indices = topology_cpu["indices"].to(torch.int64)
    degrees = topology_cpu["degrees"].to(torch.int64)

    assert candidate.device.type == "npu", f"candidate device is {candidate.device}"
    assert reference.device.type == "cpu", f"reference device is {reference.device}"
    assert tuple(candidate.shape) == tuple(reference.shape), (
        f"{case['name']}: candidate shape {tuple(candidate.shape)} != reference "
        f"shape {tuple(reference.shape)}"
    )
    assert candidate.dtype == reference.dtype == torch.float32

    cand_cpu = candidate.cpu()
    assert torch.isfinite(cand_cpu).all(), f"{case['name']}: non-finite candidate"
    assert torch.isfinite(reference).all(), f"{case['name']}: non-finite reference"
    torch.testing.assert_close(
        cand_cpu, reference, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )

    sums, row_degrees = _weight_sums(cand_cpu, offsets, indices, num_nodes, width)
    assert np.array_equal(row_degrees, degrees.numpy()), (
        f"{case['name']}: CSR degrees disagree with the native degree vector"
    )
    with_destination = row_degrees > 0
    assert int(with_destination.sum()) > 0 or int(logits.shape[0]) == 0
    if int(with_destination.sum()):
        deviation = float(np.abs(sums[with_destination] - 1.0).max())
        assert deviation <= WEIGHT_SUM_ATOL, (
            f"{case['name']}: per-destination/per-column weight sum deviates by "
            f"{deviation!r} > {WEIGHT_SUM_ATOL}"
        )
    else:
        deviation = 0.0

    assert int(cand_cpu.shape[0]) == int(logits.shape[0]), (
        f"{case['name']}: the candidate must return one row per edge"
    )

    # Semantic control: destinations that own no edge keep zero rows and the
    # low-logit destination group keeps non-degenerate, sum-to-one weights
    # (exactly what a graph-wide maximum shift would destroy).
    low_group_from = case["low_group_from"]
    if low_group_from is not None:
        low_mask = (dst >= int(low_group_from)).reshape(-1, 1).expand(-1, width)
        assert bool((cand_cpu[low_mask] > 0).all()), (
            f"{case['name']}: low-logit destination group has zero weights; a "
            "graph-wide maximum shift would produce exactly this"
        )

    _, _, _, _, outside = _errors(cand_cpu, reference)
    assert outside == 0, (
        f"{case['name']}: {outside} element(s) outside atol={PARITY_ATOL} + "
        f"rtol={PARITY_RTOL} * |reference|"
    )

    _record_case(
        case,
        topology_cpu,
        candidate,
        reference,
        offsets,
        indices,
        sums,
        degrees,
        outside,
    )


# --------------------------------------------------------------------------- #
# Hardware / report route evidence
# --------------------------------------------------------------------------- #
def test_edge_softmax_route_evidence(native_graph_ops, npu_device, tmp_path):
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
    (_EVIDENCE_PATH / "edge_softmax_route_evidence.json").write_text(
        json.dumps(route_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
