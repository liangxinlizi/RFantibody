"""Native AscendC edge-destination reductions: route artifacts, tiny execution,
CPU-DGL parity and hardware/report route evidence.

The capability under test is one native edge-topology + segment-reduction
family:

* ``ascend_graph::edge_topology`` -> native AscendC kernel ``edge_topology``
  (deterministic CSR ``offsets``/``indices`` and exact in-degrees),
* ``ascend_graph::copy_e_sum`` -> native AscendC kernel ``copy_e_sum``,
* ``ascend_graph::copy_e_mean`` -> native AscendC kernel ``copy_e_mean``.

Three fixed, pre-declared things are checked:

* the compiled artifact for the observed SoC exists and exports the generated
  ``aclrtlaunch_edge_topology`` / ``aclrtlaunch_copy_e_sum`` /
  ``aclrtlaunch_copy_e_mean`` entry points, the three operators are registered
  for dispatch key ``PrivateUse1``, and CPU tensors are rejected,
* a hand-computed tiny case (E=5, N=3, W=7) is executed by the native kernels on
  ``npu:0`` and compared with exact expected values, including the stable CSR
  order and the in-degree vector,
* ``copy_e_sum``/``copy_e_mean``/``in_degrees`` on ``npu:0`` match **real DGL
  1.1.3 on CPU** (``dgl.ops.copy_e_sum``, ``dgl.ops.copy_e_mean``,
  ``DGLGraph.in_degrees()``) built from the same ``src``/``dst``/``feat``
  arrays - an independent reference, never the candidate.  The matrix covers
  multiple destinations/heads, duplicate edges, zero-in-degree nodes, ordinary
  logits, destination groups around ``+1000``/``-1000``, empty and single-node
  graphs, the production ``make_topk_graph`` topology with the production
  attention width, and non-32-byte-aligned widths.

Tolerances are declared here, before any candidate result is inspected, and are
never relaxed afterwards: ``atol=1e-5``, ``rtol=1e-4`` (float32).  In-degree
counts are integers and must match exactly.  Passing this operator parity does
not by itself establish full-model parity.

The route test uses the vendor's two measured channels for the same public-API
bundle (each measurement in a fresh interpreter through
``route_evidence._run_measurement``): exactly four ``KERNEL_*`` hardware task
events for the four native launches, and ``AI_CORE`` report rows whose names are
exactly ``edge_topology``, ``copy_e_sum`` and ``copy_e_mean``, each token
resolved from the raw vendor artefacts with an empty negative-token resolution.
"""

from __future__ import annotations

import hashlib
import json
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_PATH = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "outputs" / "evidence")
) / "graph_parity"

_ROUTE_CASE = "test.graph_parity.route_case:run_edge_reductions"
_EXPECTED_LAUNCHES = 4
_EXPECTED_KERNEL_NAMES = ("edge_topology", "copy_e_sum", "copy_e_mean")

_EVIDENCE = {
    "capability": (
        "native edge topology + DGL copy_e_sum/copy_e_mean/in_degrees "
        "destination-edge reductions"
    ),
    "candidate": (
        "ascend_graph::edge_topology / ascend_graph::copy_e_sum / "
        "ascend_graph::copy_e_mean -> native AscendC kernels "
        "edge_topology/copy_e_sum/copy_e_mean executed on npu:0"
    ),
    "reference": (
        "dgl {} dgl.graph((src, dst), num_nodes=N) + dgl.ops.copy_e_sum + "
        "dgl.ops.copy_e_mean + DGLGraph.in_degrees() on CPU; the graph and "
        "tensors are built from the same src/dst/feat arrays (independent "
        "reference, never the candidate)"
    ).format(dgl.__version__),
    "tolerances": {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "max_abs_error_definition": "max |candidate - reference| over all elements",
        "max_rel_error_definition": (
            "max |candidate - reference| / max(|reference|, 1e-12) over all elements"
        ),
        "in_degrees": "exact integer equality",
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
    cand = candidate.detach().cpu()
    ref = reference.detach().cpu()
    diff = (cand - ref).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    if ref.numel():
        max_rel = float((diff / ref.abs().clamp_min(1e-12)).max())
    else:
        max_rel = 0.0
    return cand, ref, max_abs, max_rel


def _record_case(case, topology_cpu, candidate_sum, candidate_mean, candidate_deg,
                 reference_sum, reference_mean, reference_deg):
    """Append one machine-readable parity row and persist its exact inputs."""
    out_dir = _EVIDENCE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = case["feat"]
    src = case["src"]
    dst = case["dst"]
    offsets = topology_cpu["offsets"]
    indices = topology_cpu["indices"]
    degrees = topology_cpu["degrees"]

    cand_sum, ref_sum, sum_abs, sum_rel = _errors(candidate_sum, reference_sum)
    cand_mean, ref_mean, mean_abs, mean_rel = _errors(candidate_mean, reference_mean)
    cand_deg = candidate_deg.detach().cpu().to(torch.int64)
    ref_deg = reference_deg.detach().cpu().to(torch.int64)

    npz_path = out_dir / f"segment_reduce_{case['name']}.npz"
    np.savez(
        npz_path,
        src=src.numpy(),
        dst=dst.numpy(),
        feat=feat.numpy(),
        offsets=offsets.numpy(),
        indices=indices.numpy(),
        candidate_sum=cand_sum.numpy(),
        reference_sum=ref_sum.numpy(),
        candidate_mean=cand_mean.numpy(),
        reference_mean=ref_mean.numpy(),
        candidate_degrees=cand_deg.numpy(),
        reference_degrees=ref_deg.numpy(),
    )

    width = 1
    for size in feat.shape[1:]:
        width *= int(size)
    zero_degree = int((ref_deg == 0).sum())
    duplicate_edges = bool(
        int(dst.numel()) != int(torch.unique(torch.stack([src, dst], dim=1), dim=0).size(0))
    )

    _EVIDENCE["cases"].append(
        {
            "case": case["name"],
            "description": case["description"],
            "provenance": case.get("provenance", {}),
            "shapes": {
                "feat": list(feat.shape),
                "src": list(src.shape),
                "dst": list(dst.shape),
                "offsets": list(offsets.shape),
                "indices": list(indices.shape),
                "degrees": list(degrees.shape),
                "num_edges": int(feat.shape[0]),
                "num_nodes": int(offsets.numel()) - 1,
                "feature_width": int(width),
            },
            "dtypes": {
                "feat": str(feat.dtype),
                "src": str(src.dtype),
                "dst": str(dst.dtype),
                "offsets": str(offsets.dtype),
                "indices": str(indices.dtype),
                "degrees": str(degrees.dtype),
                "candidate": str(candidate_sum.dtype),
                "reference": str(ref_sum.dtype),
            },
            "devices": {
                "candidate": str(candidate_sum.device),
                "reference": str(reference_sum.device),
                "reference_identity": (
                    "dgl {} dgl.ops.copy_e_sum/copy_e_mean and "
                    "DGLGraph.in_degrees on CPU".format(dgl.__version__)
                ),
            },
            "input_sha256": {
                "src": _tensor_digest(src),
                "dst": _tensor_digest(dst),
                "feat": _tensor_digest(feat),
            },
            "topology_sha256": {
                "offsets": _tensor_digest(offsets),
                "indices": _tensor_digest(indices),
                "degrees": _tensor_digest(degrees),
            },
            "topology_matches_dst": True,
            "non_32_byte_aligned": bool((int(width) * 4) % 32 != 0),
            "duplicate_edges": duplicate_edges,
            "zero_in_degree_nodes": zero_degree,
            "operators": {
                "copy_e_sum": {
                    "max_abs_error": sum_abs,
                    "max_rel_error": sum_rel,
                    "all_finite": bool(
                        torch.isfinite(cand_sum).all()
                        and torch.isfinite(ref_sum).all()
                    ),
                    "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
                    "assert_close_passed": True,
                },
                "copy_e_mean": {
                    "max_abs_error": mean_abs,
                    "max_rel_error": mean_rel,
                    "all_finite": bool(
                        torch.isfinite(cand_mean).all()
                        and torch.isfinite(ref_mean).all()
                    ),
                    "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
                    "assert_close_passed": True,
                },
                "in_degrees": {
                    "exact_equal": bool(torch.equal(cand_deg, ref_deg)),
                    "max_abs_error": (
                        float((cand_deg - ref_deg).abs().max())
                        if cand_deg.numel()
                        else 0.0
                    ),
                },
            },
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
    (_EVIDENCE_PATH / "segment_reduce.json").write_text(
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


def _reference_reductions(src, dst, num_nodes, feat):
    graph = _reference_graph(src, dst, num_nodes)
    ref_sum = dgl.ops.copy_e_sum(graph, feat)
    ref_mean = dgl.ops.copy_e_mean(graph, feat)
    ref_deg = graph.in_degrees().to(torch.int64)
    return ref_sum, ref_mean, ref_deg


def _check_topology(case, topology):
    """Validate the native CSR table against the recorded ``dst`` on CPU.

    Returns the device tensors copied to CPU with their original dtypes (for the
    evidence record); the numeric checks use int64 views of the same values.
    """
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
# Parity case construction
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


def _random_feat(num_edges: int, trailing, seed: int, scale: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    if isinstance(trailing, int):
        trailing = (trailing,)
    feat = torch.randn((num_edges,) + tuple(trailing), generator=gen) * scale
    return feat.to(torch.float32)


def _case(name, description, num_nodes, src, dst, feat, provenance=None):
    return {
        "name": name,
        "description": description,
        "num_nodes": int(num_nodes),
        "src": src,
        "dst": dst,
        "feat": feat,
        "provenance": provenance or {},
    }


def _case_model_aligned_w32():
    num_edges, num_nodes, width = 1536, 128, 32
    src, dst = _random_graph(num_edges, num_nodes, 5001)
    feat = _random_feat(num_edges, width, 6001)
    return _case(
        "model_aligned_w32",
        "ordinary features, model-like aligned width 32",
        num_nodes,
        src,
        dst,
        feat,
    )


def _case_model_attention_wide96():
    num_edges, num_nodes = 1536, 128
    src, dst = _random_graph(num_edges, num_nodes, 5002)
    feat = _random_feat(num_edges, (4, 8, 3), 6002)
    return _case(
        "model_attention_wide96",
        "production attention reduction shape [E, 4, 8, 3] -> width 96",
        num_nodes,
        src,
        dst,
        feat,
        provenance={
            "shape_source": (
                "SE3-Transformer attention: n_heads=4, num_channels=32, "
                "channels_div=4 -> [E, 4, 8, 3] "
                "(include/SE3Transformer/se3_transformer/model/layers/attention.py)"
            )
        },
    )


def _case_duplicate_edges():
    pairs = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 3), (0, 3), (2, 2)]
    src = torch.tensor([pair[0] for pair in pairs] * 8, dtype=torch.int64)
    dst = torch.tensor([pair[1] for pair in pairs] * 8, dtype=torch.int64)
    feat = _random_feat(src.numel(), 16, 6003)
    return _case(
        "duplicate_edges",
        "repeated duplicate (src, dst) edges, width 16",
        6,
        src,
        dst,
        feat,
    )


def _case_zero_in_degree():
    num_nodes = 40
    src, dst = _random_graph(
        120, num_nodes, 5004, dst_choices=list(range(0, num_nodes, 2))
    )
    feat = _random_feat(src.numel(), 12, 6004)
    return _case(
        "zero_in_degree",
        "20 of 40 destinations have zero in-degree, width 12",
        num_nodes,
        src,
        dst,
        feat,
    )


def _case_unaligned_w20():
    num_edges, num_nodes, width = 257, 33, 20
    src, dst = _random_graph(num_edges, num_nodes, 5005)
    feat = _random_feat(num_edges, width, 6005)
    return _case(
        "unaligned_w20",
        "non-32-byte-aligned width 20 (80-byte rows)",
        num_nodes,
        src,
        dst,
        feat,
    )


def _case_unaligned_w39():
    num_edges, num_nodes, width = 333, 64, 39
    src, dst = _random_graph(num_edges, num_nodes, 5006)
    feat = _random_feat(num_edges, width, 6006)
    return _case(
        "unaligned_w39",
        "non-32-byte-aligned width 39 (156-byte rows)",
        num_nodes,
        src,
        dst,
        feat,
    )


def _case_empty_edges():
    num_nodes, width = 8, 8
    src = torch.empty(0, dtype=torch.int64)
    dst = torch.empty(0, dtype=torch.int64)
    feat = torch.empty(0, width, dtype=torch.float32)
    return _case(
        "empty_edges",
        "degenerate graph with no edges (E=0): zeros, one offset per node",
        num_nodes,
        src,
        dst,
        feat,
    )


def _case_single_node():
    num_edges, width = 9, 7
    src = torch.zeros(num_edges, dtype=torch.int64)
    dst = torch.zeros(num_edges, dtype=torch.int64)
    feat = _random_feat(num_edges, width, 6007)
    return _case(
        "single_node",
        "degenerate graph with a single node (N=1, all edges into it)",
        1,
        src,
        dst,
        feat,
    )


def _case_signed1000_destination_groups():
    # Destinations are assigned by argmax over two groups whose logits are
    # +1000 / -1000, the same grouping the distogram/probability head produces.
    # Features are integer-valued in [-1000, 1000], so every float32 partial sum
    # below 2**24 is exact and a zero-sum destination cannot hide a rounding
    # difference behind the absolute tolerance.
    num_nodes, width, num_edges = 64, 24, 2000
    group_logits = torch.full((num_nodes, 2), -1000.0, dtype=torch.float32)
    half = num_nodes // 2
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
    feat = torch.randint(
        -1000, 1001, (num_edges, width), generator=gen, dtype=torch.int64
    ).to(torch.float32)
    return _case(
        "signed1000_groups",
        "destinations from argmax over +/-1000 logit groups; integer-valued "
        "features with |feat| <= 1000, width 24",
        num_nodes,
        src,
        dst,
        feat,
        provenance={
            "grouping_source": (
                "destinations grouped by argmax over +1000/-1000 logits, the "
                "distogram/probability grouping used by the model"
            )
        },
    )


def _case_real_model_topk_w96():
    """Production topology and production attention reduction shape."""
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
    feat = _random_feat(int(src.numel()), (4, 8, 3), 6102)
    return _case(
        "real_model_topk_w96",
        "topology from rfantibody.rf2.network.util_module.make_topk_graph "
        "(kmin=32, top_k=128) with the production attention reduction shape "
        "[E, 4, 8, 3] -> width 96",
        int(graph.num_nodes()),
        src,
        dst,
        feat,
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
    pytest.param(_case_model_aligned_w32, id="model_aligned_w32"),
    pytest.param(_case_model_attention_wide96, id="model_attention_wide96"),
    pytest.param(_case_duplicate_edges, id="duplicate_edges"),
    pytest.param(_case_zero_in_degree, id="zero_in_degree"),
    pytest.param(_case_unaligned_w20, id="unaligned_w20"),
    pytest.param(_case_unaligned_w39, id="unaligned_w39"),
    pytest.param(_case_empty_edges, id="empty_edges"),
    pytest.param(_case_single_node, id="single_node"),
    pytest.param(_case_signed1000_destination_groups, id="signed1000_groups"),
    pytest.param(_case_real_model_topk_w96, id="real_model_topk_w96"),
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


def test_segment_reduce_native_artifacts(native_graph_ops, npu_device):
    """Compiled artifact symbols, ``PrivateUse1`` kernels and no CPU fallback."""
    artifact = Path(native_graph_ops["artifact"])
    assert artifact.is_file(), f"compiled artifact missing: {artifact}"
    symbols = _dynamic_symbols(artifact)
    for entry in (
        "aclrtlaunch_edge_topology",
        "aclrtlaunch_copy_e_sum",
        "aclrtlaunch_copy_e_mean",
    ):
        assert entry in symbols, f"generated route entry point missing: {entry}"

    for op in ("edge_topology", "copy_e_sum", "copy_e_mean"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            f"ascend_graph::{op}", "PrivateUse1"
        ), f"ascend_graph::{op} has no PrivateUse1 kernel"

    ops = native_graph_ops["ops"]
    cpu_dst = torch.tensor([0, 1, 0], dtype=torch.int64)
    cpu_feat = torch.ones(3, 4, dtype=torch.float32)
    cpu_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    cpu_offsets = torch.tensor([0, 2, 3], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        ops.build_edge_topology(cpu_dst, 2)
    with pytest.raises(RuntimeError):
        ops.copy_e_sum(cpu_feat, ops.EdgeTopology(cpu_offsets, cpu_indices, torch.tensor([2, 1])))
    with pytest.raises(RuntimeError):
        ops.copy_e_mean(cpu_feat, ops.EdgeTopology(cpu_offsets, cpu_indices, torch.tensor([2, 1])))
    with pytest.raises(RuntimeError):
        ops.in_degrees(cpu_dst, 2)
    with pytest.raises(ValueError):
        ops.build_edge_topology(cpu_dst, None)

    _EVIDENCE["artifacts"] = {
        "kernel_library": str(artifact),
        "kernel_library_sha256": native_graph_ops["artifact_sha256"],
        "python_binding": native_graph_ops["binding"],
        "python_binding_sha256": native_graph_ops["binding_sha256"],
        "exported_entry_points": [
            name
            for name in (
                "aclrtlaunch_edge_topology",
                "aclrtlaunch_copy_e_sum",
                "aclrtlaunch_copy_e_mean",
            )
            if name in symbols
        ],
        "dispatch": "PrivateUse1",
        "device": str(npu_device),
        "cpu_tensors_rejected": True,
    }


def test_segment_reduce_tiny_native_execution(
    native_graph_ops, npu_device, tmp_path
):
    """Hand-computed case (E=5, N=3, W=7) executed by the native kernels."""
    ops = native_graph_ops["ops"]
    feat = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=npu_device,
    )
    dst = torch.tensor([0, 2, 0, 1, 2], dtype=torch.int64, device=npu_device)

    topology = ops.build_edge_topology(dst, 3)
    total = ops.copy_e_sum(feat, topology)
    mean = ops.copy_e_mean(feat, topology)
    degrees = ops.in_degrees(topology)
    torch.npu.synchronize()

    for tensor in (total, mean, degrees):
        assert tensor.device.type == "npu", f"output device is {tensor.device}"
    assert tuple(total.shape) == (3, 7)
    assert tuple(mean.shape) == (3, 7)
    assert total.dtype == mean.dtype == torch.float32
    assert degrees.dtype == torch.int64

    expected_offsets = torch.tensor([0, 2, 3, 5], dtype=torch.int64)
    expected_indices = torch.tensor([0, 2, 3, 1, 4], dtype=torch.int32)
    expected_degrees = torch.tensor([2, 1, 2], dtype=torch.int64)
    expected_sum = torch.tensor(
        [
            [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    expected_mean = torch.tensor(
        [
            [0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )

    # Exactly representable values: assert exact equality, not just tolerance.
    torch.testing.assert_close(total.cpu(), expected_sum, atol=0.0, rtol=0.0)
    torch.testing.assert_close(mean.cpu(), expected_mean, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        topology.offsets.cpu(), expected_offsets, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        topology.indices.cpu(), expected_indices, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(degrees.cpu(), expected_degrees, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        topology.degrees.cpu(), expected_degrees, atol=0.0, rtol=0.0
    )

    trace = route_evidence.collect_level0(
        lambda: (ops.copy_e_sum(feat, topology), ops.copy_e_mean(feat, topology)),
        tmp_path / "segment_reduce_tiny_hardware.json",
    )
    torch.npu.synchronize()

    _EVIDENCE["tiny_native_execution"] = {
        "shapes": {"feat": [5, 7], "dst": [5]},
        "num_edges": 5,
        "num_nodes": 3,
        "expected": {
            "offsets": expected_offsets.tolist(),
            "indices": expected_indices.tolist(),
            "degrees": expected_degrees.tolist(),
            "copy_e_sum": expected_sum.tolist(),
            "copy_e_mean": expected_mean.tolist(),
        },
        "observed": {
            "offsets": topology.offsets.cpu().tolist(),
            "indices": topology.indices.cpu().tolist(),
            "degrees": degrees.cpu().tolist(),
            "copy_e_sum": total.cpu().tolist(),
            "copy_e_mean": mean.cpu().tolist(),
        },
        "exact_equality": True,
        "hardware_trace": trace,
    }


# --------------------------------------------------------------------------- #
# Parity against real DGL on CPU
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case_factory", _CASES)
def test_segment_reduce_parity_vs_cpu_dgl(native_graph_ops, npu_device, case_factory):
    """``copy_e_sum``/``copy_e_mean``/``in_degrees``: NPU kernel vs CPU DGL."""
    case = case_factory()
    src = case["src"]
    dst = case["dst"]
    feat = case["feat"]
    num_nodes = case["num_nodes"]

    reference_sum, reference_mean, reference_deg = _reference_reductions(
        src, dst, num_nodes, feat
    )

    ops = native_graph_ops["ops"]
    topology = ops.build_edge_topology(dst.to(npu_device), num_nodes)
    candidate_sum = ops.copy_e_sum(feat.to(npu_device), topology)
    candidate_mean = ops.copy_e_mean(feat.to(npu_device), topology)
    candidate_deg = ops.in_degrees(topology)
    torch.npu.synchronize()

    topology_cpu = _check_topology(case, topology)

    assert candidate_sum.device.type == "npu", (
        f"candidate device is {candidate_sum.device}"
    )
    assert reference_sum.device.type == "cpu", (
        f"reference device is {reference_sum.device}"
    )
    assert tuple(candidate_sum.shape) == tuple(reference_sum.shape), (
        f"{case['name']}: copy_e_sum candidate shape "
        f"{tuple(candidate_sum.shape)} != reference shape "
        f"{tuple(reference_sum.shape)}"
    )
    assert tuple(candidate_mean.shape) == tuple(reference_mean.shape)
    assert candidate_sum.dtype == reference_sum.dtype == torch.float32
    assert candidate_mean.dtype == reference_mean.dtype == torch.float32

    cand_sum_cpu = candidate_sum.cpu()
    cand_mean_cpu = candidate_mean.cpu()
    assert torch.isfinite(cand_sum_cpu).all(), f"{case['name']}: non-finite sum"
    assert torch.isfinite(cand_mean_cpu).all(), f"{case['name']}: non-finite mean"
    assert torch.isfinite(reference_sum).all()
    assert torch.isfinite(reference_mean).all()
    torch.testing.assert_close(
        cand_sum_cpu, reference_sum, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )
    torch.testing.assert_close(
        cand_mean_cpu, reference_mean, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )

    cand_deg_cpu = candidate_deg.cpu().to(torch.int64)
    assert torch.equal(cand_deg_cpu, reference_deg.to(torch.int64)), (
        f"{case['name']}: in-degrees differ from DGL"
    )

    _record_case(
        case,
        topology_cpu,
        candidate_sum,
        candidate_mean,
        candidate_deg,
        reference_sum,
        reference_mean,
        reference_deg,
    )


# --------------------------------------------------------------------------- #
# Hardware / report route evidence
# --------------------------------------------------------------------------- #
def test_segment_reduce_route_evidence(native_graph_ops, npu_device, tmp_path):
    """Four native launches, four ``KERNEL_*`` tasks, three report kernel names."""
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
            "ai_core_op_summary_rows": len(
                ai_core_rows.get("op_summary") or []
            ),
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
    (_EVIDENCE_PATH / "segment_reduce_route_evidence.json").write_text(
        json.dumps(route_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Channel 1: exactly one hardware task per native launch.
    assert hardware.get("hardware_event_count") == _EXPECTED_LAUNCHES, (
        "expected exactly "
        f"{_EXPECTED_LAUNCHES} KERNEL_* hardware task events for the four native "
        f"launches, got {hardware.get('hardware_event_count')}: "
        f"{json.dumps(hardware.get('task_counts'), sort_keys=True)} "
        f"(trace: {hardware.get('trace')})"
    )
    assert hardware.get("task_counts"), "hardware channel recorded no task counts"
    assert all(
        str(key).split("|")[-1].startswith("KERNEL_")
        for key in hardware["task_counts"]
    ), f"non-hardware task type counted: {hardware['task_counts']}"

    # Channel 2: the vendor report names the three kernels of this bundle.
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
            f"kernel name {name!r} resolved without a channel: "
            f"{resolved[name]}"
        )
    assert report.get("negative_resolved") == [], (
        f"negative control token {report.get('negative_token')!r} resolved to "
        f"{report.get('negative_resolved')}"
    )
