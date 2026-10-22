"""Native AscendC route control and DGL ``e_dot_v`` parity on the NPU.

Three things are checked against fixed, pre-declared criteria:

* the compiled artifact for the observed SoC exists and exports the generated
  ``aclrtlaunch_*`` entry points,
* the public API executes a hardware kernel after registering on ``PrivateUse1``.
  In this vendor build the exported chrome trace names a hardware task event by
  its CANN *task type* (``KERNEL_AIVEC``) and not by the kernel name, so the
  route proof uses two measured channels collected for the same call in two
  fresh interpreters: a probe-identical profile whose trace must contain exactly
  one ``KERNEL_*`` hardware task event, and an op-level profile whose vendor
  report CSVs must contain exactly one ``AI_CORE`` row named ``edge_dot_v``
  while the negative token ``__no_such_kernel__`` resolves nothing.  A separate
  native AscendC kernel (``ascend_graph::graph_probe``, positive control) and a
  CPU-only window (negative control) are measured with the identical harness so a
  missing event is attributable to the operator rather than to the environment;
  a builtin aclnn op is measured too, but only recorded as a vendor-naming
  observation (this build exports builtin device tasks with ``Task Type: N/A``).
* ``ascend_graph::e_dot_v`` on ``npu:0`` matches **real DGL 1.1.3
  ``dgl.ops.e_dot_v`` on CPU** (an independent reference graph/tensor set built
  from the same arrays), including duplicate edges, zero-in-degree nodes, the
  topology and head/feature shape reached in the real model, and non-32-byte
  aligned widths.

Tolerances are declared here, before any candidate result is inspected, and are
never relaxed afterwards: ``atol=1e-5``, ``rtol=1e-4`` (float32).  Passing this
operator parity does not by itself establish full-model parity.
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
ROUTE_ATOL = 1e-6
ROUTE_RTOL = 1e-6

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_PATH = Path(
    os.environ.get("RF_EVIDENCE_DIR") or (_PROJECT_ROOT / "migration_reports")
) / "graph_parity"

_TRACE_CHANNEL_NOTE = (
    "In this vendor build (CANN 8.1.RC1 / torch_npu 2.5.1) the exported chrome "
    "trace names hardware task events by their CANN task type (KERNEL_AIVEC) and "
    "carries no kernel name.  The route gate is therefore split into two measured "
    "channels for the same call, each recorded in a fresh interpreter: the "
    "KERNEL_* hardware task event of a probe-identical profile, and the kernel "
    "name from the op-level vendor report CSVs (op_summary Op Name / task_time "
    "kernel_name).  The positive control is a separate native AscendC kernel "
    "(ascend_graph::graph_probe) and the negative control a CPU-only window, both "
    "measured with the identical harness; a builtin aclnn op is recorded as a "
    "vendor-naming observation because this build exports its device tasks with "
    "Task Type 'N/A'."
)

_EVIDENCE = {
    "operator": "dgl.ops.e_dot_v",
    "candidate": (
        "ascend_graph::e_dot_v -> native AscendC kernel edge_dot_v, executed on npu:0"
    ),
    "reference": (
        "dgl {} dgl.ops.e_dot_v on CPU; the graph and tensors are built from the "
        "same src/dst/key/query arrays (independent reference, never the candidate)"
    ).format(dgl.__version__),
    "tolerances": {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "max_abs_error_definition": "max |candidate - reference| over all elements",
        "max_rel_error_definition": (
            "max |candidate - reference| / max(|reference|, 1e-12) over all elements"
        ),
    },
    "route_control": {"note": _TRACE_CHANNEL_NOTE},
    "tiny_native_execution": {},
    "kernel_identity": {},
    "cases": [],
    "pending_operator_families": [
        "edge_softmax(norm_by='dst'): per-destination softmax sum-to-one and "
        "+/-1000 group stress are asserted in the edge_softmax iteration",
        "copy_e_sum / copy_e_mean / AvgPooling destination reductions",
        "in_degrees and graph construction/edges/edata plumbing",
    ],
}


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #
def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    return _sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _record_case(case: dict, src, dst, num_nodes, key, query, candidate, reference):
    """Append one machine-readable parity row and persist its exact inputs."""
    out_dir = _EVIDENCE_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_cpu = candidate.detach().cpu()
    ref_cpu = reference.detach().cpu()
    diff = (cand_cpu - ref_cpu).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    if ref_cpu.numel():
        rel = diff / ref_cpu.abs().clamp_min(1e-12)
        max_rel = float(rel.max())
    else:
        max_rel = 0.0

    npz_path = out_dir / f"e_dot_v_{case['name']}.npz"
    np.savez(
        npz_path,
        src=src.numpy(),
        dst=dst.numpy(),
        key=key.numpy(),
        query=query.numpy(),
        reference=ref_cpu.numpy(),
        candidate=cand_cpu.numpy(),
    )

    _EVIDENCE["cases"].append(
        {
            "case": case["name"],
            "description": case["description"],
            "provenance": case.get("provenance", {}),
            "shapes": {
                "key": list(key.shape),
                "query": list(query.shape),
                "dst": list(dst.shape),
                "num_nodes": int(num_nodes),
                "num_heads": int(case["num_heads"]),
                "num_feat": int(case["num_feat"]),
                "num_edges": int(key.shape[0]),
            },
            "dtypes": {
                "key": str(key.dtype),
                "query": str(query.dtype),
                "dst": str(dst.dtype),
                "reference": str(ref_cpu.dtype),
                "candidate": str(candidate.dtype),
            },
            "devices": {
                "candidate": str(candidate.device),
                "reference": str(reference.device),
                "reference_identity": (
                    "dgl {} dgl.ops.e_dot_v on CPU".format(dgl.__version__)
                ),
            },
            "input_sha256": {
                "src": _tensor_digest(src),
                "dst": _tensor_digest(dst),
                "key": _tensor_digest(key),
                "query": _tensor_digest(query),
            },
            "non_32_byte_aligned": bool(
                (int(case["num_heads"]) * int(case["num_feat"])) % 8 != 0
            ),
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "all_finite": bool(torch.isfinite(cand_cpu).all()),
            "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
            "assert_close_passed": True,
            "inputs_npz": str(npz_path),
        }
    )


@pytest.fixture(scope="module", autouse=True)
def _flush_evidence():
    """Write the operator-parity table even if a case fails."""
    yield
    if not _EVIDENCE["cases"]:
        return
    _EVIDENCE_PATH.mkdir(parents=True, exist_ok=True)
    _EVIDENCE["generated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    (_EVIDENCE_PATH / "e_dot_v.json").write_text(
        json.dumps(_EVIDENCE, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# Route-control helpers
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


def _assert_one_hardware_kernel(channel: dict, label: str):
    """Exactly one ``KERNEL_*`` hardware task for a call that issues one kernel."""
    assert channel["hardware_event_count"] == 1, (
        f"{label}: expected exactly one KERNEL_* hardware task event for a single "
        f"kernel launch, got {channel['hardware_event_count']}: "
        f"{json.dumps(channel['task_counts'], sort_keys=True)} "
        f"(trace: {channel['trace']})"
    )
    event = channel["events"][0]
    assert str(event["task_type"]).startswith("KERNEL_"), event
    return event


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


def _random_features(
    num_edges: int,
    num_nodes: int,
    num_heads: int,
    num_feat: int,
    seed: int,
    key_scale: float = 1.0,
    query_scale: float = 1.0,
):
    gen = torch.Generator().manual_seed(seed)
    key = torch.randn(num_edges, num_heads, num_feat, generator=gen) * key_scale
    query = torch.randn(num_nodes, num_heads, num_feat, generator=gen) * query_scale
    return key.to(torch.float32), query.to(torch.float32)


def _case(
    name,
    description,
    num_heads,
    num_feat,
    num_nodes,
    src,
    dst,
    key,
    query,
    provenance=None,
):
    return {
        "name": name,
        "description": description,
        "num_heads": num_heads,
        "num_feat": num_feat,
        "num_nodes": num_nodes,
        "src": src,
        "dst": dst,
        "key": key,
        "query": query,
        "provenance": provenance or {},
    }


def _case_model_aligned_h4_f32():
    # Real-model-like attention shape: 4 heads, reduced width 32 (32-byte rows).
    num_edges, num_nodes, heads, feat = 1536, 128, 4, 32
    src, dst = _random_graph(num_edges, num_nodes, 1001)
    key, query = _random_features(num_edges, num_nodes, heads, feat, 2001)
    return _case(
        "model_aligned_h4_f32",
        "ordinary logits, model-like aligned shape (H=4, F=32)",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_model_aligned_h1_f64():
    num_edges, num_nodes, heads, feat = 2048, 128, 1, 64
    src, dst = _random_graph(num_edges, num_nodes, 1002)
    key, query = _random_features(num_edges, num_nodes, heads, feat, 2002)
    return _case(
        "model_aligned_h1_f64",
        "ordinary logits, single head aligned shape (H=1, F=64)",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_duplicate_edges():
    pairs = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 3), (0, 3), (2, 2)]
    src = torch.tensor([pair[0] for pair in pairs] * 8, dtype=torch.int64)
    dst = torch.tensor([pair[1] for pair in pairs] * 8, dtype=torch.int64)
    heads, feat, num_nodes = 2, 8, 6
    key, query = _random_features(src.numel(), num_nodes, heads, feat, 2003)
    return _case(
        "duplicate_edges",
        "repeated duplicate (src,dst) edges, H=2, F=8",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_zero_in_degree():
    num_nodes, heads, feat = 40, 3, 4
    # Only even destinations receive edges; odd nodes have zero in-degree.
    src, dst = _random_graph(120, num_nodes, 1004, dst_choices=list(range(0, num_nodes, 2)))
    key, query = _random_features(src.numel(), num_nodes, heads, feat, 2004)
    return _case(
        "zero_in_degree",
        "20 of 40 nodes have zero in-degree, H=3, F=4",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_unaligned_h4_f5():
    num_edges, num_nodes, heads, feat = 257, 33, 4, 5
    src, dst = _random_graph(num_edges, num_nodes, 1005)
    key, query = _random_features(num_edges, num_nodes, heads, feat, 2005)
    return _case(
        "unaligned_h4_f5",
        "non-32-byte-aligned flattened width (H=4, F=5 -> width 20)",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_unaligned_h1_f3():
    num_edges, num_nodes, heads, feat = 101, 17, 1, 3
    src, dst = _random_graph(num_edges, num_nodes, 1006)
    key, query = _random_features(num_edges, num_nodes, heads, feat, 2006)
    return _case(
        "unaligned_h1_f3",
        "non-32-byte-aligned flattened width (H=1, F=3)",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_unaligned_h3_f13():
    num_edges, num_nodes, heads, feat = 333, 64, 3, 13
    src, dst = _random_graph(num_edges, num_nodes, 1007)
    key, query = _random_features(num_edges, num_nodes, heads, feat, 2007)
    return _case(
        "unaligned_h3_f13",
        "non-32-byte-aligned flattened width (H=3, F=13 -> width 39)",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_empty_edges():
    num_nodes, heads, feat = 8, 2, 4
    src, dst = _random_graph(0, num_nodes, 1008)
    key, query = _random_features(0, num_nodes, heads, feat, 2008)
    return _case(
        "empty_edges",
        "degenerate graph with no edges (E=0)",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_single_node():
    num_edges, heads, feat = 9, 2, 8
    # N=1: every edge endpoint is node 0.  The previous version used
    # src = arange(num_edges), which made the DGL reference itself invalid
    # ("num_nodes argument must be larger than the max ID in the data"), so the
    # case could never compare anything.  The purpose is unchanged: a degenerate
    # graph with a single destination and no other node.
    src = torch.zeros(num_edges, dtype=torch.int64)
    dst = torch.zeros(num_edges, dtype=torch.int64)
    key, query = _random_features(num_edges, 1, heads, feat, 2009)
    return _case(
        "single_node",
        "degenerate graph with a single node (N=1, all edges into it)",
        heads,
        feat,
        1,
        src,
        dst,
        key,
        query,
    )


def _case_signed1000_destination_groups():
    # Destinations are assigned by argmax over two groups whose logits are
    # +1000 / -1000, the same grouping the distogram/probability head produces.
    num_nodes, heads, feat, num_edges = 64, 2, 16, 2000
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
    # Scaled so products stay O(1): |key| ~ 1e3, |query| ~ 1e-3.
    key, query = _random_features(
        num_edges, num_nodes, heads, feat, 2010, key_scale=1e3, query_scale=1e-3
    )
    return _case(
        "signed1000_destination_groups",
        "destinations from argmax over +/-1000 logit groups; |key|~1e3, |query|~1e-3",
        heads,
        feat,
        num_nodes,
        src,
        dst,
        key,
        query,
    )


def _case_real_model_topk_h4_f8():
    """Production topology and production attention shape.

    The graph is built by the project's own RF2 graph builder
    (``make_topk_graph``, ``kmin=32``, ``top_k=128``) from a deterministic CA
    trace, and the features use the SE3-Transformer attention shape of the real
    model (``n_heads=4``, ``num_channels=32``, ``channels_div=4`` -> per-head
    width 8).  If the production builder cannot be imported or used, this case
    fails instead of silently degrading to a hand-made graph.
    """
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
    # Only the shape of `pair` is consumed by the graph builder.
    pair = torch.zeros(batch, length, length, 8, dtype=torch.float32)
    residue_index = torch.arange(length, dtype=torch.int64).reshape(batch, length)
    graph, _ = make_topk_graph(xyz, pair, residue_index)
    src, dst = graph.edges()
    heads, feat = 4, 8
    key, query = _random_features(
        int(src.numel()), int(graph.num_nodes()), heads, feat, 4102
    )
    return _case(
        "real_model_topk_h4_f8",
        "topology from rfantibody.rf2.network.util_module.make_topk_graph "
        "(kmin=32, top_k=128) with the production attention shape H=4, F=8",
        heads,
        feat,
        int(graph.num_nodes()),
        src,
        dst,
        key,
        query,
        provenance={
            "topology_source": (
                "rfantibody.rf2.network.util_module.make_topk_graph (kmin=32, "
                "top_k=128, B=1, L=120)"
            ),
            "model_shape_source": (
                "SE3-Transformer attention: n_heads=4, num_channels=32, "
                "channels_div=4 -> per-head width 8 "
                "(include/SE3Transformer/se3_transformer/model/layers/attention.py)"
            ),
            "graph_num_nodes": int(graph.num_nodes()),
            "graph_num_edges": int(src.numel()),
        },
    )


_CASES = [
    pytest.param(_case_model_aligned_h4_f32, id="model_aligned_h4_f32"),
    pytest.param(_case_model_aligned_h1_f64, id="model_aligned_h1_f64"),
    pytest.param(_case_duplicate_edges, id="duplicate_edges"),
    pytest.param(_case_zero_in_degree, id="zero_in_degree"),
    pytest.param(_case_unaligned_h4_f5, id="unaligned_h4_f5"),
    pytest.param(_case_unaligned_h1_f3, id="unaligned_h1_f3"),
    pytest.param(_case_unaligned_h3_f13, id="unaligned_h3_f13"),
    pytest.param(_case_empty_edges, id="empty_edges"),
    pytest.param(_case_single_node, id="single_node"),
    pytest.param(
        _case_signed1000_destination_groups, id="signed1000_destination_groups"
    ),
    pytest.param(_case_real_model_topk_h4_f8, id="real_model_topk_h4_f8"),
]


def _reference_e_dot_v(src, dst, num_nodes, key, query):
    """Independent reference: real DGL 1.1.3 ``dgl.ops.e_dot_v`` on CPU."""
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    # The reference graph must really be the requested one (this is what the
    # single_node case asserts explicitly: one node, every edge into it).
    assert int(graph.num_nodes()) == int(num_nodes), (
        f"reference graph has {int(graph.num_nodes())} nodes, "
        f"expected num_nodes={int(num_nodes)}"
    )
    ref_src, ref_dst = graph.edges()
    assert torch.equal(ref_src, src), "DGL graph source edge order changed"
    assert torch.equal(ref_dst, dst), "DGL graph destination edge order changed"
    return dgl.ops.e_dot_v(graph, key, query)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_native_route_control(native_graph_ops, npu_device, tmp_path):
    """Build -> load -> register -> launch -> sync -> copy back for graph_probe."""
    artifact = Path(native_graph_ops["artifact"])
    assert artifact.is_file(), f"compiled artifact missing: {artifact}"
    symbols = _dynamic_symbols(artifact)
    assert "aclrtlaunch_graph_probe" in symbols, "generated route entry point missing"
    assert "aclrtlaunch_edge_dot_v" in symbols, "generated route entry point missing"
    assert len(native_graph_ops["artifact_sha256"]) == 64

    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "ascend_graph::graph_probe", "PrivateUse1"
    ), "ascend_graph::graph_probe has no PrivateUse1 kernel"

    ops = native_graph_ops["ops"]
    x = torch.arange(37, dtype=torch.float32, device=npu_device)
    y = ops.graph_probe(x)
    torch.npu.synchronize()

    assert y.device.type == "npu", f"graph_probe output device is {y.device}"
    assert y.dtype == torch.float32
    assert tuple(y.shape) == (37,)
    assert torch.isfinite(y).all(), "graph_probe produced NaN/Inf"
    reference = torch.arange(37, dtype=torch.float32) + 1.0
    torch.testing.assert_close(
        y.cpu(), reference, atol=ROUTE_ATOL, rtol=ROUTE_RTOL
    )

    control = route_evidence.collect_level0(
        lambda: ops.graph_probe(x), tmp_path / "graph_probe_trace.json"
    )
    event = _assert_one_hardware_kernel(control, "graph_probe")
    _EVIDENCE["route_control"] = {
        "note": _TRACE_CHANNEL_NOTE,
        "artifacts": {
            "kernel_library": str(artifact),
            "kernel_library_sha256": native_graph_ops["artifact_sha256"],
            "python_binding": native_graph_ops["binding"],
            "python_binding_sha256": native_graph_ops["binding_sha256"],
            "exported_entry_points": [
                name
                for name in ("aclrtlaunch_graph_probe", "aclrtlaunch_edge_dot_v")
                if name in symbols
            ],
        },
        "dispatch": "PrivateUse1",
        "device": str(npu_device),
        "numerical_check": {"expected": "x + 1", "atol": ROUTE_ATOL, "rtol": ROUTE_RTOL},
        "trace": control,
        "hardware_task": event,
    }

    # A non-NPU tensor must not silently take this route.
    with pytest.raises(RuntimeError):
        ops.graph_probe(torch.arange(4, dtype=torch.float32))


def test_e_dot_v_tiny_native_execution(native_graph_ops, npu_device, tmp_path):
    """Bounded hand-computed case executed by the native kernel on npu:0."""
    ops = native_graph_ops["ops"]
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "ascend_graph::e_dot_v", "PrivateUse1"
    ), "ascend_graph::e_dot_v has no PrivateUse1 kernel"

    key = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, 0.5, 2.0], [3.0, -2.0, 1.0]],
        ],
        dtype=torch.float32,
        device=npu_device,
    )
    query = torch.tensor(
        [
            [[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.5, -1.0, 2.0], [1.0, 2.0, 3.0]],
        ],
        dtype=torch.float32,
        device=npu_device,
    )
    dst = torch.tensor([0, 2], dtype=torch.int64, device=npu_device)
    out = ops.e_dot_v(key, query, dst, 3)
    torch.npu.synchronize()

    assert out.device.type == "npu", f"e_dot_v output device is {out.device}"
    assert out.dtype == torch.float32
    assert tuple(out.shape) == (2, 2, 1)
    expected = torch.tensor(
        [[[6.0], [14.0]], [[3.0], [2.0]]], dtype=torch.float32
    )
    # Kernel/DGL shape is (E, H, 1); compare like with like (no reshape) on CPU,
    # exactly like the parity cases below.
    torch.testing.assert_close(
        out.cpu(), expected, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )

    tiny = route_evidence.collect_level0(
        lambda: ops.e_dot_v(key, query, dst, 3), tmp_path / "e_dot_v_trace.json"
    )
    event = _assert_one_hardware_kernel(tiny, "e_dot_v")
    _EVIDENCE["tiny_native_execution"] = {
        "note": _TRACE_CHANNEL_NOTE,
        "shapes": {"key": list(key.shape), "query": list(query.shape), "dst": [2]},
        "num_heads": 2,
        "num_feat": 3,
        "expected": expected.tolist(),
        "observed": out.cpu().tolist(),
        "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        "trace": tiny,
        "hardware_task": event,
    }

    # Wrong dtype must be rejected by the host bridge (no silent conversion).
    with pytest.raises(RuntimeError):
        ops.e_dot_v(
            key.to(torch.float16), query.to(torch.float16), dst, 3
        )


def test_e_dot_v_kernel_identity(native_graph_ops, npu_device):
    """The public-API call reaches one hardware task and is named ``edge_dot_v``.

    Two measured channels of the *same* public call, each collected in its own
    fresh interpreter, plus instrumentation controls:

    * ``hardware``: probe-identical profile; the chrome trace must contain exactly
      one ``KERNEL_*`` hardware task event for a call that issues one kernel.
    * ``identity``: op-level profile; the vendor report CSVs must contain exactly
      one ``AI_CORE`` row whose ``Op Name`` (``op_summary*.csv``) and
      ``kernel_name`` (``task_time*.csv``) both equal ``edge_dot_v``, while the
      negative token ``__no_such_kernel__`` resolves nowhere in the same
      artefacts.
    * positive control: a *separate* native AscendC kernel (``ascend_graph::
      graph_probe``, ``x + 1``, own kernel body and generated
      ``aclrtlaunch_graph_probe`` entry point) measured with the identical harness
      must also show a ``KERNEL_*`` event, so a missing event for the custom op is
      an operator finding and not a broken environment.  A builtin aclnn op is
      measured as well and recorded as a vendor-naming observation; it never
      gates, because this build exports builtin device tasks with
      ``Task Type: "N/A"`` and therefore cannot produce a ``KERNEL_*`` event.
    * negative control: a CPU-only window must show none, so a counted event is a
      real device task and not profiler noise.

    The exported chrome trace of this vendor build carries the CANN task type as
    the event name and never the kernel name; that measured naming fact is
    recorded verbatim (``trace_event_names``) instead of being asserted away.
    """
    del npu_device  # only used to keep the device requirement explicit
    evidence = route_evidence.collect_route_identity(route_evidence.DEFAULT_CASE)

    # The host-owned, unmodified probe is invoked twice and both invocations are
    # recorded verbatim: once with this build's hardware task type and once with
    # the kernel-name pattern.  Only the task-type invocation is asserted; the
    # kernel-name invocation is an environment finding, not an operator gate.
    probe_task_type = route_evidence.run_npu_kernel_probe(
        kernel=route_evidence.KERNEL_TASK_PREFIX + "AIVEC"
    )
    route_evidence.record_probe_invocation(probe_task_type)
    probe_kernel_name = route_evidence.run_npu_kernel_probe(
        kernel=route_evidence.EXPECTED_KERNEL_TOKEN
    )
    route_evidence.record_probe_invocation(probe_kernel_name)

    hardware = evidence["hardware"]
    identity = evidence["identity"]
    positive = evidence["positive_instrumentation_control"]
    negative = evidence["negative_instrumentation_control"]
    builtin = evidence["builtin_naming_observation"]

    def _probe_record(record):
        return {
            key: record.get(key)
            for key in (
                "available",
                "exit_code",
                "kernel_argument",
                "summary",
                "log",
                "command",
                "note",
            )
        }

    _EVIDENCE["kernel_identity"] = {
        "case": evidence["case"],
        "isolation": evidence["isolation"],
        "profiler_levels": {
            "hardware": hardware.get("profiler_level"),
            "identity": identity.get("profiler_level"),
        },
        "hardware_channel": {
            "count": hardware.get("count"),
            "events": hardware.get("events"),
            "task_counts": hardware.get("task_counts"),
            "trace": hardware.get("trace"),
            "log": hardware.get("log"),
        },
        "report_channel": {
            "ai_core_rows": identity.get("ai_core_rows"),
            "op_names": identity.get("op_names"),
            "kernel_names": identity.get("kernel_names"),
            "resolved_names": identity.get("resolved_names"),
            "resolved_channels": identity.get("resolved_channels"),
            "resolved_rows": identity.get("resolved_rows"),
            "negative_token": identity.get("negative_token"),
            "negative_resolved": identity.get("negative_resolved"),
            "trace_events_observed": identity.get("trace_events_observed"),
            "trace": identity.get("trace"),
            "log": identity.get("log"),
        },
        "positive_instrumentation_control": {
            "case": positive.get("case"),
            "count": positive.get("count"),
            "traced_task_types": positive.get("traced_task_types"),
            "trace_event_names": positive.get("trace_event_names"),
            "trace": positive.get("trace"),
            "log": positive.get("log"),
            "note": positive.get("note"),
        },
        "builtin_naming_observation": {
            "case": builtin.get("case"),
            "count": builtin.get("count"),
            "traced_task_types": builtin.get("traced_task_types"),
            "trace_event_names": builtin.get("trace_event_names"),
            "trace": builtin.get("trace"),
            "log": builtin.get("log"),
            "note": builtin.get("note"),
        },
        "negative_instrumentation_control": {
            "case": negative.get("case"),
            "count": negative.get("count"),
            "trace_event_names": negative.get("trace_event_names"),
            "trace": negative.get("trace"),
            "log": negative.get("log"),
            "note": negative.get("note"),
        },
        "probe_invocations": [
            _probe_record(probe_task_type),
            _probe_record(probe_kernel_name),
        ],
        "route_evidence_file": evidence.get("evidence_file"),
    }

    # Channel 1: exactly one hardware task for a call that issues one kernel.
    assert hardware.get("count") == 1, (
        "expected exactly one KERNEL_* hardware task event for the profiled "
        f"e_dot_v call, got {hardware.get('count')}: "
        f"{json.dumps(hardware.get('task_counts'), sort_keys=True)} "
        f"(trace: {hardware.get('trace')})"
    )
    assert hardware.get("task_counts"), "hardware channel recorded no task counts"
    assert all(
        str(key).split("|")[-1].startswith("KERNEL_")
        for key in hardware["task_counts"]
    ), f"non-hardware task type counted: {hardware['task_counts']}"

    # Channel 2: the vendor report channel names the kernel for the same call.
    ai_core_rows = identity.get("ai_core_rows") or {}
    op_rows = ai_core_rows.get("op_summary") or []
    task_rows = ai_core_rows.get("task_time") or []
    assert len(op_rows) == 1, (
        "expected exactly one AI_CORE row in the op_summary report for the "
        f"profiled call, got {len(op_rows)}: {json.dumps(op_rows, sort_keys=True)}"
    )
    assert op_rows[0].get("Op Name") == route_evidence.EXPECTED_KERNEL_TOKEN, (
        f"op_summary Op Name is {op_rows[0].get('Op Name')!r}, expected "
        f"{route_evidence.EXPECTED_KERNEL_TOKEN!r}"
    )
    assert str(op_rows[0].get("Task Type", "")).upper() == "AI_CORE", op_rows[0]
    assert len(task_rows) == 1, (
        "expected exactly one AI_CORE row in the task_time report for the "
        f"profiled call, got {len(task_rows)}: {json.dumps(task_rows, sort_keys=True)}"
    )
    assert task_rows[0].get("kernel_name") == route_evidence.EXPECTED_KERNEL_TOKEN, (
        f"task_time kernel_name is {task_rows[0].get('kernel_name')!r}, expected "
        f"{route_evidence.EXPECTED_KERNEL_TOKEN!r}"
    )
    assert str(task_rows[0].get("kernel_type", "")).upper() == "AI_CORE", task_rows[0]
    assert identity.get("resolved_channels"), (
        "the vendor report/hash channel recorded no channel for the kernel name token; "
        f"artifacts: {identity.get('profile_artifacts')}"
    )
    assert identity.get("negative_resolved") == [], (
        f"negative control token {identity.get('negative_token')!r} resolved to "
        f"{identity.get('negative_resolved')}"
    )

    # Instrumentation controls: the harness can see native hardware work when it
    # exists and reports nothing when there is none.
    # Positive control: a separate native AscendC kernel must be observed.
    assert positive.get("case") == route_evidence.POSITIVE_CONTROL_CASE, (
        positive.get("case")
    )
    assert positive.get("count", 0) >= 1, (
        "positive instrumentation control (native graph_probe kernel on npu:0) produced "
        f"no KERNEL_* hardware task event; harness is not trustworthy here "
        f"(trace: {positive.get('trace')})"
    )
    assert all(
        str(event.get("task_type", "")).startswith("KERNEL_")
        for event in positive.get("events") or []
    ), positive.get("events")
    # Recorded, not gated: builtin aclnn tasks are exported with Task Type "N/A".
    assert "count" in builtin, builtin
    assert negative.get("count") == 0, (
        "negative instrumentation control (CPU-only window) produced "
        f"{negative.get('count')} KERNEL_* hardware task event(s): "
        f"{json.dumps(negative.get('task_counts'), sort_keys=True)} "
        f"(trace: {negative.get('trace')})"
    )

    # The host-owned, unmodified probe is exercised with this vendor build's
    # hardware task type.  When the host probe is not part of the tree (for
    # example in a standalone copy) the two native channels above stay the gate.
    if probe_task_type["available"]:
        assert probe_task_type["exit_code"] == 0, (
            f"npu_kernel_probe failed ({probe_task_type['exit_code']}); "
            f"see {probe_task_type['log']}"
        )
        summary = probe_task_type["summary"] or {}
        assert summary.get("status") == "pass", summary
        assert str(summary.get("dispatch")) == "PrivateUse1", summary
        assert str(summary.get("task_type", "")).startswith("KERNEL_"), summary
        assert all(
            str(device).startswith("npu") for device in summary.get("output_devices", [])
        ), summary

    # The kernel-name probe invocation is recorded verbatim above; it is not
    # asserted because this vendor chrome trace carries task types only.  A
    # regression that starts carrying kernel names is visible in the evidence
    # file (``probe_invocations[1]``) without weakening any operator gate.


@pytest.mark.parametrize("case_factory", _CASES)
def test_e_dot_v_parity_vs_cpu_dgl(native_graph_ops, npu_device, case_factory):
    """``e_dot_v`` parity: native kernel on npu:0 vs real DGL on CPU."""
    case = case_factory()
    src = case["src"]
    dst = case["dst"]
    key = case["key"]
    query = case["query"]
    num_nodes = case["num_nodes"]

    reference = _reference_e_dot_v(src, dst, num_nodes, key, query)

    ops = native_graph_ops["ops"]
    candidate = ops.e_dot_v(key.to(npu_device), query.to(npu_device), dst.to(npu_device))
    torch.npu.synchronize()

    assert candidate.device.type == "npu", f"candidate device is {candidate.device}"
    assert reference.device.type == "cpu", f"reference device is {reference.device}"
    assert tuple(candidate.shape) == tuple(reference.shape), (
        f"{case['name']}: candidate shape {tuple(candidate.shape)} != "
        f"reference shape {tuple(reference.shape)}"
    )
    assert candidate.dtype == reference.dtype == torch.float32

    cand_cpu = candidate.cpu()
    assert torch.isfinite(cand_cpu).all(), f"{case['name']}: non-finite output"
    torch.testing.assert_close(
        cand_cpu, reference, atol=PARITY_ATOL, rtol=PARITY_RTOL
    )

    _record_case(
        case, src, dst, num_nodes, key, query, candidate, reference
    )
