#!/usr/bin/env python3
"""End-to-end RFantibody acceptance workflow: RFdiffusion -> ProteinMPNN -> RF2.

The runner mirrors the repository's own test surface (``test/rfdiffusion/scripts``,
``test/proteinmpnn/scripts`` and ``test/rf2/scripts``): the same installed CLI
entry points, the same input assets and the same deterministic settings are used
for the three-stage chain, and each stage consumes the previous stage's output.

The default workload is the acceptance workload:

1. RFdiffusion designs one antibody backbone against the repository RSV site 3
   target with the ``hu-4D5-8_Fv`` framework, hotspots ``T305,T456`` and loop
   lengths ``L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13``; ``diffuser.T=50``,
   ``final_step=1``, deterministic mode.
2. ProteinMPNN redesigns loops ``H1,H2,H3,L1,L2,L3`` with
   ``seqs_per_struct=1``, ``temperature=0.2``, ``augment_eps=0``.
3. RF2 runs its normal ``10`` recycling iterations (11 forward cycles).

No silent CPU fallback is accepted.  The runner requires a working ``torch_npu``
device and a real NPU tensor smoke test before any stage, and every stage runs
with ``e2e_support/sitecustomize.py`` injected through ``PYTHONPATH``.  That
guard records module placement and module-forward output devices per process
(including child processes) and a stage without NPU forward activity fails.

Weights, inputs and outputs are resolved from explicit flags, documented
environment variables (``RFANTIBODY_WEIGHTS``, ``RFANTIBODY_ROOT``,
``RFANTIBODY_SCRIPTS``) and paths inside this repository only.

Every stage records its exact command, exit code, wall time, input/output
SHA-256 provenance, device evidence and log path.  Machine-readable results are
written to ``<output-dir>/e2e_result.json``.  The exit code is 0 only when all
stage gates and all chain-level gates pass.

Usage (from the repository root, inside the project environment)::

    .venv/bin/python e2e_run.py
    .venv/bin/python e2e_run.py --weights-dir /path/to/weights
    .venv/bin/python e2e_run.py --output-dir outputs/chain_001
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
GUARD_DIR = PROJECT_ROOT / "e2e_support"

DEFAULT_TARGET = PROJECT_ROOT / "test" / "rfdiffusion" / "inputs_for_test" / "rsv_site3.pdb"
DEFAULT_FRAMEWORK = PROJECT_ROOT / "test" / "rfdiffusion" / "inputs_for_test" / "hu-4D5-8_Fv.pdb"

DESIGN_LOOPS = "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13"
MPNN_LOOPS = "H1,H2,H3,L1,L2,L3"
HOTSPOTS = "T305,T456"

# Checkpoints supplied read-only by the user.  The first three are required by
# the acceptance chain; the fourth is exercised by the repository RF2 JSON test.
CHECKPOINT_FILES = {
    "rfdiffusion": "RFdiffusion_Ab.pt",
    "proteinmpnn": "ProteinMPNN_v48_noise_0.2.pt",
    "rf2": "RF2_ab.pt",
}
OPTIONAL_CHECKPOINT_FILES = {
    "rf2_legacy": "RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt",
}
ALL_EXTERNAL_WEIGHTS = {**CHECKPOINT_FILES, **OPTIONAL_CHECKPOINT_FILES}

CPU_FALLBACK_MARKERS = (
    "running proteinmpnn on cpu",
    "no gpu found, running",
    "falling back to cpu",
    "using cpu device",
    "device='cpu'",
    'device="cpu"',
)

# Cautious-mode skips exit 0 without producing a fresh artifact; they must not
# be accepted as a successful stage.
SKIP_MARKERS = (
    "as output already exists",
    "skipping this design because",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Preflight, environment and weights
# ---------------------------------------------------------------------------
def npu_preflight(device_index: int) -> dict[str, Any]:
    """Hard gate: the NPU must be importable, available and compute correctly."""
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            f"NPU preflight failed: cannot import torch/torch_npu: "
            f"{type(exc).__name__}: {exc}"
        )
    if not torch.npu.is_available():
        raise SystemExit("NPU preflight failed: torch.npu.is_available() is False")
    count = int(torch.npu.device_count())
    if device_index >= count:
        raise SystemExit(
            f"NPU preflight failed: requested npu:{device_index} but only "
            f"{count} device(s) are visible"
        )
    probe = torch.ones(8, device=f"npu:{device_index}") + 1
    if probe.cpu().tolist() != [2.0] * 8:
        raise SystemExit("NPU preflight failed: NPU tensor smoke test returned wrong values")
    return {
        "torch": torch.__version__,
        "torch_npu": str(getattr(torch_npu, "__version__", "unknown")),
        "device_index": int(device_index),
        "device_name": str(torch.npu.get_device_name(device_index)),
        "device_count": count,
        "current_device": int(torch.npu.current_device()),
        "npu_tensor_smoke": "passed",
    }


def resolve_weights_dir(explicit: Path | None) -> Path:
    """Locate the directory holding the external checkpoints.

    Resolution order: ``--weights-dir`` -> ``RFANTIBODY_WEIGHTS`` ->
    ``RFANTIBODY_WEIGHTS_DIR`` -> ``<project>/weights``.  The first candidate
    that contains every required checkpoint wins; otherwise the runner fails.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    for env_name in ("RFANTIBODY_WEIGHTS", "RFANTIBODY_WEIGHTS_DIR"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))
    candidates.append(PROJECT_ROOT / "weights")

    tried: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if all((candidate / name).is_file() for name in CHECKPOINT_FILES.values()):
            return candidate
        tried.append(str(candidate))
    raise SystemExit(
        "Cannot find the required checkpoints "
        f"({', '.join(CHECKPOINT_FILES.values())}). Looked in: {tried}. "
        "Pass --weights-dir, set RFANTIBODY_WEIGHTS, or place them under "
        f"{PROJECT_ROOT / 'weights'}."
    )


def resolve_weights(weights_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    weights_dir = Path(weights_dir).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for model, filename in CHECKPOINT_FILES.items():
        path = weights_dir / filename
        if not path.is_file():
            raise SystemExit(f"Missing required checkpoint for {model}: {path}")
        try:
            path.relative_to(weights_dir)
        except ValueError:
            raise SystemExit(
                f"Resolved {model} checkpoint escapes the configured directory: {path}"
            )
        records.append(
            {
                "model": model,
                "required": True,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for model, filename in OPTIONAL_CHECKPOINT_FILES.items():
        path = weights_dir / filename
        record: dict[str, Any] = {
            "model": model,
            "required": False,
            "path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            record.update({"size": path.stat().st_size, "sha256": sha256_file(path)})
        records.append(record)
    return weights_dir, records


def resolve_inputs(target: Path | None, framework: Path | None) -> dict[str, Path]:
    target_path = Path(target).expanduser().resolve() if target else DEFAULT_TARGET.resolve()
    framework_path = (
        Path(framework).expanduser().resolve() if framework else DEFAULT_FRAMEWORK.resolve()
    )
    for path in (target_path, framework_path):
        if not path.is_file():
            raise SystemExit(f"Required input PDB is missing: {path}")
    return {"target": target_path, "framework": framework_path}


def stage_environment(guard_output: Path, extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    venv_bin = str(Path(sys.prefix) / "bin")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(sys.prefix)
    pythonpath_parts = [
        str(GUARD_DIR),
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT / "include" / "SE3Transformer"),
        str(PROJECT_ROOT),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["AE_DEVICE_GUARD_OUT"] = str(guard_output)
    # On an Ascend host there is no CUDA device; make an accidental CUDA path
    # fail instead of silently running on the wrong accelerator.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Result inspection helpers
# ---------------------------------------------------------------------------
def _aggregate_guard(guard_base: Path) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "devices_seen": [],
        "npu_used": False,
        "cuda_used": False,
        "cpu_used": False,
        "modules_moved": 0,
        "move_devices": {},
        "forward_calls": 0,
        "forward_devices": {},
        "torch_npu_imported": False,
        "processes": [],
        "errors": [],
        "files": [],
    }
    for path in sorted(guard_base.parent.glob(f"{guard_base.name}.*.json")):
        aggregate["files"].append(str(path))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            aggregate["errors"].append(f"unparseable guard file {path}: {exc}")
            continue
        aggregate["processes"].append(data.get("pid"))
        aggregate["npu_used"] = aggregate["npu_used"] or bool(data.get("npu_used"))
        aggregate["cuda_used"] = aggregate["cuda_used"] or bool(data.get("cuda_used"))
        aggregate["cpu_used"] = aggregate["cpu_used"] or bool(data.get("cpu_used"))
        aggregate["modules_moved"] += int(data.get("modules_moved", 0))
        aggregate["forward_calls"] += int(data.get("forward_calls", 0))
        aggregate["torch_npu_imported"] = aggregate["torch_npu_imported"] or bool(
            data.get("torch_npu_imported")
        )
        for key in ("move_devices", "forward_devices"):
            for device_type, count in (data.get(key) or {}).items():
                bucket = aggregate[key]
                bucket[device_type] = bucket.get(device_type, 0) + int(count)
        for device_type in data.get("devices_seen", []):
            if device_type not in aggregate["devices_seen"]:
                aggregate["devices_seen"].append(device_type)
        for error in data.get("errors", []):
            aggregate["errors"].append(error)
    return aggregate


def _looks_like_pdb(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _ in range(5000):
                line = handle.readline()
                if not line:
                    break
                if line.startswith(("ATOM", "HETATM")):
                    return True
    except OSError:
        return False
    return False


def _score_values(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("SCORE "):
                    continue
                name, _, raw = line[len("SCORE "):].partition(":")
                try:
                    scores[name.strip()] = float(raw.strip())
                except ValueError:
                    continue
    except OSError:
        return {}
    return scores


def _file_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        path = Path(path)
        records.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def run_stage(
    name: str,
    cmd: list[str],
    stage_dir: Path,
    extra_env: dict[str, str],
    artifact_globs: list[str],
    declared_inputs: list[Path],
    timeout: int = 0,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{name}.log"
    guard_base = stage_dir / f"{name}.device_guard"
    env = stage_environment(guard_base, extra_env)

    started_wall = time.time()
    started = time.monotonic()
    timed_out = False
    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {shlex.join(cmd)}\n")
        log_handle.write(f"# cwd={PROJECT_ROOT}\n")
        log_handle.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout or None,
            )
            returncode: int | None = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True
    seconds = time.monotonic() - started

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    lower = log_text.lower()
    cpu_markers = sorted({marker for marker in CPU_FALLBACK_MARKERS if marker in lower})
    skip_markers = sorted({marker for marker in SKIP_MARKERS if marker in lower})
    guard = _aggregate_guard(guard_base)

    artifacts: list[Path] = []
    for pattern in artifact_globs:
        for match in sorted(glob.glob(str(stage_dir / pattern))):
            candidate = Path(match)
            if candidate.is_file() and candidate.stat().st_size > 0:
                artifacts.append(candidate)
    pdb_artifacts = [path for path in artifacts if path.suffix == ".pdb"]
    pdbs_valid = bool(pdb_artifacts) and all(_looks_like_pdb(path) for path in pdb_artifacts)
    fresh_artifacts = [
        path for path in artifacts if path.stat().st_mtime >= started_wall - 2
    ]

    npu_forward_outputs = int(guard["forward_devices"].get("npu", 0))
    checks = {
        "completed_without_timeout": not timed_out,
        "exit_code_zero": returncode == 0,
        "artifacts_nonempty_pdb": pdbs_valid,
        "artifacts_fresh": bool(fresh_artifacts),
        "no_cautious_skip_markers": not skip_markers,
        "npu_device_used": bool(guard.get("npu_used")),
        "npu_forward_outputs_observed": npu_forward_outputs > 0,
        "cuda_not_used": not bool(guard.get("cuda_used")),
        "no_cpu_fallback_markers": not cpu_markers,
    }

    return {
        "name": name,
        "cmd": cmd,
        "cmd_pretty": shlex.join(cmd),
        "cwd": str(PROJECT_ROOT),
        "exit_code": returncode,
        "timed_out": timed_out,
        "seconds": round(seconds, 3),
        "log": str(log_path),
        "inputs": _file_records(declared_inputs),
        "inputs_paths": [str(Path(path)) for path in declared_inputs],
        "artifacts": _file_records(artifacts),
        "fresh_artifacts": [str(path) for path in fresh_artifacts],
        "device_evidence": guard,
        "npu_forward_outputs": npu_forward_outputs,
        "cpu_fallback_markers": cpu_markers,
        "skip_markers": skip_markers,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def _pre_existing_files(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(path for path in output_dir.rglob("*") if path.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: a fresh outputs/acceptance_npu_<UTC> directory.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory holding the external checkpoints (default: $RFANTIBODY_WEIGHTS, "
        "$RFANTIBODY_WEIGHTS_DIR or <project>/weights)",
    )
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--framework", type=Path, default=None)
    parser.add_argument("--hotspots", default=HOTSPOTS)
    parser.add_argument("--design-loops", default=DESIGN_LOOPS)
    parser.add_argument("--mpnn-loops", default=MPNN_LOOPS)
    parser.add_argument("--num-designs", type=int, default=1)
    parser.add_argument("--diffuser-t", type=int, default=50)
    parser.add_argument("--final-step", type=int, default=1)
    parser.add_argument("--seqs-per-struct", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--augment-eps", type=float, default=0.0)
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                        help="RF2 seed (default: 0)")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--stage-timeout", type=int, default=0)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face/Transformers downloads (default: offline mode).",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Archive a non-empty --output-dir instead of refusing to reuse it.",
    )
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only verify NPU availability, weights, inputs and entry points; run no model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "outputs" / f"acceptance_npu_{stamp}"
    else:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (PROJECT_ROOT / output_dir).resolve()
    result_path = args.result_json or (output_dir / "e2e_result.json")
    if not Path(result_path).is_absolute():
        result_path = (PROJECT_ROOT / result_path).resolve()

    pre_existing = _pre_existing_files(output_dir)
    if pre_existing and not args.overwrite_existing:
        raise SystemExit(
            f"Refusing to reuse a non-empty output directory ({len(pre_existing)} "
            f"file(s)): {output_dir}. RFdiffusion cautious mode would skip designs "
            "and still exit 0. Choose a new --output-dir or pass --overwrite-existing."
        )
    if pre_existing:
        stamp = utc_now().replace(":", "").replace("-", "")
        shutil.move(str(output_dir), str(output_dir.parent / f"{output_dir.name}.archive-{stamp}"))

    hardware = npu_preflight(args.device_index)
    weights_dir, weight_records = resolve_weights(resolve_weights_dir(args.weights_dir))
    inputs = resolve_inputs(args.target, args.framework)

    offline_env: dict[str, str] = {}
    if not args.allow_download:
        offline_env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": str(output_dir / ".hf_cache"),
            }
        )

    base_env = {
        "RFANTIBODY_ROOT": str(PROJECT_ROOT),
        "RFANTIBODY_WEIGHTS": str(weights_dir),
        "RFANTIBODY_SCRIPTS": str(PROJECT_ROOT / "scripts"),
        "PYTHONHASHSEED": "0",
        "TORCH_DETERMINISTIC": "1",
        **offline_env,
    }
    venv_bin = Path(sys.prefix) / "bin"
    entry_points = {name: venv_bin / name for name in ("rfdiffusion", "proteinmpnn", "rf2")}
    for name, binary in entry_points.items():
        if not binary.is_file():
            raise SystemExit(f"Entry point not installed in the selected environment: {binary}")

    common_result = {
        "kind": "rfantibody_e2e_acceptance",
        "created_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "sys_prefix": sys.prefix,
        "weights_dir": str(weights_dir),
        "weights": weight_records,
        "hardware": hardware,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "entry_points": {name: str(path) for name, path in entry_points.items()},
        "offline": not args.allow_download,
    }

    if args.preflight_only:
        result = {
            **common_result,
            "kind": "rfantibody_e2e_preflight",
            "passed": True,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"preflight OK: npu={hardware['device_name']} weights={weights_dir} "
            f"result={result_path}"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = output_dir / "1_rfdiffusion"
    stage2_dir = output_dir / "2_proteinmpnn"
    stage3_dir = output_dir / "3_rf2"
    for path in (stage1_dir, stage2_dir, stage3_dir):
        path.mkdir(parents=True, exist_ok=True)

    stages: list[dict[str, Any]] = []

    # Stage 1: RFdiffusion backbone design.
    stage1_cmd = [
        str(entry_points["rfdiffusion"]),
        "--target", str(inputs["target"]),
        "--framework", str(inputs["framework"]),
        "--output", str(stage1_dir / "ab_des"),
        "--num-designs", str(args.num_designs),
        "--design-loops", args.design_loops,
        "--hotspots", args.hotspots,
        "--diffuser-t", str(args.diffuser_t),
        "--final-step", str(args.final_step),
        "--deterministic",
        "--no-trajectory",
        "--weights", str(weights_dir / CHECKPOINT_FILES["rfdiffusion"]),
    ]
    stage1 = run_stage(
        "rfdiffusion",
        stage1_cmd,
        stage1_dir,
        base_env,
        artifact_globs=["ab_des_*.pdb"],
        declared_inputs=[
            inputs["target"],
            inputs["framework"],
            weights_dir / CHECKPOINT_FILES["rfdiffusion"],
        ],
        timeout=args.stage_timeout,
    )
    stages.append(stage1)
    stage1_designs = sorted(Path(path) for path in stage1["fresh_artifacts"])

    # Stage 2: ProteinMPNN sequence design on the stage-1 backbones.
    if stage1["passed"] and stage1_designs:
        stage2_cmd = [
            str(entry_points["proteinmpnn"]),
            "--input-dir", str(stage1_dir),
            "--output-dir", str(stage2_dir),
            "--loops", args.mpnn_loops,
            "--seqs-per-struct", str(args.seqs_per_struct),
            "--temperature", str(args.temperature),
            "--augment-eps", str(args.augment_eps),
            "--deterministic",
            "--debug",
            "--weights", str(weights_dir / CHECKPOINT_FILES["proteinmpnn"]),
        ]
        stage2 = run_stage(
            "proteinmpnn",
            stage2_cmd,
            stage2_dir,
            base_env,
            artifact_globs=["*.pdb"],
            declared_inputs=stage1_designs
            + [weights_dir / CHECKPOINT_FILES["proteinmpnn"]],
            timeout=args.stage_timeout,
        )
        stages.append(stage2)
    else:
        stage2 = None

    # Stage 3: RF2 structure prediction on the stage-2 sequences.
    stage3 = None
    if isinstance(stage2, dict) and stage2["passed"]:
        stage2_designs = sorted(Path(path) for path in stage2["fresh_artifacts"])
        stage3_cmd = [
            str(entry_points["rf2"]),
            "--input-dir", str(stage2_dir),
            "--output-dir", str(stage3_dir),
            "--num-recycles", str(args.num_recycles),
            "--no-cautious",
            "--seed", str(args.seed),
            "--hotspot-show-prop", "0",
            "--weights", str(weights_dir / CHECKPOINT_FILES["rf2"]),
        ]
        stage3 = run_stage(
            "rf2",
            stage3_cmd,
            stage3_dir,
            base_env,
            artifact_globs=["*_best.pdb"],
            declared_inputs=stage2_designs + [weights_dir / CHECKPOINT_FILES["rf2"]],
            timeout=args.stage_timeout,
        )
        stages.append(stage3)

    rf2_scores: dict[str, float] = {}
    if stage3 is not None:
        for artifact in stage3["artifacts"]:
            if artifact["path"].endswith(".pdb"):
                rf2_scores = _score_values(Path(artifact["path"]))
                if rf2_scores:
                    break
    pred_lddt = rf2_scores.get("pred_lddt")
    rf2_scores_valid = bool(rf2_scores) and pred_lddt is not None and (
        0.0 <= pred_lddt <= 100.0
    )

    chain_checks = {
        "three_stages_executed": len(stages) == 3,
        "all_stages_passed": bool(stages) and all(stage["passed"] for stage in stages),
        "stage_outputs_are_next_stage_inputs": (
            len(stages) == 3
            and bool(stages[0]["fresh_artifacts"])
            and set(stages[1]["inputs_paths"]) >= set(stages[0]["fresh_artifacts"])
            and set(stages[2]["inputs_paths"]) >= set(stages[1]["fresh_artifacts"])
        ),
        "all_stages_used_npu": bool(stages)
        and all(stage["checks"]["npu_device_used"] for stage in stages),
        "all_stages_observed_npu_forward_outputs": bool(stages)
        and all(stage["checks"]["npu_forward_outputs_observed"] for stage in stages),
        "no_cuda_execution": bool(stages)
        and all(not stage["device_evidence"].get("cuda_used") for stage in stages),
        "rf2_score_lines_present": bool(rf2_scores),
        "rf2_pred_lddt_finite": rf2_scores_valid,
        "normal_recycles_requested": args.num_recycles == 10,
    }

    result = {
        **common_result,
        "output_dir": str(output_dir),
        "workload": {
            "target": str(inputs["target"]),
            "framework": str(inputs["framework"]),
            "hotspots": args.hotspots,
            "design_loops": args.design_loops,
            "mpnn_loops": args.mpnn_loops,
            "num_designs": args.num_designs,
            "diffuser_t": args.diffuser_t,
            "final_step": args.final_step,
            "seqs_per_struct": args.seqs_per_struct,
            "temperature": args.temperature,
            "augment_eps": args.augment_eps,
            "num_recycles": args.num_recycles,
            "forward_cycles": args.num_recycles + 1,
            "seed": args.seed,
            "deterministic": True,
            "allow_download": args.allow_download,
        },
        "stages": stages,
        "rf2_scores": rf2_scores,
        "total_stage_seconds": round(sum(stage["seconds"] for stage in stages), 3),
        "chain_checks": chain_checks,
        "passed": all(chain_checks.values()),
        "failed_checks": [name for name, ok in chain_checks.items() if not ok],
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print(f"RFantibody NPU end-to-end acceptance: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"result: {result_path}")
    for stage in stages:
        failed = [name for name, ok in stage["checks"].items() if not ok]
        print(
            f"  {stage['name']:<14} exit={stage['exit_code']} "
            f"{stage['seconds']:>8.1f}s npu={stage['device_evidence'].get('npu_used')} "
            f"npu_forwards={stage['npu_forward_outputs']} "
            f"failed_checks={failed or 'none'} log={stage['log']}"
        )
    for name in result["failed_checks"]:
        print(f"  chain check failed: {name}")
    print("=" * 78)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
