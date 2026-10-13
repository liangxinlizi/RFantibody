#!/usr/bin/env python3
"""End-to-end RFantibody workflow: RFdiffusion -> ProteinMPNN -> RF2 on NPU.

This is the main acceptance target for the CUDA-to-Ascend migration.  It reuses
the source repository's existing test scripts and inputs
(``test/rfdiffusion/scripts/antibody_pdbdesign.sh``,
``test/proteinmpnn/scripts/ab_seq_design.sh``,
``test/rf2/scripts/ab_prediction.sh``) but chains the three stages directly
through PDB directories instead of Quiver files, as required by the task:

    1. RFdiffusion designs one antibody backbone against RSV site 3 using the
       hu-4D5-8 Fv framework, hotspots T305,T456 and the documented CDR lengths.
    2. ProteinMPNN reads that output directory directly (``--input-dir``).
    3. RF2 reads the ProteinMPNN output directory directly (``--input-dir``) and
       runs the normal 10-recycle / 11-forward-cycle path.

The script never falls back to CPU silently: every stage is wrapped by a device
guard (``e2e_support/sitecustomize.py``) that records whether models were moved
onto an NPU, and the whole run aborts unless ``torch.npu.is_available()``.

It is intentionally not self-verifying against H100/A4000 references: it records
commands, exit codes, wall-clock timings, device evidence, weight hashes and
artifacts so that the deterministic evaluator (``ae-temp/validate.py``) can do
the acceptance checks.

Usage (from ``workspace/project``)::

    .venv/bin/python e2e_run.py --output-dir ../../experiments/npu_chain
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
TASK_ROOT = PROJECT_ROOT.parents[1]
MIGRATION_YAML = TASK_ROOT / "migration.yaml"
DEFAULT_OUTPUT_DIR = TASK_ROOT / "experiments" / "npu_chain"
GUARD_DIR = PROJECT_ROOT / "e2e_support"

DESIGN_LOOPS = "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13"
HOTSPOTS = "T305,T456"

CPU_FALLBACK_MARKERS = (
    "running proteinmpnn on cpu",
    "no gpu found, running",
    "device='cpu'",
    'device="cpu"',
    "using cpu",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_migration_config() -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"PyYAML is required to read {MIGRATION_YAML}: {exc}")
    if not MIGRATION_YAML.exists():
        raise SystemExit(f"migration.yaml not found at {MIGRATION_YAML}")
    with open(MIGRATION_YAML, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"migration.yaml is not a mapping: {MIGRATION_YAML}")
    return data


def npu_preflight() -> dict[str, Any]:
    """Hard gate: NPU must be importable and available before any stage runs."""
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"NPU preflight failed: cannot import torch/torch_npu: {type(exc).__name__}: {exc}"
        )
    if not torch.npu.is_available():
        raise SystemExit("NPU preflight failed: torch.npu.is_available() is False")
    device_index = torch.npu.current_device()
    probe = torch.ones(8, device=f"npu:{device_index}") + 1
    if probe.cpu().tolist() != [2.0] * 8:
        raise SystemExit("NPU preflight failed: NPU tensor smoke test returned wrong values")
    return {
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "device_index": int(device_index),
        "device_name": torch.npu.get_device_name(device_index),
        "device_count": int(torch.npu.device_count()),
        "npu_tensor_smoke": "passed",
    }


def resolve_weights(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    aux = config.get("auxiliary_models") or {}
    weights_dir = aux.get("local_path")
    if not weights_dir:
        raise SystemExit("migration.yaml auxiliary_models.local_path is empty")
    weights_dir = Path(weights_dir).resolve()

    expected = {
        "rfdiffusion": weights_dir / "RFdiffusion_Ab.pt",
        "proteinmpnn": weights_dir / "ProteinMPNN_v48_noise_0.2.pt",
        "rf2": weights_dir / "RF2_ab.pt",
    }
    records = []
    for name, path in expected.items():
        if not path.is_file():
            raise SystemExit(f"Missing required checkpoint for {name}: {path}")
        try:
            path.relative_to(weights_dir)
        except ValueError:
            raise SystemExit(
                f"Resolved {name} checkpoint escapes configured directory: {path}"
            )
        records.append(
            {
                "model": name,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return weights_dir, records


def resolve_inputs(config: dict[str, Any]) -> dict[str, Path]:
    """Prefer the copied repository test inputs; fall back to migration.yaml."""
    local_dir = PROJECT_ROOT / "test" / "rfdiffusion" / "inputs_for_test"
    target = local_dir / "rsv_site3.pdb"
    framework = local_dir / "hu-4D5-8_Fv.pdb"
    if not target.is_file():
        sample = (config.get("input") or {}).get("sample_path")
        if not sample:
            raise SystemExit("Cannot find RSV site 3 target PDB in test inputs or migration.yaml")
        target = Path(sample)
    if not framework.is_file():
        raise SystemExit(f"Cannot find hu-4D5-8 Fv framework PDB: {framework}")
    for path in (target, framework):
        if not path.is_file():
            raise SystemExit(f"Configured input PDB missing: {path}")
    return {"target": target.resolve(), "framework": framework.resolve()}


def stage_environment(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    # sys.prefix points at the active venv even though .venv/bin/python is a
    # symlink to the system interpreter; resolving sys.executable would leave the
    # venv and hide the installed console scripts.
    venv_bin = str(Path(sys.prefix) / "bin")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(PROJECT_ROOT / ".venv")
    env["PYTHONPATH"] = (
        str(GUARD_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        if env.get("PYTHONPATH")
        else str(GUARD_DIR)
    )
    env.update(extra)
    return env


def run_stage(
    name: str,
    cmd: list[str],
    stage_dir: Path,
    extra_env: dict[str, str],
    artifact_globs: list[str],
    require_recycles: int | None = None,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / f"{name}.log"
    guard_base = stage_dir / f"{name}.device_guard"
    env = stage_environment({**extra_env, "AE_DEVICE_GUARD_OUT": str(guard_base)})

    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        log_handle.write(f"# cwd={PROJECT_ROOT}\n")
        log_handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = time.monotonic() - started

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    lower = log_text.lower()
    cpu_markers = sorted({marker for marker in CPU_FALLBACK_MARKERS if marker in lower})

    guard = _aggregate_guard(guard_base)

    artifacts: list[str] = []
    for pattern in artifact_globs:
        for path in sorted(glob.glob(str(stage_dir / pattern))):
            p = Path(path)
            if p.is_file() and p.stat().st_size > 0:
                artifacts.append(str(p))

    pdb_parseable = all(_looks_like_pdb(Path(a)) for a in artifacts) if artifacts else False

    recycles_confirmed: bool | None = None
    if require_recycles is not None:
        recycles_confirmed = bool(
            re.search(rf"num_recycles[\"'\s:=]{{0,6}}{require_recycles}\b", log_text)
        )

    checks = {
        "exit_code_zero": proc.returncode == 0,
        "artifacts_nonempty_pdb": bool(artifacts) and pdb_parseable,
        "npu_device_used": bool(guard.get("npu_used")),
        "cuda_not_used": not bool(guard.get("cuda_used")),
        "no_cpu_fallback_markers": not cpu_markers,
    }
    if require_recycles is not None:
        checks["rf2_recycles_confirmed"] = bool(recycles_confirmed)

    return {
        "name": name,
        "cmd": cmd,
        "cmd_pretty": " ".join(shlex.quote(c) for c in cmd),
        "cwd": str(PROJECT_ROOT),
        "exit_code": proc.returncode,
        "seconds": round(seconds, 3),
        "log": str(log_path),
        "artifacts": artifacts,
        "device_guard_files": [str(p) for p in sorted(guard_base.parent.glob(f"{guard_base.name}.*.json"))],
        "device_evidence": guard,
        "cpu_fallback_markers": cpu_markers,
        "rf2_recycles": require_recycles,
        "rf2_recycles_confirmed": recycles_confirmed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _aggregate_guard(guard_base: Path) -> dict[str, Any]:
    """Union the per-process device-guard files written for one stage."""
    aggregate: dict[str, Any] = {
        "devices_seen": [],
        "npu_used": False,
        "cuda_used": False,
        "cpu_used": False,
        "modules_moved": 0,
        "processes": [],
        "torch_npu_imported": False,
        "errors": [],
    }
    for path in sorted(guard_base.parent.glob(f"{guard_base.name}.*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            aggregate["errors"].append(f"unparseable guard file: {path}")
            continue
        aggregate["processes"].append(data.get("pid"))
        aggregate["npu_used"] = aggregate["npu_used"] or bool(data.get("npu_used"))
        aggregate["cuda_used"] = aggregate["cuda_used"] or bool(data.get("cuda_used"))
        aggregate["cpu_used"] = aggregate["cpu_used"] or bool(data.get("cpu_used"))
        aggregate["modules_moved"] += int(data.get("modules_moved", 0))
        aggregate["torch_npu_imported"] = aggregate["torch_npu_imported"] or bool(
            data.get("torch_npu_imported")
        )
        for device in data.get("devices_seen", []):
            if device not in aggregate["devices_seen"]:
                aggregate["devices_seen"].append(device)
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


def _run(args: argparse.Namespace, output_dir: Path) -> int:
    config = load_migration_config()
    allow_download = bool(
        (config.get("model") or {}).get("allow_download")
        or (config.get("auxiliary_models") or {}).get("allow_download")
    )

    hardware = npu_preflight()
    weights_dir, weights = resolve_weights(config)
    inputs = resolve_inputs(config)

    offline_env: dict[str, str] = {}
    if not allow_download:
        offline_env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )

    rfdiff_dir = output_dir / "1_rfdiffusion"
    mpnn_dir = output_dir / "2_proteinmpnn"
    rf2_dir = output_dir / "3_rf2"
    for path in (rfdiff_dir, mpnn_dir, rf2_dir):
        path.mkdir(parents=True, exist_ok=True)

    venv_bin = Path(sys.prefix) / "bin"
    rfdiffusion_bin = str(venv_bin / "rfdiffusion")
    proteinmpnn_bin = str(venv_bin / "proteinmpnn")
    rf2_bin = str(venv_bin / "rf2")
    for binary in (rfdiffusion_bin, proteinmpnn_bin, rf2_bin):
        if not Path(binary).is_file():
            raise SystemExit(f"Entry point not installed in the selected environment: {binary}")

    base_env = {
        "RFANTIBODY_ROOT": str(PROJECT_ROOT),
        "RFANTIBODY_WEIGHTS": str(weights_dir),
        "PYTHONHASHSEED": "0",
        **offline_env,
    }

    stage_specs = [
        (
            "1_rfdiffusion",
            rfdiff_dir,
            [
                rfdiffusion_bin,
                "--target",
                str(inputs["target"]),
                "--framework",
                str(inputs["framework"]),
                "--output",
                str(rfdiff_dir / "ab_des"),
                "--num-designs",
                str(args.num_designs),
                "--design-loops",
                DESIGN_LOOPS,
                "--hotspots",
                HOTSPOTS,
                "--diffuser-t",
                str(args.diffuser_t),
                "--final-step",
                str(args.final_step),
                "--deterministic",
            ],
            ["*.pdb"],
            None,
        ),
        (
            "2_proteinmpnn",
            mpnn_dir,
            [
                proteinmpnn_bin,
                "--input-dir",
                str(rfdiff_dir),
                "--output-dir",
                str(mpnn_dir),
                "--loops",
                "H1,H2,H3,L1,L2,L3",
                "--seqs-per-struct",
                str(args.seqs_per_struct),
                "--temperature",
                str(args.temperature),
                "--augment-eps",
                "0",
                "--deterministic",
                "--debug",
            ],
            ["*.pdb"],
            None,
        ),
        (
            "3_rf2",
            rf2_dir,
            [
                rf2_bin,
                "--input-dir",
                str(mpnn_dir),
                "--output-dir",
                str(rf2_dir),
                "--num-recycles",
                str(args.num_recycles),
                "--no-cautious",
                "--seed",
                "42",
                "--extra",
                "inference.hotspot_show_proportion=0",
            ],
            ["*_best.pdb"],
            args.num_recycles,
        ),
    ]

    report: dict[str, Any] = {
        "task": "RFantibody RFdiffusion -> ProteinMPNN -> RF2 NPU chain",
        "run_start_utc": utc_now(),
        "run_end_utc": None,
        "total_seconds": None,
        "output_dir": str(output_dir),
        "overall_status": "running",
        "hardware": hardware,
        "weights_dir": str(weights_dir),
        "weights": weights,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "design_loops": DESIGN_LOOPS,
        "hotspots": HOTSPOTS,
        "num_designs": args.num_designs,
        "diffuser_t": args.diffuser_t,
        "final_step": args.final_step,
        "num_recycles": args.num_recycles,
        "allow_download": allow_download,
        "offline_env": offline_env,
        "stages": [],
        "error": None,
    }

    overall_start = time.monotonic()
    try:
        for name, stage_dir, cmd, globs, recycles in stage_specs:
            result = run_stage(name, cmd, stage_dir, base_env, globs, recycles)
            report["stages"].append(result)
            if not result["passed"]:
                report["error"] = (
                    f"stage {name} failed: exit_code={result['exit_code']} "
                    f"checks={result['checks']} cpu_markers={result['cpu_fallback_markers']}"
                )
                report["overall_status"] = "failed"
                report["total_seconds"] = round(time.monotonic() - overall_start, 3)
                report["run_end_utc"] = utc_now()
                _write_report(output_dir, report)
                return 1
        report["overall_status"] = "passed"
    finally:
        if report["total_seconds"] is None:
            report["total_seconds"] = round(time.monotonic() - overall_start, 3)
        if report["run_end_utc"] is None:
            report["run_end_utc"] = utc_now()

    _write_report(output_dir, report)
    return 0 if report["overall_status"] == "passed" else 1


def _fail(output_dir: Path, message: str) -> int:
    report = {
        "task": "RFantibody RFdiffusion -> ProteinMPNN -> RF2 NPU chain",
        "run_start_utc": utc_now(),
        "run_end_utc": utc_now(),
        "total_seconds": 0.0,
        "output_dir": str(output_dir),
        "overall_status": "failed",
        "stages": [],
        "error": message,
    }
    _write_report(output_dir, report)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that receives stage logs, device evidence and artifacts.",
    )
    parser.add_argument("--num-designs", type=int, default=1)
    parser.add_argument("--diffuser-t", type=int, default=50)
    parser.add_argument("--final-step", type=int, default=1)
    parser.add_argument("--seqs-per-struct", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _run(args, output_dir)
    except SystemExit as exc:
        return _fail(output_dir, f"setup failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        return _fail(
            output_dir,
            f"setup raised {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    report_path = output_dir / "e2e_result.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "stages"}, indent=2))
    for stage in report["stages"]:
        print(
            f"[{stage['name']}] exit={stage['exit_code']} seconds={stage['seconds']} "
            f"npu_used={stage['device_evidence'].get('npu_used')} passed={stage['passed']}"
        )
    print(f"e2e result written to {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
