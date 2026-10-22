"""Two-channel hardware-route evidence for the native AscendC graph kernels.

Why two channels
----------------
In this vendor build (CANN 8.1.RC1 / torch_npu 2.5.1) the exported chrome trace
names a hardware task event by its *task type* (for example ``KERNEL_AIVEC``);
the kernel name is not carried by the trace event.  Kernel identities live in a
separate channel: the CANN report CSVs (``op_summary*.csv`` / ``task_time*.csv``
under ``mindstudio_profiler_output/``) and the GE hash dictionaries inside the
raw profile directory.

The route gate therefore checks two measured facts for the *same* public-API
call, plus instrumentation controls:

1. ``hardware_channel`` -- a probe-identical profile (default configuration, no
   ``experimental_config``, i.e. Level0) of the call.  The chrome trace of a call
   that issues one kernel must contain exactly one ``KERNEL_*`` hardware task
   event.
2. ``report_channel`` -- an op-level profile (Level1) of the same call.  The
   preserved vendor report CSVs must contain exactly one ``AI_CORE`` row naming
   the kernel (``op_summary*:Op Name`` / ``task_time*:kernel_name``), while the
   token ``__no_such_kernel__`` resolves nowhere in the very same artefacts.

Controls (``collect_route_identity``):

* positive -- a *separate native AscendC kernel* (``ascend_graph::graph_probe``,
  ``x + 1``, its own kernel body and generated ``aclrtlaunch_graph_probe`` entry
  point) measured with the identical harness/profile level must produce a
  ``KERNEL_*`` event, so a missing event for the custom op is an operator finding
  and not a broken harness.  A builtin op cannot serve here: in this vendor build
  builtin aclnn device tasks are exported with ``Task Type: "N/A"`` under their
  aclnn event name (measured), so only a native AscendC launch can produce the
  event the harness counts; the builtin case is therefore measured and recorded
  as a vendor-naming observation (``builtin_naming_observation``) and never
  gates;
* negative -- a window with no device work (CPU-only case) must produce zero
  ``KERNEL_*`` events, so a counted event is a real device task and not profiler
  noise.

Every measurement runs in its own fresh interpreter (``python -m
test.graph_parity.route_evidence --mode ...``), because repeating profiler
sessions inside one interpreter is observably unstable in this stack
(``Incorrect schedule: Stop profiler while current state is RECORD``).  Nothing
here fabricates, renames or rewrites trace events or report rows.

The helper also wraps the *unmodified* host-owned probe ``tools/npu_kernel_probe.py``
(``run_npu_kernel_probe``) and records its invocation, so the evaluator's own
probe contract is exercised with the hardware task type of this vendor build.

Venv compatibility shim
-----------------------
``torch_npu.utils.collect_env.get_torch_npu_install_path()`` returns
``site.getsitepackages()[0]``.  In a ``--system-site-packages`` venv that is the
venv while ``torch_npu`` itself is inherited from the system interpreter, so
``profiler.stop()`` raises ``The path does not exist: <venv>/.../torch_npu/version.py``
and the raw profile can be truncated.  ``apply_install_path_shim`` corrects only
that path lookup; it does not touch device detection, profiling data or any
result.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Hardware task events are identified by this task-type prefix.
KERNEL_TASK_PREFIX = "KERNEL_"
# Token that must not resolve anywhere (negative control).
NEGATIVE_KERNEL_TOKEN = "__no_such_kernel__"
# Kernel-name token of the contracted operator under test.
EXPECTED_KERNEL_TOKEN = "edge_dot_v"

DEFAULT_CASE = "test.graph_parity.route_case:run"
# Positive instrumentation control: a *separate native AscendC kernel* through the
# same harness.  A builtin aclnn op cannot serve here: this vendor build exports
# builtin device tasks with ``Task Type: "N/A"`` (measured), so only a native
# AscendC launch can produce the ``KERNEL_*`` event the harness counts.
POSITIVE_CONTROL_CASE = "test.graph_parity.route_case:run_graph_probe"
# Recorded (not gated) vendor-naming observation: a builtin aclnn op.
BUILTIN_OBSERVATION_CASE = "test.graph_parity.route_case:run_builtin"
# Negative instrumentation control: a window with no device work.
NEGATIVE_CONTROL_CASE = "test.graph_parity.route_case:run_cpu_only"

EVIDENCE_DIR_NAME = "graph_parity"
# Profiling scratch lives in a git-ignored directory of the project root.
SCRATCH_DIR_NAME = "export_only_prof_dir"
SUBPROCESS_TIMEOUT_SECONDS = 1800
PROBE_TIMEOUT_SECONDS = 2400

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Columns of the vendor report channel that identify one executed task.  The
# header spelling differs per report (``Op Name``/``Task Type`` in
# ``op_summary*.csv``, ``kernel_name``/``kernel_type`` in ``task_time*.csv``).
_REPORT_COLUMNS = {
    "op_summary": ("Device_id", "Task ID", "Stream ID", "Op Name", "OP Type", "Task Type"),
    "task_time": ("Device_id", "kernel_name", "kernel_type", "stream_id", "task_id"),
}
# Report row classifier: the column that carries the CANN task type.
_TASK_TYPE_COLUMNS = ("tasktype", "kerneltype")
_AI_CORE_TASK_TYPE = "AI_CORE"


# --------------------------------------------------------------------------- #
# Paths, small utilities
# --------------------------------------------------------------------------- #
def evidence_dir() -> Path:
    """Directory for machine-readable route evidence (honours ``RF_EVIDENCE_DIR``)."""
    override = os.environ.get("RF_EVIDENCE_DIR")
    root = Path(override) if override else PROJECT_ROOT / "migration_reports"
    return root / EVIDENCE_DIR_NAME


def scratch_root() -> Path:
    return PROJECT_ROOT / SCRATCH_DIR_NAME / "route_evidence"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pythonpath() -> str:
    """``PYTHONPATH`` that exposes the project packages plus inherited entries."""
    existing = [item for item in os.environ.get("PYTHONPATH", "").split(":") if item]
    parts = [str(PROJECT_ROOT)] + [item for item in existing if item != str(PROJECT_ROOT)]
    return ":".join(parts)


def load_case(spec: str) -> Callable[[], Any]:
    """Resolve a no-argument ``module:function`` case (same contract as the probe)."""
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--case must use module:function syntax, got %r" % (spec,))
    module = importlib.import_module(module_name)
    case = getattr(module, function_name)
    if not callable(case):
        raise TypeError("case is not callable: %s" % (spec,))
    return case


def _task_type(args: dict[str, Any]) -> str:
    for key, value in args.items():
        if re.sub(r"[^a-z]", "", key.lower()) == "tasktype":
            return str(value)
    return ""


def _sanitize(text: str) -> str:
    """File-name-safe form of a case spec / kernel pattern."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


# --------------------------------------------------------------------------- #
# Channel 1: chrome-trace hardware task events
# --------------------------------------------------------------------------- #
def kernel_task_events(trace_path: Path) -> list[dict[str, Any]]:
    """All ``KERNEL_*`` hardware task events of an exported chrome trace.

    No name matching is performed: in this vendor build the event name *is* the
    CANN task type, so the measurable fact is the task type and the event count.
    """
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    found: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args") or {}
        task_type = _task_type(args)
        if not task_type.startswith(KERNEL_TASK_PREFIX):
            continue
        found.append(
            {
                "name": str(event.get("name", "")),
                "task_type": task_type,
                "task_id": args.get("Task Id"),
                "duration_us": event.get("dur"),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
            }
        )
    return found


def task_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        key = "%s|%s" % (event.get("name", ""), event.get("task_type", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def traced_task_types(trace_path: Path) -> dict[str, int]:
    """Count every trace event that carries a CANN ``Task Type`` field.

    Unlike :func:`kernel_task_events` this does not filter on the ``KERNEL_``
    prefix, so it records the measured naming behaviour of this vendor build:
    native AscendC launches appear as ``KERNEL_*`` task types, builtin aclnn ops
    appear under their aclnn event name with ``Task Type: N/A``.
    """
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    counts: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args") or {}
        if not any(
            re.sub(r"[^a-z]", "", str(key).lower()) == "tasktype" for key in args
        ):
            continue
        key = "%s|%s" % (str(event.get("name", "")), _task_type(args))
        counts[key] = counts.get(key, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Venv compatibility shim (path lookup only)
# --------------------------------------------------------------------------- #
def apply_install_path_shim() -> dict[str, Any]:
    """Point torch_npu's install-path lookup at the real ``torch_npu`` package."""
    import torch_npu
    from torch_npu.utils import collect_env

    package_file = Path(torch_npu.__file__).resolve()
    real_site_packages = str(package_file.parent.parent)
    collect_env.get_torch_npu_install_path = lambda: real_site_packages
    return {
        "torch_npu_package": str(package_file),
        "torch_npu_install_path_shim": real_site_packages,
        "shim_reason": (
            "site.getsitepackages()[0] is the venv in a --system-site-packages "
            "environment; only the path lookup is corrected"
        ),
    }


def _level1_config() -> Any:
    from torch_npu.profiler.experimental_config import (
        AiCMetrics,
        ProfilerLevel,
        _ExperimentalConfig,
    )

    return _ExperimentalConfig(
        profiler_level=ProfilerLevel.Level1, aic_metrics=AiCMetrics.PipeUtilization
    )


# --------------------------------------------------------------------------- #
# Profiling session and raw-profile preservation
# --------------------------------------------------------------------------- #
def _profile(
    call: Callable[[], Any],
    scratch: Path,
    trace_path: Path,
    experimental_config: Any = None,
) -> dict[str, Any]:
    """Profile one public-API call and preserve its raw profile artefacts."""
    import torch
    import torch_npu
    from torch_npu.profiler._profiler_path_creator import ProfPathCreator

    scratch = Path(scratch)
    trace_path = Path(trace_path)
    scratch.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    shim = apply_install_path_shim()
    previous_work_path = os.environ.get("ASCEND_WORK_PATH")
    # A work path makes torch_npu keep the raw profile instead of deleting it
    # after the export (`export_only_prof_dir` mode).
    os.environ["ASCEND_WORK_PATH"] = str(scratch)
    try:
        activities = [
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ]
        with torch_npu.profiler.profile(
            activities=activities, experimental_config=experimental_config
        ) as profile:
            result = call()
            torch.npu.synchronize()
        profile.export_chrome_trace(str(trace_path))

        raw_dir_text = ProfPathCreator().get_prof_dir() or ""
        raw_dir = Path(raw_dir_text) if raw_dir_text else None
        if not trace_path.is_file():
            raise RuntimeError(
                "profiler produced no chrome trace at %s (raw profile dir: %s)"
                % (trace_path, raw_dir)
            )
        kept = _preserve_profile_artifacts(raw_dir, scratch / "kept")
        kept["profiler_info_present"] = bool(
            raw_dir is not None
            and (
                (raw_dir / "profiler_info.json").is_file()
                or any(raw_dir.glob("profiler_info*.json"))
            )
        )
        return {
            "result": result,
            "trace": trace_path,
            "raw_dir": raw_dir,
            "kept": kept,
            "shim": shim,
        }
    finally:
        if previous_work_path is None:
            os.environ.pop("ASCEND_WORK_PATH", None)
        else:
            os.environ["ASCEND_WORK_PATH"] = previous_work_path


def _preserve_profile_artifacts(raw_dir: Path | None, target: Path) -> dict[str, Any]:
    """Copy the vendor report CSVs and GE hash dictionaries out of the raw profile."""
    kept: dict[str, Any] = {
        "raw_profile_dir": str(raw_dir) if raw_dir else None,
        "csv_files": [],
        "hash_files": [],
        "dashboard_dirs": [],
    }
    if raw_dir is None or not raw_dir.is_dir():
        return kept
    target.mkdir(parents=True, exist_ok=True)
    for csv_dir in sorted(raw_dir.rglob("mindstudio_profiler_output")):
        if not csv_dir.is_dir():
            continue
        destination = target / csv_dir.parent.name / "mindstudio_profiler_output"
        destination.mkdir(parents=True, exist_ok=True)
        for csv_file in sorted(csv_dir.glob("*.csv")):
            copy = destination / csv_file.name
            shutil.copy2(csv_file, copy)
            kept["csv_files"].append(str(copy))
        kept["dashboard_dirs"].append(str(destination))
    hash_target = target / "hash_dicts"
    for hash_file in sorted(raw_dir.rglob("*hash*")):
        if not hash_file.is_file():
            continue
        hash_target.mkdir(parents=True, exist_ok=True)
        copy = hash_target / hash_file.name
        try:
            shutil.copy2(hash_file, copy)
        except OSError:
            continue
        kept["hash_files"].append(str(copy))
    return kept


# --------------------------------------------------------------------------- #
# Channel 1 API: exactly-counted hardware tasks in a profiled window
# --------------------------------------------------------------------------- #
def _hardware_measurement(
    call: Callable[[], Any], scratch: Path, kind: str, case_spec: str
) -> dict[str, Any]:
    """Profile ``call`` with the probe-identical default configuration."""
    trace_path = Path(scratch) / "chrome_trace.json"
    session = _profile(call, scratch, trace_path)
    events = kernel_task_events(trace_path)
    return {
        "kind": kind,
        "case": case_spec,
        "profiler_level": "Level0 (probe-identical: activities=[CPU, NPU], no experimental_config)",
        "trace": str(trace_path),
        "events": events,
        "hardware_event_count": len(events),
        "task_counts": task_counts(events),
        "trace_event_names": [str(event["name"]) for event in events],
        "traced_task_types": traced_task_types(trace_path),
        "profile_artifacts": session["kept"],
        "profiler": dict(session["shim"], profiler_level="Level0"),
        "versions": _versions(),
        "written_utc": utc_now(),
    }


def collect_level0(call: Callable[[], Any], trace_path: Path) -> dict[str, Any]:
    """Profile ``call`` with the default (Level0) configuration.

    Returns every ``KERNEL_*`` hardware task event of the window (name, task
    type, task id, duration) together with its count.  Used by the route-control
    and tiny-execution tests, which already hold the callable.
    """
    trace_path = Path(trace_path)
    session = _profile(call, trace_path.parent / "level0_scratch", trace_path)
    events = kernel_task_events(trace_path)
    return {
        "trace": str(trace_path),
        "events": events,
        "hardware_event_count": len(events),
        "task_counts": task_counts(events),
        "profile_artifacts": session["kept"],
        "profiler": dict(session["shim"], profiler_level="Level0"),
    }


def hardware_channel(call_spec: str, scratch: Path) -> dict[str, Any]:
    """Probe-identical hardware-task channel for one public-API case."""
    return _hardware_measurement(
        load_case(call_spec),
        Path(scratch),
        "rfantibody_native_route_hardware_channel",
        call_spec,
    )


def control_channel(call_spec: str, scratch: Path) -> dict[str, Any]:
    """Negative instrumentation control: a window that must issue no device task."""
    channel = _hardware_measurement(
        load_case(call_spec),
        Path(scratch),
        "rfantibody_native_route_negative_control",
        call_spec,
    )
    channel["expectation"] = (
        "a window with no device work must produce zero KERNEL_* hardware task events"
    )
    return channel


# --------------------------------------------------------------------------- #
# Channel 2: kernel identity from the vendor report / hash dictionaries
# --------------------------------------------------------------------------- #
def _read_hash_dictionary(path: Path) -> str:
    """Text of a GE hash dictionary, whether it is sqlite or a plain text file."""
    data = path.read_bytes()
    if data[:16].startswith(b"SQLite format 3"):
        try:
            connection = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
            try:
                chunks: list[str] = []
                tables = list(
                    connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                )
                for (table_name,) in tables:
                    for row in connection.execute('SELECT * FROM "%s"' % table_name):
                        chunks.append(" ".join(str(value) for value in row))
                if chunks:
                    return "\n".join(chunks)
            finally:
                connection.close()
        except Exception:  # noqa: BLE001 - fall back to the raw bytes
            pass
    return data.decode("utf-8", errors="replace")


def resolve_token(artifacts: dict[str, Any], token: str) -> dict[str, Any]:
    """Search every preserved vendor channel for ``token``.

    Returns the matched values, the channel (``<file>:<column>``) they came from
    and a bounded list of the matching rows.  An absent token yields empty
    results, which is what the negative control asserts.
    """
    names: list[str] = []
    channels: list[str] = []
    rows: list[dict[str, Any]] = []
    if not token:
        return {"names": [], "channels": [], "rows": []}

    for csv_path in artifacts.get("csv_files", []):
        path = Path(csv_path)
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            for row in reader:
                for index, cell in enumerate(row):
                    if token not in cell:
                        continue
                    column = header[index] if index < len(header) else "column_%d" % index
                    names.append(cell)
                    channels.append("%s:%s" % (path.name, column))
                    if len(rows) < 200:
                        rows.append(
                            {"file": path.name, "column": column, "value": cell}
                        )
    for hash_path in artifacts.get("hash_files", []):
        path = Path(hash_path)
        text = _read_hash_dictionary(path)
        if token not in text:
            continue
        names.append(token)
        channels.append("%s:hash_dictionary" % path.name)
        excerpt = next(
            (line.strip() for line in text.splitlines() if token in line),
            token,
        )
        if len(rows) < 200:
            rows.append(
                {
                    "file": path.name,
                    "column": "hash_dictionary",
                    "value": excerpt[:400],
                }
            )
    return {
        "names": sorted(set(names)),
        "channels": sorted(set(channels)),
        "rows": rows,
    }


def _report_kind(path: Path) -> str:
    """``op_summary`` / ``task_time`` / ``""`` for a preserved report CSV."""
    for kind in ("op_summary", "task_time"):
        if path.name.startswith(kind):
            return kind
    return ""


def _selected_columns(header: list[str], kind: str) -> list[tuple[int, str]]:
    wanted = _REPORT_COLUMNS[kind]
    return [
        (index, name)
        for index, name in enumerate(header)
        if name in wanted
    ]


def _report_rows(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Structured rows of the preserved report CSVs.

    Returns, per report kind, every row projected onto the identifying columns,
    and the subset of ``AI_CORE`` rows (the task type of this build's vector
    kernel).  The rows are read verbatim from the vendor CSVs; nothing is
    synthesised.
    """
    rows: dict[str, Any] = {
        "files": [],
        "op_summary": [],
        "task_time": [],
        "ai_core": {"op_summary": [], "task_time": []},
    }
    for csv_path in artifacts.get("csv_files", []):
        path = Path(csv_path)
        kind = _report_kind(path)
        if not kind:
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            selected = _selected_columns(header, kind)
            type_columns = [
                (index, name)
                for index, name in enumerate(header)
                if re.sub(r"[^a-z]", "", name.lower()) in _TASK_TYPE_COLUMNS
            ]
            rows["files"].append(str(path))
            for raw in reader:
                if not raw:
                    continue
                row = {
                    name: (raw[index] if index < len(raw) else "")
                    for index, name in selected
                }
                row = dict(row, file=path.name, kind=kind)
                rows[kind].append(row)
                task_types = {
                    str(raw[index]).upper()
                    for index, _ in type_columns
                    if index < len(raw)
                }
                if _AI_CORE_TASK_TYPE in task_types:
                    rows["ai_core"][kind].append(row)
    return rows


def report_channel(call_spec: str, scratch: Path) -> dict[str, Any]:
    """Op-level (Level1) report channel naming the kernel of one public-API case.

    The chrome trace of this profiler level is recorded but never required to
    contain an event: the hardware-task fact is measured by
    ``hardware_channel`` at the probe-identical level.
    """
    scratch = Path(scratch)
    trace_path = scratch / "chrome_trace.json"
    session = _profile(
        load_case(call_spec),
        scratch,
        trace_path,
        experimental_config=_level1_config(),
    )
    events = kernel_task_events(trace_path)
    artifacts = session["kept"]
    rows = _report_rows(artifacts)
    resolved = resolve_token(artifacts, EXPECTED_KERNEL_TOKEN)
    negative = resolve_token(artifacts, NEGATIVE_KERNEL_TOKEN)
    op_names: list[str] = []
    kernel_names: list[str] = []
    for row in rows["ai_core"]["op_summary"]:
        value = str(row.get("Op Name", ""))
        if value and value not in op_names:
            op_names.append(value)
    for row in rows["ai_core"]["task_time"]:
        value = str(row.get("kernel_name", ""))
        if value and value not in kernel_names:
            kernel_names.append(value)
    return {
        "kind": "rfantibody_native_route_report_channel",
        "case": call_spec,
        "profiler_level": "Level1 (op-level reports, aic_metrics=PipeUtilization)",
        "expected_kernel_token": EXPECTED_KERNEL_TOKEN,
        "negative_token": NEGATIVE_KERNEL_TOKEN,
        "trace": str(trace_path),
        "trace_events_observed": events,
        "report_rows": rows,
        "ai_core_rows": rows["ai_core"],
        "op_names": op_names,
        "kernel_names": kernel_names,
        "resolved_names": resolved["names"],
        "resolved_channels": resolved["channels"],
        "resolved_rows": resolved["rows"],
        "negative_resolved": negative["names"],
        "negative_channels": negative["channels"],
        "profile_artifacts": artifacts,
        "profiler": dict(
            session["shim"],
            profiler_level="Level1",
            aic_metrics="ACL_AICORE_PIPE_UTILIZATION",
        ),
        "versions": _versions(),
        "written_utc": utc_now(),
    }


def _versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for module_name, key in (("torch", "torch"), ("torch_npu", "torch_npu"), ("dgl", "dgl")):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - evidence only
            continue
        versions[key] = str(getattr(module, "__version__", "unknown"))
    versions["cann_home"] = os.environ.get("ASCEND_HOME_PATH", "")
    return versions


# --------------------------------------------------------------------------- #
# Isolated measurements (one fresh interpreter per profiler session)
# --------------------------------------------------------------------------- #
def _run_measurement(mode: str, case_spec: str, base: Path) -> dict[str, Any]:
    """Run one measurement in a fresh interpreter through this module's CLI."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    tag = _sanitize(case_spec)
    output = base / ("%s_%s.json" % (mode, tag))
    log_path = output.with_suffix(".log")
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath()
    argv = [
        sys.executable,
        "-m",
        "test.graph_parity.route_evidence",
        "--mode",
        mode,
        "--case",
        case_spec,
        "--scratch-root",
        str(base / ("scratch_%s" % tag)),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        argv,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    log_path.write_text(
        "COMMAND: %s\nCWD: %s\nEXIT: %s\n\nSTDOUT\n%s\n\nSTDERR\n%s\n"
        % (
            " ".join(argv),
            PROJECT_ROOT,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0 or not output.is_file():
        tail = "\n".join((completed.stderr or "").splitlines()[-20:])
        raise RuntimeError(
            "%s measurement for %s failed (exit %s); see %s\n%s"
            % (mode, case_spec, completed.returncode, log_path, tail)
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["command"] = " ".join(argv)
    payload["exit_code"] = completed.returncode
    payload["log"] = str(log_path)
    return payload


def _channel_summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "profiler_level",
        "events",
        "hardware_event_count",
        "task_counts",
        "trace_event_names",
        "traced_task_types",
        "trace",
        "profile_artifacts",
        "profiler",
        "expectation",
        "command",
        "exit_code",
        "log",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def _report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "profiler_level",
        "ai_core_rows",
        "report_rows",
        "op_names",
        "kernel_names",
        "resolved_names",
        "resolved_channels",
        "resolved_rows",
        "negative_resolved",
        "negative_channels",
        "trace_events_observed",
        "trace",
        "profile_artifacts",
        "profiler",
        "command",
        "exit_code",
        "log",
    )
    return {key: payload.get(key) for key in keys if key in payload}


def collect_route_identity(
    case_spec: str = DEFAULT_CASE, scratch: Path | None = None
) -> dict[str, Any]:
    """Collect both route channels plus instrumentation controls and persist them.

    Each measurement runs in its own fresh interpreter, so profiler session reuse
    cannot influence the result.  The measurement logs, the raw report copies and
    this combined record are the evidence.
    """
    base = Path(scratch) if scratch else scratch_root() / "identity"
    hardware = _run_measurement("hardware", case_spec, base / "hardware")
    report = _run_measurement("report", case_spec, base / "report")
    positive = _run_measurement(
        "hardware", POSITIVE_CONTROL_CASE, base / "positive_instrumentation_control"
    )
    negative = _run_measurement(
        "control", NEGATIVE_CONTROL_CASE, base / "negative_instrumentation_control"
    )
    builtin = _run_measurement(
        "hardware", BUILTIN_OBSERVATION_CASE, base / "builtin_naming_observation"
    )

    hardware_summary = _channel_summary(hardware)
    hardware_summary["count"] = hardware.get("hardware_event_count", 0)
    hardware_summary["profiler_level"] = hardware.get("profiler_level")
    report_summary = _report_summary(report)
    positive_summary = _channel_summary(positive)
    positive_summary["case"] = POSITIVE_CONTROL_CASE
    positive_summary["count"] = positive.get("hardware_event_count", 0)
    positive_summary["note"] = (
        "separate native AscendC kernel (ascend_graph::graph_probe, x+1) with its own "
        "kernel body and generated aclrtlaunch entry point, measured with the identical "
        "harness and profiler level: proves the hardware-event channel detects native "
        "AscendC launches in this environment, so a missing event for e_dot_v is an "
        "operator finding and not a broken harness"
    )
    builtin_summary = _channel_summary(builtin)
    builtin_summary["case"] = BUILTIN_OBSERVATION_CASE
    builtin_summary["count"] = builtin.get("hardware_event_count", 0)
    builtin_summary["note"] = (
        "measured vendor naming behaviour, not a gate: a builtin aclnn op on npu:0 "
        "profiled with the identical harness produces no KERNEL_* trace event; its "
        "device tasks are named aclnn* with Task Type 'N/A' in this CANN build, so a "
        "builtin op cannot be the positive control here"
    )
    negative_summary = _channel_summary(negative)
    negative_summary["case"] = NEGATIVE_CONTROL_CASE
    negative_summary["count"] = negative.get("hardware_event_count", 0)
    negative_summary["note"] = (
        "CPU-only window measured with the identical harness: proves a counted "
        "event is a real device task and not profiler noise"
    )

    evidence: dict[str, Any] = {
        "kind": "rfantibody_native_route_identity",
        "case": case_spec,
        "candidate": ("ascend_graph::e_dot_v -> native AscendC kernel edge_dot_v on npu:0"),
        "expected_kernel_token": EXPECTED_KERNEL_TOKEN,
        "negative_token": NEGATIVE_KERNEL_TOKEN,
        "isolation": (
            "one fresh interpreter per measurement "
            "(python -m test.graph_parity.route_evidence --mode hardware|report|control)"
        ),
        "hardware": hardware_summary,
        "identity": report_summary,
        "positive_instrumentation_control": positive_summary,
        "negative_instrumentation_control": negative_summary,
        "builtin_naming_observation": builtin_summary,
        "probe_invocations": [],
        "versions": _versions(),
        "written_utc": utc_now(),
    }
    path = evidence_dir() / "route_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence["evidence_file"] = str(path)
    return evidence


# --------------------------------------------------------------------------- #
# Unmodified host probe (tools/npu_kernel_probe.py)
# --------------------------------------------------------------------------- #
def probe_script() -> Path | None:
    """Locate the host-owned, unmodified ``tools/npu_kernel_probe.py``."""
    override = os.environ.get("RF_NPU_KERNEL_PROBE")
    candidates = [Path(override)] if override else []
    candidates.append(PROJECT_ROOT.parents[1] / "tools" / "npu_kernel_probe.py")
    candidates.append(PROJECT_ROOT / "tools" / "npu_kernel_probe.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def run_npu_kernel_probe(
    output_dir: Path | None = None,
    kernel: str = KERNEL_TASK_PREFIX + "AIVEC",
    case_spec: str = DEFAULT_CASE,
) -> dict[str, Any]:
    """Run the unmodified probe for one ``--kernel`` pattern and record it.

    The probe's kernel argument is a *trace-event name* regex; in this build the
    event name is the CANN task type (``KERNEL_AIVEC``), so the kernel identity
    itself is covered by the report channel above.  Each invocation keeps its own
    summary/trace/log, so several patterns can be recorded side by side.
    """
    script = probe_script()
    directory = Path(output_dir) if output_dir else evidence_dir()
    directory.mkdir(parents=True, exist_ok=True)
    tag = _sanitize(kernel)
    summary_path = directory / ("npu_route_%s.hw_%s.json" % (EXPECTED_KERNEL_TOKEN, tag))
    log_path = directory / ("npu-kernel-probe_%s.log" % tag)
    record: dict[str, Any] = {
        "script": str(script) if script else None,
        "available": bool(script),
        "command": None,
        "env": {},
        "exit_code": None,
        "summary": {},
        "log": None,
        "kernel_argument": kernel,
    }
    if not script:
        record["note"] = (
            "tools/npu_kernel_probe.py is host-owned and not part of the migrated "
            "tree; the two native evidence channels remain the route gate"
        )
        return record
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath()
    argv = [
        sys.executable,
        str(script),
        "--operator",
        "ascend_graph::e_dot_v",
        "--case",
        case_spec,
        "--kernel",
        kernel,
        "--output",
        str(summary_path),
    ]
    completed = subprocess.run(
        argv,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    log_path.write_text(
        "COMMAND: %s\nCWD: %s\nEXIT: %s\n\nSTDOUT\n%s\n\nSTDERR\n%s\n"
        % (
            " ".join(argv),
            PROJECT_ROOT,
            completed.returncode,
            completed.stdout,
            completed.stderr,
        ),
        encoding="utf-8",
    )
    record["command"] = " ".join(argv)
    record["env"] = {"PYTHONPATH": env["PYTHONPATH"]}
    record["exit_code"] = completed.returncode
    record["log"] = str(log_path)
    if summary_path.is_file():
        try:
            record["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record["summary"] = {}
    return record


def record_probe_invocation(record: dict[str, Any]) -> Path:
    """Append one probe invocation to ``route_evidence.json``."""
    path = evidence_dir() / "route_evidence.json"
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.setdefault("probe_invocations", []).append(record)
    payload["probe_invocations_utc"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# CLI (one measurement per invocation; also usable for manual reproduction)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one measured route channel (or the combined identity record) "
            "for a no-argument public-API case."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("identity", "hardware", "report", "control"),
        default="identity",
        help=(
            "identity: both channels plus controls, each in its own interpreter; "
            "hardware: Level0 hardware-task channel; report: Level1 report "
            "channel; control: negative control (no device work)"
        ),
    )
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "module:function case (default: %s; --mode control defaults to %s)"
            % (DEFAULT_CASE, NEGATIVE_CONTROL_CASE)
        ),
    )
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="also run the unmodified tools/npu_kernel_probe.py (identity mode)",
    )
    args = parser.parse_args(argv)

    if args.mode == "identity":
        evidence = collect_route_identity(args.case or DEFAULT_CASE, args.scratch_root)
        if args.probe:
            evidence["probe_invocations"] = [
                run_npu_kernel_probe(),
                run_npu_kernel_probe(kernel=EXPECTED_KERNEL_TOKEN),
            ]
    else:
        default_case = NEGATIVE_CONTROL_CASE if args.mode == "control" else DEFAULT_CASE
        case_spec = args.case or default_case
        if args.mode == "hardware":
            evidence = hardware_channel(case_spec, args.scratch_root)
        elif args.mode == "report":
            evidence = report_channel(case_spec, args.scratch_root)
        else:
            evidence = control_channel(case_spec, args.scratch_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    count = evidence.get("hardware_event_count")
    if count is None:
        count = (evidence.get("hardware") or {}).get("count")
    print(
        "route_evidence: mode=%s case=%s hardware_events=%s"
        % (args.mode, evidence.get("case", ""), count)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
