"""Device-usage guard injected into RFantibody NPU stages through PYTHONPATH.

The e2e harness prepends this directory to ``PYTHONPATH`` and sets
``AE_DEVICE_GUARD_OUT`` for every stage subprocess.  Python imports this
``sitecustomize`` module automatically at interpreter start, before any project
code runs, so it can observe where ``torch.nn.Module`` objects are actually
placed.

It records, per process:

* every device type any module was moved to (``devices_seen``);
* whether an NPU device was actually used (``npu_used``);
* whether CUDA was used (``cuda_used``);
* whether a CPU device was used (``cpu_used``).

The e2e harness requires ``npu_used`` and rejects any stage that only placed
modules on CPU (silent CPU fallback).  The guard only observes the final
``Module.to``/``Module._apply`` implementations; it never changes model
behaviour.
"""

from __future__ import annotations

import atexit
import json
import os

_OUT_PATH = os.environ.get("AE_DEVICE_GUARD_OUT")
_OWNER_PID = os.getpid()

_STATE = {
    "pid": _OWNER_PID,
    "devices_seen": [],
    "npu_used": False,
    "cuda_used": False,
    "cpu_used": False,
    "modules_moved": 0,
    "torch_npu_imported": False,
    "errors": [],
}


def _record(device_type: str) -> None:
    if not device_type:
        return
    if device_type not in _STATE["devices_seen"]:
        _STATE["devices_seen"].append(device_type)
    if device_type == "npu":
        _STATE["npu_used"] = True
    elif device_type == "cuda":
        _STATE["cuda_used"] = True
    elif device_type == "cpu":
        _STATE["cpu_used"] = True


def _module_device(module) -> str:  # type: ignore[no-untyped-def]
    try:
        return next(module.parameters()).device.type
    except StopIteration:
        try:
            return next(module.buffers()).device.type
        except StopIteration:
            return ""


def _install_torch_guard() -> None:
    import torch  # noqa: WPS433 (runtime import is intentional)

    try:
        import torch_npu  # type: ignore  # noqa: F401

        _STATE["torch_npu_imported"] = True
    except Exception as exc:  # pragma: no cover - environment dependent
        _STATE["errors"].append(f"torch_npu import failed: {type(exc).__name__}: {exc}")

    # ``torch_npu`` patches Module.to/cast_weight; wrap the final implementation
    # so that both native and vendor device moves are observed.
    original_apply = torch.nn.Module._apply
    original_to = torch.nn.Module.to

    def _observe(module) -> None:  # type: ignore[no-untyped-def]
        device = _module_device(module)
        if device:
            _STATE["modules_moved"] += 1
            _record(device)

    def _patched_apply(self, fn, recurse=True):  # type: ignore[no-untyped-def]
        result = original_apply(self, fn, recurse=recurse)
        _observe(self)
        return result

    def _patched_to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_to(self, *args, **kwargs)
        if isinstance(result, torch.nn.Module):
            _observe(result)
        return result

    torch.nn.Module._apply = _patched_apply  # type: ignore[assignment]
    torch.nn.Module.to = _patched_to  # type: ignore[assignment]


def _write_state() -> None:
    if not _OUT_PATH:
        return
    # torch_npu forks/execs short-lived helper processes that inherit this
    # environment.  Writing to a per-process file prevents an empty helper from
    # clobbering the real stage result; the harness unions all of them.
    if not (_STATE["modules_moved"] or _STATE["errors"]):
        return
    target = f"{_OUT_PATH}.{os.getpid()}.json"
    try:
        payload = json.dumps(_STATE, indent=2, sort_keys=True) + "\n"
        tmp_path = f"{target}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_path, target)
    except Exception:
        # Evidence writing must never mask the real stage result.
        pass


try:
    _install_torch_guard()
except Exception as exc:  # pragma: no cover - environment dependent
    _STATE["errors"].append(f"guard install failed: {type(exc).__name__}: {exc}")

atexit.register(_write_state)
