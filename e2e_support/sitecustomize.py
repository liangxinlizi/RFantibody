"""Device-usage guard injected into RFantibody NPU stages via ``PYTHONPATH``.

``e2e_run.py`` prepends this directory to ``PYTHONPATH`` and sets
``AE_DEVICE_GUARD_OUT`` for every pipeline stage subprocess.  CPython imports
``sitecustomize`` automatically at interpreter startup, before project code, so
this module can observe device placement without modifying the product source.

Per process it records:

* ``devices_seen``: device types observed for module placement/forward output;
* ``npu_used`` / ``cuda_used`` / ``cpu_used``: whether those devices were seen;
* ``modules_moved`` and per-device move counters;
* ``forward_calls`` and per-device module-forward output counters;
* ``torch_npu_imported`` and any guard errors.

The guard only observes; it never changes device selection or model behaviour
and it cannot turn a CPU execution into a passing NPU execution.
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
    "move_devices": {},
    "forward_calls": 0,
    "forward_devices": {},
    "torch_npu_imported": False,
    "errors": [],
}


def _record_device(device_type: str) -> None:
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


def _scan(value, buckets: dict) -> None:  # type: ignore[no-untyped-def]
    """Collect tensor device types from nested outputs without touching data."""
    import torch

    if torch.is_tensor(value):
        device_type = value.device.type
        buckets[device_type] = buckets.get(device_type, 0) + 1
        _record_device(device_type)
    elif isinstance(value, dict):
        for item in value.values():
            _scan(item, buckets)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _scan(item, buckets)


def _module_device(module):  # type: ignore[no-untyped-def]
    try:
        return next(module.parameters()).device.type
    except StopIteration:
        try:
            return next(module.buffers()).device.type
        except StopIteration:
            return ""


def _install_torch_guard() -> None:
    import torch  # imported at guard installation time on purpose

    try:
        import torch_npu  # type: ignore  # noqa: F401

        _STATE["torch_npu_imported"] = True
    except Exception as exc:  # pragma: no cover - environment dependent
        _STATE["errors"].append(f"torch_npu import failed: {type(exc).__name__}: {exc}")

    original_apply = torch.nn.Module._apply
    original_to = torch.nn.Module.to
    original_call = torch.nn.Module.__call__

    def _observe_module(module) -> None:  # type: ignore[no-untyped-def]
        device_type = _module_device(module)
        if not device_type:
            return
        _STATE["modules_moved"] += 1
        moves = _STATE["move_devices"]
        moves[device_type] = moves.get(device_type, 0) + 1
        _record_device(device_type)

    def _patched_apply(self, fn, recurse=True):  # type: ignore[no-untyped-def]
        result = original_apply(self, fn, recurse=recurse)
        _observe_module(self)
        return result

    def _patched_to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_to(self, *args, **kwargs)
        if isinstance(result, torch.nn.Module):
            _observe_module(result)
        return result

    def _patched_call(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_call(self, *args, **kwargs)
        _STATE["forward_calls"] += 1
        buckets: dict = {}
        _scan(result, buckets)
        for device_type, count in buckets.items():
            forwards = _STATE["forward_devices"]
            forwards[device_type] = forwards.get(device_type, 0) + count
        return result

    torch.nn.Module._apply = _patched_apply  # type: ignore[assignment]
    torch.nn.Module.to = _patched_to  # type: ignore[assignment]
    torch.nn.Module.__call__ = _patched_call  # type: ignore[assignment]


def _write_state() -> None:
    if not _OUT_PATH:
        return
    if not (_STATE["modules_moved"] or _STATE["forward_calls"] or _STATE["errors"]):
        # Short-lived helper/interpreter processes leave no evidence behind.
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
