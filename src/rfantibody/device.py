"""Accelerator device selection without silent CPU fallback.

The migrated product path must run on the Ascend NPU (the runtime constant
``constraints.no_cpu_fallback``), the documented CUDA installation path must keep
selecting CUDA, and CPU-only hosts must remain usable for the CPU-DGL reference
tests.  ``get_accelerator_device`` expresses exactly that, preferring a genuine
``torch.npu`` device, then CUDA, and only then CPU -- unless the caller requires
an accelerator, in which case it raises instead of degrading to CPU.

``torch_npu`` is imported lazily, so a CUDA/CPU install never needs the NPU
vendor stack.  Device detection is genuine: it asks ``torch.npu.is_available()``
and ``torch.cuda.is_available()`` from the real vendor builds.  Nothing here
fabricates or spoofs a device.
"""

from __future__ import annotations

import torch


def npu_available() -> bool:
    """Whether a genuine Ascend NPU is usable through ``torch.npu``."""
    try:
        import torch_npu  # noqa: F401  (registers torch.npu)
    except Exception:  # pragma: no cover - CPU/CUDA installs have no torch_npu
        return False
    try:
        return bool(torch.npu.is_available()) and int(torch.npu.device_count()) >= 1
    except Exception:  # pragma: no cover - defensive
        return False


def get_accelerator_device(
    index: int = 0, require_accelerator: bool = False
) -> torch.device:
    """Return ``npu:index``, else ``cuda:index``, else ``cpu`` (or raise).

    With ``require_accelerator=True`` no CPU device is ever returned: a host
    without a usable accelerator raises ``RuntimeError`` so that a caller cannot
    silently execute neural computation on the CPU.
    """
    index = int(index)

    if npu_available():
        if int(torch.npu.device_count()) > index:
            return torch.device(f"npu:{index}")
        if require_accelerator:
            raise RuntimeError(
                f"torch.npu reports {int(torch.npu.device_count())} device(s), so "
                f"npu:{index} does not exist; refusing to fall back to CPU"
            )

    if torch.cuda.is_available() and int(torch.cuda.device_count()) > index:
        return torch.device(f"cuda:{index}")

    if require_accelerator:
        raise RuntimeError(
            "no Ascend NPU or CUDA device is available for "
            f"device index {index}; refusing to fall back to CPU"
        )
    return torch.device("cpu")


__all__ = ["get_accelerator_device", "npu_available"]
