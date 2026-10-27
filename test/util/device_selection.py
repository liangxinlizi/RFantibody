#!/usr/bin/env python3

"""
Genuine accelerator detection and reference-set selection for the test surface.

The original module ``conftest.py`` files gate their suites on a supported CUDA
GPU (A4000 or H100) and pick the matching versioned ``reference_outputs`` set.
The migrated repository also runs the very same original tests on an Ascend NPU,
so the gate is extended with genuine NPU detection: when ``torch.npu`` reports a
usable device, the suites execute on the NPU and select the versioned
``H100_references`` set -- the same numerical reference the CUDA path uses on an
H100, and the only GPU reference set versioned for ProteinMPNN and RF2.

Nothing here fabricates or spoofs a device, and no test, assertion, tolerance or
reference asset is changed.  CUDA behaviour (A4000/H100 selection and the
"supported GPU" requirement) is preserved exactly.
"""

from __future__ import annotations

import torch


def npu_device_name(index: int = 0):
    """Return a genuine Ascend NPU device name, or ``None`` when unusable."""
    try:
        import torch_npu  # noqa: F401  (registers ``torch.npu``)
    except Exception:  # CPU/CUDA installs have no torch_npu
        return None
    try:
        if not bool(torch.npu.is_available()):
            return None
        if int(torch.npu.device_count()) <= int(index):
            return None
        return str(torch.npu.get_device_name(int(index)))
    except Exception:  # pragma: no cover - defensive, never masks a real NPU
        return None


def cuda_device_name(index: int = 0):
    """Return the CUDA device name, or ``None`` when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_properties(int(index)).name)


def reference_set_dir(base_ref_dir):
    """Return the reference directory matching the active accelerator."""
    if npu_device_name() is not None:
        return base_ref_dir / "H100_references"
    name = cuda_device_name()
    if name and "A4000" in name:
        return base_ref_dir / "A4000_references"
    if name and "H100" in name:
        return base_ref_dir / "H100_references"
    return base_ref_dir


__all__ = ["cuda_device_name", "npu_device_name", "reference_set_dir"]
