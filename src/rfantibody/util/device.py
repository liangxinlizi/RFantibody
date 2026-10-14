"""NPU-only device selection shared by the RFantibody stages.

This migrated build runs on an Ascend NPU.  Importing ``torch_npu`` registers
the ``PrivateUse1`` (NPU) backend; if that import fails the real error is
propagated.  ``get_device`` returns the current NPU device and raises an explicit
``RuntimeError`` when no usable NPU is present.  There is deliberately no CUDA or
CPU fallback: a silent fallback would run the model on the wrong device, which
this migration forbids.
"""

import torch


def import_torch_npu():
    """Import ``torch_npu``, registering the ``PrivateUse1`` (NPU) backend.

    The real import error (missing or broken CANN/torch_npu) is propagated so the
    process fails immediately and visibly instead of silently continuing.
    """
    import torch_npu  # noqa: F401

    return torch_npu


def get_device():
    """Return the Ascend NPU device; CPU/CUDA fallback is disabled.

    Raises ``RuntimeError`` when no usable NPU backend is present rather than
    silently selecting ``cuda`` or ``cpu``.
    """
    import_torch_npu()  # register the PrivateUse1 backend, or raise
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError(
            "No Ascend NPU available: this migrated RFantibody build requires a "
            "working torch_npu/NPU; CUDA and CPU fallback are disabled."
        )
    return torch.device(f"npu:{torch.npu.current_device()}")
