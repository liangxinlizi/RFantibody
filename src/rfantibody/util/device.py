"""NPU-first device selection shared by the RFantibody stages.

The upstream code selected ``cuda`` only when ``torch.cuda.is_available()`` and
otherwise silently fell back to ``cpu``.  On an Ascend host there is no CUDA, so
that fallback would run the real model on CPU, which this migration forbids.

Importing ``torch_npu`` registers the ``PrivateUse1`` (NPU) backend.  If no NPU
is available the e2e device guard fails the stage rather than silently accepting
a CPU run.
"""

import torch


def import_torch_npu():
    try:
        import torch_npu  # noqa: F401

        return True
    except ImportError:
        return False


def get_device():
    import_torch_npu()  # register the PrivateUse1 backend
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{torch.npu.current_device()}")
    if torch.cuda.is_available():  # only used if a genuine CUDA stack exists
        return torch.device("cuda")
    return torch.device("cpu")
