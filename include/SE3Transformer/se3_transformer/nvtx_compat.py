"""Device-agnostic NVTX range shim for the SE(3)/RFdiffusion path.

Upstream wraps SE(3) hot sections in ``torch.cuda.nvtx.range``.  On a CPU-only
``torch`` build (this vendor stack is ``torch 2.5.1+cpu`` + ``torch_npu 2.5.1``)
``torch._C._nvtx`` is absent, so every call raises
``RuntimeError: NVTX functions not installed. Are you sure you have a CUDA
build?`` and aborts the forward pass on Ascend NPU.  NVTX ranges are pure
profiler annotations and never affect computation.

This shim delegates to the real CUDA NVTX range only when a genuine CUDA build
is present; otherwise it yields a no-op context manager.  Behaviour on CUDA
hosts is unchanged; on NPU/CPU it is a safe no-op.
"""

from contextlib import contextmanager

try:  # only importable when the CUDA extension was built
    from torch.cuda.nvtx import range as _cuda_nvtx_range
except Exception:  # pragma: no cover - defensive
    _cuda_nvtx_range = None


def _cuda_nvtx_available() -> bool:
    if _cuda_nvtx_range is None:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


@contextmanager
def range(msg, *args, **kwargs):
    """Context manager matching ``torch.cuda.nvtx.range`` (no-op on NPU/CPU)."""
    if _cuda_nvtx_available():
        with _cuda_nvtx_range(msg, *args, **kwargs):
            yield
    else:
        yield
