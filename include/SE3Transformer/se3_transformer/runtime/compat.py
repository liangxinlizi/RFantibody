# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""CUDA-optional runtime shims for the SE3-Transformer model.

The upstream model imports two CUDA-only facilities unconditionally:

* ``torch.cuda.nvtx.range`` as the ``nvtx_range`` profiling context, and
* ``torch.cuda.amp.autocast(enabled=False)`` as a "keep this block in float32"
  guard.

On a CUDA build both keep exactly their original behaviour.  On a CPU or Ascend
NPU build the vendor stack has no CUDA runtime, and importing
``torch.cuda.nvtx`` raises ``RuntimeError: NVTX functions not installed. Are you
sure you have a CUDA build?``.  These shims therefore resolve to inert
equivalents there, so the model keeps one spelling on every device without
changing what the code computes:

* :func:`nvtx_range` returns the real CUDA NVTX range when CUDA is available and
  an inert ``contextlib.nullcontext()`` otherwise (profiling markers only).
* :func:`without_autocast` returns the real CUDA autocast context when CUDA is
  available and an inert replacement that works both as a context manager and as
  a decorator otherwise.  It never *enables* autocast, so it cannot change
  numerics; float32 stays float32 on NPU/CPU just as ``enabled=False`` keeps
  float32 on CUDA.
"""

from __future__ import annotations

import contextlib
import functools


class _NoAutocast:
    """Inert stand-in for ``torch.cuda.amp.autocast(enabled=False)``.

    Usable both as ``with without_autocast():`` and as ``@without_autocast()``,
    matching the two call spellings in the model and in ``Track_module``.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with _NoAutocast():
                return func(*args, **kwargs)

        return wrapped


def nvtx_range(name, *args, **kwargs):
    """CUDA NVTX range on a CUDA build, inert context manager everywhere else."""
    try:
        import torch

        if torch.cuda.is_available():
            from torch.cuda.nvtx import range as _cuda_range

            return _cuda_range(name, *args, **kwargs)
    except Exception:  # pragma: no cover - defensive: fall back to the no-op
        pass
    return contextlib.nullcontext()


def without_autocast():
    """``torch.cuda.amp.autocast(enabled=False)`` on CUDA, inert elsewhere."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.amp.autocast(enabled=False)
    except Exception:  # pragma: no cover - defensive: fall back to the no-op
        pass
    return _NoAutocast()
