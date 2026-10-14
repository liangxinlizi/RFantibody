"""Thin Python bridge to the native AscendC graph operators.

The compiled extension registers ``npu::graph_probe``, ``npu::edge_softmax``,
``npu::copy_e_sum``, ``npu::copy_e_mean``, ``npu::avg_pool`` and
``npu::e_dot_v`` for dispatch key ``PrivateUse1``; this module makes them
importable from the SE3Transformer package without changing the existing
``graph_compat`` surface.
"""

from __future__ import annotations

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_BRIDGE_DIR = _PROJECT_ROOT / "ascend-kernel" / "python"
if _BRIDGE_DIR.is_dir() and str(_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_DIR))

import ascend_kernel  # noqa: E402
from ascend_kernel import (  # noqa: E402,F401
    avg_pool,
    copy_e_mean,
    copy_e_sum,
    e_dot_v,
    edge_softmax,
    graph_probe,
)

__all__ = [
    "graph_probe",
    "edge_softmax",
    "copy_e_sum",
    "copy_e_mean",
    "avg_pool",
    "e_dot_v",
    "ascend_kernel",
]
