"""Fixtures for the native AscendC graph-parity / route tests.

The fixtures here do three things, in this order:

1. detect a *genuine* Ascend NPU through ``torch_npu`` / ``torch.npu`` (never a
   spoofed CUDA device and never a fabricated reference),
2. build the native operator library from source with the project's documented
   build driver when the artifact is missing or older than its sources,
3. import the compiled binding and expose the artifact path/hash alongside the
   public ``rfantibody.ascend_ops`` API.

Building is part of the evaluated gate: if an NPU exists and the build fails, the
tests fail (they do not skip).  A skip is only produced when this host genuinely
has no NPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL_DIR = PROJECT_ROOT / "csrc" / "ascend_graph"
BUILD_SCRIPT = KERNEL_DIR / "build.sh"
INSTALL_DIR = KERNEL_DIR / "install"
PACKAGE_DIR = PROJECT_ROOT / "src" / "rfantibody" / "ascend_ops"
DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / "outputs" / "evidence" / "graph_parity"
BUILD_TIMEOUT_SECONDS = 1800
KERNEL_LIB_NAME = "libascend_graph.so"


def evidence_dir() -> Path:
    """Directory for machine-readable operator-parity evidence.

    Defaults to the git-ignored ``outputs/evidence`` tree; override the root
    with ``RF_EVIDENCE_DIR`` when a run needs a different location.
    """
    override = os.environ.get("RF_EVIDENCE_DIR")
    root = Path(override) if override else PROJECT_ROOT / "outputs" / "evidence"
    return root / "graph_parity"


def _build_inputs() -> list[Path]:
    inputs = [KERNEL_DIR / "CMakeLists.txt", KERNEL_DIR / "setup.py"]
    inputs.extend(sorted((KERNEL_DIR / "op_kernel").glob("*.cpp")))
    # Shared kernel bodies are header-only (`segment_reduce_common.h`), so a
    # header edit must invalidate a stale artifact exactly like a `.cpp` edit.
    inputs.extend(sorted((KERNEL_DIR / "op_kernel").glob("*.h")))
    inputs.append(KERNEL_DIR / "bindings.cpp")
    return [path for path in inputs if path.is_file()]


def _installed_kernel_library() -> "Path | None":
    preferred = INSTALL_DIR / "lib" / KERNEL_LIB_NAME
    if preferred.is_file():
        return preferred
    matches = sorted(INSTALL_DIR.glob("lib*/" + KERNEL_LIB_NAME))
    return matches[0] if matches else None


def _python_binding() -> "Path | None":
    matches = sorted(PACKAGE_DIR.glob("_ascend_graph*.so"))
    return matches[0] if matches else None


def _needs_build() -> bool:
    library = _installed_kernel_library()
    binding = _python_binding()
    if library is None or binding is None:
        return True
    newest_input = max(path.stat().st_mtime for path in _build_inputs())
    return min(library.stat().st_mtime, binding.stat().st_mtime) < newest_input


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npu_available() -> bool:
    """Genuine NPU detection through the vendor torch_npu stack."""
    try:
        import torch_npu  # noqa: F401
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.npu.is_available()) and int(torch.npu.device_count()) >= 1
    except Exception:  # pragma: no cover - defensive
        return False


@pytest.fixture(scope="session")
def npu_device():
    """The single Ascend device under test, or a genuine skip if none exists."""
    if not npu_available():
        pytest.skip("no Ascend NPU on this host (torch.npu.is_available() is False)")
    import torch

    return torch.device("npu:0")


@pytest.fixture(scope="session")
def native_graph_ops(npu_device):
    """Build (if stale) and import the native graph operators.

    Returns a mapping with the imported module, the compiled artifact path and
    hash, the build log path and the evidence directory.
    """
    build_log = None
    if _needs_build():
        out_dir = evidence_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        build_log = out_dir / "build.log"
        env = dict(os.environ)
        env["PYTHON"] = sys.executable
        env.setdefault("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        completed = subprocess.run(
            ["bash", str(BUILD_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        build_log.write_text(
            "COMMAND: bash {}\nCWD: {}\nEXIT: {}\n\nSTDOUT\n{}\n\nSTDERR\n{}\n".format(
                BUILD_SCRIPT,
                PROJECT_ROOT,
                completed.returncode,
                completed.stdout,
                completed.stderr,
            ),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or "").splitlines()[-40:])
            raise AssertionError(
                "native AscendC build failed "
                f"(exit {completed.returncode}); see {build_log}\n{tail}"
            )

    library = _installed_kernel_library()
    binding = _python_binding()
    assert library is not None, f"kernel library missing under {INSTALL_DIR}"
    assert binding is not None, f"python binding missing under {PACKAGE_DIR}"

    import rfantibody.ascend_ops as ops

    info = {
        "ops": ops,
        "artifact": str(library),
        "artifact_sha256": _sha256(library),
        "binding": str(binding),
        "binding_sha256": _sha256(binding),
        "build_log": str(build_log) if build_log else None,
        "evidence_dir": str(evidence_dir()),
        "npu_available": ops.is_available(),
    }
    out_dir = evidence_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "native_artifacts.json").write_text(
        json.dumps(
            {
                "kernel_library": str(library),
                "kernel_library_sha256": info["artifact_sha256"],
                "python_binding": str(binding),
                "python_binding_sha256": info["binding_sha256"],
                "build_log": info["build_log"],
                "soc_version_expected": "ascend910b2c",
                "build_driver": str(BUILD_SCRIPT),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return info
