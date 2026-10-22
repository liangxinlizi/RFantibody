#!/usr/bin/env python3
"""Build the thin PyTorch binding for the native AscendC graph kernels.

The device kernels and their generated host stub are produced by the installed
CANN route (``csrc/ascend_graph/CMakeLists.txt`` -> ``ascendc_library(...)``).
This setup only compiles ``csrc/bindings.cpp`` against the generated
``aclrtlaunch_*.h`` headers and links the resulting ``libascend_graph.so``,
placing the compiled extension in the real ``rfantibody.ascend_ops`` package
directory (the project itself is installed in editable mode, so no extra
``pip install`` of this binding is required or performed).

Run through ``csrc/ascend_graph/build.sh`` so the kernel library is staged first.
"""

import os
import pathlib
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
INSTALL = HERE / "install"
PACKAGE_DIR = PROJECT_ROOT / "src" / "rfantibody" / "ascend_ops"

KERNEL_LIB_NAME = "libascend_graph.so"

# Stage the native library beside the Python package so the extension's
# `$ORIGIN` rpath resolves it without any environment variable.
KERNEL_SO = INSTALL / "lib" / KERNEL_LIB_NAME
if not KERNEL_SO.is_file():
    matches = sorted(INSTALL.glob("lib*/" + KERNEL_LIB_NAME))
    if matches:
        KERNEL_SO = matches[0]
if KERNEL_SO.is_file():
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(KERNEL_SO, PACKAGE_DIR / KERNEL_LIB_NAME)

CANN = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

_INCLUDE_DIRS = [
    str(INSTALL / "include" / "ascend_graph"),
    os.path.join(CANN, "include"),
]
_LIBRARY_DIRS = [
    str(INSTALL / "lib"),
    str(INSTALL / "lib64"),
    os.path.join(CANN, "lib64"),
]
# `$ORIGIN` covers the staged copy next to the module; the absolute entries let
# the same build run before staging.  `--disable-new-dtags` emits DT_RPATH so the
# CANN directories also resolve the transitive dependencies of
# libascend_graph.so (the official target is built with CMAKE_SKIP_RPATH).
_RPATHS = ["$ORIGIN"] + _LIBRARY_DIRS

setup(
    name="rfantibody_ascend_ops",
    version="0.0.1",
    description="Native AscendC graph operators for the RFantibody NPU migration",
    package_dir={
        "rfantibody.ascend_ops": os.path.relpath(PACKAGE_DIR, HERE)
    },
    ext_modules=[
        CppExtension(
            name="rfantibody.ascend_ops._ascend_graph",
            sources=[str(HERE / "bindings.cpp")],
            include_dirs=_INCLUDE_DIRS,
            library_dirs=_LIBRARY_DIRS,
            libraries=["ascend_graph", "ascendcl"],
            extra_compile_args=["-std=c++17"],
            extra_link_args=["-Wl,--disable-new-dtags"]
            + ["-Wl,-rpath," + path for path in _RPATHS],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
