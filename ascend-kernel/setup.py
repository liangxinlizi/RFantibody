#!/usr/bin/env python3
"""Build the thin PyTorch binding for the native AscendC `graph_probe` kernel.

The device kernel and its generated host stub are produced by the installed
CANN route (``ascend-kernel/CMakeLists.txt`` -> ``ascendc_library``).  This
setup only compiles ``csrc/bindings.cpp`` against the generated header and links
the resulting ``libgraph_probe.so``, packaging both into the ``ascend_kernel``
Python package.
"""

import os
import pathlib
import shutil

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = pathlib.Path(__file__).resolve().parent
INSTALL = HERE / "install"
PACKAGE = HERE / "python" / "ascend_kernel"

# The kernel library is built by build.sh before setup.py is invoked; stage it
# inside the package so the wheel is self-contained and `import ascend_kernel`
# resolves it through the extension's `$ORIGIN` rpath.
_KERNEL_SO = INSTALL / "lib" / "libgraph_probe.so"
if _KERNEL_SO.is_file():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_KERNEL_SO, PACKAGE / "libgraph_probe.so")

CANN = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

import torch_npu  # noqa: E402  (installed vendor stack; provides include/lib dirs)

_TNPU_ROOT = pathlib.Path(torch_npu.__file__).resolve().parent

_INCLUDE_DIRS = [
    str(INSTALL / "include" / "graph_probe"),
    os.path.join(CANN, "include"),
]
_LIBRARY_DIRS = [
    str(INSTALL / "lib"),
    os.path.join(CANN, "lib64"),
    str(_TNPU_ROOT / "lib"),
]

setup(
    name="ascend_kernel",
    version="0.0.1",
    description="Native AscendC graph operators for the RFantibody NPU migration",
    packages=["ascend_kernel"],
    package_dir={"": "python"},
    package_data={"ascend_kernel": ["*.so"]},
    ext_modules=[
        CppExtension(
            name="ascend_kernel._graph_probe_binding",
            sources=[str(HERE / "csrc" / "bindings.cpp")],
            include_dirs=_INCLUDE_DIRS,
            library_dirs=_LIBRARY_DIRS,
            libraries=["graph_probe", "ascendcl"],
            extra_compile_args=["-std=c++17"],
            extra_link_args=[
                "-Wl,-rpath,$ORIGIN",
                "-Wl,-rpath," + str(INSTALL / "lib"),
                "-Wl,-rpath," + os.path.join(CANN, "lib64"),
                "-Wl,-rpath," + str(_TNPU_ROOT / "lib"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
