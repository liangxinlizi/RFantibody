#!/usr/bin/env bash
# Build the native AscendC graph kernels and their PyTorch bindings.
#
# Official route: the installed CANN `ascendc_kernel_cmake/ascendc.cmake`
# provides `ascendc_library(... SHARED)`, which owns device compilation, the
# generated launcher/host stub, kernel registration and packaging.  This script
# only drives that target and the thin binding package.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/.." && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
PY="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

# The CANN runtime/compiler environment must be present for the generated build.
if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
fi
export ASCEND_HOME_PATH="${CANN}"

BUILD_DIR="${HERE}/build"
INSTALL_DIR="${HERE}/install"
rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${HERE}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DSOC_VERSION="ascend910b2c" \
    -DASCEND_CANN_PACKAGE_PATH="${CANN}" \
    -DASCEND_PYTHON_EXECUTABLE="${PY}" \
    -DRUN_MODE=npu

cmake --build "${BUILD_DIR}" --target graph_probe -j"$(nproc)"
cmake --install "${BUILD_DIR}"

# The official target installs under the GNUInstallDirs libdir (observed: lib64
# on this CANN).  Resolve the real file instead of assuming "lib", then stage it
# both at the declared artifact path and beside the Python package.
NATIVE_SO="$(find "${INSTALL_DIR}" -type f -name 'libgraph_probe.so' -print -quit)"
if [[ -z "${NATIVE_SO}" ]]; then
    echo "build.sh: libgraph_probe.so not found under ${INSTALL_DIR}" >&2
    exit 1
fi
mkdir -p "${INSTALL_DIR}/lib" "${HERE}/python/ascend_kernel"
cp "${NATIVE_SO}" "${INSTALL_DIR}/lib/libgraph_probe.so"
cp "${NATIVE_SO}" "${HERE}/python/ascend_kernel/libgraph_probe.so"
# setuptools resolves relative CppExtension sources against the process cwd, so
# build the binding from the package directory rather than the caller's cwd.
(
    cd "${HERE}"
    "${PY}" setup.py build_ext --inplace
    "${PY}" -m pip install --force-reinstall --no-deps --no-build-isolation .
)

echo "ascend-kernel build complete: ${INSTALL_DIR}/lib/libgraph_probe.so (from ${NATIVE_SO})"
