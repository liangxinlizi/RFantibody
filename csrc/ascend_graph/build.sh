#!/usr/bin/env bash
# Build the native AscendC graph kernels and the thin PyTorch binding.
#
# Official route only: the installed CANN
# `x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake` provides
# `ascendc_library(ascend_graph SHARED ...)`, which owns device compilation, the
# generated launcher/host stub, kernel registration and packaging.  This script
# only drives that target and then compiles `bindings.cpp` against the generated
# headers.  It is idempotent: it rebuilds from a clean build directory and stages
# the resulting artifacts in fixed locations.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
PY="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
PACKAGE_DIR="${PROJECT_ROOT}/src/rfantibody/ascend_ops"

if [[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
fi
export ASCEND_HOME_PATH="${CANN}"

if [[ ! -d "${CANN}" ]]; then
    echo "build.sh: CANN root not found: ${CANN}" >&2
    exit 1
fi
if [[ ! -x "${PY}" ]]; then
    echo "build.sh: python interpreter not found: ${PY}" >&2
    exit 1
fi

BUILD_DIR="${HERE}/build"
INSTALL_DIR="${HERE}/install"
rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"

CMAKE_CONFIGURE=(
    cmake -S "${HERE}" -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    -DSOC_VERSION=ascend910b2c
    -DASCEND_CANN_PACKAGE_PATH="${CANN}"
    -DASCEND_PYTHON_EXECUTABLE="${PY}"
    -DRUN_MODE=npu
)
echo "build.sh: ${CMAKE_CONFIGURE[*]}"
"${CMAKE_CONFIGURE[@]}"

echo "build.sh: cmake --build ${BUILD_DIR} --target ascend_graph -j$(nproc)"
cmake --build "${BUILD_DIR}" --target ascend_graph -j"$(nproc)"
cmake --install "${BUILD_DIR}"

# The official target installs under the GNUInstallDirs libdir (observed: lib64
# on this CANN).  Resolve the real file instead of assuming "lib", then stage it
# at the declared artifact path and beside the Python package so the extension's
# `$ORIGIN` rpath resolves it.
NATIVE_SO="$(find "${INSTALL_DIR}" -type f -name 'libascend_graph.so' -print -quit)"
if [[ -z "${NATIVE_SO}" ]]; then
    echo "build.sh: libascend_graph.so not found under ${INSTALL_DIR}" >&2
    exit 1
fi
mkdir -p "${INSTALL_DIR}/lib" "${PACKAGE_DIR}"
cp "${NATIVE_SO}" "${INSTALL_DIR}/lib/libascend_graph.so"
cp "${NATIVE_SO}" "${PACKAGE_DIR}/libascend_graph.so"

# setuptools resolves relative CppExtension sources and package_dir entries
# against the process cwd, so build the binding from this directory.
(
    cd "${HERE}"
    "${PY}" setup.py build_ext --inplace
)

BINDING="$(find "${PACKAGE_DIR}" -maxdepth 1 -type f -name '_ascend_graph*.so' -print -quit)"
if [[ -z "${BINDING}" ]]; then
    # Fall back to the in-tree build output if the in-place copy did not land in
    # the package directory.
    BINDING="$(find "${HERE}/build" -type f -name '_ascend_graph*.so' -print -quit)"
    if [[ -z "${BINDING}" ]]; then
        echo "build.sh: _ascend_graph*.so not found after build" >&2
        exit 1
    fi
    cp "${BINDING}" "${PACKAGE_DIR}/"
    BINDING="${PACKAGE_DIR}/$(basename "${BINDING}")"
fi

EXT_SUFFIX="$("${PY}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
echo "build.sh: native kernel library: ${INSTALL_DIR}/lib/libascend_graph.so (from ${NATIVE_SO})"
echo "build.sh: python binding:        ${BINDING}"
echo "build.sh: expected module:      rfantibody.ascend_ops._ascend_graph${EXT_SUFFIX}"
