cmake_minimum_required(VERSION 3.16.0)
project(rfantibody_ascend_graph LANGUAGES CXX)

# Official complete route: the installed CANN ships
# `x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake`, whose
# `ascendc_library(... SHARED)` target owns device compilation, the generated
# launcher/host stub, kernel registration and packaging for this CANN/chip.  No
# ccec/merge/pack stage is hand-rolled here.
if(DEFINED ENV{ASCEND_HOME_PATH})
    get_filename_component(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" ABSOLUTE)
elseif(DEFINED ASCEND_CANN_PACKAGE_PATH)
    get_filename_component(ASCEND_HOME "${ASCEND_CANN_PACKAGE_PATH}" ABSOLUTE)
else()
    message(FATAL_ERROR
        "ASCEND_HOME_PATH is not set; source the CANN set_env.sh or pass "
        "-DASCEND_CANN_PACKAGE_PATH=<CANN root>")
endif()

# Ascend SoC targeted by the shipped kernels; override with -DSOC_VERSION=<soc>.
set(SOC_VERSION "ascend910b2c" CACHE STRING "Ascend SoC version")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME}" CACHE STRING "CANN package path")
set(RUN_MODE "npu" CACHE STRING "run mode")

set(ASCENDC_CMAKE
    "${ASCEND_HOME}/x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
if(NOT EXISTS "${ASCENDC_CMAKE}")
    message(FATAL_ERROR "installed ascendc.cmake not found: ${ASCENDC_CMAKE}")
endif()

include("${ASCENDC_CMAKE}")

# Produces ${CMAKE_INSTALL_PREFIX}/lib(64)/libascend_graph.so (exporting the
# generated `aclrtlaunch_graph_probe`, `aclrtlaunch_edge_dot_v`,
# `aclrtlaunch_edge_topology`, `aclrtlaunch_copy_e_sum`,
# `aclrtlaunch_copy_e_mean`, `aclrtlaunch_edge_softmax_dst`,
# `aclrtlaunch_avg_pooling`, `aclrtlaunch_max_pooling` and
# `aclrtlaunch_tied_attention_logits` host entry points) and
# ${CMAKE_INSTALL_PREFIX}/include/ascend_graph/aclrtlaunch_*.h.  The shared
# reduction body lives in `op_kernel/segment_reduce_common.h` (included by the
# two thin reduction entry translation units), the shared global-pooling body
# lives in `op_kernel/pooling_common.h` (included by the two pooling entry
# translation units) and the ordered compensated-FMA contraction primitives live
# in `op_kernel/compensated_fma_ordered.h` (included by the tied-attention
# entry translation unit).
ascendc_library(ascend_graph SHARED
    op_kernel/graph_probe.cpp
    op_kernel/e_dot_v.cpp
    op_kernel/edge_topology.cpp
    op_kernel/copy_e_sum.cpp
    op_kernel/copy_e_mean.cpp
    op_kernel/edge_softmax_dst.cpp
    op_kernel/avg_pooling.cpp
    op_kernel/max_pooling.cpp
    op_kernel/tied_attention_logits.cpp)
