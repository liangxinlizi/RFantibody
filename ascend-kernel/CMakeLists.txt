cmake_minimum_required(VERSION 3.16.0)
project(ascend_graph_probe LANGUAGES CXX)

# Official complete route: the CANN installation ships
# `ascendc_kernel_cmake/ascendc.cmake`, whose `ascendc_library(... SHARED)`
# target owns device compilation, generated launcher/host-stub, registration
# and packaging for this CANN/chip.  Do not hand-roll those stages.
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set (source the CANN set_env.sh)")
endif()
get_filename_component(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" ABSOLUTE)

set(SOC_VERSION "ascend910b2c" CACHE STRING "Ascend SoC version")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME}" CACHE STRING "CANN package path")
set(RUN_MODE "npu" CACHE STRING "run mode")

set(ASCENDC_CMAKE
    "${ASCEND_HOME}/x86_64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
if(NOT EXISTS "${ASCENDC_CMAKE}")
    message(FATAL_ERROR "installed ascendc.cmake not found: ${ASCENDC_CMAKE}")
endif()

include("${ASCENDC_CMAKE}")

# Produces ${CMAKE_INSTALL_PREFIX}/lib/libgraph_probe.so (exports the generated
# `aclrtlaunch_graph_probe`, `aclrtlaunch_edge_softmax`, ... ,
# `aclrtlaunch_graph_scatter_sum` and `aclrtlaunch_edge_dot_v` host entry
# points) and
# ${CMAKE_INSTALL_PREFIX}/include/graph_probe/aclrtlaunch_triple_chevrons_func.h.
ascendc_library(graph_probe SHARED
    csrc/op_kernel/graph_probe.cpp
    csrc/op_kernel/edge_softmax.cpp
    csrc/op_kernel/graph_reduce.cpp
    csrc/op_kernel/edge_dot_v.cpp)
