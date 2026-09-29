cmake_minimum_required(VERSION 3.24)

project(thermal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA is optional at configure time: THERMAL must configure and build its
# non-GPU pieces even on machines without an NVIDIA toolchain. Native CUDA
# kernels (native/cuda) land in Phase 7 and enable this language then.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "THERMAL: CUDA compiler found (${CMAKE_CUDA_COMPILER}) — GPU targets enabled")
    set(THERMAL_CUDA_ENABLED ON)
else()
    message(WARNING "THERMAL: no CUDA compiler found at configure time — GPU targets disabled")
    set(THERMAL_CUDA_ENABLED OFF)
endif()

# Subdirectories are added only once they contain real build targets
# (Phase 7). An empty add_subdirectory() on a directory with no
# CMakeLists.txt is a configure-time error, so we check explicitly rather
# than pretending targets exist.
if(EXISTS "${CMAKE_SOURCE_DIR}/native/cpp/CMakeLists.txt")
    add_subdirectory(native/cpp)
endif()
if(THERMAL_CUDA_ENABLED AND EXISTS "${CMAKE_SOURCE_DIR}/native/cuda/CMakeLists.txt")
    add_subdirectory(native/cuda)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/native/benchmarks/CMakeLists.txt")
    add_subdirectory(native/benchmarks)
endif()
