# NERT (Neuriplo Engine Runtime): first-party ONNX inference runtime (CPU
# reference spine). Depends on nothing beyond the C++ standard library.

cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NERT_IS_TOP_LEVEL ON)
else()
    set(NERT_IS_TOP_LEVEL OFF)
endif()

project(nert VERSION 0.2.0 LANGUAGES CXX)

if(NERT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

option(NERT_BUILD_TESTS "Build the NERT tests" ${NERT_IS_TOP_LEVEL})

add_library(nert STATIC
    src/engine.cpp
    src/wire_reader.cpp
    src/model_loader.cpp
    src/shape_inference.cpp
    src/memory_planner.cpp
    src/cpu_device.cpp
    src/executor.cpp
    src/constant_folding.cpp
    src/kernels/cpu_kernels.cpp
    src/kernels/relu.cpp
    src/kernels/add.cpp
    src/kernels/reshape.cpp
    src/kernels/reduce_mean.cpp
    src/kernels/gemm.cpp
    src/kernels/mat_mul.cpp
    src/kernels/conv.cpp
    src/kernels/max_pool.cpp
    src/kernels/mul.cpp
    src/kernels/div.cpp
    src/kernels/sub.cpp
    src/kernels/sigmoid.cpp
    src/kernels/cast.cpp
    src/kernels/softmax.cpp
    src/kernels/reduce_max.cpp
    src/kernels/reduce_sum.cpp
    src/kernels/range.cpp
    src/kernels/pad.cpp
    src/kernels/mod.cpp
    src/kernels/equal.cpp
    src/kernels/greater.cpp
    src/kernels/less.cpp
    src/kernels/and.cpp
    src/kernels/not.cpp
    src/kernels/where.cpp
    src/kernels/concat.cpp
    src/kernels/split.cpp
    src/kernels/unsqueeze.cpp
    src/kernels/expand.cpp
    src/kernels/transpose.cpp
    src/kernels/slice.cpp
    src/kernels/gather.cpp
    src/kernels/gather_elements.cpp
    src/kernels/resize.cpp
    src/kernels/flatten.cpp
    src/kernels/shape.cpp
    src/kernels/constant_of_shape.cpp
    src/kernels/top_k.cpp
    src/kernels/floor.cpp
    src/kernels/identity.cpp
    src/kernels/pow.cpp
    src/kernels/tile.cpp
    src/kernels/average_pool.cpp
    src/kernels/grid_sample.cpp
    src/kernels/einsum.cpp
    src/kernels/conv_transpose.cpp
    src/kernels/erf.cpp
    src/kernels/sqrt.cpp
    src/kernels/tanh.cpp
    src/kernels/is_nan.cpp
    src/kernels/abs.cpp
    src/kernels/neg.cpp
    src/kernels/exp.cpp
    src/kernels/log.cpp
    src/kernels/cos.cpp
    src/kernels/sin.cpp
    src/kernels/clip.cpp
    src/kernels/squeeze.cpp
    src/kernels/layer_normalization.cpp
    src/kernels/instance_normalization.cpp
    src/kernels/elu.cpp
    src/kernels/max.cpp
    src/kernels/min.cpp
    src/kernels/less_or_equal.cpp
    src/kernels/greater_or_equal.cpp
    src/kernels/or.cpp
    src/kernels/reduce_l2.cpp
    src/kernels/arg_max.cpp
    src/kernels/cum_sum.cpp
    src/kernels/eye_like.cpp
    src/kernels/scatter_nd.cpp)
target_compile_features(nert PUBLIC cxx_std_17)
target_include_directories(nert
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")

add_library(nert::nert ALIAS nert)
# Compatibility alias for consumers that still link the old target name.
add_library(neuriplo_engine ALIAS nert)

# CUDA device layer (Phase N1), opt-in so the default configure/build/test
# never needs a toolkit: NERT_ENABLE_CUDA=ON requires CUDAToolkit and links
# the CUDA runtime into `nert`; OFF (default) leaves the link closure
# standard-library-only.
option(NERT_ENABLE_CUDA "Build NERT with the CUDA device layer" OFF)
if(NERT_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "NERT_ENABLE_CUDA=ON requires a CUDA compiler (nvcc); none was found.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # PUBLIC includes: test targets (and consumers building CUDA code) need
    # the toolkit headers; libraries stay PRIVATE (static-link propagation
    # still resolves the final link).
    target_include_directories(nert PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_sources(nert PRIVATE src/cuda_device.cpp src/cuda/gemm.cu src/cuda/relu.cu src/cuda/add.cu
                                src/cuda/mul.cu src/cuda/sub.cu src/cuda/div.cu src/cuda/max_pool.cu
                                src/cuda/average_pool.cu src/cuda/sigmoid.cu src/cuda/tanh.cu
                                src/cuda/softmax.cu src/cuda/reshape.cu src/cuda/unsqueeze.cu
                                src/cuda/flatten.cu src/cuda/expand.cu src/cuda/concat.cu
                                src/cuda/split.cu src/cuda/slice.cu src/cuda/transpose.cu
                                src/cuda/gather.cu src/cuda/gather_elements.cu src/cuda/mat_mul.cu
                                src/cuda/cast.cu src/cuda/equal.cu src/cuda/where.cu src/cuda/shape.cu
                                src/cuda/constant_of_shape.cu src/cuda/mod.cu src/cuda/reduce_max.cu
                                src/cuda/top_k.cu src/cuda/resize.cu src/cuda/table.cu)
    # PRIVATE: no CUDA type leaks through the public headers, so consumers
    # stay toolkit-free.
    target_link_libraries(nert PRIVATE CUDA::cudart CUDA::cublas)
    # cuDNN (N1b Conv) is optional even with CUDA on: the host has CUDA but
    # no cuDNN, and its ON build must keep working on the cuBLAS pilot.
    find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h PATHS /usr/include /usr/include/x86_64-linux-gnu
                                                /usr/local/cuda/include "$ENV{CUDNN_ROOT}" "$ENV{CUDNN_ROOT}/include")
    find_library(CUDNN_LIBRARY NAMES cudnn PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
                                               "$ENV{CUDNN_ROOT}" "$ENV{CUDNN_ROOT}/lib64")
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        target_sources(nert PRIVATE src/cuda/conv.cu)
        target_compile_definitions(nert PRIVATE NERT_HAS_CUDNN)
        target_include_directories(nert PRIVATE ${CUDNN_INCLUDE_DIR})
        target_link_libraries(nert PRIVATE ${CUDNN_LIBRARY})
    else()
        message(STATUS "cuDNN not found - CUDA Conv disabled (cuBLAS pilot only)")
    endif()
endif()

# OpenCL device layer (Phase N1d), opt-in like CUDA: NERT_ENABLE_OPENCL=ON
# requires the OpenCL headers + ICD loader and links OpenCL into `nert`;
# OFF (default) leaves the link closure standard-library-only. Independent
# of NERT_ENABLE_CUDA; both may be on at once (distinct devices).
option(NERT_ENABLE_OPENCL "Build NERT with the OpenCL device layer" OFF)
if(NERT_ENABLE_OPENCL)
    find_package(OpenCL)
    if(NOT OpenCL_FOUND)
        message(FATAL_ERROR
            "NERT_ENABLE_OPENCL=ON requires the OpenCL headers and ICD loader; none were found.")
    endif()
    target_sources(nert PRIVATE src/opencl_device.cpp)
    target_sources(nert PRIVATE src/opencl/runtime.cpp src/opencl/relu.cpp src/opencl/add.cpp src/opencl/table.cpp)
    target_sources(nert PRIVATE src/opencl/mul.cpp src/opencl/sub.cpp src/opencl/div.cpp)
    # PRIVATE: no OpenCL type leaks through the public headers, so consumers
    # stay toolkit-free.
    target_link_libraries(nert PRIVATE OpenCL::OpenCL)
endif()

# HIP device layer (Phase N1c, NVIDIA target), opt-in like CUDA/OpenCL:
# NERT_ENABLE_HIP=ON requires a CUDA compiler (the NVIDIA backend drives
# nvcc) plus the HIP headers (HIP_INCLUDE_DIR or $HIP_PATH, e.g. from the
# hip-dev package); OFF (default) leaves the link closure
# standard-library-only. Independent of the other device flags. The AMD
# half (hipcc/clang, rocBLAS/MIOpen) is a later slice on the AMD runner.
option(NERT_ENABLE_HIP "Build NERT with the HIP device layer (NVIDIA target)" OFF)
if(NERT_ENABLE_HIP)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "NERT_ENABLE_HIP=ON requires a CUDA compiler (nvcc) for the NVIDIA target; none was found.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h
        HINTS $ENV{HIP_PATH} /opt/rocm
        PATH_SUFFIXES include
        DOC "HIP headers (hip-dev package or HIP_PATH)")
    if(NOT HIP_INCLUDE_DIR)
        message(FATAL_ERROR
            "NERT_ENABLE_HIP=ON requires the HIP headers (set HIP_INCLUDE_DIR or HIP_PATH); none were found.")
    endif()
    target_sources(nert PRIVATE src/hip_device.cpp)
    # The HIP dialect compiles with nvcc on the NVIDIA target (what
    # hipcc -platform=nvidia does); the backend selection + headers stay
    # file-scoped so the rest of `nert` is untouched.
    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/hip_device.cpp PROPERTY LANGUAGE CUDA)
    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/hip_device.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        __HIP_PLATFORM_NVIDIA__)
    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/hip_device.cpp APPEND PROPERTY INCLUDE_DIRECTORIES
        ${HIP_INCLUDE_DIR})
    # PRIVATE: no HIP type leaks through the public headers, so consumers
    # stay toolkit-free.
    target_link_libraries(nert PRIVATE CUDA::cudart)
endif()

# Tests. BUILD_INFERENCE_ENGINE_TESTS is honoured for parent projects
# (Neuriplo) that set it.
if(NERT_BUILD_TESTS OR BUILD_INFERENCE_ENGINE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Install and package config.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NERT_IS_TOP_LEVEL)
    install(TARGETS nert
        EXPORT nertTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY include/nert DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT nertTargets
        NAMESPACE nert::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nert)

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/nertConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/nertConfig.cmake"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/nertTargets.cmake\")
")
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/nertConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/nertConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nert)
endif()

# Bench harness (opt-in tooling under bench/, like tools/). OFF by default so
# the runtime link closure and the default configure/build/test stay unchanged.
# Exactly one device backend may be selected (default: CUDA); see
# bench/README.md for the flag map.
option(NERT_BUILD_BENCH "Build the standalone kernel benchmark harness" OFF)
option(NERT_BENCH_USE_OPENCL "Bench: use the OpenCL C API backend" OFF)
option(NERT_BENCH_USE_OPENCL_CPP "Bench: use the OpenCL C++ wrapper backend" OFF)
if(NERT_BENCH_USE_OPENCL AND NERT_BENCH_USE_OPENCL_CPP)
    message(FATAL_ERROR
        "NERT_BENCH_USE_OPENCL and NERT_BENCH_USE_OPENCL_CPP are mutually "
        "exclusive; enable at most one (default is CUDA).")
endif()
if(NERT_BUILD_BENCH)
    add_subdirectory(bench)
endif()
