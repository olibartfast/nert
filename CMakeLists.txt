# NERT (Neuriplo Engine Runtime): first-party ONNX inference runtime (CPU
# reference spine). Depends on nothing beyond the C++ standard library.

cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(NERT_IS_TOP_LEVEL ON)
else()
    set(NERT_IS_TOP_LEVEL OFF)
endif()

project(nert VERSION 0.1.0 LANGUAGES CXX)

if(NERT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

option(NERT_BUILD_TESTS "Build the NERT tests" ${NERT_IS_TOP_LEVEL})

add_library(nert STATIC
    src/Engine.cpp
    src/WireReader.cpp
    src/ModelLoader.cpp
    src/ShapeInference.cpp
    src/MemoryPlanner.cpp
    src/CpuDevice.cpp
    src/Executor.cpp
    src/ConstantFolding.cpp
    src/kernels/CpuKernels.cpp
    src/kernels/Relu.cpp
    src/kernels/Add.cpp
    src/kernels/Reshape.cpp
    src/kernels/ReduceMean.cpp
    src/kernels/Gemm.cpp
    src/kernels/MatMul.cpp
    src/kernels/Conv.cpp
    src/kernels/MaxPool.cpp
    src/kernels/Mul.cpp
    src/kernels/Div.cpp
    src/kernels/Sub.cpp
    src/kernels/Sigmoid.cpp
    src/kernels/Cast.cpp
    src/kernels/Softmax.cpp
    src/kernels/ReduceMax.cpp
    src/kernels/ReduceSum.cpp
    src/kernels/Range.cpp
    src/kernels/Pad.cpp
    src/kernels/Mod.cpp
    src/kernels/Equal.cpp
    src/kernels/Greater.cpp
    src/kernels/Less.cpp
    src/kernels/And.cpp
    src/kernels/Not.cpp
    src/kernels/Where.cpp
    src/kernels/Concat.cpp
    src/kernels/Split.cpp
    src/kernels/Unsqueeze.cpp
    src/kernels/Expand.cpp
    src/kernels/Transpose.cpp
    src/kernels/Slice.cpp
    src/kernels/Gather.cpp
    src/kernels/GatherElements.cpp
    src/kernels/Resize.cpp
    src/kernels/Flatten.cpp
    src/kernels/Shape.cpp
    src/kernels/ConstantOfShape.cpp
    src/kernels/TopK.cpp
    src/kernels/Floor.cpp
    src/kernels/Identity.cpp
    src/kernels/Pow.cpp
    src/kernels/Tile.cpp
    src/kernels/AveragePool.cpp
    src/kernels/GridSample.cpp
    src/kernels/Einsum.cpp
    src/kernels/ConvTranspose.cpp
    src/kernels/Erf.cpp
    src/kernels/Sqrt.cpp
    src/kernels/Tanh.cpp
    src/kernels/IsNaN.cpp
    src/kernels/Abs.cpp
    src/kernels/Neg.cpp
    src/kernels/Exp.cpp
    src/kernels/Log.cpp
    src/kernels/Cos.cpp
    src/kernels/Sin.cpp
    src/kernels/Clip.cpp
    src/kernels/Squeeze.cpp
    src/kernels/LayerNormalization.cpp)
target_compile_features(nert PUBLIC cxx_std_17)
target_include_directories(nert
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")

add_library(nert::nert ALIAS nert)
# Compatibility alias for consumers that still link the old target name.
add_library(neuriplo_engine ALIAS nert)

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
    install(DIRECTORY include/engine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
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
