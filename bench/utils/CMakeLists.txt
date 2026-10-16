# Header-only bench utilities, vendored (with attribution, see ATTRIBUTION.md)
# from the gpu-playground reference. Bench is opt-in (NERT_BUILD_BENCH=ON),
# so requiring the device SDKs here never affects the default build: the
# runtime `nert` target stays standard-library-only. When bench is enabled,
# both the CUDA toolkit and the OpenCL SDK are required at configure time
# (all helper headers are compile-checked together); which device actually
# runs is still chosen per backend flag, and every device half runtime-SKIPs
# without a GPU.

find_package(CUDAToolkit REQUIRED)
find_package(OpenCL REQUIRED)

add_library(nert_bench_cuda_helpers INTERFACE)
target_include_directories(nert_bench_cuda_helpers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nert_bench_cuda_helpers INTERFACE CUDA::cudart)

add_library(nert_bench_opencl_helpers INTERFACE)
target_include_directories(nert_bench_opencl_helpers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nert_bench_opencl_helpers INTERFACE OpenCL::OpenCL)

add_library(nert_bench_opencl_cpp_helpers INTERFACE)
target_include_directories(nert_bench_opencl_cpp_helpers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nert_bench_opencl_cpp_helpers INTERFACE OpenCL::OpenCL)
if(TARGET OpenCL::HeadersCpp)
    target_link_libraries(nert_bench_opencl_cpp_helpers INTERFACE OpenCL::HeadersCpp)
endif()

# Compile-time check that the vendored headers are self-contained and that the
# core statistics/unit converters agree on their documented values. Links all
# three helper targets so every vendored header is included; no device is
# touched, so this passes on CPU-only runners.
add_executable(nert_bench_headers_check test/headers_compile.cpp)
target_compile_features(nert_bench_headers_check PRIVATE cxx_std_17)
target_compile_options(nert_bench_headers_check PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_bench_headers_check PRIVATE nert_bench_cuda_helpers nert_bench_opencl_helpers
                                                       nert_bench_opencl_cpp_helpers)

add_test(NAME bench_utils_headers COMMAND nert_bench_headers_check)
