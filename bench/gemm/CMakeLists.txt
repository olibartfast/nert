# GEMM bench pilot. G2 builds the CPU oracle + harness with the CUDA-path
# fallback TU (gemmGPU throws -> SKIP); G3 swaps in the real CUDA
# implementation, G4 the OpenCL C / C++ backends. main.cpp already carries
# the 3-way backend include switch.

set(NERT_BENCH_GEMM_SOURCES main.cpp cpu/gemm.cpp)

if(NERT_BENCH_USE_OPENCL_CPP)
    list(APPEND NERT_BENCH_GEMM_SOURCES opencl_cpp/gemm.cpp)
    set(NERT_BENCH_GEMM_BACKEND_DEF NERT_BENCH_OPENCL_CPP_BACKEND)
elseif(NERT_BENCH_USE_OPENCL)
    list(APPEND NERT_BENCH_GEMM_SOURCES opencl/gemm.cpp)
    set(NERT_BENCH_GEMM_BACKEND_DEF NERT_BENCH_OPENCL_BACKEND)
else()
    list(APPEND NERT_BENCH_GEMM_SOURCES cuda/gemm.cpp)
endif()

add_executable(nert_bench_gemm ${NERT_BENCH_GEMM_SOURCES})
target_compile_features(nert_bench_gemm PRIVATE cxx_std_17)
target_compile_options(nert_bench_gemm PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(nert_bench_gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nert_bench_gemm PRIVATE nert_bench_cuda_helpers)
if(DEFINED NERT_BENCH_GEMM_BACKEND_DEF)
    target_compile_definitions(nert_bench_gemm PRIVATE ${NERT_BENCH_GEMM_BACKEND_DEF})
endif()

add_test(NAME bench_gemm COMMAND nert_bench_gemm)
set_tests_properties(bench_gemm PROPERTIES TIMEOUT 1200)
