# Runtime-backed bench (N1f): parity + timing for the shipped kernels.
#
# Unlike bench/gemm + bench/matmul (vendored kernel copies), this driver
# implements no kernel logic: it links `nert` and drives the runtime's own
# CpuDevice/CudaDevice tables. Built only when the CUDA device layer is on
# (OFF builds carry no CudaDevice symbol); without a GPU at run time the
# device halves SKIP and the CPU halves still run.

if(NOT NERT_ENABLE_CUDA)
    return()
endif()

add_executable(nert_bench_runtime main.cpp)
target_compile_features(nert_bench_runtime PRIVATE cxx_std_17)
target_compile_options(nert_bench_runtime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_bench_runtime PRIVATE nert)

add_test(NAME bench_runtime COMMAND nert_bench_runtime)
set_tests_properties(bench_runtime PROPERTIES TIMEOUT 1200)
