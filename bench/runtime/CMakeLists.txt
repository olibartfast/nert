# Runtime-backed bench (N1f/N1d/N1c): parity + timing for the shipped
# kernels on every enabled device layer.
#
# This driver implements no kernel logic: it links `nert` and drives the
# runtime's own device tables. Built when at least one device layer is on
# (OFF builds carry no device symbols); a device that is compiled in but
# absent at run time prints SKIP and still exits 0.

if(NOT (NERT_ENABLE_CUDA OR NERT_ENABLE_OPENCL OR NERT_ENABLE_HIP))
    return()
endif()

add_executable(nert_bench_runtime main.cpp)
target_compile_features(nert_bench_runtime PRIVATE cxx_std_17)
target_compile_options(nert_bench_runtime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_bench_runtime PRIVATE nert)

# One definition per enabled backend; the driver enumerates exactly the
# devices this binary was built with (never a symbol that is absent).
if(NERT_ENABLE_CUDA)
    target_compile_definitions(nert_bench_runtime PRIVATE NERT_BENCH_CUDA)
endif()
if(NERT_ENABLE_OPENCL)
    target_compile_definitions(nert_bench_runtime PRIVATE NERT_BENCH_OPENCL)
endif()
if(NERT_ENABLE_HIP)
    target_compile_definitions(nert_bench_runtime PRIVATE NERT_BENCH_HIP)
endif()

add_test(NAME bench_runtime COMMAND nert_bench_runtime)
set_tests_properties(bench_runtime PROPERTIES TIMEOUT 1200)
