# MatMul bench kernel. Same pattern as bench/gemm: CPU oracle + 3-way
# backend switch (CUDA default, OpenCL C, OpenCL C++). Built whenever bench
# is ON; see bench/gemm/CMakeLists.txt for the shared conventions.

set(NERT_BENCH_MATMUL_SOURCES main.cpp cpu/matmul.cpp)

if(NERT_BENCH_USE_OPENCL_CPP)
    list(APPEND NERT_BENCH_MATMUL_SOURCES opencl_cpp/matmul.cpp)
    set(NERT_BENCH_MATMUL_BACKEND_DEF NERT_BENCH_OPENCL_CPP_BACKEND)
elseif(NERT_BENCH_USE_OPENCL)
    list(APPEND NERT_BENCH_MATMUL_SOURCES opencl/matmul.cpp)
    set(NERT_BENCH_MATMUL_BACKEND_DEF NERT_BENCH_OPENCL_BACKEND)
else()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "NERT_BUILD_BENCH CUDA path needs nvcc (no CUDA compiler found). Use an OpenCL flag or install the CUDA toolkit.")
    endif()
    enable_language(CUDA)
    set_source_files_properties(cuda/matmul.cpp PROPERTIES LANGUAGE CUDA)
    list(APPEND NERT_BENCH_MATMUL_SOURCES cuda/matmul.cpp)
endif()

add_executable(nert_bench_matmul ${NERT_BENCH_MATMUL_SOURCES})
target_compile_features(nert_bench_matmul PRIVATE cxx_std_17)
target_compile_options(nert_bench_matmul PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(nert_bench_matmul PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(nert_bench_matmul PRIVATE nert_bench_cuda_helpers)
if(NERT_BENCH_USE_OPENCL)
    target_link_libraries(nert_bench_matmul PRIVATE nert_bench_opencl_helpers)
elseif(NERT_BENCH_USE_OPENCL_CPP)
    target_link_libraries(nert_bench_matmul PRIVATE nert_bench_opencl_cpp_helpers)
endif()
if(DEFINED NERT_BENCH_MATMUL_BACKEND_DEF)
    target_compile_definitions(nert_bench_matmul PRIVATE ${NERT_BENCH_MATMUL_BACKEND_DEF})
endif()

add_test(NAME bench_matmul COMMAND nert_bench_matmul)
set_tests_properties(bench_matmul PROPERTIES TIMEOUT 1200)
