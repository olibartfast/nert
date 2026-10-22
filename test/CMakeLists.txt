# NERT tests. GoogleTest is scoped to this directory; the library itself
# links nothing but the standard library.

find_package(GTest REQUIRED)

add_executable(nert_loader_test loader_test.cpp)
target_compile_features(nert_loader_test PRIVATE cxx_std_17)
target_compile_options(nert_loader_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_loader_test PRIVATE
    nert
    GTest::gtest
    GTest::gtest_main)

# Fixture resolvable via env var or CMake cache variable of the same name.
# When unset, fall back to the well-known local export location if it exists;
# the tests themselves ASSERT (fail, never skip) when the fixture is absent.
set(NERT_FIXTURE "" CACHE PATH
    "Path to the provisioned ResNet-18 ONNX fixture for the NERT tests")
if(NOT NERT_FIXTURE AND EXISTS
    "/tmp/opencode/resnet18_op19.onnx")
    set(NERT_FIXTURE
        "/tmp/opencode/resnet18_op19.onnx")
endif()

add_test(NAME nert_loader
    COMMAND nert_loader_test --gtest_filter=EngineLoader.*)
add_test(NAME nert_loader_negative
    COMMAND nert_loader_test --gtest_filter=EngineLoaderNegative.*)

add_executable(nert_shapes_test shape_inference_test.cpp)
target_compile_features(nert_shapes_test PRIVATE cxx_std_17)
target_compile_options(nert_shapes_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_shapes_test PRIVATE
    nert
    GTest::gtest
    GTest::gtest_main)

add_test(NAME nert_shapes
    COMMAND nert_shapes_test --gtest_filter=NertShapes.*)
add_test(NAME nert_shapes_negative
    COMMAND nert_shapes_test --gtest_filter=NertShapesNegative.*)

add_executable(nert_plan_test memory_planner_test.cpp)
target_compile_features(nert_plan_test PRIVATE cxx_std_17)
target_compile_options(nert_plan_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_plan_test PRIVATE
    nert
    GTest::gtest
    GTest::gtest_main)

add_test(NAME nert_plan
    COMMAND nert_plan_test --gtest_filter=NertPlan.*)
add_test(NAME nert_plan_negative
    COMMAND nert_plan_test --gtest_filter=NertPlanNegative.*)

add_executable(nert_device_test device_test.cpp)
target_compile_features(nert_device_test PRIVATE cxx_std_17)
target_compile_options(nert_device_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_device_test PRIVATE
    nert
    GTest::gtest
    GTest::gtest_main)

add_test(NAME nert_device
    COMMAND nert_device_test --gtest_filter=NertDevice.*)

add_executable(nert_kernels_test kernels_test.cpp)
target_compile_features(nert_kernels_test PRIVATE cxx_std_17)
target_compile_options(nert_kernels_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_kernels_test PRIVATE
    nert
    GTest::gtest
    GTest::gtest_main)

add_test(NAME nert_kernels
    COMMAND nert_kernels_test --gtest_filter=NertKernels.*)
add_test(NAME nert_kernels_negative
    COMMAND nert_kernels_test --gtest_filter=NertKernelsNegative.*)

add_executable(nert_executor_test executor_test.cpp)
target_compile_features(nert_executor_test PRIVATE cxx_std_17)
target_compile_options(nert_executor_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_executor_test PRIVATE
    nert
    GTest::gtest
    GTest::gtest_main)

add_test(NAME nert_executor
    COMMAND nert_executor_test --gtest_filter=NertExecutor.*)
add_test(NAME nert_executor_negative
    COMMAND nert_executor_test --gtest_filter=NertExecutorNegative.*)

# The fixture end-to-end run may take seconds; give it room beyond the default.
set_tests_properties(nert_executor PROPERTIES TIMEOUT 1800)

if(NERT_FIXTURE)
    set_tests_properties(nert_loader nert_loader_negative
        nert_shapes nert_shapes_negative
        nert_plan nert_plan_negative
        nert_executor nert_executor_negative PROPERTIES
        ENVIRONMENT "NERT_FIXTURE=${NERT_FIXTURE}")
endif()

add_executable(nert_constant_folding_test constant_folding_test.cpp)
target_compile_features(nert_constant_folding_test PRIVATE cxx_std_17)
target_compile_options(nert_constant_folding_test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(nert_constant_folding_test PRIVATE
    nert
    GTest::gtest
    GTest::gtest_main)
add_test(NAME nert_constant_folding COMMAND nert_constant_folding_test)

# CUDA device scaffold test (N1a-M1). Built only when the CUDA device layer
# is enabled; SKIPs at run time without a GPU so toolkit-only runners stay
# green.
if(NERT_ENABLE_CUDA)
    add_executable(nert_cuda_device_test cuda_device_test.cpp)
    target_compile_features(nert_cuda_device_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_device_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_device_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_device COMMAND nert_cuda_device_test)
    add_executable(nert_cuda_kernels_test cuda_kernels_test.cpp)
    target_compile_features(nert_cuda_kernels_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_kernels_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_kernels_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_kernels COMMAND nert_cuda_kernels_test)
    add_executable(nert_cuda_conv_test cuda_conv_test.cpp)
    target_compile_features(nert_cuda_conv_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_conv_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_conv_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_conv COMMAND nert_cuda_conv_test)
    add_executable(nert_cuda_pool_test cuda_pool_test.cpp)
    target_compile_features(nert_cuda_pool_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_pool_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_pool_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_pool COMMAND nert_cuda_pool_test)
    add_executable(nert_cuda_shape_test cuda_shape_test.cpp)
    target_compile_features(nert_cuda_shape_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_shape_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_shape_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_shape COMMAND nert_cuda_shape_test)
    add_executable(nert_cuda_concat_test cuda_concat_test.cpp)
    target_compile_features(nert_cuda_concat_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_concat_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_concat_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_concat COMMAND nert_cuda_concat_test)
    add_executable(nert_cuda_index_test cuda_index_test.cpp)
    target_compile_features(nert_cuda_index_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_index_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_index_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_index COMMAND nert_cuda_index_test)
    add_executable(nert_cuda_misc_test cuda_misc_test.cpp)
    target_compile_features(nert_cuda_misc_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_misc_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_misc_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_misc COMMAND nert_cuda_misc_test)
    add_executable(nert_cuda_activations_test cuda_activations_test.cpp)
    target_compile_features(nert_cuda_activations_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_activations_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_activations_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_activations COMMAND nert_cuda_activations_test)
    add_executable(nert_cuda_norm_test cuda_norm_test.cpp)
    target_compile_features(nert_cuda_norm_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_norm_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_norm_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_norm COMMAND nert_cuda_norm_test)
    add_executable(nert_cuda_move_test cuda_move_test.cpp)
    target_compile_features(nert_cuda_move_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_move_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_move_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_move COMMAND nert_cuda_move_test)
    add_executable(nert_cuda_sample_test cuda_sample_test.cpp)
    target_compile_features(nert_cuda_sample_test PRIVATE cxx_std_17)
    target_compile_options(nert_cuda_sample_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_cuda_sample_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_cuda_sample COMMAND nert_cuda_sample_test)
    # cuDNN presence (N1b Conv row) is visible to the CUDA tests so they can
    # assert precisely instead of guessing.
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        target_compile_definitions(nert_cuda_device_test PRIVATE NERT_HAS_CUDNN)
        target_compile_definitions(nert_cuda_kernels_test PRIVATE NERT_HAS_CUDNN)
        target_compile_definitions(nert_cuda_conv_test PRIVATE NERT_HAS_CUDNN)
    endif()
endif()

# OpenCL device scaffold test (N1d-M1). Built only when the OpenCL device
# layer is enabled; SKIPs at run time without an OpenCL GPU platform.
if(NERT_ENABLE_OPENCL)
    add_executable(nert_opencl_device_test opencl_device_test.cpp)
    target_compile_features(nert_opencl_device_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_device_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_device_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_device COMMAND nert_opencl_device_test)
    add_executable(nert_opencl_kernels_test opencl_kernels_test.cpp)
    target_compile_features(nert_opencl_kernels_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_kernels_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_kernels_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_kernels COMMAND nert_opencl_kernels_test)
    add_executable(nert_opencl_mat_mul_test opencl_mat_mul_test.cpp)
    target_compile_features(nert_opencl_mat_mul_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_mat_mul_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_mat_mul_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_mat_mul COMMAND nert_opencl_mat_mul_test)
    add_executable(nert_opencl_layer_norm_test opencl_layer_norm_test.cpp)
    target_compile_features(nert_opencl_layer_norm_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_layer_norm_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_layer_norm_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_layer_norm COMMAND nert_opencl_layer_norm_test)
    add_executable(nert_opencl_expand_test opencl_expand_test.cpp)
    target_compile_features(nert_opencl_expand_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_expand_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_expand_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_expand COMMAND nert_opencl_expand_test)
    add_executable(nert_opencl_where_test opencl_where_test.cpp)
    target_compile_features(nert_opencl_where_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_where_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_where_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_where COMMAND nert_opencl_where_test)
    add_executable(nert_opencl_owlv2_rows_test opencl_owlv2_rows_test.cpp)
    target_compile_features(nert_opencl_owlv2_rows_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_owlv2_rows_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_owlv2_rows_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_owlv2_rows COMMAND nert_opencl_owlv2_rows_test)
    add_executable(nert_opencl_math_extra_test opencl_math_extra_test.cpp)
    target_compile_features(nert_opencl_math_extra_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_math_extra_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_math_extra_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_math_extra COMMAND nert_opencl_math_extra_test)
    add_executable(nert_opencl_reduce_extra_test opencl_reduce_extra_test.cpp)
    target_compile_features(nert_opencl_reduce_extra_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_reduce_extra_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_reduce_extra_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_reduce_extra COMMAND nert_opencl_reduce_extra_test)
    add_executable(nert_opencl_index_move_test opencl_index_move_test.cpp)
    target_compile_features(nert_opencl_index_move_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_index_move_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_index_move_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_index_move COMMAND nert_opencl_index_move_test)
    add_executable(nert_opencl_resize_test opencl_resize_test.cpp)
    target_compile_features(nert_opencl_resize_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_resize_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_resize_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_resize COMMAND nert_opencl_resize_test)
    add_executable(nert_opencl_grid_sample_test opencl_grid_sample_test.cpp)
    target_compile_features(nert_opencl_grid_sample_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_grid_sample_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_grid_sample_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_grid_sample COMMAND nert_opencl_grid_sample_test)
    add_executable(nert_opencl_norm_range_pad_test opencl_norm_range_pad_test.cpp)
    target_compile_features(nert_opencl_norm_range_pad_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_norm_range_pad_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_norm_range_pad_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_norm_range_pad COMMAND nert_opencl_norm_range_pad_test)
    add_executable(nert_opencl_math_extra2_test opencl_math_extra2_test.cpp)
    target_compile_features(nert_opencl_math_extra2_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_math_extra2_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_math_extra2_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_math_extra2 COMMAND nert_opencl_math_extra2_test)
    add_executable(nert_opencl_index_extra_test opencl_index_extra_test.cpp)
    target_compile_features(nert_opencl_index_extra_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_index_extra_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_index_extra_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_index_extra COMMAND nert_opencl_index_extra_test)
    add_executable(nert_opencl_conv_transpose_test opencl_conv_transpose_test.cpp)
    target_compile_features(nert_opencl_conv_transpose_test PRIVATE cxx_std_17)
    target_compile_options(nert_opencl_conv_transpose_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_opencl_conv_transpose_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_opencl_conv_transpose COMMAND nert_opencl_conv_transpose_test)
endif()

# Generated full-history schema catalog freshness. The generator is a
# development-only Python tool; add the check only when Python and the pinned
# ONNX package are available, so runtime-only builds do not acquire a Python
# dependency.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import onnx"
        RESULT_VARIABLE NERT_ONNX_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET)
    if(NERT_ONNX_IMPORT_RESULT EQUAL 0)
        add_test(NAME nert_opset_table
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_opset_compat.py"
                --check)
    endif()
endif()

# HIP device and first-party kernel tests (N1c on NVIDIA). Built only when the
# HIP device layer is enabled; SKIPs at run time without a HIP-usable GPU.
if(NERT_ENABLE_HIP)
    add_executable(nert_hip_device_test hip_device_test.cpp)
    target_compile_features(nert_hip_device_test PRIVATE cxx_std_17)
    target_compile_options(nert_hip_device_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_hip_device_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_hip_device COMMAND nert_hip_device_test)
    add_executable(nert_hip_kernels_test hip_kernels_test.cpp)
    target_compile_features(nert_hip_kernels_test PRIVATE cxx_std_17)
    target_compile_options(nert_hip_kernels_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(nert_hip_kernels_test PRIVATE
        nert
        GTest::gtest
        GTest::gtest_main)
    add_test(NAME nert_hip_kernels COMMAND nert_hip_kernels_test)
endif()
