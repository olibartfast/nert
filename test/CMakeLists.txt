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
    "Path to the opset-18 ResNet-18 ONNX fixture for the NERT tests")
if(NOT NERT_FIXTURE AND EXISTS
    "/tmp/opencode/resnet18_export/resnet18.onnx")
    set(NERT_FIXTURE
        "/tmp/opencode/resnet18_export/resnet18.onnx")
endif()

add_test(NAME nert_loader
    COMMAND nert_loader_test --gtest_filter=NertLoader.*)
add_test(NAME nert_loader_negative
    COMMAND nert_loader_test --gtest_filter=NertLoaderNegative.*)

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
    # cuDNN presence (N1b Conv row) is visible to the CUDA tests so they can
    # assert precisely instead of guessing.
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        target_compile_definitions(nert_cuda_device_test PRIVATE NERT_HAS_CUDNN)
        target_compile_definitions(nert_cuda_kernels_test PRIVATE NERT_HAS_CUDNN)
        target_compile_definitions(nert_cuda_conv_test PRIVATE NERT_HAS_CUDNN)
    endif()
endif()
