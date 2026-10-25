#include "device_test_helpers.hpp"
#include "nert/shapes.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace nert_test;
const nert::Device& Device() {
#if defined(TEST_CPU)
    return nert::CpuDevice();
#elif defined(TEST_CUDA)
    return nert::CudaDevice();
#elif defined(TEST_OPENCL)
    return nert::OpenCLDevice();
#else
    return nert::HipDevice();
#endif
}

TEST(ReleaseKernels, SignSpecialValuesAndIntegerExtremes) {
    ASSERT_TRUE(DeviceAvailable(&Device));
    auto& dev = const_cast<nert::Device&>(Device());
    auto kernel = dev.kernels().find("Sign");
    ASSERT_NE(kernel, nullptr);
    auto node = MakeNode("sign", "Sign");
    std::vector<float> x{
        -std::numeric_limits<float>::infinity(), -2, -0.0f, 0, 3, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    std::vector<float> y(7);
    void* dx = Upload(dev, x);
    void* dy = Upload(dev, y);
    kernel(node, {DeviceFloat(dx, {7})}, {DeviceFloat(dy, {7})});
    y = Download(dev, dy, 7);
    EXPECT_EQ(std::vector<float>(y.begin(), y.begin() + 6), (std::vector<float>{-1, -1, 0, 0, 1, 1}));
    EXPECT_TRUE(std::isnan(y[6]));
    EXPECT_THROW(kernel(node, {DeviceFloat(dx, {7})}, {DeviceFloat(dy, {1, 7})}), nert::InferenceException);
    dev.allocator().release(dx);
    dev.allocator().release(dy);
    std::vector<int64_t> ix{INT64_MIN, -1, 0, 1, INT64_MAX}, iy(5);
    dx = UploadInt64(dev, ix);
    dy = UploadInt64(dev, iy);
    kernel(node, {DeviceInt64(dx, {5})}, {DeviceInt64(dy, {5})});
    EXPECT_EQ(DownloadInt64(dev, dy, 5), (std::vector<int64_t>{-1, -1, 0, 1, 1}));
    dev.allocator().release(dx);
    dev.allocator().release(dy);
}

TEST(ReleaseKernels, SplitNumOutputsUneven) {
    ASSERT_TRUE(DeviceAvailable(&Device));
    auto& dev = const_cast<nert::Device&>(Device());
    auto kernel = dev.kernels().find("Split");
    ASSERT_NE(kernel, nullptr);
    auto node = MakeNode("split18", "Split");
    node.schema_version = 18;
    node.attributes = {nert::Attribute{"axis", int64_t{-1}}, nert::Attribute{"num_outputs", int64_t{3}}};
    std::vector<int64_t> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, a(4), b(4), c(2);
    void* dx = UploadInt64(dev, x);
    void* da = UploadInt64(dev, a);
    void* db = UploadInt64(dev, b);
    void* dc = UploadInt64(dev, c);
    kernel(node, {DeviceInt64(dx, {2, 5})},
           {DeviceInt64(da, {2, 2}), DeviceInt64(db, {2, 2}), DeviceInt64(dc, {2, 1})});
    EXPECT_EQ(DownloadInt64(dev, da, 4), (std::vector<int64_t>{0, 1, 5, 6}));
    EXPECT_EQ(DownloadInt64(dev, db, 4), (std::vector<int64_t>{2, 3, 7, 8}));
    EXPECT_EQ(DownloadInt64(dev, dc, 2), (std::vector<int64_t>{4, 9}));
    dev.allocator().release(dx);
    dev.allocator().release(da);
    dev.allocator().release(db);
    dev.allocator().release(dc);
}

TEST(ReleaseKernels, SplitRejectsNonPositiveFinalChunkInShapesAndDevice) {
    ASSERT_TRUE(DeviceAvailable(&Device));
    auto& dev = const_cast<nert::Device&>(Device());
    auto kernel = dev.kernels().find("Split");
    ASSERT_NE(kernel, nullptr);
    for (const auto& form : std::vector<std::pair<int64_t, int64_t>>{{2, 3}, {6, 4}}) {
        const auto dim = form.first;
        const auto count = form.second;
        auto node = MakeNode("non_positive_last", "Split");
        node.schema_version = 18;
        node.inputs = {"x"};
        for (int64_t i = 0; i < count; ++i)
            node.outputs.push_back("out" + std::to_string(i));
        node.attributes = {nert::Attribute{"num_outputs", count}};
        nert::Graph graph;
        graph.inputs.push_back({"x", {nert::DataType::Int64, {dim}}});
        graph.nodes.push_back(node);
        graph.outputs.push_back({"out0", {nert::DataType::Int64, {1}}});
        EXPECT_THROW(nert::InferShapes(graph, {{"x", {dim}}}), nert::ModelLoadException);
        std::vector<int64_t> x(static_cast<size_t>(dim), 1), y(2);
        void* dx = UploadInt64(dev, x);
        void* dy = UploadInt64(dev, y);
        std::vector<nert::TensorView> outputs(static_cast<size_t>(count), DeviceInt64(dy, {dim / count + 1}));
        outputs.back().dims = {0};
        EXPECT_THROW(kernel(node, {DeviceInt64(dx, {dim})}, outputs), nert::InferenceException);
        node.attributes = {nert::Attribute{"num_outputs", int64_t{0}}};
        EXPECT_THROW(kernel(node, {DeviceInt64(dx, {dim})}, {DeviceInt64(dy, {1})}), nert::InferenceException);
        dev.allocator().release(dx);
        dev.allocator().release(dy);
    }
}
} // namespace
