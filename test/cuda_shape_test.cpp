// CUDA reshape family (N1g-1): Reshape/Unsqueeze/Flatten/Expand vs the CPU
// oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails). Shape
// and axes operands ride on device like any staged tensor.

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

std::vector<float> Iota(std::size_t count) {
    std::vector<float> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = static_cast<float>(i) + 0.5F;
    }
    return data;
}

} // namespace

// Reshape {2,3,4} via shape operand {-1,6} -> {4,6}, identical on both.
TEST(NertCudaShape, ReshapeMinusOne) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = Iota(24);
    std::vector<int64_t> shape = {-1, 6};
    std::vector<float> cpuOut(24, 0.0F);
    nert::Node node = MakeNode("rs0", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Reshape")(node, {HostFloat(x, {2, 3, 4}), HostInt64(shape, {2})},
                                                {HostFloat(cpuOut, {4, 6})});
    EXPECT_EQ(cpuOut, x);

    void* dx = Upload(cuda, x);
    void* ds = UploadInt64(cuda, shape);
    std::vector<float> gpuOut(24, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Reshape")(node, {DeviceFloat(dx, {2, 3, 4}), DeviceInt64(ds, {2})},
                                                 {DeviceFloat(dy, {4, 6})});
    gpuOut = Download(cuda, dy, 24);
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Reshape allowzero=0 copies the matching input dim: {2,3} via {0,-1}.
TEST(NertCudaShape, ReshapeAllowZeroInt64) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> shape = {0, -1};
    std::vector<int64_t> cpuOut(6, 0);
    nert::Node node = MakeNode("rs1", "Reshape");
    node.attributes.emplace_back("allowzero", int64_t{0});
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Reshape")(node, {HostInt64(x, {2, 3}), HostInt64(shape, {2})},
                                                {HostInt64(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, x);

    void* dx = UploadInt64(cuda, x);
    void* ds = UploadInt64(cuda, shape);
    std::vector<int64_t> gpuOut(6, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Reshape")(node, {DeviceInt64(dx, {2, 3}), DeviceInt64(ds, {2})},
                                                 {DeviceInt64(dy, {2, 3})});
    gpuOut = DownloadInt64(cuda, dy, 6);
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Unsqueeze {2,3} with axes {-1,0} -> {1,2,3,1} on both tables.
TEST(NertCudaShape, UnsqueezeNegativeAxes) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = Iota(6);
    std::vector<int64_t> axes = {-1, 0};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("us0", "Unsqueeze");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Unsqueeze")(node, {HostFloat(x, {2, 3}), HostInt64(axes, {2})},
                                                  {HostFloat(cpuOut, {1, 2, 3, 1})});
    EXPECT_EQ(cpuOut, x);

    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Unsqueeze")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {2})},
                                                   {DeviceFloat(dy, {1, 2, 3, 1})});
    gpuOut = Download(cuda, dy, 6);
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Flatten axis 0 ({2,3,4} -> {1,24}) and axis 2 (-> {6,4}), both tables.
TEST(NertCudaShape, FlattenAxes) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    for (int64_t axis : {0, 2}) {
        std::vector<float> x = Iota(24);
        const std::vector<int64_t> yd = axis == 0 ? std::vector<int64_t>{1, 24} : std::vector<int64_t>{6, 4};
        std::vector<float> cpuOut(24, 0.0F);
        nert::Node node = MakeNode("fl0", "Flatten");
        node.attributes.emplace_back("axis", axis);
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("Flatten")(node, {HostFloat(x, {2, 3, 4})}, {HostFloat(cpuOut, yd)});
        EXPECT_EQ(cpuOut, x);

        void* dx = Upload(cuda, x);
        std::vector<float> gpuOut(24, 0.0F);
        void* dy = Upload(cuda, gpuOut);
        nert::CudaDevice().kernels().find("Flatten")(node, {DeviceFloat(dx, {2, 3, 4})}, {DeviceFloat(dy, yd)});
        gpuOut = Download(cuda, dy, 24);
        cuda.allocator().release(dx);
        cuda.allocator().release(dy);
        EXPECT_EQ(gpuOut, cpuOut);
    }
}

// Expand {1,3} -> {2,3} float and {3} -> {2,3} int64, both tables.
TEST(NertCudaShape, ExpandBroadcast) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F};
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("ex0", "Expand");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Expand")(node, {HostFloat(x, {1, 3}), HostInt64(shape, {2})},
                                               {HostFloat(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 2.0F, 3.0F, 1.0F, 2.0F, 3.0F}));

    void* dx = Upload(cuda, x);
    void* ds = UploadInt64(cuda, shape);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Expand")(node, {DeviceFloat(dx, {1, 3}), DeviceInt64(ds, {2})},
                                                {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(cuda, dy, 6);
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);

    std::vector<int64_t> xi = {7, 8, 9};
    std::vector<int64_t> cpuOutI(6, 0);
    nert::Node nodeI = MakeNode("ex1", "Expand");
    nodeI.inputs = {"x", "shape"};
    nodeI.outputs = {"y"};
    nert::CpuDevice().kernels().find("Expand")(nodeI, {HostInt64(xi, {3}), HostInt64(shape, {2})},
                                               {HostInt64(cpuOutI, {2, 3})});
    EXPECT_EQ(cpuOutI, (std::vector<int64_t>{7, 8, 9, 7, 8, 9}));

    void* dxi = UploadInt64(cuda, xi);
    void* dsi = UploadInt64(cuda, shape);
    std::vector<int64_t> gpuOutI(6, 0);
    void* dyi = UploadInt64(cuda, gpuOutI);
    nert::CudaDevice().kernels().find("Expand")(nodeI, {DeviceInt64(dxi, {3}), DeviceInt64(dsi, {2})},
                                                {DeviceInt64(dyi, {2, 3})});
    gpuOutI = DownloadInt64(cuda, dyi, 6);
    cuda.allocator().release(dxi);
    cuda.allocator().release(dsi);
    cuda.allocator().release(dyi);
    EXPECT_EQ(gpuOutI, cpuOutI);
}

// Reshape rejects a target product mismatch, naming node + op.
TEST(NertCudaShape, ReshapeRejectsBadProduct) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = Iota(24);
    std::vector<int64_t> shape = {5, 5};
    std::vector<float> y(25, 0.0F);
    nert::Node node = MakeNode("rs_bad", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* ds = UploadInt64(cuda, shape);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {2, 3, 4});
    nert::TensorView si = DeviceInt64(ds, {2});
    nert::TensorView yi = DeviceFloat(dy, {5, 5});
    try {
        nert::CudaDevice().kernels().find("Reshape")(node, {xi, si}, {yi});
        FAIL() << "expected InferenceException for a target product mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("rs_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Reshape"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
}

// Unsqueeze rejects duplicate axes, naming node + op.
TEST(NertCudaShape, UnsqueezeRejectsDuplicateAxes) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = Iota(6);
    std::vector<int64_t> axes = {0, 0};
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("us_bad", "Unsqueeze");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Unsqueeze")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {2})},
                                                       {DeviceFloat(dy, {1, 1, 2, 3})});
        FAIL() << "expected InferenceException for duplicate axes";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("us_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Unsqueeze"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
}

// Flatten rejects an out-of-range axis, naming node + op.
TEST(NertCudaShape, FlattenRejectsBadAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = Iota(24);
    std::vector<float> y(24, 0.0F);
    nert::Node node = MakeNode("fl_bad", "Flatten");
    node.attributes.emplace_back("axis", int64_t{5});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Flatten")(node, {DeviceFloat(dx, {2, 3, 4})}, {DeviceFloat(dy, {2, 12})});
        FAIL() << "expected InferenceException for an out-of-range axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("fl_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Flatten"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Expand rejects a non-broadcastable target, naming node + op.
TEST(NertCudaShape, ExpandRejectsNonBroadcastable) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F};
    std::vector<int64_t> shape = {3};
    std::vector<float> y(3, 0.0F);
    nert::Node node = MakeNode("ex_bad", "Expand");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* ds = UploadInt64(cuda, shape);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Expand")(node, {DeviceFloat(dx, {2}), DeviceInt64(ds, {1})},
                                                    {DeviceFloat(dy, {3})});
        FAIL() << "expected InferenceException for a non-broadcastable target";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ex_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Expand"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
}
