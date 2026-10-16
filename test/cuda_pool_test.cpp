// CUDA pooling (N1e-2): MaxPool/AveragePool vs the CPU oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

bool HasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count >= 1;
}

nert::TensorView HostFloat(std::vector<float>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

void* Upload(nert::Device& cuda, const std::vector<float>& host) {
    void* device = cuda.allocator().allocate(static_cast<int64_t>(host.size() * sizeof(float)));
    if (device == nullptr) {
        throw std::runtime_error("Upload: device allocate failed");
    }
    cuda.transfer().host_to_device(device, host.data(), static_cast<int64_t>(host.size() * sizeof(float)));
    return device;
}

std::vector<float> Download(nert::Device& cuda, void* device, std::size_t count) {
    std::vector<float> host(count, 0.0F);
    cuda.transfer().device_to_host(host.data(), device, static_cast<int64_t>(count * sizeof(float)));
    return host;
}

nert::TensorView DeviceFloat(void* device, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = device;
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

nert::Node MakeNode(const std::string& name, const std::string& op) {
    nert::Node node;
    node.name = name;
    node.op_type = op;
    return node;
}

std::vector<float> RunCudaPool(nert::Device& cuda, const std::string& op, nert::Node node, std::vector<float>& x,
                               std::vector<int64_t> xd, std::vector<int64_t> yd) {
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    std::vector<float> out(static_cast<std::size_t>(yd[0] * yd[1] * yd[2] * yd[3]), 0.0F);
    void* dy = Upload(cuda, out);
    nert::CudaDevice().kernels().find(op)(node, {DeviceFloat(dx, xd)}, {DeviceFloat(dy, yd)});
    out = Download(cuda, dy, out.size());
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    return out;
}

std::vector<float> RunCpuPool(const std::string& op, nert::Node node, std::vector<float>& x, std::vector<int64_t> xd,
                              std::vector<int64_t> yd) {
    node.inputs = {"x"};
    node.outputs = {"y"};
    std::vector<float> out(static_cast<std::size_t>(yd[0] * yd[1] * yd[2] * yd[3]), 0.0F);
    nert::CpuDevice().kernels().find(op)(node, {HostFloat(x, xd)}, {HostFloat(out, yd)});
    return out;
}

} // namespace

// MaxPool 2x2 stride 2 over 1..16: [[6,8],[14,16]] on both tables.
TEST(NertCudaPool, MaxPoolStride2Exact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                            9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
    nert::Node node = MakeNode("mp0", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    const std::vector<float> expected = {6.0F, 8.0F, 14.0F, 16.0F};
    EXPECT_EQ(RunCpuPool("MaxPool", node, x, {1, 1, 4, 4}, {1, 1, 2, 2}), expected);
    EXPECT_EQ(RunCudaPool(cuda, "MaxPool", node, x, {1, 1, 4, 4}, {1, 1, 2, 2}), expected);
}

// MaxPool 3x3 stride 1 pad 1 over 1..9: out-of-bounds samples -inf, so the
// corners see partial windows: [5,6,6,8,9,9,8,9,9].
TEST(NertCudaPool, MaxPoolPaddedExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    nert::Node node = MakeNode("mp1", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{3, 3});
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    const std::vector<float> expected = {5.0F, 6.0F, 6.0F, 8.0F, 9.0F, 9.0F, 8.0F, 9.0F, 9.0F};
    EXPECT_EQ(RunCpuPool("MaxPool", node, x, {1, 1, 3, 3}, {1, 1, 3, 3}), expected);
    EXPECT_EQ(RunCudaPool(cuda, "MaxPool", node, x, {1, 1, 3, 3}, {1, 1, 3, 3}), expected);
}

// MaxPool 2x2 dilation 2 over 1..16: strided samples.
// Window (0,0) sees (0,0),(0,2),(2,0),(2,2) -> 11; full -> [11,12,15,16].
TEST(NertCudaPool, MaxPoolDilatedExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                            9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
    nert::Node node = MakeNode("mp2", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("dilations", std::vector<int64_t>{2, 2});
    const std::vector<float> expected = {11.0F, 12.0F, 15.0F, 16.0F};
    EXPECT_EQ(RunCpuPool("MaxPool", node, x, {1, 1, 4, 4}, {1, 1, 2, 2}), expected);
    EXPECT_EQ(RunCudaPool(cuda, "MaxPool", node, x, {1, 1, 4, 4}, {1, 1, 2, 2}), expected);
}

// AveragePool 2x2 stride 2 over 1..16: [[3.5,5.5],[11.5,13.5]].
TEST(NertCudaPool, AveragePoolStride2Exact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                            9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
    nert::Node node = MakeNode("ap0", "AveragePool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    const std::vector<float> expected = {3.5F, 5.5F, 11.5F, 13.5F};
    EXPECT_EQ(RunCpuPool("AveragePool", node, x, {1, 1, 4, 4}, {1, 1, 2, 2}), expected);
    EXPECT_EQ(RunCudaPool(cuda, "AveragePool", node, x, {1, 1, 4, 4}, {1, 1, 2, 2}), expected);
}

// AveragePool count_include_pad over a bottom/right-padded 3x3: padded cells
// join the divisor -> [3,2.25,3.75,2.25]; without -> [3,4.5,7.5,9].
TEST(NertCudaPool, AveragePoolCountIncludePad) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    nert::Node node = MakeNode("ap1", "AveragePool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{0, 0, 1, 1});
    node.attributes.emplace_back("count_include_pad", int64_t{1});
    const std::vector<float> expected = {3.0F, 2.25F, 3.75F, 2.25F};
    EXPECT_EQ(RunCpuPool("AveragePool", node, x, {1, 1, 3, 3}, {1, 1, 2, 2}), expected);
    EXPECT_EQ(RunCudaPool(cuda, "AveragePool", node, x, {1, 1, 3, 3}, {1, 1, 2, 2}), expected);

    nert::Node plain = MakeNode("ap2", "AveragePool");
    plain.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    plain.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    plain.attributes.emplace_back("pads", std::vector<int64_t>{0, 0, 1, 1});
    const std::vector<float> expectedPlain = {3.0F, 4.5F, 7.5F, 9.0F};
    EXPECT_EQ(RunCpuPool("AveragePool", plain, x, {1, 1, 3, 3}, {1, 1, 2, 2}), expectedPlain);
    EXPECT_EQ(RunCudaPool(cuda, "AveragePool", plain, x, {1, 1, 3, 3}, {1, 1, 2, 2}), expectedPlain);
}

// MaxPool rejects a missing kernel_shape, naming node + op.
TEST(NertCudaPool, MaxPoolRejectsMissingKernelShape) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(16, 1.0F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("mp_bad", "MaxPool");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {1, 1, 4, 4});
    nert::TensorView yi = DeviceFloat(dy, {1, 1, 2, 2});
    try {
        nert::CudaDevice().kernels().find("MaxPool")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for a missing kernel_shape";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mp_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("MaxPool"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// AveragePool rejects a rank-3 input, naming node + op.
TEST(NertCudaPool, AveragePoolRejectsRank3) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(12, 1.0F);
    std::vector<float> y(12, 0.0F);
    nert::Node node = MakeNode("ap_bad", "AveragePool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {1, 3, 4});
    nert::TensorView yi = DeviceFloat(dy, {1, 3, 4});
    try {
        nert::CudaDevice().kernels().find("AveragePool")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for a rank-3 input";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ap_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("AveragePool"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}
