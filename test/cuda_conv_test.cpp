// CUDA Conv via cuDNN (N1b): parity against the CPU oracle.
//
// Needs a cuDNN build AND a CUDA device; without either every test SKIPs
// (never fails). The Conv table row exists only under NERT_HAS_CUDNN, so
// the table-missing check doubles as the no-cuDNN skip.

#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

// GTEST_SKIP must fire in the test body (it only returns from its own
// function), so helpers report readiness and each TEST skips explicitly.
bool HasCudaConv() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        return false;
    }
#ifdef NERT_HAS_CUDNN
    return nert::CudaDevice().kernels().find("Conv") != nullptr;
#else
    return false;
#endif
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

void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3F * (1.0F + std::fabs(ref[i]))) << "index " << i;
    }
}

std::vector<float> RunCpuConv(nert::Node node, std::vector<float>& x, std::vector<int64_t> xd, std::vector<float>& w,
                              std::vector<int64_t> wd, std::vector<float>* bias, std::vector<int64_t> outDims) {
    std::vector<nert::TensorView> inputs = {HostFloat(x, xd), HostFloat(w, wd)};
    if (bias != nullptr) {
        inputs.push_back(HostFloat(*bias, {wd[0]}));
    }
    const int64_t count = outDims[0] * outDims[1] * outDims[2] * (outDims.size() > 3 ? outDims[3] : 1);
    std::vector<float> out(static_cast<std::size_t>(count), 0.0F);
    nert::CpuDevice().kernels().find("Conv")(node, inputs, {HostFloat(out, outDims)});
    return out;
}

} // namespace

// 1x1 conv, no bias: y[m] = sum_c w[m,c] * x[c] per position.
// X[c0]=[[1,2],[3,4]], X[c1]=[[5,6],[7,8]]; W = [[1,0],[0,1],[1,1]].
TEST(NertCudaConv, PointwiseNoBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> w = {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    nert::Node node = MakeNode("cv0", "Conv");
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    const std::vector<float> expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 6.0F, 8.0F, 10.0F, 12.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 2, 2, 2}, w, {3, 2, 1, 1}, nullptr, {1, 3, 2, 2}), expected);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    std::vector<float> gpu(12, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {3, 2, 1, 1})},
                                              {DeviceFloat(dy, {1, 3, 2, 2})});
    gpu = Download(cuda, dy, 12);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// 3x3 stride-1 pad-1 with bias 10 over 1..9: hand-computed window sums
// [22,31,26,37,55,43,34,49,38], all within tolerance of cuDNN.
TEST(NertCudaConv, Padded3x3WithBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    std::vector<float> w(9, 1.0F);
    std::vector<float> b = {10.0F};
    nert::Node node = MakeNode("cv1", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    const std::vector<float> expected = {22.0F, 31.0F, 26.0F, 37.0F, 55.0F, 43.0F, 34.0F, 49.0F, 38.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 1, 3, 3}, w, {1, 1, 3, 3}, &b, {1, 1, 3, 3}), expected);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(9, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 1, 3, 3}), DeviceFloat(dw, {1, 1, 3, 3}), DeviceFloat(db, {1})},
        {DeviceFloat(dy, {1, 1, 3, 3})});
    gpu = Download(cuda, dy, 9);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// Depthwise (group=2=C=M) 1x1 with bias: y[m] = w[m]*x[m] + b[m].
// y0 = 2*x0+1 -> [3,5,7,9]; y1 = 3*x1-1 -> [14,17,20,23].
TEST(NertCudaConv, DepthwiseWithBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> w = {2.0F, 3.0F};
    std::vector<float> b = {1.0F, -1.0F};
    nert::Node node = MakeNode("cv2", "Conv");
    node.attributes.emplace_back("group", int64_t{2});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    const std::vector<float> expected = {3.0F, 5.0F, 7.0F, 9.0F, 14.0F, 17.0F, 20.0F, 23.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 2, 2, 2}, w, {2, 1, 1, 1}, &b, {1, 2, 2, 2}), expected);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(8, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {2, 1, 1, 1}), DeviceFloat(db, {2})},
        {DeviceFloat(dy, {1, 2, 2, 2})});
    gpu = Download(cuda, dy, 8);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// Conv rejects a rank-1 bias whose length is not M, naming node + op.
TEST(NertCudaConv, RejectsBadBiasShape) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(8, 1.0F);
    std::vector<float> w(6, 1.0F);
    std::vector<float> b = {0.0F, 0.0F};
    std::vector<float> y(12, 0.0F);
    nert::Node node = MakeNode("cv_bad", "Conv");
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Conv")(
            node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {3, 2, 1, 1}), DeviceFloat(db, {2})},
            {DeviceFloat(dy, {1, 3, 2, 2})});
        FAIL() << "expected InferenceException for a bad bias shape";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cv_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Conv"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// Single-Conv graph through the executor on CUDA matches CPU, proving
// staging + dispatch for the new row.
TEST(NertCudaConv, ModelConvMatchesCpu) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }

    nert::Graph graph;
    graph.inputs = {{"x", {nert::DataType::Float32, {}}},
                    {"w", {nert::DataType::Float32, {}}},
                    {"b", {nert::DataType::Float32, {}}}};
    nert::Node conv = MakeNode("cv", "Conv");
    conv.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    conv.inputs = {"x", "w", "b"};
    conv.outputs = {"y"};
    graph.nodes = {conv};
    graph.outputs = {{"y", {nert::DataType::Float32, {}}}};
    const nert::ShapeMap dims = {{"x", {1, 1, 3, 3}}, {"w", {1, 1, 3, 3}}, {"b", {1}}};
    const std::map<std::string, std::vector<float>> inputs = {
        {"x", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F}},
        {"w", std::vector<float>(9, 1.0F)},
        {"b", {10.0F}}};

    nert::Model cpu(graph, dims, nert::CpuDevice());
    const std::vector<float> expected = cpu.Run(inputs).outputs.at("y");
    EXPECT_EQ(expected, (std::vector<float>{22.0F, 31.0F, 26.0F, 37.0F, 55.0F, 43.0F, 34.0F, 49.0F, 38.0F}));

    nert::Model cuda(graph, dims, nert::CudaDevice());
    ExpectNear(cuda.Run(inputs).outputs.at("y"), expected);
}
