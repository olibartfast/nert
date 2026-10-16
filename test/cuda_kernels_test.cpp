// CUDA pilot kernels (N1a-M2): Gemm via cuBLAS, first-party Relu/Add.
//
// Every test needs a CUDA device at run time; without one it SKIPs (never
// fails). Kernel-level tests run the CPU oracle table and the CUDA table on
// identical inputs; the model test runs one micro-graph on both devices.

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

// Skips the test when no CUDA device answers (never fails).
void RequireCudaDevice() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        GTEST_SKIP() << "no CUDA device (cudaGetDeviceCount found none)";
    }
}

nert::TensorView HostFloat(std::vector<float>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

nert::TensorView HostInt64(std::vector<int64_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

// A device copy of a host float buffer, released by the caller.
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

// |a - b| <= 1e-3 * (1 + |b|) elementwise, the bench tolerance.
void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3F * (1.0F + std::fabs(ref[i]))) << "index " << i;
    }
}

} // namespace

// Gemm small exact: [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] =
// [[58,64],[139,154]] on both tables, bit-exact here.
TEST(NertCudaKernels, GemmSmallExact) {
    RequireCudaDevice();
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> b = {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("gemm0", "Gemm");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Gemm")(node, {HostFloat(a, {2, 3}), HostFloat(b, {3, 2})},
                                             {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{58.0F, 64.0F, 139.0F, 154.0F}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Gemm")(node, {DeviceFloat(da, {2, 3}), DeviceFloat(db, {3, 2})},
                                              {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Gemm with transposes, non-default alpha/beta, and a rank-1 [N] bias matches
// the CPU oracle within tolerance.
TEST(NertCudaKernels, GemmTransposedWithBiasWithinTolerance) {
    RequireCudaDevice();
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    // A stored [3,2] transA=1 -> op(A) [2,3]; B stored [4,3] transB=1 ->
    // op(B) [3,4]; output [2,4] with a rank-1 [4] bias.
    std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> b = {1.0F, 0.0F, 0.0F, 1.0F, 2.0F, 1.0F, 0.0F, 2.0F, 1.0F, 1.0F, 0.0F, 1.0F};
    std::vector<float> c = {0.5F, -0.5F, 1.0F, 2.0F};
    nert::Node node = MakeNode("gemm1", "Gemm");
    node.attributes.emplace_back("transA", int64_t{1});
    node.attributes.emplace_back("transB", int64_t{1});
    node.attributes.emplace_back("alpha", 2.0F);
    node.attributes.emplace_back("beta", 0.5F);
    node.inputs = {"a", "b", "c"};
    node.outputs = {"y"};

    std::vector<float> cpuOut(8, 0.0F);
    nert::CpuDevice().kernels().find("Gemm")(node, {HostFloat(a, {3, 2}), HostFloat(b, {4, 3}), HostFloat(c, {4})},
                                             {HostFloat(cpuOut, {2, 4})});

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dc = Upload(cuda, c);
    std::vector<float> gpuOut(8, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Gemm")(
        node, {DeviceFloat(da, {3, 2}), DeviceFloat(db, {4, 3}), DeviceFloat(dc, {4})}, {DeviceFloat(dy, {2, 4})});
    gpuOut = Download(cuda, dy, 8);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dc);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// Relu matches the CPU oracle exactly, incl. the zero boundary.
TEST(NertCudaKernels, ReluExact) {
    RequireCudaDevice();
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> in = {-2.0F, -0.0F, 0.0F, 3.0F, 1.5F};
    std::vector<float> cpuOut(5, 0.0F);
    nert::Node node = MakeNode("relu0", "Relu");
    nert::CpuDevice().kernels().find("Relu")(node, {HostFloat(in, {5})}, {HostFloat(cpuOut, {5})});

    void* dx = Upload(cuda, in);
    std::vector<float> gpuOut(5, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Relu")(node, {DeviceFloat(dx, {5})}, {DeviceFloat(dy, {5})});
    gpuOut = Download(cuda, dy, 5);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Add broadcasts [2,1] int64 against [1,3] exactly like the CPU oracle.
TEST(NertCudaKernels, AddBroadcastInt64Exact) {
    RequireCudaDevice();
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {10, 20, 30};
    std::vector<int64_t> cpuOut(6, 0);
    nert::Node node = MakeNode("add0", "Add");
    nert::CpuDevice().kernels().find("Add")(node, {HostInt64(a, {2, 1}), HostInt64(b, {1, 3})},
                                            {HostInt64(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{11, 21, 31, 12, 22, 32}));

    auto uploadI64 = [&](const std::vector<int64_t>& host) {
        void* device = cuda.allocator().allocate(static_cast<int64_t>(host.size() * sizeof(int64_t)));
        EXPECT_NE(device, nullptr);
        cuda.transfer().host_to_device(device, host.data(), static_cast<int64_t>(host.size() * sizeof(int64_t)));
        return device;
    };
    auto downloadI64 = [&](void* device, std::size_t count) {
        std::vector<int64_t> host(count, 0);
        cuda.transfer().device_to_host(host.data(), device, static_cast<int64_t>(count * sizeof(int64_t)));
        return host;
    };
    auto devI64 = [&](void* device, std::vector<int64_t> dims) {
        nert::TensorView view;
        view.data = device;
        view.dtype = nert::DataType::Int64;
        view.dims = std::move(dims);
        return view;
    };
    void* da = uploadI64(a);
    void* db = uploadI64(b);
    std::vector<int64_t> gpuOut(6, 0);
    void* dy = uploadI64(gpuOut);
    nert::CudaDevice().kernels().find("Add")(node, {devI64(da, {2, 1}), devI64(db, {1, 3})}, {devI64(dy, {2, 3})});
    gpuOut = downloadI64(dy, 6);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Micro-graph Gemm -> Relu -> Add runs on both devices and agrees within
// tolerance, proving staging (upload/discharge) plus dispatch end to end.
TEST(NertCudaKernels, MicroGraphMatchesCpuWithinTolerance) {
    RequireCudaDevice();

    nert::Graph graph;
    graph.inputs = {{"a", {nert::DataType::Float32, {}}},
                    {"b", {nert::DataType::Float32, {}}},
                    {"c", {nert::DataType::Float32, {}}},
                    {"d", {nert::DataType::Float32, {}}}};
    nert::Node gemm = MakeNode("mm", "Gemm");
    gemm.inputs = {"a", "b", "c"};
    gemm.outputs = {"y"};
    nert::Node relu = MakeNode("rl", "Relu");
    relu.inputs = {"y"};
    relu.outputs = {"r"};
    nert::Node add = MakeNode("ad", "Add");
    add.inputs = {"r", "d"};
    add.outputs = {"z"};
    graph.nodes = {gemm, relu, add};
    graph.outputs = {{"z", {nert::DataType::Float32, {}}}};

    const nert::ShapeMap dims = {{"a", {2, 3}}, {"b", {3, 2}}, {"c", {2}}, {"d", {2, 2}}};
    const std::map<std::string, std::vector<float>> inputs = {{"a", {1.0F, -2.0F, 3.0F, 0.5F, 1.0F, -1.0F}},
                                                              {"b", {2.0F, 0.0F, -1.0F, 1.0F, 0.5F, 2.0F}},
                                                              {"c", {0.25F, -0.25F}},
                                                              {"d", {1.0F, 1.0F, 1.0F, 1.0F}}};

    nert::Model cpu(graph, dims, nert::CpuDevice());
    const std::vector<float> expected = cpu.Run(inputs).outputs.at("z");

    nert::Model cuda(graph, dims, nert::CudaDevice());
    const std::vector<float> got = cuda.Run(inputs).outputs.at("z");
    ExpectNear(got, expected);
}

// An op with no CUDA row is rejected at Run, naming node + op (dispatch-time
// rejection, matching the pinned CPU behavior; see N1 validation notes).
TEST(NertCudaKernels, UnknownOpRejectsNamingNodeAndOp) {
    RequireCudaDevice();

    nert::Graph graph;
    graph.inputs = {{"x", {nert::DataType::Float32, {}}}};
    nert::Node sigmoid = MakeNode("sg_bad", "Sigmoid");
    sigmoid.inputs = {"x"};
    sigmoid.outputs = {"y"};
    graph.nodes = {sigmoid};
    graph.outputs = {{"y", {nert::DataType::Float32, {}}}};

    nert::Model cuda(graph, {{"x", {4}}}, nert::CudaDevice());
    try {
        cuda.Run({{"x", std::vector<float>(4, 1.0F)}});
        FAIL() << "expected InferenceException for an op with no CUDA kernel";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sg_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Sigmoid"), std::string::npos) << message;
    }
}
