// OpenCL pilot kernels (N1d-M2): first-party Relu/Add vs the CPU oracle.
//
// Needs an OpenCL GPU platform at run time; without one every test SKIPs
// (never fails). Gemm never lands on OpenCL (cuBLAS-only), so it is the
// permanent no-row probe.

#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

bool HasOpenCLDevice() {
    try {
        (void)nert::OpenCLDevice();
        return true;
    } catch (const nert::InferenceException&) {
        return false;
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

void* Upload(nert::Device& ocl, const std::vector<float>& host) {
    void* device = ocl.allocator().allocate(static_cast<int64_t>(host.size() * sizeof(float)));
    if (device == nullptr) {
        throw std::runtime_error("Upload: device allocate failed");
    }
    ocl.transfer().host_to_device(device, host.data(), static_cast<int64_t>(host.size() * sizeof(float)));
    return device;
}

void* UploadInt64(nert::Device& ocl, const std::vector<int64_t>& host) {
    void* device = ocl.allocator().allocate(static_cast<int64_t>(host.size() * sizeof(int64_t)));
    if (device == nullptr) {
        throw std::runtime_error("UploadInt64: device allocate failed");
    }
    ocl.transfer().host_to_device(device, host.data(), static_cast<int64_t>(host.size() * sizeof(int64_t)));
    return device;
}

std::vector<float> Download(nert::Device& ocl, void* device, std::size_t count) {
    std::vector<float> host(count, 0.0F);
    ocl.transfer().device_to_host(host.data(), device, static_cast<int64_t>(count * sizeof(float)));
    return host;
}

std::vector<int64_t> DownloadInt64(nert::Device& ocl, void* device, std::size_t count) {
    std::vector<int64_t> host(count, 0);
    ocl.transfer().device_to_host(host.data(), device, static_cast<int64_t>(count * sizeof(int64_t)));
    return host;
}

nert::TensorView DeviceFloat(void* device, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = device;
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

nert::TensorView DeviceInt64(void* device, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = device;
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

nert::Node MakeNode(const std::string& name, const std::string& op) {
    nert::Node node;
    node.name = name;
    node.op_type = op;
    return node;
}

} // namespace

// Relu over {-2,-0.5,0,3}: negatives clamp, exact on both tables.
TEST(NertOpenCLKernels, ReluExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {-2.0F, -0.5F, 0.0F, 3.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("relu0", "Relu");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Relu")(node, {HostFloat(x, {4})}, {HostFloat(cpuOut, {4})});
    EXPECT_EQ(cpuOut, (std::vector<float>{0.0F, 0.0F, 0.0F, 3.0F}));

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Relu")(node, {DeviceFloat(dx, {4})}, {DeviceFloat(dy, {4})});
    gpuOut = Download(ocl, dy, 4);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Add broadcasts [2,1] int64 against [1,3] exactly like the CPU oracle.
TEST(NertOpenCLKernels, AddBroadcastInt64Exact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {10, 20, 30};
    std::vector<int64_t> cpuOut(6, 0);
    nert::Node node = MakeNode("add0", "Add");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Add")(node, {HostInt64(a, {2, 1}), HostInt64(b, {1, 3})},
                                            {HostInt64(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{11, 21, 31, 12, 22, 32}));

    void* da = UploadInt64(ocl, a);
    void* db = UploadInt64(ocl, b);
    std::vector<int64_t> gpuOut(6, 0);
    void* dy = UploadInt64(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Add")(node, {DeviceInt64(da, {2, 1}), DeviceInt64(db, {1, 3})},
                                               {DeviceInt64(dy, {2, 3})});
    gpuOut = DownloadInt64(ocl, dy, 6);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Gemm never lands on OpenCL: rejected at Run, naming node + op.
TEST(NertOpenCLKernels, UnknownOpRejectsNamingNodeAndOp) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }

    nert::Graph graph;
    graph.inputs = {{"a", {nert::DataType::Float32, {}}}, {"b", {nert::DataType::Float32, {}}}};
    nert::Node gemm = MakeNode("mm_bad", "Gemm");
    gemm.inputs = {"a", "b"};
    gemm.outputs = {"y"};
    graph.nodes = {gemm};
    graph.outputs = {{"y", {nert::DataType::Float32, {}}}};

    nert::Model ocl(graph, {{"a", {2, 3}}, {"b", {3, 2}}}, nert::OpenCLDevice());
    try {
        ocl.Run({{"a", std::vector<float>(6, 1.0F)}, {"b", std::vector<float>(6, 1.0F)}});
        FAIL() << "expected InferenceException for an op with no OpenCL kernel";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mm_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Gemm"), std::string::npos) << message;
    }
}

// Mul broadcasts [2,1] float against [1,3] exactly like the CPU oracle.
TEST(NertOpenCLKernels, MulBroadcastExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> a = {1.5F, -2.0F};
    std::vector<float> b = {2.0F, 0.5F, -1.0F};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("mul0", "Mul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Mul")(node, {HostFloat(a, {2, 1}), HostFloat(b, {1, 3})},
                                            {HostFloat(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{3.0F, 0.75F, -1.5F, -4.0F, -1.0F, 2.0F}));

    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Mul")(node, {DeviceFloat(da, {2, 1}), DeviceFloat(db, {1, 3})},
                                               {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(ocl, dy, 6);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Sub over int64 matches the CPU oracle exactly, incl. negatives.
TEST(NertOpenCLKernels, SubInt64Exact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<int64_t> a = {5, -1, 0};
    std::vector<int64_t> b = {3, 2, -7};
    std::vector<int64_t> cpuOut(3, 0);
    nert::Node node = MakeNode("sub0", "Sub");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Sub")(node, {HostInt64(a, {3}), HostInt64(b, {3})}, {HostInt64(cpuOut, {3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{2, -3, 7}));

    void* da = UploadInt64(ocl, a);
    void* db = UploadInt64(ocl, b);
    std::vector<int64_t> gpuOut(3, 0);
    void* dy = UploadInt64(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Sub")(node, {DeviceInt64(da, {3}), DeviceInt64(db, {3})},
                                               {DeviceInt64(dy, {3})});
    gpuOut = DownloadInt64(ocl, dy, 3);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Div float follows IEEE: 1/0 = inf, -1/0 = -inf — like the oracle (NaN
// compared by predicate, never by equality).
TEST(NertOpenCLKernels, DivFloatIEEE) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> a = {1.0F, -1.0F, 5.0F};
    std::vector<float> b = {0.0F, 0.0F, 2.0F};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("div0", "Div");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Div")(node, {HostFloat(a, {3}), HostFloat(b, {3})}, {HostFloat(cpuOut, {3})});
    EXPECT_EQ(cpuOut[2], 2.5F);

    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    std::vector<float> gpuOut(3, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Div")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {3})},
                                               {DeviceFloat(dy, {3})});
    gpuOut = Download(ocl, dy, 3);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut[2], cpuOut[2]);
    EXPECT_TRUE(std::isinf(gpuOut[0]) && gpuOut[0] > 0);
    EXPECT_TRUE(std::isinf(gpuOut[1]) && gpuOut[1] < 0);
}

// Div int64 with a zero divisor throws naming node + op, like the oracle.
TEST(NertOpenCLKernels, DivInt64ZeroThrows) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {2, 0};
    std::vector<int64_t> y(2, 0);
    nert::Node node = MakeNode("div_bad", "Div");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = UploadInt64(ocl, a);
    void* db = UploadInt64(ocl, b);
    void* dy = UploadInt64(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("Div")(node, {DeviceInt64(da, {2}), DeviceInt64(db, {2})},
                                                   {DeviceInt64(dy, {2})});
        FAIL() << "expected InferenceException for int64 division by zero";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("div_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Div"), std::string::npos) << message;
    }
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
}

// OpenCL activations run in float (libm + float accumulation), so they meet
// the oracle within the M2+ tolerance bar, not bit for bit.
void ExpectNearOcl(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3 * (1.0 + std::fabs(ref[i]))) << "i=" << i;
    }
}

std::vector<float> RunOcl(const std::string& op, nert::Node node, const std::vector<float>& x,
                          const std::vector<int64_t>& dims) {
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> cpuOut(x.size(), 0.0F);
    nert::CpuDevice().kernels().find(op)(node, {HostFloat(const_cast<std::vector<float>&>(x), dims)},
                                         {HostFloat(cpuOut, dims)});
    void* dx = Upload(ocl, x);
    std::vector<float> zero(x.size(), 0.0F);
    void* dy = Upload(ocl, zero);
    nert::OpenCLDevice().kernels().find(op)(node, {DeviceFloat(dx, dims)}, {DeviceFloat(dy, dims)});
    std::vector<float> gpuOut = Download(ocl, dy, x.size());
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    ExpectNearOcl(gpuOut, cpuOut);
    return gpuOut;
}

// Sigmoid over {-2,-0.5,0,2} matches the oracle within tolerance.
TEST(NertOpenCLKernels, SigmoidNear) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("sig0", "Sigmoid");
    node.inputs = {"x"};
    node.outputs = {"y"};
    RunOcl("Sigmoid", node, {-2.0F, -0.5F, 0.0F, 2.0F}, {4});
}

// Tanh over {-2,-0.5,0,2} matches the oracle within tolerance.
TEST(NertOpenCLKernels, TanhNear) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("tanh0", "Tanh");
    node.inputs = {"x"};
    node.outputs = {"y"};
    RunOcl("Tanh", node, {-2.0F, -0.5F, 0.0F, 2.0F}, {4});
}

// Softmax default axis (-1) over [2,3]: each row is softmax([1,2,3]).
TEST(NertOpenCLKernels, SoftmaxLastAxis) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("sm0", "Softmax");
    node.inputs = {"x"};
    node.outputs = {"y"};
    const std::vector<float> gpu = RunOcl("Softmax", node, {1.0F, 2.0F, 3.0F, 1.0F, 2.0F, 3.0F}, {2, 3});
    for (int r = 0; r < 2; ++r) {
        EXPECT_NEAR(gpu[r * 3 + 0] + gpu[r * 3 + 1] + gpu[r * 3 + 2], 1.0F, 1e-3) << "row " << r;
    }
}

// Softmax axis=0 over [3,2] with large values: stability via max subtraction.
TEST(NertOpenCLKernels, SoftmaxAxisZeroStable) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("sm1", "Softmax");
    node.inputs = {"x"};
    node.outputs = {"y"};
    node.attributes.emplace_back("axis", int64_t{0});
    const std::vector<float> gpu =
        RunOcl("Softmax", node, {1000.0F, 1000.0F, 1001.0F, 1001.0F, 1002.0F, 1002.0F}, {3, 2});
    for (int c = 0; c < 2; ++c) {
        EXPECT_NEAR(gpu[0 * 2 + c] + gpu[1 * 2 + c] + gpu[2 * 2 + c], 1.0F, 1e-3) << "col " << c;
    }
}
