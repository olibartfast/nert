// OpenCL pilot kernels (N1d-M2): first-party Relu/Add vs the CPU oracle.
//
// Needs an OpenCL GPU platform at run time; without one every test SKIPs
// (never fails). Gemm never lands on OpenCL (cuBLAS-only), so it is the
// permanent no-row probe.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using nert_test::DeviceFloat;
using nert_test::DeviceInt64;
using nert_test::Download;
using nert_test::DownloadInt64;
using nert_test::HostFloat;
using nert_test::HostInt64;
using nert_test::MakeNode;
using nert_test::Upload;
using nert_test::UploadInt64;

bool HasOpenCLDevice() { return nert_test::DeviceAvailable(&nert::OpenCLDevice); }

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

// Reshape [2,3] -> [3,2] preserves flat order; the shape operand is read
// back from device memory and validated on the host.
TEST(NertOpenCLKernels, ReshapeFloatPreservesOrder) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> shape = {3, 2};
    nert::Node node = MakeNode("reshape0", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};

    void* dx = Upload(ocl, x);
    void* dshape = UploadInt64(ocl, shape);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Reshape")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dshape, {2})},
                                                   {DeviceFloat(dy, {3, 2})});
    gpuOut = Download(ocl, dy, 6);
    ocl.allocator().release(dx);
    ocl.allocator().release(dshape);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, x);
}

// Reshape shape {0,-1} copies the matching input dim (2) and infers 3.
TEST(NertOpenCLKernels, ReshapeZeroAndInferred) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> shape = {0, -1};
    nert::Node node = MakeNode("reshape1", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};

    void* dx = Upload(ocl, x);
    void* dshape = UploadInt64(ocl, shape);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Reshape")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dshape, {2})},
                                                   {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(ocl, dy, 6);
    ocl.allocator().release(dx);
    ocl.allocator().release(dshape);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, x);
}

// Reshape int64 copies raw elements unchanged.
TEST(NertOpenCLKernels, ReshapeInt64Copies) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<int64_t> x = {7, -8, 9, 10};
    nert::Node node = MakeNode("reshape2", "Reshape");
    node.inputs = {"x"};
    node.outputs = {"y"};

    void* dx = UploadInt64(ocl, x);
    std::vector<int64_t> gpuOut(4, 0);
    void* dy = UploadInt64(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Reshape")(node, {DeviceInt64(dx, {2, 2})}, {DeviceInt64(dy, {4})});
    gpuOut = DownloadInt64(ocl, dy, 4);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, x);
}

// Reshape whose target product differs from the element count is rejected,
// naming the op and node.
TEST(NertOpenCLKernels, ReshapeRejectsCountMismatch) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> shape = {4};
    nert::Node node = MakeNode("reshape_bad", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};

    void* dx = Upload(ocl, x);
    void* dshape = UploadInt64(ocl, shape);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    try {
        nert::OpenCLDevice().kernels().find("Reshape")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dshape, {1})},
                                                       {DeviceFloat(dy, {4})});
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("Reshape"), std::string::npos) << message;
        EXPECT_NE(message.find("reshape_bad"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(dshape);
    ocl.allocator().release(dy);
}

// Transpose [2,3] default (reverse) vs oracle.
TEST(NertOpenCLKernels, TransposeDefaultExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("tr0", "Transpose");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Transpose")(node, {HostFloat(x, {2, 3})}, {HostFloat(cpuOut, {3, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}));

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Transpose")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {3, 2})});
    gpuOut = Download(ocl, dy, 6);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Transpose int64 with explicit perm {2, 0, 1} vs oracle.
TEST(NertOpenCLKernels, TransposePermInt64Exact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<int64_t> x = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> cpuOut(8, 0);
    nert::Node node = MakeNode("tr1", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{2, 0, 1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Transpose")(node, {HostInt64(x, {2, 2, 2})}, {HostInt64(cpuOut, {2, 2, 2})});

    void* dx = UploadInt64(ocl, x);
    std::vector<int64_t> gpuOut(8, 0);
    void* dy = UploadInt64(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Transpose")(node, {DeviceInt64(dx, {2, 2, 2})}, {DeviceInt64(dy, {2, 2, 2})});
    gpuOut = DownloadInt64(ocl, dy, 8);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Transpose rejects a non-permutation, naming node + op.
TEST(NertOpenCLKernels, TransposeRejectsBadPerm) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("tr_bad", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{0, 0});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("Transpose")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {2, 3})});
        FAIL() << "expected InferenceException for a non-permutation";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("tr_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Transpose"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Gather axis 1 with a negative index vs oracle.
TEST(NertOpenCLKernels, GatherAxisNegIndexExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> idx = {0, -1};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("ga0", "Gather");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Gather")(node, {HostFloat(x, {2, 3}), HostInt64(idx, {2})},
                                               {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 3.0F, 4.0F, 6.0F}));

    void* dx = Upload(ocl, x);
    void* di = UploadInt64(ocl, idx);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Gather")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(di, {2})},
                                                  {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(ocl, dy, 4);
    ocl.allocator().release(dx);
    ocl.allocator().release(di);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Gather rejects an out-of-range index, naming node + op.
TEST(NertOpenCLKernels, GatherRejectsOutOfRange) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> idx = {5};
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("ga_bad", "Gather");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* di = UploadInt64(ocl, idx);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("Gather")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(di, {1})},
                                                      {DeviceFloat(dy, {2, 1})});
        FAIL() << "expected InferenceException for an out-of-range index";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ga_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Gather"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(di);
    ocl.allocator().release(dy);
}

// Slice [2,4] rows 1..2 of axis 1 with negative end, both tables.
TEST(NertOpenCLKernels, SliceNegEndExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<int64_t> starts = {1};
    std::vector<int64_t> ends = {-1};
    std::vector<int64_t> axes = {1};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("sl0", "Slice");
    node.inputs = {"x", "starts", "ends", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Slice")(
        node, {HostFloat(x, {2, 4}), HostInt64(starts, {1}), HostInt64(ends, {1}), HostInt64(axes, {1})},
        {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{2.0F, 3.0F, 6.0F, 7.0F}));

    void* dx = Upload(ocl, x);
    void* ds = UploadInt64(ocl, starts);
    void* de = UploadInt64(ocl, ends);
    void* dax = UploadInt64(ocl, axes);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Slice")(
        node, {DeviceFloat(dx, {2, 4}), DeviceInt64(ds, {1}), DeviceInt64(de, {1}), DeviceInt64(dax, {1})},
        {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(ocl, dy, 4);
    ocl.allocator().release(dx);
    ocl.allocator().release(ds);
    ocl.allocator().release(de);
    ocl.allocator().release(dax);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Slice rejects a zero step, naming node + op.
TEST(NertOpenCLKernels, SliceRejectsZeroStep) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> starts = {0};
    std::vector<int64_t> ends = {2};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {0};
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("sl_bad", "Slice");
    node.inputs = {"x", "starts", "ends", "axes", "steps"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* ds = UploadInt64(ocl, starts);
    void* de = UploadInt64(ocl, ends);
    void* da = UploadInt64(ocl, axes);
    void* dt = UploadInt64(ocl, steps);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("Slice")(node,
                                                     {DeviceFloat(dx, {2, 3}), DeviceInt64(ds, {1}),
                                                      DeviceInt64(de, {1}), DeviceInt64(da, {1}), DeviceInt64(dt, {1})},
                                                     {DeviceFloat(dy, {2, 3})});
        FAIL() << "expected InferenceException for a zero step";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sl_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Slice"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(ds);
    ocl.allocator().release(de);
    ocl.allocator().release(da);
    ocl.allocator().release(dt);
    ocl.allocator().release(dy);
}

// Concat two [2,2] along axis -1, both tables.
TEST(NertOpenCLKernels, ConcatNegAxisExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> b = {5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> cpuOut(8, 0.0F);
    nert::Node node = MakeNode("cc0", "Concat");
    node.attributes.emplace_back("axis", int64_t{-1});
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Concat")(node, {HostFloat(a, {2, 2}), HostFloat(b, {2, 2})},
                                               {HostFloat(cpuOut, {2, 4})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 2.0F, 5.0F, 6.0F, 3.0F, 4.0F, 7.0F, 8.0F}));

    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    std::vector<float> gpuOut(8, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Concat")(node, {DeviceFloat(da, {2, 2}), DeviceFloat(db, {2, 2})},
                                                  {DeviceFloat(dy, {2, 4})});
    gpuOut = Download(ocl, dy, 8);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Concat rejects off-axis disagreement, naming node + op.
TEST(NertOpenCLKernels, ConcatRejectsOffAxisMismatch) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> a(4, 1.0F);
    std::vector<float> b(6, 2.0F);
    std::vector<float> y(10, 0.0F);
    nert::Node node = MakeNode("cc_bad", "Concat");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("Concat")(node, {DeviceFloat(da, {2, 2}), DeviceFloat(db, {3, 2})},
                                                      {DeviceFloat(dy, {5, 2})});
        FAIL() << "expected InferenceException for off-axis disagreement";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cc_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Concat"), std::string::npos) << message;
    }
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
}

// Shape start=1 end=-1 over [2,3,4] -> [3], both tables.
TEST(NertOpenCLKernels, ShapeSliceExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(24, 1.0F);
    std::vector<int64_t> cpuOut(1, 0);
    nert::Node node = MakeNode("sh0", "Shape");
    node.attributes.emplace_back("start", int64_t{1});
    node.attributes.emplace_back("end", int64_t{-1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Shape")(node, {HostFloat(x, {2, 3, 4})}, {HostInt64(cpuOut, {1})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{3}));

    void* dx = Upload(ocl, x);
    std::vector<int64_t> gpuOut(1, 0);
    void* dy = UploadInt64(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Shape")(node, {DeviceFloat(dx, {2, 3, 4})}, {DeviceInt64(dy, {1})});
    gpuOut = DownloadInt64(ocl, dy, 1);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Shape rejects a non-int64 output, naming node + op.
TEST(NertOpenCLKernels, ShapeRejectsNonInt64Out) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> y(3, 0.0F);
    nert::Node node = MakeNode("sh_bad", "Shape");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("Shape")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {3})});
        FAIL() << "expected InferenceException for a non-int64 output";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sh_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Shape"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Unsqueeze [2,3] axes {0,-1} -> [1,2,3,1], both tables.
TEST(NertOpenCLKernels, UnsqueezeNegAxesExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> axes = {0, -1};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("us0", "Unsqueeze");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Unsqueeze")(node, {HostFloat(x, {2, 3}), HostInt64(axes, {2})},
                                                  {HostFloat(cpuOut, {1, 2, 3, 1})});
    EXPECT_EQ(cpuOut, x);

    void* dx = Upload(ocl, x);
    void* da = UploadInt64(ocl, axes);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Unsqueeze")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {2})},
                                                     {DeviceFloat(dy, {1, 2, 3, 1})});
    gpuOut = Download(ocl, dy, 6);
    ocl.allocator().release(dx);
    ocl.allocator().release(da);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Unsqueeze rejects duplicate axes, naming node + op.
TEST(NertOpenCLKernels, UnsqueezeRejectsDupAxes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> axes = {1, -3};
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("us_bad", "Unsqueeze");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* da = UploadInt64(ocl, axes);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("Unsqueeze")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {2})},
                                                         {DeviceFloat(dy, {1, 2, 3})});
        FAIL() << "expected InferenceException for duplicate axes";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("us_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Unsqueeze"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(da);
    ocl.allocator().release(dy);
}

// MaxPool k2/s2 over 1..16 (4x4): hand-computed window maxima.
TEST(NertOpenCLKernels, MaxPoolStride2Exact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                            9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("mp0", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("MaxPool")(node, {HostFloat(x, {1, 1, 4, 4})}, {HostFloat(cpuOut, {1, 1, 2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{6.0F, 8.0F, 14.0F, 16.0F}));

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("MaxPool")(node, {DeviceFloat(dx, {1, 1, 4, 4})},
                                                   {DeviceFloat(dy, {1, 1, 2, 2})});
    gpuOut = Download(ocl, dy, 4);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// MaxPool k2/s1/pad1/dilation2 vs oracle (exercises pads + dilation).
TEST(NertOpenCLKernels, MaxPoolDilatedPaddedMatchesOracle) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {3.0F, 1.0F, 4.0F, 1.0F, 5.0F, 9.0F, 2.0F, 6.0F, 5.0F};
    std::vector<float> cpuOut(9, 0.0F);
    nert::Node node = MakeNode("mp1", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("dilations", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("MaxPool")(node, {HostFloat(x, {1, 1, 3, 3})}, {HostFloat(cpuOut, {1, 1, 3, 3})});

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(9, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("MaxPool")(node, {DeviceFloat(dx, {1, 1, 3, 3})},
                                                   {DeviceFloat(dy, {1, 1, 3, 3})});
    gpuOut = Download(ocl, dy, 9);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// MaxPool rejects a missing kernel_shape, naming node + op.
TEST(NertOpenCLKernels, MaxPoolRejectsMissingKernel) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(16, 1.0F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("mp_bad", "MaxPool");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("MaxPool")(node, {DeviceFloat(dx, {1, 1, 4, 4})},
                                                       {DeviceFloat(dy, {1, 1, 2, 2})});
        FAIL() << "expected InferenceException for a missing kernel_shape";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mp_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("MaxPool"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// AveragePool k2/s2 over 1..16 (4x4): hand-computed window means.
TEST(NertOpenCLKernels, AveragePoolStride2Exact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F,  3.0F,  4.0F,  5.0F,  6.0F,  7.0F,  8.0F,
                            9.0F, 10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("ap0", "AveragePool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("AveragePool")(node, {HostFloat(x, {1, 1, 4, 4})},
                                                    {HostFloat(cpuOut, {1, 1, 2, 2})});
    ExpectNearOcl(cpuOut, (std::vector<float>{3.5F, 5.5F, 11.5F, 13.5F}));

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("AveragePool")(node, {DeviceFloat(dx, {1, 1, 4, 4})},
                                                       {DeviceFloat(dy, {1, 1, 2, 2})});
    gpuOut = Download(ocl, dy, 4);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    ExpectNearOcl(gpuOut, cpuOut);
}

// AveragePool count_include_pad with pads vs oracle (divisor rule).
TEST(NertOpenCLKernels, AveragePoolCountPadMatchesOracle) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    auto run = [&](int64_t inc) {
        std::vector<float> cpuOut(4, 0.0F);
        nert::Node node = MakeNode("ap1", "AveragePool");
        node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
        node.attributes.emplace_back("strides", std::vector<int64_t>{1, 1});
        node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 0, 0});
        node.attributes.emplace_back("count_include_pad", inc);
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("AveragePool")(node, {HostFloat(x, {1, 1, 2, 2})},
                                                        {HostFloat(cpuOut, {1, 1, 2, 2})});

        void* dx = Upload(ocl, x);
        std::vector<float> gpuOut(4, 0.0F);
        void* dy = Upload(ocl, gpuOut);
        nert::OpenCLDevice().kernels().find("AveragePool")(node, {DeviceFloat(dx, {1, 1, 2, 2})},
                                                           {DeviceFloat(dy, {1, 1, 2, 2})});
        gpuOut = Download(ocl, dy, 4);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
        ExpectNearOcl(gpuOut, cpuOut);
    };
    run(0);
    run(1);
}

// GlobalAveragePool over [1,2,2,2] of 1..8: channel means {2.5, 6.5}.
TEST(NertOpenCLKernels, GlobalAveragePoolExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("gap0", "GlobalAveragePool");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("GlobalAveragePool")(node, {HostFloat(x, {1, 2, 2, 2})},
                                                          {HostFloat(cpuOut, {1, 2, 1, 1})});
    ExpectNearOcl(cpuOut, (std::vector<float>{2.5F, 6.5F}));

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("GlobalAveragePool")(node, {DeviceFloat(dx, {1, 2, 2, 2})},
                                                             {DeviceFloat(dy, {1, 2, 1, 1})});
    gpuOut = Download(ocl, dy, 2);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    ExpectNearOcl(gpuOut, cpuOut);
}

// GlobalAveragePool rejects a bad output shape, naming node + op.
TEST(NertOpenCLKernels, GlobalAveragePoolRejectsBadShape) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(8, 1.0F);
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("gap_bad", "GlobalAveragePool");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("GlobalAveragePool")(node, {DeviceFloat(dx, {1, 2, 2, 2})},
                                                                 {DeviceFloat(dy, {2})});
        FAIL() << "expected InferenceException for a bad output shape";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("gap_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("GlobalAveragePool"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Concat with an empty operand: {2,0,4} + {2,3,4} along axis 1 -> {2,3,4}.
// RF-DETR's transformer concatenates a zero-length slice; a zero extent is
// legal and contributes nothing.
TEST(NertOpenCLKernels, ConcatEmptyOperandExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> empty;
    std::vector<float> b(24, 0.0F);
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(i + 1);
    }
    std::vector<float> cpuOut(24, 0.0F);
    nert::Node node = MakeNode("cc_empty", "Concat");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Concat")(node, {HostFloat(empty, {2, 0, 4}), HostFloat(b, {2, 3, 4})},
                                               {HostFloat(cpuOut, {2, 3, 4})});
    EXPECT_EQ(cpuOut, b);

    void* db = Upload(ocl, b);
    std::vector<float> gpuOut(24, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Concat")(node, {DeviceFloat(nullptr, {2, 0, 4}), DeviceFloat(db, {2, 3, 4})},
                                                  {DeviceFloat(dy, {2, 3, 4})});
    gpuOut = Download(ocl, dy, 24);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Concat bool rows: {2,2} + {2,1} along axis 1 -> {2,3}, bytes unchanged.
TEST(NertOpenCLKernels, ConcatBoolRows) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<std::uint8_t> a = {1, 0, 1, 0};
    std::vector<std::uint8_t> b = {0, 1};
    std::vector<std::uint8_t> cpuOut(6, 0);
    nert::Node node = MakeNode("cc_b", "Concat");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    auto host_view = [](std::vector<std::uint8_t>& data, std::vector<int64_t> dims) {
        nert::TensorView view;
        view.data = data.data();
        view.dtype = nert::DataType::Bool;
        view.dims = std::move(dims);
        return view;
    };
    nert::CpuDevice().kernels().find("Concat")(node, {host_view(a, {2, 2}), host_view(b, {2, 1})},
                                               {host_view(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 0, 0, 1, 0, 1}));

    auto upload = [&](const std::vector<std::uint8_t>& host) {
        void* buffer = ocl.allocator().allocate(static_cast<int64_t>(host.size()));
        ocl.transfer().host_to_device(buffer, host.data(), static_cast<int64_t>(host.size()));
        return buffer;
    };
    auto device_view = [](void* data, std::vector<int64_t> dims) {
        nert::TensorView view;
        view.data = data;
        view.dtype = nert::DataType::Bool;
        view.dims = std::move(dims);
        return view;
    };
    void* da = upload(a);
    void* db = upload(b);
    std::vector<std::uint8_t> gpuOut(6, 0);
    void* dy = upload(gpuOut);
    nert::OpenCLDevice().kernels().find("Concat")(node, {device_view(da, {2, 2}), device_view(db, {2, 1})},
                                                  {device_view(dy, {2, 3})});
    ocl.transfer().device_to_host(gpuOut.data(), dy, static_cast<int64_t>(gpuOut.size()));
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// MaxPool ceil_mode over 3x3 k2/s2 -> 2x2; the final window is clipped, so
// the out-of-range samples stay -inf and drop out of the max.
TEST(NertOpenCLKernels, MaxPoolCeilModeExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("mp_ceil", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("ceil_mode", int64_t{1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("MaxPool")(node, {HostFloat(x, {1, 1, 3, 3})}, {HostFloat(cpuOut, {1, 1, 2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{5.0F, 6.0F, 8.0F, 9.0F}));

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("MaxPool")(node, {DeviceFloat(dx, {1, 1, 3, 3})},
                                                   {DeviceFloat(dy, {1, 1, 2, 2})});
    gpuOut = Download(ocl, dy, 4);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// AveragePool ceil_mode + pads: the include-pad divisor counts padded
// positions inside the window but clips at the padded input edge.
TEST(NertOpenCLKernels, AveragePoolCeilPadDivisor) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    auto run = [&](int64_t includePad, const std::vector<float>& expected) {
        std::vector<float> cpuOut(4, 0.0F);
        nert::Node node = MakeNode("ap_ceil", "AveragePool");
        node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
        node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
        node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 0, 0});
        node.attributes.emplace_back("ceil_mode", int64_t{1});
        node.attributes.emplace_back("count_include_pad", includePad);
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("AveragePool")(node, {HostFloat(x, {1, 1, 2, 2})},
                                                        {HostFloat(cpuOut, {1, 1, 2, 2})});
        ExpectNearOcl(cpuOut, expected);

        void* dx = Upload(ocl, x);
        std::vector<float> gpuOut(4, 0.0F);
        void* dy = Upload(ocl, gpuOut);
        nert::OpenCLDevice().kernels().find("AveragePool")(node, {DeviceFloat(dx, {1, 1, 2, 2})},
                                                           {DeviceFloat(dy, {1, 1, 2, 2})});
        gpuOut = Download(ocl, dy, 4);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
        ExpectNearOcl(gpuOut, cpuOut);
    };
    run(1, std::vector<float>{0.25F, 1.0F, 1.5F, 4.0F});
    run(0, std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F});
}

// MaxPool rejects a pads list that is not two entries per spatial axis.
TEST(NertOpenCLKernels, MaxPoolRejectsBadPads) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(16, 1.0F);
    std::vector<float> y(9, 0.0F);
    nert::Node node = MakeNode("mp_pads", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 0});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(ocl, x);
    void* dy = Upload(ocl, y);
    try {
        nert::OpenCLDevice().kernels().find("MaxPool")(node, {DeviceFloat(dx, {1, 1, 4, 4})},
                                                       {DeviceFloat(dy, {1, 1, 3, 3})});
        FAIL() << "expected InferenceException for a malformed pads list";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mp_pads"), std::string::npos) << message;
        EXPECT_NE(message.find("MaxPool"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// GlobalAveragePool rank-3: [2,3,4] -> [2,3,1], one mean per (n,c) plane.
TEST(NertOpenCLKernels, GlobalAveragePoolRank3Exact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x(24, 0.0F);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i + 1);
    }
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("gap3", "GlobalAveragePool");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("GlobalAveragePool")(node, {HostFloat(x, {2, 3, 4})},
                                                          {HostFloat(cpuOut, {2, 3, 1})});
    EXPECT_EQ(cpuOut, (std::vector<float>{2.5F, 6.5F, 10.5F, 14.5F, 18.5F, 22.5F}));

    void* dx = Upload(ocl, x);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("GlobalAveragePool")(node, {DeviceFloat(dx, {2, 3, 4})},
                                                             {DeviceFloat(dy, {2, 3, 1})});
    gpuOut = Download(ocl, dy, 6);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Slice negative axis + negative step: {1..6} -> indices 5,3,1 -> {6,4,2}.
TEST(NertOpenCLKernels, SliceNegativeAxisStepExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> starts = {5};
    std::vector<int64_t> ends = {0};
    std::vector<int64_t> axes = {-1};
    std::vector<int64_t> steps = {-2};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("sl_neg", "Slice");
    node.inputs = {"x", "starts", "ends", "axes", "steps"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Slice")(
        node,
        {HostFloat(x, {6}), HostInt64(starts, {1}), HostInt64(ends, {1}), HostInt64(axes, {1}), HostInt64(steps, {1})},
        {HostFloat(cpuOut, {3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{6.0F, 4.0F, 2.0F}));

    void* dx = Upload(ocl, x);
    void* dst = UploadInt64(ocl, starts);
    void* den = UploadInt64(ocl, ends);
    void* dax = UploadInt64(ocl, axes);
    void* dstp = UploadInt64(ocl, steps);
    std::vector<float> gpuOut(3, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Slice")(node,
                                                 {DeviceFloat(dx, {6}), DeviceInt64(dst, {1}), DeviceInt64(den, {1}),
                                                  DeviceInt64(dax, {1}), DeviceInt64(dstp, {1})},
                                                 {DeviceFloat(dy, {3})});
    gpuOut = Download(ocl, dy, 3);
    ocl.allocator().release(dx);
    ocl.allocator().release(dst);
    ocl.allocator().release(den);
    ocl.allocator().release(dax);
    ocl.allocator().release(dstp);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}
