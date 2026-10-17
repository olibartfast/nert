// HIP pilot kernels (N1c-M2 on NVIDIA): hipified Relu/Add vs the CPU oracle.
//
// Needs a HIP-usable GPU at run time; without one every test SKIPs (never
// fails). Gemm never lands on HIP-NVIDIA (rocBLAS is the AMD half), so it
// is the permanent no-row probe.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
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

bool HasHIPDevice() { return nert_test::DeviceAvailable(&nert::HipDevice); }

} // namespace

// Relu over {-2,-0.5,0,3}: negatives clamp, exact on both tables.
TEST(NertHIPKernels, ReluExact) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {-2.0F, -0.5F, 0.0F, 3.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("relu0", "Relu");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Relu")(node, {HostFloat(x, {4})}, {HostFloat(cpuOut, {4})});
    EXPECT_EQ(cpuOut, (std::vector<float>{0.0F, 0.0F, 0.0F, 3.0F}));

    void* dx = Upload(hip, x);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("Relu")(node, {DeviceFloat(dx, {4})}, {DeviceFloat(dy, {4})});
    gpuOut = Download(hip, dy, 4);
    hip.allocator().release(dx);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Add broadcasts [2,1] int64 against [1,3] exactly like the CPU oracle.
TEST(NertHIPKernels, AddBroadcastInt64Exact) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {10, 20, 30};
    std::vector<int64_t> cpuOut(6, 0);
    nert::Node node = MakeNode("add0", "Add");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Add")(node, {HostInt64(a, {2, 1}), HostInt64(b, {1, 3})},
                                            {HostInt64(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{11, 21, 31, 12, 22, 32}));

    void* da = UploadInt64(hip, a);
    void* db = UploadInt64(hip, b);
    std::vector<int64_t> gpuOut(6, 0);
    void* dy = UploadInt64(hip, gpuOut);
    nert::HipDevice().kernels().find("Add")(node, {DeviceInt64(da, {2, 1}), DeviceInt64(db, {1, 3})},
                                            {DeviceInt64(dy, {2, 3})});
    gpuOut = DownloadInt64(hip, dy, 6);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Gemm never lands on HIP-NVIDIA: rejected at Run, naming node + op.
TEST(NertHIPKernels, UnknownOpRejectsNamingNodeAndOp) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }

    nert::Graph graph;
    graph.inputs = {{"a", {nert::DataType::Float32, {}}}, {"b", {nert::DataType::Float32, {}}}};
    nert::Node gemm = MakeNode("mm_bad", "Gemm");
    gemm.inputs = {"a", "b"};
    gemm.outputs = {"y"};
    graph.nodes = {gemm};
    graph.outputs = {{"y", {nert::DataType::Float32, {}}}};

    nert::Model hip(graph, {{"a", {2, 3}}, {"b", {3, 2}}}, nert::HipDevice());
    try {
        hip.Run({{"a", std::vector<float>(6, 1.0F)}, {"b", std::vector<float>(6, 1.0F)}});
        FAIL() << "expected InferenceException for an op with no HIP kernel";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mm_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Gemm"), std::string::npos) << message;
    }
}

// Mul broadcasts [2,1] float against [1,3] exactly like the CPU oracle.
TEST(NertHIPKernels, MulBroadcastExact) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> a = {1.5F, -2.0F};
    std::vector<float> b = {2.0F, 0.5F, -1.0F};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("mul0", "Mul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Mul")(node, {HostFloat(a, {2, 1}), HostFloat(b, {1, 3})},
                                            {HostFloat(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{3.0F, 0.75F, -1.5F, -4.0F, -1.0F, 2.0F}));

    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("Mul")(node, {DeviceFloat(da, {2, 1}), DeviceFloat(db, {1, 3})},
                                            {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(hip, dy, 6);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Sub over int64 matches the CPU oracle exactly, incl. negatives.
TEST(NertHIPKernels, SubInt64Exact) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<int64_t> a = {5, -1, 0};
    std::vector<int64_t> b = {3, 2, -7};
    std::vector<int64_t> cpuOut(3, 0);
    nert::Node node = MakeNode("sub0", "Sub");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Sub")(node, {HostInt64(a, {3}), HostInt64(b, {3})}, {HostInt64(cpuOut, {3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{2, -3, 7}));

    void* da = UploadInt64(hip, a);
    void* db = UploadInt64(hip, b);
    std::vector<int64_t> gpuOut(3, 0);
    void* dy = UploadInt64(hip, gpuOut);
    nert::HipDevice().kernels().find("Sub")(node, {DeviceInt64(da, {3}), DeviceInt64(db, {3})}, {DeviceInt64(dy, {3})});
    gpuOut = DownloadInt64(hip, dy, 3);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Div float follows IEEE: 1/0 = inf, -1/0 = -inf — like the oracle (NaN
// compared by predicate, never by equality).
TEST(NertHIPKernels, DivFloatIEEE) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> a = {1.0F, -1.0F, 5.0F};
    std::vector<float> b = {0.0F, 0.0F, 2.0F};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("div0", "Div");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Div")(node, {HostFloat(a, {3}), HostFloat(b, {3})}, {HostFloat(cpuOut, {3})});
    EXPECT_EQ(cpuOut[2], 2.5F);

    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    std::vector<float> gpuOut(3, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("Div")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {3})}, {DeviceFloat(dy, {3})});
    gpuOut = Download(hip, dy, 3);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut[2], cpuOut[2]);
    EXPECT_TRUE(std::isinf(gpuOut[0]) && gpuOut[0] > 0);
    EXPECT_TRUE(std::isinf(gpuOut[1]) && gpuOut[1] < 0);
}

// Div int64 with a zero divisor throws naming node + op, like the oracle.
TEST(NertHIPKernels, DivInt64ZeroThrows) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {2, 0};
    std::vector<int64_t> y(2, 0);
    nert::Node node = MakeNode("div_bad", "Div");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = UploadInt64(hip, a);
    void* db = UploadInt64(hip, b);
    void* dy = UploadInt64(hip, y);
    try {
        nert::HipDevice().kernels().find("Div")(node, {DeviceInt64(da, {2}), DeviceInt64(db, {2})},
                                                {DeviceInt64(dy, {2})});
        FAIL() << "expected InferenceException for int64 division by zero";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("div_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Div"), std::string::npos) << message;
    }
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
}

// HIP activations run the same math as CUDA (double-accumulated softmax),
// so they meet the oracle within the standard tolerance bar.
void ExpectNearHip(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3 * (1.0 + std::fabs(ref[i]))) << "i=" << i;
    }
}

std::vector<float> RunHip(const std::string& op, nert::Node node, const std::vector<float>& x,
                          const std::vector<int64_t>& dims) {
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());
    std::vector<float> cpuOut(x.size(), 0.0F);
    nert::CpuDevice().kernels().find(op)(node, {HostFloat(const_cast<std::vector<float>&>(x), dims)},
                                         {HostFloat(cpuOut, dims)});
    void* dx = Upload(hip, x);
    std::vector<float> zero(x.size(), 0.0F);
    void* dy = Upload(hip, zero);
    nert::HipDevice().kernels().find(op)(node, {DeviceFloat(dx, dims)}, {DeviceFloat(dy, dims)});
    std::vector<float> gpuOut = Download(hip, dy, x.size());
    hip.allocator().release(dx);
    hip.allocator().release(dy);
    ExpectNearHip(gpuOut, cpuOut);
    return gpuOut;
}

// Sigmoid over {-2,-0.5,0,2} matches the oracle within tolerance.
TEST(NertHIPKernels, SigmoidNear) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Node node = MakeNode("sig0", "Sigmoid");
    node.inputs = {"x"};
    node.outputs = {"y"};
    RunHip("Sigmoid", node, {-2.0F, -0.5F, 0.0F, 2.0F}, {4});
}

// Tanh over {-2,-0.5,0,2} matches the oracle within tolerance.
TEST(NertHIPKernels, TanhNear) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Node node = MakeNode("tanh0", "Tanh");
    node.inputs = {"x"};
    node.outputs = {"y"};
    RunHip("Tanh", node, {-2.0F, -0.5F, 0.0F, 2.0F}, {4});
}

// Softmax default axis (-1) over [2,3]: each row is softmax([1,2,3]).
TEST(NertHIPKernels, SoftmaxLastAxis) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Node node = MakeNode("sm0", "Softmax");
    node.inputs = {"x"};
    node.outputs = {"y"};
    const std::vector<float> gpu = RunHip("Softmax", node, {1.0F, 2.0F, 3.0F, 1.0F, 2.0F, 3.0F}, {2, 3});
    for (int r = 0; r < 2; ++r) {
        EXPECT_NEAR(gpu[r * 3 + 0] + gpu[r * 3 + 1] + gpu[r * 3 + 2], 1.0F, 1e-3) << "row " << r;
    }
}

// Softmax axis=0 over [3,2] with large values: stability via max subtraction.
TEST(NertHIPKernels, SoftmaxAxisZeroStable) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Node node = MakeNode("sm1", "Softmax");
    node.inputs = {"x"};
    node.outputs = {"y"};
    node.attributes.emplace_back("axis", int64_t{0});
    const std::vector<float> gpu =
        RunHip("Softmax", node, {1000.0F, 1000.0F, 1001.0F, 1001.0F, 1002.0F, 1002.0F}, {3, 2});
    for (int c = 0; c < 2; ++c) {
        EXPECT_NEAR(gpu[0 * 2 + c] + gpu[1 * 2 + c] + gpu[2 * 2 + c], 1.0F, 1e-3) << "col " << c;
    }
}
