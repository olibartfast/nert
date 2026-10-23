// HIP first-party kernels (N1c on NVIDIA): the 72-row table vs the CPU oracle.
//
// Needs a HIP-usable GPU at run time; without one every test SKIPs (never
// fails). The compute rows (Gemm, MatMul, Conv, ConvTranspose) are naive
// first-party device kernels with double accumulation: bit-exact on small
// integer fixtures, otherwise compared within the standard tolerance bar.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

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

// The standard tolerance bar: |got - ref| <= 1e-3 * (1 + |ref|).
void ExpectNearHip(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3 * (1.0 + std::fabs(ref[i]))) << "i=" << i;
    }
}

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

// Gemm small exact: [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] =
// [[58,64],[139,154]] on both tables, bit-exact here.
TEST(NertHIPKernels, GemmSmallExact) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> b = {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("gemm0", "Gemm");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Gemm")(node, {HostFloat(a, {2, 3}), HostFloat(b, {3, 2})},
                                             {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{58.0F, 64.0F, 139.0F, 154.0F}));

    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("Gemm")(node, {DeviceFloat(da, {2, 3}), DeviceFloat(db, {3, 2})},
                                             {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(hip, dy, 4);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Gemm with transposes, non-default alpha/beta, and a rank-1 [N] bias matches
// the CPU oracle within tolerance.
TEST(NertHIPKernels, GemmTransposedWithBiasWithinTolerance) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

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

    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    void* dc = Upload(hip, c);
    std::vector<float> gpuOut(8, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("Gemm")(
        node, {DeviceFloat(da, {3, 2}), DeviceFloat(db, {4, 3}), DeviceFloat(dc, {4})}, {DeviceFloat(dy, {2, 4})});
    gpuOut = Download(hip, dy, 8);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dc);
    hip.allocator().release(dy);
    ExpectNearHip(gpuOut, cpuOut);
}

// Gemm rejects a contraction mismatch, naming node + op.
TEST(NertHIPKernels, GemmRejectsMismatch) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> a(6, 1.0F);
    std::vector<float> b(4, 1.0F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("gemm_bad", "Gemm");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    void* dy = Upload(hip, y);
    try {
        nert::HipDevice().kernels().find("Gemm")(node, {DeviceFloat(da, {2, 3}), DeviceFloat(db, {2, 2})},
                                                 {DeviceFloat(dy, {2, 2})});
        FAIL() << "expected InferenceException for a contraction mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("gemm_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Gemm"), std::string::npos) << message;
    }
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
}

// MatMul broadcasts a {2,3} A over a {2,3,2} B: two hand-computed batches.
TEST(NertHIPKernels, MatMulBatchedBroadcast) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> b = {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2.0F, 0.0F, 0.0F, 2.0F, 1.0F, 0.0F};
    std::vector<float> cpuOut(8, 0.0F);
    nert::Node node = MakeNode("mm0", "MatMul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("MatMul")(node, {HostFloat(a, {2, 3}), HostFloat(b, {2, 3, 2})},
                                               {HostFloat(cpuOut, {2, 2, 2})});
    const std::vector<float> expected = {4.0F, 5.0F, 10.0F, 11.0F, 5.0F, 4.0F, 14.0F, 10.0F};
    EXPECT_EQ(cpuOut, expected);

    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    std::vector<float> gpuOut(8, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("MatMul")(node, {DeviceFloat(da, {2, 3}), DeviceFloat(db, {2, 3, 2})},
                                               {DeviceFloat(dy, {2, 2, 2})});
    gpuOut = Download(hip, dy, 8);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    ExpectNearHip(gpuOut, expected);
}

// MatMul promotes a 1-D A: [1,2,3] x [[1,2],[3,4],[5,6]] = [22,28].
TEST(NertHIPKernels, MatMulPromotedVector) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F};
    std::vector<float> b = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("mm1", "MatMul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("MatMul")(node, {HostFloat(a, {3}), HostFloat(b, {3, 2})},
                                               {HostFloat(cpuOut, {2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{22.0F, 28.0F}));

    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("MatMul")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {3, 2})},
                                               {DeviceFloat(dy, {2})});
    gpuOut = Download(hip, dy, 2);
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    ExpectNearHip(gpuOut, cpuOut);
}

// MatMul rejects a contraction mismatch, naming node + op.
TEST(NertHIPKernels, MatMulRejectsMismatch) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> a(6, 1.0F);
    std::vector<float> b(6, 1.0F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("mm_bad", "MatMul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(hip, a);
    void* db = Upload(hip, b);
    void* dy = Upload(hip, y);
    try {
        nert::HipDevice().kernels().find("MatMul")(node, {DeviceFloat(da, {2, 3}), DeviceFloat(db, {2, 2})},
                                                   {DeviceFloat(dy, {2, 2})});
        FAIL() << "expected InferenceException for a contraction mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mm_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("MatMul"), std::string::npos) << message;
    }
    hip.allocator().release(da);
    hip.allocator().release(db);
    hip.allocator().release(dy);
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

// Reshape [2,3] -> [3,2] preserves flat order; the shape operand is read
// back from device memory and validated on the host.
TEST(NertHIPKernels, ReshapeFloatPreservesOrder) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> shape = {3, 2};
    nert::Node node = MakeNode("reshape0", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};

    void* dx = Upload(hip, x);
    void* dshape = UploadInt64(hip, shape);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("Reshape")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dshape, {2})},
                                                {DeviceFloat(dy, {3, 2})});
    gpuOut = Download(hip, dy, 6);
    hip.allocator().release(dx);
    hip.allocator().release(dshape);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, x);
}

// Reshape shape {0,-1} copies the matching input dim (2) and infers 3.
TEST(NertHIPKernels, ReshapeZeroAndInferred) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> shape = {0, -1};
    nert::Node node = MakeNode("reshape1", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};

    void* dx = Upload(hip, x);
    void* dshape = UploadInt64(hip, shape);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("Reshape")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dshape, {2})},
                                                {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(hip, dy, 6);
    hip.allocator().release(dx);
    hip.allocator().release(dshape);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, x);
}

// Reshape int64 copies raw elements unchanged.
TEST(NertHIPKernels, ReshapeInt64Copies) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<int64_t> x = {7, -8, 9, 10};
    nert::Node node = MakeNode("reshape2", "Reshape");
    node.inputs = {"x"};
    node.outputs = {"y"};

    void* dx = UploadInt64(hip, x);
    std::vector<int64_t> gpuOut(4, 0);
    void* dy = UploadInt64(hip, gpuOut);
    nert::HipDevice().kernels().find("Reshape")(node, {DeviceInt64(dx, {2, 2})}, {DeviceInt64(dy, {4})});
    gpuOut = DownloadInt64(hip, dy, 4);
    hip.allocator().release(dx);
    hip.allocator().release(dy);
    EXPECT_EQ(gpuOut, x);
}

// Reshape whose target product differs from the element count is rejected,
// naming the op and node.
TEST(NertHIPKernels, ReshapeRejectsCountMismatch) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> shape = {4};
    nert::Node node = MakeNode("reshape_bad", "Reshape");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};

    void* dx = Upload(hip, x);
    void* dshape = UploadInt64(hip, shape);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(hip, gpuOut);
    try {
        nert::HipDevice().kernels().find("Reshape")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dshape, {1})},
                                                    {DeviceFloat(dy, {4})});
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("Reshape"), std::string::npos) << message;
        EXPECT_NE(message.find("reshape_bad"), std::string::npos) << message;
    }
    hip.allocator().release(dx);
    hip.allocator().release(dshape);
    hip.allocator().release(dy);
}

// Runs the CPU oracle Conv row on host views; the suite compares the HIP row
// against its result on device views.
std::vector<float> RunCpuConv(nert::Node node, std::vector<float>& x, std::vector<int64_t> xd, std::vector<float>& w,
                              std::vector<int64_t> wd, std::vector<float>* bias, std::vector<int64_t> outDims) {
    std::vector<nert::TensorView> inputs = {HostFloat(x, xd), HostFloat(w, wd)};
    if (bias != nullptr) {
        inputs.push_back(HostFloat(*bias, {wd[0]}));
    }
    int64_t count = 1;
    for (int64_t dim : outDims) {
        count *= dim;
    }
    std::vector<float> out(static_cast<std::size_t>(count), 0.0F);
    nert::CpuDevice().kernels().find("Conv")(node, inputs, {HostFloat(out, outDims)});
    return out;
}

// 1x1 conv, no bias: y[m] = sum_c w[m,c] * x[c] per position.
// X[c0]=[[1,2],[3,4]], X[c1]=[[5,6],[7,8]]; W = [[1,0],[0,1],[1,1]].
TEST(NertHIPKernels, ConvPointwiseNoBias) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> w = {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    nert::Node node = MakeNode("cv0", "Conv");
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    const std::vector<float> expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 6.0F, 8.0F, 10.0F, 12.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 2, 2, 2}, w, {3, 2, 1, 1}, nullptr, {1, 3, 2, 2}), expected);

    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    std::vector<float> gpu(12, 0.0F);
    void* dy = Upload(hip, gpu);
    nert::HipDevice().kernels().find("Conv")(node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {3, 2, 1, 1})},
                                             {DeviceFloat(dy, {1, 3, 2, 2})});
    gpu = Download(hip, dy, 12);
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(dy);
    ExpectNearHip(gpu, expected);
}

// 3x3 stride-1 pad-1 with bias 10 over 1..9: hand-computed window sums
// [22,31,26,37,55,43,34,49,38].
TEST(NertHIPKernels, ConvPadded3x3WithBias) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    std::vector<float> w(9, 1.0F);
    std::vector<float> b = {10.0F};
    nert::Node node = MakeNode("cv1", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    const std::vector<float> expected = {22.0F, 31.0F, 26.0F, 37.0F, 55.0F, 43.0F, 34.0F, 49.0F, 38.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 1, 3, 3}, w, {1, 1, 3, 3}, &b, {1, 1, 3, 3}), expected);

    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    void* db = Upload(hip, b);
    std::vector<float> gpu(9, 0.0F);
    void* dy = Upload(hip, gpu);
    nert::HipDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 1, 3, 3}), DeviceFloat(dw, {1, 1, 3, 3}), DeviceFloat(db, {1})},
        {DeviceFloat(dy, {1, 1, 3, 3})});
    gpu = Download(hip, dy, 9);
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    ExpectNearHip(gpu, expected);
}

// Depthwise (group=2=C=M) 1x1 with bias: y[m] = w[m]*x[m] + b[m].
// y0 = 2*x0+1 -> [3,5,7,9]; y1 = 3*x1-1 -> [14,17,20,23].
TEST(NertHIPKernels, ConvDepthwiseWithBias) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> w = {2.0F, 3.0F};
    std::vector<float> b = {1.0F, -1.0F};
    nert::Node node = MakeNode("cv2", "Conv");
    node.attributes.emplace_back("group", int64_t{2});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    const std::vector<float> expected = {3.0F, 5.0F, 7.0F, 9.0F, 14.0F, 17.0F, 20.0F, 23.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 2, 2, 2}, w, {2, 1, 1, 1}, &b, {1, 2, 2, 2}), expected);

    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    void* db = Upload(hip, b);
    std::vector<float> gpu(8, 0.0F);
    void* dy = Upload(hip, gpu);
    nert::HipDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {2, 1, 1, 1}), DeviceFloat(db, {2})},
        {DeviceFloat(dy, {1, 2, 2, 2})});
    gpu = Download(hip, dy, 8);
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    ExpectNearHip(gpu, expected);
}

// auto_pad=SAME_UPPER with an odd pad total resolves asymmetric (begin 0 /
// end 1 per dim). Compared against the CPU oracle.
TEST(NertHIPKernels, ConvSameUpperAsymmetricPad) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x(1 * 2 * 3 * 3);
    std::vector<float> w(2 * 2 * 2 * 2);
    std::vector<float> b = {1.5F, -2.5F};
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.5F;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    }
    nert::Node node = MakeNode("cv_same", "Conv");
    node.attributes.emplace_back("auto_pad", std::string{"SAME_UPPER"});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    // out = ceil(3/1) = 3, total = (3-1)*1 + 2 - 3 = 1 -> begin 0, end 1.
    const std::vector<float> expected = RunCpuConv(node, x, {1, 2, 3, 3}, w, {2, 2, 2, 2}, &b, {1, 2, 3, 3});
    ASSERT_EQ(expected.size(), 1u * 2 * 3 * 3);

    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    void* db = Upload(hip, b);
    std::vector<float> gpu(2 * 3 * 3, 0.0F);
    void* dy = Upload(hip, gpu);
    nert::HipDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 2, 3, 3}), DeviceFloat(dw, {2, 2, 2, 2}), DeviceFloat(db, {2})},
        {DeviceFloat(dy, {1, 2, 3, 3})});
    gpu = Download(hip, dy, 2 * 3 * 3);
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    ExpectNearHip(gpu, expected);
}

// Conv rejects a rank-1 bias whose length is not M, naming node + op.
TEST(NertHIPKernels, ConvRejectsBadBiasShape) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x(8, 1.0F);
    std::vector<float> w(6, 1.0F);
    std::vector<float> b = {0.0F, 0.0F};
    std::vector<float> y(12, 0.0F);
    nert::Node node = MakeNode("cv_bad", "Conv");
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    void* db = Upload(hip, b);
    void* dy = Upload(hip, y);
    try {
        nert::HipDevice().kernels().find("Conv")(
            node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {3, 2, 1, 1}), DeviceFloat(db, {2})},
            {DeviceFloat(dy, {1, 3, 2, 2})});
        FAIL() << "expected InferenceException for a bad bias shape";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cv_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Conv"), std::string::npos) << message;
    }
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(db);
    hip.allocator().release(dy);
}

// ConvTranspose 2x2 stride-2 no bias, W = I-ish {1,0,0,1}: hand-computed
// gather (each input cell lands on a strided 2x2 block).
TEST(NertHIPKernels, ConvTransposeStride2NoBias) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> w = {1.0F, 0.0F, 0.0F, 1.0F};
    nert::Node node = MakeNode("ct0", "ConvTranspose");
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    const std::vector<float> expected = {1.0F, 0.0F, 2.0F, 0.0F, 0.0F, 1.0F, 0.0F, 2.0F,
                                         3.0F, 0.0F, 4.0F, 0.0F, 0.0F, 3.0F, 0.0F, 4.0F};
    std::vector<float> cpuOut(16, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(node, {HostFloat(x, {1, 1, 2, 2}), HostFloat(w, {1, 1, 2, 2})},
                                                      {HostFloat(cpuOut, {1, 1, 4, 4})});
    ExpectNearHip(cpuOut, expected);

    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    std::vector<float> gpuOut(16, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("ConvTranspose")(
        node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dw, {1, 1, 2, 2})}, {DeviceFloat(dy, {1, 1, 4, 4})});
    gpuOut = Download(hip, dy, 16);
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(dy);
    ExpectNearHip(gpuOut, expected);
}

// ConvTranspose 3x3 stride-1 pad-1 with bias vs the oracle.
TEST(NertHIPKernels, ConvTransposePaddedWithBias) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> w = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    std::vector<float> bias = {10.0F};
    nert::Node node = MakeNode("ct1", "ConvTranspose");
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    std::vector<float> cpuOut(4, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(
        node, {HostFloat(x, {1, 1, 2, 2}), HostFloat(w, {1, 1, 3, 3}), HostFloat(bias, {1})},
        {HostFloat(cpuOut, {1, 1, 2, 2})});
    // Corner (0,0): x00*w11+x01*w10+x10*w01+x11*w00 + 10 = 1+2+3+4+10.
    ExpectNearHip(cpuOut, (std::vector<float>{20.0F, 20.0F, 20.0F, 20.0F}));

    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    void* db = Upload(hip, bias);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(hip, gpuOut);
    nert::HipDevice().kernels().find("ConvTranspose")(
        node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dw, {1, 1, 3, 3}), DeviceFloat(db, {1})},
        {DeviceFloat(dy, {1, 1, 2, 2})});
    gpuOut = Download(hip, dy, 4);
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(db);
    hip.allocator().release(dy);
    ExpectNearHip(gpuOut, cpuOut);
}

// ConvTranspose rejects an output view that does not match the transposed
// window, naming node + op.
TEST(NertHIPKernels, ConvTransposeRejectsOutputShapeMismatch) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());

    std::vector<float> x(4, 1.0F);
    std::vector<float> w(4, 1.0F);
    std::vector<float> y(16, 0.0F);
    nert::Node node = MakeNode("ct_bad", "ConvTranspose");
    node.attributes.emplace_back("strides", std::vector<int64_t>{1, 1});
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    void* dx = Upload(hip, x);
    void* dw = Upload(hip, w);
    void* dy = Upload(hip, y);
    try {
        // Stride 1 over 2x2 with a 2x2 kernel computes 3x3, not 4x4.
        nert::HipDevice().kernels().find("ConvTranspose")(
            node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dw, {1, 1, 2, 2})}, {DeviceFloat(dy, {1, 1, 4, 4})});
        FAIL() << "expected InferenceException for an output shape mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ct_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ConvTranspose"), std::string::npos) << message;
    }
    hip.allocator().release(dx);
    hip.allocator().release(dw);
    hip.allocator().release(dy);
}
