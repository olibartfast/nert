// CUDA pilot kernels (N1a-M2): Gemm via cuBLAS, first-party Relu/Add.
//
// Every test needs a CUDA device at run time; without one it SKIPs (never
// fails). Kernel-level tests run the CPU oracle table and the CUDA table on
// identical inputs; the model test runs one micro-graph on both devices.

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Gemm small exact: [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] =
// [[58,64],[139,154]] on both tables, bit-exact here.
TEST(NertCudaKernels, GemmSmallExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
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
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
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
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
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
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {10, 20, 30};
    std::vector<int64_t> cpuOut(6, 0);
    nert::Node node = MakeNode("add0", "Add");
    nert::CpuDevice().kernels().find("Add")(node, {HostInt64(a, {2, 1}), HostInt64(b, {1, 3})},
                                            {HostInt64(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{11, 21, 31, 12, 22, 32}));

    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    std::vector<int64_t> gpuOut(6, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Add")(node, {DeviceInt64(da, {2, 1}), DeviceInt64(db, {1, 3})},
                                             {DeviceInt64(dy, {2, 3})});
    gpuOut = DownloadInt64(cuda, dy, 6);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Micro-graph Gemm -> Relu -> Add runs on both devices and agrees within
// tolerance, proving staging (upload/discharge) plus dispatch end to end.
TEST(NertCudaKernels, MicroGraphMatchesCpuWithinTolerance) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }

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
// Probe op is Einsum: CPU-covered with shape inference, and no CUDA row is
// planned anywhere, so this probe never needs moving again.
TEST(NertCudaKernels, UnknownOpRejectsNamingNodeAndOp) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }

    nert::Graph graph;
    graph.inputs = {{"a", {nert::DataType::Float32, {}}}, {"b", {nert::DataType::Float32, {}}}};
    nert::Node einsum = MakeNode("ein_bad", "Einsum");
    einsum.attributes.emplace_back("equation", std::string{"ij,jh->ih"});
    einsum.inputs = {"a", "b"};
    einsum.outputs = {"y"};
    graph.nodes = {einsum};
    graph.outputs = {{"y", {nert::DataType::Float32, {}}}};

    nert::Model cuda(graph, {{"a", {2, 3}}, {"b", {3, 2}}}, nert::CudaDevice());
    try {
        cuda.Run({{"a", std::vector<float>(6, 1.0F)}, {"b", std::vector<float>(6, 1.0F)}});
        FAIL() << "expected InferenceException for an op with no CUDA kernel";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ein_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Einsum"), std::string::npos) << message;
    }
}

// Mul broadcasts [2,1] float against [1,3] exactly like the CPU oracle.
TEST(NertCudaKernels, MulBroadcastExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.5F, -2.0F};
    std::vector<float> b = {2.0F, 0.5F, -1.0F};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("mul0", "Mul");
    nert::CpuDevice().kernels().find("Mul")(node, {HostFloat(a, {2, 1}), HostFloat(b, {1, 3})},
                                            {HostFloat(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{3.0F, 0.75F, -1.5F, -4.0F, -1.0F, 2.0F}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Mul")(node, {DeviceFloat(da, {2, 1}), DeviceFloat(db, {1, 3})},
                                             {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(cuda, dy, 6);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Sub over int64 matches the CPU oracle exactly, incl. negatives.
TEST(NertCudaKernels, SubInt64Exact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {5, -1, 0};
    std::vector<int64_t> b = {3, 2, -7};
    std::vector<int64_t> cpuOut(3, 0);
    nert::Node node = MakeNode("sub0", "Sub");
    // CPU oracle via int64 views.
    {
        nert::TensorView va = HostInt64(a, {3});
        nert::TensorView vb = HostInt64(b, {3});
        nert::TensorView vo = HostInt64(cpuOut, {3});
        nert::CpuDevice().kernels().find("Sub")(node, {va, vb}, {vo});
    }
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{2, -3, 7}));

    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    std::vector<int64_t> gpuOut(3, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Sub")(node, {DeviceInt64(da, {3}), DeviceInt64(db, {3})},
                                             {DeviceInt64(dy, {3})});
    gpuOut = DownloadInt64(cuda, dy, 3);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Div float follows IEEE: 1/0 = inf, -1/0 = -inf, 0/0 = nan — like the oracle.
TEST(NertCudaKernels, DivFloatIEEE) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, -1.0F, 0.0F, 5.0F};
    std::vector<float> b = {0.0F, 0.0F, 0.0F, 2.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("div0", "Div");
    nert::CpuDevice().kernels().find("Div")(node, {HostFloat(a, {4}), HostFloat(b, {4})}, {HostFloat(cpuOut, {4})});
    EXPECT_EQ(cpuOut[3], 2.5F);
    EXPECT_TRUE(std::isinf(cpuOut[0]) && cpuOut[0] > 0);
    EXPECT_TRUE(std::isinf(cpuOut[1]) && cpuOut[1] < 0);
    EXPECT_TRUE(std::isnan(cpuOut[2]));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Div")(node, {DeviceFloat(da, {4}), DeviceFloat(db, {4})},
                                             {DeviceFloat(dy, {4})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ASSERT_EQ(gpuOut.size(), cpuOut.size());
    EXPECT_EQ(gpuOut[3], cpuOut[3]);
    EXPECT_TRUE(std::isinf(gpuOut[0]) && gpuOut[0] > 0);
    EXPECT_TRUE(std::isinf(gpuOut[1]) && gpuOut[1] < 0);
    EXPECT_TRUE(std::isnan(gpuOut[2]));
}

// Div int64 with a zero divisor throws naming node + op, like the oracle.
TEST(NertCudaKernels, DivInt64ZeroThrows) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {2, 0};
    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    std::vector<int64_t> y(2, 0);
    void* dy = UploadInt64(cuda, y);
    nert::Node node = MakeNode("div_bad", "Div");
    try {
        nert::CudaDevice().kernels().find("Div")(node, {DeviceInt64(da, {2}), DeviceInt64(db, {2})},
                                                 {DeviceInt64(dy, {2})});
        FAIL() << "expected InferenceException for int64 division by zero";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("div_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Div"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// MatMul broadcasts a {2,3} A over a {2,3,2} B: two hand-computed batches.
TEST(NertCudaKernels, MatMulBatchedBroadcast) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

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

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    std::vector<float> gpuOut(8, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("MatMul")(node, {DeviceFloat(da, {2, 3}), DeviceFloat(db, {2, 3, 2})},
                                                {DeviceFloat(dy, {2, 2, 2})});
    gpuOut = Download(cuda, dy, 8);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, expected);
}

// MatMul promotes a 1-D A: [1,2,3] x [[1,2],[3,4],[5,6]] = [22,28].
TEST(NertCudaKernels, MatMulPromotedVector) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F};
    std::vector<float> b = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("mm1", "MatMul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("MatMul")(node, {HostFloat(a, {3}), HostFloat(b, {3, 2})},
                                               {HostFloat(cpuOut, {2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{22.0F, 28.0F}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("MatMul")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {3, 2})},
                                                {DeviceFloat(dy, {2})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// MatMul rejects a contraction mismatch, naming node + op.
TEST(NertCudaKernels, MatMulRejectsMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a(6, 1.0F);
    std::vector<float> b(6, 1.0F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("mm_bad", "MatMul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("MatMul")(node, {DeviceFloat(da, {2, 3}), DeviceFloat(db, {2, 2})},
                                                    {DeviceFloat(dy, {2, 2})});
        FAIL() << "expected InferenceException for a contraction mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mm_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("MatMul"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}
