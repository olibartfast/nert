// OpenCL MatMul: matrix product with 1-D promotion and NumPy batch
// broadcast vs the CPU oracle.
//
// Needs an OpenCL GPU platform at run time; without one every test SKIPs
// (never fails).

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using nert_test::DeviceFloat;
using nert_test::Download;
using nert_test::HostFloat;
using nert_test::MakeNode;
using nert_test::Upload;

bool HasOpenCLDevice() { return nert_test::DeviceAvailable(&nert::OpenCLDevice); }

std::size_t Count(const std::vector<int64_t>& dims) {
    std::size_t count = 1;
    for (int64_t dim : dims) {
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

// The OpenCL kernel accumulates in float while the oracle accumulates in
// double, so the device result meets the oracle within the tolerance bar
// rather than bit for bit (the integer fixtures below are exact on both).
void ExpectNearOcl(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3 * (1.0 + std::fabs(ref[i]))) << "i=" << i;
    }
}

// Runs MatMul on both tables: the oracle must reproduce `expected` exactly,
// and the OpenCL result must match the oracle within tolerance.
void RunMatMul(const std::string& name, const std::vector<float>& a, const std::vector<int64_t>& aDims,
               const std::vector<float>& b, const std::vector<int64_t>& bDims, const std::vector<int64_t>& outDims,
               const std::vector<float>& expected) {
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode(name, "MatMul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    const std::size_t count = Count(outDims);

    std::vector<float> cpuOut(count, 0.0F);
    nert::CpuDevice().kernels().find("MatMul")(
        node,
        {HostFloat(const_cast<std::vector<float>&>(a), aDims), HostFloat(const_cast<std::vector<float>&>(b), bDims)},
        {HostFloat(cpuOut, outDims)});
    EXPECT_EQ(cpuOut, expected);

    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    std::vector<float> gpuOut(count, 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("MatMul")(node, {DeviceFloat(da, aDims), DeviceFloat(db, bDims)},
                                                  {DeviceFloat(dy, outDims)});
    gpuOut = Download(ocl, dy, count);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
    ExpectNearOcl(gpuOut, cpuOut);
}

// Asserts the OpenCL kernel rejects the shapes, naming node + op.
void ExpectMatMulReject(const std::string& name, const std::vector<int64_t>& aDims, const std::vector<int64_t>& bDims,
                        const std::vector<int64_t>& outDims) {
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode(name, "MatMul");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    const std::vector<float> a(Count(aDims), 1.0F);
    const std::vector<float> b(Count(bDims), 1.0F);
    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    std::vector<float> gpuOut(Count(outDims), 0.0F);
    void* dy = Upload(ocl, gpuOut);
    try {
        nert::OpenCLDevice().kernels().find("MatMul")(node, {DeviceFloat(da, aDims), DeviceFloat(db, bDims)},
                                                      {DeviceFloat(dy, outDims)});
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(name), std::string::npos) << message;
        EXPECT_NE(message.find("MatMul"), std::string::npos) << message;
    }
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
}

} // namespace

// A [2,3] * B [3,2], both 1..6: hand-computed {22,28,49,64}.
TEST(NertOpenCLMatMul, TwoByThreeTimesThreeByTwoExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunMatMul("mm0", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {3, 2}, {2, 2},
              {22.0F, 28.0F, 49.0F, 64.0F});
}

// A [3] . B [3] promotes to [1,3]*[3,1] and drops both promoted axes,
// leaving a scalar output shape {}: {1,2,3}.{4,5,6} = 32.
TEST(NertOpenCLMatMul, DotProduct1DExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunMatMul("mm1", {1.0F, 2.0F, 3.0F}, {3}, {4.0F, 5.0F, 6.0F}, {3}, {}, {32.0F});
}

// A [3] * B [3,2] promotes A to a row vector -> [2]; A [2,3] * B [3]
// promotes B to a column vector -> [2].
TEST(NertOpenCLMatMul, Promote1DOperands) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunMatMul("mm2", {1.0F, 2.0F, 3.0F}, {3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {3, 2}, {2}, {22.0F, 28.0F});
    RunMatMul("mm3", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3}, {1.0F, 2.0F, 3.0F}, {3}, {2}, {14.0F, 32.0F});
}

// Batched broadcast: A [2,1,2,3] carries a size-1 batch axis against B
// [3,2] with no batch axes -> [2,1,2,2]; A [2,2,3] against B [3,2]
// broadcasts one missing leading batch axis -> [2,2,2].
TEST(NertOpenCLMatMul, BatchedBroadcastExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunMatMul("mm4", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}, {2, 1, 2, 3},
              {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {3, 2}, {2, 1, 2, 2},
              {22.0F, 28.0F, 49.0F, 64.0F, 76.0F, 100.0F, 103.0F, 136.0F});
    RunMatMul("mm5", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}, {2, 2, 3},
              {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {3, 2}, {2, 2, 2},
              {22.0F, 28.0F, 49.0F, 64.0F, 76.0F, 100.0F, 103.0F, 136.0F});
}

// Contraction mismatch (K=3 vs K=4) is rejected, naming node + op.
TEST(NertOpenCLMatMul, RejectsContractionMismatch) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    ExpectMatMulReject("mm_k_bad", {2, 3}, {4, 2}, {2, 2});
}

// Batch dims [3] vs [2] are not broadcast-compatible.
TEST(NertOpenCLMatMul, RejectsBatchBroadcastMismatch) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    ExpectMatMulReject("mm_batch_bad", {3, 2, 3}, {2, 3, 2}, {2, 2, 2});
}

// A correct product written into a wrongly shaped output is rejected.
TEST(NertOpenCLMatMul, RejectsWrongOutputShape) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    ExpectMatMulReject("mm_shape_bad", {2, 3}, {3, 2}, {2, 3});
}
