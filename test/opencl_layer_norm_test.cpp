// OpenCL LayerNormalization vs the CPU oracle: axis handling, epsilon,
// Scale/B (per-element and scalar), and negative cases.
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

void ExpectNearOcl(const std::vector<float>& got, const std::vector<float>& ref) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], 1e-3 * (1.0 + std::fabs(ref[i]))) << "i=" << i;
    }
}

// Runs one case on the CPU oracle (must match `expected`) and on OpenCL
// (must match the oracle). Scale/B may be per-element or scalar.
void RunCase(const std::string& name, int64_t axis, float epsilon, const std::vector<float>& x,
             const std::vector<int64_t>& xDims, const std::vector<float>& scale, const std::vector<int64_t>& scaleDims,
             bool hasBias, const std::vector<float>& bias, const std::vector<int64_t>& biasDims,
             const std::vector<float>& expected) {
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode(name, "LayerNormalization");
    node.attributes.emplace_back("axis", axis);
    node.attributes.emplace_back("epsilon", epsilon);
    node.inputs = {"x", "scale"};
    if (hasBias) {
        node.inputs.push_back("bias");
    }
    node.outputs = {"y"};

    std::vector<float> xv = x;
    std::vector<float> sv = scale;
    std::vector<float> bv = bias;
    std::vector<float> cpuOut(x.size(), 0.0F);
    std::vector<nert::TensorView> cpuInputs = {HostFloat(xv, xDims), HostFloat(sv, scaleDims)};
    if (hasBias) {
        cpuInputs.push_back(HostFloat(bv, biasDims));
    }
    nert::CpuDevice().kernels().find("LayerNormalization")(node, cpuInputs, {HostFloat(cpuOut, xDims)});
    ExpectNearOcl(cpuOut, expected);

    void* dx = Upload(ocl, x);
    void* ds = Upload(ocl, scale);
    void* db = hasBias ? Upload(ocl, bias) : nullptr;
    std::vector<float> gpuOut(x.size(), 0.0F);
    void* dy = Upload(ocl, gpuOut);
    std::vector<nert::TensorView> devInputs = {DeviceFloat(dx, xDims), DeviceFloat(ds, scaleDims)};
    if (hasBias) {
        devInputs.push_back(DeviceFloat(db, biasDims));
    }
    nert::OpenCLDevice().kernels().find("LayerNormalization")(node, devInputs, {DeviceFloat(dy, xDims)});
    gpuOut = Download(ocl, dy, x.size());
    ocl.allocator().release(dx);
    ocl.allocator().release(ds);
    if (db != nullptr) {
        ocl.allocator().release(db);
    }
    ocl.allocator().release(dy);
    ExpectNearOcl(gpuOut, cpuOut);
}

// 2x3, axis=-1, per-element Scale and B: invStd = 1/sqrt(2/3 + 1e-5).
TEST(NertOpenCLLayerNorm, ScaleBiasExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunCase("ln0", -1, 1e-5F, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3}, {1.0F, 2.0F, 3.0F}, {3}, true,
            {0.5F, 0.5F, 0.5F}, {3}, {-0.7247357F, 0.5F, 4.174207F, -0.7247357F, 0.5F, 4.174207F});
}

// axis=1 is the same normalized axis for rank 2; scalar Scale means unit scale.
TEST(NertOpenCLLayerNorm, AxisOneScalarScale) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunCase("ln1", 1, 1e-5F, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3}, {1.0F}, {1}, false, {}, {},
            {-1.2247357F, 0.0F, 1.2247357F, -1.2247357F, 0.0F, 1.2247357F});
}

// Off-default epsilon: invStd = 1/sqrt(2/3 + 1e-3).
TEST(NertOpenCLLayerNorm, EpsilonNonDefault) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    const float inv = 1.0F / std::sqrt(2.0F / 3.0F + 1e-3F);
    RunCase("ln_eps", -1, 1e-3F, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3}, {1.0F, 1.0F, 1.0F}, {3}, false, {}, {},
            {-inv, 0.0F, inv, -inv, 0.0F, inv});
}

// The kernel rejects an out-of-range axis and a non-broadcastable Scale,
// naming node + op.
TEST(NertOpenCLLayerNorm, RejectsBadAxisAndScale) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    nert::Node badAxis = MakeNode("ln_bad", "LayerNormalization");
    badAxis.attributes.emplace_back("axis", int64_t{2});
    badAxis.inputs = {"x", "scale"};
    badAxis.outputs = {"y"};
    try {
        nert::OpenCLDevice().kernels().find("LayerNormalization")(
            badAxis, {DeviceFloat(nullptr, {2, 3}), DeviceFloat(nullptr, {3})}, {DeviceFloat(nullptr, {2, 3})});
        FAIL() << "expected InferenceException for an out-of-range axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ln_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("LayerNormalization"), std::string::npos) << message;
    }

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> badScale = {1.0F, 2.0F};
    std::vector<float> y(6, 0.0F);
    void* dx = Upload(ocl, x);
    void* ds = Upload(ocl, badScale);
    void* dy = Upload(ocl, y);
    nert::Node badScaleNode = MakeNode("ln_bad", "LayerNormalization");
    badScaleNode.attributes.emplace_back("axis", int64_t{-1});
    badScaleNode.inputs = {"x", "scale"};
    badScaleNode.outputs = {"y"};
    try {
        nert::OpenCLDevice().kernels().find("LayerNormalization")(
            badScaleNode, {DeviceFloat(dx, {2, 3}), DeviceFloat(ds, {2})}, {DeviceFloat(dy, {2, 3})});
        FAIL() << "expected InferenceException for a non-broadcastable Scale";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ln_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("LayerNormalization"), std::string::npos) << message;
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(ds);
    ocl.allocator().release(dy);
}

} // namespace
