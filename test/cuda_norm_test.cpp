// CUDA LayerNormalization (R1): norm corridor vs the CPU oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

// Runs LayerNormalization on CUDA with X/Scale and optional B + stats.
// Returns Y; fills mean/invstd when the corresponding output is requested.
std::vector<float> RunCudaLN(nert::Device& cuda, nert::Node node, std::vector<float>& x, std::vector<int64_t> xd,
                             std::vector<float>& scale, std::vector<int64_t> sd, std::vector<float>* bias,
                             std::vector<float>* mean, std::vector<float>* invstd) {
    node.inputs =
        bias != nullptr ? std::vector<std::string>{"x", "scale", "b"} : std::vector<std::string>{"x", "scale"};
    node.outputs = {"y"};
    if (mean != nullptr) {
        node.outputs.push_back("mean");
    }
    if (invstd != nullptr) {
        node.outputs.push_back("invstd");
    }
    void* dx = Upload(cuda, x);
    void* ds = Upload(cuda, scale);
    void* db = bias != nullptr ? Upload(cuda, *bias) : nullptr;
    std::vector<float> y(x.size(), 0.0F);
    void* dy = Upload(cuda, y);
    void* dm = nullptr;
    void* dv = nullptr;
    if (mean != nullptr) {
        dm = Upload(cuda, *mean);
    }
    if (invstd != nullptr) {
        dv = Upload(cuda, *invstd);
    }
    std::vector<nert::TensorView> inputs = {DeviceFloat(dx, xd), DeviceFloat(ds, sd)};
    if (bias != nullptr) {
        inputs.push_back(DeviceFloat(db, sd));
    }
    std::vector<nert::TensorView> outputs = {DeviceFloat(dy, xd)};
    if (mean != nullptr) {
        outputs.push_back(DeviceFloat(dm, {xd[0]}));
    }
    if (invstd != nullptr) {
        outputs.push_back(DeviceFloat(dv, {xd[0]}));
    }
    nert::CudaDevice().kernels().find("LayerNormalization")(node, inputs, outputs);
    y = Download(cuda, dy, y.size());
    if (mean != nullptr) {
        *mean = Download(cuda, dm, mean->size());
    }
    if (invstd != nullptr) {
        *invstd = Download(cuda, dv, invstd->size());
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    if (db != nullptr) {
        cuda.allocator().release(db);
    }
    cuda.allocator().release(dy);
    if (dm != nullptr) {
        cuda.allocator().release(dm);
    }
    if (dv != nullptr) {
        cuda.allocator().release(dv);
    }
    return y;
}

std::vector<float> RunCpuLN(nert::Node node, std::vector<float>& x, std::vector<int64_t> xd, std::vector<float>& scale,
                            std::vector<int64_t> sd, std::vector<float>* bias, std::vector<float>* mean,
                            std::vector<float>* invstd) {
    node.inputs =
        bias != nullptr ? std::vector<std::string>{"x", "scale", "b"} : std::vector<std::string>{"x", "scale"};
    node.outputs = {"y"};
    if (mean != nullptr) {
        node.outputs.push_back("mean");
    }
    if (invstd != nullptr) {
        node.outputs.push_back("invstd");
    }
    std::vector<float> y(x.size(), 0.0F);
    std::vector<nert::TensorView> inputs = {HostFloat(x, xd), HostFloat(scale, sd)};
    if (bias != nullptr) {
        inputs.push_back(HostFloat(*bias, sd));
    }
    std::vector<nert::TensorView> outputs = {HostFloat(y, xd)};
    if (mean != nullptr) {
        outputs.push_back(HostFloat(*mean, {xd[0]}));
    }
    if (invstd != nullptr) {
        outputs.push_back(HostFloat(*invstd, {xd[0]}));
    }
    nert::CpuDevice().kernels().find("LayerNormalization")(node, inputs, outputs);
    return y;
}

} // namespace

// Default axis (-1) over [2,3] with unit scale and no B: each row is
// (x - mean) / sqrt(var + eps); row {1,2,3} has mean 2, var 2/3.
TEST(NertCudaNorm, DefaultAxisExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> scale = {1.0F, 1.0F, 1.0F};
    nert::Node node = MakeNode("ln0", "LayerNormalization");
    const std::vector<float> expected = {-1.2247357F, 0.0F, 1.2247357F, -1.2247357F, 0.0F, 1.2247357F};
    ExpectNear(RunCpuLN(node, x, {2, 3}, scale, {3}, nullptr, nullptr, nullptr), expected);
    ExpectNear(RunCudaLN(cuda, node, x, {2, 3}, scale, {3}, nullptr, nullptr, nullptr), expected);
}

// axis=1 with per-element scale/B plus Mean/InvStdDev outputs.
TEST(NertCudaNorm, ScaleBiasStatsExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 2.0F, 1.0F, 3.0F};
    std::vector<float> scale = {2.0F, 0.5F};
    std::vector<float> bias = {1.0F, -1.0F};
    nert::Node node = MakeNode("ln1", "LayerNormalization");
    node.attributes.emplace_back("axis", int64_t{1});
    const std::vector<float> expectedY = {-0.99999F, -0.5000025F, -0.99999F, -0.5000025F};
    const std::vector<float> expectedMean = {1.0F, 2.0F};
    const std::vector<float> expectedInvStd = {0.999995F, 0.999995F};
    std::vector<float> cpuMean(2, 0.0F);
    std::vector<float> cpuInv(2, 0.0F);
    ExpectNear(RunCpuLN(node, x, {2, 2}, scale, {2}, &bias, &cpuMean, &cpuInv), expectedY);
    ExpectNear(cpuMean, expectedMean);
    ExpectNear(cpuInv, expectedInvStd);
    std::vector<float> gpuMean(2, 0.0F);
    std::vector<float> gpuInv(2, 0.0F);
    ExpectNear(RunCudaLN(cuda, node, x, {2, 2}, scale, {2}, &bias, &gpuMean, &gpuInv), expectedY);
    ExpectNear(gpuMean, expectedMean);
    ExpectNear(gpuInv, expectedInvStd);
}

// axis=0 with scalar scale: single row of 6 against the oracle.
TEST(NertCudaNorm, AxisZeroScalarScale) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> scale = {3.0F};
    nert::Node node = MakeNode("ln2", "LayerNormalization");
    node.attributes.emplace_back("axis", int64_t{0});
    const std::vector<float> cpu = RunCpuLN(node, x, {2, 3}, scale, {1}, nullptr, nullptr, nullptr);
    ExpectNear(RunCudaLN(cuda, node, x, {2, 3}, scale, {1}, nullptr, nullptr, nullptr), cpu);
    // The normalized row sums to ~0 before the scalar scale.
    double sum = 0.0;
    for (float v : cpu) {
        sum += v;
    }
    EXPECT_NEAR(sum, 0.0, 1e-3);
}

// LayerNormalization rejects an out-of-range axis, naming node + op.
TEST(NertCudaNorm, RejectsBadAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> scale(3, 1.0F);
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("ln_bad", "LayerNormalization");
    node.attributes.emplace_back("axis", int64_t{5});
    node.inputs = {"x", "scale"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* ds = Upload(cuda, scale);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("LayerNormalization")(node, {DeviceFloat(dx, {2, 3}), DeviceFloat(ds, {3})},
                                                                {DeviceFloat(dy, {2, 3})});
        FAIL() << "expected InferenceException for an out-of-range axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ln_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("LayerNormalization"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
}

// LayerNormalization rejects a wrong-sized Scale, naming node + op.
TEST(NertCudaNorm, RejectsBadScaleSize) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> scale(2, 1.0F);
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("ln_bad_scale", "LayerNormalization");
    node.inputs = {"x", "scale"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* ds = Upload(cuda, scale);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("LayerNormalization")(node, {DeviceFloat(dx, {2, 3}), DeviceFloat(ds, {2})},
                                                                {DeviceFloat(dy, {2, 3})});
        FAIL() << "expected InferenceException for a wrong-sized Scale";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ln_bad_scale"), std::string::npos) << message;
        EXPECT_NE(message.find("LayerNormalization"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
}

// LayerNormalization rejects a Y shape mismatch, naming node + op.
TEST(NertCudaNorm, RejectsYShapeMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> scale(3, 1.0F);
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("ln_bad_y", "LayerNormalization");
    node.inputs = {"x", "scale"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* ds = Upload(cuda, scale);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("LayerNormalization")(node, {DeviceFloat(dx, {2, 3}), DeviceFloat(ds, {3})},
                                                                {DeviceFloat(dy, {3, 2})});
        FAIL() << "expected InferenceException for a Y shape mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ln_bad_y"), std::string::npos) << message;
        EXPECT_NE(message.find("LayerNormalization"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
}
