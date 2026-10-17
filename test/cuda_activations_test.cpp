// CUDA activations (N1e-3): Sigmoid/Tanh/Softmax vs the CPU oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {

std::vector<float> RunCuda(nert::Device& cuda, const std::string& op, nert::Node node, std::vector<float>& x,
                           std::vector<int64_t> xd) {
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    std::vector<float> out(x.size(), 0.0F);
    void* dy = Upload(cuda, out);
    nert::CudaDevice().kernels().find(op)(node, {DeviceFloat(dx, xd)}, {DeviceFloat(dy, xd)});
    out = Download(cuda, dy, out.size());
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    return out;
}

std::vector<float> RunCpu(const std::string& op, nert::Node node, std::vector<float>& x, std::vector<int64_t> xd) {
    node.inputs = {"x"};
    node.outputs = {"y"};
    std::vector<float> out(x.size(), 0.0F);
    nert::CpuDevice().kernels().find(op)(node, {HostFloat(x, xd)}, {HostFloat(out, xd)});
    return out;
}

} // namespace

// Sigmoid over {0, 1, -1, 2}: hand-computed 1/(1+exp(-x)).
TEST(NertCudaActivations, SigmoidExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 1.0F, -1.0F, 2.0F};
    nert::Node node = MakeNode("sig0", "Sigmoid");
    const std::vector<float> expected = {0.5F, 0.73105858F, 0.26894142F, 0.88079708F};
    ExpectNear(RunCpu("Sigmoid", node, x, {4}), expected);
    ExpectNear(RunCuda(cuda, "Sigmoid", node, x, {4}), expected);
}

// Tanh over {0, 1, -1}: hand-computed.
TEST(NertCudaActivations, TanhExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 1.0F, -1.0F};
    nert::Node node = MakeNode("tanh0", "Tanh");
    const std::vector<float> expected = {0.0F, 0.76159416F, -0.76159416F};
    ExpectNear(RunCpu("Tanh", node, x, {3}), expected);
    ExpectNear(RunCuda(cuda, "Tanh", node, x, {3}), expected);
}

// Softmax default axis (-1) over [2,3]: each row is softmax([1,2,3]).
TEST(NertCudaActivations, SoftmaxLastAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 1.0F, 2.0F, 3.0F};
    nert::Node node = MakeNode("sm0", "Softmax");
    const std::vector<float> expected = {0.09003057F, 0.24472848F, 0.66524094F, 0.09003057F, 0.24472848F, 0.66524094F};
    ExpectNear(RunCpu("Softmax", node, x, {2, 3}), expected);
    ExpectNear(RunCuda(cuda, "Softmax", node, x, {2, 3}), expected);
}

// Softmax axis=0 over [3,2] with large values: stability via max subtraction.
// Column j is softmax([1000+j, 1001+j, 1002+j]) = same as softmax([0,1,2]).
TEST(NertCudaActivations, SoftmaxAxisZeroStable) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1000.0F, 1000.0F, 1001.0F, 1001.0F, 1002.0F, 1002.0F};
    nert::Node node = MakeNode("sm1", "Softmax");
    node.attributes.emplace_back("axis", int64_t{0});
    const std::vector<float> expected = {0.09003057F, 0.09003057F, 0.24472848F, 0.24472848F, 0.66524094F, 0.66524094F};
    ExpectNear(RunCpu("Softmax", node, x, {3, 2}), expected);
    ExpectNear(RunCuda(cuda, "Softmax", node, x, {3, 2}), expected);
    // Row sums of the CUDA result are 1 within tolerance.
    const std::vector<float> gpu = RunCuda(cuda, "Softmax", node, x, {3, 2});
    for (int c = 0; c < 2; ++c) {
        EXPECT_NEAR(gpu[0 * 2 + c] + gpu[1 * 2 + c] + gpu[2 * 2 + c], 1.0F, 1e-5) << "col " << c;
    }
}

// Softmax rejects an out-of-range axis, naming node + op.
TEST(NertCudaActivations, SoftmaxRejectsBadAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("sm_bad", "Softmax");
    node.attributes.emplace_back("axis", int64_t{5});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {2, 3});
    nert::TensorView yi = DeviceFloat(dy, {2, 3});
    try {
        nert::CudaDevice().kernels().find("Softmax")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for an out-of-range axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sm_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Softmax"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Sigmoid rejects an element-count mismatch, naming node + op.
TEST(NertCudaActivations, SigmoidRejectsCountMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(3, 0.5F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("sig_bad", "Sigmoid");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {3});
    nert::TensorView yi = DeviceFloat(dy, {4});
    try {
        nert::CudaDevice().kernels().find("Sigmoid")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for an element-count mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sig_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Sigmoid"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Sqrt over {0, 1, 4, 9, -1, 0}: hand-computed, NaN for the negative input, exact zeros.
TEST(NertCudaActivations, SqrtExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 1.0F, 4.0F, 9.0F, -1.0F, 0.0F};
    nert::Node node = MakeNode("sqrt0", "Sqrt");
    const std::vector<float> expected = {0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 0.0F};
    const std::vector<float> cpu = RunCpu("Sqrt", node, x, {6});
    const std::vector<float> gpu = RunCuda(cuda, "Sqrt", node, x, {6});
    ASSERT_EQ(cpu.size(), expected.size());
    ASSERT_EQ(gpu.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (i == 4) {
            EXPECT_TRUE(std::isnan(cpu[i])) << "index " << i;
            EXPECT_TRUE(std::isnan(gpu[i])) << "index " << i;
        } else {
            EXPECT_NEAR(cpu[i], expected[i], 1e-3 * (1.0 + std::fabs(expected[i]))) << "index " << i;
            EXPECT_NEAR(gpu[i], expected[i], 1e-3 * (1.0 + std::fabs(expected[i]))) << "index " << i;
        }
    }
}

// Erf over {0, 1, -1, 0.5}: hand-computed, erf(0)=0 with odd symmetry.
TEST(NertCudaActivations, ErfExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 1.0F, -1.0F, 0.5F};
    nert::Node node = MakeNode("erf0", "Erf");
    const std::vector<float> expected = {0.0F, 0.84270079F, -0.84270079F, 0.52049988F};
    ExpectNear(RunCpu("Erf", node, x, {4}), expected);
    ExpectNear(RunCuda(cuda, "Erf", node, x, {4}), expected);
    // Odd symmetry erf(-x) = -erf(x) on both backends.
    const std::vector<float> cpu = RunCpu("Erf", node, x, {4});
    const std::vector<float> gpu = RunCuda(cuda, "Erf", node, x, {4});
    EXPECT_NEAR(cpu[1] + cpu[2], 0.0F, 1e-5) << "cpu symmetry";
    EXPECT_NEAR(gpu[1] + gpu[2], 0.0F, 1e-5) << "gpu symmetry";
}

// Sqrt rejects an element-count mismatch, naming node + op.
TEST(NertCudaActivations, SqrtRejectsCountMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(3, 4.0F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("sqrt_bad", "Sqrt");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {3});
    nert::TensorView yi = DeviceFloat(dy, {4});
    try {
        nert::CudaDevice().kernels().find("Sqrt")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for an element-count mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sqrt_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Sqrt"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Erf rejects non-float32 tensors, naming node + op.
TEST(NertCudaActivations, ErfRejectsNonFloat) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x(3, 1);
    std::vector<int64_t> y(3, 0);
    nert::Node node = MakeNode("erf_bad", "Erf");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = UploadInt64(cuda, x);
    void* dy = UploadInt64(cuda, y);
    nert::TensorView xi = DeviceInt64(dx, {3});
    nert::TensorView yi = DeviceInt64(dy, {3});
    try {
        nert::CudaDevice().kernels().find("Erf")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for non-float32 tensors";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("erf_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Erf"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Sin over {0, pi/2, pi, -pi/2}: hand-computed.
TEST(NertCudaActivations, SinExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    const float kPi = 3.14159274F;
    std::vector<float> x = {0.0F, kPi / 2.0F, kPi, -kPi / 2.0F};
    nert::Node node = MakeNode("sin0", "Sin");
    const std::vector<float> expected = {0.0F, 1.0F, 0.0F, -1.0F};
    ExpectNear(RunCpu("Sin", node, x, {4}), expected);
    ExpectNear(RunCuda(cuda, "Sin", node, x, {4}), expected);
}

// Cos over {0, pi/2, pi}: hand-computed.
TEST(NertCudaActivations, CosExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    const float kPi = 3.14159274F;
    std::vector<float> x = {0.0F, kPi / 2.0F, kPi};
    nert::Node node = MakeNode("cos0", "Cos");
    const std::vector<float> expected = {1.0F, 0.0F, -1.0F};
    ExpectNear(RunCpu("Cos", node, x, {3}), expected);
    ExpectNear(RunCuda(cuda, "Cos", node, x, {3}), expected);
}

// Exp over {0, 1, -1}: hand-computed.
TEST(NertCudaActivations, ExpExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 1.0F, -1.0F};
    nert::Node node = MakeNode("exp0", "Exp");
    const std::vector<float> expected = {1.0F, 2.71828183F, 0.36787944F};
    ExpectNear(RunCpu("Exp", node, x, {3}), expected);
    ExpectNear(RunCuda(cuda, "Exp", node, x, {3}), expected);
}

// Elu with default alpha=1 over {1, -1, 0}, plus alpha=2 over {-1}.
TEST(NertCudaActivations, EluExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, -1.0F, 0.0F};
    nert::Node node = MakeNode("elu0", "Elu");
    const std::vector<float> expected = {1.0F, -0.63212056F, 0.0F};
    ExpectNear(RunCpu("Elu", node, x, {3}), expected);
    ExpectNear(RunCuda(cuda, "Elu", node, x, {3}), expected);

    std::vector<float> xa = {-1.0F};
    nert::Node nodeA = MakeNode("elu1", "Elu");
    nodeA.attributes.emplace_back("alpha", 2.0F);
    const std::vector<float> expectedA = {-1.26424112F};
    ExpectNear(RunCpu("Elu", nodeA, xa, {1}), expectedA);
    ExpectNear(RunCuda(cuda, "Elu", nodeA, xa, {1}), expectedA);
}

// Log over {1, e, e^2}: hand-computed.
TEST(NertCudaActivations, LogExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.71828183F, 7.38905610F};
    nert::Node node = MakeNode("log0", "Log");
    const std::vector<float> expected = {0.0F, 1.0F, 2.0F};
    ExpectNear(RunCpu("Log", node, x, {3}), expected);
    ExpectNear(RunCuda(cuda, "Log", node, x, {3}), expected);
}

// Floor over {1.5, -1.5, 2.0}: hand-computed.
TEST(NertCudaActivations, FloorExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.5F, -1.5F, 2.0F};
    nert::Node node = MakeNode("floor0", "Floor");
    const std::vector<float> expected = {1.0F, -2.0F, 2.0F};
    ExpectNear(RunCpu("Floor", node, x, {3}), expected);
    ExpectNear(RunCuda(cuda, "Floor", node, x, {3}), expected);
}

// IsNaN over {0, NaN, 1}: bool out, both tables.
TEST(NertCudaActivations, IsNaNExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x = {0.0F, kNaN, 1.0F};
    std::vector<std::uint8_t> cpuOut(3, 0);
    nert::Node node = MakeNode("isnan0", "IsNaN");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::TensorView cpuYi;
    cpuYi.data = cpuOut.data();
    cpuYi.dtype = nert::DataType::Bool;
    cpuYi.dims = {3};
    nert::CpuDevice().kernels().find("IsNaN")(node, {HostFloat(x, {3})}, {cpuYi});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{0, 1, 0}));

    void* dx = Upload(cuda, x);
    void* dy = cuda.allocator().allocate(3);
    nert::TensorView deviceYi;
    deviceYi.data = dy;
    deviceYi.dtype = nert::DataType::Bool;
    deviceYi.dims = {3};
    nert::CudaDevice().kernels().find("IsNaN")(node, {DeviceFloat(dx, {3})}, {deviceYi});
    std::vector<std::uint8_t> gpuOut(3, 0);
    cuda.transfer().device_to_host(gpuOut.data(), dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Sin rejects an element-count mismatch, naming node + op.
TEST(NertCudaActivations, SinRejectsCountMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(3, 0.5F);
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("sin_bad", "Sin");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {3});
    nert::TensorView yi = DeviceFloat(dy, {4});
    try {
        nert::CudaDevice().kernels().find("Sin")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for an element-count mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sin_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Sin"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Elu rejects a non-float alpha attribute, naming node + op.
TEST(NertCudaActivations, EluRejectsBadAlpha) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(2, -1.0F);
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("elu_bad", "Elu");
    node.attributes.emplace_back("alpha", int64_t{2});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    nert::TensorView xi = DeviceFloat(dx, {2});
    nert::TensorView yi = DeviceFloat(dy, {2});
    try {
        nert::CudaDevice().kernels().find("Elu")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for a non-float alpha";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("elu_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Elu"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// IsNaN rejects non-float32 input, naming node + op.
TEST(NertCudaActivations, IsNaNRejectsNonFloat) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x(3, 1);
    std::vector<std::uint8_t> y(3, 0);
    nert::Node node = MakeNode("isnan_bad", "IsNaN");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = UploadInt64(cuda, x);
    void* dy = cuda.allocator().allocate(3);
    nert::TensorView xi = DeviceInt64(dx, {3});
    nert::TensorView yi;
    yi.data = dy;
    yi.dtype = nert::DataType::Bool;
    yi.dims = {3};
    try {
        nert::CudaDevice().kernels().find("IsNaN")(node, {xi}, {yi});
        FAIL() << "expected InferenceException for non-float32 input";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("isnan_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("IsNaN"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Abs over {-3, 0, 4} float and {-3, 0, 4} int64: hand-computed.
TEST(NertCudaActivations, AbsExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {-3.0F, 0.0F, 4.0F};
    nert::Node node = MakeNode("abs0", "Abs");
    const std::vector<float> expected = {3.0F, 0.0F, 4.0F};
    ExpectNear(RunCpu("Abs", node, x, {3}), expected);
    ExpectNear(RunCuda(cuda, "Abs", node, x, {3}), expected);

    std::vector<int64_t> xi = {-3, 0, 4};
    std::vector<int64_t> cpuI(3, 0);
    std::vector<int64_t> gpuI(3, 0);
    nert::Node nodeI = MakeNode("abs1", "Abs");
    nodeI.inputs = {"x"};
    nodeI.outputs = {"y"};
    nert::CpuDevice().kernels().find("Abs")(nodeI, {HostInt64(xi, {3})}, {HostInt64(cpuI, {3})});
    EXPECT_EQ(cpuI, (std::vector<int64_t>{3, 0, 4}));
    void* dx = UploadInt64(cuda, xi);
    void* dy = UploadInt64(cuda, gpuI);
    nert::CudaDevice().kernels().find("Abs")(nodeI, {DeviceInt64(dx, {3})}, {DeviceInt64(dy, {3})});
    gpuI = DownloadInt64(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuI, cpuI);
}

// Neg over {1.5, -2.0} float and {1, -2} int64: hand-computed.
TEST(NertCudaActivations, NegExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.5F, -2.0F};
    nert::Node node = MakeNode("neg0", "Neg");
    const std::vector<float> expected = {-1.5F, 2.0F};
    ExpectNear(RunCpu("Neg", node, x, {2}), expected);
    ExpectNear(RunCuda(cuda, "Neg", node, x, {2}), expected);

    std::vector<int64_t> xi = {1, -2};
    std::vector<int64_t> cpuI(2, 0);
    nert::Node nodeI = MakeNode("neg1", "Neg");
    nodeI.inputs = {"x"};
    nodeI.outputs = {"y"};
    nert::CpuDevice().kernels().find("Neg")(nodeI, {HostInt64(xi, {2})}, {HostInt64(cpuI, {2})});
    EXPECT_EQ(cpuI, (std::vector<int64_t>{-1, 2}));
    void* dx = UploadInt64(cuda, xi);
    std::vector<int64_t> gpuI(2, 0);
    void* dy = UploadInt64(cuda, gpuI);
    nert::CudaDevice().kernels().find("Neg")(nodeI, {DeviceInt64(dx, {2})}, {DeviceInt64(dy, {2})});
    gpuI = DownloadInt64(cuda, dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuI, cpuI);
}

// Pow {2,3}^2 with broadcast exponent: hand-computed.
TEST(NertCudaActivations, PowBroadcastExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {2.0F, 3.0F};
    std::vector<float> b = {2.0F};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("pow0", "Pow");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Pow")(node, {HostFloat(a, {2}), HostFloat(b, {1})}, {HostFloat(cpuOut, {2})});
    ExpectNear(cpuOut, (std::vector<float>{4.0F, 9.0F}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Pow")(node, {DeviceFloat(da, {2}), DeviceFloat(db, {1})},
                                             {DeviceFloat(dy, {2})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// Abs rejects dtype mismatch, naming node + op.
TEST(NertCudaActivations, AbsRejectsDtypeMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(3, -1.0F);
    std::vector<int64_t> y(3, 0);
    nert::Node node = MakeNode("abs_bad", "Abs");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = UploadInt64(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Abs")(node, {DeviceFloat(dx, {3})}, {DeviceInt64(dy, {3})});
        FAIL() << "expected InferenceException for a dtype mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("abs_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Abs"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Pow rejects non-broadcastable shapes, naming node + op.
TEST(NertCudaActivations, PowRejectsBroadcastMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a(3, 2.0F);
    std::vector<float> b(2, 2.0F);
    std::vector<float> y(3, 0.0F);
    nert::Node node = MakeNode("pow_bad", "Pow");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Pow")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {2})},
                                                 {DeviceFloat(dy, {3})});
        FAIL() << "expected InferenceException for non-broadcastable shapes";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("pow_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Pow"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}
