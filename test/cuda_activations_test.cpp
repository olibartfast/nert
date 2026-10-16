// CUDA activations (N1e-3): Sigmoid/Tanh/Softmax vs the CPU oracle.
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
