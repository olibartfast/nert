// CUDA GridSample (R5): sampler vs the CPU oracle.
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

std::vector<float> RunCudaGrid(nert::Device& cuda, nert::Node node, std::vector<float>& x, std::vector<int64_t> xd,
                               std::vector<float>& grid, std::vector<int64_t> gd) {
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dg = Upload(cuda, grid);
    const int64_t outCount = xd[0] * xd[1] * gd[1] * gd[2];
    std::vector<float> out(static_cast<std::size_t>(outCount), 0.0F);
    void* dy = Upload(cuda, out);
    nert::CudaDevice().kernels().find("GridSample")(node, {DeviceFloat(dx, xd), DeviceFloat(dg, gd)},
                                                    {DeviceFloat(dy, {xd[0], xd[1], gd[1], gd[2]})});
    out = Download(cuda, dy, out.size());
    cuda.allocator().release(dx);
    cuda.allocator().release(dg);
    cuda.allocator().release(dy);
    return out;
}

std::vector<float> RunCpuGrid(nert::Node node, std::vector<float>& x, std::vector<int64_t> xd, std::vector<float>& grid,
                              std::vector<int64_t> gd) {
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    const int64_t outCount = xd[0] * xd[1] * gd[1] * gd[2];
    std::vector<float> out(static_cast<std::size_t>(outCount), 0.0F);
    nert::CpuDevice().kernels().find("GridSample")(node, {HostFloat(x, xd), HostFloat(grid, gd)},
                                                   {HostFloat(out, {xd[0], xd[1], gd[1], gd[2]})});
    return out;
}

} // namespace

// Bilinear center of [[1,2],[3,4]] at grid (0,0), align 0: mean = 2.5.
TEST(NertCudaSample, BilinearCenterExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> grid = {0.0F, 0.0F};
    nert::Node node = MakeNode("gs0", "GridSample");
    const std::vector<float> expected = {2.5F};
    ExpectNear(RunCpuGrid(node, x, {1, 1, 2, 2}, grid, {1, 1, 1, 2}), expected);
    ExpectNear(RunCudaGrid(cuda, node, x, {1, 1, 2, 2}, grid, {1, 1, 1, 2}), expected);
}

// Bilinear corners with align_corners=1: exact pixel reads {1, 4}.
TEST(NertCudaSample, BilinearAlignCornersExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> grid = {-1.0F, -1.0F, 1.0F, 1.0F};
    nert::Node node = MakeNode("gs1", "GridSample");
    node.attributes.emplace_back("align_corners", int64_t{1});
    const std::vector<float> expected = {1.0F, 4.0F};
    ExpectNear(RunCpuGrid(node, x, {1, 1, 2, 2}, grid, {1, 1, 2, 2}), expected);
    ExpectNear(RunCudaGrid(cuda, node, x, {1, 1, 2, 2}, grid, {1, 1, 2, 2}), expected);
}

// Border clamps grid (-1,-1) to pixel (0,0) = 1.
TEST(NertCudaSample, BorderClampExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> grid = {-1.0F, -1.0F};
    nert::Node node = MakeNode("gs2", "GridSample");
    node.attributes.emplace_back("padding_mode", std::string("border"));
    const std::vector<float> expected = {1.0F};
    ExpectNear(RunCpuGrid(node, x, {1, 1, 2, 2}, grid, {1, 1, 1, 2}), expected);
    ExpectNear(RunCudaGrid(cuda, node, x, {1, 1, 2, 2}, grid, {1, 1, 1, 2}), expected);
}

// Nearest half-even: px=0.5 rounds to 0 (pixel 10), px=1.5 rounds to 2
// (out of range, zeros).
TEST(NertCudaSample, NearestHalfEvenExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {10.0F, 20.0F};
    std::vector<float> grid = {0.0F, 0.0F, 1.0F, 0.0F};
    nert::Node node = MakeNode("gs3", "GridSample");
    node.attributes.emplace_back("mode", std::string("nearest"));
    const std::vector<float> expected = {10.0F, 0.0F};
    ExpectNear(RunCpuGrid(node, x, {1, 1, 1, 2}, grid, {1, 2, 1, 2}), expected);
    ExpectNear(RunCudaGrid(cuda, node, x, {1, 1, 1, 2}, grid, {1, 2, 1, 2}), expected);
}

// Reflection matches the oracle on a 3x3 field with an out-of-range grid.
TEST(NertCudaSample, ReflectionMatchesOracle) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    std::vector<float> grid = {-1.5F, 0.5F, 0.25F, -0.75F};
    nert::Node node = MakeNode("gs4", "GridSample");
    node.attributes.emplace_back("padding_mode", std::string("reflection"));
    const std::vector<float> cpu = RunCpuGrid(node, x, {1, 1, 3, 3}, grid, {1, 2, 1, 2});
    ExpectNear(RunCudaGrid(cuda, node, x, {1, 1, 3, 3}, grid, {1, 2, 1, 2}), cpu);
}

// GridSample rejects an unsupported mode, naming node + op.
TEST(NertCudaSample, RejectsBadMode) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(4, 1.0F);
    std::vector<float> grid(2, 0.0F);
    std::vector<float> y(1, 0.0F);
    nert::Node node = MakeNode("gs_bad", "GridSample");
    node.attributes.emplace_back("mode", std::string("bicubic"));
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dg = Upload(cuda, grid);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("GridSample")(
            node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dg, {1, 1, 1, 2})}, {DeviceFloat(dy, {1, 1, 1, 1})});
        FAIL() << "expected InferenceException for an unsupported mode";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("gs_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("GridSample"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dg);
    cuda.allocator().release(dy);
}

// GridSample rejects a batch mismatch, naming node + op.
TEST(NertCudaSample, RejectsBatchMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(8, 1.0F);
    std::vector<float> grid(2, 0.0F);
    std::vector<float> y(1, 0.0F);
    nert::Node node = MakeNode("gs_bad_batch", "GridSample");
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dg = Upload(cuda, grid);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("GridSample")(
            node, {DeviceFloat(dx, {2, 1, 2, 2}), DeviceFloat(dg, {1, 1, 1, 2})}, {DeviceFloat(dy, {2, 1, 1, 1})});
        FAIL() << "expected InferenceException for a batch mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("gs_bad_batch"), std::string::npos) << message;
        EXPECT_NE(message.find("GridSample"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dg);
    cuda.allocator().release(dy);
}
