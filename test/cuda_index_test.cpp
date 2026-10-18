// CUDA index movement (N1g-3): Transpose/Gather/GatherElements vs the CPU
// oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Transpose {2,3} perm {1,0} -> {3,2}: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]].
TEST(NertCudaIndex, TransposeSwapExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("tr0", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{1, 0});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Transpose")(node, {HostFloat(x, {2, 3})}, {HostFloat(cpuOut, {3, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}));

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Transpose")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {3, 2})});
    gpuOut = Download(cuda, dy, 6);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Transpose default (reversed rank) over {2,2,2} iota, int64.
TEST(NertCudaIndex, TransposeDefaultInt64) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> cpuOut(8, 0);
    nert::Node node = MakeNode("tr1", "Transpose");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Transpose")(node, {HostInt64(x, {2, 2, 2})}, {HostInt64(cpuOut, {2, 2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{0, 4, 2, 6, 1, 5, 3, 7}));

    void* dx = UploadInt64(cuda, x);
    std::vector<int64_t> gpuOut(8, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Transpose")(node, {DeviceInt64(dx, {2, 2, 2})}, {DeviceInt64(dy, {2, 2, 2})});
    gpuOut = DownloadInt64(cuda, dy, 8);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Gather axis 1 over {2,4} with indices {0,2} -> {2,2} column picks.
TEST(NertCudaIndex, GatherExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<int64_t> idx = {0, 2};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("ga0", "Gather");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Gather")(node, {HostFloat(x, {2, 4}), HostInt64(idx, {2})},
                                               {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 3.0F, 5.0F, 7.0F}));

    void* dx = Upload(cuda, x);
    void* di = UploadInt64(cuda, idx);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Gather")(node, {DeviceFloat(dx, {2, 4}), DeviceInt64(di, {2})},
                                                {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(di);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Gather wraps a negative index: {-1} over {3} picks the last element.
TEST(NertCudaIndex, GatherNegativeIndex) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {10.0F, 20.0F, 30.0F};
    std::vector<int64_t> idx = {-1};
    std::vector<float> cpuOut(1, 0.0F);
    nert::Node node = MakeNode("ga1", "Gather");
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Gather")(node, {HostFloat(x, {3}), HostInt64(idx, {1})},
                                               {HostFloat(cpuOut, {1})});
    EXPECT_EQ(cpuOut, (std::vector<float>{30.0F}));

    void* dx = Upload(cuda, x);
    void* di = UploadInt64(cuda, idx);
    std::vector<float> gpuOut(1, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Gather")(node, {DeviceFloat(dx, {3}), DeviceInt64(di, {1})},
                                                {DeviceFloat(dy, {1})});
    gpuOut = Download(cuda, dy, 1);
    cuda.allocator().release(dx);
    cuda.allocator().release(di);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// GatherElements axis 1: per-row picks {{2,0,1},{1,2,0}} over {2,3}.
TEST(NertCudaIndex, GatherElementsExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> idx = {2, 0, 1, 1, 2, 0};
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("ge0", "GatherElements");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("GatherElements")(node, {HostFloat(x, {2, 3}), HostInt64(idx, {2, 3})},
                                                       {HostFloat(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{3.0F, 1.0F, 2.0F, 5.0F, 6.0F, 4.0F}));

    void* dx = Upload(cuda, x);
    void* di = UploadInt64(cuda, idx);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("GatherElements")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(di, {2, 3})},
                                                        {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(cuda, dy, 6);
    cuda.allocator().release(dx);
    cuda.allocator().release(di);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Transpose rejects a non-permutation, naming node + op.
TEST(NertCudaIndex, TransposeRejectsBadPerm) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("tr_bad", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{0, 0});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Transpose")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {2, 3})});
        FAIL() << "expected InferenceException for a non-permutation";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("tr_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Transpose"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Gather reports an out-of-range index, naming node + op.
TEST(NertCudaIndex, GatherRejectsOutOfRange) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F};
    std::vector<int64_t> idx = {3};
    std::vector<float> y(1, 0.0F);
    nert::Node node = MakeNode("ga_bad", "Gather");
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* di = UploadInt64(cuda, idx);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Gather")(node, {DeviceFloat(dx, {3}), DeviceInt64(di, {1})},
                                                    {DeviceFloat(dy, {1})});
        FAIL() << "expected InferenceException for an out-of-range index";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ga_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Gather index is out of range"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(di);
    cuda.allocator().release(dy);
}

// GatherElements reports an out-of-range index, naming node + op.
TEST(NertCudaIndex, GatherElementsRejectsOutOfRange) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> idx = {0, 7};
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("ge_bad", "GatherElements");
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* di = UploadInt64(cuda, idx);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("GatherElements")(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(di, {1, 2})},
                                                            {DeviceFloat(dy, {1, 2})});
        FAIL() << "expected InferenceException for an out-of-range index";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ge_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("GatherElements index is out of range"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(di);
    cuda.allocator().release(dy);
}

// Gather bool data selects raw bytes on device (OWLv2 attention mask).
TEST(NertCudaIndex, GatherBoolData) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<std::uint8_t> x = {1, 0, 1};
    std::vector<int64_t> idx = {2, 0};
    std::vector<std::uint8_t> cpuOut(2, 0);
    nert::Node node = MakeNode("gb0", "Gather");
    node.attributes.emplace_back("axis", int64_t{0});
    node.inputs = {"x", "idx"};
    node.outputs = {"y"};
    nert::TensorView cpuX;
    cpuX.data = x.data();
    cpuX.dtype = nert::DataType::Bool;
    cpuX.dims = {3};
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Bool;
    cpuY.dims = {2};
    nert::CpuDevice().kernels().find("Gather")(node, {cpuX, HostInt64(idx, {2})}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 1}));

    void* dx = cuda.allocator().allocate(3);
    ASSERT_NE(dx, nullptr);
    cuda.transfer().host_to_device(dx, x.data(), 3);
    void* di = UploadInt64(cuda, idx);
    std::vector<std::uint8_t> zero(2, 0);
    void* dy = cuda.allocator().allocate(2);
    ASSERT_NE(dy, nullptr);
    cuda.transfer().host_to_device(dy, zero.data(), 2);
    nert::TensorView in;
    in.data = dx;
    in.dtype = nert::DataType::Bool;
    in.dims = {3};
    nert::TensorView out;
    out.data = dy;
    out.dtype = nert::DataType::Bool;
    out.dims = {2};
    nert::CudaDevice().kernels().find("Gather")(node, {in, DeviceInt64(di, {2})}, {out});
    std::vector<std::uint8_t> gpu(2, 0);
    cuda.transfer().device_to_host(gpu.data(), dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(di);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpu, cpuOut);
}

// Transpose bool data reorders bytes on device.
TEST(NertCudaIndex, TransposeBoolPerm) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<std::uint8_t> x = {1, 0, 1, 0, 1, 0};
    std::vector<std::uint8_t> cpuOut(6, 0);
    nert::Node node = MakeNode("tr_b", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{1, 0});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Transpose")(node, {HostBool(x, {2, 3})}, {HostBool(cpuOut, {3, 2})});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 0, 0, 1, 1, 0}));

    void* dx = UploadBool(cuda, x);
    std::vector<std::uint8_t> zero(6, 0);
    void* dy = UploadBool(cuda, zero);
    nert::CudaDevice().kernels().find("Transpose")(node, {DeviceBool(dx, {2, 3})}, {DeviceBool(dy, {3, 2})});
    std::vector<std::uint8_t> gpuOut = DownloadBool(cuda, dy, 6);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}
