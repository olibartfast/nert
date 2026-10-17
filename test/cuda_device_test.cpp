// CUDA device scaffold (N1a-M1): allocator, transfers, empty kernel table.
//
// Needs a CUDA device at run time; without one every test SKIPs (never
// fails) so CPU-only runners with the toolkit installed stay green. The
// default build (NERT_ENABLE_CUDA=OFF) does not compile this file.

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

// The CUDA device names itself and exposes the seam trio.
TEST(NertCudaDevice, NamesItself) {
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());
    EXPECT_STREQ(nert::CudaDevice().name(), "cuda");
    (void)static_cast<nert::Allocator&>(cuda.allocator());
    (void)static_cast<nert::Transfer&>(cuda.transfer());
}

// M2 pilot rows + N1b Conv resolve; anything else is still null so the
// executor rejects the node naming op + node.
TEST(NertCudaDevice, KernelTableServesPilotRows) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    const nert::Device& cuda = nert::CudaDevice();
    EXPECT_NE(cuda.kernels().find("Gemm"), nullptr);
    EXPECT_NE(cuda.kernels().find("Relu"), nullptr);
    EXPECT_NE(cuda.kernels().find("Add"), nullptr);
    EXPECT_NE(cuda.kernels().find("Mul"), nullptr);
    EXPECT_NE(cuda.kernels().find("Sub"), nullptr);
    EXPECT_NE(cuda.kernels().find("Div"), nullptr);
    EXPECT_NE(cuda.kernels().find("MaxPool"), nullptr);
    EXPECT_NE(cuda.kernels().find("AveragePool"), nullptr);
    EXPECT_NE(cuda.kernels().find("Sigmoid"), nullptr);
    EXPECT_NE(cuda.kernels().find("Tanh"), nullptr);
    EXPECT_NE(cuda.kernels().find("Softmax"), nullptr);
    EXPECT_NE(cuda.kernels().find("Reshape"), nullptr);
    EXPECT_NE(cuda.kernels().find("Unsqueeze"), nullptr);
    EXPECT_NE(cuda.kernels().find("Flatten"), nullptr);
    EXPECT_NE(cuda.kernels().find("Expand"), nullptr);
    EXPECT_NE(cuda.kernels().find("Concat"), nullptr);
    EXPECT_NE(cuda.kernels().find("Split"), nullptr);
    EXPECT_NE(cuda.kernels().find("Slice"), nullptr);
#ifdef NERT_HAS_CUDNN
    EXPECT_NE(cuda.kernels().find("Conv"), nullptr);
#else
    EXPECT_EQ(cuda.kernels().find("Conv"), nullptr);
#endif
}

// Non-positive sizes allocate null without touching the driver.
TEST(NertCudaDevice, RejectsNonPositiveAllocate) {
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());
    EXPECT_EQ(cuda.allocator().allocate(0), nullptr);
    EXPECT_EQ(cuda.allocator().allocate(-8), nullptr);
}

// A host->device->host round trip preserves 1 KiB of patterned bytes.
TEST(NertCudaDevice, TransferRoundTrip) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());
    const std::size_t bytes = 1024;
    std::vector<std::uint8_t> host(bytes, 0);
    for (std::size_t i = 0; i < bytes; ++i) {
        host[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);
    }
    void* device = cuda.allocator().allocate(static_cast<int64_t>(bytes));
    ASSERT_NE(device, nullptr);
    cuda.transfer().host_to_device(device, host.data(), static_cast<int64_t>(bytes));
    std::vector<std::uint8_t> back(bytes, 0);
    cuda.transfer().device_to_host(back.data(), device, static_cast<int64_t>(bytes));
    // Zero-length transfers are safe no-ops.
    cuda.transfer().host_to_device(device, host.data(), 0);
    cuda.transfer().device_to_host(back.data(), device, 0);
    cuda.allocator().release(device);
    EXPECT_EQ(back, host);
}
