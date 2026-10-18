// OpenCL device scaffold (N1d-M1): allocator, transfers, empty kernel table.
//
// Needs an OpenCL GPU platform at run time; without one every test SKIPs
// (never fails). Presence is probed through the seam (constructing the
// device throws naming the cause), so GTEST_SKIP still fires in the TEST
// body, never in a helper.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {

bool HasOpenCLDevice() { return nert_test::DeviceAvailable(&nert::OpenCLDevice); }

} // namespace

// The OpenCL device names itself and exposes the seam trio.
TEST(NertOpenCLDevice, NamesItself) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    EXPECT_STREQ(nert::OpenCLDevice().name(), "opencl");
    EXPECT_FALSE(nert::OpenCLDevice().is_host());
    const nert::Device& ocl = nert::OpenCLDevice();
    (void)static_cast<nert::Allocator&>(const_cast<nert::Device&>(ocl).allocator());
    (void)static_cast<nert::Transfer&>(const_cast<nert::Device&>(ocl).transfer());
}

// Table rows resolve; Sin is still missing so the executor rejects the
// node naming op + node.
TEST(NertOpenCLDevice, KernelTableServesPilotRows) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    const nert::Device& ocl = nert::OpenCLDevice();
    EXPECT_NE(ocl.kernels().find("Relu"), nullptr);
    EXPECT_NE(ocl.kernels().find("Add"), nullptr);
    EXPECT_NE(ocl.kernels().find("Mul"), nullptr);
    EXPECT_NE(ocl.kernels().find("Sub"), nullptr);
    EXPECT_NE(ocl.kernels().find("Div"), nullptr);
    EXPECT_NE(ocl.kernels().find("Sigmoid"), nullptr);
    EXPECT_NE(ocl.kernels().find("Tanh"), nullptr);
    EXPECT_NE(ocl.kernels().find("Softmax"), nullptr);
    EXPECT_NE(ocl.kernels().find("Reshape"), nullptr);
    EXPECT_NE(ocl.kernels().find("Flatten"), nullptr);
    EXPECT_NE(ocl.kernels().find("Gemm"), nullptr);
    EXPECT_NE(ocl.kernels().find("Conv"), nullptr);
    EXPECT_EQ(ocl.kernels().find("Sin"), nullptr);
}

// Non-positive sizes allocate null without touching the driver.
TEST(NertOpenCLDevice, RejectsNonPositiveAllocate) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    EXPECT_EQ(ocl.allocator().allocate(0), nullptr);
    EXPECT_EQ(ocl.allocator().allocate(-8), nullptr);
}

// A host->device->host round trip preserves 1 KiB of patterned bytes.
TEST(NertOpenCLDevice, TransferRoundTrip) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    const std::size_t bytes = 1024;
    std::vector<std::uint8_t> host(bytes, 0);
    for (std::size_t i = 0; i < bytes; ++i) {
        host[i] = static_cast<std::uint8_t>((i * 31 + 7) & 0xFF);
    }
    void* device = ocl.allocator().allocate(static_cast<int64_t>(bytes));
    ASSERT_NE(device, nullptr);
    ocl.transfer().host_to_device(device, host.data(), static_cast<int64_t>(bytes));
    std::vector<std::uint8_t> back(bytes, 0);
    ocl.transfer().device_to_host(back.data(), device, static_cast<int64_t>(bytes));
    ocl.allocator().release(device);
    EXPECT_EQ(back, host);
}
