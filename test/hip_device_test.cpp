// HIP device scaffold (N1c-M1 on NVIDIA): allocator, transfers, empty
// kernel table.
//
// Needs a HIP-usable GPU at run time; without one every test SKIPs (never
// fails). Presence is probed through the seam (constructing the device
// throws naming the cause), so GTEST_SKIP still fires in the TEST body,
// never in a helper.

#include "nert/device.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {

bool HasHIPDevice() {
    try {
        (void)nert::HipDevice();
        return true;
    } catch (const nert::InferenceException&) {
        return false;
    }
}

} // namespace

// The HIP device names itself and exposes the seam trio.
TEST(NertHIPDevice, NamesItself) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    EXPECT_STREQ(nert::HipDevice().name(), "hip");
    EXPECT_FALSE(nert::HipDevice().is_host());
    const nert::Device& hip = nert::HipDevice();
    (void)static_cast<nert::Allocator&>(const_cast<nert::Device&>(hip).allocator());
    (void)static_cast<nert::Transfer&>(const_cast<nert::Device&>(hip).transfer());
}

// M2 pilot rows resolve; anything else is still null so the executor
// rejects the node naming op + node.
TEST(NertHIPDevice, KernelTableServesPilotRows) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    const nert::Device& hip = nert::HipDevice();
    EXPECT_NE(hip.kernels().find("Relu"), nullptr);
    EXPECT_NE(hip.kernels().find("Add"), nullptr);
    EXPECT_EQ(hip.kernels().find("Gemm"), nullptr);
}

// Non-positive sizes allocate null without touching the driver.
TEST(NertHIPDevice, RejectsNonPositiveAllocate) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());
    EXPECT_EQ(hip.allocator().allocate(0), nullptr);
    EXPECT_EQ(hip.allocator().allocate(-8), nullptr);
}

// A host->device->host round trip preserves 1 KiB of patterned bytes.
TEST(NertHIPDevice, TransferRoundTrip) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    nert::Device& hip = const_cast<nert::Device&>(nert::HipDevice());
    const std::size_t bytes = 1024;
    std::vector<std::uint8_t> host(bytes, 0);
    for (std::size_t i = 0; i < bytes; ++i) {
        host[i] = static_cast<std::uint8_t>((i * 31 + 7) & 0xFF);
    }
    void* device = hip.allocator().allocate(static_cast<int64_t>(bytes));
    ASSERT_NE(device, nullptr);
    hip.transfer().host_to_device(device, host.data(), static_cast<int64_t>(bytes));
    std::vector<std::uint8_t> back(bytes, 0);
    hip.transfer().device_to_host(back.data(), device, static_cast<int64_t>(bytes));
    hip.allocator().release(device);
    EXPECT_EQ(back, host);
}
