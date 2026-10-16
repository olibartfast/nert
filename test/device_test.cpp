// CPU device-seam tests.
//
// One suite (EngineDevice.*) covering the fixed device interface contract: a
// stable process-lifetime CPU device, 64-byte-aligned arena storage, safe
// release (including null), memcpy transfers in both directions with a
// zero-byte no-op, and an empty kernel table that resolves unknown ops to null.

#include "nert/device.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Two calls return the same object, and the device names itself "cpu".
TEST(NertDevice, ReferenceIsStableAndNamedCpu) {
    const nert::Device& first = nert::CpuDevice();
    const nert::Device& second = nert::CpuDevice();
    EXPECT_EQ(&first, &second);
    EXPECT_STREQ(first.name(), "cpu");
}

// allocate(128) returns non-null storage aligned to a 64-byte boundary, and the
// block is writable then readable.
TEST(NertDevice, AllocateIsAlignedAndUsable) {
    nert::Allocator& allocator = const_cast<nert::Device&>(nert::CpuDevice()).allocator();
    void* block = allocator.allocate(128);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block) % 64, 0U);

    std::memset(block, 0xAB, 128);
    const unsigned char* bytes = static_cast<const unsigned char*>(block);
    for (int i = 0; i < 128; ++i) {
        ASSERT_EQ(bytes[i], 0xAB) << "byte " << i;
    }

    allocator.release(block);
    allocator.release(nullptr); // safe no-op
}

// A non-positive request is refused with null rather than a partial buffer.
TEST(NertDevice, NonPositiveAllocationReturnsNull) {
    nert::Allocator& allocator = const_cast<nert::Device&>(nert::CpuDevice()).allocator();
    EXPECT_EQ(allocator.allocate(0), nullptr);
    EXPECT_EQ(allocator.allocate(-1), nullptr);
}

// host_to_device then device_to_host round-trips an arbitrary byte pattern.
TEST(NertDevice, TransferRoundTripsBytePattern) {
    nert::Device& device = const_cast<nert::Device&>(nert::CpuDevice());
    nert::Allocator& allocator = device.allocator();
    nert::Transfer& transfer = device.transfer();

    const std::vector<std::uint8_t> source = {0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF, 0x5A, 0xA5};
    void* device_block = allocator.allocate(static_cast<int64_t>(source.size()));
    ASSERT_NE(device_block, nullptr);

    transfer.host_to_device(device_block, source.data(), static_cast<int64_t>(source.size()));

    std::vector<std::uint8_t> round_trip(source.size(), 0);
    transfer.device_to_host(round_trip.data(), device_block, static_cast<int64_t>(source.size()));
    EXPECT_EQ(round_trip, source);

    allocator.release(device_block);
}

// A zero-byte copy is a no-op even when both pointers are null.
TEST(NertDevice, ZeroByteTransferIsNoOp) {
    nert::Transfer& transfer = const_cast<nert::Device&>(nert::CpuDevice()).transfer();
    transfer.host_to_device(nullptr, nullptr, 0);
    transfer.device_to_host(nullptr, nullptr, 0);
}

// The kernel table owns no known op in this layer: an unknown op resolves null.
TEST(NertDevice, UnknownOpResolvesNull) {
    const nert::KernelTable& kernels = nert::CpuDevice().kernels();
    EXPECT_EQ(kernels.find("NoSuchOp"), nullptr);
}

} // namespace
