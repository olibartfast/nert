// HIP device (N1c on NVIDIA): allocator, transfers, and the 72-row
// first-party kernel table.
//
// Needs a HIP-usable GPU at run time; without one every test SKIPs (never
// fails). Presence is probed through the seam (constructing the device
// throws naming the cause), so GTEST_SKIP still fires in the TEST body,
// never in a helper.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace {

bool HasHIPDevice() { return nert_test::DeviceAvailable(&nert::HipDevice); }

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

// The first-party table resolves every row, matching the 72-row CPU oracle
// with no library-backed exceptions.
TEST(NertHIPDevice, KernelTableServesFirstPartyRows) {
    if (!HasHIPDevice()) {
        GTEST_SKIP() << "no HIP device";
    }
    const nert::Device& hip = nert::HipDevice();
    const char* rows[] = {
        "Relu",
        "Add",
        "Mul",
        "Sub",
        "Div",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Sigmoid",
        "Tanh",
        "Sqrt",
        "Erf",
        "LayerNormalization",
        "Sin",
        "Cos",
        "Exp",
        "Elu",
        "Log",
        "Floor",
        "IsNaN",
        "Abs",
        "Neg",
        "Pow",
        "Greater",
        "Less",
        "GreaterOrEqual",
        "LessOrEqual",
        "And",
        "Or",
        "Not",
        "Max",
        "Min",
        "Clip",
        "Identity",
        "Squeeze",
        "Tile",
        "Range",
        "EyeLike",
        "Pad",
        "ReduceMean",
        "ReduceL2",
        "ArgMax",
        "CumSum",
        "ScatterND",
        "GridSample",
        "InstanceNormalization",
        "Einsum",
        "Softmax",
        "Reshape",
        "Unsqueeze",
        "Flatten",
        "Expand",
        "Concat",
        "Split",
        "Slice",
        "Transpose",
        "Gather",
        "GatherElements",
        "Cast",
        "Equal",
        "Where",
        "Shape",
        "ConstantOfShape",
        "Mod",
        "ReduceMax",
        "ReduceSum",
        "TopK",
        "Resize",
        "Gemm",
        "MatMul",
        "Conv",
        "ConvTranspose",
    };
    for (const char* row : rows) {
        EXPECT_NE(hip.kernels().find(row), nullptr) << row;
    }
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
