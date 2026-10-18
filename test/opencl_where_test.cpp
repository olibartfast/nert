// OpenCL Where vs the CPU oracle: bool condition, float32/int64 values,
// broadcast, and negatives.
//
// Needs an OpenCL GPU platform at run time; without one every test SKIPs
// (never fails).

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using nert_test::DeviceFloat;
using nert_test::DeviceInt64;
using nert_test::Download;
using nert_test::DownloadInt64;
using nert_test::HostFloat;
using nert_test::HostInt64;
using nert_test::MakeNode;
using nert_test::Upload;
using nert_test::UploadInt64;

bool HasOpenCLDevice() { return nert_test::DeviceAvailable(&nert::OpenCLDevice); }

nert::TensorView HostBool(std::vector<std::uint8_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.data();
    view.dtype = nert::DataType::Bool;
    view.dims = std::move(dims);
    return view;
}

nert::TensorView DeviceBool(void* data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data;
    view.dtype = nert::DataType::Bool;
    view.dims = std::move(dims);
    return view;
}

void RunWhereFloat(const std::string& name, const std::vector<std::uint8_t>& cond, const std::vector<int64_t>& condDims,
                   const std::vector<float>& x, const std::vector<int64_t>& xDims, const std::vector<float>& z,
                   const std::vector<int64_t>& zDims, const std::vector<int64_t>& outDims,
                   const std::vector<float>& expected) {
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode(name, "Where");
    node.inputs = {"c", "x", "z"};
    node.outputs = {"y"};

    std::vector<std::uint8_t> cv = cond;
    std::vector<float> xv = x;
    std::vector<float> zv = z;
    std::vector<float> cpuOut(expected.size(), 0.0F);
    nert::CpuDevice().kernels().find("Where")(
        node, {HostBool(cv, condDims), HostFloat(xv, xDims), HostFloat(zv, zDims)}, {HostFloat(cpuOut, outDims)});
    EXPECT_EQ(cpuOut, expected);

    auto upload_bytes = [&](const std::vector<std::uint8_t>& host) {
        void* buffer = ocl.allocator().allocate(static_cast<int64_t>(host.size()));
        ocl.transfer().host_to_device(buffer, host.data(), static_cast<int64_t>(host.size()));
        return buffer;
    };
    void* dc = upload_bytes(cond);
    void* dx = Upload(ocl, x);
    void* dz = Upload(ocl, z);
    std::vector<float> gpuOut(expected.size(), 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Where")(
        node, {DeviceBool(dc, condDims), DeviceFloat(dx, xDims), DeviceFloat(dz, zDims)}, {DeviceFloat(dy, outDims)});
    gpuOut = Download(ocl, dy, expected.size());
    ocl.allocator().release(dc);
    ocl.allocator().release(dx);
    ocl.allocator().release(dz);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// 2x2 mask selects element-wise.
TEST(NertOpenCLWhere, MaskExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunWhereFloat("wh0", {1, 0, 0, 1}, {2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}, {2, 2}, {10.0F, 20.0F, 30.0F, 40.0F}, {2, 2},
                  {2, 2}, {1.0F, 20.0F, 30.0F, 4.0F});
}

// Condition and both value operands broadcast.
TEST(NertOpenCLWhere, BroadcastExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunWhereFloat("wh_b", {1, 0}, {2, 1}, {5.0F, 6.0F}, {2, 1}, {7.0F, 8.0F}, {1, 2}, {2, 2}, {5.0F, 5.0F, 7.0F, 8.0F});
}

// int64 values keep their dtype.
TEST(NertOpenCLWhere, Int64Exact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<std::uint8_t> cond = {1, 0, 1, 0};
    std::vector<int64_t> x = {10, -20, 30, -40};
    std::vector<int64_t> z = {-1, 2, -3, 4};
    std::vector<int64_t> cpuOut(4, 0);
    nert::Node node = MakeNode("wh_i", "Where");
    node.inputs = {"c", "x", "z"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Where")(
        node, {HostBool(cond, {1, 4}), HostInt64(x, {1, 4}), HostInt64(z, {1, 4})}, {HostInt64(cpuOut, {1, 4})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{10, 2, 30, 4}));

    auto upload_bytes = [&](const std::vector<std::uint8_t>& host) {
        void* buffer = ocl.allocator().allocate(static_cast<int64_t>(host.size()));
        ocl.transfer().host_to_device(buffer, host.data(), static_cast<int64_t>(host.size()));
        return buffer;
    };
    void* dc = upload_bytes(cond);
    void* dx = UploadInt64(ocl, x);
    void* dz = UploadInt64(ocl, z);
    std::vector<int64_t> gpuOut(4, 0);
    void* dy = UploadInt64(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Where")(
        node, {DeviceBool(dc, {1, 4}), DeviceInt64(dx, {1, 4}), DeviceInt64(dz, {1, 4})}, {DeviceInt64(dy, {1, 4})});
    gpuOut = DownloadInt64(ocl, dy, 4);
    ocl.allocator().release(dc);
    ocl.allocator().release(dx);
    ocl.allocator().release(dz);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Non-bool condition, mixed value dtypes and non-broadcast shapes are
// rejected, naming node + op.
TEST(NertOpenCLWhere, RejectsBadInputs) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }

    auto expect_throw = [&](const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                            const nert::TensorView& out) {
        try {
            nert::OpenCLDevice().kernels().find("Where")(node, inputs, {out});
            FAIL() << "expected InferenceException for a malformed Where";
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find("wh_bad"), std::string::npos) << message;
            EXPECT_NE(message.find("Where"), std::string::npos) << message;
        }
    };

    nert::Node badCond = MakeNode("wh_bad", "Where");
    badCond.inputs = {"c", "x", "z"};
    badCond.outputs = {"y"};
    expect_throw(badCond, {DeviceFloat(nullptr, {4}), DeviceFloat(nullptr, {4}), DeviceFloat(nullptr, {4})},
                 DeviceFloat(nullptr, {4}));

    nert::Node badDtype = MakeNode("wh_bad", "Where");
    badDtype.inputs = {"c", "x", "z"};
    badDtype.outputs = {"y"};
    expect_throw(badDtype, {DeviceBool(nullptr, {4}), DeviceFloat(nullptr, {4}), DeviceInt64(nullptr, {4})},
                 DeviceFloat(nullptr, {4}));

    nert::Node badShape = MakeNode("wh_bad", "Where");
    badShape.inputs = {"c", "x", "z"};
    badShape.outputs = {"y"};
    expect_throw(badShape, {DeviceBool(nullptr, {3}), DeviceFloat(nullptr, {3}), DeviceFloat(nullptr, {2})},
                 DeviceFloat(nullptr, {3}));
}

} // namespace
