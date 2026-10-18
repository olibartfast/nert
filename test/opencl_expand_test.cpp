// OpenCL Expand vs the CPU oracle: runtime target shapes, float32/int64/
// bool, identity copy and broadcast, plus negatives.
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

void RunExpandFloat(const std::string& name, const std::vector<float>& x, const std::vector<int64_t>& xDims,
                    const std::vector<int64_t>& target, const std::vector<int64_t>& outDims,
                    const std::vector<float>& expected) {
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode(name, "Expand");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};

    std::vector<float> xv = x;
    std::vector<int64_t> shape = target;
    std::vector<float> cpuOut(expected.size(), 0.0F);
    const std::vector<int64_t> shapeDims = {static_cast<int64_t>(shape.size())};
    nert::CpuDevice().kernels().find("Expand")(node, {HostFloat(xv, xDims), HostInt64(shape, shapeDims)},
                                               {HostFloat(cpuOut, outDims)});
    EXPECT_EQ(cpuOut, expected);

    void* dx = Upload(ocl, x);
    void* dshape = UploadInt64(ocl, shape);
    std::vector<float> gpuOut(expected.size(), 0.0F);
    void* dy = Upload(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Expand")(node, {DeviceFloat(dx, xDims), DeviceInt64(dshape, shapeDims)},
                                                  {DeviceFloat(dy, outDims)});
    gpuOut = Download(ocl, dy, expected.size());
    ocl.allocator().release(dx);
    ocl.allocator().release(dshape);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// [1,2] -> [3,2]: rows repeat.
TEST(NertOpenCLExpand, RowsExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunExpandFloat("ex_rows", {1.5F, -2.0F}, {1, 2}, {3, 2}, {3, 2}, {1.5F, -2.0F, 1.5F, -2.0F, 1.5F, -2.0F});
}

// [2,1] -> [2,3]: columns repeat.
TEST(NertOpenCLExpand, ColumnsExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunExpandFloat("ex_cols", {1.0F, 2.0F}, {2, 1}, {2, 3}, {2, 3}, {1.0F, 1.0F, 1.0F, 2.0F, 2.0F, 2.0F});
}

// Scalar -> [2,2].
TEST(NertOpenCLExpand, ScalarExact) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunExpandFloat("ex_scalar", {7.0F}, {}, {2, 2}, {2, 2}, {7.0F, 7.0F, 7.0F, 7.0F});
}

// Identity targets take the copy path.
TEST(NertOpenCLExpand, IdentityCopy) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    RunExpandFloat("ex_copy", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {2, 3}, {2, 3}, {2, 3},
                   {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
}

// int64 rows: [1,3] -> [2,3].
TEST(NertOpenCLExpand, Int64Rows) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<int64_t> x = {10, 20, 30};
    std::vector<int64_t> target = {2, 3};
    std::vector<int64_t> cpuOut(6, 0);
    nert::Node node = MakeNode("ex_i64", "Expand");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Expand")(node, {HostInt64(x, {1, 3}), HostInt64(target, {2})},
                                               {HostInt64(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{10, 20, 30, 10, 20, 30}));

    void* dx = UploadInt64(ocl, x);
    void* dshape = UploadInt64(ocl, target);
    std::vector<int64_t> gpuOut(6, 0);
    void* dy = UploadInt64(ocl, gpuOut);
    nert::OpenCLDevice().kernels().find("Expand")(node, {DeviceInt64(dx, {1, 3}), DeviceInt64(dshape, {2})},
                                                  {DeviceInt64(dy, {2, 3})});
    gpuOut = DownloadInt64(ocl, dy, 6);
    ocl.allocator().release(dx);
    ocl.allocator().release(dshape);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// bool columns: [2,1] -> [2,2].
TEST(NertOpenCLExpand, BoolColumns) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<std::uint8_t> x = {1, 0};
    std::vector<int64_t> target = {2, 2};
    std::vector<std::uint8_t> cpuOut(4, 0);
    nert::Node node = MakeNode("ex_b", "Expand");
    node.inputs = {"x", "shape"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Expand")(node, {HostBool(x, {2, 1}), HostInt64(target, {2})},
                                               {HostBool(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 1, 0, 0}));

    auto upload = [&](const std::vector<std::uint8_t>& host) {
        void* buffer = ocl.allocator().allocate(static_cast<int64_t>(host.size()));
        ocl.transfer().host_to_device(buffer, host.data(), static_cast<int64_t>(host.size()));
        return buffer;
    };
    void* dx = upload(x);
    void* dshape = UploadInt64(ocl, target);
    std::vector<std::uint8_t> gpuOut(4, 0);
    void* dy = upload(gpuOut);
    nert::OpenCLDevice().kernels().find("Expand")(node, {DeviceBool(dx, {2, 1}), DeviceInt64(dshape, {2})},
                                                  {DeviceBool(dy, {2, 2})});
    ocl.transfer().device_to_host(gpuOut.data(), dy, static_cast<int64_t>(gpuOut.size()));
    ocl.allocator().release(dx);
    ocl.allocator().release(dshape);
    ocl.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Incompatible target and wrong output shape are rejected, naming node + op.
TEST(NertOpenCLExpand, RejectsBadShapes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    auto expect_throw = [&](const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                            const nert::TensorView& out) {
        try {
            nert::OpenCLDevice().kernels().find("Expand")(node, inputs, {out});
            FAIL() << "expected InferenceException for a malformed Expand";
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find("ex_bad"), std::string::npos) << message;
            EXPECT_NE(message.find("Expand"), std::string::npos) << message;
        }
    };

    std::vector<int64_t> target = {2, 4};
    void* dshape = UploadInt64(ocl, target);
    nert::Node badTarget = MakeNode("ex_bad", "Expand");
    badTarget.inputs = {"x", "shape"};
    badTarget.outputs = {"y"};
    expect_throw(badTarget, {DeviceFloat(nullptr, {2, 3}), DeviceInt64(dshape, {2})}, DeviceFloat(nullptr, {2, 4}));
    ocl.allocator().release(dshape);

    std::vector<int64_t> target2 = {2, 3};
    void* dshape2 = UploadInt64(ocl, target2);
    nert::Node badOut = MakeNode("ex_bad", "Expand");
    badOut.inputs = {"x", "shape"};
    badOut.outputs = {"y"};
    expect_throw(badOut, {DeviceFloat(nullptr, {2, 1}), DeviceInt64(dshape2, {2})}, DeviceFloat(nullptr, {2, 2}));
    ocl.allocator().release(dshape2);

    nert::Node badShapeDtype = MakeNode("ex_bad", "Expand");
    badShapeDtype.inputs = {"x", "shape"};
    badShapeDtype.outputs = {"y"};
    expect_throw(badShapeDtype, {DeviceFloat(nullptr, {2, 1}), DeviceFloat(nullptr, {2})},
                 DeviceFloat(nullptr, {2, 3}));
}

} // namespace
