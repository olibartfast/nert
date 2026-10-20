// OpenCL Resize row: nearest and linear resampling against the CPU oracle.
//
// Each case runs the CPU oracle first and then the OpenCL row; negatives
// assert the node/op in the error. Needs an OpenCL GPU at run time. The
// scales and sizes inputs live in device memory on this seam, so host views
// feed the oracle and device views (cl_mem handles) feed the row.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

// Defined in src/opencl/resize.cpp. Declared here (not wired into the OpenCL
// kernel table yet) so this suite links and calls the row directly on device
// views.
namespace nert {
namespace opencl_kernels {

void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

} // namespace opencl_kernels
} // namespace nert

namespace {

using nert_test::DeviceFloat;
using nert_test::Download;
using nert_test::HostFloat;
using nert_test::MakeNode;
using nert_test::Upload;

bool HasOpenCLDevice() { return nert_test::DeviceAvailable(&nert::OpenCLDevice); }

void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref, float tol = 1e-4F) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], tol * (1.0F + std::fabs(ref[i]))) << "i=" << i;
    }
}

// An omitted optional ONNX input (roi or scales) arrives as Unknown.
nert::TensorView Omitted() {
    nert::TensorView view;
    view.dtype = nert::DataType::Unknown;
    return view;
}

nert::TensorView HostInts(std::vector<int64_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.data();
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

nert::TensorView DeviceInts(void* data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data;
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

void UploadInts(nert::Device& ocl, void*& out, const std::vector<int64_t>& host) {
    const int64_t bytes = static_cast<int64_t>(host.size() * sizeof(int64_t));
    out = ocl.allocator().allocate(bytes);
    ocl.transfer().host_to_device(out, host.data(), bytes);
}

std::vector<int64_t> DownloadInts(nert::Device& ocl, void* ptr, std::size_t count) {
    std::vector<int64_t> host(count);
    ocl.transfer().device_to_host(host.data(), ptr, static_cast<int64_t>(count * sizeof(int64_t)));
    return host;
}

void SetAttr(nert::Node& node, const std::string& name, const std::string& value) {
    nert::Attribute attr;
    attr.name = name;
    attr.value = value;
    node.attributes.push_back(attr);
}

void SetIntAttr(nert::Node& node, const std::string& name, int64_t value) {
    nert::Attribute attr;
    attr.name = name;
    attr.value = value;
    node.attributes.push_back(attr);
}

nert::Node LinearHalfPixelNode(const std::string& name) {
    nert::Node node = MakeNode(name, "Resize");
    node.inputs = {"x", "", "sizes"};
    node.outputs = {"y"};
    SetAttr(node, "mode", "linear");
    SetAttr(node, "coordinate_transformation_mode", "half_pixel");
    return node;
}

// Runs the 2x2 -> 4x4 linear half_pixel upsample for `node`; returns the
// OpenCL output and fills `ref` with the CPU oracle's output for `node`.
std::vector<float> RunLinearHalfPixel(nert::Device& ocl, const nert::Node& node, std::vector<float>& ref) {
    std::vector<float> x = {0.0F, 1.0F, 2.0F, 3.0F};
    std::vector<int64_t> sizes = {1, 1, 4, 4};
    ref.assign(16, 0.0F);
    nert::CpuDevice().kernels().find("Resize")(
        node, {HostFloat(x, {1, 1, 2, 2}), Omitted(), Omitted(), HostInts(sizes, {4})}, {HostFloat(ref, {1, 1, 4, 4})});
    void* dx = Upload(ocl, x);
    void* dsizes = nullptr;
    UploadInts(ocl, dsizes, sizes);
    std::vector<float> gout(16, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::Resize(node, {DeviceFloat(dx, {1, 1, 2, 2}), Omitted(), Omitted(), DeviceInts(dsizes, {4})},
                                 {DeviceFloat(dy, {1, 1, 4, 4})});
    gout = Download(ocl, dy, 16);
    ocl.allocator().release(dx);
    ocl.allocator().release(dsizes);
    ocl.allocator().release(dy);
    return gout;
}

// Runs the attribute-absent control and one attribute-bearing variant on the
// same geometry. The variant must be an exact no-op (bit-identical to the
// control) and must still match the CPU oracle.
void ExpectNoOpAttributes(const std::string& label, const std::vector<std::pair<std::string, int64_t>>& attrs) {
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> control_ref;
    std::vector<float> control_out;
    ASSERT_NO_THROW(control_out = RunLinearHalfPixel(ocl, LinearHalfPixelNode(label + "_control"), control_ref));
    nert::Node variant = LinearHalfPixelNode(label);
    for (const auto& [name, value] : attrs) {
        SetIntAttr(variant, name, value);
    }
    std::vector<float> ref;
    std::vector<float> out;
    ASSERT_NO_THROW(out = RunLinearHalfPixel(ocl, variant, ref));
    EXPECT_EQ(out, control_out) << "no-op attribute changed the OpenCL output";
    ExpectNear(out, ref, 1e-5F);
}

// Nearest with the asymmetric transform and floor rounding, driven by the
// (X, '', sizes) form: the per-axis scale is 4/2 = 2, so the source
// coordinates are o/2 = 0, 0.5, 1, 1.5 and floor gives 0, 0, 1, 1; every
// source pixel doubles in both axes.
TEST(NertOpenCLResize, NearestAsymmetricFloor) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F}; // {{1, 2}, {3, 4}}
    std::vector<int64_t> sizes = {1, 1, 4, 4};
    nert::Node node = MakeNode("resize_nearest", "Resize");
    node.inputs = {"x", "", "sizes"};
    node.outputs = {"y"};
    SetAttr(node, "mode", "nearest");
    SetAttr(node, "coordinate_transformation_mode", "asymmetric");
    SetAttr(node, "nearest_mode", "floor");

    std::vector<float> ref(16, 0.0F);
    nert::CpuDevice().kernels().find("Resize")(
        node, {HostFloat(x, {1, 1, 2, 2}), Omitted(), Omitted(), HostInts(sizes, {4})}, {HostFloat(ref, {1, 1, 4, 4})});
    EXPECT_EQ(ref, (std::vector<float>{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));

    void* dx = Upload(ocl, x);
    void* dsizes = nullptr;
    UploadInts(ocl, dsizes, sizes);
    std::vector<float> gout(16, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::Resize(node, {DeviceFloat(dx, {1, 1, 2, 2}), Omitted(), Omitted(), DeviceInts(dsizes, {4})},
                                 {DeviceFloat(dy, {1, 1, 4, 4})});
    gout = Download(ocl, dy, 16);
    EXPECT_EQ(gout, ref);
    ocl.allocator().release(dx);
    ocl.allocator().release(dsizes);
    ocl.allocator().release(dy);
}

// Linear with the half_pixel transform (sizes form): the per-axis scale is 2,
// so the source coordinates are (o + 0.5)/2 - 0.5 = -0.25, 0.25, 0.75, 1.25,
// clamped and blended between the two neighbours. Hand anchors: the corner
// (0,0) clamps to pixel (0,0) -> X[0][0] = 0; the quarter point (1,1) samples
// 0.25 on both axes:
//   0.75*0.75*X[0][0] + 0.75*0.25*X[0][1] + 0.25*0.75*X[1][0] + 0.25*0.25*X[1][1]
//   = 0.5625*0 + 0.1875*1 + 0.1875*2 + 0.0625*3 = 0.75.
TEST(NertOpenCLResize, LinearHalfPixelUpsample) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {0.0F, 1.0F, 2.0F, 3.0F}; // {{0, 1}, {2, 3}}
    std::vector<int64_t> sizes = {1, 1, 4, 4};
    nert::Node node = MakeNode("resize_linear", "Resize");
    node.inputs = {"x", "", "sizes"};
    node.outputs = {"y"};
    SetAttr(node, "mode", "linear");
    SetAttr(node, "coordinate_transformation_mode", "half_pixel");

    std::vector<float> ref(16, 0.0F);
    nert::CpuDevice().kernels().find("Resize")(
        node, {HostFloat(x, {1, 1, 2, 2}), Omitted(), Omitted(), HostInts(sizes, {4})}, {HostFloat(ref, {1, 1, 4, 4})});
    // All sixteen values are dyadic, so the oracle's float rounding is exact.
    EXPECT_EQ(ref, (std::vector<float>{0, 0.25F, 0.75F, 1, 0.5F, 0.75F, 1.25F, 1.5F, 1.5F, 1.75F, 2.25F, 2.5F, 2, 2.25F,
                                       2.75F, 3}));
    EXPECT_EQ(ref[0], 0.0F);           // corner: clamped onto pixel (0,0)
    EXPECT_NEAR(ref[5], 0.75F, 1e-6F); // quarter point (1,1); arithmetic above

    void* dx = Upload(ocl, x);
    void* dsizes = nullptr;
    UploadInts(ocl, dsizes, sizes);
    std::vector<float> gout(16, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::Resize(node, {DeviceFloat(dx, {1, 1, 2, 2}), Omitted(), Omitted(), DeviceInts(dsizes, {4})},
                                 {DeviceFloat(dy, {1, 1, 4, 4})});
    gout = Download(ocl, dy, 16);
    ExpectNear(gout, ref, 1e-5F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dsizes);
    ocl.allocator().release(dy);
}

// ONNX's no-op defaults, one attribute at a time. `antialias=0` must be
// accepted and change nothing: the output is bit-identical to the
// attribute-absent control and still matches the CPU oracle.
TEST(NertOpenCLResize, AcceptAntialiasZero) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    ExpectNoOpAttributes("resize_aa0", {{"antialias", 0}});
}

// `exclude_outside=0` alone is the same no-op.
TEST(NertOpenCLResize, AcceptExcludeOutsideZero) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    ExpectNoOpAttributes("resize_eo0", {{"exclude_outside", 0}});
}

// Both ONNX no-op defaults present together (the Depth-Anything-V2 shape).
TEST(NertOpenCLResize, AcceptBothNoOps) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    ExpectNoOpAttributes("resize_both0", {{"antialias", 0}, {"exclude_outside", 0}});
}

// Non-no-op values and non-int values reject with node + op + attribute,
// independently of `mode` (checked here on nearest as well as linear).
TEST(NertOpenCLResize, RejectsNonNoOpAttributes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    auto expect_throw = [](const nert::Node& node, const std::string& fragment) {
        nert::TensorView in;
        in.dtype = nert::DataType::Float32;
        in.dims = {1, 1, 2, 2};
        nert::TensorView out;
        out.dtype = nert::DataType::Float32;
        out.dims = {1, 1, 2, 2};
        try {
            nert::opencl_kernels::Resize(node, {in}, {out});
            FAIL() << "expected InferenceException for " << node.name;
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find(node.name), std::string::npos) << message;
            EXPECT_NE(message.find(node.op_type), std::string::npos) << message;
            EXPECT_NE(message.find(fragment), std::string::npos) << message;
        }
    };

    // Nonzero ints (positive and negative) reject.
    for (const char* name : {"antialias", "exclude_outside"}) {
        nert::Node one = LinearHalfPixelNode(std::string("resize_") + name + "_one");
        SetIntAttr(one, name, 1);
        expect_throw(one, name);
        nert::Node negative = LinearHalfPixelNode(std::string("resize_") + name + "_negative");
        SetIntAttr(negative, name, -1);
        expect_throw(negative, name);
    }

    // A string value is not an int and rejects for each name.
    for (const char* name : {"antialias", "exclude_outside"}) {
        nert::Node wrong_type = LinearHalfPixelNode(std::string("resize_") + name + "_string");
        SetAttr(wrong_type, name, "0");
        expect_throw(wrong_type, name);
    }

    // Mode-independent: antialias=1 on the nearest path still rejects.
    nert::Node nearest = MakeNode("resize_aa1_nearest", "Resize");
    nearest.inputs = {"x", "", "sizes"};
    nearest.outputs = {"y"};
    SetAttr(nearest, "mode", "nearest");
    SetIntAttr(nearest, "antialias", 1);
    expect_throw(nearest, "antialias");
}

// The same geometry as NearestAsymmetricFloor with int64 data (nearest only);
// the row must copy the picked values exactly.
TEST(NertOpenCLResize, NearestInt64) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<int64_t> x = {1, 2, 3, 4};
    std::vector<int64_t> sizes = {1, 1, 4, 4};
    nert::Node node = MakeNode("resize_int64", "Resize");
    node.inputs = {"x", "", "sizes"};
    node.outputs = {"y"};
    SetAttr(node, "mode", "nearest");
    SetAttr(node, "coordinate_transformation_mode", "asymmetric");
    SetAttr(node, "nearest_mode", "floor");

    std::vector<int64_t> ref(16, 0);
    nert::CpuDevice().kernels().find("Resize")(
        node, {HostInts(x, {1, 1, 2, 2}), Omitted(), Omitted(), HostInts(sizes, {4})}, {HostInts(ref, {1, 1, 4, 4})});
    EXPECT_EQ(ref, (std::vector<int64_t>{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));

    void* dx = nullptr;
    UploadInts(ocl, dx, x);
    void* dsizes = nullptr;
    UploadInts(ocl, dsizes, sizes);
    std::vector<int64_t> gout(16, 0);
    void* dy = nullptr;
    UploadInts(ocl, dy, gout);
    nert::opencl_kernels::Resize(node, {DeviceInts(dx, {1, 1, 2, 2}), Omitted(), Omitted(), DeviceInts(dsizes, {4})},
                                 {DeviceInts(dy, {1, 1, 4, 4})});
    gout = DownloadInts(ocl, dy, 16);
    EXPECT_EQ(gout, ref);
    ocl.allocator().release(dx);
    ocl.allocator().release(dsizes);
    ocl.allocator().release(dy);
}

// Scales path with every attribute at its default (nearest, half_pixel,
// round_prefer_floor): scales {1, 1, 2, 2} on a 2x2 input give output
// {1, 1, 4, 4}; the source coordinates (o + 0.5)/2 - 0.5 = -0.25, 0.25, 0.75,
// 1.25 round (prefer floor) to 0, 0, 1, 1 per axis.
TEST(NertOpenCLResize, ScalesPath) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> scales = {1.0F, 1.0F, 2.0F, 2.0F};
    nert::Node node = MakeNode("resize_scales", "Resize");
    node.inputs = {"x", "", "scales"};
    node.outputs = {"y"};

    std::vector<float> ref(16, 0.0F);
    nert::CpuDevice().kernels().find("Resize")(
        node, {HostFloat(x, {1, 1, 2, 2}), Omitted(), HostFloat(scales, {4}), Omitted()},
        {HostFloat(ref, {1, 1, 4, 4})});
    EXPECT_EQ(ref, (std::vector<float>{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));

    void* dx = Upload(ocl, x);
    void* dscales = Upload(ocl, scales);
    std::vector<float> gout(16, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::Resize(node, {DeviceFloat(dx, {1, 1, 2, 2}), Omitted(), DeviceFloat(dscales, {4}), Omitted()},
                                 {DeviceFloat(dy, {1, 1, 4, 4})});
    gout = Download(ocl, dy, 16);
    EXPECT_EQ(gout, ref);
    ocl.allocator().release(dx);
    ocl.allocator().release(dscales);
    ocl.allocator().release(dy);
}

// Negative paths name the node and the op.
TEST(NertOpenCLResize, Negatives) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    auto expect_throw = [&](const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                            const std::vector<nert::TensorView>& outputs, const std::string& fragment = "") {
        try {
            nert::opencl_kernels::Resize(node, inputs, outputs);
            FAIL() << "expected InferenceException for " << node.name;
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find(node.name), std::string::npos) << message;
            EXPECT_NE(message.find(node.op_type), std::string::npos) << message;
            EXPECT_NE(message.find(fragment), std::string::npos) << message;
        }
    };
    auto tv = [](nert::DataType dtype, std::vector<int64_t> dims) {
        nert::TensorView view;
        view.dtype = dtype;
        view.dims = std::move(dims);
        return view;
    };

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> scales = {1.0F, 1.0F, 2.0F, 2.0F};
    std::vector<int64_t> sizes = {1, 1, 4, 4};
    void* dx = Upload(ocl, x);
    void* dscales = Upload(ocl, scales);
    void* dsizes = nullptr;
    UploadInts(ocl, dsizes, sizes);
    std::vector<float> zeros(16, 0.0F);
    void* dy = Upload(ocl, zeros);

    // Both scales and sizes present.
    nert::Node both = MakeNode("resize_both", "Resize");
    expect_throw(both,
                 {tv(nert::DataType::Float32, {1, 1, 2, 2}), Omitted(), tv(nert::DataType::Float32, {1, 1, 2, 2}),
                  tv(nert::DataType::Int64, {1, 1, 4, 4})},
                 {tv(nert::DataType::Float32, {1, 1, 4, 4})}, "exactly one of scales and sizes");

    // Sizes with the wrong element count for the input rank.
    nert::Node bad_rank = MakeNode("resize_sizes_rank", "Resize");
    expect_throw(bad_rank,
                 {tv(nert::DataType::Float32, {1, 1, 2, 2}), Omitted(), Omitted(), tv(nert::DataType::Int64, {2})},
                 {tv(nert::DataType::Float32, {1, 1, 4, 4})}, "one entry per input axis");

    // Output dims must match the sizes input.
    nert::Node mismatch = MakeNode("resize_out_dims", "Resize");
    expect_throw(mismatch, {DeviceFloat(dx, {1, 1, 2, 2}), Omitted(), Omitted(), DeviceInts(dsizes, {4})},
                 {tv(nert::DataType::Float32, {1, 1, 2, 4})}, "does not match the sizes input");

    // ... and floor(in * scale) on the scales path.
    nert::Node scaled = MakeNode("resize_scaled_dims", "Resize");
    expect_throw(scaled, {DeviceFloat(dx, {1, 1, 2, 2}), Omitted(), DeviceFloat(dscales, {4}), Omitted()},
                 {tv(nert::DataType::Float32, {1, 1, 3, 4})}, "does not match the scaled input");

    // Cubic mode is rejected here.
    nert::Node cubic = MakeNode("resize_cubic", "Resize");
    SetAttr(cubic, "mode", "cubic");
    expect_throw(cubic, {DeviceFloat(dx, {1, 1, 2, 2})}, {tv(nert::DataType::Float32, {1, 1, 2, 2})},
                 "'mode' must be 'nearest' or 'linear'");

    // Unsupported coordinate transform.
    nert::Node bad_transform = MakeNode("resize_transform", "Resize");
    SetAttr(bad_transform, "coordinate_transformation_mode", "wrap");
    expect_throw(bad_transform, {DeviceFloat(dx, {1, 1, 2, 2})}, {tv(nert::DataType::Float32, {1, 1, 2, 2})},
                 "coordinate_transformation_mode");

    // Null output and input buffers.
    nert::Node null_out = MakeNode("resize_null_out", "Resize");
    expect_throw(null_out, {DeviceFloat(dx, {1, 1, 2, 2}), Omitted(), Omitted(), DeviceInts(dsizes, {4})},
                 {tv(nert::DataType::Float32, {1, 1, 4, 4})}, "null buffer");
    nert::Node null_in = MakeNode("resize_null_in", "Resize");
    expect_throw(null_in, {tv(nert::DataType::Float32, {1, 1, 2, 2}), Omitted(), Omitted(), DeviceInts(dsizes, {4})},
                 {DeviceFloat(dy, {1, 1, 4, 4})}, "null buffer");

    ocl.allocator().release(dx);
    ocl.allocator().release(dscales);
    ocl.allocator().release(dsizes);
    ocl.allocator().release(dy);
}

// Zero-extent input dims are rejected, naming the op.
TEST(NertOpenCLResize, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("resize_zero", "Resize");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {1, 1, 0, 2};
    nert::TensorView out = in;
    try {
        nert::opencl_kernels::Resize(node, {in}, {out});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("Resize"), std::string::npos) << message;
    }
}

} // namespace
