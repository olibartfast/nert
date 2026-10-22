// OpenCL reduce rows: ReduceSum, ReduceMax.
//
// Each case runs the CPU oracle first and then the OpenCL row; negatives
// assert the node/op in the error. Needs an OpenCL GPU at run time.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Defined in src/opencl/reduce_extra.cpp. Declared here (not wired into
// the OpenCL kernel table yet) so this suite links and calls the rows
// directly on device views.
namespace nert {
namespace opencl_kernels {

void ReduceSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

} // namespace opencl_kernels
} // namespace nert

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

void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref, float tol = 1e-4F) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], tol * (1.0F + std::fabs(ref[i]))) << "i=" << i;
    }
}

nert::Attribute IntAttr(const std::string& name, int64_t value) {
    nert::Attribute attr;
    attr.name = name;
    attr.value = value;
    return attr;
}

nert::Node MakeReduceNode(const std::string& name, const std::string& op, std::vector<nert::Attribute> attrs) {
    nert::Node node = MakeNode(name, op);
    node.attributes = std::move(attrs);
    return node;
}

nert::TensorView TypeView(nert::DataType dtype, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.dtype = dtype;
    view.dims = std::move(dims);
    return view;
}

// ReduceSum over the last axis with keepdims: {1..6} in a {2, 3} input
// sums each row to {6, 15} in a {2, 1} output; the device must match the
// CPU oracle exactly.
TEST(NertOpenCLReduceExtra, ReduceSumLastAxisKeepdims) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> axes = {1};
    nert::Node node = MakeReduceNode("rsum_keep", "ReduceSum", {IntAttr("keepdims", 1)});
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    std::vector<float> ref(2, 0.0F);
    nert::CpuDevice().kernels().find("ReduceSum")(node, {HostFloat(x, {2, 3}), HostInt64(axes, {1})},
                                                  {HostFloat(ref, {2, 1})});
    EXPECT_EQ(ref, (std::vector<float>{6.0F, 15.0F}));
    void* dx = Upload(ocl, x);
    void* daxes = UploadInt64(ocl, axes);
    std::vector<float> gout(2, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::ReduceSum(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes, {1})},
                                    {DeviceFloat(dy, {2, 1})});
    gout = Download(ocl, dy, 2);
    ExpectNear(gout, ref, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);
}

// ReduceSum without keepdims: a negative axis ({-1}) matches axis 1 and
// axis 0 gives the column sums {5, 7, 9}.
TEST(NertOpenCLReduceExtra, ReduceSumNoKeepdimsAndNegativeAxis) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

    std::vector<int64_t> axes_neg = {-1};
    nert::Node node_neg = MakeReduceNode("rsum_neg", "ReduceSum", {IntAttr("keepdims", 0)});
    std::vector<float> ref_neg(2, 0.0F);
    nert::CpuDevice().kernels().find("ReduceSum")(node_neg, {HostFloat(x, {2, 3}), HostInt64(axes_neg, {1})},
                                                  {HostFloat(ref_neg, {2})});
    EXPECT_EQ(ref_neg, (std::vector<float>{6.0F, 15.0F}));
    void* dx = Upload(ocl, x);
    void* daxes = UploadInt64(ocl, axes_neg);
    std::vector<float> gout(2, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::ReduceSum(node_neg, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes, {1})},
                                    {DeviceFloat(dy, {2})});
    gout = Download(ocl, dy, 2);
    ExpectNear(gout, ref_neg, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);

    std::vector<int64_t> axes_ax0 = {0};
    nert::Node node_ax0 = MakeReduceNode("rsum_ax0", "ReduceSum", {IntAttr("keepdims", 0)});
    std::vector<float> ref_ax0(3, 0.0F);
    nert::CpuDevice().kernels().find("ReduceSum")(node_ax0, {HostFloat(x, {2, 3}), HostInt64(axes_ax0, {1})},
                                                  {HostFloat(ref_ax0, {3})});
    EXPECT_EQ(ref_ax0, (std::vector<float>{5.0F, 7.0F, 9.0F}));
    dx = Upload(ocl, x);
    daxes = UploadInt64(ocl, axes_ax0);
    std::vector<float> gout_ax0(3, 0.0F);
    dy = Upload(ocl, gout_ax0);
    nert::opencl_kernels::ReduceSum(node_ax0, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes, {1})},
                                    {DeviceFloat(dy, {3})});
    gout_ax0 = Download(ocl, dy, 3);
    ExpectNear(gout_ax0, ref_ax0, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);
}

// Absent axes input: reduce everything to a scalar 21 (keepdims 0), or
// copy the input through unchanged when noop_with_empty_axes is set.
TEST(NertOpenCLReduceExtra, ReduceSumEmptyAxes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

    nert::Node node_all = MakeReduceNode("rsum_all", "ReduceSum", {IntAttr("keepdims", 0)});
    node_all.inputs = {"x"};
    node_all.outputs = {"y"};
    std::vector<float> ref(1, 0.0F);
    nert::CpuDevice().kernels().find("ReduceSum")(node_all, {HostFloat(x, {2, 3})}, {HostFloat(ref, {})});
    EXPECT_EQ(ref, (std::vector<float>{21.0F}));
    void* dx = Upload(ocl, x);
    std::vector<float> gout(1, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::ReduceSum(node_all, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {})});
    gout = Download(ocl, dy, 1);
    ExpectNear(gout, ref, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);

    nert::Node node_noop = MakeReduceNode("rsum_noop", "ReduceSum", {IntAttr("noop_with_empty_axes", 1)});
    node_noop.inputs = {"x"};
    node_noop.outputs = {"y"};
    std::vector<float> ref_id(6, 0.0F);
    nert::CpuDevice().kernels().find("ReduceSum")(node_noop, {HostFloat(x, {2, 3})}, {HostFloat(ref_id, {2, 3})});
    EXPECT_EQ(ref_id, x);
    dx = Upload(ocl, x);
    std::vector<float> gout_id(6, -1.0F);
    dy = Upload(ocl, gout_id);
    nert::opencl_kernels::ReduceSum(node_noop, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {2, 3})});
    gout_id = Download(ocl, dy, 6);
    ExpectNear(gout_id, ref_id, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// ReduceSum on int64 data: exact integer sums, matching the CPU oracle
// exactly. Row sums {6, 15} with keepdims, the same sums without
// keepdims via a negative axis, an absent-axes full reduction to 21,
// and the noop_with_empty_axes identity byte copy.
TEST(NertOpenCLReduceExtra, ReduceSumInt64) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<int64_t> x = {1, 2, 3, 4, 5, 6};

    // axes {1}, keepdims 1: {6, 15} in a {2, 1} output.
    std::vector<int64_t> axes = {1};
    nert::Node node_keep = MakeReduceNode("rsum_i64_keep", "ReduceSum", {IntAttr("keepdims", 1)});
    node_keep.inputs = {"x", "axes"};
    node_keep.outputs = {"y"};
    std::vector<int64_t> ref_keep(2, 0);
    nert::CpuDevice().kernels().find("ReduceSum")(node_keep, {HostInt64(x, {2, 3}), HostInt64(axes, {1})},
                                                  {HostInt64(ref_keep, {2, 1})});
    EXPECT_EQ(ref_keep, (std::vector<int64_t>{6, 15}));
    void* dx = UploadInt64(ocl, x);
    void* daxes = UploadInt64(ocl, axes);
    std::vector<int64_t> gout_keep(2, 0);
    void* dy = UploadInt64(ocl, gout_keep);
    nert::opencl_kernels::ReduceSum(node_keep, {DeviceInt64(dx, {2, 3}), DeviceInt64(daxes, {1})},
                                    {DeviceInt64(dy, {2, 1})});
    gout_keep = DownloadInt64(ocl, dy, 2);
    EXPECT_EQ(gout_keep, ref_keep);
    ocl.allocator().release(dx);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);

    // axes {-1}, keepdims 0: the same sums in a flat {2} output.
    std::vector<int64_t> axes_neg = {-1};
    nert::Node node_neg = MakeReduceNode("rsum_i64_neg", "ReduceSum", {IntAttr("keepdims", 0)});
    std::vector<int64_t> ref_neg(2, 0);
    nert::CpuDevice().kernels().find("ReduceSum")(node_neg, {HostInt64(x, {2, 3}), HostInt64(axes_neg, {1})},
                                                  {HostInt64(ref_neg, {2})});
    EXPECT_EQ(ref_neg, (std::vector<int64_t>{6, 15}));
    dx = UploadInt64(ocl, x);
    daxes = UploadInt64(ocl, axes_neg);
    std::vector<int64_t> gout_neg(2, 0);
    dy = UploadInt64(ocl, gout_neg);
    nert::opencl_kernels::ReduceSum(node_neg, {DeviceInt64(dx, {2, 3}), DeviceInt64(daxes, {1})},
                                    {DeviceInt64(dy, {2})});
    gout_neg = DownloadInt64(ocl, dy, 2);
    EXPECT_EQ(gout_neg, ref_neg);
    ocl.allocator().release(dx);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);

    // Absent axes input: the full reduction is exactly 21 in a scalar
    // output.
    nert::Node node_all = MakeReduceNode("rsum_i64_all", "ReduceSum", {IntAttr("keepdims", 0)});
    node_all.inputs = {"x"};
    node_all.outputs = {"y"};
    std::vector<int64_t> ref_all(1, 0);
    nert::CpuDevice().kernels().find("ReduceSum")(node_all, {HostInt64(x, {2, 3})}, {HostInt64(ref_all, {})});
    EXPECT_EQ(ref_all, (std::vector<int64_t>{21}));
    dx = UploadInt64(ocl, x);
    std::vector<int64_t> gout_all(1, 0);
    dy = UploadInt64(ocl, gout_all);
    nert::opencl_kernels::ReduceSum(node_all, {DeviceInt64(dx, {2, 3})}, {DeviceInt64(dy, {})});
    gout_all = DownloadInt64(ocl, dy, 1);
    EXPECT_EQ(gout_all, ref_all);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);

    // noop_with_empty_axes with no axes input: the identity copies the
    // int64 bytes through unchanged.
    nert::Node node_noop = MakeReduceNode("rsum_i64_noop", "ReduceSum", {IntAttr("noop_with_empty_axes", 1)});
    node_noop.inputs = {"x"};
    node_noop.outputs = {"y"};
    std::vector<int64_t> ref_noop(6, 0);
    nert::CpuDevice().kernels().find("ReduceSum")(node_noop, {HostInt64(x, {2, 3})}, {HostInt64(ref_noop, {2, 3})});
    EXPECT_EQ(ref_noop, x);
    dx = UploadInt64(ocl, x);
    std::vector<int64_t> gout_noop(6, -1);
    dy = UploadInt64(ocl, gout_noop);
    nert::opencl_kernels::ReduceSum(node_noop, {DeviceInt64(dx, {2, 3})}, {DeviceInt64(dy, {2, 3})});
    gout_noop = DownloadInt64(ocl, dy, 6);
    EXPECT_EQ(gout_noop, ref_noop);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// ReduceMax over the rows ({3, 6}) and over the columns with keepdims
// (max of {1,5}, {3,4}, {2,6} is {5, 4, 6} in a {1, 3} output).
TEST(NertOpenCLReduceExtra, ReduceMaxAxes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 3.0F, 2.0F, 5.0F, 4.0F, 6.0F};

    std::vector<int64_t> axes_rows = {1};
    nert::Node node_rows = MakeReduceNode("rmax_rows", "ReduceMax", {IntAttr("keepdims", 0)});
    std::vector<float> ref_rows(2, 0.0F);
    nert::CpuDevice().kernels().find("ReduceMax")(node_rows, {HostFloat(x, {2, 3}), HostInt64(axes_rows, {1})},
                                                  {HostFloat(ref_rows, {2})});
    EXPECT_EQ(ref_rows, (std::vector<float>{3.0F, 6.0F}));
    void* dx = Upload(ocl, x);
    void* daxes = UploadInt64(ocl, axes_rows);
    std::vector<float> gout(2, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::ReduceMax(node_rows, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes, {1})},
                                    {DeviceFloat(dy, {2})});
    gout = Download(ocl, dy, 2);
    ExpectNear(gout, ref_rows, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);

    std::vector<int64_t> axes_cols = {0};
    nert::Node node_cols = MakeReduceNode("rmax_cols", "ReduceMax", {IntAttr("keepdims", 1)});
    std::vector<float> ref_cols(3, 0.0F);
    nert::CpuDevice().kernels().find("ReduceMax")(node_cols, {HostFloat(x, {2, 3}), HostInt64(axes_cols, {1})},
                                                  {HostFloat(ref_cols, {1, 3})});
    EXPECT_EQ(ref_cols, (std::vector<float>{5.0F, 4.0F, 6.0F}));
    dx = Upload(ocl, x);
    daxes = UploadInt64(ocl, axes_cols);
    std::vector<float> gout_cols(3, 0.0F);
    dy = Upload(ocl, gout_cols);
    nert::opencl_kernels::ReduceMax(node_cols, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes, {1})},
                                    {DeviceFloat(dy, {1, 3})});
    gout_cols = Download(ocl, dy, 3);
    ExpectNear(gout_cols, ref_cols, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);
}

// A declared-13 ReduceMax keeps axes in the attribute and has no axes input.
// The OpenCL row reads that form in place and matches the CPU oracle and the
// schema-18 input form.
TEST(NertOpenCLReduceExtra, ReduceMaxOldAttributeForm) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 5.0F, 3.0F, 4.0F, 2.0F, 6.0F};
    std::vector<int64_t> axes = {1};

    nert::Node attr_node = MakeReduceNode("rmax_attr", "ReduceMax", {});
    attr_node.schema_version = 13;
    attr_node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    std::vector<float> cpu_attr(2, 0.0F);
    nert::CpuDevice().kernels().find("ReduceMax")(attr_node, {HostFloat(x, {2, 3})}, {HostFloat(cpu_attr, {2, 1})});
    EXPECT_EQ(cpu_attr, (std::vector<float>{5.0F, 6.0F}));

    nert::Node input_node = MakeReduceNode("rmax_input", "ReduceMax", {});
    input_node.schema_version = 18;
    std::vector<float> cpu_input(2, 0.0F);
    nert::CpuDevice().kernels().find("ReduceMax")(input_node, {HostFloat(x, {2, 3}), HostInt64(axes, {1})},
                                                  {HostFloat(cpu_input, {2, 1})});
    EXPECT_EQ(cpu_attr, cpu_input);

    void* dx = Upload(ocl, x);
    void* dy = Upload(ocl, std::vector<float>(2, 0.0F));
    nert::opencl_kernels::ReduceMax(attr_node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {2, 1})});
    ExpectNear(Download(ocl, dy, 2), cpu_attr, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Negative paths name the node and the op.
TEST(NertOpenCLReduceExtra, Negatives) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    auto expect_throw = [&](nert::KernelFn fn, const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                            const std::vector<nert::TensorView>& outputs, const std::string& reason) {
        try {
            fn(node, inputs, outputs);
            FAIL() << "expected InferenceException for " << node.op_type;
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find(node.name), std::string::npos) << message;
            EXPECT_NE(message.find(node.op_type), std::string::npos) << message;
            EXPECT_NE(message.find(reason), std::string::npos) << message;
        }
    };

    // Bool data is not a supported dtype for the sum row; an int64 input
    // with a float32 output is rejected as a dtype mismatch.
    nert::Node bad_dtype = MakeNode("bad_dtype", "ReduceSum");
    expect_throw(&nert::opencl_kernels::ReduceSum, bad_dtype, {TypeView(nert::DataType::Bool, {2, 3})},
                 {TypeView(nert::DataType::Bool, {2, 1})}, "data tensors must be float32 or int64");

    nert::Node out_mismatch = MakeNode("out_mismatch", "ReduceSum");
    expect_throw(&nert::opencl_kernels::ReduceSum, out_mismatch, {TypeView(nert::DataType::Int64, {2, 3})},
                 {TypeView(nert::DataType::Float32, {2, 1})}, "data tensors must be float32 or int64");

    // ReduceMax stays float32-only, like its oracle.
    nert::Node max_i64 = MakeNode("max_i64", "ReduceMax");
    expect_throw(&nert::opencl_kernels::ReduceMax, max_i64, {TypeView(nert::DataType::Int64, {2, 3})},
                 {TypeView(nert::DataType::Int64, {2, 1})}, "data tensors must be float32");

    // Axis validation reads the axes values back from the device, so the
    // fixtures live in device buffers here.
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    void* dx = Upload(ocl, x);
    std::vector<float> out2(2, 0.0F);
    void* dy = Upload(ocl, out2);

    std::vector<int64_t> axis_out_of_range = {2};
    void* daxes_bad = UploadInt64(ocl, axis_out_of_range);
    nert::Node bad_axis = MakeNode("bad_axis", "ReduceSum");
    expect_throw(&nert::opencl_kernels::ReduceSum, bad_axis, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes_bad, {1})},
                 {DeviceFloat(dy, {2})}, "axis out of range");
    ocl.allocator().release(daxes_bad);

    std::vector<int64_t> axes_dup = {1, 1};
    void* daxes_dup = UploadInt64(ocl, axes_dup);
    nert::Node dup_axis = MakeNode("dup_axis", "ReduceSum");
    expect_throw(&nert::opencl_kernels::ReduceSum, dup_axis, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes_dup, {2})},
                 {DeviceFloat(dy, {2})}, "duplicate axis");
    ocl.allocator().release(daxes_dup);

    std::vector<int64_t> axes_last = {1};
    void* daxes_last = UploadInt64(ocl, axes_last);
    nert::Node bad_shape = MakeReduceNode("bad_shape", "ReduceSum", {IntAttr("keepdims", 1)});
    expect_throw(&nert::opencl_kernels::ReduceSum, bad_shape, {DeviceFloat(dx, {2, 3}), DeviceInt64(daxes_last, {1})},
                 {DeviceFloat(dy, {2})}, "output shape does not match");
    ocl.allocator().release(daxes_last);

    // ReduceMax rejects non-int64 axes before reading them.
    std::vector<float> axes_float = {1.0F};
    nert::Node bad_axes_dtype = MakeNode("bad_axes_dtype", "ReduceMax");
    expect_throw(&nert::opencl_kernels::ReduceMax, bad_axes_dtype, {HostFloat(x, {2, 3}), HostFloat(axes_float, {1})},
                 {HostFloat(out2, {2})}, "axes tensor must be int64");

    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Zero-extent dims are rejected by the dimension check, naming the op.
TEST(NertOpenCLReduceExtra, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("zero_rsum", "ReduceSum");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {0, 3};
    nert::TensorView out = in;
    try {
        nert::opencl_kernels::ReduceSum(node, {in}, {out});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("ReduceSum"), std::string::npos) << message;
        EXPECT_NE(message.find(node.name), std::string::npos) << message;
    }
}

} // namespace
