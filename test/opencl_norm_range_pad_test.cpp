// OpenCL rows for InstanceNormalization, Range, and Pad.
//
// Each case runs the CPU oracle first (with hand-computed anchors where the
// packet demands them) and then the OpenCL row on device views; negatives
// assert the node/op in the error. Needs an OpenCL GPU at run time.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Defined in src/opencl/norm_range_pad.cpp. Declared here (not wired into
// the OpenCL kernel table yet) so this suite links and calls the rows
// directly on device views.
namespace nert {
namespace opencl_kernels {

void InstanceNormalization(const Node& node, const std::vector<TensorView>& inputs,
                           const std::vector<TensorView>& outputs);
void Range(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Pad(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

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

} // namespace

// InstanceNormalization hand values: channel 0 {1,2,3,4} has mean 2.5 and
// population variance 1.25 (+1e-5 epsilon); channel 1 {2,4,6,8} has mean 5
// and population variance 5 (+1e-5). Unit scale and bias first, then a
// non-unit scale/bias case against the oracle.
TEST(NertOpenCLNormRangePad, InstanceNormHandComputed) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode("instnorm0", "InstanceNormalization");
    node.inputs = {"x", "scale", "B"};
    node.outputs = {"y"};

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 2.0F, 4.0F, 6.0F, 8.0F};
    std::vector<float> scale = {1.0F, 1.0F};
    std::vector<float> bias = {0.0F, 0.0F};
    std::vector<float> ref(8, 0.0F);
    nert::CpuDevice().kernels().find("InstanceNormalization")(
        node, {HostFloat(x, {1, 2, 2, 2}), HostFloat(scale, {2}), HostFloat(bias, {2})},
        {HostFloat(ref, {1, 2, 2, 2})});
    // Channel 0: (x - 2.5) / sqrt(1.25001); channel 1: (x - 5) / sqrt(5.00001).
    const std::vector<float> hand = {-1.3416354F, -0.4472118F, 0.4472118F, 1.3416354F,
                                     -1.3416394F, -0.4472131F, 0.4472131F, 1.3416394F};
    ExpectNear(ref, hand, 1e-5F);

    void* dx = Upload(ocl, x);
    void* dscale = Upload(ocl, scale);
    void* dbias = Upload(ocl, bias);
    std::vector<float> gout(8, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::InstanceNormalization(
        node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dscale, {2}), DeviceFloat(dbias, {2})},
        {DeviceFloat(dy, {1, 2, 2, 2})});
    gout = Download(ocl, dy, 8);
    ExpectNear(gout, ref, 1e-5F);
    ocl.allocator().release(dscale);
    ocl.allocator().release(dbias);
    ocl.allocator().release(dy);

    // Non-unit scale and bias: the device still matches the oracle.
    std::vector<float> scale2 = {2.0F, 0.5F};
    std::vector<float> bias2 = {1.0F, -1.0F};
    std::vector<float> ref2(8, 0.0F);
    nert::CpuDevice().kernels().find("InstanceNormalization")(
        node, {HostFloat(x, {1, 2, 2, 2}), HostFloat(scale2, {2}), HostFloat(bias2, {2})},
        {HostFloat(ref2, {1, 2, 2, 2})});
    void* dscale2 = Upload(ocl, scale2);
    void* dbias2 = Upload(ocl, bias2);
    void* dy2 = Upload(ocl, gout);
    nert::opencl_kernels::InstanceNormalization(
        node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dscale2, {2}), DeviceFloat(dbias2, {2})},
        {DeviceFloat(dy2, {1, 2, 2, 2})});
    gout = Download(ocl, dy2, 8);
    ExpectNear(gout, ref2, 1e-5F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dscale2);
    ocl.allocator().release(dbias2);
    ocl.allocator().release(dy2);
}

// Range int64: the device scalars are read back, the length is derived on
// the host, and the output is filled one element per work-item; exact
// equality with the oracle for a positive and a negative delta.
TEST(NertOpenCLNormRangePad, RangeInt64) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    struct Case {
        int64_t start;
        int64_t limit;
        int64_t delta;
        std::vector<int64_t> want;
    };
    const std::vector<Case> cases = {{2, 11, 3, {2, 5, 8}}, {5, -1, -2, {5, 3, 1}}};
    for (const Case& c : cases) {
        nert::Node node = MakeNode("range_i", "Range");
        node.inputs = {"start", "limit", "delta"};
        node.outputs = {"y"};
        std::vector<int64_t> start = {c.start};
        std::vector<int64_t> limit = {c.limit};
        std::vector<int64_t> delta = {c.delta};
        std::vector<int64_t> ref(c.want.size(), 0);
        nert::CpuDevice().kernels().find("Range")(node,
                                                  {HostInt64(start, {1}), HostInt64(limit, {1}), HostInt64(delta, {1})},
                                                  {HostInt64(ref, {static_cast<int64_t>(ref.size())})});
        EXPECT_EQ(ref, c.want);
        void* ds = UploadInt64(ocl, start);
        void* dl = UploadInt64(ocl, limit);
        void* dd = UploadInt64(ocl, delta);
        void* dy = UploadInt64(ocl, ref);
        nert::opencl_kernels::Range(node, {DeviceInt64(ds, {1}), DeviceInt64(dl, {1}), DeviceInt64(dd, {1})},
                                    {DeviceInt64(dy, {static_cast<int64_t>(ref.size())})});
        const std::vector<int64_t> gout = DownloadInt64(ocl, dy, c.want.size());
        EXPECT_EQ(gout, c.want);
        ocl.allocator().release(ds);
        ocl.allocator().release(dl);
        ocl.allocator().release(dd);
        ocl.allocator().release(dy);
    }
}

// Range float32: start 0, limit 1, delta 0.25 -> {0, 0.25, 0.5, 0.75}.
TEST(NertOpenCLNormRangePad, RangeFloat) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode("range_f", "Range");
    node.inputs = {"start", "limit", "delta"};
    node.outputs = {"y"};
    std::vector<float> start = {0.0F};
    std::vector<float> limit = {1.0F};
    std::vector<float> delta = {0.25F};
    std::vector<float> ref(4, 0.0F);
    nert::CpuDevice().kernels().find("Range")(
        node, {HostFloat(start, {1}), HostFloat(limit, {1}), HostFloat(delta, {1})}, {HostFloat(ref, {4})});
    ExpectNear(ref, {0.0F, 0.25F, 0.5F, 0.75F}, 1e-6F);
    void* ds = Upload(ocl, start);
    void* dl = Upload(ocl, limit);
    void* dd = Upload(ocl, delta);
    std::vector<float> gout(4, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::Range(node, {DeviceFloat(ds, {1}), DeviceFloat(dl, {1}), DeviceFloat(dd, {1})},
                                {DeviceFloat(dy, {4})});
    gout = Download(ocl, dy, 4);
    ExpectNear(gout, ref, 1e-6F);
    ocl.allocator().release(ds);
    ocl.allocator().release(dl);
    ocl.allocator().release(dd);
    ocl.allocator().release(dy);
}

// Pad constant mode: pads {0,1,0,1} grow each row to {9,1,2,9} and
// {9,3,4,9}; exact bytes against the oracle.
TEST(NertOpenCLNormRangePad, PadConstant) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode("pad_const", "Pad");
    node.inputs = {"x", "pads", "constant_value"};
    node.outputs = {"y"};
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> pads = {0, 1, 0, 1};
    std::vector<float> value = {9.0F};
    std::vector<float> ref(8, 0.0F);
    nert::CpuDevice().kernels().find("Pad")(node, {HostFloat(x, {2, 2}), HostInt64(pads, {4}), HostFloat(value, {1})},
                                            {HostFloat(ref, {2, 4})});
    EXPECT_EQ(ref, (std::vector<float>{9.0F, 1.0F, 2.0F, 9.0F, 9.0F, 3.0F, 4.0F, 9.0F}));
    void* dx = Upload(ocl, x);
    void* dpads = UploadInt64(ocl, pads);
    void* dvalue = Upload(ocl, value);
    std::vector<float> gout(8, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::Pad(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(dpads, {4}), DeviceFloat(dvalue, {1})},
                              {DeviceFloat(dy, {2, 4})});
    gout = Download(ocl, dy, 8);
    EXPECT_EQ(gout, ref);
    ocl.allocator().release(dx);
    ocl.allocator().release(dpads);
    ocl.allocator().release(dvalue);
    ocl.allocator().release(dy);
}

// Pad reflect and edge on the same fixture: row {1,2} pads to {2,1,2,1}
// (reflect) and {1,1,2,2} (edge); row {3,4} likewise.
TEST(NertOpenCLNormRangePad, PadReflectAndEdge) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    struct Case {
        const char* mode;
        std::vector<float> want;
    };
    const std::vector<Case> cases = {{"reflect", {2.0F, 1.0F, 2.0F, 1.0F, 4.0F, 3.0F, 4.0F, 3.0F}},
                                     {"edge", {1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F}}};
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> pads = {0, 1, 0, 1};
    for (const Case& c : cases) {
        nert::Node node = MakeNode("pad_re", "Pad");
        node.attributes.push_back(nert::Attribute("mode", std::string(c.mode)));
        node.inputs = {"x", "pads"};
        node.outputs = {"y"};
        std::vector<float> ref(8, 0.0F);
        nert::CpuDevice().kernels().find("Pad")(node, {HostFloat(x, {2, 2}), HostInt64(pads, {4})},
                                                {HostFloat(ref, {2, 4})});
        EXPECT_EQ(ref, c.want);
        void* dx = Upload(ocl, x);
        void* dpads = UploadInt64(ocl, pads);
        std::vector<float> gout(8, 0.0F);
        void* dy = Upload(ocl, gout);
        nert::opencl_kernels::Pad(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(dpads, {4})}, {DeviceFloat(dy, {2, 4})});
        gout = Download(ocl, dy, 8);
        EXPECT_EQ(gout, ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dpads);
        ocl.allocator().release(dy);
    }
}

// Pad int64 with an axes input selecting only the last axis (-1) and a
// constant_value of -1: each row {1,2,3} pads to {-1,1,2,3,-1,-1}.
TEST(NertOpenCLNormRangePad, PadInt64AndAxes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = MakeNode("pad_axes", "Pad");
    node.inputs = {"x", "pads", "constant_value", "axes"};
    node.outputs = {"y"};
    std::vector<int64_t> x = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> pads = {1, 2};
    std::vector<int64_t> value = {-1};
    std::vector<int64_t> axes = {-1};
    std::vector<int64_t> ref(12, 0);
    nert::CpuDevice().kernels().find("Pad")(
        node, {HostInt64(x, {2, 3}), HostInt64(pads, {2}), HostInt64(value, {1}), HostInt64(axes, {1})},
        {HostInt64(ref, {2, 6})});
    EXPECT_EQ(ref, (std::vector<int64_t>{-1, 1, 2, 3, -1, -1, -1, 4, 5, 6, -1, -1}));
    void* dx = UploadInt64(ocl, x);
    void* dpads = UploadInt64(ocl, pads);
    void* dvalue = UploadInt64(ocl, value);
    void* daxes = UploadInt64(ocl, axes);
    std::vector<int64_t> gout(12, 0);
    void* dy = UploadInt64(ocl, gout);
    nert::opencl_kernels::Pad(
        node, {DeviceInt64(dx, {2, 3}), DeviceInt64(dpads, {2}), DeviceInt64(dvalue, {1}), DeviceInt64(daxes, {1})},
        {DeviceInt64(dy, {2, 6})});
    gout = DownloadInt64(ocl, dy, 12);
    EXPECT_EQ(gout, ref);
    ocl.allocator().release(dx);
    ocl.allocator().release(dpads);
    ocl.allocator().release(dvalue);
    ocl.allocator().release(daxes);
    ocl.allocator().release(dy);
}

// Negative paths name the node and the op.
TEST(NertOpenCLNormRangePad, Negatives) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    auto expect_throw = [&](nert::KernelFn fn, const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                            const std::vector<nert::TensorView>& outputs) {
        try {
            fn(node, inputs, outputs);
            FAIL() << "expected InferenceException for " << node.op_type;
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find(node.name), std::string::npos) << message;
            EXPECT_NE(message.find(node.op_type), std::string::npos) << message;
        }
    };

    // InstanceNormalization with a scale of the wrong length.
    nert::Node bad_scale = MakeNode("bad_scale", "InstanceNormalization");
    std::vector<float> x8(8, 1.0F);
    std::vector<float> scale3(3, 1.0F);
    std::vector<float> bias2(2, 0.0F);
    std::vector<float> y8(8, 0.0F);
    void* dx = Upload(ocl, x8);
    void* dscale = Upload(ocl, scale3);
    void* dbias = Upload(ocl, bias2);
    void* dy = Upload(ocl, y8);
    expect_throw(&nert::opencl_kernels::InstanceNormalization, bad_scale,
                 {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dscale, {3}), DeviceFloat(dbias, {2})},
                 {DeviceFloat(dy, {1, 2, 2, 2})});
    ocl.allocator().release(dx);
    ocl.allocator().release(dscale);
    ocl.allocator().release(dbias);
    ocl.allocator().release(dy);

    // Range with mismatched operand dtypes.
    nert::Node bad_range_dtype = MakeNode("bad_range_dtype", "Range");
    std::vector<float> sf = {2.0F};
    std::vector<float> df = {3.0F};
    std::vector<int64_t> li = {11};
    std::vector<float> rf(3, 0.0F);
    void* ds = Upload(ocl, sf);
    void* dd = Upload(ocl, df);
    void* dl = UploadInt64(ocl, li);
    void* dyf = Upload(ocl, rf);
    expect_throw(&nert::opencl_kernels::Range, bad_range_dtype,
                 {DeviceFloat(ds, {1}), DeviceInt64(dl, {1}), DeviceFloat(dd, {1})}, {DeviceFloat(dyf, {3})});
    ocl.allocator().release(ds);
    ocl.allocator().release(dd);
    ocl.allocator().release(dl);
    ocl.allocator().release(dyf);

    // Range whose output length disagrees with the range.
    nert::Node bad_range_len = MakeNode("bad_range_len", "Range");
    std::vector<int64_t> si = {2};
    std::vector<int64_t> li2 = {11};
    std::vector<int64_t> di = {3};
    std::vector<int64_t> ri(4, 0);
    void* ds2 = UploadInt64(ocl, si);
    void* dl2 = UploadInt64(ocl, li2);
    void* dd2 = UploadInt64(ocl, di);
    void* dy4 = UploadInt64(ocl, ri);
    expect_throw(&nert::opencl_kernels::Range, bad_range_len,
                 {DeviceInt64(ds2, {1}), DeviceInt64(dl2, {1}), DeviceInt64(dd2, {1})}, {DeviceInt64(dy4, {4})});
    ocl.allocator().release(ds2);
    ocl.allocator().release(dl2);
    ocl.allocator().release(dd2);
    ocl.allocator().release(dy4);

    // Pad with one pads entry too few.
    nert::Node bad_pads = MakeNode("bad_pads", "Pad");
    std::vector<float> x4 = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> pads3 = {0, 1, 0};
    std::vector<float> y8b(8, 0.0F);
    void* dxp = Upload(ocl, x4);
    void* dpads = UploadInt64(ocl, pads3);
    void* dyp = Upload(ocl, y8b);
    expect_throw(&nert::opencl_kernels::Pad, bad_pads, {DeviceFloat(dxp, {2, 2}), DeviceInt64(dpads, {3})},
                 {DeviceFloat(dyp, {2, 4})});
    ocl.allocator().release(dxp);
    ocl.allocator().release(dpads);
    ocl.allocator().release(dyp);

    // Pad rejects the unsupported "wrap" mode.
    nert::Node bad_mode = MakeNode("bad_mode", "Pad");
    bad_mode.attributes.push_back(nert::Attribute("mode", std::string("wrap")));
    std::vector<int64_t> pads4 = {0, 1, 0, 1};
    void* dxm = Upload(ocl, x4);
    void* dpadm = UploadInt64(ocl, pads4);
    void* dym = Upload(ocl, y8b);
    expect_throw(&nert::opencl_kernels::Pad, bad_mode, {DeviceFloat(dxm, {2, 2}), DeviceInt64(dpadm, {4})},
                 {DeviceFloat(dym, {2, 4})});
    ocl.allocator().release(dxm);
    ocl.allocator().release(dpadm);
    ocl.allocator().release(dym);

    // Pad with a constant_value of the wrong dtype.
    nert::Node bad_value = MakeNode("bad_value", "Pad");
    std::vector<int64_t> value = {-1};
    void* dxv = Upload(ocl, x4);
    void* dpadv = UploadInt64(ocl, pads4);
    void* dvalv = UploadInt64(ocl, value);
    void* dyv = Upload(ocl, y8b);
    expect_throw(&nert::opencl_kernels::Pad, bad_value,
                 {DeviceFloat(dxv, {2, 2}), DeviceInt64(dpadv, {4}), DeviceInt64(dvalv, {1})},
                 {DeviceFloat(dyv, {2, 4})});
    ocl.allocator().release(dxv);
    ocl.allocator().release(dpadv);
    ocl.allocator().release(dvalv);
    ocl.allocator().release(dyv);
}

// Zero-extent dims are rejected by the dimension check, naming the op.
TEST(NertOpenCLNormRangePad, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("zero_instnorm", "InstanceNormalization");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {0, 2, 2, 2};
    nert::TensorView scale = in;
    scale.dims = {2};
    nert::TensorView bias = scale;
    nert::TensorView out = in;
    try {
        nert::opencl_kernels::InstanceNormalization(node, {in, scale, bias}, {out});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("InstanceNormalization"), std::string::npos) << message;
    }
}
