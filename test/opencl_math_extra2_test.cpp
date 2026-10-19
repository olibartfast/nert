// OpenCL math rows, second slice: Clip, Log, Pow, Max, Min, Or.
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

// Defined in src/opencl/math_extra2.cpp. Declared here (not wired into the
// OpenCL kernel table yet) so this suite links and calls the rows directly on
// device views.
namespace nert {
namespace opencl_kernels {

void Clip(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Log(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Pow(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Max(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Min(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Or(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

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

// Bool helpers copied from opencl_owlv2_rows_test.cpp.
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

void UploadBytes(nert::Device& ocl, void*& out, const std::vector<std::uint8_t>& host) {
    out = ocl.allocator().allocate(static_cast<int64_t>(host.size()));
    ocl.transfer().host_to_device(out, host.data(), static_cast<int64_t>(host.size()));
}

void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref, float tol = 1e-6F) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], tol * (1.0F + std::fabs(ref[i]))) << "i=" << i;
    }
}

// Clip float32 and int64 with both bounds, with only a min bound, and with no
// bounds (identity). Hand values on the CPU oracle, then the OpenCL row with
// the scalar bounds living in device memory.
TEST(NertOpenCLMathExtra2, ClipFloatAndInt64) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {-5.0F, 0.0F, 7.0F, 20.0F};
    std::vector<float> lo = {-1.0F};
    std::vector<float> hi = {10.0F};

    nert::Node node_f = MakeNode("clip_f", "Clip");
    node_f.inputs = {"x", "lo", "hi"};
    node_f.outputs = {"y"};

    // Both bounds.
    std::vector<float> ref_f(4, 0.0F);
    nert::CpuDevice().kernels().find("Clip")(node_f, {HostFloat(x, {4}), HostFloat(lo, {1}), HostFloat(hi, {1})},
                                             {HostFloat(ref_f, {4})});
    EXPECT_EQ(ref_f, (std::vector<float>{-1.0F, 0.0F, 7.0F, 10.0F}));

    // Only a min bound: the absent max is +infinity.
    std::vector<float> ref_min(4, 0.0F);
    nert::Node node_min = MakeNode("clip_min_only", "Clip");
    node_min.inputs = {"x", "lo"};
    node_min.outputs = {"y"};
    nert::CpuDevice().kernels().find("Clip")(node_min, {HostFloat(x, {4}), HostFloat(lo, {1})},
                                             {HostFloat(ref_min, {4})});
    EXPECT_EQ(ref_min, (std::vector<float>{-1.0F, 0.0F, 7.0F, 20.0F}));

    // No bounds: an exact identity.
    std::vector<float> ref_id(4, 0.0F);
    nert::Node node_id = MakeNode("clip_identity", "Clip");
    node_id.inputs = {"x"};
    node_id.outputs = {"y"};
    nert::CpuDevice().kernels().find("Clip")(node_id, {HostFloat(x, {4})}, {HostFloat(ref_id, {4})});
    EXPECT_EQ(ref_id, x);

    void* dx = Upload(ocl, x);
    void* dlo = Upload(ocl, lo);
    void* dhi = Upload(ocl, hi);
    void* dy = Upload(ocl, std::vector<float>(4, 0.0F));
    nert::opencl_kernels::Clip(node_f, {DeviceFloat(dx, {4}), DeviceFloat(dlo, {1}), DeviceFloat(dhi, {1})},
                               {DeviceFloat(dy, {4})});
    ExpectNear(Download(ocl, dy, 4), ref_f, 1e-6F);
    nert::opencl_kernels::Clip(node_min, {DeviceFloat(dx, {4}), DeviceFloat(dlo, {1})}, {DeviceFloat(dy, {4})});
    ExpectNear(Download(ocl, dy, 4), ref_min, 1e-6F);
    nert::opencl_kernels::Clip(node_id, {DeviceFloat(dx, {4})}, {DeviceFloat(dy, {4})});
    ExpectNear(Download(ocl, dy, 4), ref_id, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dlo);
    ocl.allocator().release(dhi);
    ocl.allocator().release(dy);

    // int64: the same hand values, integer comparisons only.
    std::vector<int64_t> xi = {-5, 0, 7, 20};
    std::vector<int64_t> loi = {-1};
    std::vector<int64_t> hii = {10};
    nert::Node node_i = MakeNode("clip_i", "Clip");
    node_i.inputs = {"x", "lo", "hi"};
    node_i.outputs = {"y"};

    std::vector<int64_t> ref_i(4, 0);
    nert::CpuDevice().kernels().find("Clip")(node_i, {HostInt64(xi, {4}), HostInt64(loi, {1}), HostInt64(hii, {1})},
                                             {HostInt64(ref_i, {4})});
    EXPECT_EQ(ref_i, (std::vector<int64_t>{-1, 0, 7, 10}));

    std::vector<int64_t> ref_i_min(4, 0);
    nert::CpuDevice().kernels().find("Clip")(node_min, {HostInt64(xi, {4}), HostInt64(loi, {1})},
                                             {HostInt64(ref_i_min, {4})});
    EXPECT_EQ(ref_i_min, (std::vector<int64_t>{-1, 0, 7, 20}));

    std::vector<int64_t> ref_i_id(4, 0);
    nert::CpuDevice().kernels().find("Clip")(node_id, {HostInt64(xi, {4})}, {HostInt64(ref_i_id, {4})});
    EXPECT_EQ(ref_i_id, xi);

    void* dxi = UploadInt64(ocl, xi);
    void* dloi = UploadInt64(ocl, loi);
    void* dhii = UploadInt64(ocl, hii);
    void* dyi = UploadInt64(ocl, std::vector<int64_t>(4, 0));
    nert::opencl_kernels::Clip(node_i, {DeviceInt64(dxi, {4}), DeviceInt64(dloi, {1}), DeviceInt64(dhii, {1})},
                               {DeviceInt64(dyi, {4})});
    EXPECT_EQ(DownloadInt64(ocl, dyi, 4), ref_i);
    nert::opencl_kernels::Clip(node_min, {DeviceInt64(dxi, {4}), DeviceInt64(dloi, {1})}, {DeviceInt64(dyi, {4})});
    EXPECT_EQ(DownloadInt64(ocl, dyi, 4), ref_i_min);
    nert::opencl_kernels::Clip(node_id, {DeviceInt64(dxi, {4})}, {DeviceInt64(dyi, {4})});
    EXPECT_EQ(DownloadInt64(ocl, dyi, 4), ref_i_id);
    ocl.allocator().release(dxi);
    ocl.allocator().release(dloi);
    ocl.allocator().release(dhii);
    ocl.allocator().release(dyi);
}

// Log hand anchors {1, e, 0.5} -> {0, 1, -0.6931}; a negative input produces
// NaN on both the oracle and the device (no special-casing on either side).
TEST(NertOpenCLMathExtra2, LogValues) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    const float e = std::exp(1.0F);
    std::vector<float> x = {1.0F, e, 0.5F, -2.0F};
    std::vector<float> ref(4, 0.0F);
    nert::Node node = MakeNode("log0", "Log");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Log")(node, {HostFloat(x, {4})}, {HostFloat(ref, {4})});
    EXPECT_NEAR(ref[0], 0.0F, 1e-6F);
    EXPECT_NEAR(ref[1], 1.0F, 1e-6F);
    EXPECT_NEAR(ref[2], -0.6931F, 1e-4F);
    EXPECT_TRUE(std::isnan(ref[3]));

    void* dx = Upload(ocl, x);
    void* dy = Upload(ocl, std::vector<float>(4, 0.0F));
    nert::opencl_kernels::Log(node, {DeviceFloat(dx, {4})}, {DeviceFloat(dy, {4})});
    std::vector<float> got = Download(ocl, dy, 4);
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (std::isnan(ref[i])) {
            EXPECT_TRUE(std::isnan(got[i])) << "i=" << i;
        } else {
            EXPECT_NEAR(got[i], ref[i], 1e-6F) << "i=" << i;
        }
    }
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Pow with NumPy broadcast: a scalar exponent of exactly 2 squares every
// element (the oracle's shortcut), and a {1,2} exponent broadcasts over the
// {2,2} base.
TEST(NertOpenCLMathExtra2, PowBroadcast) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> base = {1.0F, 2.0F, 3.0F, 4.0F};

    std::vector<float> exp2 = {2.0F};
    nert::Node node_sq = MakeNode("pow_sq", "Pow");
    node_sq.inputs = {"a", "b"};
    node_sq.outputs = {"y"};
    std::vector<float> ref_sq(4, 0.0F);
    nert::CpuDevice().kernels().find("Pow")(node_sq, {HostFloat(base, {2, 2}), HostFloat(exp2, {1})},
                                            {HostFloat(ref_sq, {2, 2})});
    EXPECT_EQ(ref_sq, (std::vector<float>{1.0F, 4.0F, 9.0F, 16.0F}));

    std::vector<float> exp12 = {1.0F, 2.0F};
    nert::Node node_bcast = MakeNode("pow_bcast", "Pow");
    node_bcast.inputs = {"a", "b"};
    node_bcast.outputs = {"y"};
    std::vector<float> ref_bcast(4, 0.0F);
    nert::CpuDevice().kernels().find("Pow")(node_bcast, {HostFloat(base, {2, 2}), HostFloat(exp12, {1, 2})},
                                            {HostFloat(ref_bcast, {2, 2})});
    EXPECT_EQ(ref_bcast, (std::vector<float>{1.0F, 4.0F, 3.0F, 16.0F}));

    void* dbase = Upload(ocl, base);
    void* dexp2 = Upload(ocl, exp2);
    void* dexp12 = Upload(ocl, exp12);
    void* dy = Upload(ocl, std::vector<float>(4, 0.0F));
    nert::opencl_kernels::Pow(node_sq, {DeviceFloat(dbase, {2, 2}), DeviceFloat(dexp2, {1})},
                              {DeviceFloat(dy, {2, 2})});
    ExpectNear(Download(ocl, dy, 4), ref_sq, 1e-6F);
    nert::opencl_kernels::Pow(node_bcast, {DeviceFloat(dbase, {2, 2}), DeviceFloat(dexp12, {1, 2})},
                              {DeviceFloat(dy, {2, 2})});
    ExpectNear(Download(ocl, dy, 4), ref_bcast, 1e-6F);
    ocl.allocator().release(dbase);
    ocl.allocator().release(dexp2);
    ocl.allocator().release(dexp12);
    ocl.allocator().release(dy);
}

// Max/Min fold three broadcast inputs in input order, float32 and int64.
// Hand-computed cells: a {2,1} = {{1},{4}}, b {1,2} = {{2,3}}, c {1} = {2.5}
// broadcast to {2,2}; max = {{2.5,3},{4,4}}, min = {{1,1},{2,2.5}}. The
// int64 case is a {2,1} = {{5},{-7}}, b {1} = {3}, c {1,2} = {{10,0}};
// max = {{10,5},{10,3}}, min = {{3,0},{-7,-7}}.
TEST(NertOpenCLMathExtra2, MaxMinVariadic) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> fa = {1.0F, 4.0F};
    std::vector<float> fb = {2.0F, 3.0F};
    std::vector<float> fc = {2.5F};
    nert::Node max_f = MakeNode("max3_f", "Max");
    max_f.inputs = {"a", "b", "c"};
    max_f.outputs = {"y"};
    std::vector<float> max_ref_f(4, 0.0F);
    nert::CpuDevice().kernels().find("Max")(max_f, {HostFloat(fa, {2, 1}), HostFloat(fb, {1, 2}), HostFloat(fc, {1})},
                                            {HostFloat(max_ref_f, {2, 2})});
    EXPECT_EQ(max_ref_f, (std::vector<float>{2.5F, 3.0F, 4.0F, 4.0F}));

    nert::Node min_f = MakeNode("min3_f", "Min");
    min_f.inputs = {"a", "b", "c"};
    min_f.outputs = {"y"};
    std::vector<float> min_ref_f(4, 0.0F);
    nert::CpuDevice().kernels().find("Min")(min_f, {HostFloat(fa, {2, 1}), HostFloat(fb, {1, 2}), HostFloat(fc, {1})},
                                            {HostFloat(min_ref_f, {2, 2})});
    EXPECT_EQ(min_ref_f, (std::vector<float>{1.0F, 1.0F, 2.0F, 2.5F}));

    void* dfa = Upload(ocl, fa);
    void* dfb = Upload(ocl, fb);
    void* dfc = Upload(ocl, fc);
    void* dfy = Upload(ocl, std::vector<float>(4, 0.0F));
    nert::opencl_kernels::Max(max_f, {DeviceFloat(dfa, {2, 1}), DeviceFloat(dfb, {1, 2}), DeviceFloat(dfc, {1})},
                              {DeviceFloat(dfy, {2, 2})});
    ExpectNear(Download(ocl, dfy, 4), max_ref_f, 1e-6F);
    nert::opencl_kernels::Min(min_f, {DeviceFloat(dfa, {2, 1}), DeviceFloat(dfb, {1, 2}), DeviceFloat(dfc, {1})},
                              {DeviceFloat(dfy, {2, 2})});
    ExpectNear(Download(ocl, dfy, 4), min_ref_f, 1e-6F);
    ocl.allocator().release(dfa);
    ocl.allocator().release(dfb);
    ocl.allocator().release(dfc);
    ocl.allocator().release(dfy);

    std::vector<int64_t> ia = {5, -7};
    std::vector<int64_t> ib = {3};
    std::vector<int64_t> ic = {10, 0};
    nert::Node max_i = MakeNode("max3_i", "Max");
    max_i.inputs = {"a", "b", "c"};
    max_i.outputs = {"y"};
    std::vector<int64_t> max_ref_i(4, 0);
    nert::CpuDevice().kernels().find("Max")(max_i, {HostInt64(ia, {2, 1}), HostInt64(ib, {1}), HostInt64(ic, {1, 2})},
                                            {HostInt64(max_ref_i, {2, 2})});
    EXPECT_EQ(max_ref_i, (std::vector<int64_t>{10, 5, 10, 3}));

    nert::Node min_i = MakeNode("min3_i", "Min");
    min_i.inputs = {"a", "b", "c"};
    min_i.outputs = {"y"};
    std::vector<int64_t> min_ref_i(4, 0);
    nert::CpuDevice().kernels().find("Min")(min_i, {HostInt64(ia, {2, 1}), HostInt64(ib, {1}), HostInt64(ic, {1, 2})},
                                            {HostInt64(min_ref_i, {2, 2})});
    EXPECT_EQ(min_ref_i, (std::vector<int64_t>{3, 0, -7, -7}));

    void* dia = UploadInt64(ocl, ia);
    void* dib = UploadInt64(ocl, ib);
    void* dic = UploadInt64(ocl, ic);
    void* diy = UploadInt64(ocl, std::vector<int64_t>(4, 0));
    nert::opencl_kernels::Max(max_i, {DeviceInt64(dia, {2, 1}), DeviceInt64(dib, {1}), DeviceInt64(dic, {1, 2})},
                              {DeviceInt64(diy, {2, 2})});
    EXPECT_EQ(DownloadInt64(ocl, diy, 4), max_ref_i);
    nert::opencl_kernels::Min(min_i, {DeviceInt64(dia, {2, 1}), DeviceInt64(dib, {1}), DeviceInt64(dic, {1, 2})},
                              {DeviceInt64(diy, {2, 2})});
    EXPECT_EQ(DownloadInt64(ocl, diy, 4), min_ref_i);
    ocl.allocator().release(dia);
    ocl.allocator().release(dib);
    ocl.allocator().release(dic);
    ocl.allocator().release(diy);
}

// Or with NumPy broadcast: a {2,1} = {{1},{0}} against b {1,2} = {{1,0}},
// one byte per element. Hand table: {{1,1},{1,0}}.
TEST(NertOpenCLMathExtra2, OrBroadcast) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<std::uint8_t> a = {1, 0};
    std::vector<std::uint8_t> b = {1, 0};
    std::vector<std::uint8_t> ref(4, 0);
    nert::Node node = MakeNode("or0", "Or");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Or")(node, {HostBool(a, {2, 1}), HostBool(b, {1, 2})}, {HostBool(ref, {2, 2})});
    EXPECT_EQ(ref, (std::vector<std::uint8_t>{1, 1, 1, 0}));

    void* da = nullptr;
    void* db = nullptr;
    void* dy = nullptr;
    UploadBytes(ocl, da, a);
    UploadBytes(ocl, db, b);
    UploadBytes(ocl, dy, std::vector<std::uint8_t>(4, 0));
    nert::opencl_kernels::Or(node, {DeviceBool(da, {2, 1}), DeviceBool(db, {1, 2})}, {DeviceBool(dy, {2, 2})});
    std::vector<std::uint8_t> got(4, 0);
    ocl.transfer().device_to_host(got.data(), dy, 4);
    EXPECT_EQ(got, ref);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
}

// Negative paths name the node and the op.
TEST(NertOpenCLMathExtra2, Negatives) {
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
    auto tv = [](nert::DataType dtype, std::vector<int64_t> dims) {
        nert::TensorView view;
        view.dtype = dtype;
        view.dims = std::move(dims);
        return view;
    };

    // Clip: the min bound must carry the data dtype. Real device buffers keep
    // the row on its normal path until the bound dtype check fires.
    void* dx = Upload(ocl, std::vector<float>{-5.0F, 0.0F, 7.0F, 20.0F});
    void* dy = Upload(ocl, std::vector<float>(4, 0.0F));
    nert::Node bad_clip = MakeNode("bad_clip", "Clip");
    bad_clip.inputs = {"x", "lo"};
    bad_clip.outputs = {"y"};
    expect_throw(&nert::opencl_kernels::Clip, bad_clip, {DeviceFloat(dx, {4}), tv(nert::DataType::Int64, {1})},
                 {DeviceFloat(dy, {4})});
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);

    // Log: wrong arity (two inputs).
    nert::Node bad_log = MakeNode("bad_log", "Log");
    expect_throw(&nert::opencl_kernels::Log, bad_log,
                 {tv(nert::DataType::Float32, {3}), tv(nert::DataType::Float32, {3})},
                 {tv(nert::DataType::Float32, {3})});

    // Pow: the exponent carries a different dtype than the base.
    nert::Node bad_pow = MakeNode("bad_pow", "Pow");
    expect_throw(&nert::opencl_kernels::Pow, bad_pow,
                 {tv(nert::DataType::Float32, {2}), tv(nert::DataType::Int64, {2})},
                 {tv(nert::DataType::Float32, {2})});

    // Max: mixed input dtypes.
    nert::Node bad_max = MakeNode("bad_max", "Max");
    expect_throw(&nert::opencl_kernels::Max, bad_max,
                 {tv(nert::DataType::Float32, {2}), tv(nert::DataType::Int64, {2})},
                 {tv(nert::DataType::Float32, {2})});

    // Min: shapes that cannot broadcast together.
    nert::Node bad_min = MakeNode("bad_min", "Min");
    expect_throw(&nert::opencl_kernels::Min, bad_min,
                 {tv(nert::DataType::Float32, {2, 3}), tv(nert::DataType::Float32, {2, 2})},
                 {tv(nert::DataType::Float32, {2, 3})});

    // Or: float input.
    nert::Node bad_or = MakeNode("bad_or", "Or");
    expect_throw(&nert::opencl_kernels::Or, bad_or, {tv(nert::DataType::Float32, {2}), tv(nert::DataType::Bool, {2})},
                 {tv(nert::DataType::Bool, {2})});
}

// Zero-extent dims are rejected by the dimension check, naming the op.
TEST(NertOpenCLMathExtra2, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("zero_log", "Log");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {0};
    nert::TensorView out = in;
    try {
        nert::opencl_kernels::Log(node, {in}, {out});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("Log"), std::string::npos) << message;
    }
}

} // namespace
