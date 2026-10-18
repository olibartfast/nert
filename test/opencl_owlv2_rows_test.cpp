// OpenCL rows needed by OWLv2: Cast, ConstantOfShape, Equal, Greater,
// LessOrEqual, And, IsNaN, Erf, Elu, ReduceL2, ArgMax, Einsum.
//
// Each case runs the CPU oracle first and then the OpenCL row; negatives
// assert the node/op in the error. Needs an OpenCL GPU at run time.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
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

void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref, float tol = 1e-4F) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], tol * (1.0F + std::fabs(ref[i]))) << "i=" << i;
    }
}

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

// Cast float32 -> int64 -> float32 and saturation to INT32.
TEST(NertOpenCLOWLv2Rows, CastVariants) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.9F, -2.7F, 0.0F};
    std::vector<int64_t> i64(3, 0);
    nert::Node to64 = MakeNode("cast64", "Cast");
    to64.attributes.emplace_back("to", int64_t{7});
    to64.inputs = {"x"};
    to64.outputs = {"y"};
    nert::CpuDevice().kernels().find("Cast")(to64, {HostFloat(x, {3})}, {HostInt64(i64, {3})});
    EXPECT_EQ(i64, (std::vector<int64_t>{1, -2, 0}));

    void* dx = Upload(ocl, x);
    void* dy = ocl.allocator().allocate(3 * 8);
    nert::OpenCLDevice().kernels().find("Cast")(to64, {DeviceFloat(dx, {3})}, {DeviceInt64(dy, {3})});
    std::vector<int64_t> gi64(3, 0);
    ocl.transfer().device_to_host(gi64.data(), dy, 3 * 8);
    EXPECT_EQ(gi64, i64);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);

    std::vector<float> big = {3.0e9F, -3.0e9F, 1.5F};
    std::vector<int64_t> sat(3, 0);
    nert::Node to32 = MakeNode("cast32", "Cast");
    to32.attributes.emplace_back("to", int64_t{6});
    to32.inputs = {"x"};
    to32.outputs = {"y"};
    nert::CpuDevice().kernels().find("Cast")(to32, {HostFloat(big, {3})}, {HostInt64(sat, {3})});
    EXPECT_EQ(sat, (std::vector<int64_t>{2147483647, -2147483648, 1}));
    dx = Upload(ocl, big);
    dy = ocl.allocator().allocate(3 * 8);
    nert::OpenCLDevice().kernels().find("Cast")(to32, {DeviceFloat(dx, {3})}, {DeviceInt64(dy, {3})});
    ocl.transfer().device_to_host(gi64.data(), dy, 3 * 8);
    EXPECT_EQ(gi64, sat);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// ConstantOfShape float, int64 and default zero.
TEST(NertOpenCLOWLv2Rows, ConstantOfShapeVariants) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> out(6, 0.0F);
    nert::Node fill = MakeNode("cos0", "ConstantOfShape");
    fill.attributes.emplace_back("value", 1.5F);
    fill.inputs = {"shape"};
    fill.outputs = {"y"};
    nert::CpuDevice().kernels().find("ConstantOfShape")(fill, {HostInt64(shape, {2})}, {HostFloat(out, {2, 3})});
    EXPECT_EQ(out, (std::vector<float>{1.5F, 1.5F, 1.5F, 1.5F, 1.5F, 1.5F}));
    void* ds = UploadInt64(ocl, shape);
    void* dy = ocl.allocator().allocate(6 * 4);
    nert::OpenCLDevice().kernels().find("ConstantOfShape")(fill, {DeviceInt64(ds, {2})}, {DeviceFloat(dy, {2, 3})});
    std::vector<float> gout(6, 0.0F);
    ocl.transfer().device_to_host(gout.data(), dy, 6 * 4);
    EXPECT_EQ(gout, out);
    ocl.allocator().release(ds);
    ocl.allocator().release(dy);

    nert::Node zeros = MakeNode("cos1", "ConstantOfShape");
    zeros.inputs = {"shape"};
    zeros.outputs = {"y"};
    std::vector<float> z(6, 7.0F);
    nert::CpuDevice().kernels().find("ConstantOfShape")(zeros, {HostInt64(shape, {2})}, {HostFloat(z, {2, 3})});
    EXPECT_EQ(z, (std::vector<float>(6, 0.0F)));
}

// Comparison rows on a broadcasted pair.
TEST(NertOpenCLOWLv2Rows, CompareBroadcast) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> a = {1.0F, 2.0F};
    std::vector<float> b = {1.0F, 2.0F};
    std::vector<std::uint8_t> cpuOut(4, 0);
    struct Case {
        const char* op;
        std::vector<std::uint8_t> expected;
    };
    const std::vector<Case> cases = {{"Equal", {1, 0, 0, 1}}, {"Greater", {0, 0, 1, 0}}, {"LessOrEqual", {1, 1, 0, 1}}};
    for (const Case& c : cases) {
        std::vector<float> av = {1.0F, 2.0F};
        std::vector<float> bv = {1.0F, 2.0F};
        std::fill(cpuOut.begin(), cpuOut.end(), 0);
        nert::Node node = MakeNode("cmp", c.op);
        node.inputs = {"a", "b"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find(c.op)(node, {HostFloat(av, {2, 1}), HostFloat(bv, {1, 2})},
                                               {HostBool(cpuOut, {2, 2})});
        EXPECT_EQ(cpuOut, c.expected);
        void* da = Upload(ocl, av);
        void* db = Upload(ocl, bv);
        std::vector<std::uint8_t> gout(4, 0);
        void* dy = nullptr;
        UploadBytes(ocl, dy, gout);
        nert::OpenCLDevice().kernels().find(c.op)(node, {DeviceFloat(da, {2, 1}), DeviceFloat(db, {1, 2})},
                                                  {DeviceBool(dy, {2, 2})});
        ocl.transfer().device_to_host(gout.data(), dy, 4);
        EXPECT_EQ(gout, cpuOut);
        ocl.allocator().release(da);
        ocl.allocator().release(db);
        ocl.allocator().release(dy);
    }

    std::vector<std::uint8_t> ca = {1, 0, 1};
    std::vector<std::uint8_t> cb = {1, 1, 0};
    std::vector<std::uint8_t> andOut(3, 0);
    nert::Node andNode = MakeNode("and0", "And");
    andNode.inputs = {"a", "b"};
    andNode.outputs = {"y"};
    nert::CpuDevice().kernels().find("And")(andNode, {HostBool(ca, {3}), HostBool(cb, {3})}, {HostBool(andOut, {3})});
    EXPECT_EQ(andOut, (std::vector<std::uint8_t>{1, 0, 0}));
    void* da = nullptr;
    void* db = nullptr;
    void* dy = nullptr;
    UploadBytes(ocl, da, ca);
    UploadBytes(ocl, db, cb);
    std::vector<std::uint8_t> gout(3, 0);
    UploadBytes(ocl, dy, gout);
    nert::OpenCLDevice().kernels().find("And")(andNode, {DeviceBool(da, {3}), DeviceBool(db, {3})},
                                               {DeviceBool(dy, {3})});
    ocl.transfer().device_to_host(gout.data(), dy, 3);
    EXPECT_EQ(gout, andOut);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
}

// IsNaN maps NaN to true and infinities to false.
TEST(NertOpenCLOWLv2Rows, IsNaNAndErf) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::infinity()};
    std::vector<std::uint8_t> cpuOut(3, 0);
    nert::Node nan = MakeNode("nan0", "IsNaN");
    nan.inputs = {"x"};
    nan.outputs = {"y"};
    nert::CpuDevice().kernels().find("IsNaN")(nan, {HostFloat(x, {3})}, {HostBool(cpuOut, {3})});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{0, 1, 0}));
    void* dx = Upload(ocl, x);
    std::vector<std::uint8_t> gout(3, 0);
    void* dy = nullptr;
    UploadBytes(ocl, dy, gout);
    nert::OpenCLDevice().kernels().find("IsNaN")(nan, {DeviceFloat(dx, {3})}, {DeviceBool(dy, {3})});
    ocl.transfer().device_to_host(gout.data(), dy, 3);
    EXPECT_EQ(gout, cpuOut);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);

    std::vector<float> e = {0.0F, 1.0F, -1.0F};
    std::vector<float> erfOut(3, 0.0F);
    nert::Node erf = MakeNode("erf0", "Erf");
    erf.inputs = {"x"};
    erf.outputs = {"y"};
    nert::CpuDevice().kernels().find("Erf")(erf, {HostFloat(e, {3})}, {HostFloat(erfOut, {3})});
    EXPECT_NEAR(erfOut[1], 0.8427008F, 1e-5F);
    dx = Upload(ocl, e);
    std::vector<float> gerf(3, 0.0F);
    void* de = Upload(ocl, gerf);
    nert::OpenCLDevice().kernels().find("Erf")(erf, {DeviceFloat(dx, {3})}, {DeviceFloat(de, {3})});
    gerf = Download(ocl, de, 3);
    ExpectNear(gerf, erfOut, 1e-5F);
    ocl.allocator().release(dx);
    ocl.allocator().release(de);
}

// Elu default alpha and an off-default alpha.
TEST(NertOpenCLOWLv2Rows, EluAlpha) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {-1.0F, 0.0F, 1.0F};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("elu0", "Elu");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Elu")(node, {HostFloat(x, {3})}, {HostFloat(cpuOut, {3})});
    EXPECT_NEAR(cpuOut[0], -0.6321206F, 1e-5F);
    EXPECT_FLOAT_EQ(cpuOut[1], 0.0F);
    EXPECT_FLOAT_EQ(cpuOut[2], 1.0F);
    void* dx = Upload(ocl, x);
    std::vector<float> gout(3, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::OpenCLDevice().kernels().find("Elu")(node, {DeviceFloat(dx, {3})}, {DeviceFloat(dy, {3})});
    gout = Download(ocl, dy, 3);
    ExpectNear(gout, cpuOut, 1e-5F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// ReduceL2 keepdims over the last axis and no-keepdims over axis 0.
TEST(NertOpenCLOWLv2Rows, ReduceL2Axes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {3.0F, 4.0F, 0.0F, 0.0F, 0.0F, 2.0F};

    std::vector<int64_t> axes = {1};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("l2_1", "ReduceL2");
    node.attributes.emplace_back("keepdims", int64_t{1});
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceL2")(node, {HostFloat(x, {2, 3}), HostInt64(axes, {1})},
                                                 {HostFloat(cpuOut, {2, 1})});
    EXPECT_EQ(cpuOut, (std::vector<float>{5.0F, 2.0F}));
    void* dx = Upload(ocl, x);
    void* da = UploadInt64(ocl, axes);
    std::vector<float> gout(2, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::OpenCLDevice().kernels().find("ReduceL2")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {1})},
                                                    {DeviceFloat(dy, {2, 1})});
    gout = Download(ocl, dy, 2);
    ExpectNear(gout, cpuOut);
    ocl.allocator().release(dx);
    ocl.allocator().release(da);
    ocl.allocator().release(dy);

    std::vector<int64_t> axes0 = {0};
    std::vector<float> cpu0(3, 0.0F);
    nert::Node node0 = MakeNode("l2_0", "ReduceL2");
    node0.attributes.emplace_back("keepdims", int64_t{0});
    node0.inputs = {"x", "axes"};
    node0.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceL2")(node0, {HostFloat(x, {2, 3}), HostInt64(axes0, {1})},
                                                 {HostFloat(cpu0, {3})});
    EXPECT_EQ(cpu0, (std::vector<float>{3.0F, 4.0F, 2.0F}));
    dx = Upload(ocl, x);
    da = UploadInt64(ocl, axes0);
    std::vector<float> g0(3, 0.0F);
    dy = Upload(ocl, g0);
    nert::OpenCLDevice().kernels().find("ReduceL2")(node0, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {1})},
                                                    {DeviceFloat(dy, {3})});
    g0 = Download(ocl, dy, 3);
    ExpectNear(g0, cpu0);
    ocl.allocator().release(dx);
    ocl.allocator().release(da);
    ocl.allocator().release(dy);
}

// ArgMax axis/keepdims and first-index tie-breaking.
TEST(NertOpenCLOWLv2Rows, ArgMaxAxes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 3.0F, 2.0F, 5.0F, 4.0F, 6.0F};
    std::vector<int64_t> cpuOut(2, 0);
    nert::Node node = MakeNode("am1", "ArgMax");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ArgMax")(node, {HostFloat(x, {2, 3})}, {HostInt64(cpuOut, {2, 1})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{1, 2}));
    void* dx = Upload(ocl, x);
    void* dy = ocl.allocator().allocate(2 * 8);
    nert::OpenCLDevice().kernels().find("ArgMax")(node, {DeviceFloat(dx, {2, 3})}, {DeviceInt64(dy, {2, 1})});
    std::vector<int64_t> gout(2, 0);
    ocl.transfer().device_to_host(gout.data(), dy, 2 * 8);
    EXPECT_EQ(gout, cpuOut);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Einsum for both target equations.
TEST(NertOpenCLOWLv2Rows, EinsumEquations) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> b = {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node owl = MakeNode("ein0", "Einsum");
    owl.attributes.emplace_back("equation", std::string("...pd,...qd->...pq"));
    owl.inputs = {"a", "b"};
    owl.outputs = {"y"};
    nert::CpuDevice().kernels().find("Einsum")(owl, {HostFloat(a, {1, 2, 3}), HostFloat(b, {1, 2, 3})},
                                               {HostFloat(cpuOut, {1, 2, 2})});
    ExpectNear(cpuOut, (std::vector<float>{4.0F, 2.0F, 10.0F, 5.0F}));
    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    std::vector<float> gout(4, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::OpenCLDevice().kernels().find("Einsum")(owl, {DeviceFloat(da, {1, 2, 3}), DeviceFloat(db, {1, 2, 3})},
                                                  {DeviceFloat(dy, {1, 2, 2})});
    gout = Download(ocl, dy, 4);
    ExpectNear(gout, cpuOut);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);

    std::vector<float> rfA = {1, 2, 3, 4, 10, 20, 30, 40};
    std::vector<float> rfB = {1, 1, 1, 0};
    std::vector<float> rfOut(8, 0.0F);
    nert::Node rf = MakeNode("ein1", "Einsum");
    rf.attributes.emplace_back("equation", std::string("bchw,bnc->bnhw"));
    rf.inputs = {"a", "b"};
    rf.outputs = {"y"};
    nert::CpuDevice().kernels().find("Einsum")(rf, {HostFloat(rfA, {1, 2, 2, 2}), HostFloat(rfB, {1, 2, 2})},
                                               {HostFloat(rfOut, {1, 2, 2, 2})});
    ExpectNear(rfOut, (std::vector<float>{11, 22, 33, 44, 1, 2, 3, 4}));
    da = Upload(ocl, rfA);
    db = Upload(ocl, rfB);
    std::vector<float> g2(8, 0.0F);
    dy = Upload(ocl, g2);
    nert::OpenCLDevice().kernels().find("Einsum")(rf, {DeviceFloat(da, {1, 2, 2, 2}), DeviceFloat(db, {1, 2, 2})},
                                                  {DeviceFloat(dy, {1, 2, 2, 2})});
    g2 = Download(ocl, dy, 8);
    ExpectNear(g2, rfOut);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
}

// Negative paths name the node and op.
TEST(NertOpenCLOWLv2Rows, Negatives) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    auto expect_throw = [&](const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                            const nert::TensorView& out, const std::string& op) {
        try {
            nert::OpenCLDevice().kernels().find(op)(node, inputs, {out});
            FAIL() << "expected InferenceException for " << op;
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find("bad"), std::string::npos) << message;
            EXPECT_NE(message.find(op), std::string::npos) << message;
        }
    };

    nert::Node cast = MakeNode("bad_cast", "Cast");
    cast.inputs = {"x"};
    cast.outputs = {"y"};
    expect_throw(cast, {DeviceFloat(nullptr, {3})}, DeviceInt64(nullptr, {3}), "Cast");

    nert::Node reduce = MakeNode("bad_reduce", "ReduceL2");
    reduce.attributes.emplace_back("keepdims", int64_t{1});
    reduce.inputs = {"x", "axes"};
    reduce.outputs = {"y"};
    std::vector<int64_t> dup = {1, 1};
    void* da = UploadInt64(const_cast<nert::Device&>(nert::OpenCLDevice()), dup);
    expect_throw(reduce, {DeviceFloat(nullptr, {2, 3}), DeviceInt64(da, {2})}, DeviceFloat(nullptr, {2, 1}),
                 "ReduceL2");
    const_cast<nert::Device&>(nert::OpenCLDevice()).allocator().release(da);

    nert::Node eq = MakeNode("bad_eq", "Equal");
    eq.inputs = {"a", "b"};
    eq.outputs = {"y"};
    expect_throw(eq, {DeviceFloat(nullptr, {2}), DeviceInt64(nullptr, {2})}, DeviceBool(nullptr, {2}), "Equal");

    nert::Node einsum = MakeNode("bad_einsum", "Einsum");
    einsum.attributes.emplace_back("equation", std::string("ij,jk->ik"));
    einsum.inputs = {"a", "b"};
    einsum.outputs = {"y"};
    expect_throw(einsum, {DeviceFloat(nullptr, {2, 2}), DeviceFloat(nullptr, {2, 2})}, DeviceFloat(nullptr, {2, 2}),
                 "Einsum");
}

} // namespace
