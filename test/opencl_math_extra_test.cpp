// OpenCL extra math rows: Sqrt, Cos, Sin, Exp, Not, Less.
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

// Defined in src/opencl/math_unary_extra.cpp. Declared here (not wired into
// the OpenCL kernel table yet) so this suite links and calls the rows
// directly on device views.
namespace nert {
namespace opencl_kernels {

void Sqrt(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Cos(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Sin(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Exp(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Not(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Less(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

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

// Sqrt hand values on the CPU oracle, then the OpenCL row vs the oracle.
TEST(NertOpenCLMathExtra, SqrtHandComputed) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {4.0F, 0.25F, 2.25F};
    std::vector<float> ref(3, 0.0F);
    nert::Node node = MakeNode("sqrt0", "Sqrt");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Sqrt")(node, {HostFloat(x, {3})}, {HostFloat(ref, {3})});
    ExpectNear(ref, {2.0F, 0.5F, 1.5F}, 1e-6F);
    void* dx = Upload(ocl, x);
    std::vector<float> gout(3, 0.0F);
    void* dy = Upload(ocl, gout);
    nert::opencl_kernels::Sqrt(node, {DeviceFloat(dx, {3})}, {DeviceFloat(dy, {3})});
    gout = Download(ocl, dy, 3);
    ExpectNear(gout, ref, 1e-6F);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Cos/Sin/Exp hand anchors on the CPU oracle, then each OpenCL row vs the
// oracle on {0, pi/4, pi/2}.
TEST(NertOpenCLMathExtra, TrigAndExp) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    const float pi = std::acos(-1.0F);
    struct Case {
        const char* op;
        nert::KernelFn fn;
        std::size_t hand_index;
        float hand_value;
    };
    const std::vector<Case> cases = {
        {"Cos", &nert::opencl_kernels::Cos, 0, 1.0F},
        {"Sin", &nert::opencl_kernels::Sin, 2, 1.0F},
        {"Exp", &nert::opencl_kernels::Exp, 0, 1.0F},
    };
    std::vector<float> x = {0.0F, pi / 4.0F, pi / 2.0F};
    for (const Case& c : cases) {
        std::vector<float> ref(3, 0.0F);
        nert::Node node = MakeNode("trig", c.op);
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find(c.op)(node, {HostFloat(x, {3})}, {HostFloat(ref, {3})});
        EXPECT_NEAR(ref[c.hand_index], c.hand_value, 1e-6F);
        void* dx = Upload(ocl, x);
        std::vector<float> gout(3, 0.0F);
        void* dy = Upload(ocl, gout);
        c.fn(node, {DeviceFloat(dx, {3})}, {DeviceFloat(dy, {3})});
        gout = Download(ocl, dy, 3);
        ExpectNear(gout, ref, 1e-5F);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    }
}

// Not flips bools; hand values then exact device vs oracle bytes.
TEST(NertOpenCLMathExtra, NotBool) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<std::uint8_t> x = {0, 1, 0};
    std::vector<std::uint8_t> ref(3, 0);
    nert::Node node = MakeNode("not0", "Not");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Not")(node, {HostBool(x, {3})}, {HostBool(ref, {3})});
    EXPECT_EQ(ref, (std::vector<std::uint8_t>{1, 0, 1}));
    void* dx = nullptr;
    UploadBytes(ocl, dx, x);
    std::vector<std::uint8_t> gout(3, 0);
    void* dy = nullptr;
    UploadBytes(ocl, dy, gout);
    nert::opencl_kernels::Not(node, {DeviceBool(dx, {3})}, {DeviceBool(dy, {3})});
    ocl.transfer().device_to_host(gout.data(), dy, 3);
    EXPECT_EQ(gout, ref);
    ocl.allocator().release(dx);
    ocl.allocator().release(dy);
}

// Less with NumPy broadcast: a {2,1} against b {1,2}, strict ordering.
TEST(NertOpenCLMathExtra, LessBroadcast) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> a = {2.0F, 1.0F};
    std::vector<float> b = {1.0F, 2.0F};
    std::vector<std::uint8_t> ref(4, 0);
    nert::Node node = MakeNode("less0", "Less");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Less")(node, {HostFloat(a, {2, 1}), HostFloat(b, {1, 2})},
                                             {HostBool(ref, {2, 2})});
    EXPECT_EQ(ref, (std::vector<std::uint8_t>{0, 0, 0, 1}));
    void* da = Upload(ocl, a);
    void* db = Upload(ocl, b);
    std::vector<std::uint8_t> gout(4, 0);
    void* dy = nullptr;
    UploadBytes(ocl, dy, gout);
    nert::opencl_kernels::Less(node, {DeviceFloat(da, {2, 1}), DeviceFloat(db, {1, 2})}, {DeviceBool(dy, {2, 2})});
    ocl.transfer().device_to_host(gout.data(), dy, 4);
    EXPECT_EQ(gout, ref);
    ocl.allocator().release(da);
    ocl.allocator().release(db);
    ocl.allocator().release(dy);
}

// Negative paths name the node and the op.
TEST(NertOpenCLMathExtra, Negatives) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
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

    nert::Node bad_sqrt = MakeNode("bad_sqrt", "Sqrt");
    expect_throw(&nert::opencl_kernels::Sqrt, bad_sqrt, {tv(nert::DataType::Int64, {3})},
                 {tv(nert::DataType::Float32, {3})});

    nert::Node bad_cos = MakeNode("bad_cos", "Cos");
    expect_throw(&nert::opencl_kernels::Cos, bad_cos, {tv(nert::DataType::Float32, {3})},
                 {tv(nert::DataType::Float32, {3}), tv(nert::DataType::Float32, {3})});

    nert::Node bad_sin = MakeNode("bad_sin", "Sin");
    expect_throw(&nert::opencl_kernels::Sin, bad_sin, {tv(nert::DataType::Float32, {3})},
                 {tv(nert::DataType::Float32, {4})});

    nert::Node bad_exp = MakeNode("bad_exp", "Exp");
    expect_throw(&nert::opencl_kernels::Exp, bad_exp, {tv(nert::DataType::Float32, {3})},
                 {tv(nert::DataType::Float32, {3})});

    nert::Node bad_not = MakeNode("bad_not", "Not");
    expect_throw(&nert::opencl_kernels::Not, bad_not, {tv(nert::DataType::Float32, {3})},
                 {tv(nert::DataType::Bool, {3})});

    nert::Node bad_less = MakeNode("bad_less", "Less");
    expect_throw(&nert::opencl_kernels::Less, bad_less,
                 {tv(nert::DataType::Float32, {2, 1}), tv(nert::DataType::Int64, {1, 2})},
                 {tv(nert::DataType::Bool, {2, 2})});
}

// Zero-extent dims are rejected by the dimension check, naming the op.
TEST(NertOpenCLMathExtra, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("zero_sqrt", "Sqrt");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {0};
    nert::TensorView out = in;
    try {
        nert::opencl_kernels::Sqrt(node, {in}, {out});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("Sqrt"), std::string::npos) << message;
    }
}

} // namespace
