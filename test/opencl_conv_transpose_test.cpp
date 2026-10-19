// OpenCL ConvTranspose row: the direct (gather) transposed convolution.
//
// Each positive case hand-computes the CPU oracle's result first and then
// compares the OpenCL row against the oracle on device views; negatives
// assert the node/op in the error. Needs an OpenCL GPU at run time.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Defined in src/opencl/conv_transpose.cpp. Declared here (not wired into
// the OpenCL kernel table yet) so this suite links and calls the row
// directly on device views.
namespace nert {
namespace opencl_kernels {

void ConvTranspose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

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

// A ConvTranspose node with the input/weight/output value names set.
nert::Node ConvTransposeNode(const std::string& name) {
    nert::Node node = MakeNode(name, "ConvTranspose");
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    return node;
}

void SetIntList(nert::Node& node, const std::string& name, std::vector<int64_t> values) {
    nert::Attribute attr;
    attr.name = name;
    attr.value = std::move(values);
    node.attributes.push_back(std::move(attr));
}

void SetInt(nert::Node& node, const std::string& name, int64_t value) {
    nert::Attribute attr;
    attr.name = name;
    attr.value = value;
    node.attributes.push_back(std::move(attr));
}

nert::TensorView View(nert::DataType dtype, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.dtype = dtype;
    view.dims = std::move(dims);
    return view;
}

// The shared fixture: X = {{1,2},{3,4}}, a 2x2 kernel of ones. Stride 1 gives
// the full window sums {{1,3,2},{4,10,6},{3,7,4}}; stride 2 scatters each
// input cell into a 2x2 block, giving {{1,1,2,2},{1,1,2,2},{3,3,4,4},{3,3,4,4}}.
// The inputs return non-const refs: HostFloat borrows them mutably.
std::vector<float>& FixtureX() {
    static std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    return x;
}

std::vector<float>& FixtureW() {
    static std::vector<float> w = {1.0F, 1.0F, 1.0F, 1.0F};
    return w;
}

const std::vector<float>& Stride1Expected() {
    static const std::vector<float> expected = {1.0F, 3.0F, 2.0F, 4.0F, 10.0F, 6.0F, 3.0F, 7.0F, 4.0F};
    return expected;
}

// Element count of a dimension vector (all dims are positive in these tests).
std::size_t CountOf(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        count *= dim;
    }
    return static_cast<std::size_t>(count);
}

// Runs the OpenCL row on device copies of the given host inputs and downloads
// the result.
std::vector<float> RunDevice(nert::Device& ocl, const nert::Node& node, const std::vector<nert::TensorView>& hostIn,
                             const std::vector<int64_t>& outDims) {
    std::vector<void*> deviceBuffers;
    for (const nert::TensorView& view : hostIn) {
        const std::size_t bytes = sizeof(float) * CountOf(view.dims);
        void* buffer = ocl.allocator().allocate(static_cast<int64_t>(bytes));
        ocl.transfer().host_to_device(buffer, view.data, static_cast<int64_t>(bytes));
        deviceBuffers.push_back(buffer);
    }
    void* dy = ocl.allocator().allocate(static_cast<int64_t>(sizeof(float) * CountOf(outDims)));
    std::vector<nert::TensorView> deviceIn;
    for (std::size_t i = 0; i < hostIn.size(); ++i) {
        deviceIn.push_back(DeviceFloat(deviceBuffers[i], hostIn[i].dims));
    }
    nert::opencl_kernels::ConvTranspose(node, deviceIn, {DeviceFloat(dy, outDims)});
    std::vector<float> got = Download(ocl, dy, CountOf(outDims));
    for (void* buffer : deviceBuffers) {
        ocl.allocator().release(buffer);
    }
    ocl.allocator().release(dy);
    return got;
}

// 2x2 ones kernel, stride 1, no padding, no bias: all nine cells hand-computed.
TEST(NertOpenCLConvTranspose, ConvTranspose2x2) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = ConvTransposeNode("ct2x2");
    SetIntList(node, "strides", {1, 1});

    // CPU oracle: y[oh][ow] sums the overlapping input window (kernel is ones).
    std::vector<float> ref(9, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(
        node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})},
        {HostFloat(ref, {1, 1, 3, 3})});
    EXPECT_EQ(ref, (std::vector<float>{1.0F, 3.0F, 2.0F, 4.0F, 10.0F, 6.0F, 3.0F, 7.0F, 4.0F}));
    ASSERT_EQ(ref, Stride1Expected());

    const std::vector<float> got =
        RunDevice(ocl, node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})}, {1, 1, 3, 3});
    ExpectNear(got, ref, 1e-6F);
}

// Stride 2: each input cell lands on a 2x2 output block; hand-checked cells.
TEST(NertOpenCLConvTranspose, ConvTransposeStride2) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = ConvTransposeNode("ct_stride2");
    SetIntList(node, "strides", {2, 2});

    std::vector<float> ref(16, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(
        node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})},
        {HostFloat(ref, {1, 1, 4, 4})});
    // Hand anchors: x00 covers rows/cols 0-1, x01 cols 2-3, x11 rows/cols 2-3.
    EXPECT_EQ(ref[0], 1.0F);  // x00
    EXPECT_EQ(ref[3], 2.0F);  // x01
    EXPECT_EQ(ref[5], 1.0F);  // x00
    EXPECT_EQ(ref[10], 4.0F); // x11
    EXPECT_EQ(ref[15], 4.0F); // x11
    ASSERT_EQ(ref, (std::vector<float>{1.0F, 1.0F, 2.0F, 2.0F, 1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F, 3.0F,
                                       3.0F, 4.0F, 4.0F}));

    const std::vector<float> got =
        RunDevice(ocl, node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})}, {1, 1, 4, 4});
    ExpectNear(got, ref, 1e-6F);
}

// Rank-1 bias [10] shifts every output cell by +10.
TEST(NertOpenCLConvTranspose, ConvTransposeWithBias) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = ConvTransposeNode("ct_bias");
    node.inputs = {"x", "w", "b"};
    SetIntList(node, "strides", {1, 1});
    std::vector<float> bias = {10.0F};

    std::vector<float> ref(9, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(
        node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2}), HostFloat(bias, {1})},
        {HostFloat(ref, {1, 1, 3, 3})});
    EXPECT_EQ(ref, (std::vector<float>{11.0F, 13.0F, 12.0F, 14.0F, 20.0F, 16.0F, 13.0F, 17.0F, 14.0F}));
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_FLOAT_EQ(ref[i], Stride1Expected()[i] + 10.0F) << "i=" << i;
    }

    const std::vector<float> got = RunDevice(
        ocl, node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2}), HostFloat(bias, {1})},
        {1, 1, 3, 3});
    ExpectNear(got, ref, 1e-6F);
}

// The output_shape attribute fixes the output view; the row fills exactly it.
TEST(NertOpenCLConvTranspose, ConvTransposeOutputShape) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    nert::Node node = ConvTransposeNode("ct_outshape");
    SetIntList(node, "strides", {1, 1});
    // The transposed window for this fixture computes {1,1,3,3}; the attribute
    // pins the same dims (the oracle derives the size from the formula alone
    // and requires the output view to match it).
    SetIntList(node, "output_shape", {1, 1, 3, 3});
    const std::vector<int64_t> outDims = {1, 1, 3, 3};

    std::vector<float> ref(9, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(
        node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})}, {HostFloat(ref, outDims)});
    ASSERT_EQ(ref, Stride1Expected());

    const std::vector<float> got =
        RunDevice(ocl, node, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})}, outDims);
    ExpectNear(got, ref, 1e-6F);
}

// Negative paths name the op, the node, and the oracle's reason.
TEST(NertOpenCLConvTranspose, Negatives) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    auto expectThrow = [&](const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                           const std::vector<nert::TensorView>& outputs, const std::string& fragment) {
        try {
            nert::opencl_kernels::ConvTranspose(node, inputs, outputs);
            FAIL() << "expected InferenceException for " << node.name;
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find("ConvTranspose"), std::string::npos) << message;
            EXPECT_NE(message.find(node.name), std::string::npos) << message;
            EXPECT_NE(message.find(fragment), std::string::npos) << message;
        }
    };

    // Wrong input arity: only the input tensor is given.
    nert::Node arity = ConvTransposeNode("ct_arity");
    expectThrow(arity, {HostFloat(FixtureX(), {1, 1, 2, 2})}, {View(nert::DataType::Float32, {1, 1, 3, 3})},
                "expected input, weight, optional bias, and one output");

    // Wrong dtype on X.
    nert::Node dtype = ConvTransposeNode("ct_dtype");
    expectThrow(dtype, {View(nert::DataType::Int64, {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})},
                {View(nert::DataType::Float32, {1, 1, 3, 3})}, "tensors must be float32");

    // Output dims mismatch: stride 1 computes 3x3, the view says 4x4.
    nert::Node dims = ConvTransposeNode("ct_dims");
    SetIntList(dims, "strides", {1, 1});
    expectThrow(dims, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})},
                {View(nert::DataType::Float32, {1, 1, 4, 4})}, "output shape does not match the transposed window");

    // Unsupported group: two groups over one input channel.
    nert::Node group = ConvTransposeNode("ct_group");
    SetIntList(group, "strides", {1, 1});
    SetInt(group, "group", 2);
    expectThrow(group, {HostFloat(FixtureX(), {1, 1, 2, 2}), HostFloat(FixtureW(), {1, 1, 2, 2})},
                {View(nert::DataType::Float32, {1, 2, 3, 3})}, "weight channels do not match the input and group");

    // Missing weight: the weight view carries no buffer.
    nert::Node missing = ConvTransposeNode("ct_missing");
    SetIntList(missing, "strides", {1, 1});
    expectThrow(missing, {HostFloat(FixtureX(), {1, 1, 2, 2}), View(nert::DataType::Float32, {1, 1, 2, 2})},
                {View(nert::DataType::Float32, {1, 1, 3, 3})}, "null buffer");
}

// Zero-extent dims are rejected by the dimension check, naming the op.
TEST(NertOpenCLConvTranspose, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = ConvTransposeNode("ct_zero");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {1, 1, 0, 2};
    try {
        nert::opencl_kernels::ConvTranspose(node, {in, View(nert::DataType::Float32, {1, 1, 2, 2})},
                                            {View(nert::DataType::Float32, {1, 1, 3, 3})});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("ConvTranspose"), std::string::npos) << message;
    }
}

} // namespace
