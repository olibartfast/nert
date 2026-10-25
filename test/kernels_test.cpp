// CPU kernel tests: Relu, Add, Reshape, ReduceMean, Mul, Div, Sub, Sigmoid,
// Cast, Softmax, ReduceMax, Mod, Equal, Where, Concat, Split, Unsqueeze,
// Expand, Transpose, Slice, Gather, GatherElements, Resize, Flatten, Shape,
// ConstantOfShape, TopK, Floor, Identity, Pow, Tile, AveragePool,
// ConvTranspose, Erf, Sqrt, Tanh, IsNaN, Squeeze, LayerNormalization,
// InstanceNormalization, Abs, Neg, Exp, Log, Cos, Sin, Clip, Greater, Less,
// And, Not, ReduceSum, Range, Pad, GridSample, and Einsum.
//
// Expected values are hand-computed small cases written inline, not captured
// from another runtime. Kernels are reached through the CPU device's kernel
// table (CpuDevice().kernels().find(...)), the same way the executor will.

#include "nert/device.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {

// A float32 tensor view over `data`, with its concrete shape.
nert::TensorView FloatView(std::vector<float>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Float32;
    view.dims = std::move(dims);
    return view;
}

// An int64 tensor view over `data`, with its concrete shape.
nert::TensorView Int64View(std::vector<int64_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

// A bool tensor view over `data` (one byte per element, 0 or 1), with its
// concrete shape.
nert::TensorView BoolView(std::vector<std::uint8_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.empty() ? nullptr : data.data();
    view.dtype = nert::DataType::Bool;
    view.dims = std::move(dims);
    return view;
}

// Reads a bool output buffer back as bytes (0 or 1) for comparison.
std::vector<std::uint8_t> BoolBytes(const std::vector<std::uint8_t>& data) {
    const bool* flags = reinterpret_cast<const bool*>(data.data());
    std::vector<std::uint8_t> out(data.size(), 0);
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[i] = flags[i] ? std::uint8_t{1} : std::uint8_t{0};
    }
    return out;
}

nert::Node MakeNode(const std::string& name, const std::string& op_type) {
    nert::Node node;
    node.name = name;
    node.op_type = op_type;
    return node;
}

} // namespace

// Relu clamps a mixed-sign vector at zero.
TEST(NertKernels, ReluMixedSignVector) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Relu");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {-2.0f, -0.5f, 0.0f, 3.0f, 1.5f};
    std::vector<float> out(in.size(), 0.0f);
    nert::Node node = MakeNode("relu0", "Relu");
    std::vector<nert::TensorView> inputs = {FloatView(in, {5})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {5})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{0.0f, 0.0f, 0.0f, 3.0f, 1.5f}));
}

// Add broadcasts a trailing [3] input across a [2,3] input.
TEST(NertKernels, AddBroadcastsLowerRank) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Add");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("add0", "Add");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f}));
}

// Add broadcasts [2,1] against [1,3] into [2,3].
TEST(NertKernels, AddBroadcastsBothAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Add");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("add1", "Add");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f}));
}

// Reshape reorders [2,3] into [3,2] without touching the flat buffer.
TEST(NertKernels, ReshapeReordersDims) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    nert::Node node = MakeNode("reshape0", "Reshape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Reshape changes rank to a flat [6] and validates an int64 shape operand.
TEST(NertKernels, ReshapeRankChangeWithShapeOperand) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    std::vector<int64_t> shape = {6};
    nert::Node node = MakeNode("reshape1", "Reshape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(shape, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {6})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Reshape infers a single -1 from the element count (the /model.23/Reshape
// fixture pattern: [1,4,2,2] with shape [1,4,-1] resolves to [1,4,4]).
TEST(NertKernels, ReshapeInfersMinusOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                             9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> out(in.size(), 0.0f);
    std::vector<int64_t> shape = {1, 4, -1};
    nert::Node node = MakeNode("reshape_minus1", "Reshape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 4, 2, 2}), Int64View(shape, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 4, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

// Reshape copies the input dimension for a 0 entry when allowzero is unset.
TEST(NertKernels, ReshapeZeroCopiesInputDim) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    std::vector<int64_t> shape = {0, 3};
    nert::Node node = MakeNode("reshape_zero", "Reshape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// ReduceMean over axis 1 of [2,3] with the default keepdims == 1.
TEST(NertKernels, ReduceMeanOneAxisKeepdimsDefault) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    nert::Node node = MakeNode("reduce0", "ReduceMean");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.0f, 5.0f}));
}

// ReduceMean over axis 1 with keepdims == 0 squeezes the reduced axis.
TEST(NertKernels, ReduceMeanOneAxisKeepdimsZero) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    nert::Node node = MakeNode("reduce1", "ReduceMean");
    node.attributes.emplace_back("keepdims", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.0f, 5.0f}));
}

// ReduceMean with no axes input reduces every axis, leaving [1,1] by default.
TEST(NertKernels, ReduceMeanAllAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("reduce2", "ReduceMean");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.5f}));
}

// A present-but-empty axes tensor is the ONNX way to express an empty axes
// operand; with noop_with_empty_axes it is an identity copy, not a dimension
// error.
TEST(NertKernels, ReduceMeanEmptyAxesNoopIdentity) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> axes;
    nert::Node node = MakeNode("reduce_empty_noop", "ReduceMean");
    node.attributes.emplace_back("noop_with_empty_axes", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(axes, {0})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

// An omitted optional axes operand arrives as a TensorView with Unknown
// dtype and must behave like an absent input: reduce every axis.
TEST(NertKernels, ReduceMeanOmittedAxesUnknownDtype) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("reduce_omitted", "ReduceMean");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), nert::TensorView{}};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.5f}));
}

// ReduceMax has no identity form here; a present empty axes input reduces
// every axis, matching its absent form.
TEST(NertKernels, ReduceMaxEmptyAxesReducesAll) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 4.0f, 3.0f, 2.0f};
    std::vector<float> out(1, 0.0f);
    std::vector<int64_t> axes;
    nert::Node node = MakeNode("max_empty", "ReduceMax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(axes, {0})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f}));
}

// A pre-18 ReduceMean carries axes in an attribute and has no axes input. The
// kernel reads that form in place; the result matches the schema-18 input
// form.
TEST(NertKernels, ReduceMeanOldAttributeFormMatchesInputForm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> axes = {1};

    std::vector<float> old_out(2, 0.0f);
    nert::Node old_node = MakeNode("reduce_attr", "ReduceMean");
    old_node.schema_version = 13;
    old_node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    kernel(old_node, {FloatView(in, {2, 3})}, {FloatView(old_out, {2, 1})});

    std::vector<float> new_out(2, 0.0f);
    nert::Node new_node = MakeNode("reduce_input", "ReduceMean");
    new_node.schema_version = 18;
    kernel(new_node, {FloatView(in, {2, 3}), Int64View(axes, {1})}, {FloatView(new_out, {2, 1})});

    EXPECT_EQ(old_out, (std::vector<float>{2.0f, 5.0f}));
    EXPECT_EQ(old_out, new_out);
}

// Pre-18 ReduceL2 with axes in an attribute matches the input form.
TEST(NertKernels, ReduceL2OldAttributeFormMatchesInputForm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceL2");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {3.0f, 4.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<int64_t> axes = {1};

    std::vector<float> old_out(3, -1.0f);
    nert::Node old_node = MakeNode("reduce_l2_attr", "ReduceL2");
    old_node.schema_version = 13;
    old_node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    kernel(old_node, {FloatView(in, {3, 2})}, {FloatView(old_out, {3, 1})});

    std::vector<float> new_out(3, -1.0f);
    nert::Node new_node = MakeNode("reduce_l2_input", "ReduceL2");
    new_node.schema_version = 18;
    kernel(new_node, {FloatView(in, {3, 2}), Int64View(axes, {1})}, {FloatView(new_out, {3, 1})});

    EXPECT_FLOAT_EQ(old_out[0], 5.0f);
    EXPECT_FLOAT_EQ(old_out[1], 0.0f);
    EXPECT_NEAR(old_out[2], static_cast<float>(std::sqrt(2.0)), 1e-6f);
    EXPECT_EQ(old_out, new_out);
}

// Pre-18 ReduceMax with axes in an attribute matches the input form.
TEST(NertKernels, ReduceMaxOldAttributeFormMatchesInputForm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    std::vector<int64_t> axes = {1};

    std::vector<float> old_out(2, 0.0f);
    nert::Node old_node = MakeNode("reduce_max_attr", "ReduceMax");
    old_node.schema_version = 13;
    old_node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    kernel(old_node, {FloatView(in, {2, 3})}, {FloatView(old_out, {2, 1})});

    std::vector<float> new_out(2, 0.0f);
    nert::Node new_node = MakeNode("reduce_max_input", "ReduceMax");
    new_node.schema_version = 18;
    kernel(new_node, {FloatView(in, {2, 3}), Int64View(axes, {1})}, {FloatView(new_out, {2, 1})});

    EXPECT_EQ(old_out, (std::vector<float>{5.0f, 6.0f}));
    EXPECT_EQ(old_out, new_out);
}

// Gemm computes the plain matrix product without transposes or scaling.
TEST(NertKernels, GemmPlainProduct) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("gemm0", "Gemm");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

// Gemm transA transposes a [K,M] lhs, leaving the same product as the plain case.
TEST(NertKernels, GemmTransA) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("gemm_transA", "Gemm");
    node.attributes.emplace_back("transA", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(a, {3, 2}), FloatView(b, {3, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{89.0f, 98.0f, 116.0f, 128.0f}));
}

// Gemm transB transposes a [N,K] rhs, matching the plain product.
TEST(NertKernels, GemmTransB) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("gemm_transB", "Gemm");
    node.attributes.emplace_back("transB", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

// Gemm scales by alpha, adds beta times a rank-1 [N] bias, and broadcasts it.
TEST(NertKernels, GemmAlphaBetaRankOneBias) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> c = {10.0f, 20.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("gemm_scale", "Gemm");
    node.attributes.emplace_back("alpha", 2.0f);
    node.attributes.emplace_back("beta", 0.5f);
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 2}), FloatView(b, {2, 2}), FloatView(c, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{43.0f, 54.0f, 91.0f, 110.0f}));
}

// MatMul on two rank-2 operands contracts the shared inner dimension.
TEST(NertKernels, MatMulTwoDimensional) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("matmul0", "MatMul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{58.0f, 64.0f, 139.0f, 154.0f}));
}

// MatMul on rank-3 operands multiplies each batch independently.
TEST(NertKernels, MatMulBatchedThreeDimensional) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f};
    std::vector<float> out(8, 0.0f);
    nert::Node node = MakeNode("matmul_batch", "MatMul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 2, 3}), FloatView(b, {2, 3, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 5.0f, 10.0f, 11.0f, 2.0f, 0.0f, 0.0f, 2.0f}));
}

// MatMul promotes a 1-D lhs to [1,K] and drops the promoted axis.
TEST(NertKernels, MatMulPromotesOneDimensionalLhs) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("matmul_lhs1d", "MatMul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 5.0f}));
}

// MatMul promotes a 1-D rhs to [K,1] and drops the promoted axis.
TEST(NertKernels, MatMulPromotesOneDimensionalRhs) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {1.0f, 0.0f, 1.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("matmul_rhs1d", "MatMul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 10.0f}));
}

// Conv with a 1x1 identity weight copies the input through unchanged.
TEST(NertKernels, ConvIdentityOneByOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> w = {1.0f};
    std::vector<float> out(9, 0.0f);
    nert::Node node = MakeNode("conv_id", "Conv");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 3, 3}), FloatView(w, {1, 1, 1, 1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 3, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}));
}

// Conv with a 2x2 all-ones kernel sums each 2x2 input window.
TEST(NertKernels, ConvKnownTwoByTwoKernel) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("conv_2x2", "Conv");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 3, 3}), FloatView(w, {1, 1, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{12.0f, 16.0f, 24.0f, 28.0f}));
}

// Conv adds a rank-1 [M] bias to every output channel.
TEST(NertKernels, ConvWithBias) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> b = {0.5f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("conv_bias", "Conv");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 3, 3}), FloatView(w, {1, 1, 2, 2}), FloatView(b, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{12.5f, 16.5f, 24.5f, 28.5f}));
}

// Conv with group == C is depthwise: each output channel sees only its own
// input channel.
TEST(NertKernels, ConvDepthwiseGroupEqualsChannels) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> w = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("conv_dw", "Conv");
    node.attributes.emplace_back("group", int64_t{2});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2, 2, 2}), FloatView(w, {2, 1, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 2, 1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 50.0f}));
}

// Conv with stride 2 and a one-cell pad samples the padded input on the
// computation lattice.
TEST(NertKernels, ConvStrideAndPadding) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("conv_stride_pad", "Conv");
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 4, 4}), FloatView(w, {1, 1, 3, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{14.0f, 30.0f, 57.0f, 99.0f}));
}

// MaxPool with a 2x2 window and stride 2 takes the max of each disjoint block.
TEST(NertKernels, MaxPoolTwoByTwoStrideTwo) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MaxPool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("pool_2x2", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 4, 4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{6.0f, 8.0f, 14.0f, 16.0f}));
}

// MaxPool rounds the output up with ceil_mode and ignores the padded border.
TEST(NertKernels, MaxPoolPaddingAndCeilMode) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MaxPool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("pool_pad_ceil", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{0, 0, 1, 1});
    node.attributes.emplace_back("ceil_mode", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 3, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f, 8.0f, 9.0f}));
}

// Mul broadcasts a trailing [3] input across a [2,3] input.
TEST(NertKernels, MulBroadcastsLowerRank) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("mul0", "Mul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{10.0f, 40.0f, 90.0f, 40.0f, 100.0f, 180.0f}));
}

// Mul broadcasts [2,1] against [1,3] into [2,3].
TEST(NertKernels, MulBroadcastsBothAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("mul1", "Mul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{20.0f, 40.0f, 60.0f, 30.0f, 60.0f, 90.0f}));
}

// Div divides elementwise without broadcasting.
TEST(NertKernels, DivPlainQuotient) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {6.0f, 8.0f, 10.0f};
    std::vector<float> b = {2.0f, 4.0f, 5.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("div0", "Div");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{3.0f, 2.0f, 2.0f}));
}

// Div broadcasts a trailing [3] divisor across a [2,3] dividend.
TEST(NertKernels, DivBroadcastsLowerRank) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {6.0f, 12.0f, 18.0f, 24.0f, 30.0f, 36.0f};
    std::vector<float> b = {2.0f, 3.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("div1", "Div");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{3.0f, 4.0f, 3.0f, 12.0f, 10.0f, 6.0f}));
}

// Div by zero follows IEEE float semantics: inf, -inf, nan, never a throw.
TEST(NertKernels, DivByZeroYieldsInfAndNan) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, -1.0f, 0.0f, 6.0f};
    std::vector<float> b = {0.0f, 0.0f, 0.0f, 3.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("div_zero", "Div");
    std::vector<nert::TensorView> inputs = {FloatView(a, {4}), FloatView(b, {4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_TRUE(std::isinf(out[0]) && out[0] > 0.0f);
    EXPECT_TRUE(std::isinf(out[1]) && out[1] < 0.0f);
    EXPECT_TRUE(std::isnan(out[2]));
    EXPECT_EQ(out[3], 2.0f);
}

// Sub subtracts elementwise without broadcasting.
TEST(NertKernels, SubPlainDifference) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sub");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {5.0f, 7.0f, 9.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("sub0", "Sub");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{4.0f, 5.0f, 6.0f}));
}

// Sub broadcasts [2,1] against [1,3] into [2,3].
TEST(NertKernels, SubBroadcastsBothAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sub");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {10.0f, 20.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("sub1", "Sub");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{9.0f, 8.0f, 7.0f, 19.0f, 18.0f, 17.0f}));
}

// Sigmoid maps 0 to 0.5 and symmetric inputs to complementary outputs.
TEST(NertKernels, SigmoidBasicValues) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sigmoid");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {0.0f, 2.0f, -2.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("sigmoid0", "Sigmoid");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[0], 0.5f);
    EXPECT_NEAR(out[1], 0.88079708f, 1e-6f);
    EXPECT_NEAR(out[2], 0.11920292f, 1e-6f);
}

// Sigmoid saturates: large inputs approach 1, large negatives approach 0.
TEST(NertKernels, SigmoidSaturatesForLargeMagnitudes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sigmoid");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {10.0f, -10.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("sigmoid1", "Sigmoid");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_NEAR(out[0], 0.9999546f, 1e-6f);
    EXPECT_NEAR(out[1], 0.0000453979f, 1e-8f);
}

// Cast converts float32 to int64 with truncation toward zero.
TEST(NertKernels, CastFloatToInt64Truncates) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.9f, -2.7f, 0.0f, 3.0f};
    std::vector<int64_t> out(4, 0);
    nert::Node node = MakeNode("cast0", "Cast");
    node.attributes.emplace_back("to", int64_t{7});
    std::vector<nert::TensorView> inputs = {FloatView(in, {4})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, -2, 0, 3}));
}

// Cast converts int64 to float32 exactly for small magnitudes.
TEST(NertKernels, CastInt64ToFloat) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, -2, 0};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("cast1", "Cast");
    node.attributes.emplace_back("to", int64_t{1});
    std::vector<nert::TensorView> inputs = {Int64View(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, -2.0f, 0.0f}));
}

// Cast converts float32 to bool: only exact zero maps to false.
TEST(NertKernels, CastFloatToBool) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {0.0f, 2.5f, -1.0f};
    std::vector<std::uint8_t> out(3, 0);
    nert::Node node = MakeNode("cast2", "Cast");
    node.attributes.emplace_back("to", int64_t{9});
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 1, 1}));
}

// Cast converts bool to int64 as 1/0.
TEST(NertKernels, CastBoolToInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1};
    std::vector<int64_t> out(3, 0);
    nert::Node node = MakeNode("cast3", "Cast");
    node.attributes.emplace_back("to", int64_t{7});
    std::vector<nert::TensorView> inputs = {BoolView(in, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 0, 1}));
}

// Cast converts int64 to bool: only zero maps to false.
TEST(NertKernels, CastInt64ToBool) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {0, 5, -3};
    std::vector<std::uint8_t> out(3, 0);
    nert::Node node = MakeNode("cast4", "Cast");
    node.attributes.emplace_back("to", int64_t{9});
    std::vector<nert::TensorView> inputs = {Int64View(in, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 1, 1}));
}

// Cast converts bool to float32 as 1.0/0.0.
TEST(NertKernels, CastBoolToFloat) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("cast5", "Cast");
    node.attributes.emplace_back("to", int64_t{1});
    std::vector<nert::TensorView> inputs = {BoolView(in, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 0.0f}));
}

// Cast to=6 emulates int32 on int64 lanes: truncation toward zero with
// saturation at the int32 range (NaN becomes 0). Matches ORT bit for bit
// in range; past the range it follows portable saturation (ARM ORT)
// rather than the x86 hardware-conversion quirk — see the kernel doc.
TEST(NertKernels, CastFloatToInt32Emulation) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> in = {1.9f, -2.7f, 0.0f, 3e9f, -3e9f, nan};
    std::vector<int64_t> out(in.size(), 0);
    nert::Node node = MakeNode("cast6", "Cast");
    node.attributes.emplace_back("to", int64_t{6});
    std::vector<nert::TensorView> inputs = {FloatView(in, {static_cast<int64_t>(in.size())})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {static_cast<int64_t>(out.size())})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, -2, 0, 2147483647LL, -2147483648LL, 0}));
}

// Softmax over the default last axis normalizes each row of [2,3].
TEST(NertKernels, SoftmaxDefaultLastAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("softmax0", "Softmax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_NEAR(out[0], 0.09003057f, 1e-6f);
    EXPECT_NEAR(out[1], 0.24472847f, 1e-6f);
    EXPECT_NEAR(out[2], 0.66524096f, 1e-6f);
    EXPECT_NEAR(out[3], 0.09003057f, 1e-6f);
    EXPECT_NEAR(out[4], 0.24472847f, 1e-6f);
    EXPECT_NEAR(out[5], 0.66524096f, 1e-6f);
}

// Softmax over axis 0 normalizes each column of [2,3]; every column differs
// by 3, so every column shares the same pair of outputs.
TEST(NertKernels, SoftmaxFirstAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("softmax1", "Softmax");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_NEAR(out[0], 0.04742587f, 1e-6f);
    EXPECT_NEAR(out[1], 0.04742587f, 1e-6f);
    EXPECT_NEAR(out[2], 0.04742587f, 1e-6f);
    EXPECT_NEAR(out[3], 0.95257413f, 1e-6f);
    EXPECT_NEAR(out[4], 0.95257413f, 1e-6f);
    EXPECT_NEAR(out[5], 0.95257413f, 1e-6f);
}

// Pre-13 Softmax coerces to 2D at the default axis 1, so a [2,2,3] input
// becomes [2,6] and each six-element row is normalized.
TEST(NertKernels, SoftmaxLegacyDefaultAxisCoerces2D) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(12, 0.0f);
    std::vector<float> out(12, 0.0f);
    nert::Node node = MakeNode("softmax_legacy", "Softmax");
    node.schema_version = 11;
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2, 3})};

    kernel(node, inputs, outputs);

    for (const float value : out) {
        EXPECT_NEAR(value, 1.0f / 6.0f, 1e-6f);
    }
}

// ReduceMax over axis 1 of [2,3] with the default keepdims == 1.
TEST(NertKernels, ReduceMaxOneAxisKeepdimsDefault) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    nert::Node node = MakeNode("reducemax0", "ReduceMax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f}));
}

// ReduceMax over axis 1 with keepdims == 0 squeezes the reduced axis.
TEST(NertKernels, ReduceMaxOneAxisKeepdimsZero) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    nert::Node node = MakeNode("reducemax1", "ReduceMax");
    node.attributes.emplace_back("keepdims", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f}));
}

// ReduceMax with no axes input reduces every axis, leaving [1,1] by default.
TEST(NertKernels, ReduceMaxAllAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("reducemax2", "ReduceMax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f}));
}

// ReduceMax accepts a negative axis: -1 names the last axis of [2,3].
TEST(NertKernels, ReduceMaxNegativeAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {-1};
    nert::Node node = MakeNode("reducemax3", "ReduceMax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f}));
}

// Mod takes the truncating remainder on int64; the sign follows the dividend.
TEST(NertKernels, ModDividendSign) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, -7, 7, -7};
    std::vector<int64_t> b = {3, 3, -3, -3};
    std::vector<int64_t> out(4, 0);
    nert::Node node = MakeNode("mod0", "Mod");
    std::vector<nert::TensorView> inputs = {Int64View(a, {4}), Int64View(b, {4})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, -1, 1, -1}));
}

// Mod broadcasts [2,1] against [3] into [2,3].
TEST(NertKernels, ModBroadcastsLowerRank) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 4, 5};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("mod1", "Mod");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2, 1}), Int64View(b, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 3, 2, 2, 0, 3}));
}

// Equal compares float32 vectors elementwise into a bool output.
TEST(NertKernels, EqualFloatVectors) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Equal");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 0.0f, 3.0f};
    std::vector<std::uint8_t> out(3, 0);
    nert::Node node = MakeNode("equal0", "Equal");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 0, 1}));
}

// Equal broadcasts int64 [2,1] against [3] into a [2,3] bool output.
TEST(NertKernels, EqualInt64Broadcast) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Equal");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {5, 6};
    std::vector<int64_t> b = {5, 5, 5};
    std::vector<std::uint8_t> out(6, 0);
    nert::Node node = MakeNode("equal1", "Equal");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2, 1}), Int64View(b, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 1, 1, 0, 0, 0}));
}

// Where selects elementwise between two flat vectors.
TEST(NertKernels, WhereFlatSelect) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> cond = {1, 0, 1};
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {10.0f, 20.0f, 30.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("where0", "Where");
    std::vector<nert::TensorView> inputs = {BoolView(cond, {3}), FloatView(x, {3}), FloatView(y, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 20.0f, 3.0f}));
}

// Where broadcasts all three inputs: cond [2,1], X [3], scalar Y into [2,3].
TEST(NertKernels, WhereBroadcastsAllThree) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> cond = {1, 0};
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y = {9.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("where1", "Where");
    std::vector<nert::TensorView> inputs = {BoolView(cond, {2, 1}), FloatView(x, {3}), FloatView(y, {})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 9.0f, 9.0f, 9.0f}));
}

// The op-type table resolves the implemented kernels and rejects unknown ops.
TEST(NertKernels, TableResolvesKnownOpsAndRejectsUnknown) {
    const nert::KernelTable& table = nert::CpuDevice().kernels();
    EXPECT_NE(table.find("Relu"), nullptr);
    EXPECT_NE(table.find("Add"), nullptr);
    EXPECT_NE(table.find("Reshape"), nullptr);
    EXPECT_NE(table.find("ReduceMean"), nullptr);
    EXPECT_NE(table.find("Gemm"), nullptr);
    EXPECT_NE(table.find("MatMul"), nullptr);
    EXPECT_NE(table.find("Conv"), nullptr);
    EXPECT_NE(table.find("MaxPool"), nullptr);
    EXPECT_NE(table.find("Mul"), nullptr);
    EXPECT_NE(table.find("Div"), nullptr);
    EXPECT_NE(table.find("Sub"), nullptr);
    EXPECT_NE(table.find("Sigmoid"), nullptr);
    EXPECT_NE(table.find("Cast"), nullptr);
    EXPECT_NE(table.find("Softmax"), nullptr);
    EXPECT_NE(table.find("ReduceMax"), nullptr);
    EXPECT_NE(table.find("Mod"), nullptr);
    EXPECT_NE(table.find("Equal"), nullptr);
    EXPECT_NE(table.find("Where"), nullptr);
    EXPECT_NE(table.find("Concat"), nullptr);
    EXPECT_NE(table.find("Split"), nullptr);
    EXPECT_NE(table.find("Unsqueeze"), nullptr);
    EXPECT_NE(table.find("Expand"), nullptr);
    EXPECT_NE(table.find("Transpose"), nullptr);
    EXPECT_NE(table.find("Slice"), nullptr);
    EXPECT_NE(table.find("Gather"), nullptr);
    EXPECT_NE(table.find("GatherElements"), nullptr);
    EXPECT_NE(table.find("Resize"), nullptr);
    EXPECT_NE(table.find("Flatten"), nullptr);
    EXPECT_NE(table.find("Shape"), nullptr);
    EXPECT_NE(table.find("ConstantOfShape"), nullptr);
    EXPECT_NE(table.find("TopK"), nullptr);
    EXPECT_EQ(table.find("NoSuchOp"), nullptr);
}

// A float32-expected op rejects an int64 data input, naming the op.
TEST(NertKernelsNegative, ReluRejectsDtypeMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Relu");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("relu_bad", "Relu");
    std::vector<nert::TensorView> inputs = {Int64View(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Relu"), std::string::npos) << error.what();
    }
}

// Add rejects the wrong input arity, naming the op.
TEST(NertKernelsNegative, AddRejectsWrongArity) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Add");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("add_bad", "Add");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Add"), std::string::npos) << error.what();
    }
}

// Reshape rejects an element-count mismatch, naming the op.
TEST(NertKernelsNegative, ReshapeRejectsSizeMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(5, 0.0f);
    nert::Node node = MakeNode("reshape_bad", "Reshape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {5})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Reshape"), std::string::npos) << error.what();
    }
}

// Reshape with allowzero keeps a literal 0, so [0,3] resolves to a zero
// product and is rejected for six input elements.
TEST(NertKernelsNegative, ReshapeAllowzeroKeepsZero) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    std::vector<int64_t> shape = {0, 3};
    nert::Node node = MakeNode("reshape_allowzero", "Reshape");
    node.attributes.emplace_back("allowzero", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Reshape"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("does not match"), std::string::npos) << error.what();
    }
}

// Reshape rejects two -1 entries, naming the op.
TEST(NertKernelsNegative, ReshapeRejectsTwoMinusOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    std::vector<int64_t> shape = {-1, -1};
    nert::Node node = MakeNode("reshape_two_minus1", "Reshape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Reshape"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("at most one -1"), std::string::npos) << error.what();
    }
}

// Reshape rejects a -1 that does not divide the element count (6 is not
// divisible by 4), naming the op.
TEST(NertKernelsNegative, ReshapeRejectsIndivisibleMinusOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Reshape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(in.size(), 0.0f);
    std::vector<int64_t> shape = {4, -1};
    nert::Node node = MakeNode("reshape_indivisible", "Reshape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Reshape"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("does not match"), std::string::npos) << error.what();
    }
}

// Gemm rejects a contraction-dimension mismatch, naming the op.
TEST(NertKernelsNegative, GemmRejectsInnerDimMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("gemm_bad", "Gemm");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Gemm"), std::string::npos) << error.what();
    }
}

// MatMul rejects a contraction-dimension mismatch, naming the op.
TEST(NertKernelsNegative, MatMulRejectsInnerDimMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("matmul_bad", "MatMul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("MatMul"), std::string::npos) << error.what();
    }
}

// Conv rejects a weight whose channel dimension disagrees with input/group.
TEST(NertKernelsNegative, ConvRejectsChannelMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("conv_bad", "Conv");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2, 3, 2}), FloatView(w, {1, 1, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Conv"), std::string::npos) << error.what();
    }
}

// MaxPool rejects a declared output shape that does not match its window.
TEST(NertKernelsNegative, MaxPoolRejectsShapeMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MaxPool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> out(9, 0.0f);
    nert::Node node = MakeNode("pool_bad", "MaxPool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 3, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 3, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("MaxPool"), std::string::npos) << error.what();
    }
}

// Mul rejects mixed float/int inputs, naming the op.
TEST(NertKernelsNegative, MulRejectsMixedDtypes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<int64_t> b = {1, 2, 3};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("mul_bad", "Mul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3}), Int64View(b, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Mul"), std::string::npos) << error.what();
    }
}

// Div rejects shapes that cannot broadcast, naming the op.
TEST(NertKernelsNegative, DivRejectsBroadcastMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("div_bad", "Div");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Div"), std::string::npos) << error.what();
    }
}

// Sub rejects the wrong input arity, naming the op.
TEST(NertKernelsNegative, SubRejectsWrongArity) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sub");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("sub_bad", "Sub");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Sub"), std::string::npos) << error.what();
    }
}

// Sigmoid rejects an int64 data input, naming the op.
TEST(NertKernelsNegative, SigmoidRejectsDtypeMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sigmoid");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("sigmoid_bad", "Sigmoid");
    std::vector<nert::TensorView> inputs = {Int64View(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Sigmoid"), std::string::npos) << error.what();
    }
}

// Cast rejects an unsupported `to` code, naming the op.
TEST(NertKernelsNegative, CastRejectsUnsupportedTo) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("cast_bad", "Cast");
    node.attributes.emplace_back("to", int64_t{6});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Cast"), std::string::npos) << error.what();
    }
}

// Cast rejects a missing `to` attribute, naming the op.
TEST(NertKernelsNegative, CastRejectsMissingTo) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cast");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("cast_noto", "Cast");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Cast"), std::string::npos) << error.what();
    }
}

// Softmax rejects an out-of-range axis, naming the op.
TEST(NertKernelsNegative, SoftmaxRejectsAxisOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("softmax_bad", "Softmax");
    node.attributes.emplace_back("axis", int64_t{2});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Softmax"), std::string::npos) << error.what();
    }
}

// ReduceMax rejects a duplicated axis, naming the op.
TEST(NertKernelsNegative, ReduceMaxRejectsDuplicateAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    std::vector<int64_t> axes = {0, 0};
    nert::Node node = MakeNode("reducemax_bad", "ReduceMax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(axes, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ReduceMax"), std::string::npos) << error.what();
    }
}

// Mod rejects a zero divisor, naming the op.
TEST(NertKernelsNegative, ModRejectsZeroDivisor) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 0};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("mod_bad", "Mod");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Mod"), std::string::npos) << error.what();
    }
}

// Mod rejects fmod=1, naming the op.
TEST(NertKernelsNegative, ModRejectsFmodOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mod");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 4};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("mod_fmod", "Mod");
    node.attributes.emplace_back("fmod", int64_t{1});
    std::vector<nert::TensorView> inputs = {Int64View(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Mod"), std::string::npos) << error.what();
    }
}

// Equal rejects mixed float/int inputs, naming the op.
TEST(NertKernelsNegative, EqualRejectsMixedDtypes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Equal");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<int64_t> b = {1, 2};
    std::vector<std::uint8_t> out(2, 0);
    nert::Node node = MakeNode("equal_bad", "Equal");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Equal"), std::string::npos) << error.what();
    }
}

// Where rejects a non-bool condition, naming the op.
TEST(NertKernelsNegative, WhereRejectsNonBoolCondition) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> cond = {1.0f, 0.0f};
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> y = {10.0f, 20.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("where_bad", "Where");
    std::vector<nert::TensorView> inputs = {FloatView(cond, {2}), FloatView(x, {2}), FloatView(y, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Where"), std::string::npos) << error.what();
    }
}

// Concat joins [2,2] and [2,1] along axis 1 into [2,3].
TEST(NertKernels, ConcatAxisOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("concat0", "Concat");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 2}), FloatView(b, {2, 1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f}));
}

// Concat with axis -1 names the last axis; int64 data path.
TEST(NertKernels, ConcatNegativeAxisInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {3, 4, 5, 6};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("concat1", "Concat");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<nert::TensorView> inputs = {Int64View(a, {2, 1}), Int64View(b, {2, 2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 3, 4, 2, 5, 6}));
}

// Concat with an empty operand: {2,0,4} + {2,3,4} along axis 1 -> {2,3,4}.
// A zero-length slice concatenated with a non-empty input is legal ONNX and
// appears in RF-DETR's transformer; the empty operand contributes nothing.
TEST(NertKernels, ConcatEmptyOperand) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a;
    std::vector<float> b(24, 0.0f);
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(i + 1);
    }
    std::vector<float> out(24, 0.0f);
    nert::Node node = MakeNode("concat_empty", "Concat");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 0, 4}), FloatView(b, {2, 3, 4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, b);
}

// Split divides [2,4] into unequal [2,1] and [2,3] parts along axis 1.
TEST(NertKernels, SplitUnequalSizes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out0(2, 0.0f);
    std::vector<float> out1(6, 0.0f);
    std::vector<int64_t> sizes = {1, 3};
    nert::Node node = MakeNode("split0", "Split");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 4}), Int64View(sizes, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out0, {2, 1}), FloatView(out1, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out0, (std::vector<float>{1.0f, 5.0f}));
    EXPECT_EQ(out1, (std::vector<float>{2.0f, 3.0f, 4.0f, 6.0f, 7.0f, 8.0f}));
}

// A pre-13 Split carries split sizes in an attribute, not an input.
TEST(NertKernels, SplitOldAttributeForm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out0(2, 0.0f);
    std::vector<float> out1(6, 0.0f);
    nert::Node node = MakeNode("split_attr", "Split");
    node.schema_version = 11;
    node.attributes.emplace_back("axis", int64_t{1});
    node.attributes.emplace_back("split", std::vector<int64_t>{1, 3});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 4})};
    std::vector<nert::TensorView> outputs = {FloatView(out0, {2, 1}), FloatView(out1, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out0, (std::vector<float>{1.0f, 5.0f}));
    EXPECT_EQ(out1, (std::vector<float>{2.0f, 3.0f, 4.0f, 6.0f, 7.0f, 8.0f}));
}

// Split with no sizes input divides evenly; negative axis, int64 data.
TEST(NertKernels, SplitEvenNegativeAxisInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4};
    std::vector<int64_t> out0(2, 0);
    std::vector<int64_t> out1(2, 0);
    nert::Node node = MakeNode("split1", "Split");
    node.attributes.emplace_back("axis", int64_t{-2});
    std::vector<nert::TensorView> inputs = {Int64View(in, {4, 1})};
    std::vector<nert::TensorView> outputs = {Int64View(out0, {2, 1}), Int64View(out1, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out0, (std::vector<int64_t>{1, 2}));
    EXPECT_EQ(out1, (std::vector<int64_t>{3, 4}));
}

// Unsqueeze inserts a size-1 axis at position 1 of [2,3].
TEST(NertKernels, UnsqueezeMiddleAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> axes = {1};
    nert::Node node = MakeNode("unsqueeze0", "Unsqueeze");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

// Unsqueeze with axis -1 appends a trailing singleton; int64 data path.
TEST(NertKernels, UnsqueezeNegativeAxisInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {10, 20};
    std::vector<int64_t> out(2, 0);
    std::vector<int64_t> axes = {-1};
    nert::Node node = MakeNode("unsqueeze1", "Unsqueeze");
    std::vector<nert::TensorView> inputs = {Int64View(in, {2}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{10, 20}));
}

// A pre-13 Unsqueeze carries axes in an attribute and has no axes input.
TEST(NertKernels, UnsqueezeOldAttributeForm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("unsqueeze_attr", "Unsqueeze");
    node.schema_version = 11;
    node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

// Expand broadcasts [1,3] to [2,3] by repeating the leading axis.
TEST(NertKernels, ExpandLeadingAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> shape = {2, 3};
    nert::Node node = MakeNode("expand0", "Expand");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 3}), Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}));
}

// Expand to a higher-rank target right-aligns; int64 data path.
TEST(NertKernels, ExpandHigherRankInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {4, 5, 6};
    std::vector<int64_t> out(6, 0);
    std::vector<int64_t> shape = {2, 1, 3};
    nert::Node node = MakeNode("expand1", "Expand");
    std::vector<nert::TensorView> inputs = {Int64View(in, {3}), Int64View(shape, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 1, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{4, 5, 6, 4, 5, 6}));
}

// Expand max-broadcast: [1,300,1] against [1,1,80] resolves to [1,300,80].
TEST(NertKernels, ExpandMaxBroadcastDataAndTarget) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(300, 0.0f);
    for (int64_t i = 0; i < 300; ++i) {
        in[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }
    std::vector<float> out(300 * 80, 0.0f);
    std::vector<int64_t> shape = {1, 1, 80};
    nert::Node node = MakeNode("expand_max", "Expand");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 300, 1}), Int64View(shape, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 300, 80})};

    kernel(node, inputs, outputs);

    ASSERT_EQ(out.size(), static_cast<std::size_t>(300 * 80));
    for (int64_t r = 0; r < 300; ++r) {
        for (int64_t c = 0; c < 80; ++c) {
            EXPECT_EQ(out[static_cast<std::size_t>(r * 80 + c)], static_cast<float>(r + 1))
                << "row " << r << " col " << c;
        }
    }
}

// Transpose with perm [1,0] swaps [2,3] to [3,2].
TEST(NertKernels, TransposeSwapTwoDimensional) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Transpose");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("transpose0", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{1, 0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
}

// Transpose with no perm reverses [2,1,3] to [3,1,2]; int64 data path.
TEST(NertKernels, TransposeDefaultReverseInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Transpose");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("transpose1", "Transpose");
    std::vector<nert::TensorView> inputs = {Int64View(in, {2, 1, 3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3, 1, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 4, 2, 5, 3, 6}));
}

// Slice takes [1,4) with step 1 from a length-6 vector.
TEST(NertKernels, SlicePlainRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    std::vector<float> out(3, 0.0f);
    std::vector<int64_t> starts = {1};
    std::vector<int64_t> ends = {4};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {1};
    nert::Node node = MakeNode("slice0", "Slice");
    std::vector<nert::TensorView> inputs = {FloatView(in, {6}), Int64View(starts, {1}), Int64View(ends, {1}),
                                            Int64View(axes, {1}), Int64View(steps, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{20.0f, 30.0f, 40.0f}));
}

// Slice clamps out-of-range bounds and strides the second axis; int64 data.
TEST(NertKernels, SliceClampAndStepInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int64_t> out(4, 0);
    std::vector<int64_t> starts = {-100, 1};
    std::vector<int64_t> ends = {100, 4};
    std::vector<int64_t> axes = {0, 1};
    std::vector<int64_t> steps = {1, 2};
    nert::Node node = MakeNode("slice1", "Slice");
    std::vector<nert::TensorView> inputs = {Int64View(in, {2, 5}), Int64View(starts, {2}), Int64View(ends, {2}),
                                            Int64View(axes, {2}), Int64View(steps, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 4, 7, 9}));
}

// Slice with negative steps on a [2,5] matrix: axis 0 reverses fully (start
// -1 -> 1, end -INT64_MAX clamps to -1); axis 1 starts at 100 (clamped to 4)
// and runs down by -2 to the exclusive end 0, picking columns 4 and 2.
TEST(NertKernels, SliceNegativeStepsReverse) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> starts = {-1, 100};
    std::vector<int64_t> ends = {-9223372036854775807LL, 0};
    std::vector<int64_t> axes = {0, 1};
    std::vector<int64_t> steps = {-1, -2};
    nert::Node node = MakeNode("slice_neg", "Slice");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 5}), Int64View(starts, {2}), Int64View(ends, {2}),
                                            Int64View(axes, {2}), Int64View(steps, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{10.0f, 8.0f, 5.0f, 3.0f}));
}

// Slice with step -3 on an int64 vector of 7: start 5 down to the exclusive
// end -6 (wrapped to 1) picks indices 5 and 2.
TEST(NertKernels, SliceNegativeStepInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {10, 11, 12, 13, 14, 15, 16};
    std::vector<int64_t> out(2, 0);
    std::vector<int64_t> starts = {5};
    std::vector<int64_t> ends = {-6};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {-3};
    nert::Node node = MakeNode("slice_neg64", "Slice");
    std::vector<nert::TensorView> inputs = {Int64View(in, {7}), Int64View(starts, {1}), Int64View(ends, {1}),
                                            Int64View(axes, {1}), Int64View(steps, {1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{15, 12}));
}

// Gather picks rows 2 and 0 of a [3,2] matrix along axis 0.
TEST(NertKernels, GatherRowsAxisZero) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> indices = {2, 0};
    nert::Node node = MakeNode("gather0", "Gather");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {3, 2}), Int64View(indices, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{5.0f, 6.0f, 1.0f, 2.0f}));
}

// Gather with axis -1 and a wrapped negative index; int64 data path.
TEST(NertKernels, GatherNegativeAxisWrapInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> out(4, 0);
    std::vector<int64_t> indices = {2, -3};
    nert::Node node = MakeNode("gather1", "Gather");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<nert::TensorView> inputs = {Int64View(in, {2, 3}), Int64View(indices, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{3, 1, 6, 4}));
}

// GatherElements selects per-position entries along axis 1 of [2,3].
TEST(NertKernels, GatherElementsAxisOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> indices = {2, 0, 1, 2};
    nert::Node node = MakeNode("gatherelem0", "GatherElements");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(indices, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{3.0f, 1.0f, 5.0f, 6.0f}));
}

// GatherElements with axis -1 and wrapped indices; int64 data path.
TEST(NertKernels, GatherElementsNegativeAxisInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4};
    std::vector<int64_t> out(2, 0);
    std::vector<int64_t> indices = {1, -2};
    nert::Node node = MakeNode("gatherelem1", "GatherElements");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<nert::TensorView> inputs = {Int64View(in, {2, 2}), Int64View(indices, {2, 1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 3}));
}

// GatherElements with UNEQUAL data/indices trailing dims, so the gather-axis
// stride of the data grid (in_strides[axis] = 4) differs from the indices-grid
// stride (strides[axis] = 3). Data [2,3,4] sequential 0..23, indices [2,1,3],
// axis 1. Per ONNX, y[pos] = data[pos with axis-coord <- k], i.e.
// y[r,0,c] = data[r][k=idx[r][c]][c] at offset r*12 + k*4 + c.
// Hand-computed: indices {r0: 0,1,2; r1: 2,1,0} yield offsets
// r0: 0*12 + {0,4,8} + {0,1,2} = {0,5,10};
// r1: 1*12 + {8,4,0} + {0,1,2} = {20,17,14}.
// Pre-fix, the offset used strides[axis]=3 (the indices stride), producing
// {0,3,6,18,15,12}.
TEST(NertKernels, GatherElementsUnequalStride) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(24);
    for (int i = 0; i < 24; ++i) {
        in[i] = static_cast<float>(i);
    }
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> indices = {0, 1, 2, 2, 1, 0};
    nert::Node node = MakeNode("gatherelem2", "GatherElements");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3, 4}), Int64View(indices, {2, 1, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{0.0f, 5.0f, 10.0f, 20.0f, 17.0f, 14.0f}));
}

// Resize doubles a [1,1,2,2] input to [1,1,4,4] with nearest-floor sampling.
TEST(NertKernels, ResizeNearestDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(16, 0.0f);
    std::vector<float> scales = {1.0f, 1.0f, 2.0f, 2.0f};
    nert::Node node = MakeNode("resize0", "Resize");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 1, 2, 2}), nert::TensorView{}, FloatView(scales, {4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 4, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 3.0f,
                                       3.0f, 4.0f, 4.0f}));
}

// Resize triples one axis of a [1,2] row; int64 data path.
TEST(NertKernels, ResizeTripleWidthInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {5, 6};
    std::vector<int64_t> out(6, 0);
    std::vector<float> scales = {1.0f, 3.0f};
    nert::Node node = MakeNode("resize1", "Resize");
    std::vector<nert::TensorView> inputs = {Int64View(in, {1, 2}), nert::TensorView{}, FloatView(scales, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {1, 6})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{5, 5, 5, 6, 6, 6}));
}

// Flatten collapses [2,3,4] to [2,12] with axis 1.
TEST(NertKernels, FlattenAxisOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Flatten");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(24, 0.0f);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<float>(i + 1);
    }
    std::vector<float> out(24, 0.0f);
    nert::Node node = MakeNode("flatten0", "Flatten");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3, 4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 12})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

// Flatten with axis -1 keeps a trailing singleton; int64 data path.
TEST(NertKernels, FlattenNegativeAxisInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Flatten");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("flatten1", "Flatten");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<nert::TensorView> inputs = {Int64View(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {6, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 2, 3, 4, 5, 6}));
}

// Shape emits the [2,3,4] dims as an int64 shape tensor.
TEST(NertKernels, ShapeThreeDimensional) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Shape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(24, 1.0f);
    std::vector<int64_t> out(3, 0);
    nert::Node node = MakeNode("shape0", "Shape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3, 4})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 3, 4}));
}

// Shape reads the rank of a bool input.
TEST(NertKernels, ShapeBoolInput) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Shape");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1, 0};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("shape1", "Shape");
    std::vector<nert::TensorView> inputs = {BoolView(in, {2, 2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2, 2}));
}

// ConstantOfShape fills [2,3] with the int64 value attribute.
TEST(NertKernels, ConstantOfShapeFill) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2, 3};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("cos0", "ConstantOfShape");
    node.attributes.emplace_back("value", int64_t{7});
    std::vector<nert::TensorView> inputs = {Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{7, 7, 7, 7, 7, 7}));
}

// ConstantOfShape with an empty shape input produces a scalar.
TEST(NertKernels, ConstantOfShapeScalar) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape;
    std::vector<int64_t> out(1, 0);
    nert::Node node = MakeNode("cos1", "ConstantOfShape");
    node.attributes.emplace_back("value", int64_t{-3});
    std::vector<nert::TensorView> inputs = {Int64View(shape, {0})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{-3}));
}

// TopK picks the 2 largest of 5 values, descending, with int64 indices.
TEST(NertKernels, TopKTwoLargest) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f};
    std::vector<float> values(2, 0.0f);
    std::vector<int64_t> k = {2};
    std::vector<int64_t> indices(2, 0);
    nert::Node node = MakeNode("topk0", "TopK");
    std::vector<nert::TensorView> inputs = {FloatView(in, {5}), Int64View(k, {})};
    std::vector<nert::TensorView> outputs = {FloatView(values, {2}), Int64View(indices, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(values, (std::vector<float>{5.0f, 4.0f}));
    EXPECT_EQ(indices, (std::vector<int64_t>{1, 3}));
}

// TopK with K < dim selects per row of [2,4] along the default last axis.
TEST(NertKernels, TopKRowsSmallerK) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 4.0f, 2.0f, 3.0f, 8.0f, 5.0f, 7.0f, 6.0f};
    std::vector<float> values(6, 0.0f);
    std::vector<int64_t> k = {3};
    std::vector<int64_t> indices(6, 0);
    nert::Node node = MakeNode("topk1", "TopK");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 4}), Int64View(k, {})};
    std::vector<nert::TensorView> outputs = {FloatView(values, {2, 3}), Int64View(indices, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(values, (std::vector<float>{4.0f, 3.0f, 2.0f, 8.0f, 7.0f, 6.0f}));
    EXPECT_EQ(indices, (std::vector<int64_t>{1, 3, 2, 0, 2, 3}));
}

// Concat rejects inputs that differ off the concat axis, naming the op.
TEST(NertKernelsNegative, ConcatRejectsOffAxisMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {3.0f, 4.0f, 5.0f};
    std::vector<float> out(5, 0.0f);
    nert::Node node = MakeNode("concat_bad", "Concat");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(a, {1, 2}), FloatView(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 5})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Concat"), std::string::npos) << error.what();
    }
}

// Concat rejects an out-of-range axis, naming the op.
TEST(NertKernelsNegative, ConcatRejectsAxisOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("concat_axis", "Concat");
    node.attributes.emplace_back("axis", int64_t{2});
    std::vector<nert::TensorView> inputs = {FloatView(a, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Concat"), std::string::npos) << error.what();
    }
}

// Split rejects sizes that do not sum to the axis dim, naming the op.
TEST(NertKernelsNegative, SplitRejectsSizeSumMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out0(1, 0.0f);
    std::vector<float> out1(2, 0.0f);
    std::vector<int64_t> sizes = {1, 2};
    nert::Node node = MakeNode("split_bad", "Split");
    std::vector<nert::TensorView> inputs = {FloatView(in, {4}), Int64View(sizes, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out0, {1}), FloatView(out1, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Split"), std::string::npos) << error.what();
    }
}

// Split rejects an axis dim that does not divide evenly, naming the op.
TEST(NertKernelsNegative, SplitRejectsUnevenDefault) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Split");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out0(1, 0.0f);
    std::vector<float> out1(1, 0.0f);
    nert::Node node = MakeNode("split_even", "Split");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out0, {1}), FloatView(out1, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Split"), std::string::npos) << error.what();
    }
}

// Unsqueeze rejects duplicated axes, naming the op.
TEST(NertKernelsNegative, UnsqueezeRejectsDuplicateAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1, 1};
    nert::Node node = MakeNode("unsqueeze_bad", "Unsqueeze");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2}), Int64View(axes, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Unsqueeze"), std::string::npos) << error.what();
    }
}

// Unsqueeze rejects an out-of-range axis, naming the op.
TEST(NertKernelsNegative, UnsqueezeRejectsAxisOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {5};
    nert::Node node = MakeNode("unsqueeze_axis", "Unsqueeze");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Unsqueeze"), std::string::npos) << error.what();
    }
}

// Expand rejects a target the input cannot broadcast to, naming the op.
TEST(NertKernelsNegative, ExpandRejectsUnbroadcastable) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> shape = {3, 2};
    nert::Node node = MakeNode("expand_bad", "Expand");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Expand"), std::string::npos) << error.what();
    }
}

// Expand rejects genuinely incompatible dims ([2,3] vs [2,4]), naming the op.
TEST(NertKernelsNegative, ExpandRejectsIncompatibleDims) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Expand");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(8, 0.0f);
    std::vector<int64_t> shape = {2, 4};
    nert::Node node = MakeNode("expand_incompat", "Expand");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Expand"), std::string::npos) << error.what();
    }
}

// Transpose rejects a perm that is not a permutation, naming the op.
TEST(NertKernelsNegative, TransposeRejectsBadPerm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Transpose");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("transpose_bad", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{0, 0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Transpose"), std::string::npos) << error.what();
    }
}

// Slice rejects starts/ends of different lengths, naming the op.
TEST(NertKernelsNegative, SliceRejectsStartsEndsMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> starts = {0, 0};
    std::vector<int64_t> ends = {2};
    nert::Node node = MakeNode("slice_bad", "Slice");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3}), Int64View(starts, {2}), Int64View(ends, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Slice"), std::string::npos) << error.what();
    }
}

// Slice rejects a zero step (negative steps are valid), naming the op.
TEST(NertKernelsNegative, SliceRejectsNonPositiveStep) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(3, 0.0f);
    std::vector<int64_t> starts = {0};
    std::vector<int64_t> ends = {3};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {0};
    nert::Node node = MakeNode("slice_step", "Slice");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3}), Int64View(starts, {1}), Int64View(ends, {1}),
                                            Int64View(axes, {1}), Int64View(steps, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Slice"), std::string::npos) << error.what();
    }
}

// Gather rejects an out-of-range index, naming the op.
TEST(NertKernelsNegative, GatherRejectsIndexOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(1, 0.0f);
    std::vector<int64_t> indices = {5};
    nert::Node node = MakeNode("gather_bad", "Gather");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3}), Int64View(indices, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Gather"), std::string::npos) << error.what();
    }
}

// Gather rejects an out-of-range axis, naming the op.
TEST(NertKernelsNegative, GatherRejectsAxisOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<int64_t> indices = {0, 1};
    nert::Node node = MakeNode("gather_axis", "Gather");
    node.attributes.emplace_back("axis", int64_t{2});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(indices, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Gather"), std::string::npos) << error.what();
    }
}

// GatherElements rejects data/indices of different rank, naming the op.
TEST(NertKernelsNegative, GatherElementsRejectsRankMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> indices = {0, 1};
    nert::Node node = MakeNode("gatherelem_bad", "GatherElements");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(indices, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("GatherElements"), std::string::npos) << error.what();
    }
}

// GatherElements rejects an out-of-range index, naming the op.
TEST(NertKernelsNegative, GatherElementsRejectsIndexOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GatherElements");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> indices = {0, 7};
    nert::Node node = MakeNode("gatherelem_idx", "GatherElements");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2}), Int64View(indices, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("GatherElements"), std::string::npos) << error.what();
    }
}

// Resize rejects a node that supplies both scales and sizes, naming the op.
TEST(NertKernelsNegative, ResizeRejectsSizesInput) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<float> scales = {1.0f, 2.0f};
    std::vector<int64_t> sizes = {1, 4};
    nert::Node node = MakeNode("resize_bad", "Resize");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 2}), nert::TensorView{}, FloatView(scales, {2}),
                                            Int64View(sizes, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Resize"), std::string::npos) << error.what();
    }
}

// Resize rejects scales whose rank misses the input rank, naming the op.
TEST(NertKernelsNegative, ResizeRejectsScalesRankMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(4, 0.0f);
    std::vector<float> scales = {2.0f, 2.0f, 2.0f};
    nert::Node node = MakeNode("resize_scales", "Resize");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 2}), nert::TensorView{}, FloatView(scales, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Resize"), std::string::npos) << error.what();
    }
}

// Flatten rejects an out-of-range axis, naming the op.
TEST(NertKernelsNegative, FlattenRejectsAxisOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Flatten");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("flatten_bad", "Flatten");
    node.attributes.emplace_back("axis", int64_t{3});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Flatten"), std::string::npos) << error.what();
    }
}

// Shape rejects a non-int64 output, naming the op.
TEST(NertKernelsNegative, ShapeRejectsNonInt64Output) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Shape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("shape_bad", "Shape");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Shape"), std::string::npos) << error.what();
    }
}

// ConstantOfShape rejects a negative shape value, naming the op.
TEST(NertKernelsNegative, ConstantOfShapeRejectsNegativeDim) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2, -1};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("cos_bad", "ConstantOfShape");
    node.attributes.emplace_back("value", int64_t{0});
    std::vector<nert::TensorView> inputs = {Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ConstantOfShape"), std::string::npos) << error.what();
    }
}

// ConstantOfShape without a value attribute fills float32 zero (opset 18), so an
// int64 output no longer matches and rejects, naming the op. Converted from the
// former ConstantOfShapeRejectsMissingValue: a missing value is now accepted
// (see ConstantOfShapeDefaultFillsFloatZero); the dtype mismatch still rejects.
TEST(NertKernelsNegative, ConstantOfShapeMissingValueRejectsInt64Output) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("cos_novalue", "ConstantOfShape");
    std::vector<nert::TensorView> inputs = {Int64View(shape, {1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ConstantOfShape"), std::string::npos) << error.what();
    }
}

// TopK rejects K larger than the axis dim, naming the op.
TEST(NertKernelsNegative, TopKRejectsKExceedsDim) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> values(3, 0.0f);
    std::vector<int64_t> k = {4};
    std::vector<int64_t> indices(3, 0);
    nert::Node node = MakeNode("topk_bad", "TopK");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3}), Int64View(k, {})};
    std::vector<nert::TensorView> outputs = {FloatView(values, {3}), Int64View(indices, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("TopK"), std::string::npos) << error.what();
    }
}

// TopK rejects largest=0, naming the op.
TEST(NertKernelsNegative, TopKRejectsLargestZero) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> values(2, 0.0f);
    std::vector<int64_t> k = {2};
    std::vector<int64_t> indices(2, 0);
    nert::Node node = MakeNode("topk_largest", "TopK");
    node.attributes.emplace_back("largest", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {3}), Int64View(k, {})};
    std::vector<nert::TensorView> outputs = {FloatView(values, {2}), Int64View(indices, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("TopK"), std::string::npos) << error.what();
    }
}

// TopK rejects a non-scalar K input, naming the op.
TEST(NertKernelsNegative, TopKRejectsNonScalarK) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("TopK");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<float> values(2, 0.0f);
    std::vector<int64_t> k = {2, 2};
    std::vector<int64_t> indices(2, 0);
    nert::Node node = MakeNode("topk_kscalar", "TopK");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3}), Int64View(k, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(values, {2}), Int64View(indices, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("TopK"), std::string::npos) << error.what();
    }
}

// Add broadcasts int64 [2,1] against [1,3] into [2,3].
TEST(NertKernels, AddInt64BroadcastsBothAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Add");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {10, 20, 30};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("add_i64", "Add");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2, 1}), Int64View(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{11, 21, 31, 12, 22, 32}));
}

// Sub subtracts int64 elementwise without broadcasting.
TEST(NertKernels, SubInt64PlainDifference) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sub");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {5, 7, 9};
    std::vector<int64_t> b = {1, 2, 3};
    std::vector<int64_t> out(3, 0);
    nert::Node node = MakeNode("sub_i64", "Sub");
    std::vector<nert::TensorView> inputs = {Int64View(a, {3}), Int64View(b, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{4, 5, 6}));
}

// Mul broadcasts an int64 trailing [3] input across a [2,3] input.
TEST(NertKernels, MulInt64BroadcastsLowerRank) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Mul");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> b = {10, 20, 30};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("mul_i64", "Mul");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2, 3}), Int64View(b, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{10, 40, 90, 40, 100, 180}));
}

// Div on int64 truncates toward zero, matching the static evaluator.
TEST(NertKernels, DivInt64TruncatesTowardZero) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, -7, 7, -7};
    std::vector<int64_t> b = {2, 2, -2, -2};
    std::vector<int64_t> out(4, 0);
    nert::Node node = MakeNode("div_i64", "Div");
    std::vector<nert::TensorView> inputs = {Int64View(a, {4}), Int64View(b, {4})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{3, -3, -3, 3}));
}

// Where selects elementwise between two int64 vectors.
TEST(NertKernels, WhereInt64FlatSelect) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> cond = {1, 0, 1};
    std::vector<int64_t> x = {1, 2, 3};
    std::vector<int64_t> y = {10, 20, 30};
    std::vector<int64_t> out(3, 0);
    nert::Node node = MakeNode("where_i64", "Where");
    std::vector<nert::TensorView> inputs = {BoolView(cond, {3}), Int64View(x, {3}), Int64View(y, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 20, 3}));
}

// Where broadcasts int64 X/Y with the bool condition into [2,3].
TEST(NertKernels, WhereInt64BroadcastsAllThree) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> cond = {1, 0};
    std::vector<int64_t> x = {1, 2, 3};
    std::vector<int64_t> y = {9};
    std::vector<int64_t> out(6, 0);
    nert::Node node = MakeNode("where_i64b", "Where");
    std::vector<nert::TensorView> inputs = {BoolView(cond, {2, 1}), Int64View(x, {3}), Int64View(y, {})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 2, 3, 9, 9, 9}));
}

// Div rejects an int64 zero divisor, naming the op.
TEST(NertKernelsNegative, DivRejectsInt64ZeroDivisor) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Div");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {2, 0};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("div_zero_i64", "Div");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Div"), std::string::npos) << error.what();
    }
}

// Where rejects mixed float/int value inputs, naming the op.
TEST(NertKernelsNegative, WhereRejectsMixedValueDtypes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Where");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> cond = {1, 0};
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<int64_t> y = {10, 20};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("where_mixed", "Where");
    std::vector<nert::TensorView> inputs = {BoolView(cond, {2}), FloatView(x, {2}), Int64View(y, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Where"), std::string::npos) << error.what();
    }
}

// Floor rounds toward negative infinity on both signs.
TEST(NertKernels, FloorRoundsDown) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Floor");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {-1.5f, 0.5f, 2.0f, 2.7f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("floor0", "Floor");
    std::vector<nert::TensorView> inputs = {FloatView(in, {4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{-2.0f, 0.0f, 2.0f, 2.0f}));
}

// Identity copies an int64 tensor unchanged.
TEST(NertKernels, IdentityCopiesInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Identity");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {7, -3, 42};
    std::vector<int64_t> out(3, 0);
    nert::Node node = MakeNode("identity0", "Identity");
    std::vector<nert::TensorView> inputs = {Int64View(in, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

// Pow squares with a scalar exponent and broadcasts a per-column exponent.
TEST(NertKernels, PowScalarAndBroadcastExponent) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Pow");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> two = {2.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("pow0", "Pow");
    std::vector<nert::TensorView> inputs = {FloatView(base, {2, 2}), FloatView(two, {})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{1.0f, 4.0f, 9.0f, 16.0f}));

    std::vector<float> base2 = {4.0f, 2.0f, 9.0f, 3.0f};
    std::vector<float> exps = {0.5f, 3.0f};
    inputs = {FloatView(base2, {2, 2}), FloatView(exps, {2})};
    kernel(node, inputs, outputs);
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[1], 8.0f);
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    EXPECT_FLOAT_EQ(out[3], 27.0f);
}

// Tile repeats a [1,2] int64 tensor twice along each axis.
TEST(NertKernels, TileRepeatsEachAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Tile");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2};
    std::vector<int64_t> repeats = {2, 2};
    std::vector<int64_t> out(8, 0);
    nert::Node node = MakeNode("tile0", "Tile");
    std::vector<nert::TensorView> inputs = {Int64View(in, {1, 2}), Int64View(repeats, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 2, 1, 2, 1, 2, 1, 2}));
}

// AveragePool with a 2x2 window and stride 1 averages each overlapping block.
TEST(NertKernels, AveragePoolTwoByTwoStrideOne) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("AveragePool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("avg_2x2", "AveragePool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 3, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{3.0f, 4.0f, 6.0f, 7.0f}));
}

// AveragePool over a padded border divides by the in-bounds count unless
// count_include_pad is set, in which case padded positions count too.
TEST(NertKernels, AveragePoolCountIncludePad) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("AveragePool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("avg_pad", "AveragePool");
    node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));

    node.attributes.emplace_back("count_include_pad", int64_t{1});
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{0.25f, 0.5f, 0.75f, 1.0f}));
}

// GlobalAveragePool over {1,2,2,2} 1..8 averages each channel plane:
// {(1+2+3+4)/4, (5+6+7+8)/4} = {2.5, 6.5}.
TEST(NertKernels, GlobalAveragePoolChannelMeans) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GlobalAveragePool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("gap0", "GlobalAveragePool");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 2, 1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.5f, 6.5f}));
}

// GlobalAveragePool rejects a rank-1 input, naming node + op.
TEST(NertKernels, GlobalAveragePoolRankOneRejected) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GlobalAveragePool");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("gap_bad", "GlobalAveragePool");
    std::vector<nert::TensorView> inputs = {FloatView(x, {4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1})};
    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException for a rank-1 GlobalAveragePool input";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("gap_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("GlobalAveragePool"), std::string::npos) << message;
    }
}

// ConvTranspose with a 2x2 all-ones kernel and stride 2 expands each input
// pixel into a 2x2 block, plus the bias.
TEST(NertKernels, ConvTransposeStrideTwoUpsamples) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConvTranspose");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> b = {0.5f};
    std::vector<float> out(16, 0.0f);
    nert::Node node = MakeNode("deconv_s2", "ConvTranspose");
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 2, 2}), FloatView(w, {1, 1, 2, 2}), FloatView(b, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 4, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.5f, 1.5f, 2.5f, 2.5f, 1.5f, 1.5f, 2.5f, 2.5f, 3.5f, 3.5f, 4.5f, 4.5f, 3.5f,
                                       3.5f, 4.5f, 4.5f}));
}

// ConvTranspose with stride 1 sums overlapping contributions; pads crop the
// border back to the centre.
TEST(NertKernels, ConvTransposeOverlapAndPads) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConvTranspose");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(9, 0.0f);
    nert::Node node = MakeNode("deconv_s1", "ConvTranspose");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 2, 2}), FloatView(w, {1, 1, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 3, 3})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f, 10.0f, 6.0f, 3.0f, 7.0f, 4.0f}));

    std::vector<float> centre(1, 0.0f);
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    outputs = {FloatView(centre, {1, 1, 1, 1})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(centre, (std::vector<float>{10.0f}));
}

// Resize driven by a sizes input doubles the width with nearest/floor.
TEST(NertKernels, ResizeFromSizesInput) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f};
    std::vector<int64_t> sizes = {1, 1, 1, 4};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("resize_sizes", "Resize");
    node.attributes.emplace_back("mode", std::string("nearest"));
    node.attributes.emplace_back("coordinate_transformation_mode", std::string("asymmetric"));
    node.attributes.emplace_back("nearest_mode", std::string("floor"));
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 1, 2}), nert::TensorView{}, nert::TensorView{},
                                            Int64View(sizes, {4})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 1, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}));
}

// ConvTranspose rejects an output view whose shape disagrees with the window.
TEST(NertKernelsNegative, ConvTransposeRejectsWrongOutputShape) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConvTranspose");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("deconv_bad", "ConvTranspose");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 2, 2}), FloatView(w, {1, 1, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ConvTranspose"), std::string::npos) << error.what();
    }
}

// Erf, Sqrt, and Tanh match the C library on a few points.
TEST(NertKernels, ErfSqrtTanhUnary) {
    std::vector<float> in = {0.0f, 0.5f, 1.0f, 4.0f};
    for (const char* op : {"Erf", "Sqrt", "Tanh"}) {
        nert::KernelFn kernel = nert::CpuDevice().kernels().find(op);
        ASSERT_NE(kernel, nullptr) << op;
        std::vector<float> out(4, -1.0f);
        nert::Node node = MakeNode("unary", op);
        std::vector<nert::TensorView> inputs = {FloatView(in, {4})};
        std::vector<nert::TensorView> outputs = {FloatView(out, {4})};
        kernel(node, inputs, outputs);
        for (std::size_t i = 0; i < in.size(); ++i) {
            const float want = std::string(op) == "Erf"    ? std::erf(in[i])
                               : std::string(op) == "Sqrt" ? std::sqrt(in[i])
                                                           : std::tanh(in[i]);
            EXPECT_FLOAT_EQ(out[i], want) << op << " at " << i;
        }
    }
}

// IsNaN flags only the NaN element.
TEST(NertKernels, IsNaNFlagsNaN) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("IsNaN");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, std::nanf(""), -2.0f};
    std::vector<std::uint8_t> out(3, 7);
    nert::Node node = MakeNode("isnan0", "IsNaN");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 1, 0}));
}

// Squeeze keeps the element order; only the view's dims change.
TEST(NertKernels, SqueezeCopiesInOrder) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Squeeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f};
    std::vector<int64_t> axes = {0};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("squeeze0", "Squeeze");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

// LayerNormalization over the last axis: rows [1,2,3] and [2,2,2] normalize to
// [-sqrt(1.5), 0, sqrt(1.5)] and [0,0,0] (epsilon 0), then scale and shift;
// the optional Mean and InvStdDev outputs carry the row statistics.
TEST(NertKernels, LayerNormalizationLastAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("LayerNormalization");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> scale = {2.0f, 2.0f, 2.0f};
    std::vector<float> bias = {1.0f, 1.0f, 1.0f};
    std::vector<float> y(6, 0.0f);
    std::vector<float> mean(2, 0.0f);
    std::vector<float> inv(2, 0.0f);
    nert::Node node = MakeNode("ln0", "LayerNormalization");
    node.attributes.emplace_back("epsilon", 1e-12f);
    std::vector<nert::TensorView> inputs = {FloatView(x, {2, 3}), FloatView(scale, {3}), FloatView(bias, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(y, {2, 3}), FloatView(mean, {2, 1}), FloatView(inv, {2, 1})};

    kernel(node, inputs, outputs);

    const float r = std::sqrt(1.5f);
    EXPECT_NEAR(y[0], 1.0f - 2.0f * r, 1e-5f);
    EXPECT_NEAR(y[1], 1.0f, 1e-5f);
    EXPECT_NEAR(y[2], 1.0f + 2.0f * r, 1e-5f);
    EXPECT_NEAR(mean[0], 2.0f, 1e-6f);
    EXPECT_NEAR(inv[0], 1.0f / std::sqrt(2.0f / 3.0f), 1e-4f);
    EXPECT_NEAR(mean[1], 2.0f, 1e-6f);
}

// InstanceNormalization on [2,2,2,2]: each (n, c) plane of four values is
// normalized on its own (population variance, epsilon 1e-12 so the constant
// plane stays finite), then scaled by
// scale[c] and shifted by B[c]. Plane [1,2,3,4] has mean 2.5 and variance
// 1.25, so it maps to [-3,-1,1,3]/sqrt(5); a constant plane maps to 0, giving
// B[c]; plane [0,0,0,8] has mean 2 and variance 12, so it maps to
// [-1,-1,-1,3]/sqrt(3).
TEST(NertKernels, InstanceNormalizationPerChannelPlanes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("InstanceNormalization");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 5.0f, 5.0f, 5.0f,
                            0.0f, 0.0f, 0.0f, 8.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> scale = {2.0f, -1.0f};
    std::vector<float> bias = {0.5f, 3.0f};
    std::vector<float> y(16, 0.0f);
    nert::Node node = MakeNode("in0", "InstanceNormalization");
    node.attributes.emplace_back("epsilon", 1e-12f);
    std::vector<nert::TensorView> inputs = {FloatView(x, {2, 2, 2, 2}), FloatView(scale, {2}), FloatView(bias, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(y, {2, 2, 2, 2})};

    kernel(node, inputs, outputs);

    const float s5 = std::sqrt(5.0f);
    const float s3 = std::sqrt(3.0f);
    const std::vector<float> expected = {0.5f - 6.0f / s5,
                                         0.5f - 2.0f / s5,
                                         0.5f + 2.0f / s5,
                                         0.5f + 6.0f / s5,
                                         3.0f,
                                         3.0f,
                                         3.0f,
                                         3.0f,
                                         0.5f - 2.0f / s3,
                                         0.5f - 2.0f / s3,
                                         0.5f - 2.0f / s3,
                                         0.5f + 6.0f / s3,
                                         3.0f + 3.0f / s5,
                                         3.0f + 1.0f / s5,
                                         3.0f - 1.0f / s5,
                                         3.0f - 3.0f / s5};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 1e-5f) << "index " << i;
    }
}

// InstanceNormalization defaults epsilon to 1e-5: plane [0, 2] has variance 1,
// so it maps to [-1, 1] / sqrt(1 + 1e-5).
TEST(NertKernels, InstanceNormalizationDefaultEpsilon) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("InstanceNormalization");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {0.0f, 2.0f};
    std::vector<float> scale = {1.0f};
    std::vector<float> bias = {0.0f};
    std::vector<float> y(2, 0.0f);
    nert::Node node = MakeNode("in_eps", "InstanceNormalization");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 2}), FloatView(scale, {1}), FloatView(bias, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(y, {1, 1, 2})};

    kernel(node, inputs, outputs);

    const double expected = 1.0 / std::sqrt(1.0 + 1e-5);
    EXPECT_NEAR(y[0], -expected, 1e-7);
    EXPECT_NEAR(y[1], expected, 1e-7);
    EXPECT_NE(y[1], 1.0f);
}

// Shape with start/end emits only the selected dims; negatives count back.
TEST(NertKernels, ShapeStartEndSelectsDims) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Shape");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(2 * 3 * 4, 0.0f);
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("shape_slice", "Shape");
    node.attributes.emplace_back("start", int64_t{1});
    node.attributes.emplace_back("end", int64_t{3});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3, 4})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<int64_t>{3, 4}));

    std::vector<int64_t> first(1, 0);
    nert::Node head = MakeNode("shape_head", "Shape");
    head.attributes.emplace_back("end", int64_t{-2});
    outputs = {Int64View(first, {1})};
    kernel(head, inputs, outputs);
    EXPECT_EQ(first, (std::vector<int64_t>{2}));
}

// Linear Resize with align_corners doubles [0, 3] to [0, 1, 2, 3]; with
// half_pixel the samples at -0.25 and 1.25 clamp to the border.
TEST(NertKernels, ResizeLinearTransforms) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {0.0f, 3.0f};
    std::vector<int64_t> sizes = {1, 4};
    std::vector<float> out(4, -1.0f);
    nert::Node node = MakeNode("resize_linear", "Resize");
    node.attributes.emplace_back("mode", std::string("linear"));
    node.attributes.emplace_back("coordinate_transformation_mode", std::string("align_corners"));
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2}), nert::TensorView{}, nert::TensorView{},
                                            Int64View(sizes, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 4})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f}));

    nert::Node half = MakeNode("resize_half", "Resize");
    half.attributes.emplace_back("mode", std::string("linear"));
    kernel(half, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{0.0f, 0.75f, 2.25f, 3.0f}));
}

// Nearest Resize defaults to half_pixel + round_prefer_floor: upsampling
// [1, 2] by 2 samples 0, 0, 1, 1.
TEST(NertKernels, ResizeNearestOnnxDefaults) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Resize");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> scales = {1.0f, 2.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("resize_default", "Resize");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2}), nert::TensorView{}, FloatView(scales, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}));
}

// A 3-D Conv with a 2x1x1 all-ones kernel and depth stride 2 sums each pair
// of frames, the tubelet pattern of video transformers.
TEST(NertKernels, Conv3dTubelet) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f, 1000.0f, 2000.0f};
    std::vector<float> w = {1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("conv3d", "Conv");
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 1, 1});
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 1, 4, 1, 2}), FloatView(w, {1, 1, 2, 1, 1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 2, 1, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{11.0f, 22.0f, 1100.0f, 2200.0f}));
}

// LayerNormalization rejects a Scale that is neither per-element nor scalar.
TEST(NertKernelsNegative, LayerNormalizationRejectsScaleShape) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("LayerNormalization");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> scale = {1.0f, 1.0f};
    std::vector<float> y(3, 0.0f);
    nert::Node node = MakeNode("ln_bad", "LayerNormalization");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 3}), FloatView(scale, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(y, {1, 3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("LayerNormalization"), std::string::npos) << error.what();
    }
}

// InstanceNormalization rejects a scale whose length is not C.
TEST(NertKernelsNegative, InstanceNormalizationRejectsScaleLength) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("InstanceNormalization");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> scale = {1.0f, 1.0f, 1.0f};
    std::vector<float> bias = {0.0f, 0.0f};
    std::vector<float> y(4, 0.0f);
    nert::Node node = MakeNode("in_bad", "InstanceNormalization");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2, 2}), FloatView(scale, {3}), FloatView(bias, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(y, {1, 2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("InstanceNormalization"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("in_bad"), std::string::npos) << error.what();
    }
}

// Abs on float32 flips negatives and keeps positives.
TEST(NertKernels, AbsFloat) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Abs");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {-1.5f, 0.0f, 2.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("abs0", "Abs");
    std::vector<nert::TensorView> inputs = {FloatView(x, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.5f, 0.0f, 2.0f}));
}

// Abs on int64 keeps the dtype.
TEST(NertKernels, AbsInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Abs");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> x = {-7, 3, 0};
    std::vector<int64_t> out(3, 0);
    nert::Node node = MakeNode("abs_i", "Abs");
    std::vector<nert::TensorView> inputs = {Int64View(x, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{7, 3, 0}));
}

// Neg on float32 flips signs.
TEST(NertKernels, NegFloat) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Neg");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {-1.5f, 2.0f, 4.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("neg0", "Neg");
    std::vector<nert::TensorView> inputs = {FloatView(x, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.5f, -2.0f, -4.0f}));
}

// Neg on int64 keeps the dtype.
TEST(NertKernels, NegInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Neg");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> x = {-7, 3, 0};
    std::vector<int64_t> out(3, 0);
    nert::Node node = MakeNode("neg_i", "Neg");
    std::vector<nert::TensorView> inputs = {Int64View(x, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{7, -3, 0}));
}

// Exp of 0 is 1 and of 1 is e.
TEST(NertKernels, ExpValues) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Exp");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {0.0f, 1.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("exp0", "Exp");
    std::vector<nert::TensorView> inputs = {FloatView(x, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_NEAR(out[1], 2.718281828f, 1e-6f);
}

// Log of 1 is 0 and of e^2 is 2.
TEST(NertKernels, LogValues) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Log");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0f, 7.389056099f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("log0", "Log");
    std::vector<nert::TensorView> inputs = {FloatView(x, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_NEAR(out[1], 2.0f, 1e-6f);
}

// Cos of 0 is 1 and of pi is -1.
TEST(NertKernels, CosValues) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Cos");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {0.0f, 3.14159265f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("cos0", "Cos");
    std::vector<nert::TensorView> inputs = {FloatView(x, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_NEAR(out[1], -1.0f, 1e-6f);
}

// Sin of 0 is 0 and of pi/2 is 1.
TEST(NertKernels, SinValues) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Sin");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {0.0f, 1.57079633f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("sin0", "Sin");
    std::vector<nert::TensorView> inputs = {FloatView(x, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[0], 0.0f);
    EXPECT_NEAR(out[1], 1.0f, 1e-6f);
}

// Clip with both bounds clamps to [0, 6].
TEST(NertKernels, ClipBothBounds) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {-2.0f, 3.0f, 9.0f};
    std::vector<float> lo = {0.0f};
    std::vector<float> hi = {6.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("clip_both", "Clip");
    std::vector<nert::TensorView> inputs = {FloatView(x, {3}), FloatView(lo, {}), FloatView(hi, {})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{0.0f, 3.0f, 6.0f}));
}

// Clip with only min leaves the upper side unbounded.
TEST(NertKernels, ClipOnlyMin) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {-2.0f, 3.0f, 9.0f};
    std::vector<float> lo = {0.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("clip_min", "Clip");
    std::vector<nert::TensorView> inputs = {FloatView(x, {3}), FloatView(lo, {}), nert::TensorView{}};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{0.0f, 3.0f, 9.0f}));
}

// Clip with only max (min omitted) leaves the lower side unbounded.
TEST(NertKernels, ClipOnlyMax) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {-2.0f, 3.0f, 9.0f};
    std::vector<float> hi = {6.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("clip_max", "Clip");
    std::vector<nert::TensorView> inputs = {FloatView(x, {3}), nert::TensorView{}, FloatView(hi, {})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{-2.0f, 3.0f, 6.0f}));
}

// Clip on int64 data clamps to [min, max] with exact integer semantics.
TEST(NertKernels, ClipInt64Bounds) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> x = {-5, 0, 7, 20};
    std::vector<int64_t> lo = {-1};
    std::vector<int64_t> hi = {10};
    std::vector<int64_t> out(4, 0);
    nert::Node node = MakeNode("clip_i64", "Clip");
    std::vector<nert::TensorView> inputs = {Int64View(x, {4}), Int64View(lo, {}), Int64View(hi, {})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{-1, 0, 7, 10}));
}

// Clip on int64 with only a min bound leaves the upper side unbounded.
TEST(NertKernels, ClipInt64OnlyMin) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> x = {-5, 0, 7, 20};
    std::vector<int64_t> lo = {-1};
    std::vector<int64_t> out(4, 0);
    nert::Node node = MakeNode("clip_i64_min", "Clip");
    std::vector<nert::TensorView> inputs = {Int64View(x, {4}), Int64View(lo, {}), nert::TensorView{}};
    std::vector<nert::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{-1, 0, 7, 20}));
}

// Clip on int64 with only a max bound (min omitted) leaves the lower side
// unbounded.
TEST(NertKernels, ClipInt64OnlyMax) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> x = {-5, 0, 7, 20};
    std::vector<int64_t> hi = {10};
    std::vector<int64_t> out(4, 0);
    nert::Node node = MakeNode("clip_i64_max", "Clip");
    std::vector<nert::TensorView> inputs = {Int64View(x, {4}), nert::TensorView{}, Int64View(hi, {})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {4})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{-5, 0, 7, 10}));
}

// Clip on int64 with no bounds is the identity, int64 extremes included: the
// absent-bound fallbacks (INT64_MIN/INT64_MAX) never disturb a value.
TEST(NertKernels, ClipInt64Identity) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> x = {-5, 0, 7, 20, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    std::vector<int64_t> out(x.size(), 0);
    nert::Node node = MakeNode("clip_i64_id", "Clip");
    std::vector<nert::TensorView> inputs = {Int64View(x, {static_cast<int64_t>(x.size())})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {static_cast<int64_t>(out.size())})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, x);
}

// Clip rejects a float32/int64 dtype mix (int64 min on float32 data), naming
// node + op.
TEST(NertKernelsNegative, ClipRejectsMixedDtypes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Clip");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {-5.0f, 0.0f, 7.0f, 20.0f};
    std::vector<int64_t> lo = {-1};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("clip_mixed", "Clip");
    std::vector<nert::TensorView> inputs = {FloatView(x, {4}), Int64View(lo, {})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {4})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("Clip"), std::string::npos) << message;
        EXPECT_NE(message.find("clip_mixed"), std::string::npos) << message;
    }
}

// Exp rejects int64 input and names the op.
TEST(NertKernelsNegative, ExpRejectsInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Exp");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> x = {1, 2};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("exp_bad", "Exp");
    std::vector<nert::TensorView> inputs = {Int64View(x, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Exp"), std::string::npos) << error.what();
    }
}

// Greater broadcasts float32 [2,1] against [3] into a [2,3] bool output.
TEST(NertKernels, GreaterFloatBroadcast) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Greater");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {2.0f, 5.0f};
    std::vector<float> b = {1.0f, 3.0f, 5.0f};
    std::vector<std::uint8_t> out(6, 0);
    nert::Node node = MakeNode("greater0", "Greater");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 0, 0, 1, 1, 0}));
}

// Greater compares int64 vectors elementwise.
TEST(NertKernels, GreaterInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Greater");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {3, -1, 7};
    std::vector<int64_t> b = {2, -1, 9};
    std::vector<std::uint8_t> out(3, 0);
    nert::Node node = MakeNode("greater1", "Greater");
    std::vector<nert::TensorView> inputs = {Int64View(a, {3}), Int64View(b, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 0, 0}));
}

// Less broadcasts float32 [2,1] against [3] into a [2,3] bool output.
TEST(NertKernels, LessFloatBroadcast) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Less");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {2.0f, 5.0f};
    std::vector<float> b = {1.0f, 3.0f, 5.0f};
    std::vector<std::uint8_t> out(6, 0);
    nert::Node node = MakeNode("less0", "Less");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 1}), FloatView(b, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 1, 1, 0, 0, 0}));
}

// Less compares int64 vectors elementwise.
TEST(NertKernels, LessInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Less");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {3, -1, 7};
    std::vector<int64_t> b = {2, -1, 9};
    std::vector<std::uint8_t> out(3, 0);
    nert::Node node = MakeNode("less1", "Less");
    std::vector<nert::TensorView> inputs = {Int64View(a, {3}), Int64View(b, {3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 0, 1}));
}

// And broadcasts bool [2,1] against [2] into a [2,2] bool output.
TEST(NertKernels, AndBroadcast) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("And");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> a = {1, 0};
    std::vector<std::uint8_t> b = {1, 0};
    std::vector<std::uint8_t> out(4, 0);
    nert::Node node = MakeNode("and0", "And");
    std::vector<nert::TensorView> inputs = {BoolView(a, {2, 1}), BoolView(b, {2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 0, 0, 0}));
}

// Not negates a bool vector.
TEST(NertKernels, NotFlipsBools) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Not");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> a = {1, 0, 0, 1};
    std::vector<std::uint8_t> out(4, 0);
    nert::Node node = MakeNode("not0", "Not");
    std::vector<nert::TensorView> inputs = {BoolView(a, {2, 2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{0, 1, 1, 0}));
}

// Greater rejects mixed float/int inputs, naming the op.
TEST(NertKernelsNegative, GreaterRejectsMixedDtypes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Greater");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<int64_t> b = {1, 2};
    std::vector<std::uint8_t> out(2, 0);
    nert::Node node = MakeNode("greater_bad", "Greater");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Greater"), std::string::npos) << error.what();
    }
}

// Runs Pad over a [1,n] float row with the given mode, pads, and optional constant.
std::vector<float> RunPad(const std::string& mode, std::vector<float> in, std::vector<int64_t> pads,
                          std::vector<int64_t> outDims, const std::vector<float>* constant) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Pad");
    EXPECT_NE(kernel, nullptr);
    int64_t outCount = 1;
    for (int64_t dim : outDims) {
        outCount *= dim;
    }
    std::vector<float> out(static_cast<std::size_t>(outCount), -1.0f);
    nert::Node node = MakeNode("pad0", "Pad");
    node.attributes.emplace_back("mode", std::string(mode));
    std::vector<float> constantCopy = constant != nullptr ? *constant : std::vector<float>{};
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, static_cast<int64_t>(in.size())}),
                                            Int64View(pads, {static_cast<int64_t>(pads.size())})};
    if (constant != nullptr) {
        inputs.push_back(FloatView(constantCopy, {}));
    }
    std::vector<nert::TensorView> outputs = {FloatView(out, outDims)};
    kernel(node, inputs, outputs);
    return out;
}

// Pad constant mode fills with the constant_value input.
TEST(NertKernels, PadConstantWithValue) {
    const std::vector<float> value = {9.0f};
    EXPECT_EQ(RunPad("constant", {1.0f, 2.0f, 3.0f}, {0, 1, 0, 2}, {1, 6}, &value),
              (std::vector<float>{9.0f, 1.0f, 2.0f, 3.0f, 9.0f, 9.0f}));
}

// Pad constant mode defaults to zero when constant_value is omitted.
TEST(NertKernels, PadConstantDefaultZero) {
    EXPECT_EQ(RunPad("constant", {1.0f, 2.0f}, {0, 1, 0, 1}, {1, 4}, nullptr),
              (std::vector<float>{0.0f, 1.0f, 2.0f, 0.0f}));
}

// Pad reflect mirrors around the edge element without repeating it.
TEST(NertKernels, PadReflect) {
    EXPECT_EQ(RunPad("reflect", {1.0f, 2.0f, 3.0f, 4.0f}, {0, 2, 0, 1}, {1, 7}, nullptr),
              (std::vector<float>{3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f}));
}

// Pad edge repeats the border element.
TEST(NertKernels, PadEdge) {
    EXPECT_EQ(RunPad("edge", {1.0f, 2.0f, 3.0f, 4.0f}, {0, 2, 0, 1}, {1, 7}, nullptr),
              (std::vector<float>{1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f, 4.0f}));
}

// Negative pads crop the input.
TEST(NertKernels, PadNegativeCrops) {
    EXPECT_EQ(RunPad("constant", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {0, -1, 0, -2}, {1, 2}, nullptr),
              (std::vector<float>{2.0f, 3.0f}));
}

// Pad rejects an unknown mode, naming the op.
TEST(NertKernelsNegative, PadRejectsWrapMode) {
    try {
        RunPad("wrap", {1.0f, 2.0f}, {0, 1, 0, 1}, {1, 4}, nullptr);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Pad"), std::string::npos) << error.what();
    }
}

// Range over float32: ceil((2.5 - 0) / 0.5) == 5 elements.
TEST(NertKernels, RangeFloat32) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Range");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> start = {0.0f};
    std::vector<float> limit = {2.5f};
    std::vector<float> delta = {0.5f};
    std::vector<float> out(5, -1.0f);
    nert::Node node = MakeNode("range0", "Range");
    std::vector<nert::TensorView> inputs = {FloatView(start, {}), FloatView(limit, {}), FloatView(delta, {})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {5})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{0.0f, 0.5f, 1.0f, 1.5f, 2.0f}));
}

// Range over int64 with a negative delta: ceil((4 - 10) / -3) == 2 elements.
TEST(NertKernels, RangeInt64NegativeDelta) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Range");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> start = {10};
    std::vector<int64_t> limit = {4};
    std::vector<int64_t> delta = {-3};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("range1", "Range");
    std::vector<nert::TensorView> inputs = {Int64View(start, {}), Int64View(limit, {}), Int64View(delta, {})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{10, 7}));
}

// Range rejects a zero delta, naming the op.
TEST(NertKernelsNegative, RangeRejectsZeroDelta) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Range");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> start = {0};
    std::vector<int64_t> limit = {4};
    std::vector<int64_t> delta = {0};
    std::vector<int64_t> out(1, 0);
    nert::Node node = MakeNode("range_bad", "Range");
    std::vector<nert::TensorView> inputs = {Int64View(start, {}), Int64View(limit, {}), Int64View(delta, {})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Range"), std::string::npos) << error.what();
    }
}

// ReduceSum over axis 1 of [2,3] with the default keepdims == 1.
TEST(NertKernels, ReduceSumKeepdimsDefault) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    std::vector<int64_t> axes = {1};
    nert::Node node = MakeNode("rsum0", "ReduceSum");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{6.0f, 15.0f}));
}

// A pre-13 ReduceSum carries axes in an attribute and has no axes input.
TEST(NertKernels, ReduceSumOldAttributeForm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("rsum_attr", "ReduceSum");
    node.schema_version = 11;
    node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{6.0f, 15.0f}));
}

// ReduceSum over axis -2 (axis 0) with keepdims == 0 on int64 data.
TEST(NertKernels, ReduceSumInt64KeepdimsZeroNegativeAxis) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> out(3, 0);
    std::vector<int64_t> axes = {-2};
    nert::Node node = MakeNode("rsum1", "ReduceSum");
    node.attributes.emplace_back("keepdims", int64_t{0});
    std::vector<nert::TensorView> inputs = {Int64View(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{5, 7, 9}));
}

// ReduceSum with an empty axes input and noop_with_empty_axes == 1 copies the input.
TEST(NertKernels, ReduceSumNoopWithEmptyAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    std::vector<int64_t> axes;
    nert::Node node = MakeNode("rsum2", "ReduceSum");
    node.attributes.emplace_back("noop_with_empty_axes", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axes, {0})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

namespace {

// Run GridSample on a 2x2 input {1,2,3,4} and the given grid points.
std::vector<float> RunGridSample(const std::vector<std::pair<std::string, std::string>>& strs, int64_t align,
                                 std::vector<float> grid) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GridSample");
    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};
    const int64_t points = static_cast<int64_t>(grid.size() / 2);
    std::vector<float> out(static_cast<size_t>(points), -1.0f);
    nert::Node node = MakeNode("gs", "GridSample");
    for (const auto& kv : strs) {
        node.attributes.emplace_back(kv.first, kv.second);
    }
    node.attributes.emplace_back("align_corners", align);
    std::vector<nert::TensorView> inputs = {FloatView(in, {1, 1, 2, 2}), FloatView(grid, {1, 1, points, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 1, points})};
    kernel(node, inputs, outputs);
    return out;
}

void ExpectNear(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "index " << i;
    }
}

} // namespace

// Bilinear, align_corners 0: (0.25, 0.25) maps to pixel (0.75, 0.75), weights .0625/.1875/.1875/.5625.
TEST(NertKernels, GridSampleBilinearAlignCornersZero) {
    ASSERT_NE(nert::CpuDevice().kernels().find("GridSample"), nullptr);
    ExpectNear(RunGridSample({{"mode", "bilinear"}}, 0, {0.25f, 0.25f}), {3.25f});
}

// Bilinear, align_corners 1: (0.5, -0.5) maps to pixel (0.75, 0.25), giving 2.25.
TEST(NertKernels, GridSampleBilinearAlignCornersOne) {
    ASSERT_NE(nert::CpuDevice().kernels().find("GridSample"), nullptr);
    ExpectNear(RunGridSample({}, 1, {0.5f, -0.5f}), {2.25f});
}

// Nearest: pixel (0.75, 0.75) rounds to (1, 1) = 4; pixel (0.5, 0.5) rounds half to even to (0, 0) = 1.
TEST(NertKernels, GridSampleNearest) {
    ASSERT_NE(nert::CpuDevice().kernels().find("GridSample"), nullptr);
    ExpectNear(RunGridSample({{"mode", "nearest"}}, 0, {0.25f, 0.25f, 0.0f, 0.0f}), {4.0f, 1.0f});
}

// x = -2 (align_corners 0) is pixel (-1.5, 0.5): zeros gives 0 (both neighbours outside).
TEST(NertKernels, GridSamplePaddingZeros) {
    ASSERT_NE(nert::CpuDevice().kernels().find("GridSample"), nullptr);
    ExpectNear(RunGridSample({{"padding_mode", "zeros"}}, 0, {-2.0f, 0.0f}), {0.0f});
}

// Border clamps to column 0: mean of rows (1, 3) = 2.
TEST(NertKernels, GridSamplePaddingBorder) {
    ASSERT_NE(nert::CpuDevice().kernels().find("GridSample"), nullptr);
    ExpectNear(RunGridSample({{"padding_mode", "border"}}, 0, {-2.0f, 0.0f}), {2.0f});
}

// Reflection about -0.5 maps x pixel -1.5 to 0.5, so the sample is (0.5, 0.5) = 2.5.
TEST(NertKernels, GridSamplePaddingReflection) {
    ASSERT_NE(nert::CpuDevice().kernels().find("GridSample"), nullptr);
    ExpectNear(RunGridSample({{"padding_mode", "reflection"}}, 0, {-2.0f, 0.0f}), {2.5f});
}

// Einsum "bchw,bnc->bnhw": n0 selects channel 0 = {1,2}, n1 sums both channels = {4,6}.
TEST(NertKernels, EinsumBchwBncToBnhw) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w = {1.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("es", "Einsum");
    node.attributes.emplace_back("equation", std::string("bchw,bnc->bnhw"));
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2, 1, 2}), FloatView(w, {1, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 2, 1, 2})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 4.0f, 6.0f}));
}

// Einsum "ij,jk->ik" is a matrix product.
TEST(NertKernels, EinsumMatrixProduct) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("es", "Einsum");
    node.attributes.emplace_back("equation", std::string("ij, jk -> ik"));
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 2}), FloatView(b, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{19.0f, 22.0f, 43.0f, 50.0f}));
}

// Einsum "ii->" is the trace; implicit "ii" also sums the diagonal.
TEST(NertKernels, EinsumTrace) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    for (const char* equation : {"ii->", "ii"}) {
        std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> out(1, 0.0f);
        nert::Node node = MakeNode("es", "Einsum");
        node.attributes.emplace_back("equation", std::string(equation));
        std::vector<nert::TensorView> inputs = {FloatView(a, {2, 2})};
        std::vector<nert::TensorView> outputs = {FloatView(out, {})};
        kernel(node, inputs, outputs);
        EXPECT_EQ(out, (std::vector<float>{5.0f})) << equation;
    }
}

// Einsum "ij->ji" transposes.
TEST(NertKernels, EinsumTranspose) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("es", "Einsum");
    node.attributes.emplace_back("equation", std::string("ij->ji"));
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3, 2})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}));
}

// second operand: y[b,i,j] = sum_k x[b,i,k]*z[b,j,k].
TEST(NertKernels, EinsumEllipsisBatchMatMul) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> z = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> out(8, 0.0f);
    nert::Node node = MakeNode("es", "Einsum");
    node.attributes.emplace_back("equation", std::string("...pd,...qd->...pq"));
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 2, 3}), FloatView(z, {1, 4, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 2, 4})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{14.0f, 32.0f, 50.0f, 68.0f, 32.0f, 77.0f, 122.0f, 167.0f}));
}

// Einsum "...i,...i->...i" broadcasts the shorter ellipsis: x[3] against
// z[2,3] multiplies elementwise across both rows.
TEST(NertKernels, EinsumEllipsisBroadcasts) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> z = {10.0f, 20.0f, 30.0f, 100.0f, 200.0f, 300.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("es", "Einsum");
    node.attributes.emplace_back("equation", std::string("...i,...i->...i"));
    std::vector<nert::TensorView> inputs = {FloatView(x, {3}), FloatView(z, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};
    kernel(node, inputs, outputs);
    EXPECT_EQ(out, (std::vector<float>{10.0f, 40.0f, 90.0f, 100.0f, 400.0f, 900.0f}));
}

// Einsum with malformed dots ("..i") is rejected, naming the op.
TEST(NertKernelsNegative, EinsumRejectsMalformedDots) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("es_bad", "Einsum");
    node.attributes.emplace_back("equation", std::string("..i,i->i"));
    std::vector<nert::TensorView> inputs = {FloatView(a, {2}), FloatView(a, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};
    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Einsum"), std::string::npos) << error.what();
    }
}

// Einsum with an ellipsis but no explicit output is rejected, naming the op.
TEST(NertKernelsNegative, EinsumRejectsImplicitEllipsisOutput) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("es_bad", "Einsum");
    node.attributes.emplace_back("equation", std::string("...i"));
    std::vector<nert::TensorView> inputs = {FloatView(a, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};
    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Einsum"), std::string::npos) << error.what();
    }
}

// GridSample mode "bicubic" is rejected, naming the op.
TEST(NertKernelsNegative, GridSampleRejectsBicubic) {
    ASSERT_NE(nert::CpuDevice().kernels().find("GridSample"), nullptr);
    try {
        RunGridSample({{"mode", "bicubic"}}, 0, {0.0f, 0.0f});
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("GridSample"), std::string::npos) << error.what();
    }
}

// ConstantOfShape fills [2,2] with the float32 value attribute 2.5.
TEST(NertKernels, ConstantOfShapeFloatFill) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2, 2};
    std::vector<float> out(4, 0.0f);
    nert::Node node = MakeNode("cos_f", "ConstantOfShape");
    node.attributes.emplace_back("value", 2.5f);
    std::vector<nert::TensorView> inputs = {Int64View(shape, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.5f, 2.5f, 2.5f, 2.5f}));
}

// ConstantOfShape fills [3] with the bool value attribute true.
TEST(NertKernels, ConstantOfShapeBoolFill) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {3};
    std::vector<std::uint8_t> out(3, 0);
    nert::Node node = MakeNode("cos_b", "ConstantOfShape");
    node.attributes.emplace_back("value", true);
    std::vector<nert::TensorView> inputs = {Int64View(shape, {1})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 1, 1}));
}

// ConstantOfShape without a value attribute fills float32 zero over [3].
TEST(NertKernels, ConstantOfShapeDefaultFillsFloatZero) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {3};
    std::vector<float> out(3, 7.0f);
    nert::Node node = MakeNode("cos_d", "ConstantOfShape");
    std::vector<nert::TensorView> inputs = {Int64View(shape, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{0.0f, 0.0f, 0.0f}));
}

// ConstantOfShape rejects a bool value written into a float32 output, naming the op.
TEST(NertKernelsNegative, ConstantOfShapeBoolValueRejectsFloatOutput) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("cos_mismatch", "ConstantOfShape");
    node.attributes.emplace_back("value", true);
    std::vector<nert::TensorView> inputs = {Int64View(shape, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ConstantOfShape"), std::string::npos) << error.what();
    }
}

// ConstantOfShape rejects a string value attribute, naming the op.
TEST(NertKernelsNegative, ConstantOfShapeRejectsStringValue) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ConstantOfShape");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> shape = {2};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("cos_str", "ConstantOfShape");
    node.attributes.emplace_back("value", std::string("one"));
    std::vector<nert::TensorView> inputs = {Int64View(shape, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ConstantOfShape"), std::string::npos) << error.what();
    }
}

// Slice [0:0] of an int64 shape vector yields the empty vector (dims {0}).
TEST(NertKernels, SliceEmptyRangeInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {2, 6};
    std::vector<int64_t> out;
    std::vector<int64_t> starts = {0};
    std::vector<int64_t> ends = {0};
    nert::Node node = MakeNode("slice_empty", "Slice");
    std::vector<nert::TensorView> inputs = {Int64View(in, {2}), Int64View(starts, {1}), Int64View(ends, {1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {0})};

    kernel(node, inputs, outputs);

    EXPECT_TRUE(out.empty());
}

// Concat of an empty int64 vector with [3,4] is [3,4].
TEST(NertKernels, ConcatEmptyInputInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> empty;
    std::vector<int64_t> tail = {3, 4};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("concat_empty", "Concat");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<nert::TensorView> inputs = {Int64View(empty, {0}), Int64View(tail, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{3, 4}));
}

// Double accumulation: 1e8 + 1 - 1e8 is 1. A float32 running sum loses the 1
// (1e8 + 1 rounds back to 1e8, whose float32 ulp is 8) and would give 0.
TEST(NertKernels, MatMulAccumulatesInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("MatMul");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0e8f, 1.0f, -1.0e8f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("mm_acc", "MatMul");
    std::vector<nert::TensorView> inputs = {FloatView(a, {1, 3}), FloatView(b, {3, 1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f}));
}

// Gemm accumulates 1e8 + 1 - 1e8 in double, giving 1 (float32 would give 0).
TEST(NertKernels, GemmAccumulatesInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gemm");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0e8f, 1.0f, -1.0e8f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("gemm_acc", "Gemm");
    std::vector<nert::TensorView> inputs = {FloatView(a, {1, 3}), FloatView(b, {3, 1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f}));
}

// Conv (1x1 kernel over three channels) accumulates 1e8 + 1 - 1e8 in double,
// giving 1 (float32 would give 0).
TEST(NertKernels, ConvAccumulatesInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Conv");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> x = {1.0e8f, 1.0f, -1.0e8f};
    std::vector<float> w = {1.0f, 1.0f, 1.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("conv_acc", "Conv");
    std::vector<nert::TensorView> inputs = {FloatView(x, {1, 3, 1, 1}), FloatView(w, {1, 3, 1, 1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1, 1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f}));
}

// ReduceSum accumulates 1e8 + 1 - 1e8 in double, giving 1 (float32 would give 0).
TEST(NertKernels, ReduceSumAccumulatesInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0e8f, 1.0f, -1.0e8f};
    std::vector<int64_t> axes = {0};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("rs_acc", "ReduceSum");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f}));
}

// ReduceMean of {1e8, 1, -1e8} is 1/3, rounded once to float32 (a float32
// running sum would give 0).
TEST(NertKernels, ReduceMeanAccumulatesInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceMean");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0e8f, 1.0f, -1.0e8f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("rm_acc", "ReduceMean");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{static_cast<float>(1.0 / 3.0)}));
}

// Einsum "ij,jk->ik" accumulates 1e8 + 1 - 1e8 in double, giving 1 (float32
// would give 0).
TEST(NertKernels, EinsumAccumulatesInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Einsum");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0e8f, 1.0f, -1.0e8f};
    std::vector<float> b = {1.0f, 1.0f, 1.0f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("es_acc", "Einsum");
    node.attributes.emplace_back("equation", std::string("ij,jk->ik"));
    std::vector<nert::TensorView> inputs = {FloatView(a, {1, 3}), FloatView(b, {3, 1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f}));
}

// Softmax over {0, -17 x 1000}: each exp(-17) = 4.14e-8 is below half a float32
// ulp of 1, so a float32 running sum stays 1 and y0 would be exactly 1. The
// double sum keeps the tail: y0 = 1 / (1 + 1000 e^-17) = 0.99995860...
TEST(NertKernels, SoftmaxSumsInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Softmax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(1001, -17.0f);
    in[0] = 0.0f;
    std::vector<float> out(1001, 0.0f);
    nert::Node node = MakeNode("sm_acc", "Softmax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {1001})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {1001})};

    kernel(node, inputs, outputs);

    const double tail = std::exp(-17.0);
    const double total = 1.0 + 1000.0 * tail;
    EXPECT_EQ(out[0], static_cast<float>(1.0 / total));
    EXPECT_LT(out[0], 0.99996f);
    EXPECT_EQ(out[1], static_cast<float>(tail / total));
}

// Elu with default alpha 1.0: x >= 0 passes through, x < 0 maps to exp(x)-1.
// in = [-1, 0, 1]: exp(-1)-1 = -0.63212055...
TEST(NertKernels, EluDefaultAlpha) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Elu");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {-1.0f, 0.0f, 1.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("elu0", "Elu");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 1.0f);
    EXPECT_NEAR(out[0], static_cast<float>(std::exp(-1.0) - 1.0), 1e-6f);
}

// Elu with alpha 2.0: negative side scales by alpha.
// in = [-1, 2]: 2*(exp(-1)-1) = -1.2642411..., 2 passes through.
TEST(NertKernels, EluNonDefaultAlpha) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Elu");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {-1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("elu_alpha", "Elu");
    node.attributes.emplace_back("alpha", 2.0f);
    std::vector<nert::TensorView> inputs = {FloatView(in, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[1], 2.0f);
    EXPECT_NEAR(out[0], static_cast<float>(2.0 * (std::exp(-1.0) - 1.0)), 1e-6f);
}

// Max broadcasts [2,1] against [1,3] and takes the elementwise maximum:
// [[1,4,2],[3,0,5]] vs [[2,1,6]] -> [[2,4,6],[3,1,6]].
TEST(NertKernels, MaxBroadcastsTwoInputs) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Max");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 4.0f, 2.0f, 3.0f, 0.0f, 5.0f};
    std::vector<float> b = {2.0f, 1.0f, 6.0f};
    std::vector<float> out(6, 0.0f);
    nert::Node node = MakeNode("max0", "Max");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{2.0f, 4.0f, 6.0f, 3.0f, 1.0f, 6.0f}));
}

// Min over three int64 inputs: elementwise minimum, incl. negative values.
// [5, -1] vs [3, 2] vs [4, -3] -> [3, -3].
TEST(NertKernels, MinThreeInt64Inputs) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Min");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {5, -1};
    std::vector<int64_t> b = {3, 2};
    std::vector<int64_t> c = {4, -3};
    std::vector<int64_t> out(2, 0);
    nert::Node node = MakeNode("min0", "Min");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2}), Int64View(b, {2}), Int64View(c, {2})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{3, -3}));
}

// Elu rejects an int64 input, naming the op and node.
TEST(NertKernelsNegative, EluRejectsInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Elu");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("elu_bad", "Elu");
    std::vector<nert::TensorView> inputs = {Int64View(in, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Elu"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("elu_bad"), std::string::npos) << error.what();
    }
}

// Max rejects mixed float32/int64 inputs, naming the op and node.
TEST(NertKernelsNegative, MaxRejectsMixedDtypes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Max");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<int64_t> b = {1, 2};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("max_bad", "Max");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Max"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("max_bad"), std::string::npos) << error.what();
    }
}

// Min rejects non-broadcastable shapes, naming the op and node.
TEST(NertKernelsNegative, MinRejectsBroadcastMismatch) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Min");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f};
    std::vector<float> out(3, 0.0f);
    nert::Node node = MakeNode("min_bad", "Min");
    std::vector<nert::TensorView> inputs = {FloatView(a, {3}), FloatView(b, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Min"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("min_bad"), std::string::npos) << error.what();
    }
}

// LessOrEqual broadcasts [2,1] against [1,3]: [[1,4,2],[3,0,5]] <= [[2,1,6]]
// -> [[1,0,1],[0,1,1]].
TEST(NertKernels, LessOrEqualBroadcastsTwoInputs) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("LessOrEqual");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 4.0f, 2.0f, 3.0f, 0.0f, 5.0f};
    std::vector<float> b = {2.0f, 1.0f, 6.0f};
    std::vector<std::uint8_t> out(6, 0);
    nert::Node node = MakeNode("le0", "LessOrEqual");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2, 3}), FloatView(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 0, 1, 0, 1, 1}));
}

// GreaterOrEqual over int64 with equality on the diagonal:
// [1,2] >= [1,3] -> [1,0]; equality counts as true (off-default vs Greater).
TEST(NertKernels, GreaterOrEqualInt64WithEquality) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GreaterOrEqual");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {1, 3};
    std::vector<std::uint8_t> out(2, 0);
    nert::Node node = MakeNode("ge0", "GreaterOrEqual");
    std::vector<nert::TensorView> inputs = {Int64View(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 0}));
}

// Or broadcasts [2,1] bool against [1,3] bool:
// [[1,0,0],[0,1,0]] || [[0,1,0]] -> [[1,1,0],[0,1,0]].
TEST(NertKernels, OrBroadcastsBoolInputs) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Or");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> a = {1, 0, 0, 0, 1, 0};
    std::vector<std::uint8_t> b = {0, 1, 0};
    std::vector<std::uint8_t> out(6, 0);
    nert::Node node = MakeNode("or0", "Or");
    std::vector<nert::TensorView> inputs = {BoolView(a, {2, 3}), BoolView(b, {1, 3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(BoolBytes(out), (std::vector<std::uint8_t>{1, 1, 0, 0, 1, 0}));
}

// LessOrEqual rejects mixed float32/int64 inputs, naming op and node.
TEST(NertKernelsNegative, LessOrEqualRejectsMixedDtypes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("LessOrEqual");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<int64_t> b = {1, 2};
    std::vector<std::uint8_t> out(2, 0);
    nert::Node node = MakeNode("le_bad", "LessOrEqual");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2}), Int64View(b, {2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("LessOrEqual"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("le_bad"), std::string::npos) << error.what();
    }
}

// GreaterOrEqual rejects a non-bool output, naming op and node.
TEST(NertKernelsNegative, GreaterOrEqualRejectsNonBoolOutput) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("GreaterOrEqual");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f, 3.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("ge_bad", "GreaterOrEqual");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2}), FloatView(b, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("GreaterOrEqual"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("ge_bad"), std::string::npos) << error.what();
    }
}

// Or rejects float32 inputs, naming op and node.
TEST(NertKernelsNegative, OrRejectsNonBoolInput) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Or");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> a = {1.0f, 0.0f};
    std::vector<std::uint8_t> b = {1, 0};
    std::vector<std::uint8_t> out(2, 0);
    nert::Node node = MakeNode("or_bad", "Or");
    std::vector<nert::TensorView> inputs = {FloatView(a, {2}), BoolView(b, {2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("Or"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("or_bad"), std::string::npos) << error.what();
    }
}

// ReduceL2 over axis 1 keepdims: rows [3,4] -> 5, [0,0] -> 0, [1,1] -> sqrt(2).
// sqrt(2) = 1.41421356...
TEST(NertKernels, ReduceL2AxisKeepdims) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceL2");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {3.0f, 4.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<int64_t> axes = {1};
    std::vector<float> out(3, -1.0f);
    nert::Node node = MakeNode("rl2_0", "ReduceL2");
    std::vector<nert::TensorView> inputs = {FloatView(in, {3, 2}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {3, 1})};

    kernel(node, inputs, outputs);

    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_NEAR(out[2], static_cast<float>(std::sqrt(2.0)), 1e-6f);
}

// ReduceL2 over all axes no-keepdims with double accumulation:
// {1e8, 1, -1e8} squares sum to 2e16 + 1; float32 sum of squares loses the 1
// (2e16 ulp is 4), the double path keeps it: sqrt(2e16+1) = 141421356.24...
TEST(NertKernels, ReduceL2AccumulatesInDouble) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceL2");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0e8f, 1.0f, -1.0e8f};
    std::vector<float> out(1, 0.0f);
    nert::Node node = MakeNode("rl2_acc", "ReduceL2");
    node.attributes.emplace_back("keepdims", int64_t{0});
    std::vector<nert::TensorView> inputs = {FloatView(in, {3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out[0], static_cast<float>(std::sqrt(2.0e16 + 1.0)));
    EXPECT_GT(out[0], 1.4142135e8f);
}

// ReduceL2 rejects int64 data, naming op and node.
TEST(NertKernelsNegative, ReduceL2RejectsInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ReduceL2");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {1, 2, 3};
    std::vector<int64_t> out(1, 0);
    nert::Node node = MakeNode("rl2_bad", "ReduceL2");
    std::vector<nert::TensorView> inputs = {Int64View(in, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {1})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ReduceL2"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("rl2_bad"), std::string::npos) << error.what();
    }
}

// ArgMax over axis 1 keepdims: rows [1,4,2] -> 1, [3,3,5] -> 2 (first max).
TEST(NertKernels, ArgMaxAxisKeepdimsFirstMax) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ArgMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 4.0f, 2.0f, 3.0f, 3.0f, 5.0f};
    std::vector<int64_t> out(2, -1);
    nert::Node node = MakeNode("argmax0", "ArgMax");
    node.attributes.emplace_back("axis", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {2, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{1, 2}));
}

// ArgMax with select_last_index 1 and keepdims 0: [2, 5, 5] -> 2 (last tie),
// output rank drops to scalar shape {}.
TEST(NertKernels, ArgMaxLastIndexNoKeepdims) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ArgMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {2, 5, 5};
    std::vector<int64_t> out(1, -1);
    nert::Node node = MakeNode("argmax_last", "ArgMax");
    node.attributes.emplace_back("axis", int64_t{0});
    node.attributes.emplace_back("keepdims", int64_t{0});
    node.attributes.emplace_back("select_last_index", int64_t{1});
    std::vector<nert::TensorView> inputs = {Int64View(in, {3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{2}));
}

// ArgMax rejects a float32 output, naming op and node.
TEST(NertKernelsNegative, ArgMaxRejectsNonInt64Output) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ArgMax");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("argmax_bad", "ArgMax");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ArgMax"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("argmax_bad"), std::string::npos) << error.what();
    }
}

// CumSum along axis 1 inclusive: rows [1,2,3] -> [1,3,6], [4,0,1] -> [4,4,5].
TEST(NertKernels, CumSumAxisInclusive) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("CumSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 1.0f};
    std::vector<int64_t> axis = {1};
    std::vector<float> out(6, -1.0f);
    nert::Node node = MakeNode("cs0", "CumSum");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3}), Int64View(axis, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 3.0f, 6.0f, 4.0f, 4.0f, 5.0f}));
}

// CumSum exclusive + reverse over int64: [5,1,4] reversed-exclusive
// -> [5,4,0] (suffix sums excluding self, from the end).
TEST(NertKernels, CumSumExclusiveReverseInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("CumSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> in = {5, 1, 4};
    std::vector<int64_t> axis = {0};
    std::vector<int64_t> out(3, -1);
    nert::Node node = MakeNode("cs_rev", "CumSum");
    node.attributes.emplace_back("exclusive", int64_t{1});
    node.attributes.emplace_back("reverse", int64_t{1});
    std::vector<nert::TensorView> inputs = {Int64View(in, {3}), Int64View(axis, {1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{5, 4, 0}));
}

// CumSum rejects an axis out of range, naming op and node.
TEST(NertKernelsNegative, CumSumRejectsAxisOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("CumSum");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in = {1.0f, 2.0f};
    std::vector<int64_t> axis = {5};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("cs_bad", "CumSum");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2}), Int64View(axis, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("CumSum"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("cs_bad"), std::string::npos) << error.what();
    }
}

// EyeLike default: 2x3 float input shape gives [[1,0,0],[0,1,0]].
TEST(NertKernels, EyeLikeDefaultDiagonal) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("EyeLike");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(6, 7.0f);
    std::vector<float> out(6, -1.0f);
    nert::Node node = MakeNode("eye0", "EyeLike");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}));
}

// EyeLike with k=1 and int64 output: 3x3 superdiagonal
// [[0,1,0],[0,0,1],[0,0,0]].
TEST(NertKernels, EyeLikeSuperdiagonalInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("EyeLike");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(9, 0.0f);
    std::vector<int64_t> out(9, -1);
    nert::Node node = MakeNode("eye_k", "EyeLike");
    node.attributes.emplace_back("k", int64_t{1});
    std::vector<nert::TensorView> inputs = {FloatView(in, {3, 3})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{0, 1, 0, 0, 0, 1, 0, 0, 0}));
}

// EyeLike rejects a rank-3 input, naming op and node.
TEST(NertKernelsNegative, EyeLikeRejectsRank3) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("EyeLike");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> in(8, 0.0f);
    std::vector<float> out(8, 0.0f);
    nert::Node node = MakeNode("eye_bad", "EyeLike");
    std::vector<nert::TensorView> inputs = {FloatView(in, {2, 2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2, 2, 2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("EyeLike"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("eye_bad"), std::string::npos) << error.what();
    }
}

// ScatterND scatters two rows into a 4x2 zero matrix:
// indices [[0],[2]], updates [[1,2],[3,4]] -> [[1,2],[0,0],[3,4],[0,0]].
TEST(NertKernels, ScatterNDRowScatter) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ScatterND");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> data(8, 0.0f);
    std::vector<int64_t> indices = {0, 2};
    std::vector<float> updates = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(8, -1.0f);
    nert::Node node = MakeNode("snd0", "ScatterND");
    std::vector<nert::TensorView> inputs = {FloatView(data, {4, 2}), Int64View(indices, {2, 1}),
                                            FloatView(updates, {2, 2})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {4, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f}));
}

// ScatterND with a negative index and int64 data: data [10,20,30],
// indices [[-1]], updates [99] -> [10,20,99].
TEST(NertKernels, ScatterNDNegativeIndexInt64) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ScatterND");
    ASSERT_NE(kernel, nullptr);

    std::vector<int64_t> data = {10, 20, 30};
    std::vector<int64_t> indices = {-1};
    std::vector<int64_t> updates = {99};
    std::vector<int64_t> out(3, -1);
    nert::Node node = MakeNode("snd_neg", "ScatterND");
    std::vector<nert::TensorView> inputs = {Int64View(data, {3}), Int64View(indices, {1, 1}), Int64View(updates, {1})};
    std::vector<nert::TensorView> outputs = {Int64View(out, {3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<int64_t>{10, 20, 99}));
}

// ScatterND rejects an out-of-range index, naming op and node.
TEST(NertKernelsNegative, ScatterNDRejectsIndexOutOfRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("ScatterND");
    ASSERT_NE(kernel, nullptr);

    std::vector<float> data = {1.0f, 2.0f};
    std::vector<int64_t> indices = {5};
    std::vector<float> updates = {9.0f};
    std::vector<float> out(2, 0.0f);
    nert::Node node = MakeNode("snd_bad", "ScatterND");
    std::vector<nert::TensorView> inputs = {FloatView(data, {2}), Int64View(indices, {1, 1}), FloatView(updates, {1})};
    std::vector<nert::TensorView> outputs = {FloatView(out, {2})};

    try {
        kernel(node, inputs, outputs);
        FAIL() << "expected InferenceException";
    } catch (const nert::InferenceException& error) {
        EXPECT_NE(std::string(error.what()).find("ScatterND"), std::string::npos) << error.what();
        EXPECT_NE(std::string(error.what()).find("snd_bad"), std::string::npos) << error.what();
    }
}

// Flatten bool (OWLv2's text attention mask) copies bytes in flat order.
TEST(NertKernels, FlattenBoolCopies) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Flatten");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1, 1};
    std::vector<std::uint8_t> out(4, 0);
    nert::Node node = MakeNode("flatten_b", "Flatten");
    node.attributes.emplace_back("axis", int64_t{-1});
    std::vector<nert::TensorView> inputs = {BoolView(in, {2, 2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {4, 1})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<std::uint8_t>{1, 0, 1, 1}));
}

// Gather bool data (OWLv2's text attention mask) selects bytes unchanged.
TEST(NertKernels, GatherBoolData) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Gather");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1};
    std::vector<int64_t> indices = {2, 0};
    std::vector<std::uint8_t> out(2, 0);
    nert::Node node = MakeNode("gather_b", "Gather");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<nert::TensorView> inputs = {BoolView(in, {3}), Int64View(indices, {2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<std::uint8_t>{1, 1}));
}

// Concat bool rows along axis 0 copies bytes unchanged.
TEST(NertKernels, ConcatBoolRows) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Concat");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> a = {1, 0};
    std::vector<std::uint8_t> b = {0, 1};
    std::vector<std::uint8_t> out(4, 0);
    nert::Node node = MakeNode("cc_b", "Concat");
    node.attributes.emplace_back("axis", int64_t{0});
    std::vector<nert::TensorView> inputs = {BoolView(a, {1, 2}), BoolView(b, {1, 2})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<std::uint8_t>{1, 0, 0, 1}));
}

// Slice bool data selects bytes unchanged.
TEST(NertKernels, SliceBoolRange) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Slice");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1, 1};
    std::vector<int64_t> starts = {1};
    std::vector<int64_t> ends = {3};
    std::vector<std::uint8_t> out(2, 0);
    nert::Node node = MakeNode("sl_b", "Slice");
    std::vector<nert::TensorView> inputs = {BoolView(in, {4}), Int64View(starts, {1}), Int64View(ends, {1})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<std::uint8_t>{0, 1}));
}

// Transpose bool data with perm {1,0} reorders bytes.
TEST(NertKernels, TransposeBoolPerm) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Transpose");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1, 0, 1, 0};
    std::vector<std::uint8_t> out(6, 0);
    nert::Node node = MakeNode("tr_b", "Transpose");
    node.attributes.emplace_back("perm", std::vector<int64_t>{1, 0});
    std::vector<nert::TensorView> inputs = {BoolView(in, {2, 3})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {3, 2})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, (std::vector<std::uint8_t>{1, 0, 0, 1, 1, 0}));
}

// Unsqueeze bool data only changes the shape; byte order is preserved.
TEST(NertKernels, UnsqueezeBoolAxes) {
    nert::KernelFn kernel = nert::CpuDevice().kernels().find("Unsqueeze");
    ASSERT_NE(kernel, nullptr);

    std::vector<std::uint8_t> in = {1, 0, 1, 0, 1, 0};
    std::vector<int64_t> axes = {0};
    std::vector<std::uint8_t> out(6, 0);
    nert::Node node = MakeNode("un_b", "Unsqueeze");
    std::vector<nert::TensorView> inputs = {BoolView(in, {2, 3}), Int64View(axes, {1})};
    std::vector<nert::TensorView> outputs = {BoolView(out, {1, 2, 3})};

    kernel(node, inputs, outputs);

    EXPECT_EQ(out, in);
}

TEST(NertKernels, SplitNumOutputsUnevenAndConflicts) {
    auto kernel = nert::CpuDevice().kernels().find("Split");
    nert::Node node = MakeNode("split18", "Split");
    node.schema_version = 18;
    node.attributes = {nert::Attribute{"axis", int64_t{-1}}, nert::Attribute{"num_outputs", int64_t{3}}};
    std::vector<int64_t> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, a(4), b(4), c(2);
    kernel(node, {Int64View(x, {2, 5})}, {Int64View(a, {2, 2}), Int64View(b, {2, 2}), Int64View(c, {2, 1})});
    EXPECT_EQ(a, (std::vector<int64_t>{0, 1, 5, 6}));
    EXPECT_EQ(b, (std::vector<int64_t>{2, 3, 7, 8}));
    EXPECT_EQ(c, (std::vector<int64_t>{4, 9}));
    std::vector<int64_t> sizes{2, 2, 1};
    EXPECT_THROW(kernel(node, {Int64View(x, {2, 5}), Int64View(sizes, {3})},
                        {Int64View(a, {2, 2}), Int64View(b, {2, 2}), Int64View(c, {2, 1})}),
                 nert::InferenceException);
    node.schema_version = 13;
    EXPECT_THROW(
        kernel(node, {Int64View(x, {2, 5})}, {Int64View(a, {2, 2}), Int64View(b, {2, 2}), Int64View(c, {2, 1})}),
        nert::InferenceException);
}

TEST(NertKernels, SignSpecialValuesAndIntegerExtremes) {
    auto kernel = nert::CpuDevice().kernels().find("Sign");
    ASSERT_NE(kernel, nullptr);
    nert::Node node = MakeNode("sign", "Sign");
    std::vector<float> x{-INFINITY, -2, -0.0f, 0, 3, INFINITY, NAN}, y(7);
    kernel(node, {FloatView(x, {7})}, {FloatView(y, {7})});
    EXPECT_EQ(std::vector<float>(y.begin(), y.begin() + 6), (std::vector<float>{-1, -1, 0, 0, 1, 1}));
    EXPECT_TRUE(std::isnan(y[6]));
    std::vector<int64_t> ix{INT64_MIN, -1, 0, 1, INT64_MAX}, iy(5);
    kernel(node, {Int64View(ix, {5})}, {Int64View(iy, {5})});
    EXPECT_EQ(iy, (std::vector<int64_t>{-1, -1, 0, 1, 1}));
    EXPECT_THROW(kernel(node, {FloatView(x, {7})}, {FloatView(y, {1, 7})}), nert::InferenceException);
}
