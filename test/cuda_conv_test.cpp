// CUDA Conv via cuDNN (N1b): parity against the CPU oracle.
//
// Needs a cuDNN build AND a CUDA device; without either every test SKIPs
// (never fails). The Conv table row exists only under NERT_HAS_CUDNN, so
// the table-missing check doubles as the no-cuDNN skip.

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

// GTEST_SKIP must fire in the test body (it only returns from its own
// function), so helpers report readiness and each TEST skips explicitly.
bool HasCudaConv() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        return false;
    }
#ifdef NERT_HAS_CUDNN
    return nert::CudaDevice().kernels().find("Conv") != nullptr;
#else
    return false;
#endif
}

std::vector<float> RunCpuConv(nert::Node node, std::vector<float>& x, std::vector<int64_t> xd, std::vector<float>& w,
                              std::vector<int64_t> wd, std::vector<float>* bias, std::vector<int64_t> outDims) {
    std::vector<nert::TensorView> inputs = {HostFloat(x, xd), HostFloat(w, wd)};
    if (bias != nullptr) {
        inputs.push_back(HostFloat(*bias, {wd[0]}));
    }
    const int64_t count = outDims[0] * outDims[1] * outDims[2] * (outDims.size() > 3 ? outDims[3] : 1);
    std::vector<float> out(static_cast<std::size_t>(count), 0.0F);
    nert::CpuDevice().kernels().find("Conv")(node, inputs, {HostFloat(out, outDims)});
    return out;
}

} // namespace

// 1x1 conv, no bias: y[m] = sum_c w[m,c] * x[c] per position.
// X[c0]=[[1,2],[3,4]], X[c1]=[[5,6],[7,8]]; W = [[1,0],[0,1],[1,1]].
TEST(NertCudaConv, PointwiseNoBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> w = {1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    nert::Node node = MakeNode("cv0", "Conv");
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    const std::vector<float> expected = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 6.0F, 8.0F, 10.0F, 12.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 2, 2, 2}, w, {3, 2, 1, 1}, nullptr, {1, 3, 2, 2}), expected);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    std::vector<float> gpu(12, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {3, 2, 1, 1})},
                                              {DeviceFloat(dy, {1, 3, 2, 2})});
    gpu = Download(cuda, dy, 12);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// 3x3 stride-1 pad-1 with bias 10 over 1..9: hand-computed window sums
// [22,31,26,37,55,43,34,49,38], all within tolerance of cuDNN.
TEST(NertCudaConv, Padded3x3WithBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    std::vector<float> w(9, 1.0F);
    std::vector<float> b = {10.0F};
    nert::Node node = MakeNode("cv1", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    const std::vector<float> expected = {22.0F, 31.0F, 26.0F, 37.0F, 55.0F, 43.0F, 34.0F, 49.0F, 38.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 1, 3, 3}, w, {1, 1, 3, 3}, &b, {1, 1, 3, 3}), expected);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(9, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 1, 3, 3}), DeviceFloat(dw, {1, 1, 3, 3}), DeviceFloat(db, {1})},
        {DeviceFloat(dy, {1, 1, 3, 3})});
    gpu = Download(cuda, dy, 9);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// Depthwise (group=2=C=M) 1x1 with bias: y[m] = w[m]*x[m] + b[m].
// y0 = 2*x0+1 -> [3,5,7,9]; y1 = 3*x1-1 -> [14,17,20,23].
TEST(NertCudaConv, DepthwiseWithBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<float> w = {2.0F, 3.0F};
    std::vector<float> b = {1.0F, -1.0F};
    nert::Node node = MakeNode("cv2", "Conv");
    node.attributes.emplace_back("group", int64_t{2});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    const std::vector<float> expected = {3.0F, 5.0F, 7.0F, 9.0F, 14.0F, 17.0F, 20.0F, 23.0F};
    EXPECT_EQ(RunCpuConv(node, x, {1, 2, 2, 2}, w, {2, 1, 1, 1}, &b, {1, 2, 2, 2}), expected);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(8, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {2, 1, 1, 1}), DeviceFloat(db, {2})},
        {DeviceFloat(dy, {1, 2, 2, 2})});
    gpu = Download(cuda, dy, 8);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// Conv rejects a rank-1 bias whose length is not M, naming node + op.
TEST(NertCudaConv, RejectsBadBiasShape) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(8, 1.0F);
    std::vector<float> w(6, 1.0F);
    std::vector<float> b = {0.0F, 0.0F};
    std::vector<float> y(12, 0.0F);
    nert::Node node = MakeNode("cv_bad", "Conv");
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Conv")(
            node, {DeviceFloat(dx, {1, 2, 2, 2}), DeviceFloat(dw, {3, 2, 1, 1}), DeviceFloat(db, {2})},
            {DeviceFloat(dy, {1, 3, 2, 2})});
        FAIL() << "expected InferenceException for a bad bias shape";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cv_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Conv"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// Single-Conv graph through the executor on CUDA matches CPU, proving
// staging + dispatch for the new row.
TEST(NertCudaConv, ModelConvMatchesCpu) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }

    nert::Graph graph;
    graph.inputs = {{"x", {nert::DataType::Float32, {}}},
                    {"w", {nert::DataType::Float32, {}}},
                    {"b", {nert::DataType::Float32, {}}}};
    nert::Node conv = MakeNode("cv", "Conv");
    conv.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    conv.inputs = {"x", "w", "b"};
    conv.outputs = {"y"};
    graph.nodes = {conv};
    graph.outputs = {{"y", {nert::DataType::Float32, {}}}};
    const nert::ShapeMap dims = {{"x", {1, 1, 3, 3}}, {"w", {1, 1, 3, 3}}, {"b", {1}}};
    const std::map<std::string, std::vector<float>> inputs = {
        {"x", {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F}},
        {"w", std::vector<float>(9, 1.0F)},
        {"b", {10.0F}}};

    nert::Model cpu(graph, dims, nert::CpuDevice());
    const std::vector<float> expected = cpu.Run(inputs).outputs.at("y");
    EXPECT_EQ(expected, (std::vector<float>{22.0F, 31.0F, 26.0F, 37.0F, 55.0F, 43.0F, 34.0F, 49.0F, 38.0F}));

    nert::Model cuda(graph, dims, nert::CudaDevice());
    ExpectNear(cuda.Run(inputs).outputs.at("y"), expected);
}

// The RT-DETRv4-s stem geometry that CUDA used to reject ('/model/backbone/
// stem/stem2b/conv/Conv'): kernel 2x2, pads=[0,0,1,1] (top/left 0,
// bottom/right 1), strides/dilations 1, group 1, float32, rank-1 bias, and
// the 16x8x2x2 weight shape of onnx::Conv_2844. The row must serve it via
// the zero pre-pad and match the CPU oracle.
TEST(NertCudaConv, AsymmetricPadBottomRightWithBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(1 * 8 * 3 * 3);
    std::vector<float> w(16 * 8 * 2 * 2);
    std::vector<float> b(16);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25F;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.125F;
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = static_cast<float>(i) * 0.5F - 4.0F;
    }
    nert::Node node = MakeNode("cv_asym", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{0, 0, 1, 1});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    // H = W = 3 + 0 + 1 - 2 + 1 = 3: the ring is bottom row + right column.
    const std::vector<float> expected = RunCpuConv(node, x, {1, 8, 3, 3}, w, {16, 8, 2, 2}, &b, {1, 16, 3, 3});
    ASSERT_EQ(expected.size(), 1u * 16 * 3 * 3);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(16 * 3 * 3, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 8, 3, 3}), DeviceFloat(dw, {16, 8, 2, 2}), DeviceFloat(db, {16})},
        {DeviceFloat(dy, {1, 16, 3, 3})});
    gpu = Download(cuda, dy, 16 * 3 * 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// auto_pad=SAME_UPPER with an odd pad total resolves asymmetric (here
// begin 0 / end 1 per dim) — the same path the stem geometry takes when a
// model pads automatically. Compared against the CPU oracle.
TEST(NertCudaConv, SameUpperAsymmetricPad) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(1 * 2 * 3 * 3);
    std::vector<float> w(2 * 2 * 2 * 2);
    std::vector<float> b = {1.5F, -2.5F};
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.5F;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    }
    nert::Node node = MakeNode("cv_same", "Conv");
    node.attributes.emplace_back("auto_pad", std::string{"SAME_UPPER"});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    // out = ceil(3/1) = 3, total = (3-1)*1 + 2 - 3 = 1 -> begin 0, end 1.
    const std::vector<float> expected = RunCpuConv(node, x, {1, 2, 3, 3}, w, {2, 2, 2, 2}, &b, {1, 2, 3, 3});
    ASSERT_EQ(expected.size(), 1u * 2 * 3 * 3);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(2 * 3 * 3, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 2, 3, 3}), DeviceFloat(dw, {2, 2, 2, 2}), DeviceFloat(db, {2})},
        {DeviceFloat(dy, {1, 2, 3, 3})});
    gpu = Download(cuda, dy, 2 * 3 * 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// Begin-pad asymmetry: pads=[1,0,0,2] (top/left 1,0; bottom/right 0,2)
// resolves pad_begin=[1,0], pad_end=[0,2], so the input lands at spatial
// offset (1,0) inside the zero pre-pad. Both dims are asymmetric and the
// begin pad is non-zero in dim 0 — unlike the two cases above, whose
// begin pads are [0,0] — so a row that ignored pad_begin, or that only
// served a bottom/right ring, would shift the windows and miss the CPU
// oracle. The pre-remedy row rejected any pS[i] != eS[i] outright, so this
// geometry threw there and cannot pass without the pre-pad.
TEST(NertCudaConv, AsymmetricBeginPadMixedWithBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(1 * 2 * 3 * 3);
    std::vector<float> w(2 * 2 * 2 * 2);
    std::vector<float> b = {0.75F, -1.25F};
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.5F;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    }
    nert::Node node = MakeNode("cv_beginpad", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 0, 0, 2});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    // pad_begin = [1, 0], pad_end = [0, 2] -> pre-padded input 4x5.
    // H = 3 + 1 + 0 - 2 + 1 = 3; W = 3 + 0 + 2 - 2 + 1 = 4.
    const std::vector<float> expected = RunCpuConv(node, x, {1, 2, 3, 3}, w, {2, 2, 2, 2}, &b, {1, 2, 3, 4});
    ASSERT_EQ(expected.size(), 1u * 2 * 3 * 4);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(2 * 3 * 4, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 2, 3, 3}), DeviceFloat(dw, {2, 2, 2, 2}), DeviceFloat(db, {2})},
        {DeviceFloat(dy, {1, 2, 3, 4})});
    gpu = Download(cuda, dy, 2 * 3 * 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// Width-axis begin-pad asymmetry: pads=[0,1,0,0] (top/left 0,1;
// bottom/right 0,0) resolves pad_begin=[0,1], pad_end=[0,0], so the input
// lands at spatial offset (0,1) inside the 3x4 zero pre-pad. The three
// asymmetric cases above resolve pad_begin=[0,0], [0,0], [1,0]: only dim 0
// is offset there, so srcOff[1]/o3 (the inner width offset) is never
// exercised. Here it is 1. A regression confined to the width axis would
// leave the input in columns 0..2 of the same 3x4 temporary and give
// y[0,0,0,0] = 10.5 (and y[0,1,0,0] = 2.125) where the oracle has
// 0.75 and -1.25; ExpectNear fails on that shift.
TEST(NertCudaConv, AsymmetricWidthBeginPadWithBias) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(1 * 2 * 3 * 3);
    std::vector<float> w(2 * 2 * 2 * 2);
    std::vector<float> b = {0.75F, -1.25F};
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.5F;
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    }
    nert::Node node = MakeNode("cv_width_beginpad", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{0, 1, 0, 0});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    // pad_begin = [0, 1], pad_end = [0, 0] -> pre-padded input 3x4.
    // H = 3 + 0 + 0 - 2 + 1 = 2; W = 3 + 1 + 0 - 2 + 1 = 3.
    const std::vector<float> expected = RunCpuConv(node, x, {1, 2, 3, 3}, w, {2, 2, 2, 2}, &b, {1, 2, 2, 3});
    ASSERT_EQ(expected.size(), 1u * 2 * 2 * 3);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, b);
    std::vector<float> gpu(2 * 2 * 3, 0.0F);
    void* dy = Upload(cuda, gpu);
    nert::CudaDevice().kernels().find("Conv")(
        node, {DeviceFloat(dx, {1, 2, 3, 3}), DeviceFloat(dw, {2, 2, 2, 2}), DeviceFloat(db, {2})},
        {DeviceFloat(dy, {1, 2, 2, 3})});
    gpu = Download(cuda, dy, 2 * 2 * 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpu, expected);
}

// pads=[-1,0,0,1] resolves pad_begin=[-1,0], pad_end=[0,1]: the dims are
// asymmetric (prePad true) and a negative begin pad would make the pre-pad
// copy offset negative, writing before the temporary. The row must reject
// it with InferenceException before allocating, with this exact reason.
TEST(NertCudaConv, RejectsNegativePadding) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(1 * 1 * 3 * 3, 1.0F);
    std::vector<float> w(1 * 1 * 2 * 2, 1.0F);
    std::vector<float> y(1 * 1 * 1 * 3, 0.0F);
    nert::Node node = MakeNode("cv_negpad", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{-1, 0, 0, 1});
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* dy = Upload(cuda, y);
    std::string message;
    auto run = [&] {
        try {
            nert::CudaDevice().kernels().find("Conv")(
                node, {DeviceFloat(dx, {1, 1, 3, 3}), DeviceFloat(dw, {1, 1, 2, 2})}, {DeviceFloat(dy, {1, 1, 1, 3})});
        } catch (const nert::InferenceException& e) {
            message = e.what();
            throw;
        }
    };
    EXPECT_THROW(run(), nert::InferenceException);
    EXPECT_NE(message.find("negative padding is not supported on CUDA here"), std::string::npos) << message;
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(dy);
}

// pads=[1,0,0,-1] resolves pad_begin=[1,0], pad_end=[0,-1]: the begin pad
// makes the dims asymmetric (prePad true) and the negative end pad shrinks
// the destination width while the copy still lands at offset pad_begin.
// RejectsNegativePadding above makes only pad_begin[0] negative; this case
// exercises the pad_end[i] < 0 half of the guard. InferenceException must
// carry the node name and the same named reason, before any allocation.
TEST(NertCudaConv, RejectsNegativeEndPadding) {
    if (!HasCudaConv()) {
        GTEST_SKIP() << "no CUDA Conv (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(1 * 1 * 3 * 3, 1.0F);
    std::vector<float> w(1 * 1 * 2 * 2, 1.0F);
    std::vector<float> y(1 * 1 * 3 * 1, 0.0F);
    nert::Node node = MakeNode("cv_negpad_end", "Conv");
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 0, 0, -1});
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* dy = Upload(cuda, y);
    std::string message;
    auto run = [&] {
        try {
            nert::CudaDevice().kernels().find("Conv")(
                node, {DeviceFloat(dx, {1, 1, 3, 3}), DeviceFloat(dw, {1, 1, 2, 2})}, {DeviceFloat(dy, {1, 1, 3, 1})});
        } catch (const nert::InferenceException& e) {
            message = e.what();
            throw;
        }
    };
    EXPECT_THROW(run(), nert::InferenceException);
    EXPECT_NE(message.find("cv_negpad_end"), std::string::npos) << message;
    EXPECT_EQ(message, "Conv (node 'cv_negpad_end'): negative padding is not supported on CUDA here");
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(dy);
}

bool HasCudaConvTranspose() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        return false;
    }
#ifdef NERT_HAS_CUDNN
    return nert::CudaDevice().kernels().find("ConvTranspose") != nullptr;
#else
    return false;
#endif
}

// 2x2 stride-2 no bias, W = I-ish {1,0,0,1}: hand-computed scatter.
TEST(NertCudaConv, ConvTransposeStride2NoBias) {
    if (!HasCudaConvTranspose()) {
        GTEST_SKIP() << "no CUDA ConvTranspose (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> w = {1.0F, 0.0F, 0.0F, 1.0F};
    nert::Node node = MakeNode("ct0", "ConvTranspose");
    node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    const std::vector<float> expected = {1.0F, 0.0F, 2.0F, 0.0F, 0.0F, 1.0F, 0.0F, 2.0F,
                                         3.0F, 0.0F, 4.0F, 0.0F, 0.0F, 3.0F, 0.0F, 4.0F};
    std::vector<float> cpuOut(16, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(node, {HostFloat(x, {1, 1, 2, 2}), HostFloat(w, {1, 1, 2, 2})},
                                                      {HostFloat(cpuOut, {1, 1, 4, 4})});
    ExpectNear(cpuOut, expected);

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    std::vector<float> gpuOut(16, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ConvTranspose")(
        node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dw, {1, 1, 2, 2})}, {DeviceFloat(dy, {1, 1, 4, 4})});
    gpuOut = Download(cuda, dy, 16);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, expected);
}

// 3x3 stride-1 pad-1 with bias vs the oracle.
TEST(NertCudaConv, ConvTransposePaddedWithBias) {
    if (!HasCudaConvTranspose()) {
        GTEST_SKIP() << "no CUDA ConvTranspose (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> w = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    std::vector<float> bias = {10.0F};
    nert::Node node = MakeNode("ct1", "ConvTranspose");
    node.attributes.emplace_back("pads", std::vector<int64_t>{1, 1, 1, 1});
    node.inputs = {"x", "w", "b"};
    node.outputs = {"y"};
    std::vector<float> cpuOut(4, 0.0F);
    nert::CpuDevice().kernels().find("ConvTranspose")(
        node, {HostFloat(x, {1, 1, 2, 2}), HostFloat(w, {1, 1, 3, 3}), HostFloat(bias, {1})},
        {HostFloat(cpuOut, {1, 1, 2, 2})});
    // Corner (0,0): x00*w11+x01*w10+x10*w01+x11*w00 + 10 = 1+2+3+4+10.
    ExpectNear(cpuOut, (std::vector<float>{20.0F, 20.0F, 20.0F, 20.0F}));

    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* db = Upload(cuda, bias);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ConvTranspose")(
        node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dw, {1, 1, 3, 3}), DeviceFloat(db, {1})},
        {DeviceFloat(dy, {1, 1, 2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// ConvTranspose rejects non-zero output_padding on CUDA, naming node + op.
TEST(NertCudaConv, ConvTransposeRejectsOutputPadding) {
    if (!HasCudaConvTranspose()) {
        GTEST_SKIP() << "no CUDA ConvTranspose (device or cuDNN build)";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(4, 1.0F);
    std::vector<float> w(4, 1.0F);
    std::vector<float> y(9, 0.0F);
    nert::Node node = MakeNode("ct_bad", "ConvTranspose");
    node.attributes.emplace_back("output_padding", std::vector<int64_t>{1, 0});
    node.inputs = {"x", "w"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dw = Upload(cuda, w);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ConvTranspose")(
            node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dw, {1, 1, 2, 2})}, {DeviceFloat(dy, {1, 1, 3, 3})});
        FAIL() << "expected InferenceException for non-zero output_padding";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ct_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ConvTranspose"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dw);
    cuda.allocator().release(dy);
}
