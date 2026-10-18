// CUDA concat family (N1g-2): Concat/Split/Slice vs the CPU oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Concat axis 1: {2,2} + {2,3} -> {2,5}, row blocks join on both tables.
TEST(NertCudaConcat, AxisOneExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> b = {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
    std::vector<float> cpuOut(10, 0.0F);
    nert::Node node = MakeNode("cc0", "Concat");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Concat")(node, {HostFloat(a, {2, 2}), HostFloat(b, {2, 3})},
                                               {HostFloat(cpuOut, {2, 5})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 2.0F, 5.0F, 6.0F, 7.0F, 3.0F, 4.0F, 8.0F, 9.0F, 10.0F}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    std::vector<float> gpuOut(10, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Concat")(node, {DeviceFloat(da, {2, 2}), DeviceFloat(db, {2, 3})},
                                                {DeviceFloat(dy, {2, 5})});
    gpuOut = Download(cuda, dy, 10);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Concat negative axis int64: {2,1} + {2,2} -> {2,3}.
TEST(NertCudaConcat, NegativeAxisInt64) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {1, 2};
    std::vector<int64_t> b = {3, 4, 5, 6};
    std::vector<int64_t> cpuOut(6, 0);
    nert::Node node = MakeNode("cc1", "Concat");
    node.attributes.emplace_back("axis", int64_t{-1});
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Concat")(node, {HostInt64(a, {2, 1}), HostInt64(b, {2, 2})},
                                               {HostInt64(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{1, 3, 4, 2, 5, 6}));

    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    std::vector<int64_t> gpuOut(6, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Concat")(node, {DeviceInt64(da, {2, 1}), DeviceInt64(db, {2, 2})},
                                                {DeviceInt64(dy, {2, 3})});
    gpuOut = DownloadInt64(cuda, dy, 6);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Split equal thirds: {2,6} -> 3x {2,2} along axis 1.
TEST(NertCudaConcat, SplitEqualExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    nert::Node node = MakeNode("sp0", "Split");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"x"};
    node.outputs = {"y0", "y1", "y2"};
    std::vector<float> c0(4, 0.0F), c1(4, 0.0F), c2(4, 0.0F);
    nert::CpuDevice().kernels().find("Split")(node, {HostFloat(x, {2, 6})},
                                              {HostFloat(c0, {2, 2}), HostFloat(c1, {2, 2}), HostFloat(c2, {2, 2})});
    EXPECT_EQ(c0, (std::vector<float>{1.0F, 2.0F, 7.0F, 8.0F}));
    EXPECT_EQ(c2, (std::vector<float>{5.0F, 6.0F, 11.0F, 12.0F}));

    void* dx = Upload(cuda, x);
    std::vector<float> g0(4, 0.0F), g1(4, 0.0F), g2(4, 0.0F);
    void* dy0 = Upload(cuda, g0);
    void* dy1 = Upload(cuda, g1);
    void* dy2 = Upload(cuda, g2);
    nert::CudaDevice().kernels().find("Split")(
        node, {DeviceFloat(dx, {2, 6})},
        {DeviceFloat(dy0, {2, 2}), DeviceFloat(dy1, {2, 2}), DeviceFloat(dy2, {2, 2})});
    g0 = Download(cuda, dy0, 4);
    g1 = Download(cuda, dy1, 4);
    g2 = Download(cuda, dy2, 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy0);
    cuda.allocator().release(dy1);
    cuda.allocator().release(dy2);
    EXPECT_EQ(g0, c0);
    EXPECT_EQ(g1, c1);
    EXPECT_EQ(g2, c2);
}

// Split with sizes operand {1,3,1} over {2,5}: uneven parts match.
TEST(NertCudaConcat, SplitSizesExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
    std::vector<int64_t> sizes = {1, 3, 1};
    nert::Node node = MakeNode("sp1", "Split");
    node.attributes.emplace_back("axis", int64_t{1});
    node.inputs = {"x", "sizes"};
    node.outputs = {"y0", "y1", "y2"};
    std::vector<float> c0(2, 0.0F), c1(6, 0.0F), c2(2, 0.0F);
    nert::CpuDevice().kernels().find("Split")(node, {HostFloat(x, {2, 5}), HostInt64(sizes, {3})},
                                              {HostFloat(c0, {2, 1}), HostFloat(c1, {2, 3}), HostFloat(c2, {2, 1})});
    EXPECT_EQ(c1, (std::vector<float>{2.0F, 3.0F, 4.0F, 7.0F, 8.0F, 9.0F}));

    void* dx = Upload(cuda, x);
    void* ds = UploadInt64(cuda, sizes);
    std::vector<float> g0(2, 0.0F), g1(6, 0.0F), g2(2, 0.0F);
    void* dy0 = Upload(cuda, g0);
    void* dy1 = Upload(cuda, g1);
    void* dy2 = Upload(cuda, g2);
    nert::CudaDevice().kernels().find("Split")(
        node, {DeviceFloat(dx, {2, 5}), DeviceInt64(ds, {3})},
        {DeviceFloat(dy0, {2, 1}), DeviceFloat(dy1, {2, 3}), DeviceFloat(dy2, {2, 1})});
    g0 = Download(cuda, dy0, 2);
    g1 = Download(cuda, dy1, 6);
    g2 = Download(cuda, dy2, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy0);
    cuda.allocator().release(dy1);
    cuda.allocator().release(dy2);
    EXPECT_EQ(g0, c0);
    EXPECT_EQ(g1, c1);
    EXPECT_EQ(g2, c2);
}

// Slice {3,4} rows 1..3 cols 1..3 -> {2,2} [[5,6],[9,10]].
TEST(NertCudaConcat, SliceBasicExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
    std::vector<int64_t> starts = {1, 1};
    std::vector<int64_t> ends = {3, 3};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("sl0", "Slice");
    node.inputs = {"x", "starts", "ends"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Slice")(
        node, {HostFloat(x, {3, 4}), HostInt64(starts, {2}), HostInt64(ends, {2})}, {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{5.0F, 6.0F, 9.0F, 10.0F}));

    void* dx = Upload(cuda, x);
    void* dst = UploadInt64(cuda, starts);
    void* den = UploadInt64(cuda, ends);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Slice")(
        node, {DeviceFloat(dx, {3, 4}), DeviceInt64(dst, {2}), DeviceInt64(den, {2})}, {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(dst);
    cuda.allocator().release(den);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Slice with step 2 over {0..5} -> {0,2,4}, exercising axes+steps operands.
TEST(NertCudaConcat, SliceSteppedExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    std::vector<int64_t> starts = {0};
    std::vector<int64_t> ends = {6};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {2};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("sl1", "Slice");
    node.inputs = {"x", "starts", "ends", "axes", "steps"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Slice")(
        node,
        {HostFloat(x, {6}), HostInt64(starts, {1}), HostInt64(ends, {1}), HostInt64(axes, {1}), HostInt64(steps, {1})},
        {HostFloat(cpuOut, {3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{0.0F, 2.0F, 4.0F}));

    void* dx = Upload(cuda, x);
    void* dst = UploadInt64(cuda, starts);
    void* den = UploadInt64(cuda, ends);
    void* dax = UploadInt64(cuda, axes);
    void* dstp = UploadInt64(cuda, steps);
    std::vector<float> gpuOut(3, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Slice")(node,
                                               {DeviceFloat(dx, {6}), DeviceInt64(dst, {1}), DeviceInt64(den, {1}),
                                                DeviceInt64(dax, {1}), DeviceInt64(dstp, {1})},
                                               {DeviceFloat(dy, {3})});
    gpuOut = Download(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dst);
    cuda.allocator().release(den);
    cuda.allocator().release(dax);
    cuda.allocator().release(dstp);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Concat rejects a rank mismatch, naming node + op.
TEST(NertCudaConcat, ConcatRejectsRankMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a(4, 1.0F);
    std::vector<float> b(4, 2.0F);
    std::vector<float> y(8, 0.0F);
    nert::Node node = MakeNode("cc_bad", "Concat");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Concat")(node, {DeviceFloat(da, {2, 2}), DeviceFloat(db, {4})},
                                                    {DeviceFloat(dy, {8})});
        FAIL() << "expected InferenceException for a rank mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cc_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Concat"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// Split rejects sizes that do not sum to the axis dim, naming node + op.
TEST(NertCudaConcat, SplitRejectsBadSizes) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(10, 1.0F);
    std::vector<int64_t> sizes = {2, 2};
    std::vector<float> y0(4, 0.0F), y1(4, 0.0F);
    nert::Node node = MakeNode("sp_bad", "Split");
    node.inputs = {"x", "sizes"};
    node.outputs = {"y0", "y1"};
    void* dx = Upload(cuda, x);
    void* ds = UploadInt64(cuda, sizes);
    void* dy0 = Upload(cuda, y0);
    void* dy1 = Upload(cuda, y1);
    try {
        nert::CudaDevice().kernels().find("Split")(node, {DeviceFloat(dx, {2, 5}), DeviceInt64(ds, {2})},
                                                   {DeviceFloat(dy0, {2, 2}), DeviceFloat(dy1, {2, 2})});
        FAIL() << "expected InferenceException for sizes that do not sum to the axis dim";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sp_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Split"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy0);
    cuda.allocator().release(dy1);
}

// Slice rejects a zero step, naming node + op.
TEST(NertCudaConcat, SliceRejectsZeroStep) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> starts = {0};
    std::vector<int64_t> ends = {6};
    std::vector<int64_t> axes = {0};
    std::vector<int64_t> steps = {0};
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("sl_bad", "Slice");
    node.inputs = {"x", "starts", "ends", "axes", "steps"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dst = UploadInt64(cuda, starts);
    void* den = UploadInt64(cuda, ends);
    void* dax = UploadInt64(cuda, axes);
    void* dstp = UploadInt64(cuda, steps);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Slice")(node,
                                                   {DeviceFloat(dx, {6}), DeviceInt64(dst, {1}), DeviceInt64(den, {1}),
                                                    DeviceInt64(dax, {1}), DeviceInt64(dstp, {1})},
                                                   {DeviceFloat(dy, {6})});
        FAIL() << "expected InferenceException for a zero step";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sl_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Slice"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dst);
    cuda.allocator().release(den);
    cuda.allocator().release(dax);
    cuda.allocator().release(dstp);
    cuda.allocator().release(dy);
}

// Concat bool rows copies bytes unchanged on device.
TEST(NertCudaConcat, ConcatBoolRows) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<std::uint8_t> a = {1, 0};
    std::vector<std::uint8_t> b = {0, 1};
    std::vector<std::uint8_t> cpuOut(4, 0);
    nert::Node node = MakeNode("cc_b", "Concat");
    node.attributes.emplace_back("axis", int64_t{0});
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Concat")(node, {HostBool(a, {1, 2}), HostBool(b, {1, 2})},
                                               {HostBool(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 0, 0, 1}));

    void* da = UploadBool(cuda, a);
    void* db = UploadBool(cuda, b);
    std::vector<std::uint8_t> zero(4, 0);
    void* dy = UploadBool(cuda, zero);
    nert::CudaDevice().kernels().find("Concat")(node, {DeviceBool(da, {1, 2}), DeviceBool(db, {1, 2})},
                                                {DeviceBool(dy, {2, 2})});
    std::vector<std::uint8_t> gpuOut = DownloadBool(cuda, dy, 4);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Slice bool data selects bytes unchanged on device.
TEST(NertCudaConcat, SliceBoolRange) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<std::uint8_t> in = {1, 0, 1, 1};
    std::vector<int64_t> starts = {1};
    std::vector<int64_t> ends = {3};
    std::vector<std::uint8_t> cpuOut(2, 0);
    nert::Node node = MakeNode("sl_b", "Slice");
    node.inputs = {"x", "starts", "ends"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Slice")(node, {HostBool(in, {4}), HostInt64(starts, {1}), HostInt64(ends, {1})},
                                              {HostBool(cpuOut, {2})});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{0, 1}));

    void* dx = UploadBool(cuda, in);
    void* ds = UploadInt64(cuda, starts);
    void* de = UploadInt64(cuda, ends);
    std::vector<std::uint8_t> zero(2, 0);
    void* dy = UploadBool(cuda, zero);
    nert::CudaDevice().kernels().find("Slice")(node, {DeviceBool(dx, {4}), DeviceInt64(ds, {1}), DeviceInt64(de, {1})},
                                               {DeviceBool(dy, {2})});
    std::vector<std::uint8_t> gpuOut = DownloadBool(cuda, dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(ds);
    cuda.allocator().release(de);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}
