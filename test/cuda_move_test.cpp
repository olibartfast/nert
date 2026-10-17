// CUDA data movement (R6b/Tier-5 Phase E): Identity/Squeeze/Tile/Range/
// EyeLike/Pad vs the CPU oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Identity copy over float + bool, both tables.
TEST(NertCudaMove, IdentityExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, -2.0F, 3.0F};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("id0", "Identity");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Identity")(node, {HostFloat(x, {3})}, {HostFloat(cpuOut, {3})});
    EXPECT_EQ(cpuOut, x);

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(3, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Identity")(node, {DeviceFloat(dx, {3})}, {DeviceFloat(dy, {3})});
    gpuOut = Download(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Squeeze [1,3,1] -> [3]: buffer copy, both tables.
TEST(NertCudaMove, SqueezeExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("sq0", "Squeeze");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Squeeze")(node, {HostFloat(x, {1, 3, 1})}, {HostFloat(cpuOut, {3})});
    EXPECT_EQ(cpuOut, x);

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(3, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Squeeze")(node, {DeviceFloat(dx, {1, 3, 1})}, {DeviceFloat(dy, {3})});
    gpuOut = Download(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Tile [2,1] by {2,3} -> [4,3]: hand-computed, both tables.
TEST(NertCudaMove, TileExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F};
    std::vector<int64_t> reps = {2, 3};
    std::vector<float> cpuOut(12, 0.0F);
    nert::Node node = MakeNode("tile0", "Tile");
    node.inputs = {"x", "reps"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Tile")(node, {HostFloat(x, {2, 1}), HostInt64(reps, {2})},
                                             {HostFloat(cpuOut, {4, 3})});
    const std::vector<float> expected = {1.0F, 1.0F, 1.0F, 2.0F, 2.0F, 2.0F, 1.0F, 1.0F, 1.0F, 2.0F, 2.0F, 2.0F};
    EXPECT_EQ(cpuOut, expected);

    void* dx = Upload(cuda, x);
    void* dr = UploadInt64(cuda, reps);
    std::vector<float> gpuOut(12, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Tile")(node, {DeviceFloat(dx, {2, 1}), DeviceInt64(dr, {2})},
                                              {DeviceFloat(dy, {4, 3})});
    gpuOut = Download(cuda, dy, 12);
    cuda.allocator().release(dx);
    cuda.allocator().release(dr);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Range float 0..5 step 1 and int64 5..0 step -2, both tables.
TEST(NertCudaMove, RangeExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> start = {0.0F};
    std::vector<float> limit = {5.0F};
    std::vector<float> delta = {1.0F};
    std::vector<float> cpuOut(5, 0.0F);
    nert::Node node = MakeNode("range0", "Range");
    node.inputs = {"s", "l", "d"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Range")(
        node, {HostFloat(start, {1}), HostFloat(limit, {1}), HostFloat(delta, {1})}, {HostFloat(cpuOut, {5})});
    ExpectNear(cpuOut, (std::vector<float>{0.0F, 1.0F, 2.0F, 3.0F, 4.0F}));

    void* ds = Upload(cuda, start);
    void* dl = Upload(cuda, limit);
    void* dd = Upload(cuda, delta);
    std::vector<float> gpuOut(5, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Range")(node, {DeviceFloat(ds, {1}), DeviceFloat(dl, {1}), DeviceFloat(dd, {1})},
                                               {DeviceFloat(dy, {5})});
    gpuOut = Download(cuda, dy, 5);
    cuda.allocator().release(ds);
    cuda.allocator().release(dl);
    cuda.allocator().release(dd);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);

    std::vector<int64_t> is = {5};
    std::vector<int64_t> il = {0};
    std::vector<int64_t> id = {-2};
    std::vector<int64_t> cpuI(3, 0);
    nert::Node nodeI = MakeNode("range1", "Range");
    nodeI.inputs = {"s", "l", "d"};
    nodeI.outputs = {"y"};
    nert::CpuDevice().kernels().find("Range")(nodeI, {HostInt64(is, {1}), HostInt64(il, {1}), HostInt64(id, {1})},
                                              {HostInt64(cpuI, {3})});
    EXPECT_EQ(cpuI, (std::vector<int64_t>{5, 3, 1}));
    void* dis = UploadInt64(cuda, is);
    void* dil = UploadInt64(cuda, il);
    void* did = UploadInt64(cuda, id);
    std::vector<int64_t> gpuI(3, 0);
    void* diy = UploadInt64(cuda, gpuI);
    nert::CudaDevice().kernels().find("Range")(
        nodeI, {DeviceInt64(dis, {1}), DeviceInt64(dil, {1}), DeviceInt64(did, {1})}, {DeviceInt64(diy, {3})});
    gpuI = DownloadInt64(cuda, diy, 3);
    cuda.allocator().release(dis);
    cuda.allocator().release(dil);
    cuda.allocator().release(did);
    cuda.allocator().release(diy);
    EXPECT_EQ(gpuI, cpuI);
}

// EyeLike k=0 over [2,3] and k=1 over [3,3], both tables.
TEST(NertCudaMove, EyeLikeExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 0.0F);
    std::vector<float> cpuOut(6, 0.0F);
    nert::Node node = MakeNode("eye0", "EyeLike");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("EyeLike")(node, {HostFloat(x, {2, 3})}, {HostFloat(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F}));

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("EyeLike")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(cuda, dy, 6);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);

    std::vector<float> x3(9, 0.0F);
    std::vector<float> cpu3(9, 0.0F);
    nert::Node nodeK = MakeNode("eye1", "EyeLike");
    nodeK.attributes.emplace_back("k", int64_t{1});
    nodeK.inputs = {"x"};
    nodeK.outputs = {"y"};
    nert::CpuDevice().kernels().find("EyeLike")(nodeK, {HostFloat(x3, {3, 3})}, {HostFloat(cpu3, {3, 3})});
    EXPECT_EQ(cpu3, (std::vector<float>{0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F}));
    void* dx3 = Upload(cuda, x3);
    std::vector<float> gpu3(9, 0.0F);
    void* dy3 = Upload(cuda, gpu3);
    nert::CudaDevice().kernels().find("EyeLike")(nodeK, {DeviceFloat(dx3, {3, 3})}, {DeviceFloat(dy3, {3, 3})});
    gpu3 = Download(cuda, dy3, 9);
    cuda.allocator().release(dx3);
    cuda.allocator().release(dy3);
    EXPECT_EQ(gpu3, cpu3);
}

// Pad constant/reflect/edge over {1,2,3} with pads {1,1}, both tables.
TEST(NertCudaMove, PadModesExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F};
    std::vector<int64_t> pads = {1, 1};
    std::vector<float> value = {9.0F};
    auto runMode = [&](const std::string& op, const std::vector<std::string>& extraInputs,
                       const std::vector<nert::TensorView>& extraHost,
                       const std::vector<std::pair<void*, std::vector<int64_t>>>& extraDevice,
                       const std::vector<float>& expected) {
        nert::Node node = MakeNode("pad0", op);
        node.inputs = {"x", "pads"};
        for (const std::string& e : extraInputs) {
            node.inputs.push_back(e);
        }
        node.outputs = {"y"};
        std::vector<float> cpuOut(5, 0.0F);
        std::vector<nert::TensorView> hostInputs = {HostFloat(x, {3}), HostInt64(pads, {2})};
        for (const nert::TensorView& v : extraHost) {
            hostInputs.push_back(v);
        }
        nert::CpuDevice().kernels().find(op)(node, hostInputs, {HostFloat(cpuOut, {5})});
        ExpectNear(cpuOut, expected);

        void* dx = Upload(cuda, x);
        void* dp = UploadInt64(cuda, pads);
        std::vector<nert::TensorView> deviceInputs = {DeviceFloat(dx, {3}), DeviceInt64(dp, {2})};
        std::vector<void*> owned{dx, dp};
        for (const auto& [buf, dims] : extraDevice) {
            nert::TensorView v;
            v.data = buf;
            v.dtype = nert::DataType::Float32;
            v.dims = dims;
            deviceInputs.push_back(v);
        }
        std::vector<float> gpuOut(5, 0.0F);
        void* dy = Upload(cuda, gpuOut);
        owned.push_back(dy);
        nert::CudaDevice().kernels().find(op)(node, deviceInputs, {DeviceFloat(dy, {5})});
        gpuOut = Download(cuda, dy, 5);
        for (void* p : owned) {
            cuda.allocator().release(p);
        }
        ExpectNear(gpuOut, cpuOut);
    };
    void* dv = Upload(cuda, value);
    runMode("Pad", {"value"}, {HostFloat(value, {1})}, {{dv, {1}}}, {9.0F, 1.0F, 2.0F, 3.0F, 9.0F});
    cuda.allocator().release(dv);
    // reflect / edge carry the mode as an attribute instead of a value input.
    auto runAttrMode = [&](const std::string& mode, const std::vector<float>& expected) {
        std::vector<float> cpuOut(5, 0.0F);
        nert::Node node = MakeNode("pad1", "Pad");
        node.attributes.emplace_back("mode", mode);
        node.inputs = {"x", "pads"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("Pad")(node, {HostFloat(x, {3}), HostInt64(pads, {2})},
                                                {HostFloat(cpuOut, {5})});
        ExpectNear(cpuOut, expected);
        void* dx = Upload(cuda, x);
        void* dp = UploadInt64(cuda, pads);
        std::vector<float> gpuOut(5, 0.0F);
        void* dy = Upload(cuda, gpuOut);
        nert::CudaDevice().kernels().find("Pad")(node, {DeviceFloat(dx, {3}), DeviceInt64(dp, {2})},
                                                 {DeviceFloat(dy, {5})});
        gpuOut = Download(cuda, dy, 5);
        cuda.allocator().release(dx);
        cuda.allocator().release(dp);
        cuda.allocator().release(dy);
        ExpectNear(gpuOut, cpuOut);
    };
    runAttrMode("reflect", {2.0F, 1.0F, 2.0F, 3.0F, 2.0F});
    runAttrMode("edge", {1.0F, 1.0F, 2.0F, 3.0F, 3.0F});
}

// Tile rejects a bad repeats rank, naming node + op.
TEST(NertCudaMove, TileRejectsBadRepeats) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(4, 1.0F);
    std::vector<int64_t> reps = {2, 2, 2};
    std::vector<float> y(8, 0.0F);
    nert::Node node = MakeNode("tile_bad", "Tile");
    node.inputs = {"x", "reps"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dr = UploadInt64(cuda, reps);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Tile")(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(dr, {3})},
                                                  {DeviceFloat(dy, {2, 4})});
        FAIL() << "expected InferenceException for a bad repeats rank";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("tile_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Tile"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dr);
    cuda.allocator().release(dy);
}

// Range rejects a zero delta, naming node + op.
TEST(NertCudaMove, RangeRejectsZeroDelta) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> s = {0};
    std::vector<int64_t> l = {5};
    std::vector<int64_t> d = {0};
    std::vector<int64_t> y(5, 0);
    nert::Node node = MakeNode("range_bad", "Range");
    node.inputs = {"s", "l", "d"};
    node.outputs = {"y"};
    void* ds = UploadInt64(cuda, s);
    void* dl = UploadInt64(cuda, l);
    void* dd = UploadInt64(cuda, d);
    void* dy = UploadInt64(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Range")(
            node, {DeviceInt64(ds, {1}), DeviceInt64(dl, {1}), DeviceInt64(dd, {1})}, {DeviceInt64(dy, {5})});
        FAIL() << "expected InferenceException for a zero delta";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("range_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Range"), std::string::npos) << message;
    }
    cuda.allocator().release(ds);
    cuda.allocator().release(dl);
    cuda.allocator().release(dd);
    cuda.allocator().release(dy);
}

// Pad rejects an unsupported mode, naming node + op.
TEST(NertCudaMove, PadRejectsBadMode) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(3, 1.0F);
    std::vector<int64_t> pads = {1, 1};
    std::vector<float> y(5, 0.0F);
    nert::Node node = MakeNode("pad_bad", "Pad");
    node.attributes.emplace_back("mode", std::string("wrap"));
    node.inputs = {"x", "pads"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dp = UploadInt64(cuda, pads);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Pad")(node, {DeviceFloat(dx, {3}), DeviceInt64(dp, {2})},
                                                 {DeviceFloat(dy, {5})});
        FAIL() << "expected InferenceException for an unsupported mode";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("pad_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Pad"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dp);
    cuda.allocator().release(dy);
}
