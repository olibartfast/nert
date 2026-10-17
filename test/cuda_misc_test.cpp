// CUDA misc elementwise (N1g-5a): Cast/Equal/Where/Shape/ConstantOfShape/Mod
// vs the CPU oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Cast int64 {0,-3,7} to float and to bool, both tables.
TEST(NertCudaMisc, CastIntToFloatAndBool) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x = {0, -3, 7};
    nert::Node toFloat = MakeNode("cast0", "Cast");
    toFloat.attributes.emplace_back("to", int64_t{1});
    toFloat.inputs = {"x"};
    toFloat.outputs = {"y"};
    std::vector<float> cpuFloat(3, 0.0F);
    nert::CpuDevice().kernels().find("Cast")(toFloat, {HostInt64(x, {3})}, {HostFloat(cpuFloat, {3})});
    EXPECT_EQ(cpuFloat, (std::vector<float>{0.0F, -3.0F, 7.0F}));

    void* dx = UploadInt64(cuda, x);
    std::vector<float> gpuFloat(3, 0.0F);
    void* dy = Upload(cuda, gpuFloat);
    nert::CudaDevice().kernels().find("Cast")(toFloat, {DeviceInt64(dx, {3})}, {DeviceFloat(dy, {3})});
    gpuFloat = Download(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuFloat, cpuFloat);

    nert::Node toBool = MakeNode("cast1", "Cast");
    toBool.attributes.emplace_back("to", int64_t{9});
    toBool.inputs = {"x"};
    toBool.outputs = {"y"};
    std::vector<std::uint8_t> cpuBool(3, 0);
    nert::TensorView boolOut;
    boolOut.data = cpuBool.data();
    boolOut.dtype = nert::DataType::Bool;
    boolOut.dims = {3};
    nert::CpuDevice().kernels().find("Cast")(toBool, {HostInt64(x, {3})}, {boolOut});
    EXPECT_EQ(cpuBool, (std::vector<std::uint8_t>{0, 1, 1}));

    void* dxb = UploadInt64(cuda, x);
    void* dyb = cuda.allocator().allocate(3);
    nert::TensorView deviceBoolOut;
    deviceBoolOut.data = dyb;
    deviceBoolOut.dtype = nert::DataType::Bool;
    deviceBoolOut.dims = {3};
    nert::CudaDevice().kernels().find("Cast")(toBool, {DeviceInt64(dxb, {3})}, {deviceBoolOut});
    std::vector<std::uint8_t> gpuBool(3, 0);
    cuda.transfer().device_to_host(gpuBool.data(), dyb, 3);
    cuda.allocator().release(dxb);
    cuda.allocator().release(dyb);
    EXPECT_EQ(gpuBool, cpuBool);
}

// Cast rejects an unsupported 'to', naming node + op.
TEST(NertCudaMisc, CastRejectsBadTo) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(2, 1.0F);
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("cast_bad", "Cast");
    node.attributes.emplace_back("to", int64_t{10});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Cast")(node, {DeviceFloat(dx, {2})}, {DeviceFloat(dy, {2})});
        FAIL() << "expected InferenceException for an unsupported 'to'";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cast_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Cast"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// Equal broadcasts {2,1} against {1,3} float, both tables.
TEST(NertCudaMisc, EqualBroadcastExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, 2.0F};
    std::vector<float> b = {1.0F, 9.0F, 2.0F};
    std::vector<std::uint8_t> cpuOut(6, 0);
    nert::TensorView cpuView;
    cpuView.data = cpuOut.data();
    cpuView.dtype = nert::DataType::Bool;
    cpuView.dims = {2, 3};
    nert::Node node = MakeNode("eq0", "Equal");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Equal")(node, {HostFloat(a, {2, 1}), HostFloat(b, {1, 3})}, {cpuView});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 0, 0, 0, 0, 1}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = cuda.allocator().allocate(6);
    nert::TensorView deviceView;
    deviceView.data = dy;
    deviceView.dtype = nert::DataType::Bool;
    deviceView.dims = {2, 3};
    nert::CudaDevice().kernels().find("Equal")(node, {DeviceFloat(da, {2, 1}), DeviceFloat(db, {1, 3})}, {deviceView});
    std::vector<std::uint8_t> gpuOut(6, 0);
    cuda.transfer().device_to_host(gpuOut.data(), dy, 6);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Where selects per broadcast condition: cond {2,1} over {2,3} values.
TEST(NertCudaMisc, WhereBroadcastExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<std::uint8_t> cond = {1, 0};
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> z = {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F};
    std::vector<float> cpuOut(6, 0.0F);
    nert::TensorView condView;
    condView.data = cond.data();
    condView.dtype = nert::DataType::Bool;
    condView.dims = {2, 1};
    nert::Node node = MakeNode("wh0", "Where");
    node.inputs = {"c", "x", "z"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Where")(node, {condView, HostFloat(x, {2, 3}), HostFloat(z, {2, 3})},
                                              {HostFloat(cpuOut, {2, 3})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 2.0F, 3.0F, 40.0F, 50.0F, 60.0F}));

    void* dc = cuda.allocator().allocate(2);
    cuda.transfer().host_to_device(dc, cond.data(), 2);
    nert::TensorView deviceCond;
    deviceCond.data = dc;
    deviceCond.dtype = nert::DataType::Bool;
    deviceCond.dims = {2, 1};
    void* dx = Upload(cuda, x);
    void* dz = Upload(cuda, z);
    std::vector<float> gpuOut(6, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Where")(node, {deviceCond, DeviceFloat(dx, {2, 3}), DeviceFloat(dz, {2, 3})},
                                               {DeviceFloat(dy, {2, 3})});
    gpuOut = Download(cuda, dy, 6);
    cuda.allocator().release(dc);
    cuda.allocator().release(dx);
    cuda.allocator().release(dz);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Where rejects a non-bool condition, naming node + op.
TEST(NertCudaMisc, WhereRejectsNonBoolCond) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> c = {1.0F, 0.0F};
    std::vector<float> x = {1.0F, 2.0F};
    std::vector<float> z = {3.0F, 4.0F};
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("wh_bad", "Where");
    node.inputs = {"c", "x", "z"};
    node.outputs = {"y"};
    void* dc = Upload(cuda, c);
    void* dx = Upload(cuda, x);
    void* dz = Upload(cuda, z);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Where")(
            node, {DeviceFloat(dc, {2}), DeviceFloat(dx, {2}), DeviceFloat(dz, {2})}, {DeviceFloat(dy, {2})});
        FAIL() << "expected InferenceException for a non-bool condition";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("wh_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Where"), std::string::npos) << message;
    }
    cuda.allocator().release(dc);
    cuda.allocator().release(dx);
    cuda.allocator().release(dz);
    cuda.allocator().release(dy);
}

// Shape reports dims {2,3,4} (and a start=1 slice) into device int64.
TEST(NertCudaMisc, ShapeExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(24, 1.0F);
    nert::Node node = MakeNode("sh0", "Shape");
    node.inputs = {"x"};
    node.outputs = {"y"};
    std::vector<int64_t> cpuOut(3, 0);
    nert::CpuDevice().kernels().find("Shape")(node, {HostFloat(x, {2, 3, 4})}, {HostInt64(cpuOut, {3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{2, 3, 4}));

    void* dx = Upload(cuda, x);
    std::vector<int64_t> gpuOut(3, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Shape")(node, {DeviceFloat(dx, {2, 3, 4})}, {DeviceInt64(dy, {3})});
    gpuOut = DownloadInt64(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);

    nert::Node sliced = MakeNode("sh1", "Shape");
    sliced.attributes.emplace_back("start", int64_t{1});
    sliced.inputs = {"x"};
    sliced.outputs = {"y"};
    std::vector<int64_t> cpuSliced(2, 0);
    nert::CpuDevice().kernels().find("Shape")(sliced, {HostFloat(x, {2, 3, 4})}, {HostInt64(cpuSliced, {2})});
    EXPECT_EQ(cpuSliced, (std::vector<int64_t>{3, 4}));

    void* dxs = Upload(cuda, x);
    std::vector<int64_t> gpuSliced(2, 0);
    void* dys = UploadInt64(cuda, gpuSliced);
    nert::CudaDevice().kernels().find("Shape")(sliced, {DeviceFloat(dxs, {2, 3, 4})}, {DeviceInt64(dys, {2})});
    gpuSliced = DownloadInt64(cuda, dys, 2);
    cuda.allocator().release(dxs);
    cuda.allocator().release(dys);
    EXPECT_EQ(gpuSliced, cpuSliced);
}

// ConstantOfShape fills {2,2} with the float value attr and {3} default.
TEST(NertCudaMisc, ConstantOfShapeExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> shape = {2, 2};
    nert::Node node = MakeNode("cos0", "ConstantOfShape");
    node.attributes.emplace_back("value", 7.5F);
    node.inputs = {"shape"};
    node.outputs = {"y"};
    std::vector<float> cpuOut(4, 0.0F);
    nert::CpuDevice().kernels().find("ConstantOfShape")(node, {HostInt64(shape, {2})}, {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{7.5F, 7.5F, 7.5F, 7.5F}));

    void* ds = UploadInt64(cuda, shape);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ConstantOfShape")(node, {DeviceInt64(ds, {2})}, {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);

    // Default fill (no value attr) is float zero.
    std::vector<int64_t> shape3 = {3};
    nert::Node plain = MakeNode("cos1", "ConstantOfShape");
    plain.inputs = {"shape"};
    plain.outputs = {"y"};
    std::vector<float> cpuPlain(3, 1.0F);
    nert::CpuDevice().kernels().find("ConstantOfShape")(plain, {HostInt64(shape3, {1})}, {HostFloat(cpuPlain, {3})});
    EXPECT_EQ(cpuPlain, (std::vector<float>{0.0F, 0.0F, 0.0F}));

    void* ds3 = UploadInt64(cuda, shape3);
    std::vector<float> gpuPlain(3, 1.0F);
    void* dy3 = Upload(cuda, gpuPlain);
    nert::CudaDevice().kernels().find("ConstantOfShape")(plain, {DeviceInt64(ds3, {1})}, {DeviceFloat(dy3, {3})});
    gpuPlain = Download(cuda, dy3, 3);
    cuda.allocator().release(ds3);
    cuda.allocator().release(dy3);
    EXPECT_EQ(gpuPlain, cpuPlain);
}

// ConstantOfShape rejects a negative shape value, naming node + op.
TEST(NertCudaMisc, ConstantOfShapeRejectsNegative) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> shape = {2, -1};
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("cos_bad", "ConstantOfShape");
    node.inputs = {"shape"};
    node.outputs = {"y"};
    void* ds = UploadInt64(cuda, shape);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ConstantOfShape")(node, {DeviceInt64(ds, {2})}, {DeviceFloat(dy, {2})});
        FAIL() << "expected InferenceException for a negative shape value";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cos_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ConstantOfShape"), std::string::npos) << message;
    }
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
}

// Mod over int64 with broadcast: {7,8} % {3,2} = {1,0}, both tables.
TEST(NertCudaMisc, ModBroadcastExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 2};
    std::vector<int64_t> cpuOut(2, 0);
    nert::Node node = MakeNode("mod0", "Mod");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Mod")(node, {HostInt64(a, {2}), HostInt64(b, {2})}, {HostInt64(cpuOut, {2})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{1, 0}));

    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    std::vector<int64_t> gpuOut(2, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Mod")(node, {DeviceInt64(da, {2}), DeviceInt64(db, {2})},
                                             {DeviceInt64(dy, {2})});
    gpuOut = DownloadInt64(cuda, dy, 2);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Mod reports a zero divisor, naming node + op.
TEST(NertCudaMisc, ModRejectsZeroDivisor) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {7, 8};
    std::vector<int64_t> b = {3, 0};
    std::vector<int64_t> y(2, 0);
    nert::Node node = MakeNode("mod_bad", "Mod");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    void* dy = UploadInt64(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Mod")(node, {DeviceInt64(da, {2}), DeviceInt64(db, {2})},
                                                 {DeviceInt64(dy, {2})});
        FAIL() << "expected InferenceException for a zero divisor";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("mod_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Mod"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// ReduceMax axis 1 keepdims: {2,3} -> {2,1} row maxima, both tables.
TEST(NertCudaMisc, ReduceMaxAxisExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 5.0F, 3.0F, 4.0F, 2.0F, 6.0F};
    std::vector<int64_t> axes = {1};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("rm0", "ReduceMax");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceMax")(node, {HostFloat(x, {2, 3}), HostInt64(axes, {1})},
                                                  {HostFloat(cpuOut, {2, 1})});
    EXPECT_EQ(cpuOut, (std::vector<float>{5.0F, 6.0F}));

    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceMax")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {1})},
                                                   {DeviceFloat(dy, {2, 1})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// ReduceMax over all axes without keepdims: single max, both tables.
TEST(NertCudaMisc, ReduceMaxGlobalNoKeepdims) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 5.0F, 3.0F, 4.0F, 2.0F, 6.0F};
    std::vector<float> cpuOut(1, 0.0F);
    nert::Node node = MakeNode("rm1", "ReduceMax");
    node.attributes.emplace_back("keepdims", int64_t{0});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceMax")(node, {HostFloat(x, {2, 3})}, {HostFloat(cpuOut, {})});
    EXPECT_EQ(cpuOut, (std::vector<float>{6.0F}));

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(1, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceMax")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {})});
    gpuOut = Download(cuda, dy, 1);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// ReduceMax rejects a duplicate axis, naming node + op.
TEST(NertCudaMisc, ReduceMaxRejectsDuplicateAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> axes = {1, -1};
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("rm_bad", "ReduceMax");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ReduceMax")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {2})},
                                                       {DeviceFloat(dy, {2, 1})});
        FAIL() << "expected InferenceException for a duplicate axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("rm_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ReduceMax"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
}

// ReduceSum over axis 1 of [2,3]: {1+2+3, 4+5+6}, both tables.
TEST(NertCudaMisc, ReduceSumAxisExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> axes = {1};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("rs0", "ReduceSum");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceSum")(node, {HostFloat(x, {2, 3}), HostInt64(axes, {1})},
                                                  {HostFloat(cpuOut, {2, 1})});
    EXPECT_EQ(cpuOut, (std::vector<float>{6.0F, 15.0F}));

    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceSum")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {1})},
                                                   {DeviceFloat(dy, {2, 1})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// ReduceSum int64 over all axes without keepdims: absent axes input, scalar out.
TEST(NertCudaMisc, ReduceSumGlobalNoKeepdimsInt64) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x = {1, 2, 3, 4};
    std::vector<int64_t> cpuOut(1, 0);
    nert::Node node = MakeNode("rs1", "ReduceSum");
    node.attributes.emplace_back("keepdims", int64_t{0});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceSum")(node, {HostInt64(x, {2, 2})}, {HostInt64(cpuOut, {})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{10}));

    void* dx = UploadInt64(cuda, x);
    std::vector<int64_t> gpuOut(1, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceSum")(node, {DeviceInt64(dx, {2, 2})}, {DeviceInt64(dy, {})});
    gpuOut = DownloadInt64(cuda, dy, 1);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// ReduceSum with noop_with_empty_axes and empty axes: identity copy, both tables.
TEST(NertCudaMisc, ReduceSumNoopIdentity) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> axes;
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("rs2", "ReduceSum");
    node.attributes.emplace_back("noop_with_empty_axes", int64_t{1});
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceSum")(node, {HostFloat(x, {2, 2}), HostInt64(axes, {0})},
                                                  {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, x);

    void* dx = Upload(cuda, x);
    // Empty axes: no buffer to upload; the kernel must not touch it.
    void* da = nullptr;
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceSum")(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(da, {0})},
                                                   {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// ReduceSum rejects a duplicate axis, naming node + op.
TEST(NertCudaMisc, ReduceSumRejectsDuplicateAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> axes = {1, -1};
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("rs_bad", "ReduceSum");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ReduceSum")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {2})},
                                                       {DeviceFloat(dy, {2, 1})});
        FAIL() << "expected InferenceException for a duplicate axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("rs_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ReduceSum"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
}

// ReduceSum rejects an out-of-range axis, naming node + op.
TEST(NertCudaMisc, ReduceSumRejectsBadAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> axes = {5};
    std::vector<float> y(6, 0.0F);
    nert::Node node = MakeNode("rs_bad_axis", "ReduceSum");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ReduceSum")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {1})},
                                                       {DeviceFloat(dy, {2, 3})});
        FAIL() << "expected InferenceException for an out-of-range axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("rs_bad_axis"), std::string::npos) << message;
        EXPECT_NE(message.find("ReduceSum"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
}

// ReduceSum rejects a non-int64 axes tensor, naming node + op.
TEST(NertCudaMisc, ReduceSumRejectsNonInt64Axes) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<float> axes = {1.0F};
    std::vector<float> y(2, 0.0F);
    nert::Node node = MakeNode("rs_bad_axes", "ReduceSum");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* da = Upload(cuda, axes);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ReduceSum")(node, {DeviceFloat(dx, {2, 3}), DeviceFloat(da, {1})},
                                                       {DeviceFloat(dy, {2, 1})});
        FAIL() << "expected InferenceException for a non-int64 axes tensor";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("rs_bad_axes"), std::string::npos) << message;
        EXPECT_NE(message.find("ReduceSum"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
}
TEST(NertCudaMisc, TopKExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 5.0F, 3.0F, 4.0F, 7.0F, 2.0F, 8.0F, 6.0F};
    std::vector<int64_t> k = {2};
    std::vector<float> cpuVals(4, 0.0F);
    std::vector<int64_t> cpuIdx(4, 0);
    nert::Node node = MakeNode("tk0", "TopK");
    node.inputs = {"x", "k"};
    node.outputs = {"values", "indices"};
    nert::CpuDevice().kernels().find("TopK")(node, {HostFloat(x, {2, 4}), HostInt64(k, {})},
                                             {HostFloat(cpuVals, {2, 2}), HostInt64(cpuIdx, {2, 2})});
    EXPECT_EQ(cpuVals, (std::vector<float>{5.0F, 4.0F, 8.0F, 7.0F}));
    EXPECT_EQ(cpuIdx, (std::vector<int64_t>{1, 3, 2, 0}));

    void* dx = Upload(cuda, x);
    void* dk = UploadInt64(cuda, k);
    std::vector<float> gpuVals(4, 0.0F);
    std::vector<int64_t> gpuIdx(4, 0);
    void* dyv = Upload(cuda, gpuVals);
    void* dyi = UploadInt64(cuda, gpuIdx);
    nert::CudaDevice().kernels().find("TopK")(node, {DeviceFloat(dx, {2, 4}), DeviceInt64(dk, {})},
                                              {DeviceFloat(dyv, {2, 2}), DeviceInt64(dyi, {2, 2})});
    gpuVals = Download(cuda, dyv, 4);
    gpuIdx = DownloadInt64(cuda, dyi, 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(dk);
    cuda.allocator().release(dyv);
    cuda.allocator().release(dyi);
    EXPECT_EQ(gpuVals, cpuVals);
    EXPECT_EQ(gpuIdx, cpuIdx);
}

// TopK rejects K exceeding the axis dim, naming node + op.
TEST(NertCudaMisc, TopKRejectsBigK) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F};
    std::vector<int64_t> k = {5};
    std::vector<float> v(5, 0.0F);
    std::vector<int64_t> idx(5, 0);
    nert::Node node = MakeNode("tk_bad", "TopK");
    node.inputs = {"x", "k"};
    node.outputs = {"values", "indices"};
    void* dx = Upload(cuda, x);
    void* dk = UploadInt64(cuda, k);
    void* dyv = Upload(cuda, v);
    void* dyi = UploadInt64(cuda, idx);
    try {
        nert::CudaDevice().kernels().find("TopK")(node, {DeviceFloat(dx, {2}), DeviceInt64(dk, {})},
                                                  {DeviceFloat(dyv, {5}), DeviceInt64(dyi, {5})});
        FAIL() << "expected InferenceException for K exceeding the axis dim";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("tk_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("TopK"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dk);
    cuda.allocator().release(dyv);
    cuda.allocator().release(dyi);
}

// Resize nearest 2x over {1,1,2,2} with scales, both tables.
TEST(NertCudaMisc, ResizeNearestExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> roi(8, 0.0F);
    std::vector<float> scales = {1.0F, 1.0F, 2.0F, 2.0F};
    std::vector<float> cpuOut(16, 0.0F);
    nert::Node node = MakeNode("rz0", "Resize");
    node.inputs = {"x", "roi", "scales"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Resize")(
        node, {HostFloat(x, {1, 1, 2, 2}), HostFloat(roi, {8}), HostFloat(scales, {4})},
        {HostFloat(cpuOut, {1, 1, 4, 4})});
    const std::vector<float> expected = {1.0F, 1.0F, 2.0F, 2.0F, 1.0F, 1.0F, 2.0F, 2.0F,
                                         3.0F, 3.0F, 4.0F, 4.0F, 3.0F, 3.0F, 4.0F, 4.0F};
    EXPECT_EQ(cpuOut, expected);

    void* dx = Upload(cuda, x);
    void* dr = Upload(cuda, roi);
    void* ds = Upload(cuda, scales);
    std::vector<float> gpuOut(16, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Resize")(
        node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dr, {8}), DeviceFloat(ds, {4})},
        {DeviceFloat(dy, {1, 1, 4, 4})});
    gpuOut = Download(cuda, dy, 16);
    cuda.allocator().release(dx);
    cuda.allocator().release(dr);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, expected);
}

// Resize linear 2x over {1,1,1,4} asymmetric: [0,5,10,15,20,25,30,30].
TEST(NertCudaMisc, ResizeLinearExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {0.0F, 10.0F, 20.0F, 30.0F};
    std::vector<float> roi(8, 0.0F);
    std::vector<float> scales = {1.0F, 1.0F, 1.0F, 2.0F};
    std::vector<float> cpuOut(8, 0.0F);
    nert::Node node = MakeNode("rz1", "Resize");
    node.attributes.emplace_back("mode", std::string{"linear"});
    node.attributes.emplace_back("coordinate_transformation_mode", std::string{"asymmetric"});
    node.inputs = {"x", "roi", "scales"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Resize")(
        node, {HostFloat(x, {1, 1, 1, 4}), HostFloat(roi, {8}), HostFloat(scales, {4})},
        {HostFloat(cpuOut, {1, 1, 1, 8})});
    const std::vector<float> expected = {0.0F, 5.0F, 10.0F, 15.0F, 20.0F, 25.0F, 30.0F, 30.0F};
    ExpectNear(cpuOut, expected);

    void* dx = Upload(cuda, x);
    void* dr = Upload(cuda, roi);
    void* ds = Upload(cuda, scales);
    std::vector<float> gpuOut(8, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Resize")(
        node, {DeviceFloat(dx, {1, 1, 1, 4}), DeviceFloat(dr, {8}), DeviceFloat(ds, {4})},
        {DeviceFloat(dy, {1, 1, 1, 8})});
    gpuOut = Download(cuda, dy, 8);
    cuda.allocator().release(dx);
    cuda.allocator().release(dr);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, expected);
}

// Resize rejects an unsupported mode, naming node + op.
TEST(NertCudaMisc, ResizeRejectsBadMode) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(4, 1.0F);
    std::vector<float> roi(8, 0.0F);
    std::vector<float> scales = {1.0F, 1.0F, 2.0F, 2.0F};
    std::vector<float> y(16, 0.0F);
    nert::Node node = MakeNode("rz_bad", "Resize");
    node.attributes.emplace_back("mode", std::string{"cubic"});
    node.inputs = {"x", "roi", "scales"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dr = Upload(cuda, roi);
    void* ds = Upload(cuda, scales);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Resize")(
            node, {DeviceFloat(dx, {1, 1, 2, 2}), DeviceFloat(dr, {8}), DeviceFloat(ds, {4})},
            {DeviceFloat(dy, {1, 1, 4, 4})});
        FAIL() << "expected InferenceException for an unsupported mode";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("rz_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Resize"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dr);
    cuda.allocator().release(ds);
    cuda.allocator().release(dy);
}
