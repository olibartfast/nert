// CUDA misc elementwise (N1g-5a): Cast/Equal/Where/Shape/ConstantOfShape/Mod
// vs the CPU oracle.
//
// Needs a CUDA device; without one every test SKIPs (never fails).

#include "cuda_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
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

// A declared-13 ReduceMax carries axes in an attribute and no axes input. The
// CUDA row reads that form in place and matches the CPU oracle.
TEST(NertCudaMisc, ReduceMaxOldAttributeFormMatchesCpu) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 5.0F, 3.0F, 4.0F, 2.0F, 6.0F};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("rm_attr", "ReduceMax");
    node.schema_version = 13;
    node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceMax")(node, {HostFloat(x, {2, 3})}, {HostFloat(cpuOut, {2, 1})});
    EXPECT_EQ(cpuOut, (std::vector<float>{5.0F, 6.0F}));

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceMax")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {2, 1})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(dx);
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

// Greater float broadcast: {1,2,3} > 2, both tables.
TEST(NertCudaMisc, GreaterBroadcastExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F};
    std::vector<float> b = {2.0F};
    std::vector<std::uint8_t> cpuOut(3, 0);
    nert::Node node = MakeNode("gt0", "Greater");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Bool;
    cpuY.dims = {3};
    nert::CpuDevice().kernels().find("Greater")(node, {HostFloat(a, {3}), HostFloat(b, {1})}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{0, 0, 1}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = cuda.allocator().allocate(3);
    nert::TensorView deviceY;
    deviceY.data = dy;
    deviceY.dtype = nert::DataType::Bool;
    deviceY.dims = {3};
    nert::CudaDevice().kernels().find("Greater")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {1})}, {deviceY});
    std::vector<std::uint8_t> gpuOut(3, 0);
    cuda.transfer().device_to_host(gpuOut.data(), dy, 3);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// LessOrEqual int64: {1,2,3} <= {1,3,2}, both tables.
TEST(NertCudaMisc, LessOrEqualInt64Exact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {1, 2, 3};
    std::vector<int64_t> b = {1, 3, 2};
    std::vector<std::uint8_t> cpuOut(3, 0);
    nert::Node node = MakeNode("le0", "LessOrEqual");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Bool;
    cpuY.dims = {3};
    nert::CpuDevice().kernels().find("LessOrEqual")(node, {HostInt64(a, {3}), HostInt64(b, {3})}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 1, 0}));

    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    void* dy = cuda.allocator().allocate(3);
    nert::TensorView deviceY;
    deviceY.data = dy;
    deviceY.dtype = nert::DataType::Bool;
    deviceY.dims = {3};
    nert::CudaDevice().kernels().find("LessOrEqual")(node, {DeviceInt64(da, {3}), DeviceInt64(db, {3})}, {deviceY});
    std::vector<std::uint8_t> gpuOut(3, 0);
    cuda.transfer().device_to_host(gpuOut.data(), dy, 3);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// And/Or truth tables with broadcast, both tables.
TEST(NertCudaMisc, AndOrBroadcastExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<std::uint8_t> a = {0, 0, 1, 1};
    std::vector<std::uint8_t> b = {1};
    auto runBoth = [&](const std::string& op, const std::vector<std::uint8_t>& expected) {
        nert::Node node = MakeNode("logic0", op);
        node.inputs = {"a", "b"};
        node.outputs = {"y"};
        nert::TensorView av;
        av.data = const_cast<std::uint8_t*>(a.data());
        av.dtype = nert::DataType::Bool;
        av.dims = {2, 2};
        nert::TensorView bv;
        bv.data = const_cast<std::uint8_t*>(b.data());
        bv.dtype = nert::DataType::Bool;
        bv.dims = {1};
        std::vector<std::uint8_t> cpuOut(4, 0);
        nert::TensorView cpuY;
        cpuY.data = cpuOut.data();
        cpuY.dtype = nert::DataType::Bool;
        cpuY.dims = {2, 2};
        nert::CpuDevice().kernels().find(op)(node, {av, bv}, {cpuY});
        EXPECT_EQ(cpuOut, expected);

        void* da = cuda.allocator().allocate(4);
        cuda.transfer().host_to_device(da, a.data(), 4);
        void* db = cuda.allocator().allocate(1);
        cuda.transfer().host_to_device(db, b.data(), 1);
        void* dy = cuda.allocator().allocate(4);
        nert::TensorView dav;
        dav.data = da;
        dav.dtype = nert::DataType::Bool;
        dav.dims = {2, 2};
        nert::TensorView dbv;
        dbv.data = db;
        dbv.dtype = nert::DataType::Bool;
        dbv.dims = {1};
        nert::TensorView deviceY;
        deviceY.data = dy;
        deviceY.dtype = nert::DataType::Bool;
        deviceY.dims = {2, 2};
        nert::CudaDevice().kernels().find(op)(node, {dav, dbv}, {deviceY});
        std::vector<std::uint8_t> gpuOut(4, 0);
        cuda.transfer().device_to_host(gpuOut.data(), dy, 4);
        cuda.allocator().release(da);
        cuda.allocator().release(db);
        cuda.allocator().release(dy);
        EXPECT_EQ(gpuOut, cpuOut);
    };
    runBoth("And", {0, 0, 1, 1});
    runBoth("Or", {1, 1, 1, 1});
}

// Not over {1, 0}: hand-computed, both tables.
TEST(NertCudaMisc, NotExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<std::uint8_t> x = {1, 0};
    std::vector<std::uint8_t> cpuOut(2, 0);
    nert::Node node = MakeNode("not0", "Not");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::TensorView xv;
    xv.data = x.data();
    xv.dtype = nert::DataType::Bool;
    xv.dims = {2};
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Bool;
    cpuY.dims = {2};
    nert::CpuDevice().kernels().find("Not")(node, {xv}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{0, 1}));

    void* dx = cuda.allocator().allocate(2);
    cuda.transfer().host_to_device(dx, x.data(), 2);
    void* dy = cuda.allocator().allocate(2);
    nert::TensorView dxv;
    dxv.data = dx;
    dxv.dtype = nert::DataType::Bool;
    dxv.dims = {2};
    nert::TensorView deviceY;
    deviceY.data = dy;
    deviceY.dtype = nert::DataType::Bool;
    deviceY.dims = {2};
    nert::CudaDevice().kernels().find("Not")(node, {dxv}, {deviceY});
    std::vector<std::uint8_t> gpuOut(2, 0);
    cuda.transfer().device_to_host(gpuOut.data(), dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Greater rejects mixed dtypes, naming node + op.
TEST(NertCudaMisc, GreaterRejectsDtypeMix) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a(3, 1.0F);
    std::vector<int64_t> b(3, 1);
    std::vector<std::uint8_t> y(3, 0);
    nert::Node node = MakeNode("gt_bad", "Greater");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(cuda, a);
    void* db = UploadInt64(cuda, b);
    void* dy = cuda.allocator().allocate(3);
    nert::TensorView deviceY;
    deviceY.data = dy;
    deviceY.dtype = nert::DataType::Bool;
    deviceY.dims = {3};
    try {
        nert::CudaDevice().kernels().find("Greater")(node, {DeviceFloat(da, {3}), DeviceInt64(db, {3})}, {deviceY});
        FAIL() << "expected InferenceException for mixed dtypes";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("gt_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Greater"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// And rejects non-bool inputs, naming node + op.
TEST(NertCudaMisc, AndRejectsNonBool) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a(2, 1.0F);
    std::vector<float> b(2, 1.0F);
    std::vector<std::uint8_t> y(2, 0);
    nert::Node node = MakeNode("and_bad", "And");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = cuda.allocator().allocate(2);
    nert::TensorView deviceY;
    deviceY.data = dy;
    deviceY.dtype = nert::DataType::Bool;
    deviceY.dims = {2};
    try {
        nert::CudaDevice().kernels().find("And")(node, {DeviceFloat(da, {2}), DeviceFloat(db, {2})}, {deviceY});
        FAIL() << "expected InferenceException for non-bool inputs";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("and_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("And"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// Less float: {1,2,3} < {2,2,2}, both tables.
TEST(NertCudaMisc, LessFloatExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, 2.0F, 3.0F};
    std::vector<float> b = {2.0F, 2.0F, 2.0F};
    std::vector<std::uint8_t> cpuOut(3, 0);
    nert::Node node = MakeNode("lt0", "Less");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Bool;
    cpuY.dims = {3};
    nert::CpuDevice().kernels().find("Less")(node, {HostFloat(a, {3}), HostFloat(b, {3})}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 0, 0}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = cuda.allocator().allocate(3);
    nert::TensorView deviceY;
    deviceY.data = dy;
    deviceY.dtype = nert::DataType::Bool;
    deviceY.dims = {3};
    nert::CudaDevice().kernels().find("Less")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {3})}, {deviceY});
    std::vector<std::uint8_t> gpuOut(3, 0);
    cuda.transfer().device_to_host(gpuOut.data(), dy, 3);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// GreaterOrEqual int64 broadcast: {2,1} >= 1, both tables.
TEST(NertCudaMisc, GreaterOrEqualBroadcastInt64Exact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {2, 1};
    std::vector<int64_t> b = {1};
    std::vector<std::uint8_t> cpuOut(2, 0);
    nert::Node node = MakeNode("ge0", "GreaterOrEqual");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Bool;
    cpuY.dims = {2};
    nert::CpuDevice().kernels().find("GreaterOrEqual")(node, {HostInt64(a, {2}), HostInt64(b, {1})}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<std::uint8_t>{1, 1}));

    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    void* dy = cuda.allocator().allocate(2);
    nert::TensorView deviceY;
    deviceY.data = dy;
    deviceY.dtype = nert::DataType::Bool;
    deviceY.dims = {2};
    nert::CudaDevice().kernels().find("GreaterOrEqual")(node, {DeviceInt64(da, {2}), DeviceInt64(db, {1})}, {deviceY});
    std::vector<std::uint8_t> gpuOut(2, 0);
    cuda.transfer().device_to_host(gpuOut.data(), dy, 2);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Max 3-input broadcast fold vs oracle, float.
TEST(NertCudaMisc, MaxBroadcastFoldExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a = {1.0F, 5.0F};
    std::vector<float> b = {4.0F, 2.0F, 3.0F, 0.0F};
    std::vector<float> c = {2.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("max0", "Max");
    node.inputs = {"a", "b", "c"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Max")(node, {HostFloat(a, {2, 1}), HostFloat(b, {2, 2}), HostFloat(c, {1})},
                                            {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{4.0F, 2.0F, 5.0F, 5.0F}));

    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dc = Upload(cuda, c);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Max")(
        node, {DeviceFloat(da, {2, 1}), DeviceFloat(db, {2, 2}), DeviceFloat(dc, {1})}, {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dc);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Min int64 2-input vs oracle.
TEST(NertCudaMisc, MinInt64Exact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> a = {1, 5, 3};
    std::vector<int64_t> b = {4, 2, 3};
    std::vector<int64_t> cpuOut(3, 0);
    nert::Node node = MakeNode("min0", "Min");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Min")(node, {HostInt64(a, {3}), HostInt64(b, {3})}, {HostInt64(cpuOut, {3})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{1, 2, 3}));

    void* da = UploadInt64(cuda, a);
    void* db = UploadInt64(cuda, b);
    std::vector<int64_t> gpuOut(3, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Min")(node, {DeviceInt64(da, {3}), DeviceInt64(db, {3})},
                                             {DeviceInt64(dy, {3})});
    gpuOut = DownloadInt64(cuda, dy, 3);
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Clip with min+max inputs, and with both absent (identity).
TEST(NertCudaMisc, ClipBoundsExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {-5.0F, 0.5F, 5.0F};
    std::vector<float> lo = {0.0F};
    std::vector<float> hi = {1.0F};
    std::vector<float> cpuOut(3, 0.0F);
    nert::Node node = MakeNode("clip0", "Clip");
    node.inputs = {"x", "lo", "hi"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Clip")(node, {HostFloat(x, {3}), HostFloat(lo, {1}), HostFloat(hi, {1})},
                                             {HostFloat(cpuOut, {3})});
    ExpectNear(cpuOut, (std::vector<float>{0.0F, 0.5F, 1.0F}));

    void* dx = Upload(cuda, x);
    void* dlo = Upload(cuda, lo);
    void* dhi = Upload(cuda, hi);
    std::vector<float> gpuOut(3, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Clip")(
        node, {DeviceFloat(dx, {3}), DeviceFloat(dlo, {1}), DeviceFloat(dhi, {1})}, {DeviceFloat(dy, {3})});
    gpuOut = Download(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dlo);
    cuda.allocator().release(dhi);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);

    // Both bounds absent: identity copy on both tables.
    nert::Node nodeI = MakeNode("clip1", "Clip");
    nodeI.inputs = {"x"};
    nodeI.outputs = {"y"};
    std::vector<float> cpuI(3, 0.0F);
    nert::CpuDevice().kernels().find("Clip")(nodeI, {HostFloat(x, {3})}, {HostFloat(cpuI, {3})});
    ExpectNear(cpuI, x);
    void* dx2 = Upload(cuda, x);
    std::vector<float> gpuI(3, 0.0F);
    void* dy2 = Upload(cuda, gpuI);
    nert::CudaDevice().kernels().find("Clip")(nodeI, {DeviceFloat(dx2, {3})}, {DeviceFloat(dy2, {3})});
    gpuI = Download(cuda, dy2, 3);
    cuda.allocator().release(dx2);
    cuda.allocator().release(dy2);
    ExpectNear(gpuI, cpuI);
}

// Max rejects a broadcast mismatch, naming node + op.
TEST(NertCudaMisc, MaxRejectsBroadcastMismatch) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> a(3, 1.0F);
    std::vector<float> b(2, 2.0F);
    std::vector<float> y(3, 0.0F);
    nert::Node node = MakeNode("max_bad", "Max");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    void* da = Upload(cuda, a);
    void* db = Upload(cuda, b);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("Max")(node, {DeviceFloat(da, {3}), DeviceFloat(db, {2})},
                                                 {DeviceFloat(dy, {3})});
        FAIL() << "expected InferenceException for a broadcast mismatch";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("max_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Max"), std::string::npos) << message;
    }
    cuda.allocator().release(da);
    cuda.allocator().release(db);
    cuda.allocator().release(dy);
}

// Clip rejects a non-scalar max, naming node + op.
TEST(NertCudaMisc, ClipRejectsNonScalarBound) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(3, 0.5F);
    std::vector<float> hi = {1.0F, 2.0F};
    std::vector<float> y(3, 0.0F);
    nert::Node node = MakeNode("clip_bad", "Clip");
    node.inputs = {"x", "hi"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dhi = Upload(cuda, hi);
    void* dy = Upload(cuda, y);
    // Second input position is min; give it a scalar min via Unknown is not
    // possible positionally, so pass max-shape at min position: still scalar
    // check triggers on the 2-element bound.
    nert::TensorView loView;
    loView.data = nullptr;
    loView.dtype = nert::DataType::Unknown;
    try {
        nert::CudaDevice().kernels().find("Clip")(node, {DeviceFloat(dx, {3}), loView, DeviceFloat(dhi, {2})},
                                                  {DeviceFloat(dy, {3})});
        FAIL() << "expected InferenceException for a non-scalar bound";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("clip_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("Clip"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dhi);
    cuda.allocator().release(dy);
}

// Clip on int64 data matches the CPU oracle exactly, both bounds present.
TEST(NertCudaMisc, ClipInt64Bounds) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x = {-5, 0, 7, 20};
    std::vector<int64_t> lo = {-1};
    std::vector<int64_t> hi = {10};
    std::vector<int64_t> cpuOut(4, 0);
    nert::Node node = MakeNode("clip_i64", "Clip");
    node.inputs = {"x", "lo", "hi"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Clip")(node, {HostInt64(x, {4}), HostInt64(lo, {1}), HostInt64(hi, {1})},
                                             {HostInt64(cpuOut, {4})});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{-1, 0, 7, 10}));

    void* dx = UploadInt64(cuda, x);
    void* dlo = UploadInt64(cuda, lo);
    void* dhi = UploadInt64(cuda, hi);
    std::vector<int64_t> gpuOut(4, 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Clip")(
        node, {DeviceInt64(dx, {4}), DeviceInt64(dlo, {1}), DeviceInt64(dhi, {1})}, {DeviceInt64(dy, {4})});
    gpuOut = DownloadInt64(cuda, dy, 4);
    cuda.allocator().release(dx);
    cuda.allocator().release(dlo);
    cuda.allocator().release(dhi);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Clip on int64 with no bounds is the identity, int64 extremes included.
TEST(NertCudaMisc, ClipInt64Identity) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x = {-5, 0, 7, 20, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    std::vector<int64_t> cpuOut(x.size(), 0);
    nert::Node node = MakeNode("clip_i64_id", "Clip");
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Clip")(node, {HostInt64(x, {static_cast<int64_t>(x.size())})},
                                             {HostInt64(cpuOut, {static_cast<int64_t>(cpuOut.size())})});
    EXPECT_EQ(cpuOut, x);

    void* dx = UploadInt64(cuda, x);
    std::vector<int64_t> gpuOut(x.size(), 0);
    void* dy = UploadInt64(cuda, gpuOut);
    nert::CudaDevice().kernels().find("Clip")(node, {DeviceInt64(dx, {static_cast<int64_t>(x.size())})},
                                              {DeviceInt64(dy, {static_cast<int64_t>(x.size())})});
    gpuOut = DownloadInt64(cuda, dy, x.size());
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// ReduceMean over axis 1 of [2,3] with keepdims: {2, 5}, both tables.
TEST(NertCudaMisc, ReduceMeanAxisExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<int64_t> axes = {1};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("rmn0", "ReduceMean");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceMean")(node, {HostFloat(x, {2, 3}), HostInt64(axes, {1})},
                                                   {HostFloat(cpuOut, {2, 1})});
    ExpectNear(cpuOut, (std::vector<float>{2.0F, 5.0F}));

    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceMean")(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(da, {1})},
                                                    {DeviceFloat(dy, {2, 1})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// A declared-13 ReduceMean carries axes in an attribute and no axes input. The
// CUDA row reads that form in place and matches the CPU oracle.
TEST(NertCudaMisc, ReduceMeanOldAttributeFormMatchesCpu) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("rmn_attr", "ReduceMean");
    node.schema_version = 13;
    node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceMean")(node, {HostFloat(x, {2, 3})}, {HostFloat(cpuOut, {2, 1})});
    ExpectNear(cpuOut, (std::vector<float>{2.0F, 5.0F}));

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceMean")(node, {DeviceFloat(dx, {2, 3})}, {DeviceFloat(dy, {2, 1})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// ReduceMean rejects non-float32 data, naming node + op.
TEST(NertCudaMisc, ReduceMeanRejectsInt64) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<int64_t> x(4, 1);
    std::vector<int64_t> y(1, 0);
    nert::Node node = MakeNode("rmn_bad", "ReduceMean");
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = UploadInt64(cuda, x);
    void* dy = UploadInt64(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ReduceMean")(node, {DeviceInt64(dx, {2, 2})}, {DeviceInt64(dy, {})});
        FAIL() << "expected InferenceException for non-float32 data";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("rmn_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ReduceMean"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// ReduceL2 over axis 1 of [2,2]: {sqrt(1+4), sqrt(9+16)}, both tables.
TEST(NertCudaMisc, ReduceL2AxisExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> axes = {1};
    std::vector<float> cpuOut(2, 0.0F);
    nert::Node node = MakeNode("rl2_0", "ReduceL2");
    node.inputs = {"x", "axes"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceL2")(node, {HostFloat(x, {2, 2}), HostInt64(axes, {1})},
                                                 {HostFloat(cpuOut, {2, 1})});
    ExpectNear(cpuOut, (std::vector<float>{2.23606798F, 5.0F}));

    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axes);
    std::vector<float> gpuOut(2, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceL2")(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(da, {1})},
                                                  {DeviceFloat(dy, {2, 1})});
    gpuOut = Download(cuda, dy, 2);
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// A declared-13 ReduceL2 carries axes in an attribute and no axes input. The
// CUDA row reads that form in place and matches the CPU oracle.
TEST(NertCudaMisc, ReduceL2OldAttributeFormMatchesCpu) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {3.0F, 4.0F, 0.0F, 0.0F, 1.0F, 1.0F};
    std::vector<float> cpuOut(3, -1.0F);
    nert::Node node = MakeNode("rl2_attr", "ReduceL2");
    node.schema_version = 13;
    node.attributes.emplace_back("axes", std::vector<int64_t>{1});
    node.inputs = {"x"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ReduceL2")(node, {HostFloat(x, {3, 2})}, {HostFloat(cpuOut, {3, 1})});
    ExpectNear(cpuOut, (std::vector<float>{5.0F, 0.0F, 1.41421356F}));

    void* dx = Upload(cuda, x);
    std::vector<float> gpuOut(3, -1.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ReduceL2")(node, {DeviceFloat(dx, {3, 2})}, {DeviceFloat(dy, {3, 1})});
    gpuOut = Download(cuda, dy, 3);
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
    ExpectNear(gpuOut, cpuOut);
}

// ArgMax axis 1 keepdims over [2,3] with a tie + select_last_index, both tables.
TEST(NertCudaMisc, ArgMaxTieExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 5.0F, 5.0F, 4.0F, 2.0F, 0.0F};
    auto run = [&](int64_t last, const std::vector<int64_t>& expected) {
        std::vector<int64_t> cpuOut(2, -1);
        nert::Node node = MakeNode("am0", "ArgMax");
        node.attributes.emplace_back("axis", int64_t{1});
        node.attributes.emplace_back("select_last_index", last);
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("ArgMax")(node, {HostFloat(x, {2, 3})}, {HostInt64(cpuOut, {2, 1})});
        EXPECT_EQ(cpuOut, expected);

        void* dx = Upload(cuda, x);
        std::vector<int64_t> gpuOut(2, -1);
        void* dy = UploadInt64(cuda, gpuOut);
        nert::CudaDevice().kernels().find("ArgMax")(node, {DeviceFloat(dx, {2, 3})}, {DeviceInt64(dy, {2, 1})});
        gpuOut = DownloadInt64(cuda, dy, 2);
        cuda.allocator().release(dx);
        cuda.allocator().release(dy);
        EXPECT_EQ(gpuOut, cpuOut);
    };
    run(0, {1, 0});
    run(1, {2, 0});
}

// ArgMax rejects a bad axis, naming node + op.
TEST(NertCudaMisc, ArgMaxRejectsBadAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(6, 1.0F);
    std::vector<int64_t> y(2, 0);
    nert::Node node = MakeNode("am_bad", "ArgMax");
    node.attributes.emplace_back("axis", int64_t{5});
    node.inputs = {"x"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* dy = UploadInt64(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ArgMax")(node, {DeviceFloat(dx, {2, 3})}, {DeviceInt64(dy, {2, 1})});
        FAIL() << "expected InferenceException for a bad axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("am_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ArgMax"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(dy);
}

// CumSum inclusive + exclusive-reverse over {1,2,3}, both tables.
TEST(NertCudaMisc, CumSumModesExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x = {1.0F, 2.0F, 3.0F};
    std::vector<int64_t> axis = {0};
    auto run = [&](int64_t excl, int64_t rev, const std::vector<float>& expected) {
        std::vector<float> cpuOut(3, 0.0F);
        nert::Node node = MakeNode("cs0", "CumSum");
        node.attributes.emplace_back("exclusive", excl);
        node.attributes.emplace_back("reverse", rev);
        node.inputs = {"x", "axis"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("CumSum")(node, {HostFloat(x, {3}), HostInt64(axis, {1})},
                                                   {HostFloat(cpuOut, {3})});
        ExpectNear(cpuOut, expected);

        void* dx = Upload(cuda, x);
        void* da = UploadInt64(cuda, axis);
        std::vector<float> gpuOut(3, 0.0F);
        void* dy = Upload(cuda, gpuOut);
        nert::CudaDevice().kernels().find("CumSum")(node, {DeviceFloat(dx, {3}), DeviceInt64(da, {1})},
                                                    {DeviceFloat(dy, {3})});
        gpuOut = Download(cuda, dy, 3);
        cuda.allocator().release(dx);
        cuda.allocator().release(da);
        cuda.allocator().release(dy);
        ExpectNear(gpuOut, cpuOut);
    };
    run(0, 0, {1.0F, 3.0F, 6.0F});
    run(1, 1, {5.0F, 3.0F, 0.0F});
}

// CumSum rejects a non-scalar axis, naming node + op.
TEST(NertCudaMisc, CumSumRejectsNonScalarAxis) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> x(4, 1.0F);
    std::vector<int64_t> axis = {0, 1};
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("cs_bad", "CumSum");
    node.inputs = {"x", "axis"};
    node.outputs = {"y"};
    void* dx = Upload(cuda, x);
    void* da = UploadInt64(cuda, axis);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("CumSum")(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(da, {2})},
                                                    {DeviceFloat(dy, {2, 2})});
        FAIL() << "expected InferenceException for a non-scalar axis";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("cs_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("CumSum"), std::string::npos) << message;
    }
    cuda.allocator().release(dx);
    cuda.allocator().release(da);
    cuda.allocator().release(dy);
}

// ScatterND float: copy + single-slice replace with negative index, both tables.
TEST(NertCudaMisc, ScatterNDExact) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> indices = {-1};
    std::vector<float> updates = {9.0F, 8.0F};
    std::vector<float> cpuOut(4, 0.0F);
    nert::Node node = MakeNode("sc0", "ScatterND");
    node.inputs = {"data", "indices", "updates"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("ScatterND")(
        node, {HostFloat(data, {2, 2}), HostInt64(indices, {1, 1}), HostFloat(updates, {1, 2})},
        {HostFloat(cpuOut, {2, 2})});
    EXPECT_EQ(cpuOut, (std::vector<float>{1.0F, 2.0F, 9.0F, 8.0F}));

    void* dd = Upload(cuda, data);
    void* di = UploadInt64(cuda, indices);
    void* du = Upload(cuda, updates);
    std::vector<float> gpuOut(4, 0.0F);
    void* dy = Upload(cuda, gpuOut);
    nert::CudaDevice().kernels().find("ScatterND")(
        node, {DeviceFloat(dd, {2, 2}), DeviceInt64(di, {1, 1}), DeviceFloat(du, {1, 2})}, {DeviceFloat(dy, {2, 2})});
    gpuOut = Download(cuda, dy, 4);
    cuda.allocator().release(dd);
    cuda.allocator().release(di);
    cuda.allocator().release(du);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// ScatterND rejects an out-of-range index, naming node + op.
TEST(NertCudaMisc, ScatterNDRejectsBadIndex) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    std::vector<float> data(4, 1.0F);
    std::vector<int64_t> indices = {5};
    std::vector<float> updates = {9.0F, 8.0F};
    std::vector<float> y(4, 0.0F);
    nert::Node node = MakeNode("sc_bad", "ScatterND");
    node.inputs = {"data", "indices", "updates"};
    node.outputs = {"y"};
    void* dd = Upload(cuda, data);
    void* di = UploadInt64(cuda, indices);
    void* du = Upload(cuda, updates);
    void* dy = Upload(cuda, y);
    try {
        nert::CudaDevice().kernels().find("ScatterND")(
            node, {DeviceFloat(dd, {2, 2}), DeviceInt64(di, {1, 1}), DeviceFloat(du, {1, 2})},
            {DeviceFloat(dy, {2, 2})});
        FAIL() << "expected InferenceException for an out-of-range index";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sc_bad"), std::string::npos) << message;
        EXPECT_NE(message.find("ScatterND"), std::string::npos) << message;
    }
    cuda.allocator().release(dd);
    cuda.allocator().release(di);
    cuda.allocator().release(du);
    cuda.allocator().release(dy);
}

// ConstantOfShape with an empty shape vector creates a scalar (OWLv2's
// empty text-model tensor): the CPU oracle allows zero-length shape inputs.
TEST(NertCudaMisc, ConstantOfShapeEmptyShapeScalar) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    nert::Node node = MakeNode("cos_empty", "ConstantOfShape");
    node.attributes.emplace_back("value", 7.5F);
    node.inputs = {"shape"};
    node.outputs = {"y"};
    nert::TensorView cpuShape;
    cpuShape.data = nullptr;
    cpuShape.dtype = nert::DataType::Int64;
    cpuShape.dims = {0};
    std::vector<float> cpuOut(1, 0.0F);
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Float32;
    cpuY.dims = {};
    nert::CpuDevice().kernels().find("ConstantOfShape")(node, {cpuShape}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<float>{7.5F}));

    nert::TensorView cudaShape;
    cudaShape.data = nullptr;
    cudaShape.dtype = nert::DataType::Int64;
    cudaShape.dims = {0};
    std::vector<float> zero(1, 0.0F);
    void* dy = cuda.allocator().allocate(static_cast<int64_t>(sizeof(float)));
    ASSERT_NE(dy, nullptr);
    cuda.transfer().host_to_device(dy, zero.data(), static_cast<int64_t>(sizeof(float)));
    nert::TensorView cudaY;
    cudaY.data = dy;
    cudaY.dtype = nert::DataType::Float32;
    cudaY.dims = {};
    nert::CudaDevice().kernels().find("ConstantOfShape")(node, {cudaShape}, {cudaY});
    std::vector<float> gpuOut(1, 0.0F);
    cuda.transfer().device_to_host(gpuOut.data(), dy, static_cast<int64_t>(sizeof(float)));
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);
}

// Cast to=6 emulates int32 on int64 lanes: truncation + saturation, NaN -> 0,
// identical to the CPU oracle on both the float and int64 input paths.
TEST(NertCudaMisc, CastInt32Emulation) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> xf = {1.9F, -2.7F, 0.0F, 3e9F, -3e9F, nan};
    nert::Node node = MakeNode("cast6", "Cast");
    node.attributes.emplace_back("to", int64_t{6});
    node.inputs = {"x"};
    node.outputs = {"y"};
    std::vector<int64_t> cpuOut(6, 0);
    nert::TensorView cpuY;
    cpuY.data = cpuOut.data();
    cpuY.dtype = nert::DataType::Int64;
    cpuY.dims = {6};
    nert::CpuDevice().kernels().find("Cast")(node, {HostFloat(xf, {6})}, {cpuY});
    EXPECT_EQ(cpuOut, (std::vector<int64_t>{1, -2, 0, 2147483647LL, -2147483648LL, 0}));

    void* dxf = Upload(cuda, xf);
    std::vector<int64_t> zero(6, 0);
    void* dy = UploadInt64(cuda, zero);
    nert::CudaDevice().kernels().find("Cast")(node, {DeviceFloat(dxf, {6})}, {DeviceInt64(dy, {6})});
    std::vector<int64_t> gpuOut = DownloadInt64(cuda, dy, 6);
    cuda.allocator().release(dxf);
    cuda.allocator().release(dy);
    EXPECT_EQ(gpuOut, cpuOut);

    std::vector<int64_t> xi = {3, -4, 5000000000LL, -5000000000LL};
    std::vector<int64_t> cpuInt(4, 0);
    nert::TensorView cpuIntY;
    cpuIntY.data = cpuInt.data();
    cpuIntY.dtype = nert::DataType::Int64;
    cpuIntY.dims = {4};
    nert::CpuDevice().kernels().find("Cast")(node, {HostInt64(xi, {4})}, {cpuIntY});
    EXPECT_EQ(cpuInt, (std::vector<int64_t>{3, -4, 2147483647LL, -2147483648LL}));

    void* dxi = UploadInt64(cuda, xi);
    std::vector<int64_t> zeroInt(4, 0);
    void* dyi = UploadInt64(cuda, zeroInt);
    nert::CudaDevice().kernels().find("Cast")(node, {DeviceInt64(dxi, {4})}, {DeviceInt64(dyi, {4})});
    std::vector<int64_t> gpuInt = DownloadInt64(cuda, dyi, 4);
    cuda.allocator().release(dxi);
    cuda.allocator().release(dyi);
    EXPECT_EQ(gpuInt, cpuInt);
}

// Einsum CUDA matches the CPU oracle on matrix product, DETR-style
// bchw,bnc, ellipsis batch matmul + broadcast, trace, and transpose.
TEST(NertCudaMisc, EinsumMatchesOracle) {
    if (!HasCudaDevice()) {
        GTEST_SKIP() << "no CUDA device";
    }
    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());

    struct Case {
        std::string equation;
        std::vector<std::vector<float>> data;
        std::vector<std::vector<int64_t>> dims;
        std::vector<int64_t> outDims;
    };
    std::vector<Case> cases = {
        {"ij,jk->ik", {{1.0F, 2.0F, 3.0F, 4.0F}, {5.0F, 6.0F, 7.0F, 8.0F}}, {{2, 2}, {2, 2}}, {2, 2}},
        {"bchw,bnc->bnhw",
         {{1.0F, 2.0F, 3.0F, 4.0F}, {1.0F, 0.0F, 1.0F, 1.0F}},
         {{1, 2, 1, 2}, {1, 2, 2}},
         {1, 2, 1, 2}},
        {"...pd,...qd->...pq",
         {{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
          {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F}},
         {{1, 2, 3}, {1, 4, 3}},
         {1, 2, 4}},
        {"...i,...i->...i", {{1.0F, 2.0F, 3.0F}, {10.0F, 20.0F, 30.0F, 100.0F, 200.0F, 300.0F}}, {{3}, {2, 3}}, {2, 3}},
        {"ii->", {{1.0F, 2.0F, 3.0F, 4.0F}}, {{2, 2}}, {}},
        {"ij->ji", {{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}}, {{2, 3}}, {3, 2}},
    };

    for (const Case& c : cases) {
        nert::Node node = MakeNode("es", "Einsum");
        node.attributes.emplace_back("equation", c.equation);
        int64_t outCount = 1;
        for (int64_t d : c.outDims) {
            outCount *= d;
        }

        std::vector<std::vector<float>> host = c.data;
        std::vector<nert::TensorView> cpuIns;
        for (std::size_t i = 0; i < host.size(); ++i) {
            cpuIns.push_back(HostFloat(host[i], c.dims[i]));
        }
        std::vector<float> cpuOut(static_cast<std::size_t>(outCount), 0.0F);
        nert::CpuDevice().kernels().find("Einsum")(node, cpuIns, {HostFloat(cpuOut, c.outDims)});

        std::vector<void*> deviceIns;
        std::vector<nert::TensorView> cudaIns;
        for (std::size_t i = 0; i < host.size(); ++i) {
            void* device = Upload(cuda, host[i]);
            deviceIns.push_back(device);
            cudaIns.push_back(DeviceFloat(device, c.dims[i]));
        }
        std::vector<float> zero(static_cast<std::size_t>(outCount), 0.0F);
        void* deviceOut = Upload(cuda, zero);
        nert::CudaDevice().kernels().find("Einsum")(node, cudaIns, {DeviceFloat(deviceOut, c.outDims)});
        std::vector<float> gpuOut = Download(cuda, deviceOut, static_cast<std::size_t>(outCount));
        for (void* device : deviceIns) {
            cuda.allocator().release(device);
        }
        cuda.allocator().release(deviceOut);
        EXPECT_EQ(gpuOut, cpuOut) << "equation " << c.equation;
    }
}
