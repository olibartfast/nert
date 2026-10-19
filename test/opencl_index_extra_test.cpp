// OpenCL index and shape rows: Identity, Squeeze, CumSum, EyeLike, ScatterND,
// ReduceMean.
//
// Each case runs the CPU oracle first and then the OpenCL row on device
// views; negatives assert the node/op in the error. Needs an OpenCL GPU at
// run time.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Defined in src/opencl/index_extra.cpp. Declared here (not wired into the
// OpenCL kernel table yet) so this suite links and calls the rows directly
// on device views.
namespace nert {
namespace opencl_kernels {

void Identity(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Squeeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void CumSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void EyeLike(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ScatterND(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceMean(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

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

// Bool helpers (copied from the owlv2 rows suite): bool tensors ride on
// byte buffers.
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

std::vector<std::uint8_t> DownloadBytes(nert::Device& ocl, void* buffer, std::size_t count) {
    std::vector<std::uint8_t> host(count, 0);
    ocl.transfer().device_to_host(host.data(), buffer, static_cast<int64_t>(host.size()));
    return host;
}

void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref, float tol = 1e-6F) {
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], tol) << "i=" << i;
    }
}

// Identity copies every supported dtype byte for byte; the device output is
// byte-equal to the input and to the CPU oracle.
TEST(NertOpenCLIndexExtra, IdentityDtypes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    {
        std::vector<float> x = {1.5F, -2.25F, 3.0F, 0.0F};
        std::vector<float> ref(4, 0.0F);
        nert::Node node = MakeNode("id_f", "Identity");
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("Identity")(node, {HostFloat(x, {2, 2})}, {HostFloat(ref, {2, 2})});
        EXPECT_EQ(ref, x);
        void* dx = Upload(ocl, x);
        void* dy = Upload(ocl, std::vector<float>(4, 0.0F));
        nert::opencl_kernels::Identity(node, {DeviceFloat(dx, {2, 2})}, {DeviceFloat(dy, {2, 2})});
        EXPECT_EQ(Download(ocl, dy, 4), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    }
    {
        std::vector<int64_t> x = {5, -6, 0, 42};
        std::vector<int64_t> ref(4, 0);
        nert::Node node = MakeNode("id_i", "Identity");
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("Identity")(node, {HostInt64(x, {2, 2})}, {HostInt64(ref, {2, 2})});
        EXPECT_EQ(ref, x);
        void* dx = UploadInt64(ocl, x);
        void* dy = UploadInt64(ocl, std::vector<int64_t>(4, 0));
        nert::opencl_kernels::Identity(node, {DeviceInt64(dx, {2, 2})}, {DeviceInt64(dy, {2, 2})});
        EXPECT_EQ(DownloadInt64(ocl, dy, 4), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    }
    {
        std::vector<std::uint8_t> x = {0, 1, 1, 0};
        std::vector<std::uint8_t> ref(4, 0);
        nert::Node node = MakeNode("id_b", "Identity");
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("Identity")(node, {HostBool(x, {2, 2})}, {HostBool(ref, {2, 2})});
        EXPECT_EQ(ref, x);
        void* dx = nullptr;
        void* dy = nullptr;
        UploadBytes(ocl, dx, x);
        UploadBytes(ocl, dy, std::vector<std::uint8_t>(4, 0));
        nert::opencl_kernels::Identity(node, {DeviceBool(dx, {2, 2})}, {DeviceBool(dy, {2, 2})});
        EXPECT_EQ(DownloadBytes(ocl, dy, 4), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    }
}

// Squeeze keeps the element order; the output dims come from shape
// inference and the row copies the buffer.
TEST(NertOpenCLIndexExtra, SqueezeShapes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    {
        std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        std::vector<float> ref(6, 0.0F);
        nert::Node node = MakeNode("sq_f", "Squeeze");
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("Squeeze")(node, {HostFloat(x, {2, 1, 3})}, {HostFloat(ref, {2, 3})});
        EXPECT_EQ(ref, x);
        void* dx = Upload(ocl, x);
        void* dy = Upload(ocl, std::vector<float>(6, 0.0F));
        nert::opencl_kernels::Squeeze(node, {DeviceFloat(dx, {2, 1, 3})}, {DeviceFloat(dy, {2, 3})});
        EXPECT_EQ(Download(ocl, dy, 6), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    }
    {
        std::vector<int64_t> x = {7, 8, 9, 10, 11, 12};
        std::vector<int64_t> ref(6, 0);
        nert::Node node = MakeNode("sq_i", "Squeeze");
        node.inputs = {"x"};
        node.outputs = {"y"};
        nert::CpuDevice().kernels().find("Squeeze")(node, {HostInt64(x, {2, 1, 3})}, {HostInt64(ref, {2, 3})});
        EXPECT_EQ(ref, x);
        void* dx = UploadInt64(ocl, x);
        void* dy = UploadInt64(ocl, std::vector<int64_t>(6, 0));
        nert::opencl_kernels::Squeeze(node, {DeviceInt64(dx, {2, 1, 3})}, {DeviceInt64(dy, {2, 3})});
        EXPECT_EQ(DownloadInt64(ocl, dy, 6), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    }
}

// CumSum variants on {1,2,3,4} shape {2,2}, each checked against the CPU
// oracle: axis 1 (default, exclusive, reverse, reverse+exclusive), axis 0
// (reverse, reverse+exclusive), and an exact int64 case beyond float32's
// exact integer range.
TEST(NertOpenCLIndexExtra, CumSumVariants) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};

    auto run_case = [&](const char* name, int64_t axis, int64_t excl, int64_t rev, const std::vector<float>& expect) {
        nert::Node node = MakeNode(name, "CumSum");
        node.inputs = {"x", "axis"};
        node.outputs = {"y"};
        if (excl != 0) {
            node.attributes.push_back(nert::Attribute("exclusive", excl));
        }
        if (rev != 0) {
            node.attributes.push_back(nert::Attribute("reverse", rev));
        }
        std::vector<float> ref(4, 0.0F);
        std::vector<int64_t> axisHost = {axis};
        nert::CpuDevice().kernels().find("CumSum")(node, {HostFloat(x, {2, 2}), HostInt64(axisHost, {1})},
                                                   {HostFloat(ref, {2, 2})});
        EXPECT_EQ(ref, expect) << name;
        void* dx = Upload(ocl, x);
        void* dax = UploadInt64(ocl, axisHost);
        void* dy = Upload(ocl, std::vector<float>(4, 0.0F));
        nert::opencl_kernels::CumSum(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(dax, {1})}, {DeviceFloat(dy, {2, 2})});
        EXPECT_EQ(Download(ocl, dy, 4), ref) << name;
        ocl.allocator().release(dx);
        ocl.allocator().release(dax);
        ocl.allocator().release(dy);
    };

    run_case("cum_inc", 1, 0, 0, {1.0F, 3.0F, 3.0F, 7.0F});
    run_case("cum_exc", 1, 1, 0, {0.0F, 1.0F, 0.0F, 3.0F});
    run_case("cum_rev1", 1, 0, 1, {3.0F, 2.0F, 7.0F, 4.0F});
    run_case("cum_revex1", 1, 1, 1, {2.0F, 0.0F, 4.0F, 0.0F});
    run_case("cum_rev0", 0, 0, 1, {4.0F, 6.0F, 3.0F, 4.0F});
    run_case("cum_revex0", 0, 1, 1, {3.0F, 4.0F, 0.0F, 0.0F});

    {
        // The int64 row is exact where float32 would round the running sum.
        std::vector<int64_t> x64 = {100000001, 100000002, 100000003, 100000004};
        std::vector<int64_t> ref(4, 0);
        nert::Node node = MakeNode("cum_i64", "CumSum");
        node.inputs = {"x", "axis"};
        node.outputs = {"y"};
        std::vector<int64_t> axisHost = {1};
        nert::CpuDevice().kernels().find("CumSum")(node, {HostInt64(x64, {2, 2}), HostInt64(axisHost, {1})},
                                                   {HostInt64(ref, {2, 2})});
        EXPECT_EQ(ref, (std::vector<int64_t>{100000001, 200000003, 100000003, 200000007}));
        void* dx = UploadInt64(ocl, x64);
        void* dax = UploadInt64(ocl, axisHost);
        void* dy = UploadInt64(ocl, std::vector<int64_t>(4, 0));
        nert::opencl_kernels::CumSum(node, {DeviceInt64(dx, {2, 2}), DeviceInt64(dax, {1})}, {DeviceInt64(dy, {2, 2})});
        EXPECT_EQ(DownloadInt64(ocl, dy, 4), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dax);
        ocl.allocator().release(dy);
    }
}

// EyeLike paints ones on the k-th diagonal; k = 1 and k = -1 are hand
// asserted and the int64 row is exact.
TEST(NertOpenCLIndexExtra, EyeLikeK) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    const std::vector<float> k0 = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    const std::vector<float> k1 = {0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> km1 = {0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};

    auto run_eye = [&](const char* name, int64_t k, const std::vector<float>& expect) {
        nert::Node node = MakeNode(name, "EyeLike");
        node.inputs = {"x"};
        node.outputs = {"y"};
        node.attributes.push_back(nert::Attribute("k", k));
        std::vector<float> ref(12, -1.0F);
        std::vector<float> in(12, 3.5F);
        nert::CpuDevice().kernels().find("EyeLike")(node, {HostFloat(in, {3, 4})}, {HostFloat(ref, {3, 4})});
        EXPECT_EQ(ref, expect) << name;
        void* dx = Upload(ocl, in);
        void* dy = Upload(ocl, std::vector<float>(12, 0.0F));
        nert::opencl_kernels::EyeLike(node, {DeviceFloat(dx, {3, 4})}, {DeviceFloat(dy, {3, 4})});
        EXPECT_EQ(Download(ocl, dy, 12), ref) << name;
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    };

    run_eye("eye0", 0, k0);
    run_eye("eye_pos", 1, k1);
    run_eye("eye_neg", -1, km1);

    {
        nert::Node node = MakeNode("eye_i64", "EyeLike");
        node.inputs = {"x"};
        node.outputs = {"y"};
        node.attributes.push_back(nert::Attribute("k", static_cast<int64_t>(0)));
        std::vector<int64_t> ref(12, -1);
        std::vector<float> in(12, 1.0F);
        nert::CpuDevice().kernels().find("EyeLike")(node, {HostFloat(in, {3, 4})}, {HostInt64(ref, {3, 4})});
        EXPECT_EQ(ref, (std::vector<int64_t>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}));
        void* dx = Upload(ocl, in);
        void* dy = UploadInt64(ocl, std::vector<int64_t>(12, 0));
        nert::opencl_kernels::EyeLike(node, {DeviceFloat(dx, {3, 4})}, {DeviceInt64(dy, {3, 4})});
        EXPECT_EQ(DownloadInt64(ocl, dy, 12), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dy);
    }
}

// ScatterND replaces whole slices; a negative index counts back and a K=2
// index addresses one element of the 2-D data. Device matches the oracle.
TEST(NertOpenCLIndexExtra, ScatterNDBasic) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());

    auto run_scatter = [&](const char* name, std::vector<float> data, const std::vector<int64_t>& dataDims,
                           std::vector<int64_t> idx, const std::vector<int64_t>& idxDims, std::vector<float> upd,
                           const std::vector<int64_t>& updDims, const std::vector<float>& expect) {
        nert::Node node = MakeNode(name, "ScatterND");
        node.inputs = {"data", "indices", "updates"};
        node.outputs = {"y"};
        const std::size_t count = expect.size();
        std::vector<float> ref(count, -1.0F);
        nert::CpuDevice().kernels().find("ScatterND")(
            node, {HostFloat(data, dataDims), HostInt64(idx, idxDims), HostFloat(upd, updDims)},
            {HostFloat(ref, dataDims)});
        EXPECT_EQ(ref, expect) << name;
        void* dd = Upload(ocl, data);
        void* di = UploadInt64(ocl, idx);
        void* du = Upload(ocl, upd);
        void* dou = Upload(ocl, std::vector<float>(count, 0.0F));
        nert::opencl_kernels::ScatterND(node,
                                        {DeviceFloat(dd, dataDims), DeviceInt64(di, idxDims), DeviceFloat(du, updDims)},
                                        {DeviceFloat(dou, dataDims)});
        EXPECT_EQ(Download(ocl, dou, count), ref) << name;
        ocl.allocator().release(dd);
        ocl.allocator().release(di);
        ocl.allocator().release(du);
        ocl.allocator().release(dou);
    };

    run_scatter("sc_rows", std::vector<float>(12, 0.0F), {4, 3}, {0, 2}, {2, 1}, {10, 11, 12, 20, 21, 22}, {2, 3},
                {10, 11, 12, 0, 0, 0, 20, 21, 22, 0, 0, 0});
    run_scatter("sc_neg", std::vector<float>(12, 0.0F), {4, 3}, {-1}, {1, 1}, {7, 8, 9}, {1, 3},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 9});
    run_scatter("sc_k2", std::vector<float>(12, 0.0F), {4, 3}, {1, 2}, {1, 2}, {99}, {1},
                {0, 0, 0, 0, 0, 99, 0, 0, 0, 0, 0, 0});
}

// ReduceMean over the requested axes with double accumulation; the device
// matches the oracle within 1e-6.
TEST(NertOpenCLIndexExtra, ReduceMeanAxes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

    {
        nert::Node node = MakeNode("mean1", "ReduceMean");
        node.inputs = {"x", "axes"};
        node.outputs = {"y"};
        node.attributes.push_back(nert::Attribute("keepdims", static_cast<int64_t>(1)));
        std::vector<int64_t> axesHost = {1};
        std::vector<float> ref(2, 0.0F);
        nert::CpuDevice().kernels().find("ReduceMean")(node, {HostFloat(x, {2, 3}), HostInt64(axesHost, {1})},
                                                       {HostFloat(ref, {2, 1})});
        ExpectNear(ref, {2.0F, 5.0F});
        void* dx = Upload(ocl, x);
        void* dax = UploadInt64(ocl, axesHost);
        void* dy = Upload(ocl, std::vector<float>(2, 0.0F));
        nert::opencl_kernels::ReduceMean(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dax, {1})},
                                         {DeviceFloat(dy, {2, 1})});
        ExpectNear(Download(ocl, dy, 2), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dax);
        ocl.allocator().release(dy);
    }
    {
        nert::Node node = MakeNode("mean0", "ReduceMean");
        node.inputs = {"x", "axes"};
        node.outputs = {"y"};
        node.attributes.push_back(nert::Attribute("keepdims", static_cast<int64_t>(0)));
        std::vector<int64_t> axesHost = {0};
        std::vector<float> ref(3, 0.0F);
        nert::CpuDevice().kernels().find("ReduceMean")(node, {HostFloat(x, {2, 3}), HostInt64(axesHost, {1})},
                                                       {HostFloat(ref, {3})});
        ExpectNear(ref, {2.5F, 3.5F, 4.5F});
        void* dx = Upload(ocl, x);
        void* dax = UploadInt64(ocl, axesHost);
        void* dy = Upload(ocl, std::vector<float>(3, 0.0F));
        nert::opencl_kernels::ReduceMean(node, {DeviceFloat(dx, {2, 3}), DeviceInt64(dax, {1})},
                                         {DeviceFloat(dy, {3})});
        ExpectNear(Download(ocl, dy, 3), ref);
        ocl.allocator().release(dx);
        ocl.allocator().release(dax);
        ocl.allocator().release(dy);
    }
}

// Negative paths name the node and the op.
TEST(NertOpenCLIndexExtra, Negatives) {
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

    nert::Node bad_id = MakeNode("bad_id", "Identity");
    expect_throw(&nert::opencl_kernels::Identity, bad_id, {tv(nert::DataType::Float32, {2, 3})},
                 {tv(nert::DataType::Float32, {3, 2})});

    nert::Node bad_sq = MakeNode("bad_sq", "Squeeze");
    expect_throw(&nert::opencl_kernels::Squeeze, bad_sq, {tv(nert::DataType::Float32, {2, 1, 3})},
                 {tv(nert::DataType::Float32, {2, 2})});

    nert::Node bad_cum = MakeNode("bad_cum", "CumSum");
    void* dax = UploadInt64(ocl, {3});
    expect_throw(&nert::opencl_kernels::CumSum, bad_cum, {tv(nert::DataType::Float32, {2}), DeviceInt64(dax, {1})},
                 {tv(nert::DataType::Float32, {2})});
    ocl.allocator().release(dax);

    nert::Node bad_eye = MakeNode("bad_eye", "EyeLike");
    expect_throw(&nert::opencl_kernels::EyeLike, bad_eye, {tv(nert::DataType::Float32, {2, 2, 2})},
                 {tv(nert::DataType::Float32, {2, 2, 2})});

    nert::Node bad_sc = MakeNode("bad_sc", "ScatterND");
    void* dd = Upload(ocl, std::vector<float>(4, 0.0F));
    void* di = UploadInt64(ocl, {5});
    void* du = Upload(ocl, std::vector<float>(2, 0.0F));
    void* dou = Upload(ocl, std::vector<float>(4, 0.0F));
    expect_throw(&nert::opencl_kernels::ScatterND, bad_sc,
                 {DeviceFloat(dd, {2, 2}), DeviceInt64(di, {1, 1}), DeviceFloat(du, {1, 2})},
                 {DeviceFloat(dou, {2, 2})});
    ocl.allocator().release(dd);
    ocl.allocator().release(di);
    ocl.allocator().release(du);
    ocl.allocator().release(dou);

    nert::Node bad_red = MakeNode("bad_red", "ScatterND");
    bad_red.attributes.push_back(nert::Attribute("reduction", std::string("add")));
    expect_throw(
        &nert::opencl_kernels::ScatterND, bad_red,
        {tv(nert::DataType::Float32, {2, 2}), tv(nert::DataType::Int64, {1, 1}), tv(nert::DataType::Float32, {1, 2})},
        {tv(nert::DataType::Float32, {2, 2})});

    nert::Node bad_mean = MakeNode("bad_mean", "ReduceMean");
    void* dx = Upload(ocl, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    void* dmean_ax = UploadInt64(ocl, {0, 0});
    void* dmy = Upload(ocl, std::vector<float>(3, 0.0F));
    expect_throw(&nert::opencl_kernels::ReduceMean, bad_mean, {DeviceFloat(dx, {2, 3}), DeviceInt64(dmean_ax, {2})},
                 {DeviceFloat(dmy, {3})});
    ocl.allocator().release(dx);
    ocl.allocator().release(dmean_ax);
    ocl.allocator().release(dmy);
}

// Zero-extent dims are rejected by the dimension check, naming the op.
TEST(NertOpenCLIndexExtra, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    void* dax = UploadInt64(ocl, {1});
    nert::Node node = MakeNode("zero_cum", "CumSum");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {0, 2};
    nert::TensorView out = in;
    try {
        nert::opencl_kernels::CumSum(node, {in, DeviceInt64(dax, {1})}, {out});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("CumSum"), std::string::npos) << message;
        EXPECT_NE(message.find(node.name), std::string::npos) << message;
    }
    ocl.allocator().release(dax);
}

} // namespace
