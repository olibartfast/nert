// OpenCL index-moving and integer rows: Split, GatherElements, TopK, Mod, Tile.
//
// Each case runs the CPU oracle first (hand-asserted where the packet pins
// values) and then the OpenCL row vs the oracle on device views. Negative
// cases assert the node/op in the error. Needs an OpenCL GPU at run time.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

// Defined in src/opencl/index_move.cpp. Declared here (not wired into the
// OpenCL kernel table yet) so this suite links and calls the rows directly on
// device views.
namespace nert {
namespace opencl_kernels {

void Split(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GatherElements(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void TopK(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Mod(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Tile(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

} // namespace opencl_kernels
} // namespace nert

namespace {

using nert_test::DeviceFloat;
using nert_test::DeviceInt64;
using nert_test::Download;
using nert_test::HostFloat;
using nert_test::MakeNode;
using nert_test::Upload;
using nert_test::UploadInt64;

bool HasOpenCLDevice() { return nert_test::DeviceAvailable(&nert::OpenCLDevice); }

nert::TensorView HostInt64(std::vector<int64_t>& data, std::vector<int64_t> dims) {
    nert::TensorView view;
    view.data = data.data();
    view.dtype = nert::DataType::Int64;
    view.dims = std::move(dims);
    return view;
}

std::vector<int64_t> DownloadInt64(nert::Device& ocl, void* ptr, std::size_t count) {
    std::vector<int64_t> host(count, 0);
    if (count > 0) {
        ocl.transfer().device_to_host(host.data(), ptr, static_cast<int64_t>(count * sizeof(int64_t)));
    }
    return host;
}

// Releases the device buffer when the scope ends so every upload is freed
// even when an assertion trips mid-test.
class ScopedBuffer {
  public:
    ScopedBuffer(nert::Device& ocl, void* ptr) : ocl_(ocl), ptr_(ptr) {}
    ~ScopedBuffer() {
        if (ptr_ != nullptr) {
            ocl_.allocator().release(ptr_);
        }
    }
    ScopedBuffer(const ScopedBuffer&) = delete;
    ScopedBuffer& operator=(const ScopedBuffer&) = delete;
    void* get() const { return ptr_; }

  private:
    nert::Device& ocl_;
    void* ptr_;
};

void SetAttr(nert::Node& node, const std::string& name, int64_t value) {
    nert::Attribute attr{};
    attr.name = name;
    attr.value = value;
    node.attributes.push_back(attr);
}

// Split with explicit sizes {1,3} on axis 1: hand values, then device vs oracle.
TEST(NertOpenCLIndexMove, SplitUneven) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<int64_t> sizes = {1, 3};
    std::vector<float> ref0(2, 0.0F);
    std::vector<float> ref1(6, 0.0F);
    nert::Node node = MakeNode("split_even_sizes", "Split");
    node.inputs = {"x", "sizes"};
    node.outputs = {"y0", "y1"};
    SetAttr(node, "axis", 1);
    nert::CpuDevice().kernels().find("Split")(node, {HostFloat(x, {2, 4}), HostInt64(sizes, {2})},
                                              {HostFloat(ref0, {2, 1}), HostFloat(ref1, {2, 3})});
    EXPECT_EQ(ref0, (std::vector<float>{1.0F, 5.0F}));
    EXPECT_EQ(ref1, (std::vector<float>{2.0F, 3.0F, 4.0F, 6.0F, 7.0F, 8.0F}));

    void* dx = Upload(ocl, x);
    ScopedBuffer gx(ocl, dx);
    void* dsizes = UploadInt64(ocl, sizes);
    ScopedBuffer gsizes(ocl, dsizes);
    std::vector<float> g0(2, 0.0F);
    std::vector<float> g1(6, 0.0F);
    void* dy0 = Upload(ocl, g0);
    ScopedBuffer gy0(ocl, dy0);
    void* dy1 = Upload(ocl, g1);
    ScopedBuffer gy1(ocl, dy1);
    nert::opencl_kernels::Split(node, {DeviceFloat(dx, {2, 4}), DeviceInt64(dsizes, {2})},
                                {DeviceFloat(dy0, {2, 1}), DeviceFloat(dy1, {2, 3})});
    EXPECT_EQ(Download(ocl, dy0, 2), ref0);
    EXPECT_EQ(Download(ocl, dy1, 6), ref1);
}

// A pre-13 Split carries split sizes in an attribute; OpenCL and CPU agree.
TEST(NertOpenCLIndexMove, SplitOldAttributeForm) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    nert::Node node = MakeNode("split_attr", "Split");
    node.schema_version = 11;
    node.inputs = {"x"};
    node.outputs = {"y0", "y1"};
    SetAttr(node, "axis", 1);
    nert::Attribute split;
    split.name = "split";
    split.value = std::vector<int64_t>{1, 3};
    node.attributes.push_back(split);

    std::vector<float> ref0(2, 0.0F);
    std::vector<float> ref1(6, 0.0F);
    nert::CpuDevice().kernels().find("Split")(node, {HostFloat(x, {2, 4})},
                                              {HostFloat(ref0, {2, 1}), HostFloat(ref1, {2, 3})});
    EXPECT_EQ(ref0, (std::vector<float>{1.0F, 5.0F}));
    EXPECT_EQ(ref1, (std::vector<float>{2.0F, 3.0F, 4.0F, 6.0F, 7.0F, 8.0F}));

    void* dx = Upload(ocl, x);
    ScopedBuffer gx(ocl, dx);
    std::vector<float> g0(2, 0.0F);
    std::vector<float> g1(6, 0.0F);
    void* dy0 = Upload(ocl, g0);
    ScopedBuffer gy0(ocl, dy0);
    void* dy1 = Upload(ocl, g1);
    ScopedBuffer gy1(ocl, dy1);
    nert::opencl_kernels::Split(node, {DeviceFloat(dx, {2, 4})}, {DeviceFloat(dy0, {2, 1}), DeviceFloat(dy1, {2, 3})});
    EXPECT_EQ(Download(ocl, dy0, 2), ref0);
    EXPECT_EQ(Download(ocl, dy1, 6), ref1);
}

// Split on axis -1 with no sizes input: int64 rows split evenly in two.
TEST(NertOpenCLIndexMove, SplitEvenInt64) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<int64_t> x = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> ref0(4, 0);
    std::vector<int64_t> ref1(4, 0);
    nert::Node node = MakeNode("split_even_i64", "Split");
    node.inputs = {"x"};
    node.outputs = {"y0", "y1"};
    SetAttr(node, "axis", -1);
    nert::CpuDevice().kernels().find("Split")(node, {HostInt64(x, {2, 4})},
                                              {HostInt64(ref0, {2, 2}), HostInt64(ref1, {2, 2})});
    EXPECT_EQ(ref0, (std::vector<int64_t>{1, 2, 5, 6}));
    EXPECT_EQ(ref1, (std::vector<int64_t>{3, 4, 7, 8}));

    void* dx = UploadInt64(ocl, x);
    ScopedBuffer gx(ocl, dx);
    std::vector<int64_t> g0(4, 0);
    std::vector<int64_t> g1(4, 0);
    void* dy0 = UploadInt64(ocl, g0);
    ScopedBuffer gy0(ocl, dy0);
    void* dy1 = UploadInt64(ocl, g1);
    ScopedBuffer gy1(ocl, dy1);
    nert::opencl_kernels::Split(node, {DeviceInt64(dx, {2, 4})}, {DeviceInt64(dy0, {2, 2}), DeviceInt64(dy1, {2, 2})});
    EXPECT_EQ(DownloadInt64(ocl, dy0, 4), ref0);
    EXPECT_EQ(DownloadInt64(ocl, dy1, 4), ref1);
}

// GatherElements on axis 1 with a wrapping negative index, plus a rank-3
// gather on axis 0.
TEST(NertOpenCLIndexMove, GatherElementsAxis1) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> data = {10.0F, 11.0F, 12.0F, 20.0F, 21.0F, 22.0F};
    std::vector<int64_t> idx = {2, 0, 1, -1};
    std::vector<float> ref(4, 0.0F);
    nert::Node node = MakeNode("gather_axis1", "GatherElements");
    node.inputs = {"data", "indices"};
    node.outputs = {"y"};
    SetAttr(node, "axis", 1);
    nert::CpuDevice().kernels().find("GatherElements")(node, {HostFloat(data, {2, 3}), HostInt64(idx, {2, 2})},
                                                       {HostFloat(ref, {2, 2})});
    EXPECT_EQ(ref, (std::vector<float>{12.0F, 10.0F, 21.0F, 22.0F}));

    void* ddata = Upload(ocl, data);
    ScopedBuffer gdata(ocl, ddata);
    void* didx = UploadInt64(ocl, idx);
    ScopedBuffer gidx(ocl, didx);
    std::vector<float> gout(4, 0.0F);
    void* dy = Upload(ocl, gout);
    ScopedBuffer gy(ocl, dy);
    nert::opencl_kernels::GatherElements(node, {DeviceFloat(ddata, {2, 3}), DeviceInt64(didx, {2, 2})},
                                         {DeviceFloat(dy, {2, 2})});
    EXPECT_EQ(Download(ocl, dy, 4), ref);

    // Rank 3, axis 0: out[z,i,j] = data[idx[z,i,j], i, j]. Hand-computed for
    // indices {{{1,0},{0,1}},{{0,1},{1,0}}} on data {1..8}.
    std::vector<float> data3 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<int64_t> idx3 = {1, 0, 0, 1, 0, 1, 1, 0};
    std::vector<float> ref3(8, 0.0F);
    nert::Node node3 = MakeNode("gather_axis0", "GatherElements");
    node3.inputs = {"data", "indices"};
    node3.outputs = {"y"};
    SetAttr(node3, "axis", 0);
    nert::CpuDevice().kernels().find("GatherElements")(node3, {HostFloat(data3, {2, 2, 2}), HostInt64(idx3, {2, 2, 2})},
                                                       {HostFloat(ref3, {2, 2, 2})});
    EXPECT_EQ(ref3, (std::vector<float>{5.0F, 2.0F, 3.0F, 8.0F, 1.0F, 6.0F, 7.0F, 4.0F}));

    void* ddata3 = Upload(ocl, data3);
    ScopedBuffer gdata3(ocl, ddata3);
    void* didx3 = UploadInt64(ocl, idx3);
    ScopedBuffer gidx3(ocl, didx3);
    std::vector<float> gout3(8, 0.0F);
    void* dy3 = Upload(ocl, gout3);
    ScopedBuffer gy3(ocl, dy3);
    nert::opencl_kernels::GatherElements(node3, {DeviceFloat(ddata3, {2, 2, 2}), DeviceInt64(didx3, {2, 2, 2})},
                                         {DeviceFloat(dy3, {2, 2, 2})});
    EXPECT_EQ(Download(ocl, dy3, 8), ref3);
}

// TopK on the last axis, K=2: descending values, ties broken by lower index.
TEST(NertOpenCLIndexMove, TopKLastAxis) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 4.0F, 2.0F, 3.0F, 9.0F, 7.0F, 8.0F, 6.0F};
    std::vector<int64_t> k = {2};
    std::vector<float> ref_values(4, 0.0F);
    std::vector<int64_t> ref_indices(4, 0);
    nert::Node node = MakeNode("topk_last", "TopK");
    node.inputs = {"x", "k"};
    node.outputs = {"values", "indices"};
    nert::CpuDevice().kernels().find("TopK")(node, {HostFloat(x, {2, 4}), HostInt64(k, {1})},
                                             {HostFloat(ref_values, {2, 2}), HostInt64(ref_indices, {2, 2})});
    EXPECT_EQ(ref_values, (std::vector<float>{4.0F, 3.0F, 9.0F, 8.0F}));
    EXPECT_EQ(ref_indices, (std::vector<int64_t>{1, 3, 0, 2}));

    void* dx = Upload(ocl, x);
    ScopedBuffer gx(ocl, dx);
    void* dk = UploadInt64(ocl, k);
    ScopedBuffer gk(ocl, dk);
    std::vector<float> gvalues(4, 0.0F);
    std::vector<int64_t> gindices(4, 0);
    void* dvalues = Upload(ocl, gvalues);
    ScopedBuffer gvalues_buf(ocl, dvalues);
    void* dindices = UploadInt64(ocl, gindices);
    ScopedBuffer gindices_buf(ocl, dindices);
    nert::opencl_kernels::TopK(node, {DeviceFloat(dx, {2, 4}), DeviceInt64(dk, {1})},
                               {DeviceFloat(dvalues, {2, 2}), DeviceInt64(dindices, {2, 2})});
    EXPECT_EQ(Download(ocl, dvalues, 4), ref_values);
    EXPECT_EQ(DownloadInt64(ocl, dindices, 4), ref_indices);
}

// Mod with int64 truncating remainder: a {2,3} against b {3}, then a 2-D
// broadcast on both sides.
TEST(NertOpenCLIndexMove, ModBroadcastInt64) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<int64_t> a = {7, 8, 9, 10, 11, 12};
    std::vector<int64_t> b = {3, 4, 5};
    std::vector<int64_t> ref(6, 0);
    nert::Node node = MakeNode("mod_bcast", "Mod");
    node.inputs = {"a", "b"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Mod")(node, {HostInt64(a, {2, 3}), HostInt64(b, {3})}, {HostInt64(ref, {2, 3})});
    EXPECT_EQ(ref, (std::vector<int64_t>{1, 0, 4, 1, 3, 2}));

    void* da = UploadInt64(ocl, a);
    ScopedBuffer ga(ocl, da);
    void* db = UploadInt64(ocl, b);
    ScopedBuffer gb(ocl, db);
    std::vector<int64_t> gout(6, 0);
    void* dy = UploadInt64(ocl, gout);
    ScopedBuffer gy(ocl, dy);
    nert::opencl_kernels::Mod(node, {DeviceInt64(da, {2, 3}), DeviceInt64(db, {3})}, {DeviceInt64(dy, {2, 3})});
    EXPECT_EQ(DownloadInt64(ocl, dy, 6), ref);

    // 2-D broadcast: a {2,1} against b {1,3} -> out {2,3}.
    std::vector<int64_t> a2 = {7, 10};
    std::vector<int64_t> b2 = {3, 4, 5};
    std::vector<int64_t> ref2(6, 0);
    nert::Node node2 = MakeNode("mod_bcast2d", "Mod");
    node2.inputs = {"a", "b"};
    node2.outputs = {"y"};
    nert::CpuDevice().kernels().find("Mod")(node2, {HostInt64(a2, {2, 1}), HostInt64(b2, {1, 3})},
                                            {HostInt64(ref2, {2, 3})});
    EXPECT_EQ(ref2, (std::vector<int64_t>{1, 3, 2, 1, 2, 0}));

    void* da2 = UploadInt64(ocl, a2);
    ScopedBuffer ga2(ocl, da2);
    void* db2 = UploadInt64(ocl, b2);
    ScopedBuffer gb2(ocl, db2);
    std::vector<int64_t> gout2(6, 0);
    void* dy2 = UploadInt64(ocl, gout2);
    ScopedBuffer gy2(ocl, dy2);
    nert::opencl_kernels::Mod(node2, {DeviceInt64(da2, {2, 1}), DeviceInt64(db2, {1, 3})}, {DeviceInt64(dy2, {2, 3})});
    EXPECT_EQ(DownloadInt64(ocl, dy2, 6), ref2);
}

// Tile with repeats {2,3} on a float {2,2}, then an exact int64 tile.
TEST(NertOpenCLIndexMove, TileVariants) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> reps = {2, 3};
    std::vector<float> ref(24, 0.0F);
    nert::Node node = MakeNode("tile_float", "Tile");
    node.inputs = {"x", "repeats"};
    node.outputs = {"y"};
    nert::CpuDevice().kernels().find("Tile")(node, {HostFloat(x, {2, 2}), HostInt64(reps, {2})},
                                             {HostFloat(ref, {4, 6})});
    ASSERT_EQ(ref.size(), 24U);
    EXPECT_EQ(ref[0], 1.0F);
    EXPECT_EQ(ref[3], 2.0F);
    EXPECT_EQ(ref[6], 3.0F);
    EXPECT_EQ(ref[11], 4.0F);
    EXPECT_EQ(ref[12], 1.0F);
    EXPECT_EQ(ref[23], 4.0F);

    void* dx = Upload(ocl, x);
    ScopedBuffer gx(ocl, dx);
    void* dreps = UploadInt64(ocl, reps);
    ScopedBuffer greps(ocl, dreps);
    std::vector<float> gout(24, 0.0F);
    void* dy = Upload(ocl, gout);
    ScopedBuffer gy(ocl, dy);
    nert::opencl_kernels::Tile(node, {DeviceFloat(dx, {2, 2}), DeviceInt64(dreps, {2})}, {DeviceFloat(dy, {4, 6})});
    EXPECT_EQ(Download(ocl, dy, 24), ref);

    // int64 exact match vs the oracle.
    std::vector<int64_t> xi = {1, 2, 3, 4};
    std::vector<int64_t> reps_i = {1, 2};
    std::vector<int64_t> ref_i(8, 0);
    nert::Node node_i = MakeNode("tile_int64", "Tile");
    node_i.inputs = {"x", "repeats"};
    node_i.outputs = {"y"};
    nert::CpuDevice().kernels().find("Tile")(node_i, {HostInt64(xi, {2, 2}), HostInt64(reps_i, {2})},
                                             {HostInt64(ref_i, {2, 4})});
    EXPECT_EQ(ref_i, (std::vector<int64_t>{1, 2, 1, 2, 3, 4, 3, 4}));

    void* dxi = UploadInt64(ocl, xi);
    ScopedBuffer gxi(ocl, dxi);
    void* dreps_i = UploadInt64(ocl, reps_i);
    ScopedBuffer greps_i(ocl, dreps_i);
    std::vector<int64_t> gout_i(8, 0);
    void* dy_i = UploadInt64(ocl, gout_i);
    ScopedBuffer gy_i(ocl, dy_i);
    nert::opencl_kernels::Tile(node_i, {DeviceInt64(dxi, {2, 2}), DeviceInt64(dreps_i, {2})},
                               {DeviceInt64(dy_i, {2, 4})});
    EXPECT_EQ(DownloadInt64(ocl, dy_i, 8), ref_i);
}

// Negative paths name the node and the op.
TEST(NertOpenCLIndexMove, Negatives) {
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

    // Split sizes count (3) does not match the output count (2).
    nert::Node bad_sizes = MakeNode("bad_split_sizes", "Split");
    SetAttr(bad_sizes, "axis", 1);
    std::vector<float> x24 = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    std::vector<int64_t> sizes3 = {1, 1, 2};
    void* dx = Upload(ocl, x24);
    ScopedBuffer gx(ocl, dx);
    void* dsizes3 = UploadInt64(ocl, sizes3);
    ScopedBuffer gsizes3(ocl, dsizes3);
    std::vector<float> scratch2(2, 0.0F);
    std::vector<float> scratch6(6, 0.0F);
    void* ds2 = Upload(ocl, scratch2);
    ScopedBuffer gs2(ocl, ds2);
    void* ds6 = Upload(ocl, scratch6);
    ScopedBuffer gs6(ocl, ds6);
    expect_throw(&nert::opencl_kernels::Split, bad_sizes, {DeviceFloat(dx, {2, 4}), DeviceInt64(dsizes3, {3})},
                 {DeviceFloat(ds2, {2, 1}), DeviceFloat(ds6, {2, 3})});

    // Split axis dim 4 is not divisible by the output count 3.
    nert::Node bad_div = MakeNode("bad_split_div", "Split");
    expect_throw(&nert::opencl_kernels::Split, bad_div, {DeviceFloat(dx, {2, 4})},
                 {DeviceFloat(ds2, {2, 2}), DeviceFloat(ds2, {2, 2}), DeviceFloat(ds2, {2, 2})});

    // GatherElements index 3 is out of range for an axis dim of 3.
    nert::Node bad_gather = MakeNode("bad_gather_range", "GatherElements");
    SetAttr(bad_gather, "axis", 1);
    std::vector<int64_t> idx_bad = {0, 3, 0, 0};
    void* didx_bad = UploadInt64(ocl, idx_bad);
    ScopedBuffer gidx_bad(ocl, didx_bad);
    expect_throw(&nert::opencl_kernels::GatherElements, bad_gather,
                 {DeviceFloat(dx, {2, 3}), DeviceInt64(didx_bad, {2, 2})}, {DeviceFloat(ds6, {2, 2})});

    // TopK K=5 exceeds the axis dim 4.
    nert::Node bad_topk_k = MakeNode("bad_topk_k", "TopK");
    std::vector<int64_t> k5 = {5};
    void* dk5 = UploadInt64(ocl, k5);
    ScopedBuffer gk5(ocl, dk5);
    std::vector<int64_t> scratch_i4(4, 0);
    void* dv = Upload(ocl, scratch2);
    ScopedBuffer gv(ocl, dv);
    void* di = UploadInt64(ocl, scratch_i4);
    ScopedBuffer gi(ocl, di);
    expect_throw(&nert::opencl_kernels::TopK, bad_topk_k, {DeviceFloat(dx, {2, 4}), DeviceInt64(dk5, {1})},
                 {DeviceFloat(dv, {2, 2}), DeviceInt64(di, {2, 2})});

    // TopK largest=0 is not accepted here (rejected before K is read).
    nert::Node bad_topk_largest = MakeNode("bad_topk_largest", "TopK");
    SetAttr(bad_topk_largest, "largest", 0);
    std::vector<int64_t> k2 = {2};
    void* dk2 = UploadInt64(ocl, k2);
    ScopedBuffer gk2(ocl, dk2);
    expect_throw(&nert::opencl_kernels::TopK, bad_topk_largest, {DeviceFloat(dx, {2, 4}), DeviceInt64(dk2, {1})},
                 {DeviceFloat(dv, {2, 2}), DeviceInt64(di, {2, 2})});

    // Mod with a zero divisor.
    nert::Node bad_mod = MakeNode("bad_mod_zero", "Mod");
    std::vector<int64_t> ma = {3, 5};
    std::vector<int64_t> mb = {2, 0};
    void* dma = UploadInt64(ocl, ma);
    ScopedBuffer gma(ocl, dma);
    void* dmb = UploadInt64(ocl, mb);
    ScopedBuffer gmb(ocl, dmb);
    void* dmod = UploadInt64(ocl, scratch_i4);
    ScopedBuffer gmod(ocl, dmod);
    expect_throw(&nert::opencl_kernels::Mod, bad_mod, {DeviceInt64(dma, {2}), DeviceInt64(dmb, {2})},
                 {DeviceInt64(dmod, {2})});

    // Tile repeats rank-1 length 3 for a rank-2 input.
    nert::Node bad_tile = MakeNode("bad_tile_repeats", "Tile");
    std::vector<float> x22 = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<int64_t> reps3 = {1, 1, 1};
    void* dx22 = Upload(ocl, x22);
    ScopedBuffer gx22(ocl, dx22);
    void* dreps3 = UploadInt64(ocl, reps3);
    ScopedBuffer greps3(ocl, dreps3);
    expect_throw(&nert::opencl_kernels::Tile, bad_tile, {DeviceFloat(dx22, {2, 2}), DeviceInt64(dreps3, {3})},
                 {DeviceFloat(ds2, {2, 2})});
}

// Zero-extent dims are rejected by the dimension check, naming the op.
TEST(NertOpenCLIndexMove, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("zero_split", "Split");
    nert::TensorView in;
    in.dtype = nert::DataType::Float32;
    in.dims = {0, 2};
    nert::TensorView out = in;
    try {
        nert::opencl_kernels::Split(node, {in}, {out});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("Split"), std::string::npos) << message;
        EXPECT_NE(message.find("zero_split"), std::string::npos) << message;
    }
}

} // namespace
