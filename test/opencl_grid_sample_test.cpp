// OpenCL GridSample row: bilinear and nearest sampling, all three padding
// modes, both align_corners settings.
//
// Each case hand-computes the expected values, runs the CPU oracle first
// and then the OpenCL row on device buffers; negatives assert the node/op
// in the error. Needs an OpenCL GPU at run time.

#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

// Defined in src/opencl/grid_sample.cpp. Declared here (not wired into the
// OpenCL kernel table yet) so this suite links and calls the row directly
// on device views.
namespace nert {
namespace opencl_kernels {

void GridSample(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

} // namespace opencl_kernels
} // namespace nert

namespace {

using nert_test::DeviceFloat;
using nert_test::Download;
using nert_test::HostFloat;
using nert_test::MakeNode;
using nert_test::Upload;

bool HasOpenCLDevice() { return nert_test::DeviceAvailable(&nert::OpenCLDevice); }

int64_t Count(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        count *= dim;
    }
    return count;
}

void SetStringAttribute(nert::Node& node, const std::string& name, const std::string& value) {
    nert::Attribute attr;
    attr.name = name;
    attr.value = value;
    node.attributes.push_back(attr);
}

void SetIntAttribute(nert::Node& node, const std::string& name, int64_t value) {
    nert::Attribute attr;
    attr.name = name;
    attr.value = value;
    node.attributes.push_back(attr);
}

// Run the CPU oracle on host views and return the output values.
std::vector<float> RunOracle(const nert::Node& node, std::vector<float>& x, const std::vector<int64_t>& x_dims,
                             std::vector<float>& grid, const std::vector<int64_t>& grid_dims,
                             const std::vector<int64_t>& out_dims) {
    std::vector<float> ref(static_cast<std::size_t>(Count(out_dims)), 0.0F);
    nert::CpuDevice().kernels().find("GridSample")(node, {HostFloat(x, x_dims), HostFloat(grid, grid_dims)},
                                                   {HostFloat(ref, out_dims)});
    return ref;
}

// Upload the operands, run the OpenCL row, download, release the buffers.
std::vector<float> RunDevice(nert::Device& ocl, const nert::Node& node, const std::vector<float>& x,
                             const std::vector<int64_t>& x_dims, const std::vector<float>& grid,
                             const std::vector<int64_t>& grid_dims, const std::vector<int64_t>& out_dims) {
    void* dx = Upload(ocl, x);
    void* dg = Upload(ocl, grid);
    void* dy = ocl.allocator().allocate(static_cast<int64_t>(Count(out_dims) * sizeof(float)));
    nert::opencl_kernels::GridSample(node, {DeviceFloat(dx, x_dims), DeviceFloat(dg, grid_dims)},
                                     {DeviceFloat(dy, out_dims)});
    std::vector<float> got = Download(ocl, dy, static_cast<std::size_t>(Count(out_dims)));
    ocl.allocator().release(dx);
    ocl.allocator().release(dg);
    ocl.allocator().release(dy);
    return got;
}

void ExpectNear(const std::vector<float>& got, const std::vector<float>& ref, float tol = 1e-6F) {
    EXPECT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < got.size() && i < ref.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], tol * (1.0F + std::fabs(ref[i]))) << "i=" << i;
    }
}

// X {1,1,2,2} {{1,2},{3,4}} sampled exactly at the four pixel centres under
// align_corners=1 (normalized corners -1/+1): the output equals X, exactly
// on the oracle and on the device.
TEST(NertOpenCLGridSample, BilinearIdentity) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    // Pixel centres under align_corners=1 are the corners of the normalized
    // square; grid is [Ho, Wo, 2] with (x, y) per output pixel, row-major.
    std::vector<float> grid = {-1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F};
    nert::Node node = MakeNode("gs_identity", "GridSample");
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    SetIntAttribute(node, "align_corners", 1);
    std::vector<float> ref = RunOracle(node, x, {1, 1, 2, 2}, grid, {1, 2, 2, 2}, {1, 1, 2, 2});
    EXPECT_EQ(ref, x);
    std::vector<float> got = RunDevice(ocl, node, x, {1, 1, 2, 2}, grid, {1, 2, 2, 2}, {1, 1, 2, 2});
    EXPECT_EQ(got, x);
    EXPECT_EQ(got, ref);
}

// align_corners=0 maps g to ((g + 1) * size - 1) / 2, so grid (0, 0) on the
// 2x2 plane {{1,2},{3,4}} lands on continuous (0.5, 0.5): all four taps get
// weight 0.25 and the output is (1 + 2 + 3 + 4) / 4 = 2.5.
TEST(NertOpenCLGridSample, BilinearHalfPixelAlign0) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> grid = {0.0F, 0.0F};
    nert::Node node = MakeNode("gs_half0", "GridSample");
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    SetIntAttribute(node, "align_corners", 0);
    std::vector<float> ref = RunOracle(node, x, {1, 1, 2, 2}, grid, {1, 1, 1, 2}, {1, 1, 1, 1});
    EXPECT_NEAR(ref[0], 2.5F, 1e-6F);
    std::vector<float> got = RunDevice(ocl, node, x, {1, 1, 2, 2}, grid, {1, 1, 1, 2}, {1, 1, 1, 1});
    EXPECT_NEAR(got[0], 2.5F, 1e-6F);
    ExpectNear(got, ref);
}

// align_corners=1 maps g to (g + 1) / 2 * (size - 1), so grid (0, 0) on the
// 3x3 plane 1..9 lands on continuous (1, 1): the centre pixel, 5.0.
TEST(NertOpenCLGridSample, BilinearAlign1Center) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    std::vector<float> grid = {0.0F, 0.0F};
    nert::Node node = MakeNode("gs_center1", "GridSample");
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    SetIntAttribute(node, "align_corners", 1);
    std::vector<float> ref = RunOracle(node, x, {1, 1, 3, 3}, grid, {1, 1, 1, 2}, {1, 1, 1, 1});
    EXPECT_NEAR(ref[0], 5.0F, 1e-6F);
    std::vector<float> got = RunDevice(ocl, node, x, {1, 1, 3, 3}, grid, {1, 1, 1, 2}, {1, 1, 1, 1});
    EXPECT_NEAR(got[0], 5.0F, 1e-6F);
    ExpectNear(got, ref);
}

// Nearest ties round to even, matching the oracle's std::nearbyint and the
// device's rint. With align_corners=0 and size 2 the denormalized
// coordinate is ((g + 1) * 2 - 1) / 2 = g + 0.5, so g = 0 lands on exactly
// 0.5, g = 1 on 1.5 and g = -1 on -0.5. nearbyint/rint gives 0 for 0.5
// (ties to even, in range), 2 for 1.5 (ties to even, outside the 2-wide
// axis, so the zeros padding yields 0) and -0 for -0.5 (in range). Half-up
// would return 40 for the first point and truncation would return 40 for
// the second, so {10, 0, 10} pins ties-to-even exactly.
TEST(NertOpenCLGridSample, NearestHalfToEven) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {10.0F, 20.0F, 30.0F, 40.0F};
    std::vector<float> grid = {0.0F, 0.0F, 1.0F, 1.0F, -1.0F, -1.0F};
    nert::Node node = MakeNode("gs_nearest", "GridSample");
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    SetStringAttribute(node, "mode", "nearest");
    SetIntAttribute(node, "align_corners", 0);
    std::vector<float> ref = RunOracle(node, x, {1, 1, 2, 2}, grid, {1, 1, 3, 2}, {1, 1, 1, 3});
    EXPECT_EQ(ref, (std::vector<float>{10.0F, 0.0F, 10.0F}));
    std::vector<float> got = RunDevice(ocl, node, x, {1, 1, 2, 2}, grid, {1, 1, 3, 2}, {1, 1, 1, 3});
    EXPECT_EQ(got, ref);
}

// One grid point (2, 2) outside the 3x3 plane 1..9 under align_corners=1
// lands on continuous (3.0, 3.0), one pixel past the border:
//   zeros      -> every tap outside the plane        -> 0.0
//   border     -> clamped to (2, 2), the corner      -> 9.0
//   reflection -> reflected about [0, 2] to (1, 1)   -> 5.0
TEST(NertOpenCLGridSample, PaddingModes) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Device& ocl = const_cast<nert::Device&>(nert::OpenCLDevice());
    std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
    std::vector<float> grid = {2.0F, 2.0F};
    const std::vector<std::pair<std::string, float>> cases = {{"zeros", 0.0F}, {"border", 9.0F}, {"reflection", 5.0F}};
    for (const std::pair<std::string, float>& c : cases) {
        nert::Node node = MakeNode("gs_pad_" + c.first, "GridSample");
        node.inputs = {"x", "grid"};
        node.outputs = {"y"};
        SetIntAttribute(node, "align_corners", 1);
        SetStringAttribute(node, "padding_mode", c.first);
        std::vector<float> ref = RunOracle(node, x, {1, 1, 3, 3}, grid, {1, 1, 1, 2}, {1, 1, 1, 1});
        EXPECT_NEAR(ref[0], c.second, 1e-6F) << c.first;
        std::vector<float> got = RunDevice(ocl, node, x, {1, 1, 3, 3}, grid, {1, 1, 1, 2}, {1, 1, 1, 1});
        EXPECT_NEAR(got[0], c.second, 1e-6F) << c.first;
        ExpectNear(got, ref);
    }
}

// Negative paths name the node and the op.
TEST(NertOpenCLGridSample, Negatives) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    auto expect_throw = [&](const nert::Node& node, const std::vector<nert::TensorView>& inputs,
                            const std::vector<nert::TensorView>& outputs) {
        try {
            nert::opencl_kernels::GridSample(node, inputs, outputs);
            FAIL() << "expected InferenceException for " << node.name;
        } catch (const nert::InferenceException& e) {
            const std::string message = e.what();
            EXPECT_NE(message.find(node.name), std::string::npos) << message;
            EXPECT_NE(message.find(node.op_type), std::string::npos) << message;
        }
    };
    std::vector<float> x_data = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> grid_data = {0.0F, 0.0F};
    std::vector<float> out_data = {0.0F};

    // The grid's last axis must be 2.
    nert::Node bad_last = MakeNode("gs_bad_last", "GridSample");
    expect_throw(bad_last, {HostFloat(x_data, {1, 1, 2, 2}), HostFloat(grid_data, {1, 1, 1, 1})},
                 {HostFloat(out_data, {1, 1, 1, 1})});

    // X must be rank 4.
    nert::Node bad_rank = MakeNode("gs_bad_rank", "GridSample");
    expect_throw(bad_rank, {HostFloat(x_data, {1, 2, 2}), HostFloat(grid_data, {1, 1, 1, 2})},
                 {HostFloat(out_data, {1, 1, 1, 1})});

    // Unsupported mode.
    nert::Node bad_mode = MakeNode("gs_bad_mode", "GridSample");
    SetStringAttribute(bad_mode, "mode", "bicubic");
    expect_throw(bad_mode, {HostFloat(x_data, {1, 1, 2, 2}), HostFloat(grid_data, {1, 1, 1, 2})},
                 {HostFloat(out_data, {1, 1, 1, 1})});

    // Unsupported padding mode.
    nert::Node bad_padding = MakeNode("gs_bad_padding", "GridSample");
    SetStringAttribute(bad_padding, "padding_mode", "wrap");
    expect_throw(bad_padding, {HostFloat(x_data, {1, 1, 2, 2}), HostFloat(grid_data, {1, 1, 1, 2})},
                 {HostFloat(out_data, {1, 1, 1, 1})});

    // align_corners must be 0 or 1.
    nert::Node bad_align = MakeNode("gs_bad_align", "GridSample");
    SetIntAttribute(bad_align, "align_corners", 2);
    expect_throw(bad_align, {HostFloat(x_data, {1, 1, 2, 2}), HostFloat(grid_data, {1, 1, 1, 2})},
                 {HostFloat(out_data, {1, 1, 1, 1})});

    // Null grid buffer (X and output must be non-null to reach that check).
    nert::Node null_grid = MakeNode("gs_null_grid", "GridSample");
    nert::TensorView null_view;
    null_view.data = nullptr;
    null_view.dtype = nert::DataType::Float32;
    null_view.dims = {1, 1, 1, 2};
    expect_throw(null_grid, {HostFloat(x_data, {1, 1, 2, 2}), null_view}, {HostFloat(out_data, {1, 1, 1, 1})});
}

// Zero-extent dimensions are rejected by the dimension check, naming the op.
TEST(NertOpenCLGridSample, ZeroExtentRejected) {
    if (!HasOpenCLDevice()) {
        GTEST_SKIP() << "no OpenCL GPU platform";
    }
    nert::Node node = MakeNode("gs_zero", "GridSample");
    node.inputs = {"x", "grid"};
    node.outputs = {"y"};
    std::vector<float> empty;
    std::vector<float> grid_data = {0.0F, 0.0F};
    std::vector<float> out_data = {0.0F};
    try {
        nert::opencl_kernels::GridSample(node, {HostFloat(empty, {1, 1, 0, 3}), HostFloat(grid_data, {1, 1, 1, 2})},
                                         {HostFloat(out_data, {1, 1, 1, 1})});
        FAIL() << "expected InferenceException for zero extent";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("non-positive dimension"), std::string::npos) << message;
        EXPECT_NE(message.find("GridSample"), std::string::npos) << message;
        EXPECT_NE(message.find(node.name), std::string::npos) << message;
    }
}

} // namespace
