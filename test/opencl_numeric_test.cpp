#include "device_test_helpers.hpp"
#include "nert/device.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace nert_test;

TEST(NertOpenCLNumeric, UnaryFloatSpecialValuesAndScalar) {
    if (!DeviceAvailable(&nert::OpenCLDevice))
        GTEST_SKIP() << "no OpenCL GPU";
    auto& dev = const_cast<nert::Device&>(nert::OpenCLDevice());
    for (const char* op : {"Abs", "Neg", "Floor"}) {
        for (bool scalar : {false, true}) {
            std::vector<float> x = scalar ? std::vector<float>{-1.25F}
                                          : std::vector<float>{-2.75F,
                                                               -0.0F,
                                                               0.0F,
                                                               3.25F,
                                                               std::numeric_limits<float>::infinity(),
                                                               -std::numeric_limits<float>::infinity(),
                                                               std::numeric_limits<float>::quiet_NaN()};
            const std::vector<int64_t> dims = scalar ? std::vector<int64_t>{} : std::vector<int64_t>{7};
            std::vector<float> ref(x.size());
            auto node = MakeNode("numeric", op);
            auto kernel = dev.kernels().find(op);
            ASSERT_NE(kernel, nullptr) << op;
            nert::CpuDevice().kernels().find(op)(node, {HostFloat(x, dims)}, {HostFloat(ref, dims)});
            void* dx = Upload(dev, x);
            void* dy = Upload(dev, ref);
            kernel(node, {DeviceFloat(dx, dims)}, {DeviceFloat(dy, dims)});
            const auto got = Download(dev, dy, x.size());
            dev.allocator().release(dx);
            dev.allocator().release(dy);
            for (std::size_t i = 0; i < ref.size(); ++i) {
                if (std::isnan(ref[i]))
                    EXPECT_TRUE(std::isnan(got[i])) << op;
                else {
                    EXPECT_EQ(got[i], ref[i]) << op;
                    EXPECT_EQ(std::signbit(got[i]), std::signbit(ref[i])) << op;
                }
            }
        }
    }
}

TEST(NertOpenCLNumeric, UnaryInt64ExactBeyondFloatPrecision) {
    if (!DeviceAvailable(&nert::OpenCLDevice))
        GTEST_SKIP() << "no OpenCL GPU";
    auto& dev = const_cast<nert::Device&>(nert::OpenCLDevice());
    for (const char* op : {"Abs", "Neg"}) {
        std::vector<int64_t> x = {-(int64_t{1} << 54) - 1, -1, 0, (int64_t{1} << 54) + 1};
        std::vector<int64_t> ref(4);
        auto node = MakeNode("numeric_int", op);
        nert::CpuDevice().kernels().find(op)(node, {HostInt64(x, {2, 2})}, {HostInt64(ref, {4})});
        auto kernel = dev.kernels().find(op);
        ASSERT_NE(kernel, nullptr);
        void* dx = UploadInt64(dev, x);
        void* dy = UploadInt64(dev, ref);
        kernel(node, {DeviceInt64(dx, {2, 2})}, {DeviceInt64(dy, {4})});
        const auto got = DownloadInt64(dev, dy, 4);
        dev.allocator().release(dx);
        dev.allocator().release(dy);
        EXPECT_EQ(got, ref) << op;
    }
}

TEST(NertOpenCLNumeric, GreaterEqualBroadcastFloatNaNAndInt64) {
    if (!DeviceAvailable(&nert::OpenCLDevice))
        GTEST_SKIP() << "no OpenCL GPU";
    auto& dev = const_cast<nert::Device&>(nert::OpenCLDevice());
    auto kernel = dev.kernels().find("GreaterOrEqual");
    ASSERT_NE(kernel, nullptr);
    auto node = MakeNode("ge", "GreaterOrEqual");
    for (bool integer : {false, true}) {
        std::vector<float> af = {1, std::numeric_limits<float>::quiet_NaN()};
        std::vector<float> bf = {0, 1, 2};
        std::vector<int64_t> ai = {(int64_t{1} << 54), (int64_t{1} << 54) + 1};
        std::vector<int64_t> bi = {(int64_t{1} << 54) - 1, (int64_t{1} << 54), (int64_t{1} << 54) + 1};
        void* da = integer ? UploadInt64(dev, ai) : Upload(dev, af);
        void* db = integer ? UploadInt64(dev, bi) : Upload(dev, bf);
        void* dy = dev.allocator().allocate(6);
        nert::TensorView out{dy, nert::DataType::Bool, {2, 3}};
        kernel(node,
               {integer ? DeviceInt64(da, {2, 1}) : DeviceFloat(da, {2, 1}),
                integer ? DeviceInt64(db, {1, 3}) : DeviceFloat(db, {1, 3})},
               {out});
        std::vector<uint8_t> got(6);
        dev.transfer().device_to_host(got.data(), dy, 6);
        dev.allocator().release(da);
        dev.allocator().release(db);
        dev.allocator().release(dy);
        EXPECT_EQ(got, integer ? (std::vector<uint8_t>{1, 1, 0, 1, 1, 1}) : (std::vector<uint8_t>{1, 1, 0, 0, 0, 0}));
    }
}

TEST(NertOpenCLNumeric, RejectInvalidUnaryInputs) {
    if (!DeviceAvailable(&nert::OpenCLDevice))
        GTEST_SKIP() << "no OpenCL GPU";
    for (const char* op : {"Abs", "Neg", "Floor"}) {
        auto kernel = nert::OpenCLDevice().kernels().find(op);
        ASSERT_NE(kernel, nullptr);
        auto node = MakeNode("bad_numeric", op);
        nert::TensorView empty{nullptr, nert::DataType::Float32, {1}};
        EXPECT_THROW(kernel(node, {}, {empty}), nert::InferenceException);
        EXPECT_THROW(kernel(node, {empty}, {empty}), nert::InferenceException);
        auto wrong = empty;
        wrong.dtype = nert::DataType::Bool;
        EXPECT_THROW(kernel(node, {wrong}, {wrong}), nert::InferenceException);
        wrong = empty;
        wrong.dims = {2};
        EXPECT_THROW(kernel(node, {empty}, {wrong}), nert::InferenceException);
        wrong.dims = {0};
        EXPECT_THROW(kernel(node, {wrong}, {wrong}), nert::InferenceException);
    }
}
} // namespace
