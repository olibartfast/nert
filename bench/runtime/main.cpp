// Runtime bench (N1f): parity + timing for every shipped CUDA kernel.
//
// Unlike bench/gemm + bench/matmul (vendored kernel copies), this driver
// implements ZERO kernel logic: it builds one fixture per op (shapes +
// attrs + deterministic data) and runs the runtime's own tables —
// CpuDevice (oracle) vs CudaDevice (device) — through the public seam.
// Adding kernel coverage means adding a fixture, never reimplementing.
//
// Exit 0 + PASSED when every executed case matches within 1e-3*(1+|ref|);
// a missing device or a missing CUDA row prints SKIP: and still exits 0.
// Any mismatch exits 1. Timings are steady_clock around the (internally
// synchronized) kernel calls: honest end-to-end ms.

#include "nert/device.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-3;

struct Case {
    std::string op;
    std::string label;
    nert::Node node;
    std::vector<std::vector<int64_t>> inDims;
    std::vector<std::vector<float>> inData;
    std::vector<int64_t> outDims;
};

nert::Node MakeNode(const std::string& name, const std::string& op) {
    nert::Node node;
    node.name = name;
    node.op_type = op;
    return node;
}

std::vector<float> Fill(std::mt19937_64& rng, std::size_t count, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> data(count);
    for (float& v : data) {
        v = dist(rng);
    }
    return data;
}

int64_t Count(const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (int64_t d : dims) {
        n *= d;
    }
    return n;
}

std::string Shape(const std::vector<int64_t>& dims) {
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        s += std::to_string(dims[i]);
        if (i + 1 < dims.size()) {
            s += "x";
        }
    }
    return s + "]";
}

// Mean milliseconds of fn() over iters timed runs.
template <typename Fn> double TimeMs(Fn&& fn, int iters) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

bool HasCudaDevice(nert::Device& cuda) {
    void* probe = cuda.allocator().allocate(1);
    if (probe == nullptr) {
        return false;
    }
    cuda.allocator().release(probe);
    return true;
}

int RunCase(nert::Device& cuda, bool devicePresent, const Case& c, int& passed, int& skipped, int& failed) {
    nert::KernelFn cpuFn = nert::CpuDevice().kernels().find(c.op);
    if (cpuFn == nullptr) {
        std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << "NO CPU ORACLE  FAIL\n";
        ++failed;
        return 1;
    }
    std::vector<nert::TensorView> cpuIns;
    for (std::size_t i = 0; i < c.inData.size(); ++i) {
        nert::TensorView v;
        v.data = const_cast<float*>(c.inData[i].data());
        v.dtype = nert::DataType::Float32;
        v.dims = c.inDims[i];
        cpuIns.push_back(v);
    }
    std::vector<float> ref(static_cast<std::size_t>(Count(c.outDims)), 0.0F);
    nert::TensorView refView;
    refView.data = ref.data();
    refView.dtype = nert::DataType::Float32;
    refView.dims = c.outDims;
    cpuFn(c.node, cpuIns, {refView});
    const double firstMs = TimeMs([&] { cpuFn(c.node, cpuIns, {refView}); }, 1);
    const int cpuIters = firstMs > 1500.0 ? 1 : 3;
    const double cpuMs = cpuIters == 1 ? firstMs : TimeMs([&] { cpuFn(c.node, cpuIns, {refView}); }, cpuIters);

    if (!devicePresent) {
        std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << "SKIP: no CUDA device\n";
        ++skipped;
        return 0;
    }
    nert::KernelFn cudaFn = cuda.kernels().find(c.op);
    if (cudaFn == nullptr) {
        std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << "SKIP: no CUDA row\n";
        ++skipped;
        return 0;
    }
    std::vector<void*> devicePtrs;
    std::vector<nert::TensorView> cudaIns;
    for (std::size_t i = 0; i < c.inData.size(); ++i) {
        const int64_t bytes = static_cast<int64_t>(c.inData[i].size() * sizeof(float));
        void* device = cuda.allocator().allocate(bytes);
        if (device == nullptr) {
            std::cout << "FAIL: device allocate failed for " << c.op << " " << c.label << '\n';
            ++failed;
            return 1;
        }
        cuda.transfer().host_to_device(device, c.inData[i].data(), bytes);
        devicePtrs.push_back(device);
        nert::TensorView v;
        v.data = device;
        v.dtype = nert::DataType::Float32;
        v.dims = c.inDims[i];
        cudaIns.push_back(v);
    }
    std::vector<float> zeros(static_cast<std::size_t>(Count(c.outDims)), 0.0F);
    const int64_t outBytes = static_cast<int64_t>(zeros.size() * sizeof(float));
    void* deviceOut = cuda.allocator().allocate(outBytes);
    if (deviceOut == nullptr) {
        std::cout << "FAIL: device allocate failed for " << c.op << " " << c.label << '\n';
        ++failed;
        return 1;
    }
    nert::TensorView outView;
    outView.data = deviceOut;
    outView.dtype = nert::DataType::Float32;
    outView.dims = c.outDims;
    for (int i = 0; i < 3; ++i) {
        cuda.transfer().host_to_device(deviceOut, zeros.data(), outBytes);
        cudaFn(c.node, cudaIns, {outView});
    }
    const double gpuMs = TimeMs(
        [&] {
            cuda.transfer().host_to_device(deviceOut, zeros.data(), outBytes);
            cudaFn(c.node, cudaIns, {outView});
        },
        10);
    std::vector<float> got(zeros.size(), 0.0F);
    cuda.transfer().device_to_host(got.data(), deviceOut, outBytes);
    for (void* device : devicePtrs) {
        cuda.allocator().release(device);
    }
    cuda.allocator().release(deviceOut);

    double maxDiff = 0.0;
    int mismatches = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        maxDiff = std::max(maxDiff, diff);
        if (diff > kTol * (1.0 + std::fabs(static_cast<double>(ref[i])))) {
            ++mismatches;
        }
    }
    const double speedup = gpuMs > 0.0 ? cpuMs / gpuMs : 0.0;
    std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << std::right << std::fixed
              << std::setprecision(2) << std::setw(10) << cpuMs << std::setw(10) << gpuMs << std::setw(9)
              << std::setprecision(1) << speedup << "x" << std::setw(12) << std::scientific << std::setprecision(2)
              << maxDiff << "  " << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
    if (mismatches == 0) {
        ++passed;
        return 0;
    }
    ++failed;
    return 1;
}

} // namespace

int main() {
    std::mt19937_64 rng(0xC0FFEE);
    const int64_t kBig = 4 * 1024 * 1024;
    std::vector<Case> cases;

    {
        Case c;
        c.op = "Gemm";
        c.label = Shape({512, 512}) + "x" + Shape({512, 512});
        c.node = MakeNode("bench_gemm", "Gemm");
        c.inDims = {{512, 512}, {512, 512}};
        c.inData = {Fill(rng, 512 * 512, -1.0F, 1.0F), Fill(rng, 512 * 512, -1.0F, 1.0F)};
        c.outDims = {512, 512};
        cases.push_back(std::move(c));
    }
    for (const char* op : {"Relu", "Sigmoid", "Tanh"}) {
        Case c;
        c.op = op;
        c.label = Shape({kBig});
        c.node = MakeNode("bench_unary", op);
        c.inDims = {{kBig}};
        c.inData = {Fill(rng, static_cast<std::size_t>(kBig), -6.0F, 6.0F)};
        c.outDims = {kBig};
        cases.push_back(std::move(c));
    }
    for (const char* op : {"Add", "Mul", "Sub"}) {
        Case c;
        c.op = op;
        c.label = Shape({2048, 2048}) + "+" + Shape({2048});
        c.node = MakeNode("bench_bin", op);
        c.inDims = {{2048, 2048}, {2048}};
        c.inData = {Fill(rng, 2048 * 2048, -3.0F, 3.0F), Fill(rng, 2048, -3.0F, 3.0F)};
        c.outDims = {2048, 2048};
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Div";
        c.label = Shape({kBig});
        c.node = MakeNode("bench_div", "Div");
        c.inDims = {{kBig}, {kBig}};
        std::vector<float> b = Fill(rng, static_cast<std::size_t>(kBig), 0.5F, 2.0F);
        for (float& v : b) {
            if ((rng() & 1) != 0) {
                v = -v;
            }
        }
        c.inData = {Fill(rng, static_cast<std::size_t>(kBig), -3.0F, 3.0F), std::move(b)};
        c.outDims = {kBig};
        cases.push_back(std::move(c));
    }
    for (const char* op : {"MaxPool", "AveragePool"}) {
        Case c;
        c.op = op;
        c.label = Shape({8, 64, 112, 112}) + " k3 s2";
        c.node = MakeNode("bench_pool", op);
        c.node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{3, 3});
        c.node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
        c.inDims = {{8, 64, 112, 112}};
        c.inData = {Fill(rng, 8 * 64 * 112 * 112, -3.0F, 3.0F)};
        c.outDims = {8, 64, 55, 55};
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Softmax";
        c.label = Shape({256, 4096});
        c.node = MakeNode("bench_softmax", "Softmax");
        c.inDims = {{256, 4096}};
        c.inData = {Fill(rng, 256 * 4096, -3.0F, 3.0F)};
        c.outDims = {256, 4096};
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Conv";
        c.label = Shape({8, 16, 64, 64}) + " k3+bias";
        c.node = MakeNode("bench_conv", "Conv");
        c.inDims = {{8, 16, 64, 64}, {32, 16, 3, 3}, {32}};
        c.inData = {Fill(rng, 8 * 16 * 64 * 64, -1.0F, 1.0F), Fill(rng, 32 * 16 * 3 * 3, -1.0F, 1.0F),
                    Fill(rng, 32, -1.0F, 1.0F)};
        c.outDims = {8, 32, 62, 62};
        cases.push_back(std::move(c));
    }

    nert::Device& cuda = const_cast<nert::Device&>(nert::CudaDevice());
    const bool devicePresent = HasCudaDevice(cuda);
    if (!devicePresent) {
        std::cout << "SKIP: no CUDA device (device halves skipped, CPU halves still run)\n";
    }
    std::cout << std::left << std::setw(12) << "op" << std::setw(28) << "case" << std::right << std::setw(10)
              << "cpu_ms" << std::setw(10) << "gpu_ms" << std::setw(10) << "speedup" << std::setw(12) << "maxdiff"
              << '\n';

    int passed = 0, skipped = 0, failed = 0;
    int status = 0;
    for (const Case& c : cases) {
        status |= RunCase(cuda, devicePresent, c, passed, skipped, failed);
    }
    std::cout << (failed == 0 ? "PASSED " : "FAILED ") << passed << " cases (" << skipped << " skipped, " << failed
              << " failed)\n";
    return failed == 0 ? 0 : 1;
}
