// Runtime bench (N1f/N1d/N1c): parity + timing for every shipped kernel.
//
// This driver implements ZERO kernel logic: it builds one fixture per op
// (shapes + attrs + deterministic data) and runs the runtime's own tables —
// CpuDevice (oracle) vs every device layer this binary was built with
// (CUDA, OpenCL, HIP) — through the public seam. Adding kernel coverage
// means adding a fixture, never reimplementing.
//
// Exit 0 + PASSED when every executed case matches within 1e-3*(1+|ref|);
// a device that is not compiled in is absent from the binary entirely, and
// one that is compiled in but missing at run time (or lacks a row for an
// op) prints SKIP: and still exits 0. Any mismatch exits 1. Timings are
// steady_clock around the (internally synchronized) kernel calls: honest
// end-to-end ms.

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
    // Parallel to inData: a non-empty entry makes that input int64
    // (e.g. Reshape's shape operand); empty keeps the float input.
    std::vector<std::vector<int64_t>> int64Dims;
    std::vector<std::vector<int64_t>> int64Data;
    std::vector<int64_t> outDims;
};

// One compiled-in device layer and whether it is usable right now.
struct DeviceSlot {
    std::string name;
    nert::Device* device = nullptr;
    bool present = false;
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

// A device is usable when its allocator hands out one byte. Constructing the
// device or allocating may throw (no platform/driver): both mean SKIP.
bool ProbeDevice(nert::Device& device) {
    try {
        void* probe = device.allocator().allocate(1);
        if (probe == nullptr) {
            return false;
        }
        device.allocator().release(probe);
        return true;
    } catch (...) {
        return false;
    }
}

template <typename Factory> void AddDevice(std::vector<DeviceSlot>& slots, const char* name, Factory factory) {
    DeviceSlot slot;
    slot.name = name;
    try {
        slot.device = &const_cast<nert::Device&>(factory());
        slot.present = ProbeDevice(*slot.device);
    } catch (...) {
        slot.present = false;
    }
    slots.push_back(std::move(slot));
}

std::vector<DeviceSlot> EnumerateDevices() {
    std::vector<DeviceSlot> slots;
#ifdef NERT_BENCH_CUDA
    AddDevice(slots, "CUDA", []() -> const nert::Device& { return nert::CudaDevice(); });
#endif
#ifdef NERT_BENCH_OPENCL
    AddDevice(slots, "OpenCL", []() -> const nert::Device& { return nert::OpenCLDevice(); });
#endif
#ifdef NERT_BENCH_HIP
    AddDevice(slots, "HIP", []() -> const nert::Device& { return nert::HipDevice(); });
#endif
    return slots;
}

// Runs one case on one device against the already-computed CPU reference.
void RunDeviceCase(const DeviceSlot& slot, const Case& c, const std::vector<float>& ref, double cpuMs, int& passed,
                   int& skipped, int& failed) {
    if (!slot.present) {
        std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << std::setw(9) << slot.name
                  << "SKIP: device not present\n";
        ++skipped;
        return;
    }
    nert::Device& device = *slot.device;
    nert::KernelFn deviceFn = device.kernels().find(c.op);
    if (deviceFn == nullptr) {
        std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << std::setw(9) << slot.name
                  << "SKIP: no " << slot.name << " row\n";
        ++skipped;
        return;
    }
    std::vector<void*> devicePtrs;
    std::vector<nert::TensorView> deviceIns;
    for (std::size_t i = 0; i < c.inData.size(); ++i) {
        const bool as_int64 = i < c.int64Data.size() && !c.int64Data[i].empty();
        const int64_t bytes = as_int64 ? static_cast<int64_t>(c.int64Data[i].size() * sizeof(int64_t))
                                       : static_cast<int64_t>(c.inData[i].size() * sizeof(float));
        void* deviceBuffer = device.allocator().allocate(bytes);
        if (deviceBuffer == nullptr) {
            std::cout << "FAIL: device allocate failed for " << c.op << " " << c.label << '\n';
            ++failed;
            return;
        }
        if (as_int64) {
            device.transfer().host_to_device(deviceBuffer, c.int64Data[i].data(), bytes);
        } else {
            device.transfer().host_to_device(deviceBuffer, c.inData[i].data(), bytes);
        }
        devicePtrs.push_back(deviceBuffer);
        nert::TensorView v;
        v.data = deviceBuffer;
        v.dtype = as_int64 ? nert::DataType::Int64 : nert::DataType::Float32;
        v.dims = as_int64 ? c.int64Dims[i] : c.inDims[i];
        deviceIns.push_back(v);
    }
    std::vector<float> zeros(static_cast<std::size_t>(Count(c.outDims)), 0.0F);
    const int64_t outBytes = static_cast<int64_t>(zeros.size() * sizeof(float));
    void* deviceOut = device.allocator().allocate(outBytes);
    if (deviceOut == nullptr) {
        std::cout << "FAIL: device allocate failed for " << c.op << " " << c.label << '\n';
        for (void* buffer : devicePtrs) {
            device.allocator().release(buffer);
        }
        ++failed;
        return;
    }
    nert::TensorView outView;
    outView.data = deviceOut;
    outView.dtype = nert::DataType::Float32;
    outView.dims = c.outDims;
    for (int i = 0; i < 3; ++i) {
        device.transfer().host_to_device(deviceOut, zeros.data(), outBytes);
        deviceFn(c.node, deviceIns, {outView});
    }
    const double deviceMs = TimeMs(
        [&] {
            device.transfer().host_to_device(deviceOut, zeros.data(), outBytes);
            deviceFn(c.node, deviceIns, {outView});
        },
        10);
    std::vector<float> got(zeros.size(), 0.0F);
    device.transfer().device_to_host(got.data(), deviceOut, outBytes);
    for (void* buffer : devicePtrs) {
        device.allocator().release(buffer);
    }
    device.allocator().release(deviceOut);

    double maxDiff = 0.0;
    int mismatches = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        maxDiff = std::max(maxDiff, diff);
        if (diff > kTol * (1.0 + std::fabs(static_cast<double>(ref[i])))) {
            ++mismatches;
        }
    }
    const double speedup = deviceMs > 0.0 ? cpuMs / deviceMs : 0.0;
    std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << std::setw(9) << slot.name
              << std::right << std::fixed << std::setprecision(2) << std::setw(10) << cpuMs << std::setw(10) << deviceMs
              << std::setw(9) << std::setprecision(1) << speedup << "x" << std::setw(12) << std::scientific
              << std::setprecision(2) << maxDiff << "  " << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
    if (mismatches == 0) {
        ++passed;
        return;
    }
    ++failed;
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
        c.op = "Reshape";
        c.label = Shape({2, 3, 4}) + " -> " + Shape({6, 4});
        c.node = MakeNode("bench_reshape", "Reshape");
        c.inDims = {{2, 3, 4}, {}};
        c.inData = {Fill(rng, 24, -1.0F, 1.0F), {}};
        c.int64Dims = {{}, {2}};
        c.int64Data = {{}, {6, 4}};
        c.outDims = {6, 4};
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

    std::vector<DeviceSlot> devices = EnumerateDevices();
    for (const DeviceSlot& slot : devices) {
        if (!slot.present) {
            std::cout << "SKIP: no " << slot.name << " device (its halves skipped, others still run)\n";
        }
    }
    std::cout << std::left << std::setw(12) << "op" << std::setw(28) << "case" << std::setw(9) << "device" << std::right
              << std::setw(10) << "cpu_ms" << std::setw(10) << "dev_ms" << std::setw(10) << "speedup" << std::setw(12)
              << "maxdiff" << '\n';

    int passed = 0, skipped = 0, failed = 0;
    for (const Case& c : cases) {
        nert::KernelFn cpuFn = nert::CpuDevice().kernels().find(c.op);
        if (cpuFn == nullptr) {
            std::cout << std::left << std::setw(12) << c.op << std::setw(28) << c.label << "NO CPU ORACLE  FAIL\n";
            ++failed;
            continue;
        }
        std::vector<nert::TensorView> cpuIns;
        for (std::size_t i = 0; i < c.inData.size(); ++i) {
            const bool as_int64 = i < c.int64Data.size() && !c.int64Data[i].empty();
            nert::TensorView v;
            v.data = as_int64 ? static_cast<void*>(const_cast<int64_t*>(c.int64Data[i].data()))
                              : static_cast<void*>(const_cast<float*>(c.inData[i].data()));
            v.dtype = as_int64 ? nert::DataType::Int64 : nert::DataType::Float32;
            v.dims = as_int64 ? c.int64Dims[i] : c.inDims[i];
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

        for (DeviceSlot& slot : devices) {
            RunDeviceCase(slot, c, ref, cpuMs, passed, skipped, failed);
        }
    }
    std::cout << (failed == 0 ? "PASSED " : "FAILED ") << passed << " cases (" << skipped << " skipped, " << failed
              << " failed)\n";
    return failed == 0 ? 0 : 1;
}
