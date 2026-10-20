// Runtime bench (N1f/N1d/N1c): parity + timing for every shipped kernel.
//
// This driver implements ZERO kernel logic: it builds one fixture per op
// (shapes + attrs + deterministic data) and runs the runtime's own tables —
// CpuDevice (oracle) vs every device layer this binary was built with
// (CUDA, OpenCL, HIP) — through the public seam. Adding kernel coverage
// means adding a fixture, never reimplementing.
//
// Fixtures carry float32, int64, and bool tensors (shape/axes operands,
// comparison/logic outputs, ArgMax indices). Float rows compare within
// 1e-3*(1+|ref|); int64/bool rows are exact and report maxdiff 0 or 1.
//
// Exit 0 + PASSED when every executed case matches; a device that is not
// compiled in is absent from the binary entirely, and one that is compiled
// in but missing at run time (or lacks a row for an op) prints SKIP: and
// still exits 0. Any mismatch exits 1. Timings are steady_clock around the
// (internally synchronized) kernel calls: honest end-to-end ms.

#include "nert/device.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kTol = 1e-3;

using Bytes = std::vector<std::uint8_t>;

struct Buffer {
    nert::DataType dtype = nert::DataType::Float32;
    std::vector<int64_t> dims;
    Bytes bytes;
};

struct Case {
    std::string op;
    std::string label;
    nert::Node node;
    std::vector<Buffer> inputs;
    Buffer output;
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

std::size_t ElemSize(nert::DataType dtype) {
    switch (dtype) {
    case nert::DataType::Float32:
        return sizeof(float);
    case nert::DataType::Int64:
        return sizeof(int64_t);
    case nert::DataType::Bool:
        return sizeof(std::uint8_t);
    default:
        return 0;
    }
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

std::vector<float> Fill(std::mt19937_64& rng, std::size_t count, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> data(count);
    for (float& v : data) {
        v = dist(rng);
    }
    return data;
}

std::vector<std::uint8_t> RandomBool(std::mt19937_64& rng, std::size_t count) {
    std::vector<std::uint8_t> data(count);
    for (std::uint8_t& v : data) {
        v = static_cast<std::uint8_t>(rng() & 1U);
    }
    return data;
}

template <typename T> Bytes ToBytes(const std::vector<T>& values) {
    Bytes bytes(values.size() * sizeof(T));
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

Buffer FloatBuffer(std::vector<float> values, std::vector<int64_t> dims) {
    Buffer b;
    b.dtype = nert::DataType::Float32;
    b.dims = std::move(dims);
    b.bytes = ToBytes(values);
    return b;
}

Buffer Int64Buffer(std::vector<int64_t> values, std::vector<int64_t> dims) {
    Buffer b;
    b.dtype = nert::DataType::Int64;
    b.dims = std::move(dims);
    b.bytes = ToBytes(values);
    return b;
}

Buffer BoolBuffer(std::vector<std::uint8_t> values, std::vector<int64_t> dims) {
    Buffer b;
    b.dtype = nert::DataType::Bool;
    b.dims = std::move(dims);
    b.bytes = ToBytes(values);
    return b;
}

Buffer OutputBuffer(nert::DataType dtype, std::vector<int64_t> dims) {
    Buffer b;
    b.dtype = dtype;
    b.dims = std::move(dims);
    return b;
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

nert::TensorView ViewOf(const Buffer& buffer) {
    nert::TensorView view;
    view.data = const_cast<std::uint8_t*>(buffer.bytes.data());
    view.dtype = buffer.dtype;
    view.dims = buffer.dims;
    return view;
}

// Compares got vs ref under the dtype's rule. Float: 1e-3*(1+|ref|) and the
// largest absolute difference. Int64/bool: exact; maxDiff is 0 or 1.
void Compare(const Bytes& got, const Bytes& ref, nert::DataType dtype, std::size_t count, double& maxDiff,
             int& mismatches) {
    maxDiff = 0.0;
    mismatches = 0;
    if (dtype == nert::DataType::Float32) {
        const float* g = reinterpret_cast<const float*>(got.data());
        const float* r = reinterpret_cast<const float*>(ref.data());
        for (std::size_t i = 0; i < count; ++i) {
            const double diff = std::fabs(static_cast<double>(g[i]) - static_cast<double>(r[i]));
            maxDiff = std::max(maxDiff, diff);
            if (diff > kTol * (1.0 + std::fabs(static_cast<double>(r[i])))) {
                ++mismatches;
            }
        }
    } else if (dtype == nert::DataType::Int64) {
        const int64_t* g = reinterpret_cast<const int64_t*>(got.data());
        const int64_t* r = reinterpret_cast<const int64_t*>(ref.data());
        for (std::size_t i = 0; i < count; ++i) {
            if (g[i] != r[i]) {
                ++mismatches;
            }
        }
        maxDiff = mismatches == 0 ? 0.0 : 1.0;
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            if (got[i] != ref[i]) {
                ++mismatches;
            }
        }
        maxDiff = mismatches == 0 ? 0.0 : 1.0;
    }
}

// Runs one case on one device against the already-computed CPU reference.
void RunDeviceCase(const DeviceSlot& slot, const Case& c, const Bytes& ref, double cpuMs, int& passed, int& skipped,
                   int& failed) {
    if (!slot.present) {
        std::cout << std::left << std::setw(20) << c.op << std::setw(28) << c.label << std::setw(9) << slot.name
                  << "SKIP: device not present\n";
        ++skipped;
        return;
    }
    nert::Device& device = *slot.device;
    nert::KernelFn deviceFn = device.kernels().find(c.op);
    if (deviceFn == nullptr) {
        std::cout << std::left << std::setw(20) << c.op << std::setw(28) << c.label << std::setw(9) << slot.name
                  << "SKIP: no " << slot.name << " row\n";
        ++skipped;
        return;
    }
    std::vector<void*> devicePtrs;
    std::vector<nert::TensorView> deviceIns;
    for (const Buffer& input : c.inputs) {
        const int64_t bytes = static_cast<int64_t>(input.bytes.size());
        void* deviceBuffer = device.allocator().allocate(bytes);
        if (deviceBuffer == nullptr) {
            std::cout << "FAIL: device allocate failed for " << c.op << " " << c.label << '\n';
            ++failed;
            return;
        }
        device.transfer().host_to_device(deviceBuffer, input.bytes.data(), bytes);
        devicePtrs.push_back(deviceBuffer);
        deviceIns.push_back(ViewOf(input));
        deviceIns.back().data = deviceBuffer;
    }
    const std::size_t outElems = static_cast<std::size_t>(Count(c.output.dims));
    const std::size_t outBytes = outElems * ElemSize(c.output.dtype);
    Bytes zeros(outBytes, 0);
    void* deviceOut = device.allocator().allocate(static_cast<int64_t>(outBytes));
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
    outView.dtype = c.output.dtype;
    outView.dims = c.output.dims;
    for (int i = 0; i < 3; ++i) {
        device.transfer().host_to_device(deviceOut, zeros.data(), static_cast<int64_t>(outBytes));
        deviceFn(c.node, deviceIns, {outView});
    }
    const double deviceMs = TimeMs(
        [&] {
            device.transfer().host_to_device(deviceOut, zeros.data(), static_cast<int64_t>(outBytes));
            deviceFn(c.node, deviceIns, {outView});
        },
        10);
    Bytes got(outBytes, 0);
    device.transfer().device_to_host(got.data(), deviceOut, static_cast<int64_t>(outBytes));
    for (void* buffer : devicePtrs) {
        device.allocator().release(buffer);
    }
    device.allocator().release(deviceOut);

    double maxDiff = 0.0;
    int mismatches = 0;
    Compare(got, ref, c.output.dtype, outElems, maxDiff, mismatches);
    const double speedup = deviceMs > 0.0 ? cpuMs / deviceMs : 0.0;
    std::cout << std::left << std::setw(20) << c.op << std::setw(28) << c.label << std::setw(9) << slot.name
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
    const int64_t kNew = 1024 * 1024;
    std::vector<Case> cases;

    {
        Case c;
        c.op = "Gemm";
        c.label = Shape({512, 512}) + "x" + Shape({512, 512});
        c.node = MakeNode("bench_gemm", "Gemm");
        c.inputs = {FloatBuffer(Fill(rng, 512 * 512, -1.0F, 1.0F), {512, 512}),
                    FloatBuffer(Fill(rng, 512 * 512, -1.0F, 1.0F), {512, 512})};
        c.output = OutputBuffer(nert::DataType::Float32, {512, 512});
        cases.push_back(std::move(c));
    }
    for (const char* op : {"Relu", "Sigmoid", "Tanh"}) {
        Case c;
        c.op = op;
        c.label = Shape({kBig});
        c.node = MakeNode("bench_unary", op);
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kBig), -6.0F, 6.0F), {kBig})};
        c.output = OutputBuffer(nert::DataType::Float32, {kBig});
        cases.push_back(std::move(c));
    }
    for (const char* op : {"Add", "Mul", "Sub"}) {
        Case c;
        c.op = op;
        c.label = Shape({2048, 2048}) + "+" + Shape({2048});
        c.node = MakeNode("bench_bin", op);
        c.inputs = {FloatBuffer(Fill(rng, 2048 * 2048, -3.0F, 3.0F), {2048, 2048}),
                    FloatBuffer(Fill(rng, 2048, -3.0F, 3.0F), {2048})};
        c.output = OutputBuffer(nert::DataType::Float32, {2048, 2048});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Div";
        c.label = Shape({kBig});
        c.node = MakeNode("bench_div", "Div");
        std::vector<float> b = Fill(rng, static_cast<std::size_t>(kBig), 0.5F, 2.0F);
        for (float& v : b) {
            if ((rng() & 1) != 0) {
                v = -v;
            }
        }
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kBig), -3.0F, 3.0F), {kBig}),
                    FloatBuffer(std::move(b), {kBig})};
        c.output = OutputBuffer(nert::DataType::Float32, {kBig});
        cases.push_back(std::move(c));
    }
    for (const char* op : {"MaxPool", "AveragePool"}) {
        Case c;
        c.op = op;
        c.label = Shape({8, 64, 112, 112}) + " k3 s2";
        c.node = MakeNode("bench_pool", op);
        c.node.attributes.emplace_back("kernel_shape", std::vector<int64_t>{3, 3});
        c.node.attributes.emplace_back("strides", std::vector<int64_t>{2, 2});
        c.inputs = {FloatBuffer(Fill(rng, 8 * 64 * 112 * 112, -3.0F, 3.0F), {8, 64, 112, 112})};
        c.output = OutputBuffer(nert::DataType::Float32, {8, 64, 55, 55});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Softmax";
        c.label = Shape({256, 4096});
        c.node = MakeNode("bench_softmax", "Softmax");
        c.inputs = {FloatBuffer(Fill(rng, 256 * 4096, -3.0F, 3.0F), {256, 4096})};
        c.output = OutputBuffer(nert::DataType::Float32, {256, 4096});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Reshape";
        c.label = Shape({2, 3, 4}) + " -> " + Shape({6, 4});
        c.node = MakeNode("bench_reshape", "Reshape");
        c.inputs = {FloatBuffer(Fill(rng, 24, -1.0F, 1.0F), {2, 3, 4}), Int64Buffer({6, 4}, {2})};
        c.output = OutputBuffer(nert::DataType::Float32, {6, 4});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Conv";
        c.label = Shape({8, 16, 64, 64}) + " k3+bias";
        c.node = MakeNode("bench_conv", "Conv");
        c.inputs = {FloatBuffer(Fill(rng, 8 * 16 * 64 * 64, -1.0F, 1.0F), {8, 16, 64, 64}),
                    FloatBuffer(Fill(rng, 32 * 16 * 3 * 3, -1.0F, 1.0F), {32, 16, 3, 3}),
                    FloatBuffer(Fill(rng, 32, -1.0F, 1.0F), {32})};
        c.output = OutputBuffer(nert::DataType::Float32, {8, 32, 62, 62});
        cases.push_back(std::move(c));
    }

    // R6 / Tier-5 rows.
    struct UnarySpec {
        const char* op;
        float lo;
        float hi;
    };
    const UnarySpec unary[] = {
        {"Sin", -6.0F, 6.0F},   {"Cos", -6.0F, 6.0F}, {"Exp", -6.0F, 3.0F}, {"Elu", -6.0F, 6.0F},  {"Log", 0.1F, 6.0F},
        {"Floor", -6.0F, 6.0F}, {"Abs", -6.0F, 6.0F}, {"Neg", -6.0F, 6.0F}, {"Sqrt", 0.0F, 16.0F}, {"Erf", -4.0F, 4.0F},
    };
    for (const UnarySpec& spec : unary) {
        Case c;
        c.op = spec.op;
        c.label = Shape({kNew});
        c.node = MakeNode("bench_unary2", spec.op);
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), spec.lo, spec.hi), {kNew})};
        c.output = OutputBuffer(nert::DataType::Float32, {kNew});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "IsNaN";
        c.label = Shape({kNew});
        c.node = MakeNode("bench_isnan", "IsNaN");
        std::vector<float> x = Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F);
        x[17] = std::numeric_limits<float>::quiet_NaN();
        x[9001] = -std::numeric_limits<float>::quiet_NaN();
        c.inputs = {FloatBuffer(std::move(x), {kNew})};
        c.output = OutputBuffer(nert::DataType::Bool, {kNew});
        cases.push_back(std::move(c));
    }
    for (const char* op : {"Greater", "Less", "GreaterOrEqual", "LessOrEqual"}) {
        Case c;
        c.op = op;
        c.label = Shape({1024, 1024}) + " vs " + Shape({1024});
        c.node = MakeNode("bench_cmp", op);
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024}),
                    FloatBuffer(Fill(rng, 1024, -3.0F, 3.0F), {1024})};
        c.output = OutputBuffer(nert::DataType::Bool, {1024, 1024});
        cases.push_back(std::move(c));
    }
    for (const char* op : {"And", "Or"}) {
        Case c;
        c.op = op;
        c.label = Shape({1024, 1024}) + " bcast";
        c.node = MakeNode("bench_logic", op);
        c.inputs = {BoolBuffer(RandomBool(rng, static_cast<std::size_t>(kNew)), {1024, 1024}),
                    BoolBuffer(RandomBool(rng, 1024), {1024})};
        c.output = OutputBuffer(nert::DataType::Bool, {1024, 1024});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Not";
        c.label = Shape({kNew});
        c.node = MakeNode("bench_not", "Not");
        c.inputs = {BoolBuffer(RandomBool(rng, static_cast<std::size_t>(kNew)), {kNew})};
        c.output = OutputBuffer(nert::DataType::Bool, {kNew});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Pow";
        c.label = Shape({1024, 1024}) + "^" + Shape({1024});
        c.node = MakeNode("bench_pow", "Pow");
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), 0.5F, 2.0F), {1024, 1024}),
                    FloatBuffer(Fill(rng, 1024, -1.0F, 1.0F), {1024})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024, 1024});
        cases.push_back(std::move(c));
    }
    for (const char* op : {"Max", "Min"}) {
        Case c;
        c.op = op;
        c.label = Shape({1024, 1024}) + " fold3";
        c.node = MakeNode("bench_fold", op);
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024}),
                    FloatBuffer(Fill(rng, 1024, -3.0F, 3.0F), {1024}), FloatBuffer(Fill(rng, 1, -3.0F, 3.0F), {1})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024, 1024});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Clip";
        c.label = Shape({kNew}) + " [0,1]";
        c.node = MakeNode("bench_clip", "Clip");
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -5.0F, 5.0F), {kNew}),
                    FloatBuffer({0.0F}, {1}), FloatBuffer({1.0F}, {1})};
        c.output = OutputBuffer(nert::DataType::Float32, {kNew});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Identity";
        c.label = Shape({kNew});
        c.node = MakeNode("bench_identity", "Identity");
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {kNew})};
        c.output = OutputBuffer(nert::DataType::Float32, {kNew});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Squeeze";
        c.label = Shape({1, 1024, 1}) + " -> " + Shape({1024});
        c.node = MakeNode("bench_squeeze", "Squeeze");
        c.inputs = {FloatBuffer(Fill(rng, 1024, -3.0F, 3.0F), {1, 1024, 1})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Tile";
        c.label = Shape({1024, 1}) + " x{2,3}";
        c.node = MakeNode("bench_tile", "Tile");
        c.inputs = {FloatBuffer(Fill(rng, 1024, -3.0F, 3.0F), {1024, 1}), Int64Buffer({2, 3}, {2})};
        c.output = OutputBuffer(nert::DataType::Float32, {2048, 3});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Range";
        c.label = "0..1M step 1";
        c.node = MakeNode("bench_range", "Range");
        c.inputs = {FloatBuffer({0.0F}, {1}), FloatBuffer({static_cast<float>(kNew)}, {1}), FloatBuffer({1.0F}, {1})};
        c.output = OutputBuffer(nert::DataType::Float32, {kNew});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "EyeLike";
        c.label = Shape({1024, 1024});
        c.node = MakeNode("bench_eye", "EyeLike");
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -1.0F, 1.0F), {1024, 1024})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024, 1024});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "Pad";
        c.label = Shape({1024, 1024}) + " +1";
        c.node = MakeNode("bench_pad", "Pad");
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024}),
                    Int64Buffer({1, 1, 1, 1}, {4})};
        c.output = OutputBuffer(nert::DataType::Float32, {1026, 1026});
        cases.push_back(std::move(c));
    }
    for (const char* op : {"ReduceSum", "ReduceMean", "ReduceL2"}) {
        Case c;
        c.op = op;
        c.label = Shape({1024, 1024}) + " axis1";
        c.node = MakeNode("bench_reduce", op);
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024}),
                    Int64Buffer({1}, {1})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024, 1});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "ArgMax";
        c.label = Shape({1024, 1024}) + " axis1";
        c.node = MakeNode("bench_argmax", "ArgMax");
        c.node.attributes.emplace_back("axis", int64_t{1});
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024})};
        c.output = OutputBuffer(nert::DataType::Int64, {1024, 1});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "CumSum";
        c.label = Shape({1024, 1024}) + " axis1";
        c.node = MakeNode("bench_cumsum", "CumSum");
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024}),
                    Int64Buffer({1}, {1})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024, 1024});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "ScatterND";
        c.label = Shape({1024, 1024}) + " rows";
        c.node = MakeNode("bench_scatter", "ScatterND");
        std::vector<int64_t> indices(1024);
        for (std::size_t i = 0; i < indices.size(); ++i) {
            indices[i] = static_cast<int64_t>(i);
        }
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024}),
                    Int64Buffer(std::move(indices), {1024, 1}),
                    FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024, 1024});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "LayerNormalization";
        c.label = Shape({1024, 1024}) + " axis1";
        c.node = MakeNode("bench_ln", "LayerNormalization");
        c.node.attributes.emplace_back("axis", int64_t{1});
        c.inputs = {FloatBuffer(Fill(rng, static_cast<std::size_t>(kNew), -3.0F, 3.0F), {1024, 1024}),
                    FloatBuffer(std::vector<float>(1024, 1.0F), {1024})};
        c.output = OutputBuffer(nert::DataType::Float32, {1024, 1024});
        cases.push_back(std::move(c));
    }
    {
        Case c;
        c.op = "GlobalAveragePool";
        c.label = Shape({8, 64, 112, 112});
        c.node = MakeNode("bench_gap", "GlobalAveragePool");
        c.inputs = {FloatBuffer(Fill(rng, 8 * 64 * 112 * 112, -3.0F, 3.0F), {8, 64, 112, 112})};
        c.output = OutputBuffer(nert::DataType::Float32, {8, 64, 1, 1});
        cases.push_back(std::move(c));
    }

    std::vector<DeviceSlot> devices = EnumerateDevices();
    for (const DeviceSlot& slot : devices) {
        if (!slot.present) {
            std::cout << "SKIP: no " << slot.name << " device (its halves skipped, others still run)\n";
        }
    }
    std::cout << std::left << std::setw(20) << "op" << std::setw(28) << "case" << std::setw(9) << "device" << std::right
              << std::setw(10) << "cpu_ms" << std::setw(10) << "dev_ms" << std::setw(10) << "speedup" << std::setw(12)
              << "maxdiff" << '\n';

    int passed = 0, skipped = 0, failed = 0;
    for (const Case& c : cases) {
        nert::KernelFn cpuFn = nert::CpuDevice().kernels().find(c.op);
        if (cpuFn == nullptr) {
            std::cout << std::left << std::setw(20) << c.op << std::setw(28) << c.label << "NO CPU ORACLE  FAIL\n";
            ++failed;
            continue;
        }
        std::vector<nert::TensorView> cpuIns;
        for (const Buffer& input : c.inputs) {
            cpuIns.push_back(ViewOf(input));
        }
        const std::size_t outElems = static_cast<std::size_t>(Count(c.output.dims));
        Bytes ref(outElems * ElemSize(c.output.dtype), 0);
        nert::TensorView refView;
        refView.data = ref.data();
        refView.dtype = c.output.dtype;
        refView.dims = c.output.dims;
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
