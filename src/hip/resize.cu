// Hipified from src/cuda/resize.cu via hipify-perl, hand-reviewed.
// Resize on HIP: nearest/linear resampling, one thread per output element.
//
// Mirrors the CPU oracle: float32 (int64 only for nearest), mode/transform/
// nearest_mode attrs, exactly one of scales/sizes, float-balance scale for
// the sizes path. The per-axis sample tables (lower/upper/double weight)
// build on the host exactly like the oracle's AxisTable and upload once;
// the kernel replays the tap expansion. Linear mode caps rank at 4 (2^rank
// taps in a static array); nearest has one tap per element at any rank ≤ 16.

#include "nert/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

constexpr std::size_t kMaxRank = 16;
constexpr std::size_t kMaxLinearRank = 4;
constexpr std::size_t kMaxTaps = 16;

void CheckHip(hipError_t status, const char* what, const std::string& context) {
    if (status != hipSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + hipGetErrorString(status));
    }
}

int64_t ElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim < 0) {
            throw InferenceException(context + ": negative dimension");
        }
        count *= dim;
    }
    return count;
}

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Resize: attribute '" + name + "' must be a string (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

// The input coordinate an output coordinate samples, per the ONNX transform.
double SourceCoordinate(const std::string& mode, int64_t out_coord, int64_t in_len, int64_t out_len, double scale,
                        const std::string& context) {
    const double x = static_cast<double>(out_coord);
    if (mode == "half_pixel") {
        return (x + 0.5) / scale - 0.5;
    }
    if (mode == "pytorch_half_pixel") {
        return out_len > 1 ? (x + 0.5) / scale - 0.5 : 0.0;
    }
    if (mode == "align_corners") {
        return out_len == 1 ? 0.0 : x * static_cast<double>(in_len - 1) / static_cast<double>(out_len - 1);
    }
    if (mode == "asymmetric") {
        return x / scale;
    }
    throw InferenceException(context + ": unsupported coordinate_transformation_mode '" + mode + "'");
}

int64_t NearestIndex(const std::string& mode, double x, int64_t in_len, const std::string& context) {
    int64_t index = 0;
    if (mode == "round_prefer_floor") {
        const double f = std::floor(x);
        index = static_cast<int64_t>(x - f == 0.5 ? f : std::round(x));
    } else if (mode == "round_prefer_ceil") {
        index = static_cast<int64_t>(std::round(x));
    } else if (mode == "floor") {
        index = static_cast<int64_t>(std::floor(x));
    } else if (mode == "ceil") {
        index = static_cast<int64_t>(std::ceil(x));
    } else {
        throw InferenceException(context + ": unsupported nearest_mode '" + mode + "'");
    }
    return std::min(std::max(index, int64_t{0}), in_len - 1);
}

struct AxisSample {
    int64_t lower = 0;
    int64_t upper = 0;
    double weight = 0.0;
};

std::vector<AxisSample> AxisTable(bool linear, const std::string& transform, const std::string& nearest, int64_t in_len,
                                  int64_t out_len, double scale, const std::string& context) {
    std::vector<AxisSample> table(static_cast<std::size_t>(out_len));
    for (int64_t o = 0; o < out_len; ++o) {
        const double x = SourceCoordinate(transform, o, in_len, out_len, scale, context);
        AxisSample& sample = table[static_cast<std::size_t>(o)];
        if (!linear) {
            sample.lower = sample.upper = NearestIndex(nearest, x, in_len, context);
            continue;
        }
        const double clamped = std::min(std::max(x, 0.0), static_cast<double>(in_len - 1));
        sample.lower = static_cast<int64_t>(std::floor(clamped));
        sample.upper = std::min(sample.lower + 1, in_len - 1);
        sample.weight = sample.upper == sample.lower ? 0.0 : clamped - static_cast<double>(sample.lower);
    }
    return table;
}

struct IndexSet {
    int64_t s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the resize limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = values[j];
    }
    return packed;
}

template <typename T>
__global__ void ResizeKernel(const T* x, T* y, int64_t outCount, IndexSet inStrides, IndexSet outStrides,
                             IndexSet axisOffset, int rank, const int64_t* lower, const int64_t* upper,
                             const double* weight) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t taps[kMaxTaps] = {0};
        double tapW[kMaxTaps] = {1.0};
        int tapCount = 1;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            const int64_t tab = axisOffset.s[j] + coord;
            const int64_t lo = lower[tab];
            const int64_t hi = upper[tab];
            const double w = weight[tab];
            if (w == 0.0) {
                for (int t = 0; t < tapCount; ++t) {
                    taps[t] += lo * inStrides.s[j];
                }
                continue;
            }
            int next = 0;
            int64_t taps2[kMaxTaps] = {0};
            double tapW2[kMaxTaps] = {0.0};
            for (int t = 0; t < tapCount; ++t) {
                taps2[next] = taps[t] + lo * inStrides.s[j];
                tapW2[next] = tapW[t] * (1.0 - w);
                ++next;
                taps2[next] = taps[t] + hi * inStrides.s[j];
                tapW2[next] = tapW[t] * w;
                ++next;
            }
            tapCount = next;
            for (int t = 0; t < tapCount; ++t) {
                taps[t] = taps2[t];
                tapW[t] = tapW2[t];
            }
        }
        if (tapCount == 1) {
            y[i] = x[taps[0]];
            continue;
        }
        double value = 0.0;
        for (int t = 0; t < tapCount; ++t) {
            value += tapW[t] * static_cast<double>(x[taps[t]]);
        }
        y[i] = static_cast<T>(value);
    }
}

} // namespace

void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Resize (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 4 || outputs.size() != 1) {
        throw InferenceException(context + ": expected an input, roi, scales, and an optional sizes input");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    const std::string mode = StringAttribute(node, "mode", "nearest");
    if (mode != "nearest" && mode != "linear") {
        throw InferenceException(context + ": Resize 'mode' must be 'nearest' or 'linear' here");
    }
    const bool linear = mode == "linear";
    if (in.dtype != DataType::Float32 && !(in.dtype == DataType::Int64 && !linear)) {
        throw InferenceException(context + ": tensors must be float32 (or int64 for nearest)");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const std::string transform = StringAttribute(node, "coordinate_transformation_mode", "half_pixel");
    const std::string nearest = StringAttribute(node, "nearest_mode", "round_prefer_floor");

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the resize limit of 16");
    }
    if (linear && static_cast<std::size_t>(rank) > kMaxLinearRank) {
        throw InferenceException(context + ": linear Resize supports at most 4 axes here");
    }
    std::vector<double> scale_vec(static_cast<std::size_t>(rank), 1.0);
    const bool has_scales =
        inputs.size() > 2 && inputs[2].dtype != DataType::Unknown && ElementCount(inputs[2].dims, context) > 0;
    const bool has_sizes = inputs.size() > 3 && inputs[3].dtype != DataType::Unknown;
    if (has_sizes && has_scales) {
        throw InferenceException(context + ": Resize takes exactly one of scales and sizes");
    }
    if (has_sizes) {
        // A sizes input fixes the output dims; the per-axis scale used by the
        // coordinate transform is then out / in, as ONNX defines it.
        const TensorView& sizes_view = inputs[3];
        if (sizes_view.dtype != DataType::Int64 || ElementCount(sizes_view.dims, context) != rank ||
            sizes_view.data == nullptr) {
            throw InferenceException(context + ": Resize sizes must be int64 with one entry per input axis");
        }
        std::vector<int64_t> sizes(static_cast<std::size_t>(rank), 0);
        CheckHip(hipMemcpy(sizes.data(), sizes_view.data, static_cast<std::size_t>(rank) * sizeof(int64_t),
                           hipMemcpyDeviceToHost),
                 "hipMemcpy", context);
        for (int64_t i = 0; i < rank; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            if (out.dims.size() != static_cast<std::size_t>(rank) || out.dims[ui] != sizes[ui]) {
                throw InferenceException(context + ": output shape does not match the sizes input");
            }
            scale_vec[ui] = static_cast<double>(static_cast<float>(sizes[ui]) / static_cast<float>(in.dims[ui]));
        }
    } else {
        if (!has_scales || inputs[2].dtype != DataType::Float32) {
            throw InferenceException(context + ": Resize requires a scales or sizes input");
        }
        const TensorView& scales_view = inputs[2];
        if (ElementCount(scales_view.dims, context) != rank) {
            throw InferenceException(context + ": Resize scales rank must match the input rank");
        }
        if (scales_view.data == nullptr) {
            throw InferenceException(context + ": null scales buffer");
        }
        std::vector<float> scales(static_cast<std::size_t>(rank), 0.0F);
        CheckHip(hipMemcpy(scales.data(), scales_view.data, static_cast<std::size_t>(rank) * sizeof(float),
                           hipMemcpyDeviceToHost),
                 "hipMemcpy", context);
        std::vector<int64_t> expected(static_cast<std::size_t>(rank), 0);
        for (int64_t i = 0; i < rank; ++i) {
            const std::size_t ui = static_cast<std::size_t>(i);
            const int64_t dim =
                static_cast<int64_t>(std::floor(static_cast<double>(in.dims[ui]) * static_cast<double>(scales[ui])));
            if (dim <= 0) {
                throw InferenceException(context + ": Resize produces a non-positive output dim");
            }
            expected[ui] = dim;
            scale_vec[ui] = static_cast<double>(scales[ui]);
        }
        if (out.dims != expected) {
            throw InferenceException(context + ": output shape does not match the scaled input");
        }
    }

    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    std::vector<int64_t> lowerAll;
    std::vector<int64_t> upperAll;
    std::vector<double> weightAll;
    std::vector<int64_t> axisOffset;
    for (int64_t i = 0; i < rank; ++i) {
        const std::size_t ui = static_cast<std::size_t>(i);
        axisOffset.push_back(static_cast<int64_t>(lowerAll.size()));
        const std::vector<AxisSample> table =
            AxisTable(linear, transform, nearest, in.dims[ui], out.dims[ui], scale_vec[ui], context);
        for (const AxisSample& s : table) {
            lowerAll.push_back(s.lower);
            upperAll.push_back(s.upper);
            weightAll.push_back(s.weight);
        }
    }
    int64_t* dLower = nullptr;
    int64_t* dUpper = nullptr;
    double* dWeight = nullptr;
    CheckHip(hipMalloc(&dLower, lowerAll.size() * sizeof(int64_t)), "hipMalloc", context);
    CheckHip(hipMalloc(&dUpper, upperAll.size() * sizeof(int64_t)), "hipMalloc", context);
    CheckHip(hipMalloc(&dWeight, weightAll.size() * sizeof(double)), "hipMalloc", context);
    CheckHip(hipMemcpy(dLower, lowerAll.data(), lowerAll.size() * sizeof(int64_t), hipMemcpyHostToDevice), "hipMemcpy",
             context);
    CheckHip(hipMemcpy(dUpper, upperAll.data(), upperAll.size() * sizeof(int64_t), hipMemcpyHostToDevice), "hipMemcpy",
             context);
    CheckHip(hipMemcpy(dWeight, weightAll.data(), weightAll.size() * sizeof(double), hipMemcpyHostToDevice),
             "hipMemcpy", context);

    const std::vector<int64_t> in_strides = Strides(in.dims);
    const std::vector<int64_t> out_strides = Strides(out.dims);
    const int block = 256;
    const int grid = static_cast<int>((out_count + block - 1) / block);
    if (in.dtype == DataType::Float32) {
        ResizeKernel<<<grid, block>>>(static_cast<const float*>(in.data), static_cast<float*>(out.data), out_count,
                                      Pack(in_strides, context), Pack(out_strides, context), Pack(axisOffset, context),
                                      static_cast<int>(rank), dLower, dUpper, dWeight);
    } else {
        ResizeKernel<<<grid, block>>>(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), out_count,
                                      Pack(in_strides, context), Pack(out_strides, context), Pack(axisOffset, context),
                                      static_cast<int>(rank), dLower, dUpper, dWeight);
    }
    CheckHip(hipGetLastError(), "ResizeKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
    CheckHip(hipFree(dLower), "hipFree", context);
    CheckHip(hipFree(dUpper), "hipFree", context);
    CheckHip(hipFree(dWeight), "hipFree", context);
}

} // namespace hip_kernels
} // namespace nert
