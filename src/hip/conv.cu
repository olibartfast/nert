// Conv on HIP (first-party, no library): the ONNX cross-correlation over one,
// two, or three spatial axes, with groups, strides, dilations, and explicit
// or automatic padding.
//
// Mirrors the CPU oracle's validation (group/channel checks, stride/dilation
// counts, auto_pad/pads resolution, the lifted 1-D/2-D-as-3-D geometry) with
// a naive one-thread-per-output kernel and double accumulation, so each
// output rounds once to float32 like the oracle. No MIOpen dependency, so
// the row runs on both the NVIDIA and AMD HIP backends.

#include "nert/device.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

void CheckHip(hipError_t status, const char* what, const std::string& context) {
    if (status != hipSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + hipGetErrorString(status));
    }
}

int64_t ElementCount(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0) {
            throw InferenceException(context + ": non-positive dimension");
        }
        count *= dim;
    }
    return count;
}

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

std::vector<int64_t> IntListAttribute(const Node& node, const std::string& name, const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::vector<int64_t>* value = std::get_if<std::vector<int64_t>>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be an int list (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return {};
}

std::string StringAttribute(const Node& node, const std::string& name, const std::string& fallback,
                            const std::string& op) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(op + ": attribute '" + name + "' must be a string (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

// Resolves per-spatial-axis begin/end padding from auto_pad or the explicit
// pads list. Mirrors the static shape inference, so the two never disagree.
void ResolvePads(const std::string& context, const std::string& auto_pad, const std::vector<int64_t>& in_spatial,
                 const std::vector<int64_t>& kernel, const std::vector<int64_t>& dilations,
                 const std::vector<int64_t>& strides, const std::vector<int64_t>& pads_attr,
                 std::vector<int64_t>& pad_begin, std::vector<int64_t>& pad_end) {
    const std::size_t rank = in_spatial.size();
    pad_begin.assign(rank, 0);
    pad_end.assign(rank, 0);

    if (auto_pad == "VALID") {
        return;
    }
    if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        for (std::size_t i = 0; i < rank; ++i) {
            const int64_t out = (in_spatial[i] + strides[i] - 1) / strides[i];
            const int64_t effective = dilations[i] * (kernel[i] - 1) + 1;
            int64_t total = (out - 1) * strides[i] + effective - in_spatial[i];
            if (total < 0) {
                total = 0;
            }
            if (auto_pad == "SAME_UPPER") {
                pad_begin[i] = total / 2;
                pad_end[i] = total - pad_begin[i];
            } else {
                pad_end[i] = total / 2;
                pad_begin[i] = total - pad_end[i];
            }
        }
        return;
    }

    // NOTSET (or an empty value): use the explicit pads, defaulting to zero.
    if (pads_attr.empty()) {
        return;
    }
    if (pads_attr.size() != 2 * rank) {
        throw InferenceException(context + ": pads must have two entries per spatial dimension");
    }
    for (std::size_t i = 0; i < rank; ++i) {
        pad_begin[i] = pads_attr[i];
        pad_end[i] = pads_attr[rank + i];
    }
}

// One spatial axis of the output window, matching the shape-inference formula.
int64_t WindowOutput(const std::string& context, int64_t in, int64_t kernel, int64_t dilation, int64_t pad_begin,
                     int64_t pad_end, int64_t stride) {
    const int64_t effective = dilation * (kernel - 1) + 1;
    const int64_t numer = in + pad_begin + pad_end - effective;
    if (numer < 0) {
        throw InferenceException(context + ": window is larger than the padded input");
    }
    const int64_t out = numer / stride + 1;
    if (out <= 0) {
        throw InferenceException(context + ": computes a non-positive output dimension");
    }
    return out;
}

struct ConvParams {
    int64_t n_batch;
    int64_t in_channels;
    int64_t out_channels;
    int64_t channels_per_group;
    int64_t outputs_per_group;
    int has_bias;
    int64_t in3[3];
    int64_t k3[3];
    int64_t o3[3];
    int64_t s3[3];
    int64_t d3[3];
    int64_t p3[3];
    int64_t in_plane;
    int64_t k_plane;
    int64_t out_plane;
};

__global__ void ConvKernel(const float* x, const float* w, const float* bias, float* y, int64_t count, ConvParams p) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int64_t rem = i;
        const int64_t ow = rem % p.o3[2];
        rem /= p.o3[2];
        const int64_t oh = rem % p.o3[1];
        rem /= p.o3[1];
        const int64_t od = rem % p.o3[0];
        rem /= p.o3[0];
        const int64_t m = rem % p.out_channels;
        const int64_t n = rem / p.out_channels;
        const int64_t channelBase = (m / p.outputs_per_group) * p.channels_per_group;
        double acc = p.has_bias != 0 ? static_cast<double>(bias[m]) : 0.0;
        for (int64_t kc = 0; kc < p.channels_per_group; ++kc) {
            const float* xChannel = x + (n * p.in_channels + channelBase + kc) * p.in_plane;
            const float* wChannel = w + (m * p.channels_per_group + kc) * p.k_plane;
            for (int64_t kd = 0; kd < p.k3[0]; ++kd) {
                const int64_t id = od * p.s3[0] - p.p3[0] + kd * p.d3[0];
                if (id < 0 || id >= p.in3[0]) {
                    continue;
                }
                for (int64_t kh = 0; kh < p.k3[1]; ++kh) {
                    const int64_t ih = oh * p.s3[1] - p.p3[1] + kh * p.d3[1];
                    if (ih < 0 || ih >= p.in3[1]) {
                        continue;
                    }
                    const float* xRow = xChannel + (id * p.in3[1] + ih) * p.in3[2];
                    const float* wRow = wChannel + (kd * p.k3[1] + kh) * p.k3[2];
                    for (int64_t kw = 0; kw < p.k3[2]; ++kw) {
                        const int64_t iw = ow * p.s3[2] - p.p3[2] + kw * p.d3[2];
                        if (iw < 0 || iw >= p.in3[2]) {
                            continue;
                        }
                        acc += static_cast<double>(xRow[iw]) * static_cast<double>(wRow[kw]);
                    }
                }
            }
        }
        y[i] = static_cast<float>(acc);
    }
}

} // namespace

void Conv(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Conv (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two or three inputs and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView& out = outputs[0];
    const bool hasBias = inputs.size() == 3;
    if (x.dtype != DataType::Float32 || w.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (hasBias && inputs[2].dtype != DataType::Float32) {
        throw InferenceException(context + ": bias tensor must be float32");
    }
    const std::size_t rank = x.dims.size();
    if (rank < 3 || rank > 5) {
        throw InferenceException(context + ": input must be rank 3, 4, or 5 (1-D, 2-D, or 3-D convolution)");
    }
    if (w.dims.size() != rank) {
        throw InferenceException(context + ": weight rank must match the input rank");
    }
    (void)ElementCount(x.dims, context);
    (void)ElementCount(w.dims, context);
    (void)ElementCount(out.dims, context);
    const std::size_t spatial = rank - 2;

    const int64_t n_batch = x.dims[0];
    const int64_t inChannels = x.dims[1];
    const int64_t outChannels = w.dims[0];

    const int64_t group = IntAttribute(node, "group", 1, "Conv");
    if (group <= 0) {
        throw InferenceException(context + ": group must be positive");
    }
    if (inChannels % group != 0) {
        throw InferenceException(context + ": input channels are not divisible by group");
    }
    if (w.dims[1] != inChannels / group) {
        throw InferenceException(context + ": weight channel dimension does not match input/group");
    }
    if (outChannels % group != 0) {
        throw InferenceException(context + ": output channels are not divisible by group");
    }
    const int64_t channelsPerGroup = inChannels / group;
    const int64_t outputsPerGroup = outChannels / group;

    std::vector<int64_t> strides = IntListAttribute(node, "strides", "Conv");
    std::vector<int64_t> dilations = IntListAttribute(node, "dilations", "Conv");
    if (strides.empty()) {
        strides.assign(spatial, 1);
    }
    if (dilations.empty()) {
        dilations.assign(spatial, 1);
    }
    if (strides.size() != spatial || dilations.size() != spatial) {
        throw InferenceException(context + ": stride and dilation must have one entry per spatial dimension");
    }
    const std::vector<int64_t> in_spatial(x.dims.begin() + 2, x.dims.end());
    const std::vector<int64_t> kernel(w.dims.begin() + 2, w.dims.end());
    for (std::size_t i = 0; i < spatial; ++i) {
        if (strides[i] <= 0 || dilations[i] <= 0) {
            throw InferenceException(context + ": stride and dilation must be positive");
        }
        if (kernel[i] <= 0) {
            throw InferenceException(context + ": kernel dimensions must be positive");
        }
    }

    std::vector<int64_t> pad_begin;
    std::vector<int64_t> pad_end;
    ResolvePads(context, StringAttribute(node, "auto_pad", "NOTSET", "Conv"), in_spatial, kernel, dilations, strides,
                IntListAttribute(node, "pads", "Conv"), pad_begin, pad_end);

    std::vector<int64_t> expected = {n_batch, outChannels};
    for (std::size_t i = 0; i < spatial; ++i) {
        expected.push_back(
            WindowOutput(context, in_spatial[i], kernel[i], dilations[i], pad_begin[i], pad_end[i], strides[i]));
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the convolution");
    }

    const float* bias = nullptr;
    if (hasBias) {
        const std::vector<int64_t>& biasDims = inputs[2].dims;
        if (biasDims.size() != 1 || biasDims[0] != outChannels) {
            throw InferenceException(context + ": bias must be rank-1 of length M");
        }
        bias = static_cast<const float*>(inputs[2].data);
    }
    if (x.data == nullptr || w.data == nullptr || out.data == nullptr || (hasBias && bias == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    // Lift every case to three spatial axes (depth, height, width); a missing
    // leading axis has extent 1, stride 1, no padding.
    const std::size_t lift = 3 - spatial;
    auto lifted = [lift](const std::vector<int64_t>& v, int64_t fill) {
        std::vector<int64_t> r(lift, fill);
        r.insert(r.end(), v.begin(), v.end());
        return r;
    };
    const std::vector<int64_t> in3 = lifted(in_spatial, 1);
    const std::vector<int64_t> k3 = lifted(kernel, 1);
    const std::vector<int64_t> o3 = lifted(std::vector<int64_t>(expected.begin() + 2, expected.end()), 1);
    const std::vector<int64_t> s3 = lifted(strides, 1);
    const std::vector<int64_t> d3 = lifted(dilations, 1);
    const std::vector<int64_t> p3 = lifted(pad_begin, 0);

    ConvParams p{};
    p.n_batch = n_batch;
    p.in_channels = inChannels;
    p.out_channels = outChannels;
    p.channels_per_group = channelsPerGroup;
    p.outputs_per_group = outputsPerGroup;
    p.has_bias = hasBias ? 1 : 0;
    for (std::size_t i = 0; i < 3; ++i) {
        p.in3[i] = in3[i];
        p.k3[i] = k3[i];
        p.o3[i] = o3[i];
        p.s3[i] = s3[i];
        p.d3[i] = d3[i];
        p.p3[i] = p3[i];
    }
    p.in_plane = in3[0] * in3[1] * in3[2];
    p.k_plane = k3[0] * k3[1] * k3[2];
    p.out_plane = o3[0] * o3[1] * o3[2];
    const int64_t outCount = n_batch * outChannels * p.out_plane;

    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    ConvKernel<<<grid, block>>>(static_cast<const float*>(x.data), static_cast<const float*>(w.data), bias,
                                static_cast<float*>(out.data), outCount, p);
    CheckHip(hipGetLastError(), "ConvKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
