// Hipified from src/cuda/pad.cu via hipify-perl, hand-reviewed.
// Pad on HIP: pad constant/reflect/edge around the input.
//
// Mirrors the CPU oracle: float32/int64 sharing one dtype, `mode` string
// attr (constant/reflect/edge), pads + optional axes int64 inputs
// downloaded to the host, optional scalar constant_value of the data
// dtype. One thread per output element maps back through the oracle's
// position rule. Rank cap 16 (pilot limit).

#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

constexpr std::size_t kMaxRank = 16;

enum class PadMode : int {
    Constant = 0,
    Reflect = 1,
    Edge = 2,
};

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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

struct IndexSet {
    int64_t s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pad limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = values[j];
    }
    return packed;
}

std::vector<int64_t> ReadInt64s(const TensorView& view, const char* what, const std::string& context) {
    if (view.dtype != DataType::Int64) {
        throw InferenceException(context + ": " + std::string(what) + " must be int64");
    }
    int64_t count = 1;
    for (int64_t dim : view.dims) {
        count *= dim;
    }
    std::vector<int64_t> values(static_cast<std::size_t>(count), 0);
    if (count > 0) {
        if (view.data == nullptr) {
            throw InferenceException(context + ": null " + std::string(what) + " buffer");
        }
        CheckHip(hipMemcpy(values.data(), view.data, static_cast<std::size_t>(count) * sizeof(int64_t),
                           hipMemcpyDeviceToHost),
                 "hipMemcpy", context);
    }
    return values;
}

// Maps a source position along one axis of extent `dim` into [0, dim), or -1
// for constant fill. Matches the oracle's MapPosition exactly.
__device__ int64_t MapPosition(int64_t pos, int64_t dim, int mode) {
    if (pos >= 0 && pos < dim) {
        return pos;
    }
    if (mode == static_cast<int>(PadMode::Constant)) {
        return -1;
    }
    if (mode == static_cast<int>(PadMode::Edge)) {
        return pos < 0 ? 0 : dim - 1;
    }
    if (dim == 1) {
        return 0;
    }
    const int64_t period = 2 * (dim - 1);
    int64_t m = pos % period;
    if (m < 0) {
        m += period;
    }
    return m < dim ? m : period - m;
}

template <typename T>
__global__ void PadKernel(const T* x, T* y, int64_t outCount, IndexSet outStrides, IndexSet inDims, IndexSet inStrides,
                          IndexSet begins, int rank, int mode, T fill) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        bool filled = false;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            const int64_t src = MapPosition(coord - begins.s[j], inDims.s[j], mode);
            if (src < 0) {
                filled = true;
                break;
            }
            offset += src * inStrides.s[j];
        }
        y[i] = filled ? fill : x[offset];
    }
}

template <typename T>
void LaunchPad(const T* x, T* y, int64_t outCount, const std::vector<int64_t>& outStrides,
               const std::vector<int64_t>& inDims, const std::vector<int64_t>& inStrides,
               const std::vector<int64_t>& begins, std::size_t rank, PadMode mode, T fill, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    PadKernel<<<grid, block>>>(x, y, outCount, Pack(outStrides, context), Pack(inDims, context),
                               Pack(inStrides, context), Pack(begins, context), static_cast<int>(rank),
                               static_cast<int>(mode), fill);
    CheckHip(hipGetLastError(), "PadKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void Pad(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Pad (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two to four inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if ((in.dtype != DataType::Float32 && in.dtype != DataType::Int64) || out.dtype != in.dtype) {
        throw InferenceException(context + ": data tensors must share dtype float32 or int64");
    }

    PadMode mode = PadMode::Constant;
    for (const Attribute& attr : node.attributes) {
        if (attr.name != "mode") {
            continue;
        }
        const std::string* value = std::get_if<std::string>(&attr.value);
        if (value == nullptr) {
            throw InferenceException(context + ": attribute 'mode' must be a string");
        }
        if (*value == "constant") {
            mode = PadMode::Constant;
        } else if (*value == "reflect") {
            mode = PadMode::Reflect;
        } else if (*value == "edge") {
            mode = PadMode::Edge;
        } else {
            throw InferenceException(context + ": unsupported mode '" + *value + "'");
        }
    }

    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the pad limit of 16");
    }
    (void)ElementCount(in.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> pads = ReadInt64s(inputs[1], "pads", context);
    std::vector<int64_t> axes;
    if (inputs.size() == 4 && inputs[3].dtype != DataType::Unknown) {
        axes = ReadInt64s(inputs[3], "axes", context);
    } else {
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }
    if (pads.size() != axes.size() * 2) {
        throw InferenceException(context + ": pads must hold two entries per padded axis");
    }

    std::vector<int64_t> begins(static_cast<std::size_t>(rank), 0);
    std::vector<int64_t> ends(static_cast<std::size_t>(rank), 0);
    std::vector<bool> seen(static_cast<std::size_t>(rank), false);
    for (std::size_t k = 0; k < axes.size(); ++k) {
        int64_t axis = axes[k];
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": axis out of range");
        }
        if (axis < 0) {
            axis += rank;
        }
        if (seen[static_cast<std::size_t>(axis)]) {
            throw InferenceException(context + ": duplicate axis");
        }
        seen[static_cast<std::size_t>(axis)] = true;
        begins[static_cast<std::size_t>(axis)] = pads[k];
        ends[static_cast<std::size_t>(axis)] = pads[axes.size() + k];
    }

    std::vector<int64_t> expectedDims = in.dims;
    for (std::size_t i = 0; i < expectedDims.size(); ++i) {
        expectedDims[i] += begins[i] + ends[i];
    }
    if (out.dims != expectedDims) {
        throw InferenceException(context + ": output shape does not match the pads");
    }
    const int64_t outCount = ElementCount(out.dims, context);

    const bool hasValue = inputs.size() >= 3 && inputs[2].dtype != DataType::Unknown;
    if (hasValue) {
        if (inputs[2].dtype != in.dtype) {
            throw InferenceException(context + ": constant_value must match the data dtype");
        }
        if (ElementCount(inputs[2].dims, context) != 1 || inputs[2].data == nullptr) {
            throw InferenceException(context + ": constant_value must be a scalar");
        }
    }

    if (outCount == 0) {
        return;
    }
    const std::vector<int64_t> outStrides = Strides(out.dims);
    const std::vector<int64_t> inStrides = Strides(in.dims);
    if (in.dtype == DataType::Float32) {
        float hostFill = 0.0F;
        if (hasValue) {
            CheckHip(hipMemcpy(&hostFill, inputs[2].data, sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy", context);
        }
        LaunchPad(static_cast<const float*>(in.data), static_cast<float*>(out.data), outCount, outStrides, in.dims,
                  inStrides, begins, in.dims.size(), mode, hostFill, context);
    } else {
        int64_t hostFill = 0;
        if (hasValue) {
            CheckHip(hipMemcpy(&hostFill, inputs[2].data, sizeof(int64_t), hipMemcpyDeviceToHost), "hipMemcpy",
                     context);
        }
        LaunchPad(static_cast<const int64_t*>(in.data), static_cast<int64_t*>(out.data), outCount, outStrides, in.dims,
                  inStrides, begins, in.dims.size(), mode, hostFill, context);
    }
}

} // namespace hip_kernels
} // namespace nert
