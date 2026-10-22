// Hipified from src/cuda/layer_normalization.cu via hipify-perl, hand-reviewed.
// LayerNormalization on HIP: normalize over the trailing axes from
// `axis` (default -1), then scale and shift.
//
// Mirrors the CPU oracle: one thread per prod(dims[:axis]) row, statistics
// in double, Scale/B per normalized element or scalar, optional Mean and
// InvStdDev outputs with one value per row.

#include "nert/device.hpp"

#include <cmath>
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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be an integer");
            }
            return *value;
        }
    }
    return fallback;
}

float FloatAttribute(const Node& node, const std::string& name, float fallback, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute '" + name + "' must be a float");
            }
            return *value;
        }
    }
    return fallback;
}

// A per-normalized-element parameter (Scale or B) on the device: `inner`
// values, or one. Returns the device pointer and reports scalar in `scalar`.
const float* ParamData(const TensorView& view, int64_t inner, bool& scalar, const std::string& context,
                       const char* what) {
    if (view.dtype != DataType::Float32 || view.data == nullptr) {
        throw InferenceException(context + ": " + std::string(what) + " must be a float32 tensor");
    }
    const int64_t count = ElementCount(view.dims, context);
    if (count != inner && count != 1) {
        throw InferenceException(context + ": " + std::string(what) + " must have one value per normalized element");
    }
    scalar = count == 1;
    return static_cast<const float*>(view.data);
}

__global__ void LayerNormalizationKernel(const float* x, float* y, float* meanOut, float* invStdOut, const float* scale,
                                         const float* bias, bool scaleScalar, bool hasBias, bool biasScalar,
                                         int64_t outer, int64_t inner, double epsilon) {
    for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x; r < outer; r += blockDim.x * gridDim.x) {
        const float* row = x + r * inner;
        float* outRow = y + r * inner;
        double sum = 0.0;
        for (int64_t i = 0; i < inner; ++i) {
            sum += row[i];
        }
        const double mean = sum / static_cast<double>(inner);
        double squares = 0.0;
        for (int64_t i = 0; i < inner; ++i) {
            const double centered = row[i] - mean;
            squares += centered * centered;
        }
        const double invStd = 1.0 / sqrt(squares / static_cast<double>(inner) + epsilon);
        for (int64_t i = 0; i < inner; ++i) {
            const double normalized = (row[i] - mean) * invStd;
            const double scaled = normalized * scale[scaleScalar ? 0 : i];
            outRow[i] = static_cast<float>(hasBias ? scaled + bias[biasScalar ? 0 : i] : scaled);
        }
        if (meanOut != nullptr) {
            meanOut[r] = static_cast<float>(mean);
        }
        if (invStdOut != nullptr) {
            invStdOut[r] = static_cast<float>(invStd);
        }
    }
}

} // namespace

void LayerNormalization(const Node& node, const std::vector<TensorView>& inputs,
                        const std::vector<TensorView>& outputs) {
    const std::string context = "LayerNormalization (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.empty() || outputs.size() > 3) {
        throw InferenceException(context + ": expected X, Scale, optional B, and one to three outputs");
    }
    const TensorView& x = inputs[0];
    const TensorView& y = outputs[0];
    if (x.dtype != DataType::Float32 || y.dtype != DataType::Float32) {
        throw InferenceException(context + ": X and Y must be float32");
    }
    if (x.dims.empty() || x.dims != y.dims) {
        throw InferenceException(context + ": Y must have the shape of a non-scalar X");
    }
    const int64_t rank = static_cast<int64_t>(x.dims.size());
    int64_t axis = IntAttribute(node, "axis", -1, context);
    if (axis < 0) {
        axis += rank;
    }
    if (axis < 0 || axis >= rank) {
        throw InferenceException(context + ": axis is out of range");
    }
    const float epsilon = FloatAttribute(node, "epsilon", 1e-5F, context);

    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < rank; ++i) {
        (i < axis ? outer : inner) *= x.dims[static_cast<std::size_t>(i)];
    }
    (void)ElementCount(x.dims, context);
    if (x.data == nullptr || y.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    bool scaleScalar = false;
    const float* scale = ParamData(inputs[1], inner, scaleScalar, context, "Scale");
    bool biasScalar = false;
    const float* bias = nullptr;
    if (inputs.size() == 3 && inputs[2].dtype != DataType::Unknown) {
        bias = ParamData(inputs[2], inner, biasScalar, context, "B");
    }

    float* meanOut = nullptr;
    float* invStdOut = nullptr;
    for (std::size_t o = 1; o < outputs.size(); ++o) {
        if (outputs[o].dtype == DataType::Unknown) {
            continue;
        }
        if (outputs[o].dtype != DataType::Float32 || outputs[o].data == nullptr ||
            ElementCount(outputs[o].dims, context) != outer) {
            throw InferenceException(context + ": Mean and InvStdDev must be float32 with one value per row");
        }
        (o == 1 ? meanOut : invStdOut) = static_cast<float*>(outputs[o].data);
    }

    if (outer == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((outer + block - 1) / block);
    LayerNormalizationKernel<<<grid, block>>>(static_cast<const float*>(x.data), static_cast<float*>(y.data), meanOut,
                                              invStdOut, scale, bias, scaleScalar, bias != nullptr, biasScalar, outer,
                                              inner, static_cast<double>(epsilon));
    CheckHip(hipGetLastError(), "LayerNormalizationKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
