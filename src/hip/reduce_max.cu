// Hipified from src/cuda/reduce_max.cu via hipify-perl, hand-reviewed.
// ReduceMax on HIP: max over axes, one thread per output element.
//
// Mirrors the CPU oracle: float32 only, keepdims attr default 1, axes in the
// form legal for the node's effective schema (attribute before 18, optional
// int64 input from 18), an absent or empty axes value means every axis,
// unique normalized axes, output-shape check. Each thread replays the
// oracle's input-order scan restricted to its own output cell, so ties and
// -inf behavior match without atomics. Ranks above 16 are rejected (pilot
// limit).

#include "../reduce_axes.hpp"
#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
namespace {

constexpr std::size_t kMaxRank = 16;

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

struct IndexSet {
    int64_t s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the reduce limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = values[j];
    }
    return packed;
}

__global__ void ReduceMaxKernel(const float* x, float* y, int64_t outCount, IndexSet inStrides, IndexSet outStrides,
                                IndexSet axisMap, IndexSet redStrides, int rank, int outRank, int64_t redCount,
                                float lowest) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t base = 0;
        for (int j = 0; j < outRank; ++j) {
            const int64_t coord = remaining / outStrides.s[j];
            remaining %= outStrides.s[j];
            // The input axis feeding this output axis: invert axisMap.
            for (int a = 0; a < rank; ++a) {
                if (axisMap.s[a] == j) {
                    base += coord * inStrides.s[a];
                }
            }
        }
        float best = lowest;
        for (int64_t t = 0; t < redCount; ++t) {
            int64_t rest = t;
            int64_t offset = base;
            for (int r = 0; r < rank; ++r) {
                if (axisMap.s[r] == -1) {
                    // Reduced axis r is the (r-th-slot) entry of the packed lists.
                    int slot = 0;
                    for (int q = 0; q < r; ++q) {
                        if (axisMap.s[q] == -1) {
                            ++slot;
                        }
                    }
                    const int64_t coord = rest / redStrides.s[slot];
                    rest %= redStrides.s[slot];
                    offset += coord * inStrides.s[r];
                }
            }
            const float v = x[offset];
            if (v > best) {
                best = v;
            }
        }
        y[i] = best;
    }
}

} // namespace

void ReduceMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ReduceMax (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one or two inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": data tensors must be float32");
    }

    const std::vector<int64_t>& inDims = in.dims;
    const int64_t rank = static_cast<int64_t>(inDims.size());
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the reduce limit of 16");
    }
    const int64_t inCount = ElementCount(inDims, context);
    if (in.data == nullptr) {
        throw InferenceException(context + ": null input buffer");
    }

    const int64_t keepdims = IntAttribute(node, "keepdims", 1, context);
    const ReduceAxes resolved_axes = ResolveReduceAxes(node);

    // Resolve the axes once: the old attribute form and the input form both
    // reduce every axis when the axes value is absent or empty.
    std::vector<int64_t> axes;
    if (resolved_axes.form == ReduceAxesForm::Attribute) {
        if (resolved_axes.axes_attribute != nullptr) {
            const std::vector<int64_t>* values = ReduceAttributeIntList(resolved_axes.axes_attribute);
            if (values == nullptr) {
                throw InferenceException(context + ": attribute 'axes' must be an integer list");
            }
            axes = *values;
        }
    } else if (inputs.size() == 2 && inputs[1].dtype != DataType::Unknown) {
        const TensorView& axesView = inputs[1];
        if (axesView.dtype != DataType::Int64) {
            throw InferenceException(context + ": axes tensor must be int64");
        }
        const int64_t axesCount = ReduceAxesElementCount(axesView.dims, context);
        if (axesCount > 0 && axesView.data == nullptr) {
            throw InferenceException(context + ": null axes buffer");
        }
        if (axesCount > 0) {
            axes.assign(static_cast<std::size_t>(axesCount), 0);
            CheckHip(hipMemcpy(axes.data(), axesView.data, static_cast<std::size_t>(axesCount) * sizeof(int64_t),
                               hipMemcpyDeviceToHost),
                     "hipMemcpy", context);
        }
    }
    if (axes.empty()) {
        for (int64_t i = 0; i < rank; ++i) {
            axes.push_back(i);
        }
    }

    // Normalize and validate the axes: unique and within [-rank, rank - 1].
    std::vector<bool> reduced(static_cast<std::size_t>(rank), false);
    for (int64_t axis : axes) {
        if (axis < -rank || axis >= rank) {
            throw InferenceException(context + ": axis out of range");
        }
        const int64_t normalized = axis < 0 ? axis + rank : axis;
        if (reduced[static_cast<std::size_t>(normalized)]) {
            throw InferenceException(context + ": duplicate axis");
        }
        reduced[static_cast<std::size_t>(normalized)] = true;
    }

    std::vector<int64_t> expectedDims;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            if (keepdims != 0) {
                expectedDims.push_back(1);
            }
        } else {
            expectedDims.push_back(inDims[static_cast<std::size_t>(i)]);
        }
    }
    if (out.dims != expectedDims) {
        throw InferenceException(context + ": output shape does not match the reduced axes");
    }
    const int64_t outCount = ElementCount(out.dims, context);

    std::vector<int64_t> inStride(static_cast<std::size_t>(rank), 1);
    for (std::size_t i = static_cast<std::size_t>(rank); i-- > 1;) {
        inStride[i - 1] = inStride[i] * inDims[i];
    }
    std::vector<int64_t> outStride(expectedDims.size(), 1);
    for (std::size_t i = expectedDims.size(); i-- > 1;) {
        outStride[i - 1] = outStride[i] * expectedDims[i];
    }

    // Map each surviving input axis to its output axis.
    std::vector<int64_t> axisMap(static_cast<std::size_t>(rank), -1);
    int64_t nextOutAxis = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (!reduced[static_cast<std::size_t>(i)]) {
            axisMap[static_cast<std::size_t>(i)] = keepdims != 0 ? i : nextOutAxis;
            ++nextOutAxis;
        }
    }

    // Reduced-axis dims in input-axis order, with strides for the cartesian
    // walk.
    std::vector<int64_t> redDims;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<std::size_t>(i)]) {
            redDims.push_back(inDims[static_cast<std::size_t>(i)]);
        }
    }
    std::vector<int64_t> redStrides(redDims.size(), 1);
    for (std::size_t i = redDims.size(); i-- > 1;) {
        redStrides[i - 1] = redStrides[i] * redDims[i];
    }
    int64_t redCount = 1;
    for (int64_t d : redDims) {
        redCount *= d;
    }

    if (out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    if (outCount == 0) {
        return;
    }

    const float lowest = std::numeric_limits<float>::lowest();
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    ReduceMaxKernel<<<grid, block>>>(static_cast<const float*>(in.data), static_cast<float*>(out.data), outCount,
                                     Pack(inStride, context), Pack(outStride, context), Pack(axisMap, context),
                                     Pack(redStrides, context), static_cast<int>(rank),
                                     static_cast<int>(expectedDims.size()), redCount, lowest);
    CheckHip(hipGetLastError(), "ReduceMaxKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace hip_kernels
} // namespace nert
