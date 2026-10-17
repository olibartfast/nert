// Split on CUDA: carve the input along one axis into the outputs.
//
// Mirrors the CPU oracle: f32/i64 sharing one dtype, axis attr default 0
// with negative wrap, equal parts or an int64 split-sizes operand
// (downloaded from device when present). Each (output, outer-row) block is
// one cudaMemcpy device-to-device, exactly the oracle's memcpy structure.

#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nert {
namespace cuda_kernels {
namespace {

void CheckCuda(cudaError_t status, const char* what, const std::string& context) {
    if (status != cudaSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + cudaGetErrorString(status));
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

void SplitCopy(const TensorView& in, const std::vector<TensorView>& outputs, const std::vector<int64_t>& parts,
               int64_t axis, int64_t outer, int64_t inner, std::size_t element, const std::string& context) {
    const auto* x = static_cast<const std::uint8_t*>(in.data);
    int64_t begin = 0;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        auto* y = static_cast<std::uint8_t*>(outputs[i].data);
        const int64_t extent = parts[i];
        if (extent == 0) {
            continue;
        }
        for (int64_t o = 0; o < outer; ++o) {
            const void* src =
                x + static_cast<std::size_t>(o * in.dims[static_cast<std::size_t>(axis)] + begin) * inner * element;
            void* dst = y + static_cast<std::size_t>(o * extent * inner) * element;
            CheckCuda(
                cudaMemcpy(dst, src, static_cast<std::size_t>(extent * inner) * element, cudaMemcpyDeviceToDevice),
                "cudaMemcpy", context);
        }
        begin += extent;
    }
}

} // namespace

void Split(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Split (node '" + node.name + "')";
    if (inputs.empty() || inputs.size() > 2 || outputs.empty()) {
        throw InferenceException(context + ": expected one or two inputs and at least one output");
    }
    const TensorView& in = inputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    for (const TensorView& view : outputs) {
        if (view.dtype != in.dtype) {
            throw InferenceException(context + ": output dtype must match the input dtype");
        }
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Split axis is out of range");
    }
    int64_t axis = IntAttribute(node, "axis", 0, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Split axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];

    std::vector<int64_t> parts;
    if (inputs.size() == 2 && inputs[1].dtype != DataType::Unknown) {
        const TensorView& sizes = inputs[1];
        if (sizes.dtype != DataType::Int64) {
            throw InferenceException(context + ": split sizes must be int64");
        }
        const int64_t count = ElementCount(sizes.dims, context);
        if (count > 0 && sizes.data == nullptr) {
            throw InferenceException(context + ": null split-sizes buffer");
        }
        parts.assign(static_cast<std::size_t>(count), 0);
        if (count > 0) {
            CheckCuda(cudaMemcpy(parts.data(), sizes.data, static_cast<std::size_t>(count) * sizeof(int64_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy", context);
        }
        if (parts.size() != outputs.size()) {
            throw InferenceException(context + ": Split split sizes must match the output count");
        }
        int64_t sum = 0;
        for (const int64_t v : parts) {
            if (v < 0) {
                throw InferenceException(context + ": Split split sizes must be non-negative");
            }
            sum += v;
        }
        if (sum != dim) {
            throw InferenceException(context + ": Split split sizes must sum to the axis dim");
        }
    } else {
        if (dim % static_cast<int64_t>(outputs.size()) != 0) {
            throw InferenceException(context + ": Split axis dim is not divisible by the output count");
        }
        parts.assign(outputs.size(), dim / static_cast<int64_t>(outputs.size()));
    }

    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::vector<int64_t> want = in.dims;
        want[static_cast<std::size_t>(axis)] = parts[i];
        if (outputs[i].dims != want) {
            throw InferenceException(context + ": output shape does not match the split sizes");
        }
        if (ElementCount(outputs[i].dims, context) > 0 && outputs[i].data == nullptr) {
            throw InferenceException(context + ": null output buffer");
        }
    }
    if (ElementCount(in.dims, context) > 0 && in.data == nullptr) {
        throw InferenceException(context + ": null input buffer");
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    const std::size_t element = in.dtype == DataType::Float32 ? sizeof(float) : sizeof(int64_t);
    SplitCopy(in, outputs, parts, axis, outer, inner, element, context);
}

} // namespace cuda_kernels
} // namespace nert
