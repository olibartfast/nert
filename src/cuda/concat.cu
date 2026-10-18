// Concat on CUDA: join the inputs along one axis.
//
// Mirrors the CPU oracle: f32/i64 sharing one dtype, axis attr default 1
// with negative wrap, shared rank, agreement off the concat axis. Each
// (input, outer-row) block is one cudaMemcpy device-to-device, exactly the
// oracle's memcpy structure.

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
        if (dim < 0) {
            throw InferenceException(context + ": negative dimension");
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

void ConcatCopy(const std::vector<TensorView>& inputs, void* out, int64_t axis, int64_t outer, int64_t inner,
                int64_t out_extent, std::size_t element, const std::string& context) {
    auto* dst = static_cast<std::uint8_t*>(out);
    int64_t written = 0;
    for (const TensorView& view : inputs) {
        const auto* x = static_cast<const std::uint8_t*>(view.data);
        const int64_t extent = view.dims[static_cast<std::size_t>(axis)];
        if (extent == 0) {
            continue;
        }
        for (int64_t o = 0; o < outer; ++o) {
            const void* src = x + static_cast<std::size_t>(o * extent * inner) * element;
            void* d = dst + static_cast<std::size_t>(o * out_extent + written) * inner * element;
            CheckCuda(cudaMemcpy(d, src, static_cast<std::size_t>(extent * inner) * element, cudaMemcpyDeviceToDevice),
                      "cudaMemcpy", context);
        }
        written += extent;
    }
}

} // namespace

void Concat(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Concat (node '" + node.name + "')";
    if (inputs.empty() || outputs.size() != 1) {
        throw InferenceException(context + ": expected at least one input and one output");
    }
    const TensorView& first = inputs[0];
    if (first.dtype != DataType::Float32 && first.dtype != DataType::Int64 && first.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    const TensorView& out = outputs[0];
    if (out.dtype != first.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const int64_t rank = static_cast<int64_t>(first.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Concat axis is out of range");
    }
    int64_t axis = IntAttribute(node, "axis", 1, context);
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Concat axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    std::vector<int64_t> expected = first.dims;
    for (const TensorView& view : inputs) {
        if (view.dtype != first.dtype) {
            throw InferenceException(context + ": all inputs must share the element type");
        }
        if (static_cast<int64_t>(view.dims.size()) != rank) {
            throw InferenceException(context + ": Concat inputs must share rank");
        }
        for (int64_t j = 0; j < rank; ++j) {
            if (j == axis) {
                continue;
            }
            if (view.dims[static_cast<std::size_t>(j)] != expected[static_cast<std::size_t>(j)]) {
                throw InferenceException(context + ": Concat inputs differ off the concat axis");
            }
        }
        expected[static_cast<std::size_t>(axis)] += view.dims[static_cast<std::size_t>(axis)];
    }
    // The first input's axis extent was counted twice above; correct it.
    expected[static_cast<std::size_t>(axis)] -= first.dims[static_cast<std::size_t>(axis)];
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the concatenated inputs");
    }

    for (const TensorView& view : inputs) {
        (void)ElementCount(view.dims, context);
        if (ElementCount(view.dims, context) > 0 && view.data == nullptr) {
            throw InferenceException(context + ": null input buffer");
        }
    }
    const int64_t out_count = ElementCount(out.dims, context);
    if (out_count > 0 && out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    if (out_count == 0) {
        return;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= expected[static_cast<std::size_t>(i)];
    }
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= expected[static_cast<std::size_t>(i)];
    }
    const int64_t out_extent = expected[static_cast<std::size_t>(axis)];
    const std::size_t element = first.dtype == DataType::Float32 ? sizeof(float)
                                : first.dtype == DataType::Int64 ? sizeof(int64_t)
                                                                 : sizeof(bool);
    ConcatCopy(inputs, out.data, axis, outer, inner, out_extent, element, context);
}

} // namespace cuda_kernels
} // namespace nert
