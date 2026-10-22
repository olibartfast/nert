// Unsqueeze on CUDA: validate the axes on the host, then copy.
//
// Mirrors the CPU oracle: f32/i64/bool sharing one dtype, axes resolved once
// from the node's effective schema (pre-13 attribute or schema-13+ int64
// operand, downloaded from device), wrapped negatives, uniqueness, output
// shape check. Movement is cudaMemcpy device-to-device.

#include "../axes_form.hpp"
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

} // namespace

void Unsqueeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Unsqueeze (node '" + node.name + "')";
    if (inputs.empty() || outputs.size() != 1) {
        throw InferenceException(context + ": expected a data input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    const AxesForm axes_form = ResolveAxesForm(node);
    std::vector<int64_t> axes;
    if (axes_form.source == AxesSource::Attribute) {
        if (inputs.size() != 1) {
            throw InferenceException(context + ": expected one data input for the attribute axes form");
        }
        const std::vector<int64_t>* values = NodeAttributeIntList(axes_form.axes_attribute);
        if (values == nullptr) {
            throw InferenceException(context + ": Unsqueeze requires an axes attribute");
        }
        axes = *values;
    } else {
        if (inputs.size() != 2) {
            throw InferenceException(context + ": expected a data input, an axes input, and one output");
        }
        const TensorView& axes_view = inputs[1];
        if (axes_view.dtype != DataType::Int64) {
            throw InferenceException(context + ": Unsqueeze requires an axes input");
        }
        const int64_t axes_count = ElementCount(axes_view.dims, context);
        if (axes_count > 0) {
            if (axes_view.data == nullptr) {
                throw InferenceException(context + ": null axes buffer");
            }
            axes.assign(static_cast<std::size_t>(axes_count), 0);
            CheckCuda(cudaMemcpy(axes.data(), axes_view.data, static_cast<std::size_t>(axes_count) * sizeof(int64_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy", context);
        }
    }
    if (axes.empty()) {
        throw InferenceException(context + ": Unsqueeze axes must not be empty");
    }
    const int64_t axes_count = static_cast<int64_t>(axes.size());
    const int64_t rank_out = static_cast<int64_t>(in.dims.size()) + axes_count;
    std::vector<bool> seen(static_cast<std::size_t>(rank_out), false);
    for (int64_t& axis : axes) {
        if (axis < -rank_out || axis >= rank_out) {
            throw InferenceException(context + ": Unsqueeze axis is out of range");
        }
        if (axis < 0) {
            axis += rank_out;
        }
        if (seen[static_cast<std::size_t>(axis)]) {
            throw InferenceException(context + ": Unsqueeze axes must be unique");
        }
        seen[static_cast<std::size_t>(axis)] = true;
    }
    std::vector<int64_t> expected(static_cast<std::size_t>(rank_out), 0);
    std::size_t data_pos = 0;
    for (int64_t i = 0; i < rank_out; ++i) {
        if (seen[static_cast<std::size_t>(i)]) {
            expected[static_cast<std::size_t>(i)] = 1;
        } else {
            expected[static_cast<std::size_t>(i)] = in.dims[data_pos++];
        }
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the unsqueezed input");
    }

    const int64_t in_count = ElementCount(in.dims, context);
    const int64_t out_count = ElementCount(out.dims, context);
    if (in_count != out_count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (in_count == 0) {
        return;
    }
    const std::size_t bytes = static_cast<std::size_t>(in_count) * (in.dtype == DataType::Float32 ? sizeof(float)
                                                                    : in.dtype == DataType::Int64 ? sizeof(int64_t)
                                                                                                  : sizeof(bool));
    CheckCuda(cudaMemcpy(out.data, in.data, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy", context);
}

} // namespace cuda_kernels
} // namespace nert
