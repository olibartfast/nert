// Shape on OpenCL: the input's dims are host metadata, so the kernel is
// one host-to-device copy.
//
// Mirrors the CPU oracle: float32/int64/bool input, int64 output,
// start/end attrs clamped into [0, rank] with end pulled up to start,
// output shape check.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
namespace {

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
    }
}

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Shape: attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

} // namespace

void Shape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Shape (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": Shape input must be float32, int64, or bool");
    }
    if (out.dtype != DataType::Int64) {
        throw InferenceException(context + ": Shape output must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    int64_t start = IntAttribute(node, "start", 0);
    int64_t end = IntAttribute(node, "end", rank);
    start = std::min(std::max(start < 0 ? start + rank : start, int64_t{0}), rank);
    end = std::min(std::max(end < 0 ? end + rank : end, int64_t{0}), rank);
    if (end < start) {
        end = start;
    }
    if (out.dims != std::vector<int64_t>{end - start}) {
        throw InferenceException(context + ": Shape output shape must be the selected dim count");
    }
    if (end > start && out.data == nullptr) {
        throw InferenceException(context + ": null output buffer");
    }
    if (end == start) {
        return;
    }
    CheckCl(clEnqueueWriteBuffer(opencl_runtime::Queue(), reinterpret_cast<cl_mem>(out.data), CL_TRUE, 0,
                                 static_cast<std::size_t>(end - start) * sizeof(int64_t), in.dims.data() + start, 0,
                                 nullptr, nullptr),
            "clEnqueueWriteBuffer", context);
}

} // namespace opencl_kernels
} // namespace nert
