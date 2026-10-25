// Abs/Neg (float32 and int64) and Floor (float32), matching the CPU rows.
#include "nert/device.hpp"
#include "runtime.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace nert {
namespace opencl_kernels {
namespace {
constexpr const char* kSource = R"CLC(
__kernel void abs_f(__global const float* x, __global float* y, long n) {
    long i = get_global_id(0); if (i < n) y[i] = fabs(x[i]);
}
__kernel void abs_i(__global const long* x, __global long* y, long n) {
    long i = get_global_id(0); if (i < n) y[i] = as_long(abs(x[i]));
}
__kernel void neg_f(__global const float* x, __global float* y, long n) {
    long i = get_global_id(0); if (i < n) y[i] = -x[i];
}
__kernel void neg_i(__global const long* x, __global long* y, long n) {
    long i = get_global_id(0); if (i < n) y[i] = as_long((ulong)0 - as_ulong(x[i]));
}
__kernel void floor_f(__global const float* x, __global float* y, long n) {
    long i = get_global_id(0); if (i < n) y[i] = floor(x[i]);
}
__kernel void sign_f(__global const float* x, __global float* y, long n) {
    long i = get_global_id(0); if (i < n) y[i] = isnan(x[i]) ? x[i] : (float)((0.0f < x[i]) - (x[i] < 0.0f));
}
__kernel void sign_i(__global const long* x, __global long* y, long n) {
    long i = get_global_id(0); if (i < n) y[i] = (long)(((long)0 < x[i]) - (x[i] < (long)0));
}
)CLC";

void Check(cl_int status, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": OpenCL status " + std::to_string(status));
    }
}

int64_t Count(const std::vector<int64_t>& dims, const std::string& context) {
    int64_t count = 1;
    for (int64_t dim : dims) {
        if (dim <= 0 || count > std::numeric_limits<int64_t>::max() / dim) {
            throw InferenceException(context + ": invalid dimension or element-count overflow");
        }
        count *= dim;
    }
    return count;
}

void Run(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs,
         const char* op, const char* float_entry, const char* int_entry) {
    const std::string context = std::string(op) + " (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != out.dtype ||
        (in.dtype != DataType::Float32 && (int_entry == nullptr || in.dtype != DataType::Int64))) {
        throw InferenceException(context + ": unsupported or mismatched dtype");
    }
    const cl_long count = Count(in.dims, context);
    if (Count(out.dims, context) != count) {
        throw InferenceException(context + ": element-count mismatch");
    }
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const char* entry = in.dtype == DataType::Float32 ? float_entry : int_entry;
    cl_kernel kernel = opencl_runtime::KernelFor(std::string("UnaryNumeric/") + entry, kSource, entry);
    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    Check(clSetKernelArg(kernel, 0, sizeof(x), &x), context);
    Check(clSetKernelArg(kernel, 1, sizeof(y), &y), context);
    Check(clSetKernelArg(kernel, 2, sizeof(count), &count), context);
    const std::size_t global = static_cast<std::size_t>(count);
    Check(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
          context);
    Check(clFinish(opencl_runtime::Queue()), context);
}
} // namespace

void Abs(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    Run(node, inputs, outputs, "Abs", "abs_f", "abs_i");
}
void Sign(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    if (inputs.size() == 1 && outputs.size() == 1 && inputs[0].dims != outputs[0].dims) {
        throw InferenceException("Sign: shape mismatch");
    }
    Run(node, inputs, outputs, "Sign", "sign_f", "sign_i");
}
void Neg(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    Run(node, inputs, outputs, "Neg", "neg_f", "neg_i");
}
void Floor(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    Run(node, inputs, outputs, "Floor", "floor_f", nullptr);
}
} // namespace opencl_kernels
} // namespace nert
