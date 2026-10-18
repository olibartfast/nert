// Einsum on OpenCL for the two target-model equations.
//
// OWLv2:     ...pd,...qd->...pq  (batched class/image dot products)
// RF-DETR:   bchw,bnc->bnhw      (per-pixel channel contraction)
//
// Both are direct contractions with one work-item per output. Other
// equations are rejected with a named error; no silent narrowing.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <cstddef>
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

std::string Equation(const Node& node, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "equation") {
            const std::string* value = std::get_if<std::string>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'equation' must be a string");
            }
            return *value;
        }
    }
    throw InferenceException(context + ": missing 'equation' attribute");
}

constexpr const char* kSource = R"CLC(
__kernel void einsum_owlv2(__global const float* a, __global const float* b, __global float* y, long P, long Q,
                           long D) {
    long idx = get_global_id(0);
    long q = idx % Q;
    long rest = idx / Q;
    long p = rest % P;
    long batch = rest / P;
    long aBase = (batch * P + p) * D;
    long bBase = (batch * Q + q) * D;
    float acc = 0.0f;
    for (long d = 0; d < D; ++d) {
        acc += a[aBase + d] * b[bBase + d];
    }
    y[idx] = acc;
}
__kernel void einsum_rfdetr(__global const float* a, __global const float* b, __global float* y, long N, long C,
                            long H, long W) {
    long idx = get_global_id(0);
    long w = idx % W;
    long rest = idx / W;
    long h = rest % H;
    rest /= H;
    long n = rest % N;
    long batch = rest / N;
    long aBase = ((batch * C) * H + h) * W + w;
    long bBase = (batch * N + n) * C;
    float acc = 0.0f;
    for (long c = 0; c < C; ++c) {
        acc += a[aBase + c * H * W] * b[bBase + c];
    }
    y[idx] = acc;
}
)CLC";

} // namespace

void Einsum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Einsum (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }
    const std::string equation = Equation(node, context);
    const int64_t out_count = ElementCount(out.dims, context);
    cl_mem am = reinterpret_cast<cl_mem>(const_cast<void*>(a.data));
    cl_mem bm = reinterpret_cast<cl_mem>(const_cast<void*>(b.data));
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);

    if (equation == "...pd,...qd->...pq") {
        if (a.dims.size() != 3 || b.dims.size() != 3 || out.dims.size() != 3) {
            throw InferenceException(context + ": equation expects rank-3 operands and output");
        }
        const int64_t batch = a.dims[0];
        const int64_t p = a.dims[1];
        const int64_t d = a.dims[2];
        if (b.dims[0] != batch || b.dims[2] != d) {
            throw InferenceException(context + ": operand dimensions do not agree");
        }
        if (out.dims != std::vector<int64_t>{batch, p, b.dims[1]}) {
            throw InferenceException(context + ": output shape does not match the operands");
        }
        const cl_long p_arg = p;
        const cl_long q_arg = b.dims[1];
        const cl_long d_arg = d;
        cl_kernel kernel = opencl_runtime::KernelFor("EinsumOWLv2", kSource, "einsum_owlv2");
        CheckCl(clSetKernelArg(kernel, 0, sizeof(am), &am), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 1, sizeof(bm), &bm), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 2, sizeof(ym), &ym), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 3, sizeof(p_arg), &p_arg), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 4, sizeof(q_arg), &q_arg), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 5, sizeof(d_arg), &d_arg), "clSetKernelArg", context);
        const std::size_t global = static_cast<std::size_t>(out_count);
        CheckCl(
            clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
        CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
        return;
    }

    if (equation == "bchw,bnc->bnhw") {
        if (a.dims.size() != 4 || b.dims.size() != 3 || out.dims.size() != 4) {
            throw InferenceException(context + ": equation expects 4-D, 3-D operands and a 4-D output");
        }
        const int64_t batch = a.dims[0];
        const int64_t c = a.dims[1];
        const int64_t h = a.dims[2];
        const int64_t w = a.dims[3];
        if (b.dims[0] != batch || b.dims[2] != c) {
            throw InferenceException(context + ": operand dimensions do not agree");
        }
        if (out.dims != std::vector<int64_t>{batch, b.dims[1], h, w}) {
            throw InferenceException(context + ": output shape does not match the operands");
        }
        const cl_long n_arg = b.dims[1];
        const cl_long c_arg = c;
        const cl_long h_arg = h;
        const cl_long w_arg = w;
        cl_kernel kernel = opencl_runtime::KernelFor("EinsumRFDETR", kSource, "einsum_rfdetr");
        CheckCl(clSetKernelArg(kernel, 0, sizeof(am), &am), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 1, sizeof(bm), &bm), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 2, sizeof(ym), &ym), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 3, sizeof(n_arg), &n_arg), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 4, sizeof(c_arg), &c_arg), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 5, sizeof(h_arg), &h_arg), "clSetKernelArg", context);
        CheckCl(clSetKernelArg(kernel, 6, sizeof(w_arg), &w_arg), "clSetKernelArg", context);
        const std::size_t global = static_cast<std::size_t>(out_count);
        CheckCl(
            clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
        CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
        return;
    }

    throw InferenceException(context + ": equation '" + equation +
                             "' is not supported by the OpenCL row (supported: ...pd,...qd->...pq and bchw,bnc->bnhw)");
}

} // namespace opencl_kernels
} // namespace nert
