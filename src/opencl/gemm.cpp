// Gemm on OpenCL: Y = alpha * op(A) * op(B) + beta * C.
//
// Mirrors the CPU oracle: rank-2 float32 A/B, optional transA/transB,
// alpha/beta float attributes, bias C broadcast from [N] or [M,N]. One
// work-item per output element, accumulating in float32 with one
// rounding at the store (the CPU oracle accumulates in double, so
// comparisons use the bench tolerance).

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

int64_t IntAttribute(const Node& node, const std::string& name, int64_t fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Gemm: attribute '" + name + "' must be an integer (node '" + node.name +
                                         "')");
            }
            return *value;
        }
    }
    return fallback;
}

float FloatAttribute(const Node& node, const std::string& name, float fallback) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == name) {
            const float* value = std::get_if<float>(&attr.value);
            if (value == nullptr) {
                throw InferenceException("Gemm: attribute '" + name + "' must be a float (node '" + node.name + "')");
            }
            return *value;
        }
    }
    return fallback;
}

struct GemmParams {
    cl_long m;
    cl_long n;
    cl_long k;
    cl_int transA;
    cl_int transB;
    cl_int hasBias;
    cl_int biasRank;
    cl_float alpha;
    cl_float beta;
};

constexpr const char* kSource = R"CLC(
typedef struct {
    long m; long n; long k;
    int transA; int transB; int hasBias; int biasRank;
    float alpha; float beta;
} GemmParams;
__kernel void gemm_f(__global const float* a, __global const float* b, __global const float* c, __global float* y,
                     GemmParams p) {
    long idx = get_global_id(0);
    long total = p.m * p.n;
    if (idx < total) {
        long row = idx / p.n;
        long col = idx % p.n;
        float acc = 0.0f;
        for (long kk = 0; kk < p.k; ++kk) {
            float av = p.transA == 0 ? a[row * p.k + kk] : a[kk * p.m + row];
            float bv = p.transB == 0 ? b[kk * p.n + col] : b[col * p.k + kk];
            acc += av * bv;
        }
        float value = p.alpha * acc;
        if (p.hasBias != 0) {
            value += p.beta * (p.biasRank == 1 ? c[col] : c[row * p.n + col]);
        }
        y[idx] = value;
    }
}
)CLC";

} // namespace

void Gemm(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Gemm (node '" + node.name + "')";
    if (inputs.size() < 2 || inputs.size() > 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected two or three inputs and one output");
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    const TensorView& out = outputs[0];
    const bool hasC = inputs.size() == 3;
    if (a.dtype != DataType::Float32 || b.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (hasC && inputs[2].dtype != DataType::Float32) {
        throw InferenceException(context + ": bias tensor must be float32");
    }
    if (a.dims.size() != 2 || b.dims.size() != 2) {
        throw InferenceException(context + ": A and B must be rank-2");
    }
    (void)ElementCount(a.dims, context);
    (void)ElementCount(b.dims, context);

    const int64_t transA = IntAttribute(node, "transA", 0);
    const int64_t transB = IntAttribute(node, "transB", 0);
    const float alpha = FloatAttribute(node, "alpha", 1.0f);
    const float beta = FloatAttribute(node, "beta", 1.0f);

    const int64_t m = transA == 0 ? a.dims[0] : a.dims[1];
    const int64_t k = transA == 0 ? a.dims[1] : a.dims[0];
    const int64_t bK = transB == 0 ? b.dims[0] : b.dims[1];
    const int64_t n = transB == 0 ? b.dims[1] : b.dims[0];
    if (k != bK) {
        throw InferenceException(context + ": contraction dimension mismatch");
    }
    if (out.dims != std::vector<int64_t>{m, n}) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    (void)ElementCount(out.dims, context);

    const TensorView* c = nullptr;
    int64_t biasRank = 0;
    if (hasC) {
        const TensorView& bias = inputs[2];
        const bool rankOne = bias.dims.size() == 1 && bias.dims[0] == n;
        const bool rankTwo = bias.dims.size() == 2 && bias.dims[0] == m && bias.dims[1] == n;
        if (!rankOne && !rankTwo) {
            throw InferenceException(context + ": bias shape must be [N] or [M,N]");
        }
        (void)ElementCount(bias.dims, context);
        c = &bias;
        biasRank = static_cast<int64_t>(bias.dims.size());
    }
    if (a.data == nullptr || b.data == nullptr || out.data == nullptr || (hasC && c->data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }

    GemmParams p{};
    p.m = m;
    p.n = n;
    p.k = k;
    p.transA = transA == 0 ? 0 : 1;
    p.transB = transB == 0 ? 0 : 1;
    p.hasBias = hasC ? 1 : 0;
    p.biasRank = biasRank == 1 ? 1 : 2;
    p.alpha = alpha;
    p.beta = beta;

    cl_mem am = reinterpret_cast<cl_mem>(const_cast<void*>(a.data));
    cl_mem bm = reinterpret_cast<cl_mem>(const_cast<void*>(b.data));
    cl_mem cm = hasC ? reinterpret_cast<cl_mem>(const_cast<void*>(c->data)) : nullptr;
    cl_mem ym = reinterpret_cast<cl_mem>(out.data);
    cl_kernel kernel = opencl_runtime::KernelFor("Gemm", kSource, "gemm_f");
    CheckCl(clSetKernelArg(kernel, 0, sizeof(am), &am), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(bm), &bm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 2, sizeof(cm), &cm), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 3, sizeof(ym), &ym), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 4, sizeof(p), &p), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(m * n);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
