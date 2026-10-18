// Cast on OpenCL: float32/int64/bool conversion per the `to` attribute.
//
// Mirrors the CPU oracle: `to = 6` (INT32) is emulated on int64 lanes with
// saturation, float-to-int truncates toward zero, any nonzero value is true.
// Same-dtype casts are buffer copies; every other pair has a dedicated
// kernel and cache key.

#include "nert/device.hpp"
#include "runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
namespace {

constexpr int64_t kElemFloat = 1;
constexpr int64_t kElemInt32 = 6;
constexpr int64_t kElemInt64 = 7;
constexpr int64_t kElemBool = 9;

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

int64_t ToAttribute(const Node& node, const std::string& context) {
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "to") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'to' must be an integer");
            }
            return *value;
        }
    }
    throw InferenceException(context + ": missing required attribute 'to'");
}

constexpr const char* kSource = R"CLC(
__kernel void cast_f2i(__global const float* x, __global long* y) {
    y[get_global_id(0)] = (long)x[get_global_id(0)];
}
__kernel void cast_f2i32(__global const float* x, __global long* y) {
    y[get_global_id(0)] = (long)convert_int_sat(x[get_global_id(0)]);
}
__kernel void cast_f2b(__global const float* x, __global uchar* y) {
    y[get_global_id(0)] = x[get_global_id(0)] != 0.0f ? 1 : 0;
}
__kernel void cast_i2f(__global const long* x, __global float* y) {
    y[get_global_id(0)] = (float)x[get_global_id(0)];
}
__kernel void cast_i2i32(__global const long* x, __global long* y) {
    long v = x[get_global_id(0)];
    y[get_global_id(0)] = v > 2147483647L ? 2147483647L : (v < -2147483648L ? -2147483648L : v);
}
__kernel void cast_i2b(__global const long* x, __global uchar* y) {
    y[get_global_id(0)] = x[get_global_id(0)] != 0 ? 1 : 0;
}
__kernel void cast_b2f(__global const uchar* x, __global float* y) {
    y[get_global_id(0)] = x[get_global_id(0)] != 0 ? 1.0f : 0.0f;
}
__kernel void cast_b2i(__global const uchar* x, __global long* y) {
    y[get_global_id(0)] = x[get_global_id(0)] != 0 ? 1 : 0;
}
)CLC";

void LaunchCopy(const std::string& context, cl_mem x, cl_mem y, int64_t count, std::size_t element) {
    CheckCl(clEnqueueCopyBuffer(opencl_runtime::Queue(), x, y, 0, 0, static_cast<std::size_t>(count) * element, 0,
                                nullptr, nullptr),
            "clEnqueueCopyBuffer", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace

void Cast(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Cast (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": input must be float32, int64, or bool");
    }
    const int64_t to = ToAttribute(node, context);
    DataType want = DataType::Unknown;
    if (to == kElemFloat) {
        want = DataType::Float32;
    } else if (to == kElemInt32 || to == kElemInt64) {
        want = DataType::Int64;
    } else if (to == kElemBool) {
        want = DataType::Bool;
    } else {
        throw InferenceException(context + ": attribute 'to' names an unsupported element type");
    }
    if (out.dtype != want) {
        throw InferenceException(context + ": output dtype does not match attribute 'to'");
    }
    const int64_t count = ElementCount(in.dims, context);
    if (in.dims != out.dims) {
        throw InferenceException(context + ": output shape must match the input shape");
    }
    (void)ElementCount(out.dims, context);
    if (in.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    cl_mem x = reinterpret_cast<cl_mem>(const_cast<void*>(in.data));
    cl_mem y = reinterpret_cast<cl_mem>(out.data);
    if (want == DataType::Float32 && in.dtype == DataType::Float32) {
        LaunchCopy(context, x, y, count, sizeof(float));
        return;
    }
    if (want == DataType::Int64 && in.dtype == DataType::Int64 && to != kElemInt32) {
        LaunchCopy(context, x, y, count, sizeof(int64_t));
        return;
    }
    if (want == DataType::Bool && in.dtype == DataType::Bool) {
        LaunchCopy(context, x, y, count, sizeof(bool));
        return;
    }

    const char* entry = nullptr;
    const char* cache_key = nullptr;
    TensorView src = in;
    TensorView dst = out;
    if (want == DataType::Float32) {
        entry = in.dtype == DataType::Int64 ? "cast_i2f" : "cast_b2f";
        cache_key = in.dtype == DataType::Int64 ? "CastI2F" : "CastB2F";
    } else if (want == DataType::Int64) {
        if (to == kElemInt32) {
            entry = in.dtype == DataType::Float32 ? "cast_f2i32"
                    : in.dtype == DataType::Int64 ? "cast_i2i32"
                                                  : "cast_b2i";
            cache_key = in.dtype == DataType::Float32 ? "CastF2I32"
                        : in.dtype == DataType::Int64 ? "CastI2I32"
                                                      : "CastB2I";
        } else {
            entry = in.dtype == DataType::Float32 ? "cast_f2i" : "cast_b2i";
            cache_key = in.dtype == DataType::Float32 ? "CastF2I64" : "CastB2I";
        }
    } else {
        entry = in.dtype == DataType::Float32 ? "cast_f2b" : "cast_i2b";
        cache_key = in.dtype == DataType::Float32 ? "CastF2B" : "CastI2B";
    }
    (void)src;
    (void)dst;
    cl_kernel kernel = opencl_runtime::KernelFor(cache_key, kSource, entry);
    CheckCl(clSetKernelArg(kernel, 0, sizeof(x), &x), "clSetKernelArg", context);
    CheckCl(clSetKernelArg(kernel, 1, sizeof(y), &y), "clSetKernelArg", context);
    const std::size_t global = static_cast<std::size_t>(count);
    CheckCl(clEnqueueNDRangeKernel(opencl_runtime::Queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel", context);
    CheckCl(clFinish(opencl_runtime::Queue()), "clFinish", context);
}

} // namespace opencl_kernels
} // namespace nert
