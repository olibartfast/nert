// Hipified from src/cuda/gather.cu via hipify-perl, hand-reviewed.
// Gather on HIP: indexed copy along one axis, one thread per output
// element.
//
// Mirrors the CPU oracle: f32/i64 sharing one dtype, int64 indices, axis
// attr default 0 with negative wrap, negative index wrap. An out-of-range
// index is flagged on device and reported on the host (Div precedent),
// since device code cannot throw.

#include "nert/device.hpp"

#include <cstddef>
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

template <typename T>
__global__ void GatherKernel(const T* x, const int64_t* idx, T* y, int64_t outCount, int64_t indexCount, int64_t inner,
                             int64_t dim, int* oobFlag) {
    const int64_t row = indexCount * inner;
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outCount; i += blockDim.x * gridDim.x) {
        const int64_t o = i / row;
        const int64_t r = i % row;
        const int64_t j = r / inner;
        const int64_t n = r % inner;
        int64_t k = idx[j];
        if (k < 0) {
            k += dim;
        }
        if (k < 0 || k >= dim) {
            atomicExch(oobFlag, 1);
            y[i] = 0;
            continue;
        }
        y[i] = x[(o * dim + k) * inner + n];
    }
}

template <typename T>
void LaunchGather(const T* x, const int64_t* idx, T* y, int64_t outCount, int64_t indexCount, int64_t inner,
                  int64_t dim, int* oobFlag, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    GatherKernel<<<grid, block>>>(x, idx, y, outCount, indexCount, inner, dim, oobFlag);
    CheckHip(hipGetLastError(), "GatherKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void Gather(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Gather (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data and indices inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& indices = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64 && in.dtype != DataType::Bool) {
        throw InferenceException(context + ": tensors must be float32, int64, or bool");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": Gather indices must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (rank == 0) {
        throw InferenceException(context + ": Gather axis is out of range");
    }
    int64_t axis = 0;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "axis") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'axis' must be an integer");
            }
            axis = *value;
        }
    }
    if (axis < -rank || axis >= rank) {
        throw InferenceException(context + ": Gather axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }

    std::vector<int64_t> expected;
    for (int64_t i = 0; i < axis; ++i) {
        expected.push_back(in.dims[static_cast<std::size_t>(i)]);
    }
    expected.insert(expected.end(), indices.dims.begin(), indices.dims.end());
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < in.dims.size(); ++i) {
        expected.push_back(in.dims[i]);
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the gathered indices");
    }

    const int64_t index_count = ElementCount(indices.dims, context);
    const int64_t out_count = ElementCount(out.dims, context);
    (void)ElementCount(in.dims, context);
    if (index_count > 0 && indices.data == nullptr) {
        throw InferenceException(context + ": null indices buffer");
    }
    if (out_count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (out_count == 0) {
        return;
    }

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        outer *= in.dims[static_cast<std::size_t>(i)];
    }
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        inner *= in.dims[static_cast<std::size_t>(i)];
    }
    (void)outer;
    int* oobFlag = nullptr;
    CheckHip(hipMalloc(&oobFlag, sizeof(int)), "hipMalloc", context);
    CheckHip(hipMemset(oobFlag, 0, sizeof(int)), "hipMemset", context);
    if (in.dtype == DataType::Float32) {
        LaunchGather(static_cast<const float*>(in.data), static_cast<const int64_t*>(indices.data),
                     static_cast<float*>(out.data), out_count, index_count, inner, dim, oobFlag, context);
    } else if (in.dtype == DataType::Int64) {
        LaunchGather(static_cast<const int64_t*>(in.data), static_cast<const int64_t*>(indices.data),
                     static_cast<int64_t*>(out.data), out_count, index_count, inner, dim, oobFlag, context);
    } else {
        LaunchGather(static_cast<const bool*>(in.data), static_cast<const int64_t*>(indices.data),
                     static_cast<bool*>(out.data), out_count, index_count, inner, dim, oobFlag, context);
    }
    int flagged = 0;
    CheckHip(hipMemcpy(&flagged, oobFlag, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy", context);
    CheckHip(hipFree(oobFlag), "hipFree", context);
    if (flagged != 0) {
        throw InferenceException(context + ": Gather index is out of range");
    }
}

} // namespace hip_kernels
} // namespace nert
