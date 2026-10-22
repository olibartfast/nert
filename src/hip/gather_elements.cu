// Hipified from src/cuda/gather_elements.cu via hipify-perl, hand-reviewed.
// GatherElements on HIP: per-element indexed copy, one thread per output.
//
// Mirrors the CPU oracle: f32/i64 sharing one dtype, int64 indices sharing
// the data rank, axis attr default 1 with negative wrap, output shape equal
// to the indices shape, negative index wrap. Out-of-range positions are
// flagged on device and reported on the host (Div precedent), since device
// code cannot throw. Ranks above 16 are rejected (pilot limit).

#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

struct IndexSet {
    int64_t s[kMaxRank];
};

IndexSet Pack(const std::vector<int64_t>& values, const std::string& context) {
    if (values.size() > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the gather limit of 16");
    }
    IndexSet packed{};
    for (std::size_t j = 0; j < values.size(); ++j) {
        packed.s[j] = values[j];
    }
    return packed;
}

template <typename T>
__global__ void GatherElementsKernel(const T* x, const int64_t* idx, T* y, int64_t count, IndexSet strides,
                                     IndexSet inStrides, IndexSet inDims, int axis, int64_t dim, int rank,
                                     int* oobFlag) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int64_t remaining = i;
        int64_t offset = 0;
        for (int j = 0; j < rank; ++j) {
            const int64_t coord = remaining / strides.s[j];
            remaining %= strides.s[j];
            if (j != axis) {
                if (coord >= inDims.s[j]) {
                    atomicExch(oobFlag, 1);
                    y[i] = 0;
                    goto next;
                }
                offset += coord * inStrides.s[j];
            }
        }
        {
            int64_t k = idx[i];
            if (k < 0) {
                k += dim;
            }
            if (k < 0 || k >= dim) {
                atomicExch(oobFlag, 1);
                y[i] = 0;
                goto next;
            }
            offset += k * inStrides.s[axis];
        }
        y[i] = x[offset];
    next:;
    }
}

template <typename T>
void LaunchGatherElements(const T* x, const int64_t* idx, T* y, const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& inStrides, const std::vector<int64_t>& inDims, int64_t axis,
                          int64_t dim, int64_t count, int64_t rank, int* oobFlag, const std::string& context) {
    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    GatherElementsKernel<<<grid, block>>>(x, idx, y, count, Pack(strides, context), Pack(inStrides, context),
                                          Pack(inDims, context), static_cast<int>(axis), dim, static_cast<int>(rank),
                                          oobFlag);
    CheckHip(hipGetLastError(), "GatherElementsKernel launch", context);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize", context);
}

} // namespace

void GatherElements(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "GatherElements (node '" + node.name + "')";
    if (inputs.size() != 2 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data and indices inputs and one output");
    }
    const TensorView& in = inputs[0];
    const TensorView& indices = inputs[1];
    const TensorView& out = outputs[0];
    if (in.dtype != DataType::Float32 && in.dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must be float32 or int64");
    }
    if (out.dtype != in.dtype) {
        throw InferenceException(context + ": output dtype must match the input dtype");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": GatherElements indices must be int64");
    }
    const int64_t rank = static_cast<int64_t>(in.dims.size());
    if (static_cast<int64_t>(indices.dims.size()) != rank) {
        throw InferenceException(context + ": GatherElements data and indices must share rank");
    }
    if (static_cast<std::size_t>(rank) > kMaxRank) {
        throw InferenceException(context + ": rank exceeds the gather limit of 16");
    }
    int64_t axis = 1;
    for (const Attribute& attr : node.attributes) {
        if (attr.name == "axis") {
            const int64_t* value = std::get_if<int64_t>(&attr.value);
            if (value == nullptr) {
                throw InferenceException(context + ": attribute 'axis' must be an integer");
            }
            axis = *value;
        }
    }
    if (rank == 0 || axis < -rank || axis >= rank) {
        throw InferenceException(context + ": GatherElements axis is out of range");
    }
    if (axis < 0) {
        axis += rank;
    }
    if (out.dims != indices.dims) {
        throw InferenceException(context + ": output shape must match the indices shape");
    }

    const int64_t count = ElementCount(indices.dims, context);
    (void)ElementCount(in.dims, context);
    (void)ElementCount(out.dims, context);
    if (count > 0 && indices.data == nullptr) {
        throw InferenceException(context + ": null indices buffer");
    }
    if (count > 0 && (in.data == nullptr || out.data == nullptr)) {
        throw InferenceException(context + ": null buffer");
    }
    if (count == 0) {
        return;
    }

    const std::vector<int64_t> strides = Strides(indices.dims);
    const std::vector<int64_t> in_strides = Strides(in.dims);
    const int64_t dim = in.dims[static_cast<std::size_t>(axis)];
    int* oobFlag = nullptr;
    CheckHip(hipMalloc(&oobFlag, sizeof(int)), "hipMalloc", context);
    CheckHip(hipMemset(oobFlag, 0, sizeof(int)), "hipMemset", context);
    if (in.dtype == DataType::Float32) {
        LaunchGatherElements(static_cast<const float*>(in.data), static_cast<const int64_t*>(indices.data),
                             static_cast<float*>(out.data), strides, in_strides, in.dims, axis, dim, count, rank,
                             oobFlag, context);
    } else {
        LaunchGatherElements(static_cast<const int64_t*>(in.data), static_cast<const int64_t*>(indices.data),
                             static_cast<int64_t*>(out.data), strides, in_strides, in.dims, axis, dim, count, rank,
                             oobFlag, context);
    }
    int flagged = 0;
    CheckHip(hipMemcpy(&flagged, oobFlag, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy", context);
    CheckHip(hipFree(oobFlag), "hipFree", context);
    if (flagged != 0) {
        throw InferenceException(context + ": GatherElements index is out of range");
    }
}

} // namespace hip_kernels
} // namespace nert
