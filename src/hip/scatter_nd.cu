// Hipified from src/cuda/scatter_nd.cu via hipify-perl, hand-reviewed.
// ScatterND on HIP: copy `data`, then scatter `updates` slices at
// `indices`.
//
// Mirrors the CPU oracle: float32/int64/bool sharing one dtype, int64
// indices with bounds + negative-wrap checks on the host, updates shape
// `indices.dims[:-1] + data.dims[K:]`, duplicates apply in order. The copy
// is one D2D transfer; each slice scatters with its own D2D transfer
// (ConcatCopy precedent), so ordering matches without atomics.

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

std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

} // namespace

void ScatterND(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "ScatterND (node '" + node.name + "')";
    if (inputs.size() != 3 || outputs.size() != 1) {
        throw InferenceException(context + ": expected data, indices, updates, and one output");
    }
    const TensorView& data = inputs[0];
    const TensorView& indices = inputs[1];
    const TensorView& updates = inputs[2];
    const TensorView& out = outputs[0];
    if (data.dtype != DataType::Float32 && data.dtype != DataType::Int64 && data.dtype != DataType::Bool) {
        throw InferenceException(context + ": data must be float32, int64, or bool");
    }
    if (updates.dtype != data.dtype || out.dtype != data.dtype) {
        throw InferenceException(context + ": data, updates, and output must share dtype");
    }
    if (indices.dtype != DataType::Int64) {
        throw InferenceException(context + ": indices must be int64");
    }
    if (indices.dims.empty()) {
        throw InferenceException(context + ": indices must have rank >= 1");
    }
    const int64_t dataRank = static_cast<int64_t>(data.dims.size());
    const int64_t k = indices.dims.back();
    if (k < 1 || k > dataRank) {
        throw InferenceException(context + ": indices last dim must be within the data rank");
    }
    if (out.dims != data.dims) {
        throw InferenceException(context + ": output shape must match the data shape");
    }
    std::vector<int64_t> expectedUpdates(indices.dims.begin(), indices.dims.end() - 1);
    for (int64_t i = k; i < dataRank; ++i) {
        expectedUpdates.push_back(data.dims[static_cast<std::size_t>(i)]);
    }
    if (updates.dims != expectedUpdates) {
        throw InferenceException(context + ": updates shape does not match indices and data");
    }
    (void)ElementCount(data.dims, context);
    const int64_t indicesCount = ElementCount(indices.dims, context);
    (void)ElementCount(updates.dims, context);
    if (data.data == nullptr || indices.data == nullptr || updates.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const std::size_t elemSize = data.dtype == DataType::Float32 ? sizeof(float)
                                 : data.dtype == DataType::Int64 ? sizeof(int64_t)
                                                                 : sizeof(bool);
    const int64_t dataBytes = ElementCount(data.dims, context) * static_cast<int64_t>(elemSize);
    if (dataBytes > 0) {
        CheckHip(hipMemcpy(out.data, data.data, static_cast<std::size_t>(dataBytes), hipMemcpyDeviceToDevice),
                 "hipMemcpy", context);
    }

    const std::vector<int64_t> dataStrides = Strides(data.dims);
    int64_t sliceCount = 1;
    for (std::size_t i = 0; i + 1 < indices.dims.size(); ++i) {
        sliceCount *= indices.dims[i];
    }
    int64_t sliceSize = 1;
    for (int64_t i = k; i < dataRank; ++i) {
        sliceSize *= data.dims[static_cast<std::size_t>(i)];
    }
    std::vector<int64_t> idx(static_cast<std::size_t>(indicesCount), 0);
    if (indicesCount > 0) {
        CheckHip(hipMemcpy(idx.data(), indices.data, static_cast<std::size_t>(indicesCount) * sizeof(int64_t),
                           hipMemcpyDeviceToHost),
                 "hipMemcpy", context);
    }
    auto* dst = static_cast<std::uint8_t*>(out.data);
    const auto* upd = static_cast<const std::uint8_t*>(updates.data);
    for (int64_t s = 0; s < sliceCount; ++s) {
        int64_t offset = 0;
        for (int64_t j = 0; j < k; ++j) {
            int64_t coord = idx[static_cast<std::size_t>(s * k + j)];
            const int64_t dim = data.dims[static_cast<std::size_t>(j)];
            if (coord < -dim || coord >= dim) {
                throw InferenceException(context + ": index out of range");
            }
            if (coord < 0) {
                coord += dim;
            }
            offset += coord * dataStrides[static_cast<std::size_t>(j)];
        }
        if (sliceSize > 0) {
            CheckHip(hipMemcpy(dst + static_cast<std::size_t>(offset) * elemSize,
                               upd + static_cast<std::size_t>(s * sliceSize) * elemSize,
                               static_cast<std::size_t>(sliceSize) * elemSize, hipMemcpyDeviceToDevice),
                     "hipMemcpy", context);
        }
    }
}

} // namespace hip_kernels
} // namespace nert
