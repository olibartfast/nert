// ScatterND: copy `data`, then scatter `updates` slices at `indices`.
//
// `indices` is int64 with `K = indices.dims.back()` (1 <= K <= data rank);
// each K-tuple addresses `data[i0]...[i{K-1}]` with negative indices counting
// back. `updates` has shape `indices.dims[:-1] + data.dims[K:]`. Out-of-range
// indices throw; duplicate indices apply in order. Data is float32, int64,
// or bool; all three tensors share the data dtype for data/updates.

#include "kernels.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nert {
namespace kernels {
namespace {

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
    (void)ElementCount(indices.dims, context);
    (void)ElementCount(updates.dims, context);
    if (data.data == nullptr || indices.data == nullptr || updates.data == nullptr || out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
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
    const int64_t* idx = static_cast<const int64_t*>(indices.data);
    const std::size_t elemSize = data.dtype == DataType::Float32 ? sizeof(float)
                                 : data.dtype == DataType::Int64 ? sizeof(int64_t)
                                                                 : sizeof(bool);
    const char* srcData = static_cast<const char*>(data.data);
    char* dst = static_cast<char*>(out.data);
    const char* upd = updates.data == nullptr ? nullptr : static_cast<const char*>(updates.data);
    const int64_t dataCount = ElementCount(data.dims, context);
    for (int64_t i = 0; i < dataCount * static_cast<int64_t>(elemSize); ++i) {
        dst[i] = srcData[i];
    }
    for (int64_t s = 0; s < sliceCount; ++s) {
        int64_t offset = 0;
        for (int64_t j = 0; j < k; ++j) {
            int64_t coord = idx[s * k + j];
            const int64_t dim = data.dims[static_cast<std::size_t>(j)];
            if (coord < -dim || coord >= dim) {
                throw InferenceException(context + ": index out of range");
            }
            if (coord < 0) {
                coord += dim;
            }
            offset += coord * dataStrides[static_cast<std::size_t>(j)];
        }
        std::memcpy(dst + offset * elemSize, upd + s * sliceSize * elemSize,
                    static_cast<std::size_t>(sliceSize * static_cast<int64_t>(elemSize)));
    }
}

} // namespace kernels
} // namespace nert
