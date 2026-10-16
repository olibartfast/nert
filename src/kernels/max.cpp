// Max: elementwise maximum over one or more inputs with NumPy multidirectional
// broadcasting. All inputs share one element type (float32 or int64); the
// output matches.

#include "kernels.hpp"

#include <algorithm>
#include <cstdint>
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

// The NumPy multidirectional broadcast of two shapes. Throws when the shapes
// cannot be broadcast together.
std::vector<int64_t> BroadcastDims(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                   const std::string& context) {
    const std::size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (std::size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t db = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (da != db && da != 1 && db != 1) {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

// Row-major strides for a contiguous tensor.
std::vector<int64_t> Strides(const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (std::size_t i = dims.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// For each output axis, the input stride that maps it (0 where the input axis
// is broadcast). Throws when the input shape cannot broadcast to the output.
std::vector<int64_t> AlignedStrides(const std::vector<int64_t>& inDims, const std::vector<int64_t>& outDims,
                                    const std::string& context) {
    const std::size_t outRank = outDims.size();
    const std::size_t inRank = inDims.size();
    const std::vector<int64_t> inStrides = Strides(inDims);
    std::vector<int64_t> aligned(outRank, 0);
    for (std::size_t j = 0; j < outRank; ++j) {
        if (j + inRank < outRank) {
            continue; // a missing leading axis broadcasts as a size-1 axis
        }
        const std::size_t k = j - (outRank - inRank);
        if (inDims[k] == outDims[j]) {
            aligned[j] = inStrides[k];
        } else if (inDims[k] == 1) {
            aligned[j] = 0; // broadcast
        } else {
            throw InferenceException(context + ": shapes are not broadcast-compatible");
        }
    }
    return aligned;
}

template <typename T>
void MaxTyped(const std::vector<const T*>& ins, T* y, int64_t outCount, const std::vector<int64_t>& outStrides,
              const std::vector<std::vector<int64_t>>& inStrides, std::size_t rank) {
    for (int64_t i = 0; i < outCount; ++i) {
        int64_t remaining = i;
        std::vector<int64_t> offsets(ins.size(), 0);
        for (std::size_t j = 0; j < rank; ++j) {
            const int64_t coord = remaining / outStrides[j];
            remaining %= outStrides[j];
            for (std::size_t k = 0; k < ins.size(); ++k) {
                offsets[k] += coord * inStrides[k][j];
            }
        }
        T best = ins[0][offsets[0]];
        for (std::size_t k = 1; k < ins.size(); ++k) {
            best = std::max(best, ins[k][offsets[k]]);
        }
        y[i] = best;
    }
}

} // namespace

void Max(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Max (node '" + node.name + "')";
    if (inputs.size() < 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected at least one input and one output");
    }
    const DataType dtype = inputs[0].dtype;
    if (dtype != DataType::Float32 && dtype != DataType::Int64) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }
    for (const TensorView& in : inputs) {
        if (in.dtype != dtype) {
            throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
        }
    }
    const TensorView& out = outputs[0];
    if (out.dtype != dtype) {
        throw InferenceException(context + ": tensors must share one float32 or int64 dtype");
    }

    std::vector<int64_t> acc = inputs[0].dims;
    (void)ElementCount(inputs[0].dims, context);
    for (std::size_t i = 1; i < inputs.size(); ++i) {
        (void)ElementCount(inputs[i].dims, context);
        acc = BroadcastDims(acc, inputs[i].dims, context);
    }
    if (acc != out.dims) {
        throw InferenceException(context + ": output shape does not match the broadcast of the inputs");
    }
    const int64_t outCount = ElementCount(out.dims, context);
    for (const TensorView& in : inputs) {
        if (in.data == nullptr) {
            throw InferenceException(context + ": null buffer");
        }
    }
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    const std::vector<int64_t> outStrides = Strides(out.dims);
    std::vector<std::vector<int64_t>> inStrides;
    inStrides.reserve(inputs.size());
    for (const TensorView& in : inputs) {
        inStrides.push_back(AlignedStrides(in.dims, out.dims, context));
    }
    if (dtype == DataType::Float32) {
        std::vector<const float*> ins;
        for (const TensorView& in : inputs) {
            ins.push_back(static_cast<const float*>(in.data));
        }
        MaxTyped(ins, static_cast<float*>(out.data), outCount, outStrides, inStrides, out.dims.size());
    } else {
        std::vector<const int64_t*> ins;
        for (const TensorView& in : inputs) {
            ins.push_back(static_cast<const int64_t*>(in.data));
        }
        MaxTyped(ins, static_cast<int64_t*>(out.data), outCount, outStrides, inStrides, out.dims.size());
    }
}

} // namespace kernels
} // namespace nert
