// GlobalAveragePool: the ONNX whole-spatial average.
//
// Input is [N, C, *spatial] with at least the batch and channel axes; the
// output is [N, C, 1, ..., 1] with one trailing 1 per spatial axis. Each
// (n, c) output is the mean of its spatial plane, accumulated in double
// with one float32 rounding — the same accumulation the CUDA row uses, so
// the two match bit for bit. Naive, float32, single-threaded.

#include "kernels.hpp"

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

} // namespace

void GlobalAveragePool(const Node& node, const std::vector<TensorView>& inputs,
                       const std::vector<TensorView>& outputs) {
    const std::string context = "GlobalAveragePool (node '" + node.name + "')";
    if (inputs.size() != 1 || outputs.size() != 1) {
        throw InferenceException(context + ": expected one input and one output");
    }
    const TensorView& x = inputs[0];
    const TensorView& out = outputs[0];
    if (x.dtype != DataType::Float32 || out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }
    if (x.dims.size() < 2) {
        throw InferenceException(context + ": input must have batch and channel axes (rank at least 2)");
    }
    (void)ElementCount(x.dims, context);
    (void)ElementCount(out.dims, context);

    std::vector<int64_t> expected = {x.dims[0], x.dims[1]};
    expected.insert(expected.end(), x.dims.size() - 2, 1);
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape must be the input shape with all spatial axes at 1");
    }

    const float* in = static_cast<const float*>(x.data);
    float* dst = static_cast<float*>(out.data);
    const int64_t rows = x.dims[0] * x.dims[1];
    int64_t spatial = 1;
    for (std::size_t i = 2; i < x.dims.size(); ++i) {
        spatial *= x.dims[i];
    }
    for (int64_t row = 0; row < rows; ++row) {
        double total = 0.0;
        for (int64_t k = 0; k < spatial; ++k) {
            total += static_cast<double>(in[row * spatial + k]);
        }
        dst[row] = static_cast<float>(total / static_cast<double>(spatial));
    }
}

} // namespace kernels
} // namespace nert
