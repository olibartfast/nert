// Einsum on CUDA: naive one-thread-per-output contraction.
//
// Mirrors the CPU oracle exactly, including the shared ellipsis expansion
// (nert/einsum.hpp), dtype/rank/shape checks, and double accumulation with
// a single float32 rounding at the end. The host resolves labels into
// per-operand strides and a contracted-label list; the device kernel
// decomposes each output index and sums the contraction in the same
// row-major order as the CPU odometer, so results are bit-comparable.
//
// Pilot limit: at most 32 distinct labels (two operands x rank 16, the
// established broadcast cap). Larger equations are rejected on the host.

#include "nert/device.hpp"
#include "nert/einsum.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <vector>

namespace nert {
namespace cuda_kernels {
namespace {

constexpr int kMaxLabels = 32;

void CheckCuda(cudaError_t status, const char* what, const std::string& context) {
    if (status != cudaSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + cudaGetErrorString(status));
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

struct Equation {
    std::vector<std::string> inputs;
    std::string output;
    std::vector<char> fresh;
    std::vector<std::size_t> pads;
    std::vector<std::size_t> lefts;
    std::size_t width = 0;
};

Equation ParseEquation(const Node& node, const std::vector<TensorView>& inputs, const std::string& context) {
    const Attribute* attr = nullptr;
    for (const Attribute& a : node.attributes) {
        if (a.name == "equation") {
            attr = &a;
        }
    }
    if (attr == nullptr) {
        throw InferenceException(context + ": missing 'equation' attribute");
    }
    const std::string* raw = std::get_if<std::string>(&attr->value);
    if (raw == nullptr) {
        throw InferenceException(context + ": attribute 'equation' must be a string");
    }
    std::string text;
    for (char c : *raw) {
        if (c != ' ') {
            text.push_back(c);
        }
    }
    Equation eq;
    std::string equation_text = text;
    if (text.find('.') != std::string::npos) {
        const std::size_t arrow = text.find("->");
        if (arrow == std::string::npos) {
            throw InferenceException(context + ": implicit output with ellipsis is not supported");
        }
        std::vector<std::size_t> ranks;
        for (const TensorView& in : inputs) {
            ranks.push_back(in.dims.size());
        }
        const nert::einsum::Expansion expansion = nert::einsum::ExpandEllipsis(text, ranks);
        if (!expansion.ok) {
            throw InferenceException(context + ": " + expansion.error);
        }
        equation_text = "";
        for (std::size_t i = 0; i < expansion.inputs.size(); ++i) {
            if (i != 0) {
                equation_text.push_back(',');
            }
            equation_text += expansion.inputs[i];
        }
        equation_text += "->" + expansion.output;
        eq.fresh = expansion.fresh;
        eq.pads = expansion.pads;
        eq.lefts = expansion.lefts;
        eq.width = expansion.width;
    }
    const std::size_t arrow = equation_text.find("->");
    const bool explicitOutput = arrow != std::string::npos;
    const std::string lhs = explicitOutput ? equation_text.substr(0, arrow) : equation_text;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = lhs.find(',', start);
        eq.inputs.push_back(lhs.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    std::map<char, int> counts;
    for (const std::string& sub : eq.inputs) {
        for (char c : sub) {
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
                throw InferenceException(context + ": invalid character in the equation");
            }
            ++counts[c];
        }
    }
    if (explicitOutput) {
        eq.output = equation_text.substr(arrow + 2);
        for (std::size_t i = 0; i < eq.output.size(); ++i) {
            const char c = eq.output[i];
            if (counts.find(c) == counts.end()) {
                throw InferenceException(context + ": output label does not appear in the inputs");
            }
            if (eq.output.find(c) != i) {
                throw InferenceException(context + ": output label repeated");
            }
        }
    } else {
        for (const auto& entry : counts) {
            if (entry.second == 1) {
                eq.output.push_back(entry.first);
            }
        }
    }
    return eq;
}

// Kernel parameters are passed by value (about 1.4 KB, under the 4 KB
// kernel-parameter limit): no per-call device metadata allocation.
struct EinsumParams {
    int32_t label_count = 0;
    int32_t op_count = 0;
    int32_t contract_labels = 0;
    int32_t contracted[kMaxLabels] = {};
    int64_t sizes[kMaxLabels] = {};
    int64_t out_stride[kMaxLabels] = {};
    int64_t contract_stride[kMaxLabels] = {};
    int64_t op_stride[2][kMaxLabels] = {};
    int64_t contract_extent = 0;
    int64_t out_count = 0;
};

__global__ void EinsumKernel(const float* x0, const float* x1, float* y, EinsumParams p) {
    const int64_t out_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_index >= p.out_count) {
        return;
    }
    int64_t out_part[2] = {0, 0};
    for (int32_t l = 0; l < p.label_count; ++l) {
        if (p.out_stride[l] < 0) {
            continue;
        }
        const int64_t coord = (out_index / p.out_stride[l]) % p.sizes[l];
        out_part[0] += coord * p.op_stride[0][l];
        out_part[1] += coord * p.op_stride[1][l];
    }
    double acc = 0.0;
    for (int64_t c = 0; c < p.contract_extent; ++c) {
        int64_t off0 = out_part[0];
        int64_t off1 = out_part[1];
        for (int32_t k = 0; k < p.contract_labels; ++k) {
            const int32_t l = p.contracted[k];
            const int64_t coord = (c / p.contract_stride[k]) % p.sizes[l];
            off0 += coord * p.op_stride[0][l];
            off1 += coord * p.op_stride[1][l];
        }
        double prod = 1.0;
        if (p.op_count > 0) {
            prod *= static_cast<double>(x0[off0]);
        }
        if (p.op_count > 1) {
            prod *= static_cast<double>(x1[off1]);
        }
        acc += prod;
    }
    y[out_index] = static_cast<float>(acc);
}

} // namespace

void Einsum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Einsum (node '" + node.name + "')";
    const Equation eq = ParseEquation(node, inputs, context);
    if (eq.inputs.size() != inputs.size() || inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": operand count does not match the equation");
    }
    const TensorView& out = outputs[0];
    if (out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }

    std::vector<std::vector<nert::einsum::MappedDim>> mapped(inputs.size());
    std::map<char, int64_t> sizes;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].dtype != DataType::Float32) {
            throw InferenceException(context + ": tensors must be float32");
        }
        if (inputs[i].data == nullptr) {
            throw InferenceException(context + ": null buffer");
        }
        const std::size_t left = i < eq.lefts.size() ? eq.lefts[i] : eq.inputs[i].size();
        const std::size_t pad = i < eq.pads.size() ? eq.pads[i] : 0;
        if (!nert::einsum::MapOperand(eq.inputs[i], left, pad, eq.width, inputs[i].dims, mapped[i])) {
            throw InferenceException(context + ": operand rank does not match its subscript");
        }
        auto is_fresh = [&](char c) { return std::find(eq.fresh.begin(), eq.fresh.end(), c) != eq.fresh.end(); };
        for (const nert::einsum::MappedDim& entry : mapped[i]) {
            if (entry.virtual_pad) {
                const auto inserted = sizes.emplace(entry.label, 1);
                if (!inserted.second && inserted.first->second != 1) {
                    inserted.first->second = std::max<int64_t>(inserted.first->second, 1);
                }
                continue;
            }
            if (entry.dim <= 0) {
                throw InferenceException(context + ": non-positive dimension");
            }
            const auto inserted = sizes.emplace(entry.label, entry.dim);
            if (!inserted.second && inserted.first->second != entry.dim) {
                if (!is_fresh(entry.label) || (inserted.first->second != 1 && entry.dim != 1)) {
                    throw InferenceException(context + ": label bound to two different sizes");
                }
                inserted.first->second = std::max(inserted.first->second, entry.dim);
            }
        }
    }
    std::vector<int64_t> expected;
    for (char c : eq.output) {
        expected.push_back(sizes.at(c));
    }
    if (out.dims != expected) {
        throw InferenceException(context + ": output shape does not match the operands");
    }
    if (out.data == nullptr) {
        throw InferenceException(context + ": null buffer");
    }

    std::vector<char> labels;
    for (const auto& entry : sizes) {
        labels.push_back(entry.first);
    }
    if (labels.size() > static_cast<std::size_t>(kMaxLabels)) {
        throw InferenceException(context + ": equation has more than 32 labels");
    }
    const std::size_t n = labels.size();
    std::vector<std::vector<int64_t>> opStride(inputs.size(), std::vector<int64_t>(n, 0));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        int64_t stride = 1;
        for (std::size_t k = mapped[i].size(); k-- > 0;) {
            const nert::einsum::MappedDim& entry = mapped[i][k];
            const std::size_t li =
                static_cast<std::size_t>(std::find(labels.begin(), labels.end(), entry.label) - labels.begin());
            if (entry.virtual_pad || entry.dim != sizes.at(entry.label)) {
                const bool fresh = std::find(eq.fresh.begin(), eq.fresh.end(), entry.label) != eq.fresh.end();
                if (!entry.virtual_pad && (!fresh || entry.dim != 1)) {
                    throw InferenceException(context + ": broadcast diagonal is not supported");
                }
            } else {
                opStride[i][li] += stride;
            }
            stride *= entry.dim;
        }
    }
    std::vector<int64_t> outStride(n, -1);
    std::int64_t outCount = 1;
    {
        int64_t stride = 1;
        for (std::size_t j = eq.output.size(); j-- > 0;) {
            const std::size_t li =
                static_cast<std::size_t>(std::find(labels.begin(), labels.end(), eq.output[j]) - labels.begin());
            outStride[li] = stride;
            stride *= sizes.at(eq.output[j]);
        }
        outCount = stride;
    }

    EinsumParams p;
    p.label_count = static_cast<int32_t>(n);
    p.op_count = static_cast<int32_t>(inputs.size());
    p.contract_extent = 1;
    for (std::size_t l = 0; l < n; ++l) {
        p.sizes[l] = sizes.at(labels[l]);
        p.out_stride[l] = outStride[l];
        p.op_stride[0][l] = opStride[0][l];
        p.op_stride[1][l] = inputs.size() > 1 ? opStride[1][l] : 0;
    }
    for (std::size_t l = 0; l < n; ++l) {
        if (outStride[l] >= 0) {
            continue;
        }
        const int32_t k = p.contract_labels++;
        p.contracted[k] = static_cast<int32_t>(l);
    }
    for (int32_t k = p.contract_labels; k-- > 0;) {
        const int32_t l = p.contracted[k];
        p.contract_stride[k] = p.contract_extent;
        p.contract_extent *= p.sizes[l];
    }
    p.out_count = outCount;
    if (outCount == 0) {
        return;
    }

    const int block = 256;
    const int grid = static_cast<int>((outCount + block - 1) / block);
    const float* x0 = static_cast<const float*>(inputs[0].data);
    const float* x1 = inputs.size() > 1 ? static_cast<const float*>(inputs[1].data) : nullptr;
    EinsumKernel<<<grid, block>>>(x0, x1, static_cast<float*>(out.data), p);
    CheckCuda(cudaGetLastError(), "EinsumKernel launch", context);
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", context);
}

} // namespace cuda_kernels
} // namespace nert
