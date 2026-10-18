// Einsum: Einstein summation over one or two float32 operands.
//
// The equation attribute lists one subscript per operand and optionally an
// explicit output after "->". Without "->" the output is every label that
// appears exactly once across the inputs, in alphabetical order. Labels absent
// from the output are summed over, and a label repeated within one operand
// selects its diagonal. Ellipsis is rejected. Spaces are ignored. Naive,
// single-threaded; products and sums accumulate in double.

#include "nert/einsum.hpp"

#include "kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nert {
namespace kernels {
namespace {

struct Equation {
    std::vector<std::string> inputs;
    std::string output;
    // Ellipsis bookkeeping (empty/zero on the plain path): fresh broadcast
    // labels, per-operand virtual-pad counts and left sizes, batch width.
    std::vector<char> fresh;
    std::vector<std::size_t> pads;
    std::vector<std::size_t> lefts;
    std::size_t width = 0;
};

Equation ParseEquation(const Node& node, const std::vector<TensorView>& inputs, const std::string& context,
                       std::vector<char>& fresh) {
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
    // Ellipsis equations expand to plain labels once ranks are known; the
    // rest of this function then runs unchanged. Fresh labels broadcast
    // (size 1), user labels keep the strict equal-size rule.
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
    fresh = eq.fresh;
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

} // namespace

void Einsum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs) {
    const std::string context = "Einsum (node '" + node.name + "')";
    std::vector<char> fresh_labels;
    const Equation eq = ParseEquation(node, inputs, context, fresh_labels);
    if (eq.inputs.size() != inputs.size() || inputs.empty() || inputs.size() > 2 || outputs.size() != 1) {
        throw InferenceException(context + ": operand count does not match the equation");
    }
    const TensorView& out = outputs[0];
    if (out.dtype != DataType::Float32) {
        throw InferenceException(context + ": tensors must be float32");
    }

    // Per-operand mapping (plain path: identity — left is the full length,
    // pads and width are zero, so every position is real).
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
        auto is_fresh = [&](char c) {
            return std::find(fresh_labels.begin(), fresh_labels.end(), c) != fresh_labels.end();
        };
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
            if (entry.dim <= 0) {
                throw InferenceException(context + ": non-positive dimension");
            }
            const auto inserted = sizes.emplace(entry.label, entry.dim);
            if (!inserted.second && inserted.first->second != entry.dim) {
                // Ellipsis batch dims broadcast against size 1; user labels
                // keep the strict equal-size rule.
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

    // Loop labels: every distinct label, with a per-operand and output stride.
    std::vector<char> labels;
    for (const auto& entry : sizes) {
        labels.push_back(entry.first);
    }
    const std::size_t n = labels.size();
    std::vector<std::vector<int64_t>> opStride(inputs.size(), std::vector<int64_t>(n, 0));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        int64_t stride = 1;
        for (std::size_t k = mapped[i].size(); k-- > 0;) {
            const nert::einsum::MappedDim& entry = mapped[i][k];
            const std::size_t li =
                static_cast<std::size_t>(std::find(labels.begin(), labels.end(), entry.label) - labels.begin());
            // A broadcast (size-1) or virtual-pad ellipsis plane contributes
            // no stride, so it stays on its single plane while the label
            // loops. Real size mismatches still fail.
            if (entry.virtual_pad || entry.dim != sizes.at(entry.label)) {
                const bool fresh =
                    std::find(fresh_labels.begin(), fresh_labels.end(), entry.label) != fresh_labels.end();
                if (!entry.virtual_pad && (!fresh || entry.dim != 1)) {
                    throw InferenceException(context + ": broadcast diagonal is not supported");
                }
            } else {
                opStride[i][li] += stride;
            }
            stride *= entry.dim;
        }
    }
    std::vector<int64_t> outStride(n, 0);
    int64_t outCount = 1;
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

    // Products and sums accumulate in double; each output rounds once to float32.
    float* y = static_cast<float*>(out.data);
    std::vector<double> acc(static_cast<std::size_t>(outCount), 0.0);

    int64_t total = 1;
    for (char c : labels) {
        total *= sizes.at(c);
    }
    std::vector<int64_t> coord(n, 0);
    std::vector<int64_t> offset(inputs.size(), 0);
    int64_t outOffset = 0;
    for (int64_t step = 0; step < total; ++step) {
        double prod = 1.0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            prod *= static_cast<double>(static_cast<const float*>(inputs[i].data)[offset[i]]);
        }
        acc[static_cast<std::size_t>(outOffset)] += prod;
        // Advance the odometer, keeping the running offsets in sync.
        for (std::size_t li = n; li-- > 0;) {
            ++coord[li];
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                offset[i] += opStride[i][li];
            }
            outOffset += outStride[li];
            if (coord[li] < sizes.at(labels[li])) {
                break;
            }
            const int64_t size = sizes.at(labels[li]);
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                offset[i] -= opStride[i][li] * size;
            }
            outOffset -= outStride[li] * size;
            coord[li] = 0;
        }
    }
    for (int64_t i = 0; i < outCount; ++i) {
        y[i] = static_cast<float>(acc[static_cast<std::size_t>(i)]);
    }
}

} // namespace kernels

namespace einsum {
namespace {

bool IsAlpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// Splits one subscript around its "..." (at most one; syntax-checked).
// Returns false when the dots are malformed.
bool SplitEllipsis(const std::string& sub, std::string& left, std::string& right, bool& has_ellipsis) {
    has_ellipsis = false;
    left.clear();
    right.clear();
    std::size_t i = 0;
    while (i < sub.size() && sub[i] != '.') {
        left.push_back(sub[i++]);
    }
    if (i < sub.size()) {
        if (sub.compare(i, 3, "...") != 0) {
            return false;
        }
        has_ellipsis = true;
        i += 3;
        while (i < sub.size() && sub[i] != '.') {
            right.push_back(sub[i++]);
        }
        if (i < sub.size()) {
            return false; // second dotted group
        }
    }
    return true;
}

} // namespace

std::string EllipsisSyntaxError(const std::string& text) {
    const std::size_t arrow = text.find("->");
    const std::string lhs = arrow == std::string::npos ? text : text.substr(0, arrow);
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = lhs.find(',', start);
        const std::string sub = lhs.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        std::string left;
        std::string right;
        bool has_ellipsis = false;
        if (!SplitEllipsis(sub, left, right, has_ellipsis)) {
            return "dots must form exactly one '...' per subscript";
        }
        for (char c : left + right) {
            if (!IsAlpha(c)) {
                return "invalid character in the equation";
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (arrow != std::string::npos) {
        const std::string out = text.substr(arrow + 2);
        std::string left;
        std::string right;
        bool has_ellipsis = false;
        if (!SplitEllipsis(out, left, right, has_ellipsis)) {
            return "dots must form exactly one '...' per subscript";
        }
        for (char c : left + right) {
            if (!IsAlpha(c)) {
                return "invalid character in the equation";
            }
        }
    }
    return "";
}

Expansion ExpandEllipsis(const std::string& text, const std::vector<std::size_t>& ranks) {
    Expansion result;
    const std::string syntax = EllipsisSyntaxError(text);
    if (!syntax.empty()) {
        result.error = syntax;
        return result;
    }
    const std::size_t arrow = text.find("->");
    result.has_output = arrow != std::string::npos;
    const std::string lhs = result.has_output ? text.substr(0, arrow) : text;
    std::vector<std::string> subs;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = lhs.find(',', start);
        subs.push_back(lhs.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (subs.size() != ranks.size()) {
        result.error = "operand count does not match the equation";
        return result;
    }
    // Used labels first, so fresh ones never collide.
    bool used[256] = {false};
    for (char c : text) {
        if (IsAlpha(c)) {
            used[static_cast<unsigned char>(c)] = true;
        }
    }
    struct Split {
        std::string left;
        std::string right;
        bool has_ellipsis = false;
        std::size_t count = 0;
    };
    std::vector<Split> parts;
    std::size_t width = 0;
    for (std::size_t i = 0; i < subs.size(); ++i) {
        Split part;
        if (!SplitEllipsis(subs[i], part.left, part.right, part.has_ellipsis)) {
            result.error = "dots must form exactly one '...' per subscript";
            return result;
        }
        if (ranks[i] < part.left.size() + part.right.size()) {
            result.error = "operand rank is smaller than its explicit labels";
            return result;
        }
        part.count = part.has_ellipsis ? ranks[i] - part.left.size() - part.right.size() : 0;
        if (!part.has_ellipsis && ranks[i] != part.left.size() + part.right.size()) {
            result.error = "operand rank does not match its subscript";
            return result;
        }
        width = std::max(width, part.count);
        parts.push_back(part);
    }
    std::string pool;
    for (char c = 'a'; c <= 'z'; ++c) {
        if (!used[static_cast<unsigned char>(c)]) {
            pool.push_back(c);
        }
    }
    for (char c = 'A'; c <= 'Z'; ++c) {
        if (!used[static_cast<unsigned char>(c)]) {
            pool.push_back(c);
        }
    }
    if (pool.size() < width) {
        result.error = "equation leaves no fresh labels for the ellipsis";
        return result;
    }
    std::string fresh(pool.substr(0, width));
    result.fresh.assign(fresh.begin(), fresh.end());
    result.width = width;
    result.pads.assign(subs.size(), 0);
    result.lefts.assign(subs.size(), 0);
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const Split& part = parts[i];
        // Right-align: a shorter ellipsis left-pads with broadcast dims.
        // Every operand still spells all width batch labels, so positions
        // line up across operands; the leading (width - count) spellings
        // are virtual (size 1, stride 0, no storage).
        result.pads[i] = width - part.count;
        result.lefts[i] = part.left.size();
        result.inputs.push_back(part.left + fresh + part.right);
    }
    if (result.has_output) {
        std::string left;
        std::string right;
        bool has_ellipsis = false;
        if (!SplitEllipsis(text.substr(arrow + 2), left, right, has_ellipsis)) {
            result.error = "dots must form exactly one '...' per subscript";
            return result;
        }
        // The output ellipsis always spans the broadcast width.
        result.output = left + (has_ellipsis ? fresh : "") + right;
    } else {
        result.output = fresh;
    }
    result.ok = true;
    return result;
}

bool MapOperand(const std::string& sub, std::size_t left, std::size_t pad, std::size_t width,
                const std::vector<int64_t>& dims, std::vector<MappedDim>& mapped) {
    mapped.clear();
    std::size_t d = 0;
    for (std::size_t q = 0; q < left && q < sub.size(); ++q) {
        if (d >= dims.size()) {
            return false;
        }
        mapped.push_back(MappedDim{sub[q], dims[d++], false});
    }
    for (std::size_t p = 0; p < width; ++p) {
        const std::size_t q = left + p;
        if (q >= sub.size()) {
            return false;
        }
        if (p < pad) {
            mapped.push_back(MappedDim{sub[q], 1, true});
        } else {
            if (d >= dims.size()) {
                return false;
            }
            mapped.push_back(MappedDim{sub[q], dims[d++], false});
        }
    }
    for (std::size_t q = left + width; q < sub.size(); ++q) {
        if (d >= dims.size()) {
            return false;
        }
        mapped.push_back(MappedDim{sub[q], dims[d++], false});
    }
    return d == dims.size();
}

} // namespace einsum
} // namespace nert
