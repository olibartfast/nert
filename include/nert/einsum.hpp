#pragma once
// Shared Einsum "..." (ellipsis) expansion, used by the loader (syntax),
// shape inference (output dims), and the CPU kernel (execution) so the
// three agree on one expansion.
//
// An ellipsis covers zero or more batch dims. Each "..." expands to fresh
// labels (unused a-z/A-Z, deterministic order); with operand ranks known,
// the count per subscript is rank minus its explicit labels. Missing
// leading dims broadcast (size 1), exactly like NumPy: an output "..."
// always spans the broadcast of the inputs. Only the expanded equation
// (plain labels) reaches the existing per-site parsers.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nert {
namespace einsum {

// Syntax check for "..." groups in a spaceless equation. Returns an empty
// string when clean, else the reason: dots must form exactly one "..."
// per subscript, with nothing else dotted.
std::string EllipsisSyntaxError(const std::string& text);

// Expands every "..." using the operand ranks. Precondition: the syntax is
// clean and ranks holds one entry per equation operand. Fresh labels are
// returned alongside so sites can tell broadcastable labels apart from
// user labels (which keep the strict equal-size rule). A shorter ellipsis
// left-pads with broadcast dims: pads[i] counts operand i's virtual
// leading batch dims (size 1, stride 0, consuming no storage), and
// lefts[i]/width locate the batch block inside each expanded subscript.
struct Expansion {
    bool ok = false;
    std::string error;
    std::vector<std::string> inputs;
    std::string output;
    bool has_output = false;
    std::vector<char> fresh;
    std::vector<std::size_t> pads;
    std::vector<std::size_t> lefts;
    std::size_t width = 0;
};

Expansion ExpandEllipsis(const std::string& text, const std::vector<std::size_t>& ranks);

// One expanded subscript position mapped to storage: virtual pads read
// size 1 with stride 0 and consume no dimension.
struct MappedDim {
    char label = 0;
    int64_t dim = 1;
    bool virtual_pad = false;
};

// Maps an expanded subscript onto operand dims. left/pad/width come from
// the expansion (plain path: left is the full length, pad and width are
// zero). Returns false when the mapping does not consume every dim.
bool MapOperand(const std::string& sub, std::size_t left, std::size_t pad, std::size_t width,
                const std::vector<int64_t>& dims, std::vector<MappedDim>& mapped);

} // namespace einsum
} // namespace nert
