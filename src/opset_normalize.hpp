#pragma once
// Load-time canonicalization of older ONNX node forms into the opset-18 IR.
//
// The runtime keeps one implementation per operator and one canonical graph
// shape. This pass is the only place that understands pre-18 syntax; it runs
// after node decoding and before shape inference, so every consumer sees the
// opset-18 form.
//
// Must not include anything from the backend abstraction layer.

#include "nert/graph.hpp"

#include <cstdint>

namespace nert {

// Rewrites the node forms whose syntax changed between `declared_opset` and
// opset 18. Rejects forms that are illegal for the declared version or that
// would need semantics the kernels do not implement. `declared_opset` must be
// in the range the loader accepts (13..18 in the current phase).
void NormalizeGraphForOpset(Graph& graph, int64_t declared_opset);

} // namespace nert
