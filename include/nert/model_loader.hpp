#pragma once
// Reads an ONNX model file into the engine's graph.
//
// The single entry point for model loading. The protobuf wire decoding is an
// implementation detail behind this interface, and parsing is hand-written, so
// the engine pulls in no protobuf or ONNX library.
//
// Must not include anything from the backend abstraction layer.

#include <string>

#include "nert/graph.hpp"

namespace nert {

// Parse the ONNX model at `path` into the Graph IR.
//
// Accepts declared ONNX-domain opsets 11 and above. Every source node records
// its effective ONNX schema on `Node::schema_version` (Constant nodes are
// folded into initializers before a Node exists); during the version-aware
// migration, opsets 11..18 are still rewritten to the current canonical form,
// while above 18 each node must resolve to a schema version verified against
// that implementation. A node with no schema at or below the declaration is
// rejected as ONNX-invalid.
//
// Throws ModelLoadException when the file cannot be read, the wire format is
// malformed, or the model is structurally unusable. Semantic rejections
// (unsupported op, unsupported attribute, unresolvable dynamic shape) name the
// offending node and op type in the message.
Graph LoadGraphFromFile(const std::string& path);

} // namespace nert
