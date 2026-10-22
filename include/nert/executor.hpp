#pragma once
// The sequential graph executor.
//
// A Model binds one loaded graph to one concrete device and one concrete set of
// input shapes. Construction resolves shapes and plans the arena. Pointer-offset
// devices reserve the arena once; handle-based devices (OpenCL) allocate each
// live intermediate separately and release it when its last consumer has run,
// so peak memory follows the live set rather than the whole plan. Run then
// executes the nodes in order through the device's kernel table, binding graph
// inputs and initializers to their storage and intermediates to their live
// buffers.
//
// Must not include or reference the backend abstraction layer.

#include "nert/device.hpp"
#include "nert/graph.hpp"
#include "nert/plan.hpp"
#include "nert/shapes.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nert {

// One host-side tensor crossing the typed Run boundary: its element type, its
// shape, and its elements as contiguous native-endian bytes. Float32 takes 4
// bytes per element, Int64 8, and Bool 1 (0 or 1; any non-zero byte reads as
// true on input, and outputs are written as 0 or 1).
struct HostTensor {
    DataType dtype = DataType::Float32;
    std::vector<int64_t> dims;
    std::vector<uint8_t> bytes;
};

using HostTensorMap = std::map<std::string, HostTensor>;

// Bytes per element of a graph I/O dtype (Float32, Int64, Bool); 0 otherwise.
size_t HostElementSize(DataType dtype);

// The caller-visible result of one float Run: every declared graph output,
// flattened to a float32 vector and keyed by tensor name.
struct InferenceResult {
    std::map<std::string, std::vector<float>> outputs;
};

// One graph bound to a device and a concrete input shape. The arena is
// allocated exactly once, at construction, and released in the destructor.
class Model {
  public:
    Model(const Graph& graph, const ShapeMap& input_dims, const Device& device);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Graph& graph() const { return graph_; }
    const InferredShapes& shapes() const { return shapes_; }
    const MemoryPlan& plan() const { return plan_; }

    // Execute the graph for `inputs`, keyed by graph input name. Throws
    // InferenceException when an input is missing or mis-sized, when a node's op
    // has no kernel on the device, or when a node output has no arena buffer.
    // Float-only entry point: every graph input and output must be float32; a
    // graph with a non-float input or output throws InferenceException naming
    // the tensor.
    InferenceResult Run(const std::map<std::string, std::vector<float>>& inputs);

    // Typed entry point: each input's dtype and dims must match the graph
    // input's inferred dtype and shape, and its byte count must equal the
    // element count times HostElementSize(dtype). Returns every declared graph
    // output in its own dtype, keyed by name. Throws InferenceException on a
    // missing, mistyped, or mis-sized input, and as Run does otherwise.
    HostTensorMap RunTyped(const HostTensorMap& inputs);

  private:
    // Execute the nodes with the graph inputs already bound in `views`, then
    // bind initializers and node outputs; on return `views` resolves every
    // graph output (pointer-offset arena views stay valid until the next run).
    // On a non-host device the initializers are already resident (uploaded
    // once in the constructor), so only the graph inputs are staged to device
    // memory per run — one upload each, no per-op round-trips — and the graph
    // outputs are downloaded into `host_outputs` (caller-owned), so output
    // views stay host-readable. Handle-based intermediates are released as
    // soon as their last consumer has run.
    void Execute(std::map<std::string, TensorView>& views, std::map<std::string, std::vector<uint8_t>>& host_outputs);

    // The view bound to graph output `name` after Execute.
    static const TensorView& OutputView(const std::map<std::string, TensorView>& views, const std::string& name);

    // Uploads every non-empty initializer once (non-host devices only).
    // Throws InferenceException naming the initializer when its device
    // allocation fails; leaves already-uploaded residents for the caller
    // (the constructor) to release.
    void UploadResidents();

    // Releases every resident initializer buffer; safe to call when none
    // were uploaded.
    void ReleaseResidents();

    // Releases every per-offset arena buffer; safe to call when none were
    // allocated. Retained for the handle-device layout; the live-buffer path
    // no longer preallocates these slots.
    void ReleaseArenaSlots();

    const Graph& graph_; // borrowed; must outlive the model
    Device* device_;     // the device chosen once for this graph
    InferredShapes shapes_;
    MemoryPlan plan_;
    void* arena_ = nullptr; // one allocation on pointer-offset devices
    // Retained for Model layout compatibility. Empty on pointer-offset devices
    // and no longer populated by the live-buffer handle path.
    std::map<int64_t, void*> arena_slots_;
    // Device buffers holding the initializers, uploaded once in the
    // constructor on non-host devices; empty on host devices (which read
    // initializer storage in place) and for empty initializers.
    std::map<std::string, void*> resident_;
};

} // namespace nert
