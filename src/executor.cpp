// The sequential executor (T-16).
//
// Construction runs shape inference and memory planning for the supplied input
// shapes. Pointer-offset devices reserve the whole arena in one allocation;
// handle-based devices allocate each live intermediate separately at its
// defining node and release it after its last consumer, so their peak follows
// the live set instead of the total planned construction demand. Run resolves
// a TensorView for every graph input, initializer, and node output, walks the
// nodes in order, and dispatches each to the device kernel table. The planner's
// inclusive liveness guarantees that no node output shares bytes with any input
// of its defining node, so Run binds each tensor's view directly — no transient
// copies. Graph outputs are copied out before handle-backed buffers are
// released, so a second Run observes the same plan.

#include "nert/executor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace nert {
namespace {

// Product of a shape's dimensions. An empty shape is a scalar: one element.
int64_t ElementCount(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (const int64_t dim : dims) {
        count *= dim;
    }
    return count;
}

// A dtype's name for error messages.
const char* DataTypeName(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "float32";
    case DataType::Int64:
        return "int64";
    case DataType::Bool:
        return "bool";
    case DataType::Unknown:
        break;
    }
    return "unknown";
}

// The inferred shape for `name`, or a failure naming the tensor.
const TensorInfo& ShapeOf(const InferredShapes& shapes, const std::string& name) {
    const auto it = shapes.tensors.find(name);
    if (it == shapes.tensors.end()) {
        throw InferenceException("executor: tensor '" + name + "' has no inferred shape");
    }
    return it->second;
}

// Per-run device buffers that release themselves on scope exit, so a
// throwing kernel cannot leak staged inputs.
class StagedBuffers {
  public:
    explicit StagedBuffers(Device* device) : device_(device) {}

    StagedBuffers(const StagedBuffers&) = delete;
    StagedBuffers& operator=(const StagedBuffers&) = delete;

    ~StagedBuffers() {
        for (void* buffer : buffers_) {
            device_->allocator().release(buffer);
        }
    }

    void push_back(void* buffer) { buffers_.push_back(buffer); }

  private:
    Device* device_;
    std::vector<void*> buffers_;
};

constexpr int64_t kHandleFreeCacheBytes = 256LL * 1024 * 1024;

// Per-run buffers for handle-based devices (OpenCL). They are allocated when a
// node produces a tensor and released once its inclusive last-use node has run;
// the destructor releases any survivors on a kernel error. Freed buffers are
// reused up to a small cache cap so per-node driver allocation stays bounded.
class TensorBuffers {
  public:
    explicit TensorBuffers(Device* device) : device_(device) {}

    TensorBuffers(const TensorBuffers&) = delete;
    TensorBuffers& operator=(const TensorBuffers&) = delete;

    ~TensorBuffers() { ReleaseAll(); }

    // Reuse the smallest freed buffer that fits, or allocate a new one. The
    // returned capacity is at least `bytes`; kernels still see the requested
    // shape.
    void* acquire(int64_t bytes, int64_t& capacity) {
        const auto fit = free_.lower_bound(bytes);
        if (fit != free_.end()) {
            capacity = fit->first;
            void* buffer = fit->second;
            free_.erase(fit);
            free_bytes_ -= capacity;
            return buffer;
        }
        capacity = bytes;
        return device_->allocator().allocate(bytes);
    }

    void add(const std::string& name, void* buffer, int64_t capacity) {
        auto inserted = buffers_.emplace(name, Buffer{buffer, capacity});
        if (!inserted.second) {
            cache_or_release(buffer, capacity);
            throw InferenceException("executor: duplicate live buffer for tensor '" + name + "'");
        }
    }

    void release_before(int64_t node_index, const std::map<std::string, int64_t>& last_use,
                        const std::set<std::string>& keep) {
        for (auto it = buffers_.begin(); it != buffers_.end();) {
            const auto use = last_use.find(it->first);
            if (use != last_use.end() && use->second < node_index && keep.count(it->first) == 0) {
                cache_or_release(it->second.data, it->second.size);
                it = buffers_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void ReleaseAll() {
        for (const auto& entry : buffers_) {
            device_->allocator().release(entry.second.data);
        }
        buffers_.clear();
        for (const auto& entry : free_) {
            device_->allocator().release(entry.second);
        }
        free_.clear();
        free_bytes_ = 0;
    }

  private:
    struct Buffer {
        void* data;
        int64_t size;
    };

    void cache_or_release(void* buffer, int64_t bytes) {
        if (free_bytes_ + bytes <= kHandleFreeCacheBytes) {
            free_.emplace(bytes, buffer);
            free_bytes_ += bytes;
        } else {
            device_->allocator().release(buffer);
        }
    }

    Device* device_;
    std::map<std::string, Buffer> buffers_;
    std::multimap<int64_t, void*> free_; // capacity -> reusable buffer
    int64_t free_bytes_ = 0;
};

// Inclusive last-use node index for every planned arena tensor, plus the graph
// outputs that must survive until download.
void ComputeLifetimes(const Graph& graph, const MemoryPlan& plan, std::map<std::string, int64_t>& last_use,
                      std::set<std::string>& graph_outputs) {
    std::map<std::string, int64_t> definitions;
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        for (const std::string& output : graph.nodes[i].outputs) {
            if (plan.buffers.count(output) != 0) {
                definitions.emplace(output, static_cast<int64_t>(i));
            }
        }
    }
    for (const auto& entry : plan.buffers) {
        last_use[entry.first] = definitions.at(entry.first);
    }
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        for (const std::string& input : graph.nodes[i].inputs) {
            const auto use = last_use.find(input);
            if (use != last_use.end() && static_cast<int64_t>(i) > use->second) {
                use->second = static_cast<int64_t>(i);
            }
        }
    }
    for (const auto& output : graph.outputs) {
        if (plan.buffers.count(output.first) != 0) {
            graph_outputs.insert(output.first);
            last_use[output.first] = static_cast<int64_t>(graph.nodes.size()) - 1;
        }
    }
}

} // namespace

Model::Model(const Graph& graph, const ShapeMap& input_dims, const Device& device)
    : graph_(graph), device_(const_cast<Device*>(&device)), shapes_(InferShapes(graph, input_dims)),
      plan_(PlanMemory(graph, shapes_)) {
    try {
        if (plan_.arena_size > 0 && device_->supports_arena_offsets()) {
            arena_ = device_->allocator().allocate(plan_.arena_size);
            if (arena_ == nullptr) {
                throw InferenceException("executor: could not allocate the " + std::to_string(plan_.arena_size) +
                                         "-byte arena");
            }
        }
        UploadResidents();
    } catch (...) {
        // The destructor never runs for a failed construction, so release
        // the partial state here before the exception escapes.
        ReleaseResidents();
        ReleaseArenaSlots();
        if (arena_ != nullptr) {
            device_->allocator().release(arena_);
            arena_ = nullptr;
        }
        throw;
    }
}

Model::~Model() {
    ReleaseResidents();
    ReleaseArenaSlots();
    if (arena_ != nullptr) {
        device_->allocator().release(arena_);
    }
}

void Model::ReleaseArenaSlots() {
    for (const auto& entry : arena_slots_) {
        device_->allocator().release(entry.second);
    }
    arena_slots_.clear();
}

size_t HostElementSize(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return sizeof(float);
    case DataType::Int64:
        return sizeof(int64_t);
    case DataType::Bool:
        return sizeof(bool);
    case DataType::Unknown:
        break;
    }
    return 0;
}

InferenceResult Model::Run(const std::map<std::string, std::vector<float>>& inputs) {
    // Graph inputs point at the caller's vectors; Execute binds the rest.
    std::map<std::string, TensorView> views;

    for (const auto& entry : graph_.inputs) {
        const std::string& name = entry.first;
        const TensorInfo& info = ShapeOf(shapes_, name);
        if (info.dtype != DataType::Float32) {
            throw InferenceException("executor: graph input '" + name + "' is " + DataTypeName(info.dtype) +
                                     ", not float32; use RunTyped");
        }
        const auto provided = inputs.find(name);
        if (provided == inputs.end()) {
            throw InferenceException("executor: missing input '" + name + "'");
        }
        const int64_t expected = ElementCount(info.dims);
        if (static_cast<int64_t>(provided->second.size()) != expected) {
            throw InferenceException("executor: input '" + name + "' has " + std::to_string(provided->second.size()) +
                                     " values, expected " + std::to_string(expected));
        }
        TensorView view;
        view.data = const_cast<float*>(provided->second.data());
        view.dtype = info.dtype;
        view.dims = info.dims;
        views.emplace(name, std::move(view));
    }

    std::map<std::string, std::vector<uint8_t>> host_outputs;
    Execute(views, host_outputs);

    InferenceResult result;
    for (const auto& entry : graph_.outputs) {
        const std::string& name = entry.first;
        const TensorView& view = OutputView(views, name);
        if (view.dtype != DataType::Float32) {
            throw InferenceException("executor: graph output '" + name + "' is not float32");
        }
        const int64_t count = ElementCount(view.dims);
        const float* src = static_cast<const float*>(view.data);
        result.outputs.emplace(name, std::vector<float>(src, src + count));
    }
    return result;
}

HostTensorMap Model::RunTyped(const HostTensorMap& inputs) {
    std::map<std::string, TensorView> views;
    // Bool inputs are normalised to 0/1 bytes so kernels may read them as bool.
    std::map<std::string, std::vector<uint8_t>> bool_inputs;
    // Download targets for the graph outputs on non-host devices.
    std::map<std::string, std::vector<uint8_t>> host_outputs;

    for (const auto& entry : graph_.inputs) {
        const std::string& name = entry.first;
        const TensorInfo& info = ShapeOf(shapes_, name);
        const auto provided = inputs.find(name);
        if (provided == inputs.end()) {
            throw InferenceException("executor: missing input '" + name + "'");
        }
        const HostTensor& host = provided->second;
        if (host.dtype != info.dtype) {
            throw InferenceException("executor: input '" + name + "' is " + DataTypeName(host.dtype) + ", expected " +
                                     DataTypeName(info.dtype));
        }
        if (host.dims != info.dims) {
            throw InferenceException("executor: input '" + name + "' has a shape other than the one the model was " +
                                     "built for");
        }
        const size_t expected = static_cast<size_t>(ElementCount(info.dims)) * HostElementSize(info.dtype);
        if (host.bytes.size() != expected) {
            throw InferenceException("executor: input '" + name + "' has " + std::to_string(host.bytes.size()) +
                                     " bytes, expected " + std::to_string(expected));
        }
        TensorView view;
        view.dtype = info.dtype;
        view.dims = info.dims;
        if (info.dtype == DataType::Bool) {
            std::vector<uint8_t>& normalised = bool_inputs[name];
            normalised.resize(host.bytes.size());
            for (size_t i = 0; i < host.bytes.size(); ++i) {
                normalised[i] = host.bytes[i] != 0 ? 1U : 0U;
            }
            view.data = normalised.data();
        } else {
            view.data = const_cast<uint8_t*>(host.bytes.data());
        }
        views.emplace(name, std::move(view));
    }

    Execute(views, host_outputs);

    HostTensorMap result;
    for (const auto& entry : graph_.outputs) {
        const std::string& name = entry.first;
        const TensorView& view = OutputView(views, name);
        const size_t element_size = HostElementSize(view.dtype);
        if (element_size == 0) {
            throw InferenceException("executor: graph output '" + name + "' has no host dtype");
        }
        HostTensor host;
        host.dtype = view.dtype;
        host.dims = view.dims;
        host.bytes.resize(static_cast<size_t>(ElementCount(view.dims)) * element_size);
        if (view.dtype == DataType::Bool) {
            const bool* src = static_cast<const bool*>(view.data);
            for (size_t i = 0; i < host.bytes.size(); ++i) {
                host.bytes[i] = src[i] ? 1U : 0U;
            }
        } else if (!host.bytes.empty()) {
            std::memcpy(host.bytes.data(), view.data, host.bytes.size());
        }
        result.emplace(name, std::move(host));
    }
    return result;
}

const TensorView& Model::OutputView(const std::map<std::string, TensorView>& views, const std::string& name) {
    const auto it = views.find(name);
    if (it == views.end()) {
        throw InferenceException("executor: graph output '" + name + "' was never produced");
    }
    return it->second;
}

// Byte size of a host-bound view for device staging, or 0 when the view has
// no host dtype, no data, or no elements. Nulls and empties keep their exact
// CPU semantics by skipping staging (kernels see the same nulls as on CPU).
int64_t StagedBytes(const TensorView& view) {
    const size_t elem = HostElementSize(view.dtype);
    if (view.data == nullptr || elem == 0) {
        return 0;
    }
    return ElementCount(view.dims) * static_cast<int64_t>(elem);
}

void Model::UploadResidents() {
    if (device_->is_host()) {
        return;
    }
    for (const auto& entry : graph_.initializers) {
        const Initializer& init = entry.second;
        TensorView view;
        view.dtype = init.dtype;
        view.dims = init.dims;
        view.data = InitializerData(const_cast<Initializer&>(init));
        // Empty initializers stay host-bound (kernels see the same nulls as
        // on CPU); Execute still validates the non-empty-but-null case.
        if (StagedBytes(view) == 0) {
            continue;
        }
        const int64_t bytes = StagedBytes(view);
        void* device_buffer = device_->allocator().allocate(bytes);
        if (device_buffer == nullptr) {
            throw InferenceException("executor: could not stage initializer '" + entry.first + "' on the device");
        }
        device_->transfer().host_to_device(device_buffer, view.data, bytes);
        resident_.emplace(entry.first, device_buffer);
    }
}

void Model::ReleaseResidents() {
    for (const auto& entry : resident_) {
        device_->allocator().release(entry.second);
    }
    resident_.clear();
}

void Model::Execute(std::map<std::string, TensorView>& views,
                    std::map<std::string, std::vector<uint8_t>>& host_outputs) {
    for (const auto& entry : graph_.initializers) {
        const Initializer& init = entry.second;
        TensorView view;
        view.dtype = init.dtype;
        view.dims = init.dims;
        view.data = InitializerData(const_cast<Initializer&>(init));
        // An empty initializer (an omitted Resize roi, say) has no storage.
        if (view.data == nullptr && ElementCount(init.dims) != 0) {
            throw InferenceException("executor: initializer '" + entry.first + "' has no values of its dtype");
        }
        // On non-host devices the bytes already live on the device from
        // construction; empty initializers stay host-bound either way.
        const auto resident = resident_.find(entry.first);
        if (resident != resident_.end()) {
            view.data = resident->second;
        }
        views.emplace(entry.first, std::move(view));
    }

    StagedBuffers staged(device_);
    TensorBuffers tensor_buffers(device_);
    const bool pointer_arena = device_->supports_arena_offsets();
    std::map<std::string, int64_t> last_use;
    std::set<std::string> graph_outputs;
    if (!pointer_arena) {
        ComputeLifetimes(graph_, plan_, last_use, graph_outputs);
    }
    if (!device_->is_host()) {
        // Per-run staging covers the graph inputs only: initializers are
        // resident from construction, so a second Run uploads no weights.
        for (auto& staged_entry : views) {
            bool is_input = false;
            for (const auto& input : graph_.inputs) {
                if (input.first == staged_entry.first) {
                    is_input = true;
                    break;
                }
            }
            if (!is_input) {
                continue;
            }
            TensorView& view = staged_entry.second;
            const int64_t bytes = StagedBytes(view);
            if (bytes == 0) {
                continue;
            }
            void* device_buffer = device_->allocator().allocate(bytes);
            if (device_buffer == nullptr) {
                throw InferenceException("executor: could not stage '" + staged_entry.first + "' on the device");
            }
            device_->transfer().host_to_device(device_buffer, view.data, bytes);
            view.data = device_buffer;
            staged.push_back(device_buffer);
        }
    }

    for (std::size_t node_index = 0; node_index < graph_.nodes.size(); ++node_index) {
        const Node& node = graph_.nodes[node_index];
        if (!pointer_arena) {
            tensor_buffers.release_before(static_cast<int64_t>(node_index), last_use, graph_outputs);
        }
        const KernelFn kernel = device_->kernels().find(node.op_type);
        if (kernel == nullptr) {
            throw InferenceException("executor: no kernel for op '" + node.op_type + "' (node '" + node.name + "')");
        }

        std::vector<TensorView> node_inputs;
        node_inputs.reserve(node.inputs.size());
        for (const std::string& name : node.inputs) {
            if (name.empty()) {
                node_inputs.push_back(TensorView{}); // an omitted optional operand
                continue;
            }
            const auto it = views.find(name);
            if (it == views.end()) {
                throw InferenceException("executor: node '" + node.name + "' input '" + name + "' is not available");
            }
            node_inputs.push_back(it->second);
        }

        std::vector<TensorView> node_outputs;
        node_outputs.reserve(node.outputs.size());
        for (const std::string& name : node.outputs) {
            const auto buffer = plan_.buffers.find(name);
            if (buffer == plan_.buffers.end()) {
                throw InferenceException("executor: no arena buffer for output '" + name + "' of node '" + node.name +
                                         "'");
            }
            const TensorInfo& info = ShapeOf(shapes_, name);
            TensorView view;
            if (pointer_arena) {
                view.data = static_cast<char*>(arena_) + buffer->second.offset;
            } else if (buffer->second.size == 0) {
                view.data = nullptr;
            } else {
                int64_t capacity = 0;
                void* data = tensor_buffers.acquire(buffer->second.size, capacity);
                if (data == nullptr) {
                    throw InferenceException("executor: could not allocate the " + std::to_string(buffer->second.size) +
                                             "-byte buffer for output '" + name + "' of node '" + node.name + "'");
                }
                tensor_buffers.add(name, data, capacity);
                view.data = data;
            }
            view.dtype = info.dtype;
            view.dims = info.dims;
            views[name] = view;
            node_outputs.push_back(std::move(view));
        }

        kernel(node, node_inputs, node_outputs);
    }

    if (!device_->is_host()) {
        // Graph outputs are downloaded once into caller-owned host buffers so
        // the result extraction above keeps reading host memory; outputs
        // aliasing resident initializers download from those copies.
        // Staged input buffers release themselves when `staged` unwinds,
        // including when a kernel throws above.
        for (const auto& entry : graph_.outputs) {
            const auto produced = views.find(entry.first);
            if (produced == views.end()) {
                throw InferenceException("executor: graph output '" + entry.first + "' was never produced");
            }
            TensorView& view = produced->second;
            const int64_t bytes = StagedBytes(view);
            if (bytes == 0) {
                continue;
            }
            std::vector<uint8_t>& host = host_outputs[entry.first];
            host.resize(static_cast<std::size_t>(bytes));
            device_->transfer().device_to_host(host.data(), view.data, bytes);
            view.data = host.data();
        }
    }
    tensor_buffers.ReleaseAll();
}

} // namespace nert
