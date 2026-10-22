// Sequential-executor tests.
//
// Two suites, split for ctest filters:
//   EngineExecutor.*          — hermetic Add+Relu with hand-computed output, a
//                               counting device proving one arena allocation
//                               across construction plus two runs, and the
//                               ResNet-18 fixture end to end.
//   EngineExecutorNegative.*  — missing input, wrong input size, and an op with
//                               no kernel, each throwing InferenceException.
//
// The hermetic graph reuses the hand-rolled protobuf wire encoders from the
// loader tests: the field numbers mirror ModelLoader.cpp and no public header
// carries any. The fixture is resolved via NERT_FIXTURE; a missing
// fixture FAILS, never skips.

#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"
#include "nert/model_loader.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Hand-rolled protobuf wire encoders (test-side twin of the reader).
//
// Field numbers: ModelProto graph=7 opset_import=8; GraphProto node=1
// initializer=5 input=11 output=12; NodeProto input=1 output=2 name=3
// op_type=4; AttributeProto name=1; TensorProto dims=1 data_type=2 name=8
// raw_data=9; OpsetId version=2; ValueInfoProto name=1 type=2; TypeProto
// tensor_type=1; TensorType elem_type=1 shape=2; TensorShapeProto dim=1;
// Dimension dim_value=1.
// ---------------------------------------------------------------------------

void put_varint(std::string& out, uint64_t value) {
    while (value >= 0x80U) {
        out.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void put_tag(std::string& out, int field_number, uint8_t wire_type) {
    put_varint(out, (static_cast<uint64_t>(field_number) << 3) | wire_type);
}

void put_length_delimited(std::string& out, int field_number, const std::string& payload) {
    put_tag(out, field_number, 2);
    put_varint(out, payload.size());
    out.append(payload);
}

void put_sub(std::string& out, int field_number, const std::string& sub) {
    put_length_delimited(out, field_number, sub);
}

void put_varint_field(std::string& out, int field_number, uint64_t value) {
    put_tag(out, field_number, 0);
    put_varint(out, value);
}

std::string encode_opset18() {
    std::string leaving;
    put_varint_field(leaving, 2, 18);
    return leaving;
}

std::string encode_value_info(const std::string& name, int64_t elem_type, const std::vector<int64_t>& dims) {
    std::string vi;
    put_length_delimited(vi, 1, name);
    std::string tensor_type;
    put_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
    std::string shape;
    for (const int64_t dim : dims) {
        std::string dim_msg;
        put_varint_field(dim_msg, 1, static_cast<uint64_t>(dim));
        put_sub(shape, 1, dim_msg);
    }
    put_sub(tensor_type, 2, shape);
    std::string type_proto;
    put_sub(type_proto, 1, tensor_type);
    put_sub(vi, 2, type_proto);
    return vi;
}

std::string encode_node(const std::string& name, const std::string& op_type, const std::vector<std::string>& inputs,
                        const std::vector<std::string>& outputs) {
    std::string n;
    for (const std::string& input : inputs) {
        put_length_delimited(n, 1, input);
    }
    for (const std::string& output : outputs) {
        put_length_delimited(n, 2, output);
    }
    put_length_delimited(n, 3, name);
    put_length_delimited(n, 4, op_type);
    return n;
}

std::string encode_initializer(const std::string& name, const std::vector<int64_t>& dims,
                               const std::vector<float>& values) {
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 1); // FLOAT
    put_length_delimited(t, 8, name);
    const std::string raw(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    put_length_delimited(t, 9, raw);
    return t;
}

std::string encode_int64_initializer(const std::string& name, const std::vector<int64_t>& dims,
                                     const std::vector<int64_t>& values) {
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 7); // INT64
    put_length_delimited(t, 8, name);
    const std::string raw(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(int64_t));
    put_length_delimited(t, 9, raw);
    return t;
}

std::string encode_model(const std::string& graph_body) {
    std::string model;
    put_sub(model, 8, encode_opset18());
    put_sub(model, 7, graph_body);
    return model;
}

std::string write_model_bytes(const std::string& name, const std::string& bytes) {
    const std::string dir = (std::filesystem::temp_directory_path() / "engine_executor_tests").string();
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/" + name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) {
        ADD_FAILURE() << "cannot write test model file: " << path;
    }
    return path;
}

// A hermetic Add(x, c) -> y followed by Relu(y) -> z, with the float constant
// `c` supplied as an initializer. The graph input x has one element per entry
// in `constant`; the graph output is z.
nert::Graph load_hermetic_add_relu(const std::vector<float>& constant) {
    const int64_t width = static_cast<int64_t>(constant.size());
    std::string body;
    put_sub(body, 5, encode_initializer("c", {width}, constant));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "c"}, {"y"}));
    put_sub(body, 1, encode_node("relu0", "Relu", {"y"}, {"z"}));
    put_sub(body, 11, encode_value_info("x", 1, {width}));
    put_sub(body, 12, encode_value_info("z", 1, {width}));
    return nert::LoadGraphFromFile(write_model_bytes("add_relu.onnx", encode_model(body)));
}

// Add(x, c) -> t0 followed by `relu_links` Relu links, output tN. Under
// inclusive liveness the links touch at their boundaries, so only alternating
// tensors can reuse; a chain of at least three links makes the 64-byte-aligned
// arena strictly smaller than the sum of the intermediate buffers.
nert::Graph load_hermetic_chain(const std::vector<float>& constant, int relu_links) {
    const int64_t width = static_cast<int64_t>(constant.size());
    std::string body;
    put_sub(body, 5, encode_initializer("c", {width}, constant));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "c"}, {"t0"}));
    std::string previous = "t0";
    for (int i = 1; i <= relu_links; ++i) {
        const std::string next = "t" + std::to_string(i);
        put_sub(body, 1, encode_node("relu" + std::to_string(i), "Relu", {previous}, {next}));
        previous = next;
    }
    put_sub(body, 11, encode_value_info("x", 1, {width}));
    put_sub(body, 12, encode_value_info(previous, 1, {width}));
    return nert::LoadGraphFromFile(write_model_bytes("chain.onnx", encode_model(body)));
}

// A hermetic typed-I/O graph: int64 graph input `ids` [4] and int64 constant
// `c`; Add(ids, c) -> `sum` (int64 graph output) and Greater(ids, c) -> `flag`
// (bool graph output).
nert::Graph load_hermetic_int64_bool(const std::vector<int64_t>& constant) {
    const int64_t width = static_cast<int64_t>(constant.size());
    std::string body;
    put_sub(body, 5, encode_int64_initializer("c", {width}, constant));
    put_sub(body, 1, encode_node("add0", "Add", {"ids", "c"}, {"sum"}));
    put_sub(body, 1, encode_node("greater0", "Greater", {"ids", "c"}, {"flag"}));
    put_sub(body, 11, encode_value_info("ids", 7 /*INT64*/, {width}));
    put_sub(body, 12, encode_value_info("sum", 7 /*INT64*/, {width}));
    put_sub(body, 12, encode_value_info("flag", 9 /*BOOL*/, {width}));
    return nert::LoadGraphFromFile(write_model_bytes("int64_bool.onnx", encode_model(body)));
}

nert::HostTensor int64_host_tensor(const std::vector<int64_t>& values) {
    nert::HostTensor host;
    host.dtype = nert::DataType::Int64;
    host.dims = {static_cast<int64_t>(values.size())};
    host.bytes.resize(values.size() * sizeof(int64_t));
    std::memcpy(host.bytes.data(), values.data(), host.bytes.size());
    return host;
}

const char* fixture_env_name() { return "NERT_FIXTURE"; }

std::string fixture_path() {
    const char* from_env = std::getenv(fixture_env_name());
    return from_env != nullptr ? std::string(from_env) : std::string();
}

// The process-lifetime CPU device as a mutable reference, so the non-const
// allocator and transfer accessors are usable.
nert::Device& cpu_device() { return const_cast<nert::Device&>(nert::CpuDevice()); }

// Counts allocator calls while delegating to the CPU allocator.
class CountingAllocator final : public nert::Allocator {
  public:
    int allocate_calls = 0;
    int release_calls = 0;

    void* allocate(int64_t bytes) override {
        ++allocate_calls;
        return cpu_device().allocator().allocate(bytes);
    }

    void release(void* buffer) override {
        ++release_calls;
        cpu_device().allocator().release(buffer);
    }
};

// A kernel table with no entries, used to prove the executor reports a missing
// kernel rather than silently doing nothing.
class EmptyKernelTable final : public nert::KernelTable {
  public:
    nert::KernelFn find(const std::string&) const override { return nullptr; }
};

// A test device whose allocator counts calls. When `empty_kernels` is true it
// advertises no kernels; otherwise it forwards to the CPU device's table.
class TestDevice final : public nert::Device {
  public:
    explicit TestDevice(bool empty_kernels) : kernels_(empty_kernels ? nullptr : &nert::CpuDevice().kernels()) {}

    const char* name() const override { return "test-cpu"; }
    nert::Allocator& allocator() override { return allocator_; }
    nert::Transfer& transfer() override { return cpu_device().transfer(); }
    const nert::KernelTable& kernels() const override { return kernels_ != nullptr ? *kernels_ : empty_table_; }

    const CountingAllocator& counting() const { return allocator_; }

  private:
    CountingAllocator allocator_;
    const nert::KernelTable* kernels_;
    EmptyKernelTable empty_table_;
};

// Counts transfer traffic while delegating to the CPU transfer (memcpy
// semantics, so numerics stay identical).
class CountingTransfer final : public nert::Transfer {
  public:
    int h2d_calls = 0;
    int d2h_calls = 0;
    int64_t h2d_bytes = 0;
    int64_t d2h_bytes = 0;

    void host_to_device(void* dst, const void* src, int64_t bytes) override {
        ++h2d_calls;
        h2d_bytes += bytes;
        cpu_device().transfer().host_to_device(dst, src, bytes);
    }

    void device_to_host(void* dst, const void* src, int64_t bytes) override {
        ++d2h_calls;
        d2h_bytes += bytes;
        cpu_device().transfer().device_to_host(dst, src, bytes);
    }
};

// A non-host test device: malloc-backed memory, CPU kernels, counted
// traffic. Exercises the GPU executor path (resident initializers,
// per-run input staging) with no hardware.
class NonHostTestDevice final : public nert::Device {
  public:
    explicit NonHostTestDevice(const nert::KernelTable* table, bool arena_offsets = true)
        : table_(table), arena_offsets_(arena_offsets) {}

    const char* name() const override { return "test-device"; }
    nert::Allocator& allocator() override { return allocator_; }
    nert::Transfer& transfer() override { return transfer_; }
    const nert::KernelTable& kernels() const override { return *table_; }
    bool is_host() const override { return false; }
    bool supports_arena_offsets() const override { return arena_offsets_; }

    const CountingAllocator& counting() const { return allocator_; }
    const CountingTransfer& traffic() const { return transfer_; }

  private:
    CountingAllocator allocator_;
    CountingTransfer transfer_;
    const nert::KernelTable* table_;
    bool arena_offsets_;
};

void ThrowingAddKernel(const nert::Node&, const std::vector<nert::TensorView>&, const std::vector<nert::TensorView>&) {
    throw nert::InferenceException("test-device: boom");
}

// A kernel table that throws on Add and forwards everything else to the CPU
// table, used to prove staged inputs release on kernel failure.
class ThrowOnAddTable final : public nert::KernelTable {
  public:
    nert::KernelFn find(const std::string& op_type) const override {
        if (op_type == "Add") {
            return &ThrowingAddKernel;
        }
        return nert::CpuDevice().kernels().find(op_type);
    }
};

} // namespace

// ===========================================================================
// EngineExecutor.* — positive cases
// ===========================================================================
TEST(NertExecutor, HermeticAddReluMatchesHandComputed) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    nert::Model model(graph, {{"x", {2}}}, nert::CpuDevice());

    const nert::InferenceResult result = model.Run({{"x", {1.0F, -2.0F}}});
    ASSERT_EQ(result.outputs.count("z"), 1U);
    EXPECT_EQ(result.outputs.at("z"), (std::vector<float>{0.5F, 0.0F}));
}

TEST(NertExecutor, ArenaAllocatedOnceAndReused) {
    // A three-link chain under inclusive liveness: only alternating tensors can
    // share, so the 64-byte-aligned arena stays strictly smaller than the sum of
    // the intermediate buffers and the reuse proof is not swamped by alignment.
    const std::vector<float> constant(16, 0.0F);
    const nert::Graph graph = load_hermetic_chain(constant, 2);
    TestDevice device(false);
    {
        nert::Model model(graph, {{"x", {16}}}, device);
        const std::vector<float> input(16, 1.0F);
        const nert::InferenceResult first = model.Run({{"x", input}});
        const nert::InferenceResult second = model.Run({{"x", input}});
        EXPECT_EQ(first.outputs.at("t2"), second.outputs.at("t2"));
        EXPECT_EQ(device.counting().allocate_calls, 1);
        EXPECT_EQ(device.counting().release_calls, 0);

        int64_t intermediate_total = 0;
        for (const auto& buffer : model.plan().buffers) {
            intermediate_total += buffer.second.size;
        }
        EXPECT_LT(model.plan().arena_size, intermediate_total);
    }
    EXPECT_EQ(device.counting().allocate_calls, 1);
    EXPECT_EQ(device.counting().release_calls, 1);
}

// Initializers upload once at construction; each Run stages only the graph
// inputs.Hermetic Add(x, c) + Relu with a 2-float constant: 8 weight bytes
// once, 8 input bytes per run.
TEST(NertExecutor, ResidentInitializersUploadOnceAcrossRuns) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    NonHostTestDevice device(&nert::CpuDevice().kernels());
    {
        nert::Model model(graph, {{"x", {2}}}, device);
        // Construction: arena + one resident upload of `c`, nothing else.
        EXPECT_EQ(device.counting().allocate_calls, 2);
        EXPECT_EQ(device.traffic().h2d_calls, 1);
        EXPECT_EQ(device.traffic().h2d_bytes, 8);

        const nert::InferenceResult first = model.Run({{"x", {1.0F, -2.0F}}});
        EXPECT_EQ(first.outputs.at("z"), (std::vector<float>{0.5F, 0.0F}));
        // First run adds exactly one input staging (allocate + upload) and
        // releases it at the end of Execute.
        EXPECT_EQ(device.counting().allocate_calls, 3);
        EXPECT_EQ(device.counting().release_calls, 1);
        EXPECT_EQ(device.traffic().h2d_calls, 2);
        EXPECT_EQ(device.traffic().h2d_bytes, 16);

        const nert::InferenceResult second = model.Run({{"x", {1.0F, -2.0F}}});
        EXPECT_EQ(second.outputs.at("z"), (std::vector<float>{0.5F, 0.0F}));
        // Second run stages the input again but never re-uploads `c`.
        EXPECT_EQ(device.counting().allocate_calls, 4);
        EXPECT_EQ(device.counting().release_calls, 2);
        EXPECT_EQ(device.traffic().h2d_calls, 3);
        EXPECT_EQ(device.traffic().h2d_bytes, 24);
    }
    // Destruction releases the arena and the resident initializer: every
    // allocation is balanced.
    EXPECT_EQ(device.counting().allocate_calls, 4);
    EXPECT_EQ(device.counting().release_calls, 4);
}

// Handle-based devices allocate only live intermediates at run time. The
// three-node chain has one resident initializer and no construction arena; each
// run allocates the staged input plus two overlapping live outputs, then reuses
// the released buffer for the last output.
TEST(NertExecutor, HandleDeviceAllocatesOnlyLiveTensors) {
    const std::vector<float> constant(16, 0.0F);
    const nert::Graph graph = load_hermetic_chain(constant, 2);
    NonHostTestDevice device(&nert::CpuDevice().kernels(), false);
    {
        nert::Model model(graph, {{"x", {16}}}, device);
        EXPECT_EQ(device.counting().allocate_calls, 1); // resident initializer only
        EXPECT_EQ(device.counting().release_calls, 0);

        const std::vector<float> input(16, 1.0F);
        const nert::InferenceResult first = model.Run({{"x", input}});
        EXPECT_EQ(first.outputs.at("t2"), input);
        EXPECT_EQ(device.counting().allocate_calls, 4);
        EXPECT_EQ(device.counting().release_calls, 3);

        const nert::InferenceResult second = model.Run({{"x", input}});
        EXPECT_EQ(second.outputs.at("t2"), input);
        EXPECT_EQ(device.counting().allocate_calls, 7);
        EXPECT_EQ(device.counting().release_calls, 6);
    }
    EXPECT_EQ(device.counting().allocate_calls, 7);
    EXPECT_EQ(device.counting().release_calls, 7); // resident released
}

// A throwing kernel on a handle device releases both the staged input and
// the node output allocated before the launch.
TEST(NertExecutor, HandleDeviceReleasesBuffersOnKernelThrow) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    ThrowOnAddTable throwing_table;
    NonHostTestDevice device(&throwing_table, false);
    {
        nert::Model model(graph, {{"x", {2}}}, device);
        EXPECT_EQ(device.counting().allocate_calls, 1); // resident initializer
        try {
            (void)model.Run({{"x", {1.0F, -2.0F}}});
            FAIL() << "expected InferenceException from the throwing Add kernel";
        } catch (const nert::InferenceException& e) {
            EXPECT_NE(std::string(e.what()).find("boom"), std::string::npos) << e.what();
        }
        // Staged input + the Add output were released on unwind.
        EXPECT_EQ(device.counting().allocate_calls, 3);
        EXPECT_EQ(device.counting().release_calls, 2);
    }
    EXPECT_EQ(device.counting().allocate_calls, 3);
    EXPECT_EQ(device.counting().release_calls, 3);
}

// A throwing kernel still releases the staged graph inputs: the per-run
// guard owns them, not the happy path.
TEST(NertExecutor, ThrowingKernelReleasesStagedInputs) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    ThrowOnAddTable throwing_table;
    NonHostTestDevice device(&throwing_table);
    {
        nert::Model model(graph, {{"x", {2}}}, device);
        EXPECT_EQ(device.counting().allocate_calls, 2);
        try {
            (void)model.Run({{"x", {1.0F, -2.0F}}});
            FAIL() << "expected InferenceException from the throwing Add kernel";
        } catch (const nert::InferenceException& e) {
            EXPECT_NE(std::string(e.what()).find("boom"), std::string::npos) << e.what();
        }
        // The staged input released on unwind; arena + resident stay held.
        EXPECT_EQ(device.counting().allocate_calls, 3);
        EXPECT_EQ(device.counting().release_calls, 1);
    }
    EXPECT_EQ(device.counting().release_calls, 3);
}

TEST(NertExecutor, TypedRunInt64InputBoolOutput) {
    const nert::Graph graph = load_hermetic_int64_bool({2, 2, 2, 2});
    nert::Model model(graph, {{"ids", {4}}}, nert::CpuDevice());
    EXPECT_EQ(model.shapes().tensors.at("ids").dtype, nert::DataType::Int64);

    const nert::HostTensorMap result = model.RunTyped({{"ids", int64_host_tensor({1, 2, 3, -7})}});
    ASSERT_EQ(result.size(), 2U);

    const nert::HostTensor& sum = result.at("sum");
    EXPECT_EQ(sum.dtype, nert::DataType::Int64);
    EXPECT_EQ(sum.dims, (std::vector<int64_t>{4}));
    ASSERT_EQ(sum.bytes.size(), 4U * sizeof(int64_t));
    std::vector<int64_t> sums(4);
    std::memcpy(sums.data(), sum.bytes.data(), sum.bytes.size());
    EXPECT_EQ(sums, (std::vector<int64_t>{3, 4, 5, -5}));

    const nert::HostTensor& flag = result.at("flag");
    EXPECT_EQ(flag.dtype, nert::DataType::Bool);
    EXPECT_EQ(flag.dims, (std::vector<int64_t>{4}));
    EXPECT_EQ(flag.bytes, (std::vector<uint8_t>{0, 0, 1, 0}));

    // A second run reuses the arena and sees the new input.
    const nert::HostTensorMap again = model.RunTyped({{"ids", int64_host_tensor({5, 0, 2, 9})}});
    EXPECT_EQ(again.at("flag").bytes, (std::vector<uint8_t>{1, 0, 0, 1}));
}

TEST(NertExecutor, TypedRunMatchesFloatRun) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    nert::Model model(graph, {{"x", {2}}}, nert::CpuDevice());

    const std::vector<float> input{1.0F, -2.0F};
    nert::HostTensor host;
    host.dtype = nert::DataType::Float32;
    host.dims = {2};
    host.bytes.resize(input.size() * sizeof(float));
    std::memcpy(host.bytes.data(), input.data(), host.bytes.size());

    const nert::HostTensorMap typed = model.RunTyped({{"x", host}});
    const nert::HostTensor& z = typed.at("z");
    EXPECT_EQ(z.dtype, nert::DataType::Float32);
    std::vector<float> values(2);
    ASSERT_EQ(z.bytes.size(), values.size() * sizeof(float));
    std::memcpy(values.data(), z.bytes.data(), z.bytes.size());
    EXPECT_EQ(values, model.Run({{"x", input}}).outputs.at("z"));
}

TEST(NertExecutor, FixtureResNet18RunsAndIsFinite) {
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty()) << "missing fixture: set " << fixture_env_name()
                                  << " to the opset-18 ResNet-18 ONNX model; a missing fixture must fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture)) << "fixture does not exist: " << fixture;

    const nert::Graph graph = nert::LoadGraphFromFile(fixture);
    ASSERT_FALSE(graph.inputs.empty());
    ASSERT_FALSE(graph.outputs.empty());
    ASSERT_EQ(graph.inputs.front().second.dims, (std::vector<int64_t>{1, 3, 224, 224}));

    const std::string input_name = graph.inputs.front().first;
    const std::string output_name = graph.outputs.front().first;
    nert::Model model(graph, {{input_name, {1, 3, 224, 224}}}, nert::CpuDevice());

    const std::vector<float> input(static_cast<std::size_t>(1 * 3 * 224 * 224), 0.01F);
    const nert::InferenceResult result = model.Run({{input_name, input}});

    ASSERT_EQ(result.outputs.count(output_name), 1U);
    const std::vector<float>& output = result.outputs.at(output_name);
    EXPECT_EQ(output.size(), 1000U);
    for (std::size_t i = 0; i < output.size(); ++i) {
        EXPECT_TRUE(std::isfinite(output[i])) << "output[" << i << "] = " << output[i];
    }
}

// ===========================================================================
// EngineExecutorNegative.* — rejections
// ===========================================================================

TEST(NertExecutorNegative, MissingInputThrows) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    nert::Model model(graph, {{"x", {2}}}, nert::CpuDevice());
    EXPECT_THROW(model.Run({}), nert::InferenceException);
}

TEST(NertExecutorNegative, WrongInputSizeThrows) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    nert::Model model(graph, {{"x", {2}}}, nert::CpuDevice());
    EXPECT_THROW(model.Run({{"x", {1.0F}}}), nert::InferenceException);
}

TEST(NertExecutorNegative, OpWithoutKernelThrows) {
    const nert::Graph graph = load_hermetic_add_relu({-0.5F, 0.5F});
    ASSERT_FALSE(graph.nodes.empty());
    TestDevice device(true); // advertises no kernels
    nert::Model model(graph, {{"x", {2}}}, device);

    try {
        model.Run({{"x", {1.0F, -2.0F}}});
        FAIL() << "expected InferenceException for an op with no kernel";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find(graph.nodes.front().name), std::string::npos) << message;
        EXPECT_NE(message.find(graph.nodes.front().op_type), std::string::npos) << message;
    }
}

TEST(NertExecutorNegative, FloatRunRejectsNonFloatGraphInput) {
    const nert::Graph graph = load_hermetic_int64_bool({2, 2, 2, 2});
    nert::Model model(graph, {{"ids", {4}}}, nert::CpuDevice());
    try {
        model.Run({{"ids", {1.0F, 2.0F, 3.0F, 4.0F}}});
        FAIL() << "expected InferenceException for a non-float graph input";
    } catch (const nert::InferenceException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("ids"), std::string::npos) << message;
        EXPECT_NE(message.find("int64"), std::string::npos) << message;
    }
}

TEST(NertExecutorNegative, TypedRunRejectsMistypedOrMissizedInput) {
    const nert::Graph graph = load_hermetic_int64_bool({2, 2, 2, 2});
    nert::Model model(graph, {{"ids", {4}}}, nert::CpuDevice());

    EXPECT_THROW(model.RunTyped({}), nert::InferenceException);

    nert::HostTensor wrong_dtype = int64_host_tensor({1, 2, 3, 4});
    wrong_dtype.dtype = nert::DataType::Float32;
    EXPECT_THROW(model.RunTyped({{"ids", wrong_dtype}}), nert::InferenceException);

    nert::HostTensor short_bytes = int64_host_tensor({1, 2, 3, 4});
    short_bytes.bytes.resize(3 * sizeof(int64_t));
    EXPECT_THROW(model.RunTyped({{"ids", short_bytes}}), nert::InferenceException);

    nert::HostTensor wrong_dims = int64_host_tensor({1, 2, 3, 4});
    wrong_dims.dims = {2, 2};
    EXPECT_THROW(model.RunTyped({{"ids", wrong_dims}}), nert::InferenceException);
}
