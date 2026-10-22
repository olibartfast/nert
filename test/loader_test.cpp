// Model-loader tests.
//
// Two suites, split for ctest filters (suite names must not carry the ctest
// NAME colons, so the filters live in the add_test COMMAND):
//   EngineLoader.*           — positives, fixture + hermetic wire-bytes cases
//   EngineLoaderNegative.*   — rejection classes with message-content asserts
//
// Positives run on real fixture bytes (the reference ResNet-18 export, with
// external weight data) when the fixture is resolvable via the
// NERT_FIXTURE environment variable or CMake cache variable. A
// missing fixture FAILS the test — never skipped or mocked — so acceptance
// cannot silently pass without the real bytes. The hermetic cases are
// hand-encoded model bytes and keep CI green without torch or ONNX. The field
// numbers the test encoders use mirror those in ModelLoader.cpp; no public
// header carries any.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "nert/device.hpp"
#include "nert/executor.hpp"
#include "nert/graph.hpp"
#include "nert/model_loader.hpp"

namespace {

// ---------------------------------------------------------------------------
// Hand-rolled protobuf wire encoders (test-side twin of the reader).
//
// Field numbers: ModelProto graph=7 opset_import=8; GraphProto node=1
// initializer=5 input=11 output=12; NodeProto input=1 output=2 name=3
// op_type=4 attribute=5; AttributeProto name=1 f=2 i=3 s=4 type=20;
// TensorProto dims=1 data_type=2 name=8 raw_data=9; OpsetIdList
// domain=1 version=2; ValueInfoProto name=1 type=2; TypeProto tensor_type=1;
// TensorType elem_type=1 shape=2; TensorShapeProto dim=1; Dimension
// dim_value=1 dim_param=2.
// ---------------------------------------------------------------------------

void put_varint(std::string& out, uint64_t value)
{
    while (value >= 0x80U) {
        out.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void put_tag(std::string& out, int field_number, uint8_t wire_type)
{
    put_varint(out, (static_cast<uint64_t>(field_number) << 3) | wire_type);
}

void put_length_delimited(std::string& out, int field_number,
    const std::string& payload)
{
    put_tag(out, field_number, 2);
    put_varint(out, payload.size());
    out.append(payload);
}

void put_sub(std::string& out, int field_number, const std::string& sub)
{
    put_length_delimited(out, field_number, sub);
}

void put_varint_field(std::string& out, int field_number, uint64_t value)
{
    put_tag(out, field_number, 0);
    put_varint(out, value);
}

void put_fixed32_field(std::string& out, int field_number, uint32_t bits)
{
    put_tag(out, field_number, 5);
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((bits >> (8 * i)) & 0xFFU));
    }
}

std::string encode_opset(int64_t version)
{
    // Empty domain string decodes as the default/"ai.onnx" domain.
    std::string leaving; // domain (field 1) omitted in the import entry
    put_varint_field(leaving, 2, static_cast<uint64_t>(version));
    return leaving;
}

std::string encode_value_info(const std::string& name, int64_t elem_type,
    const std::vector<int64_t>& dims, bool dynamic = false)
{
    std::string vi;
    put_length_delimited(vi, 1, name);
    std::string tensor_type;
    put_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
    std::string shape;
    for (const int64_t dim : dims) {
        std::string dim_msg;
        if (dynamic) {
            put_length_delimited(dim_msg, 2, std::string("N"));
        } else {
            put_varint_field(dim_msg, 1, static_cast<uint64_t>(dim));
        }
        put_sub(shape, 1, dim_msg);
    }
    put_sub(tensor_type, 2, shape);
    std::string type_proto;
    put_sub(type_proto, 1, tensor_type);
    put_sub(vi, 2, type_proto);
    return vi;
}

std::string encode_node(const std::string& name, const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& attr_bytes)
{
    std::string n;
    for (const std::string& input : inputs) {
        put_length_delimited(n, 1, input);
    }
    for (const std::string& output : outputs) {
        put_length_delimited(n, 2, output);
    }
    if (!name.empty()) {
        put_length_delimited(n, 3, name);
    }
    put_length_delimited(n, 4, op_type);
    for (const std::string& attr : attr_bytes) {
        put_sub(n, 5, attr);
    }
    return n;
}

std::string encode_attribute_int(const std::string& name, int64_t value)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_varint_field(a, 3, static_cast<uint64_t>(value));
    put_varint_field(a, 20, 2); // INT
    return a;
}

std::string encode_attribute_float(const std::string& name, float value)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_fixed32_field(a, 2,
        *reinterpret_cast<const uint32_t*>(&value));
    put_varint_field(a, 20, 1); // FLOAT
    return a;
}

std::string encode_attribute_string(const std::string& name,
    const std::string& value)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_length_delimited(a, 4, value);
    put_varint_field(a, 20, 3); // STRING
    return a;
}

std::string encode_attribute_ints(const std::string& name,
    const std::vector<int64_t>& values)
{
    std::string a;
    put_length_delimited(a, 1, name);
    for (const int64_t value : values) {
        put_varint_field(a, 8, static_cast<uint64_t>(value));
    }
    put_varint_field(a, 20, 7); // INTS
    return a;
}

std::string encode_attribute_name_only(const std::string& name)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_varint_field(a, 20, 2); // INT payload kind, but no i field
    return a;
}

std::string encode_initializer(const std::string& name,
    const std::vector<int64_t>& dims, const std::vector<float>& values)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 1); // FLOAT
    put_length_delimited(t, 8, name);
    const std::string raw(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(float));
    put_length_delimited(t, 9, raw);
    return t;
}

// An empty FLOAT tensor, as opset-11 Resize writes for an unused roi or
// scales input. No data field is emitted for the zero-element tensor.
std::string encode_initializer_empty_float(const std::string& name)
{
    std::string t;
    put_varint_field(t, 1, 0); // dims = [0]
    put_varint_field(t, 2, 1); // FLOAT
    put_length_delimited(t, 8, name);
    return t;
}

// An integer shape-constant initializer. `via_raw_data` mirrors real exports
// (little-endian raw bytes, TensorProto.raw_data = 9); the alternative writes
// the packed int64_data field (7). Both decode paths are exercised.
std::string encode_initializer_int64(const std::string& name,
    const std::vector<int64_t>& dims, const std::vector<int64_t>& values,
    bool via_raw_data = true)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 7); // INT64
    put_length_delimited(t, 8, name);
    if (via_raw_data) {
        const std::string raw(reinterpret_cast<const char*>(values.data()),
            values.size() * sizeof(int64_t));
        put_length_delimited(t, 9, raw);
    } else {
        for (const int64_t value : values) {
            put_varint_field(t, 7, static_cast<uint64_t>(value));
        }
    }
    return t;
}

// A non-FLOAT, non-INT64 initializer used to assert the rejection message.
std::string encode_initializer_float16(const std::string& name,
    const std::vector<int64_t>& dims, const std::string& raw)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, 10); // FLOAT16
    put_length_delimited(t, 8, name);
    put_length_delimited(t, 9, raw);
    return t;
}

// Embedded TensorProto payload for a Constant 'value' attribute, stored via
// raw_data (mirrors the survey oracle, which carries all Constants raw).
std::string encode_constant_tensor(int64_t elem_type,
    const std::vector<int64_t>& dims, const std::string& raw)
{
    std::string t;
    for (const int64_t dim : dims) {
        put_varint_field(t, 1, static_cast<uint64_t>(dim));
    }
    put_varint_field(t, 2, static_cast<uint64_t>(elem_type));
    put_length_delimited(t, 9, raw);
    return t;
}

std::string raw_bytes_float(const std::vector<float>& values)
{
    return std::string(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(float));
}

std::string raw_bytes_int64(const std::vector<int64_t>& values)
{
    return std::string(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(int64_t));
}

// A Constant 'value' attribute (AttributeProto type TENSOR = 4, tensor = 5).
std::string encode_attribute_tensor(const std::string& name,
    const std::string& tensor_bytes)
{
    std::string a;
    put_length_delimited(a, 1, name);
    put_sub(a, 5, tensor_bytes);
    put_varint_field(a, 20, 4); // TENSOR
    return a;
}

std::string encode_constant_node(const std::string& name,
    const std::string& output, const std::string& tensor_bytes)
{
    return encode_node(name, "Constant", {}, {output},
        {encode_attribute_tensor("value", tensor_bytes)});
}

std::string encode_model(const std::string& graph_body, int64_t opset = 18)
{
    std::string model;
    put_sub(model, 8, encode_opset(opset));
    put_sub(model, 7, graph_body);
    return model;
}

// Row-wise softmax used as the oracle for the legacy Softmax expansion.
std::vector<float> row_softmax(const std::vector<float>& values, int64_t cols)
{
    std::vector<float> out(values.size(), 0.0F);
    for (size_t start = 0; start < values.size(); start += static_cast<size_t>(cols)) {
        double peak = values[start];
        for (int64_t i = 1; i < cols; ++i) {
            if (values[start + static_cast<size_t>(i)] > peak) {
                peak = values[start + static_cast<size_t>(i)];
            }
        }
        double total = 0.0;
        for (int64_t i = 0; i < cols; ++i) {
            total += std::exp(static_cast<double>(values[start + static_cast<size_t>(i)]) - peak);
        }
        for (int64_t i = 0; i < cols; ++i) {
            out[start + static_cast<size_t>(i)] = static_cast<float>(
                std::exp(static_cast<double>(values[start + static_cast<size_t>(i)]) - peak) / total);
        }
    }
    return out;
}

// Holds hermetic models in temporary files; returns the written path.
std::string write_model_bytes(const std::string& name, const std::string& bytes)
{
    const std::string dir =
        (std::filesystem::temp_directory_path() / "engine_loader_tests")
            .string();
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

std::string load_error_message(const std::string& bytes,
    const std::string& file_name)
{
    try {
        nert::LoadGraphFromFile(write_model_bytes(file_name, bytes));
    } catch (const nert::ModelLoadException& e) {
        return e.what();
    }
    return "<no ModelLoadException thrown>";
}

const char* fixture_env_name() { return "NERT_FIXTURE"; }

} // namespace

// ===========================================================================
// EngineLoader.* — positive cases
// ===========================================================================

class EngineLoader : public ::testing::Test {
protected:
    // Env var first; never hard-codes a path inside the test source.
    static std::string fixture_path()
    {
        const char* from_env = std::getenv(fixture_env_name());
        return from_env != nullptr ? std::string(from_env) : std::string();
    }
};

TEST_F(EngineLoader, RealFixtureLoads)
{
    const std::string fixture = fixture_path();
    ASSERT_FALSE(fixture.empty())
        << "missing fixture: set " << fixture_env_name()
        << " to the opset-18 ResNet-18 ONNX model; a missing fixture must "
           "fail, never skip";
    ASSERT_TRUE(std::filesystem::exists(fixture))
        << "fixture path from " << fixture_env_name()
        << " does not exist: " << fixture;
    EXPECT_NO_THROW(nert::LoadGraphFromFile(fixture));
}

TEST_F(EngineLoader, RealFixtureNodeCountAndOpsHistogram)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const nert::Graph graph = nert::LoadGraphFromFile(fixture);
    ASSERT_EQ(graph.nodes.size(), 49U); // reference model node count

    std::map<std::string, size_t> histogram;
    for (const nert::Node& node : graph.nodes) {
        histogram[node.op_type]++;
    }
    EXPECT_EQ(histogram.at("Conv"), 20U);
    EXPECT_EQ(histogram.at("Relu"), 17U);
    EXPECT_EQ(histogram.at("Add"), 8U);
    EXPECT_EQ(histogram.at("MaxPool"), 1U);
    EXPECT_EQ(histogram.at("ReduceMean"), 1U);
    EXPECT_EQ(histogram.at("Reshape"), 1U);
    EXPECT_EQ(histogram.at("Gemm"), 1U);
    EXPECT_EQ(histogram.size(), 7U); // no operator outside the supported set
}

TEST_F(EngineLoader, RealFixtureInitializerCount)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const nert::Graph graph = nert::LoadGraphFromFile(fixture);
    EXPECT_EQ(graph.initializers.size(), 44U); // reference model initializers
    for (const auto& init : graph.initializers) {
        EXPECT_FALSE(init.first.empty());
        if (init.second.dtype == nert::DataType::Int64) {
            EXPECT_FALSE(
                std::get<std::vector<std::int64_t>>(init.second.values).empty());
        } else {
            EXPECT_EQ(init.second.dtype, nert::DataType::Float32);
            EXPECT_FALSE(std::get<std::vector<float>>(init.second.values)
                    .empty()); // resolved, real float payloads
        }
    }
}

// The fixture carries int64 Reshape shape constants alongside FLOAT weights.
// Both decode, and the value table records their dtype.

TEST_F(EngineLoader, RealFixtureInt64ShapeConstants)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const nert::Graph graph = nert::LoadGraphFromFile(fixture);

    ASSERT_EQ(graph.initializers.count("val_230"), 1U);
    const nert::Initializer& shape = graph.initializers.at("val_230");
    EXPECT_EQ(shape.dtype, nert::DataType::Int64);
    EXPECT_EQ(shape.dims, (std::vector<std::int64_t>{2}));
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(shape.values),
        (std::vector<std::int64_t>{1, 512}));

    ASSERT_EQ(graph.tensors.count("val_230"), 1U);
    EXPECT_EQ(graph.tensors.at("val_230").dtype, nert::DataType::Int64);
}

TEST_F(EngineLoader, RealFixtureIODtypesAndShapes)
{
    const std::string fixture = fixture_path();
    ASSERT_TRUE(std::filesystem::exists(fixture));
    const nert::Graph graph = nert::LoadGraphFromFile(fixture);
    ASSERT_EQ(graph.inputs.size(), 1U);
    EXPECT_EQ(graph.inputs.front().second.dtype, nert::DataType::Float32);
    EXPECT_EQ(graph.inputs.front().second.dims,
        (std::vector<std::int64_t>{1, 3, 224, 224})); // FLOAT [1,3,224,224]
    ASSERT_EQ(graph.outputs.size(), 1U);
    EXPECT_EQ(graph.outputs.front().second.dtype, nert::DataType::Float32);
    EXPECT_EQ(graph.outputs.front().second.dims,
        (std::vector<std::int64_t>{1, 1000})); // -> FLOAT [1,1000]
}

// Hermetic positive: single Relu node, no initializers, static IO.

TEST_F(EngineLoader, HermeticReluNodeAndGraphIo)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("relu_model.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().name, "relu0");
    EXPECT_EQ(graph.nodes.front().op_type, "Relu");
    EXPECT_EQ(graph.nodes.front().inputs, (std::vector<std::string>{"x"}));
    EXPECT_EQ(graph.nodes.front().outputs, (std::vector<std::string>{"y"}));
    EXPECT_TRUE(graph.nodes.front().attributes.empty());
    EXPECT_EQ(graph.initializers.size(), 0U);
    ASSERT_EQ(graph.inputs.size(), 1U);
    EXPECT_EQ(graph.inputs.front().first, "x");
    EXPECT_EQ(graph.inputs.front().second.dtype, nert::DataType::Float32);
    EXPECT_EQ(graph.inputs.front().second.dims,
        (std::vector<int64_t>{1, 2, 3, 4}));
    ASSERT_EQ(graph.outputs.size(), 1U);
    EXPECT_EQ(graph.outputs.front().first, "y");
    EXPECT_EQ(graph.outputs.front().second.dims,
        (std::vector<int64_t>{1, 2, 3, 4}));
}

// Hermetic positive: Gemm typed attributes + FLOAT initializer via raw_data.

TEST_F(EngineLoader, HermeticGemmAttributesAndInitializer)
{
    std::string body;
    put_sub(body, 5, encode_initializer("w", {2, 3},
        {1.0F, -2.5F, 3.0F, 4.25F, 0.5F, -1.0F}));
    put_sub(body, 1, encode_node("gemm0", "Gemm", {"y", "w"}, {"z"},
        {encode_attribute_float("alpha", 1.0F),
            encode_attribute_float("beta", 1.0F),
            encode_attribute_int("transA", 0),
            encode_attribute_int("transB", 1)}));
    put_sub(body, 11, encode_value_info("y", 1, {2, 2}));
    put_sub(body, 12, encode_value_info("z", 1, {2, 3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("gemm_model.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    std::map<std::string, nert::Attribute> attrs;
    for (const nert::Attribute& a : graph.nodes.front().attributes) {
        attrs[a.name] = a;
    }
    ASSERT_EQ(attrs.size(), 4U);
    EXPECT_EQ(std::holds_alternative<float>(attrs.at("alpha").value), true);
    EXPECT_EQ(std::get<float>(attrs.at("alpha").value), 1.0F);
    EXPECT_EQ(std::get<float>(attrs.at("beta").value), 1.0F);
    EXPECT_EQ(std::get<std::int64_t>(attrs.at("transA").value), 0);
    EXPECT_EQ(std::get<std::int64_t>(attrs.at("transB").value), 1);
    ASSERT_EQ(graph.initializers.size(), 1U);
    const nert::Initializer& w = graph.initializers.at("w");
    EXPECT_EQ(w.dtype, nert::DataType::Float32);
    EXPECT_EQ(std::get<std::vector<float>>(w.values),
        (std::vector<float>{1.0F, -2.5F, 3.0F, 4.25F, 0.5F, -1.0F}));
}

// Hermetic positive: an int64 shape constant feeds Reshape, so a float32 graph
// with an integer constant loads. Both storage paths decode.

TEST_F(EngineLoader, HermeticInt64ReshapeConstant)
{
    for (const bool via_raw_data : {true, false}) {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("shape", {2}, {1, 512},
                          via_raw_data));
        put_sub(body, 1, encode_node("reshape0", "Reshape", {"x", "shape"},
                          {"y"}, {encode_attribute_int("allowzero", 0)}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {1, 48}));
        put_sub(body, 13, encode_value_info("shape", 7 /*INT64*/, {2}));

        const nert::Graph graph = nert::LoadGraphFromFile(write_model_bytes(
            via_raw_data ? "reshape_raw.onnx" : "reshape_varint.onnx",
            encode_model(body)));
        ASSERT_EQ(graph.initializers.size(), 1U);
        const nert::Initializer& shape = graph.initializers.at("shape");
        EXPECT_EQ(shape.dtype, nert::DataType::Int64);
        EXPECT_EQ(shape.dims, (std::vector<std::int64_t>{2}));
        EXPECT_EQ(std::get<std::vector<std::int64_t>>(shape.values),
            (std::vector<std::int64_t>{1, 512}));
        ASSERT_EQ(graph.tensors.count("shape"), 1U);
        EXPECT_EQ(graph.tensors.at("shape").dtype, nert::DataType::Int64);
    }
}

// Hermetic positive: a float32 Constant folds into initializers. The Constant
// node vanishes from graph.nodes and the Add consumer keeps naming the folded
// tensor, which now resolves as an initializer.
TEST_F(EngineLoader, HermeticConstantFoldingFloat)
{
    const std::vector<float> values{1.0F, -2.5F, 3.25F};
    std::string body;
    put_sub(body, 1,
        encode_constant_node("const_f0", "c0",
            encode_constant_tensor(1 /*FLOAT*/, {3}, raw_bytes_float(values))));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "c0"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {3}));
    put_sub(body, 12, encode_value_info("y", 1, {3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("const_fold_float.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().name, "add0");
    EXPECT_EQ(graph.nodes.front().op_type, "Add");
    EXPECT_EQ(graph.nodes.front().inputs,
        (std::vector<std::string>{"x", "c0"}));
    ASSERT_EQ(graph.initializers.count("c0"), 1U);
    const nert::Initializer& folded = graph.initializers.at("c0");
    EXPECT_EQ(folded.dtype, nert::DataType::Float32);
    EXPECT_EQ(folded.dims, (std::vector<std::int64_t>{3}));
    EXPECT_EQ(std::get<std::vector<float>>(folded.values), values);
}

// Hermetic positive: a DOUBLE Constant (an exporter anchor grid) and its
// DOUBLE value-info entry load as float32, the only compute type.
TEST_F(EngineLoader, HermeticDoubleConstantNarrowsToFloat)
{
    const std::vector<double> values{0.5, 79.5};
    const std::string raw(reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(double));
    std::string body;
    put_sub(body, 1,
        encode_constant_node("const_d0", "c0",
            encode_constant_tensor(11 /*DOUBLE*/, {2}, raw)));
    put_sub(body, 1, encode_node("cast0", "Cast", {"c0"}, {"c1"},
        {encode_attribute_int("to", 1)}));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "c1"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));
    put_sub(body, 13, encode_value_info("c0", 11, {2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("const_fold_double.onnx", encode_model(body)));
    ASSERT_EQ(graph.initializers.count("c0"), 1U);
    const nert::Initializer& folded = graph.initializers.at("c0");
    EXPECT_EQ(folded.dtype, nert::DataType::Float32);
    EXPECT_EQ(std::get<std::vector<float>>(folded.values),
        (std::vector<float>{0.5F, 79.5F}));
    ASSERT_EQ(graph.tensors.count("c0"), 1U);
    EXPECT_EQ(graph.tensors.at("c0").dtype, nert::DataType::Float32);
}

// Hermetic positive: an identical repeated value-info entry (torch 2.12
// exports every entry twice) loads; a conflicting repeat is rejected by name.
TEST_F(EngineLoader, HermeticDuplicateValueInfo)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"h"}, {}));
    put_sub(body, 1, encode_node("relu1", "Relu", {"h"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));
    put_sub(body, 13, encode_value_info("h", 1, {2}));
    std::string same = body;
    put_sub(same, 13, encode_value_info("h", 1, {2}));
    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("dup_value_info.onnx", encode_model(same)));
    EXPECT_EQ(graph.tensors.at("h").dims, (std::vector<std::int64_t>{2}));

    std::string conflict = body;
    put_sub(conflict, 13, encode_value_info("h", 1, {3}));
    const std::string message =
        load_error_message(encode_model(conflict), "dup_value_info_bad.onnx");
    EXPECT_NE(message.find("conflicting"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("'h'"), std::string::npos) << "message: " << message;
}

// Hermetic positive: a BOOL initializer (an attention mask) loads one byte per
// element.
TEST_F(EngineLoader, HermeticBoolInitializerLoads)
{
    std::string body;
    put_sub(body, 5, [] {
        std::string t;
        put_varint_field(t, 1, 3);
        put_varint_field(t, 2, 9 /*BOOL*/);
        put_length_delimited(t, 8, "mask");
        put_length_delimited(t, 9, std::string("\x01\x00\x01", 3));
        return t;
    }());
    put_sub(body, 1, encode_node("where0", "Where", {"mask", "x", "x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {3}));
    put_sub(body, 12, encode_value_info("y", 1, {3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("bool_init.onnx", encode_model(body)));
    const nert::Initializer& mask = graph.initializers.at("mask");
    EXPECT_EQ(mask.dtype, nert::DataType::Bool);
    EXPECT_EQ(std::get<std::vector<std::uint8_t>>(mask.values),
        (std::vector<std::uint8_t>{1, 0, 1}));
}

// Hermetic positive: an int64 Constant folds into initializers and feeds
// Reshape, mirroring the survey oracle's int64 Constants.
TEST_F(EngineLoader, HermeticConstantFoldingInt64)
{
    const std::vector<int64_t> values{1, 512};
    std::string body;
    put_sub(body, 1,
        encode_constant_node("const_i0", "shape",
            encode_constant_tensor(7 /*INT64*/, {2}, raw_bytes_int64(values))));
    put_sub(body, 1, encode_node("reshape0", "Reshape", {"x", "shape"},
                      {"y"}, {encode_attribute_int("allowzero", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 48}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("const_fold_int64.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Reshape");
    ASSERT_EQ(graph.initializers.count("shape"), 1U);
    const nert::Initializer& folded = graph.initializers.at("shape");
    EXPECT_EQ(folded.dtype, nert::DataType::Int64);
    EXPECT_EQ(folded.dims, (std::vector<std::int64_t>{2}));
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(folded.values), values);
}

// Hermetic positive: Flatten loads with its axis attribute.
TEST_F(EngineLoader, HermeticFlattenLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("flat0", "Flatten", {"x"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 12}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("flatten_model.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Flatten");
    ASSERT_EQ(graph.nodes.front().attributes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().attributes.front().name, "axis");
    EXPECT_EQ(
        std::get<std::int64_t>(graph.nodes.front().attributes.front().value),
        1);
}

// Hermetic positive: intermediate value-info entries may be float32, int64,
// or bool (graph inputs/outputs accept the same three dtypes, below).
TEST_F(EngineLoader, HermeticBoolValueInfoAccepted)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));
    put_sub(body, 13, encode_value_info("flag", 9 /*BOOL*/, {2}));
    put_sub(body, 13, encode_value_info("idx", 7 /*INT64*/, {2}));
    put_sub(body, 13, encode_value_info("act", 1 /*FLOAT*/, {2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("bool_vi.onnx", encode_model(body)));
    ASSERT_EQ(graph.tensors.count("flag"), 1U);
    EXPECT_EQ(graph.tensors.at("flag").dtype, nert::DataType::Bool);
    ASSERT_EQ(graph.tensors.count("idx"), 1U);
    EXPECT_EQ(graph.tensors.at("idx").dtype, nert::DataType::Int64);
    ASSERT_EQ(graph.tensors.count("act"), 1U);
    EXPECT_EQ(graph.tensors.at("act").dtype, nert::DataType::Float32);
}

// Hermetic positive: the no-attribute elementwise ops (Mul/Div/Sub/Sigmoid)
// chain without attributes.
TEST_F(EngineLoader, HermeticNewElementwiseLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("sig0", "Sigmoid", {"x"}, {"s"}, {}));
    put_sub(body, 1, encode_node("mul0", "Mul", {"x", "s"}, {"m"}, {}));
    put_sub(body, 1, encode_node("div0", "Div", {"m", "x"}, {"d"}, {}));
    put_sub(body, 1, encode_node("sub0", "Sub", {"d", "m"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 4, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("new_elem.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 4U);
    EXPECT_EQ(graph.nodes[0].op_type, "Sigmoid");
    EXPECT_EQ(graph.nodes[1].op_type, "Mul");
    EXPECT_EQ(graph.nodes[2].op_type, "Div");
    EXPECT_EQ(graph.nodes[3].op_type, "Sub");
    for (const nert::Node& node : graph.nodes) {
        EXPECT_TRUE(node.attributes.empty()) << node.op_type;
    }
}

// Hermetic positive: Split carries axis plus a split-sizes input, Concat
// carries axis over two inputs.
TEST_F(EngineLoader, HermeticConcatSplitLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("split", {2}, {16, 16}));
    put_sub(body, 1, encode_node("split0", "Split", {"x", "split"}, {"a", "b"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 1, encode_node("concat0", "Concat", {"a", "b"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 32, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 32, 4, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("concat_split.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Split");
    ASSERT_EQ(graph.nodes[0].attributes.size(), 1U);
    EXPECT_EQ(
        std::get<std::int64_t>(graph.nodes[0].attributes.front().value), 1);
    EXPECT_EQ(graph.nodes[1].op_type, "Concat");
    ASSERT_EQ(graph.nodes[1].attributes.size(), 1U);
    EXPECT_EQ(
        std::get<std::int64_t>(graph.nodes[1].attributes.front().value), 1);
}

// Hermetic positive: Unsqueeze/Expand take no attributes; their axes/shape
// tables arrive as inputs.
TEST_F(EngineLoader, HermeticUnsqueezeExpandLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {-1}));
    put_sub(body, 5, encode_initializer_int64("eshape", {3}, {1, 2, 2}));
    put_sub(body, 1, encode_node("unsq0", "Unsqueeze", {"x", "axes"}, {"u"},
                      {}));
    put_sub(body, 1, encode_node("exp0", "Expand", {"u", "eshape"}, {"y"},
                      {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("unsq_expand.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Unsqueeze");
    EXPECT_TRUE(graph.nodes[0].attributes.empty());
    EXPECT_EQ(graph.nodes[1].op_type, "Expand");
    EXPECT_TRUE(graph.nodes[1].attributes.empty());
}

// Hermetic positive: Transpose carries perm; Gather/GatherElements carry axis
// with indices as inputs.
TEST_F(EngineLoader, HermeticTransposeGatherLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("idx", {2}, {0, 1}));
    put_sub(body, 1, encode_node("tr0", "Transpose", {"x"}, {"t"},
                      {encode_attribute_ints("perm", {0, 2, 1})}));
    put_sub(body, 1, encode_node("g0", "Gather", {"t", "idx"}, {"g"},
                      {encode_attribute_int("axis", 0)}));
    put_sub(body, 1, encode_node("ge0", "GatherElements", {"t", "idx"}, {"h"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("h", 1, {2, 3, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("tr_gather.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 3U);
    EXPECT_EQ(graph.nodes[0].op_type, "Transpose");
    EXPECT_EQ(std::get<std::vector<std::int64_t>>(
                  graph.nodes[0].attributes.front().value),
        (std::vector<std::int64_t>{0, 2, 1}));
    EXPECT_EQ(graph.nodes[1].op_type, "Gather");
    EXPECT_EQ(graph.nodes[2].op_type, "GatherElements");
}

// Hermetic positive: Cast pins `to` to FLOAT (1) and INT64 (7).
TEST_F(EngineLoader, HermeticCastLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("cast0", "Cast", {"x"}, {"c"},
                      {encode_attribute_int("to", 7)}));
    put_sub(body, 1, encode_node("cast1", "Cast", {"c"}, {"y"},
                      {encode_attribute_int("to", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("cast.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[0].attributes.front().value),
        7);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[1].attributes.front().value),
        1);
}

// Hermetic positive: Softmax carries axis; Resize carries the surveyed nearest
// policy (mode/asymmetric/floor/-0.75) with scales as third input.
TEST_F(EngineLoader, HermeticSoftmaxResizeLoads)
{
    std::string body;
    put_sub(body, 5,
        encode_initializer("scales", {4}, {1.0F, 1.0F, 2.0F, 2.0F}));
    put_sub(body, 1, encode_node("sm0", "Softmax", {"x"}, {"s"},
                      {encode_attribute_int("axis", -1)}));
    put_sub(body, 1, encode_node("rs0", "Resize", {"x", "", "scales"}, {"y"},
                      {encode_attribute_string("mode", "nearest"),
                          encode_attribute_string(
                              "coordinate_transformation_mode", "asymmetric"),
                          encode_attribute_string("nearest_mode", "floor"),
                          encode_attribute_float("cubic_coeff_a", -0.75F)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 8, 8}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("sm_resize.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Softmax");
    EXPECT_EQ(graph.nodes[1].op_type, "Resize");
    EXPECT_EQ(graph.nodes[1].inputs,
        (std::vector<std::string>{"x", "", "scales"}));
    ASSERT_EQ(graph.nodes[1].attributes.size(), 4U);
}

// Hermetic positive: the ONNX no-op defaults antialias=0 and
// exclude_outside=0 are admitted and preserved as ints. The loader is the
// single admission point, so the OpenCL row can independently re-check the
// same values without changing the admitted graph surface.
TEST_F(EngineLoader, HermeticResizeNoOpAttributesLoad)
{
    std::string body;
    put_sub(body, 5,
        encode_initializer("scales", {4}, {1.0F, 1.0F, 2.0F, 2.0F}));
    put_sub(body, 1, encode_node("rs_noop", "Resize", {"x", "", "scales"}, {"y"},
                      {encode_attribute_string("mode", "linear"),
                          encode_attribute_string(
                              "coordinate_transformation_mode", "half_pixel"),
                          encode_attribute_int("antialias", 0),
                          encode_attribute_int("exclude_outside", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 8, 8}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("rs_noop.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    const nert::Node& node = graph.nodes[0];
    ASSERT_EQ(node.op_type, "Resize");
    bool saw_antialias = false;
    bool saw_exclude_outside = false;
    for (const nert::Attribute& attr : node.attributes) {
        if (attr.name == "antialias") {
            saw_antialias = true;
            const std::int64_t* value = std::get_if<std::int64_t>(&attr.value);
            ASSERT_NE(value, nullptr);
            EXPECT_EQ(*value, 0);
        } else if (attr.name == "exclude_outside") {
            saw_exclude_outside = true;
            const std::int64_t* value = std::get_if<std::int64_t>(&attr.value);
            ASSERT_NE(value, nullptr);
            EXPECT_EQ(*value, 0);
        }
    }
    EXPECT_TRUE(saw_antialias);
    EXPECT_TRUE(saw_exclude_outside);
}

// Hermetic positive: Slice takes no attributes (starts/ends/axes as inputs);
// TopK takes axis/largest/sorted with K as an Int64 constant input.
TEST_F(EngineLoader, HermeticSliceTopKLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("starts", {1}, {0}));
    put_sub(body, 5, encode_initializer_int64("ends", {1}, {5}));
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
    put_sub(body, 5, encode_initializer_int64("k", {1}, {3}));
    put_sub(body, 1, encode_node("sl0", "Slice",
                      {"x", "starts", "ends", "axes"}, {"s"}, {}));
    put_sub(body, 1, encode_node("tk0", "TopK", {"s", "k"}, {"v", "i"},
                      {encode_attribute_int("axis", -1),
                          encode_attribute_int("largest", 1),
                          encode_attribute_int("sorted", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 10}));
    put_sub(body, 12, encode_value_info("v", 1, {1, 3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("slice_topk.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Slice");
    EXPECT_TRUE(graph.nodes[0].attributes.empty());
    EXPECT_EQ(graph.nodes[1].op_type, "TopK");
    ASSERT_EQ(graph.nodes[1].attributes.size(), 3U);
}

// Hermetic positive: Cast `to = 6` (INT32) loads; it is emulated on int64
// lanes so the graph dtype is Int64 (see the Cast kernel doc).
TEST_F(EngineLoader, HermeticCastToInt32Loads)
{
    std::string body;
    put_sub(body, 1, encode_node("cast_i32", "Cast", {"x"}, {"c"},
                      {encode_attribute_int("to", 6)}));
    put_sub(body, 1, encode_node("cast_f", "Cast", {"c"}, {"y"},
                      {encode_attribute_int("to", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("cast_int32.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[0].attributes.front().value),
        6);
}

// Hermetic positive: Cast `to` may also be BOOL (9).
TEST_F(EngineLoader, HermeticCastToBoolLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("cast_b", "Cast", {"x"}, {"b"},
                      {encode_attribute_int("to", 9)}));
    put_sub(body, 1, encode_node("cast_f", "Cast", {"b"}, {"y"},
                      {encode_attribute_int("to", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("cast_bool.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[0].attributes.front().value),
        9);
}

// Hermetic positive: an intermediate value_info entry may record a zero dim
// (an empty tensor, such as the leading slice of a shape vector); graph
// inputs and outputs still reject it (ZeroDimensionRejected).
TEST_F(EngineLoader, ValueInfoZeroDimensionLoads)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));
    put_sub(body, 13, encode_value_info("empty", 7 /*INT64*/, {0}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("vi_zero_dim.onnx", encode_model(body)));
    const auto it = graph.tensors.find("empty");
    ASSERT_NE(it, graph.tensors.end());
    EXPECT_EQ(it->second.dims, (std::vector<std::int64_t>{0}));
    EXPECT_EQ(it->second.dtype, nert::DataType::Int64);
}

// Hermetic positive: ConstantOfShape carries an INT64 tensor `value`; Shape,
// Equal, and Where take no attributes.
TEST_F(EngineLoader, HermeticConstantOfShapeEqualWhereShapeLoads)
{
    const std::vector<int64_t> fill{1};
    std::string body;
    put_sub(body, 5, encode_initializer_int64("cshape", {1}, {3}));
    put_sub(body, 1, encode_node("cos0", "ConstantOfShape", {"cshape"},
                      {"fill"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(7 /*INT64*/, {1},
                              raw_bytes_int64(fill)))}));
    put_sub(body, 1, encode_node("sh0", "Shape", {"x"}, {"r"}, {}));
    put_sub(body, 1, encode_node("eq0", "Equal", {"x", "y"}, {"e"}, {}));
    put_sub(body, 1, encode_node("w0", "Where", {"e", "x", "y"}, {"z"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 11, encode_value_info("y", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("z", 1, {2, 3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("cos_eq_where.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 4U);
    EXPECT_EQ(graph.nodes[0].op_type, "ConstantOfShape");
    ASSERT_EQ(graph.nodes[0].attributes.size(), 1U);
    EXPECT_EQ(std::get<std::int64_t>(graph.nodes[0].attributes.front().value),
        1);
    EXPECT_EQ(graph.nodes[1].op_type, "Shape");
    EXPECT_EQ(graph.nodes[2].op_type, "Equal");
    EXPECT_EQ(graph.nodes[3].op_type, "Where");
}

// Hermetic positive: ReduceMax carries keepdims with axes as input; Mod
// carries fmod=0.
TEST_F(EngineLoader, HermeticReduceMaxModLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {-1}));
    put_sub(body, 1, encode_node("rm0", "ReduceMax", {"x", "axes"}, {"r"},
                      {encode_attribute_int("keepdims", 0)}));
    put_sub(body, 1, encode_node("mod0", "Mod", {"x", "x"}, {"y"},
                      {encode_attribute_int("fmod", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("rm_mod.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "ReduceMax");
    EXPECT_EQ(graph.nodes[1].op_type, "Mod");
}

// Hermetic positive: Conv admits kernel_shape (carried by the detection
// fixture on every Conv); shape inference derives dims from the weight.
TEST_F(EngineLoader, HermeticConvKernelShapeAdmitted)
{
    std::string body;
    put_sub(body, 5, encode_initializer("w", {4, 3, 3, 3},
                       std::vector<float>(4 * 3 * 3 * 3, 0.0F)));
    put_sub(body, 1, encode_node("conv0", "Conv", {"x", "w"}, {"y"},
                      {encode_attribute_ints("kernel_shape", {3, 3}),
                          encode_attribute_ints("strides", {1, 1}),
                          encode_attribute_ints("pads", {1, 1, 1, 1})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 4, 8, 8}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("conv_ks.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Conv");
}

// Declared opsets 11..18 load directly. The graph stays in the opset-18 form,
// so a model that only uses stable operators needs no rewrite.
TEST_F(EngineLoader, HermeticOpsetRangeElevenToEighteen)
{
    for (const int64_t opset : { int64_t{11}, int64_t{12}, int64_t{13},
             int64_t{14}, int64_t{16}, int64_t{17}, int64_t{18} }) {
        std::string body;
        put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 2}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 2}));

        const nert::Graph graph = nert::LoadGraphFromFile(write_model_bytes(
            "opset_" + std::to_string(opset) + ".onnx",
            encode_model(body, opset)));
        ASSERT_EQ(graph.nodes.size(), 1U) << "opset " << opset;
        EXPECT_EQ(graph.nodes.front().op_type, "Relu");
    }
}

// Opset-13 ReduceMean/ReduceL2/ReduceMax keep axes in an attribute. The loader
// leaves the source form in place: one input, the `axes` attribute, and no
// synthesized axes initializer.
TEST_F(EngineLoader, HermeticReduceAxesAttributePreserved)
{
    for (const std::string& op :
        { std::string("ReduceMean"), std::string("ReduceL2"),
            std::string("ReduceMax") }) {
        std::string body;
        put_sub(body, 1, encode_node("reduce0", op, {"x"}, {"y"},
                          {encode_attribute_ints("axes", {1})}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

        const nert::Graph graph = nert::LoadGraphFromFile(write_model_bytes(
            op + "_opset13.onnx", encode_model(body, 13)));
        ASSERT_EQ(graph.nodes.size(), 1U) << op;
        const nert::Node& node = graph.nodes.front();
        EXPECT_EQ(node.op_type, op);
        EXPECT_EQ(node.schema_version, 13) << op;
        ASSERT_EQ(node.inputs.size(), 1U) << op;
        EXPECT_EQ(node.inputs[0], "x");

        const nert::Attribute* axes = nullptr;
        for (const nert::Attribute& attribute : node.attributes) {
            if (attribute.name == "axes") {
                axes = &attribute;
            }
        }
        ASSERT_NE(axes, nullptr) << op;
        EXPECT_EQ(std::get<std::vector<int64_t>>(axes->value),
            (std::vector<int64_t>{1}))
            << op;
        EXPECT_EQ(node.attributes.size(), 1U) << op;

        // No synthesized axes initializer or tensor name exists.
        EXPECT_EQ(graph.initializers.count("reduce0/axes"), 0U) << op;
        EXPECT_EQ(graph.initializers.count("reduce0/axes_1"), 0U) << op;
        EXPECT_EQ(graph.tensors.count("reduce0/axes"), 0U) << op;
        EXPECT_EQ(graph.tensors.count("reduce0/axes_1"), 0U) << op;
    }
}

// The old attribute form no longer allocates a synthesized axes name, so a
// tensor that happens to use that name is left untouched and does not affect
// loading or the node's inputs.
TEST_F(EngineLoader, HermeticReduceAxesNameCollision)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("reduce0/axes", {1}, {0}));
    put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {1})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const nert::Graph first = nert::LoadGraphFromFile(
        write_model_bytes("reduce_name_collision.onnx", encode_model(body, 13)));
    ASSERT_EQ(first.nodes.front().inputs.size(), 1U);
    EXPECT_EQ(first.nodes.front().inputs[0], "x");
    EXPECT_EQ(first.initializers.count("reduce0/axes"), 1U);
    EXPECT_EQ(std::get<std::vector<int64_t>>(
                  first.initializers.at("reduce0/axes").values),
        (std::vector<int64_t>{0}));

    const nert::Graph second = nert::LoadGraphFromFile(
        write_model_bytes("reduce_name_collision.onnx", encode_model(body, 13)));
    EXPECT_EQ(second.nodes.front().inputs.size(), 1U);
    EXPECT_EQ(second.initializers.count("reduce0/axes"), 1U);
}

// An empty axes attribute and invalid axes keep their normal reduce semantics:
// empty means every axis; duplicates and out-of-range values are rejected by
// shape inference while the attribute is still in place.
TEST_F(EngineLoader, HermeticReduceAxesEmptyAndInvalid)
{
    auto load = [](const std::vector<int64_t>& axes, const std::vector<int64_t>& out_dims) {
        std::string body;
        put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x"}, {"y"},
                          {encode_attribute_ints("axes", axes)}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
        put_sub(body, 12, encode_value_info("y", 1, out_dims));
        return nert::LoadGraphFromFile(write_model_bytes("reduce_axes_edge.onnx", encode_model(body, 13)));
    };

    const nert::Graph empty = load({}, {1, 1, 1});
    nert::Model empty_model(empty, {{"x", {2, 3, 4}}}, nert::CpuDevice());
    std::vector<float> values;
    for (int i = 0; i < 24; ++i) {
        values.push_back(static_cast<float>(i));
    }
    const nert::InferenceResult result = empty_model.Run({{"x", values}});
    ASSERT_EQ(result.outputs.at("y").size(), 1U);
    EXPECT_NEAR(result.outputs.at("y").front(), 11.5F, 1e-6F);

    const nert::Graph duplicate = load({1, 1}, {2, 1, 4});
    EXPECT_THROW(nert::Model(duplicate, {{"x", {2, 3, 4}}}, nert::CpuDevice()), nert::ModelLoadException);

    const nert::Graph out_of_range = load({3}, {2, 1, 4});
    EXPECT_THROW(nert::Model(out_of_range, {{"x", {2, 3, 4}}}, nert::CpuDevice()), nert::ModelLoadException);
}

// The opset-13 graph executes exactly like the opset-18 form.
TEST_F(EngineLoader, HermeticOpset13ReduceMeanMatchesOpset18)
{
    std::vector<float> values;
    for (int i = 0; i < 24; ++i) {
        values.push_back(static_cast<float>(i));
    }

    auto load = [](int64_t opset) {
        std::string body;
        if (opset < 18) {
            put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x"}, {"y"},
                              {encode_attribute_ints("axes", {1})}));
        } else {
            put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
            put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x", "axes"},
                              {"y"}, {}));
        }
        put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));
        return nert::LoadGraphFromFile(write_model_bytes(
            "reduce_run_" + std::to_string(opset) + ".onnx",
            encode_model(body, opset)));
    };

    const nert::Graph graph13 = load(13);
    const nert::Graph graph18 = load(18);
    nert::Model model13(graph13, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
    nert::Model model18(graph18, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());

    const nert::InferenceResult result13 = model13.Run({ { "x", values } });
    const nert::InferenceResult result18 = model18.Run({ { "x", values } });
    const std::vector<float>& out13 = result13.outputs.at("y");
    const std::vector<float>& out18 = result18.outputs.at("y");
    ASSERT_EQ(out13.size(), 8U);
    ASSERT_EQ(out18.size(), out13.size());
    for (size_t i = 0; i < out13.size(); ++i) {
        EXPECT_NEAR(out13[i], out18[i], 1e-6F) << "element " << i;
    }
}

// ScatterND's explicit default reduction ("none") is legal from opset 16 and
// stays in the graph; the kernels implement the overwrite form directly.
TEST_F(EngineLoader, HermeticScatterNDReductionNonePreserved)
{
    for (const int64_t opset : { int64_t{16}, int64_t{18} }) {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("idx", {1, 1}, {0}));
        put_sub(body, 5, encode_initializer("upd", {1, 3}, {1.0F, 2.0F, 3.0F}));
        put_sub(body, 1, encode_node("scatter0", "ScatterND",
                          {"x", "idx", "upd"}, {"y"},
                          {encode_attribute_string("reduction", "none")}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

        const nert::Graph graph = nert::LoadGraphFromFile(write_model_bytes(
            "scatter_none_" + std::to_string(opset) + ".onnx",
            encode_model(body, opset)));
        ASSERT_EQ(graph.nodes.size(), 1U) << "opset " << opset;
        const nert::Node& node = graph.nodes.front();
        EXPECT_EQ(node.inputs.size(), 3U) << "opset " << opset;
        ASSERT_EQ(node.attributes.size(), 1U) << "opset " << opset;
        EXPECT_EQ(node.attributes.front().name, "reduction") << "opset " << opset;
        EXPECT_EQ(std::get<std::string>(node.attributes.front().value), "none")
            << "opset " << opset;
    }
}

// Legacy Softmax keeps the source node in the graph; the kernel implements
// the pre-13 2-D coercion directly. The default axis is 1.
TEST_F(EngineLoader, HermeticLegacySoftmaxDirect)
{
    std::vector<float> values;
    for (int i = 0; i < 24; ++i) {
        values.push_back(0.05F * static_cast<float>(i));
    }

    std::string body;
    put_sub(body, 1, encode_node("sm0", "Softmax", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("softmax_default_11.onnx", encode_model(body, 11)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Softmax");
    EXPECT_EQ(graph.nodes.front().schema_version, 11);

    nert::Model model(graph, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult result = model.Run({ { "x", values } });
    const std::vector<float> expected = row_softmax(values, 12);
    ASSERT_EQ(result.outputs.at("y").size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result.outputs.at("y")[i], expected[i], 1e-6F) << "element " << i;
    }

    // A negative axis is resolved against the input rank; legacy Softmax
    // then normalizes the trailing block from that axis.
    std::string negative;
    put_sub(negative, 1, encode_node("sm_neg", "Softmax", {"x"}, {"y"},
                           {encode_attribute_int("axis", -1)}));
    put_sub(negative, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(negative, 12, encode_value_info("y", 1, {2, 3, 4}));
    const nert::Graph negative_graph = nert::LoadGraphFromFile(
        write_model_bytes("softmax_neg_11.onnx", encode_model(negative, 11)));
    ASSERT_EQ(negative_graph.nodes.size(), 1U);

    nert::Model negative_model(negative_graph, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult negative_result = negative_model.Run({ { "x", values } });
    const std::vector<float> negative_expected = row_softmax(values, 4);
    ASSERT_EQ(negative_result.outputs.at("y").size(), negative_expected.size());
    for (size_t i = 0; i < negative_expected.size(); ++i) {
        EXPECT_NEAR(negative_result.outputs.at("y")[i], negative_expected[i], 1e-6F) << "element " << i;
    }
}

// Legacy Softmax on an intermediate input does not need value_info for that
// input: shape inference derives the rank from the producer.
TEST_F(EngineLoader, HermeticLegacySoftmaxIntermediateInput)
{
    std::vector<float> values;
    for (int i = 0; i < 24; ++i) {
        values.push_back(0.1F * static_cast<float>(i) - 1.0F);
    }
    std::vector<float> relu = values;
    for (float& value : relu) {
        value = value > 0.0F ? value : 0.0F;
    }

    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"h"}, {}));
    put_sub(body, 1, encode_node("sm0", "Softmax", {"h"}, {"y"},
                      {encode_attribute_int("axis", -1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("softmax_intermediate_11.onnx", encode_model(body, 11)));
    ASSERT_EQ(graph.nodes.size(), 2U);
    EXPECT_EQ(graph.nodes[0].op_type, "Relu");
    EXPECT_EQ(graph.nodes[1].op_type, "Softmax");

    nert::Model model(graph, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult result = model.Run({ { "x", values } });
    const std::vector<float> expected = row_softmax(relu, 4);
    ASSERT_EQ(result.outputs.at("y").size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result.outputs.at("y")[i], expected[i], 1e-6F) << "element " << i;
    }
}

// Opset-13 Softmax already has the canonical axis semantics; it must not be
// expanded.
TEST_F(EngineLoader, HermeticSoftmaxAtOpset13Unchanged)
{
    std::string body;
    put_sub(body, 1, encode_node("sm0", "Softmax", {"x"}, {"y"},
                      {encode_attribute_int("axis", -1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("softmax_13.onnx", encode_model(body, 13)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Softmax");
}

// ReduceSum's pre-13 axes attribute stays in the graph; shape inference and
// the kernel read it in place. An empty list still means reduce every axis.
TEST_F(EngineLoader, HermeticLegacyReduceSumAxesPreserved)
{
    std::vector<float> values;
    for (int i = 1; i <= 24; ++i) {
        values.push_back(static_cast<float>(i));
    }

    std::string body;
    put_sub(body, 1, encode_node("rs0", "ReduceSum", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {1}),
                          encode_attribute_int("keepdims", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("reduce_sum_axes_11.onnx", encode_model(body, 11)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    const nert::Node& node = graph.nodes.front();
    ASSERT_EQ(node.inputs.size(), 1U);
    EXPECT_EQ(node.schema_version, 11);
    ASSERT_EQ(node.attributes.size(), 2U);
    EXPECT_EQ(graph.initializers.count("rs0/axes"), 0U);

    nert::Model model(graph, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult result = model.Run({ { "x", values } });
    std::vector<float> expected;
    for (int b = 0; b < 2; ++b) {
        for (int c = 0; c < 4; ++c) {
            float sum = 0.0F;
            for (int j = 0; j < 3; ++j) {
                sum += values[static_cast<size_t>((b * 3 + j) * 4 + c)];
            }
            expected.push_back(sum);
        }
    }
    ASSERT_EQ(result.outputs.at("y").size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result.outputs.at("y")[i], expected[i], 1e-5F) << "element " << i;
    }

    std::string empty;
    put_sub(empty, 1, encode_node("rs_empty", "ReduceSum", {"x"}, {"y"},
                        {encode_attribute_ints("axes", {})}));
    put_sub(empty, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(empty, 12, encode_value_info("y", 1, {1, 1, 1}));
    const nert::Graph empty_graph = nert::LoadGraphFromFile(
        write_model_bytes("reduce_sum_empty_11.onnx", encode_model(empty, 11)));
    ASSERT_EQ(empty_graph.nodes.size(), 1U);
    EXPECT_EQ(empty_graph.nodes.front().inputs.size(), 1U);
    ASSERT_EQ(empty_graph.nodes.front().attributes.size(), 1U);
    EXPECT_EQ(empty_graph.nodes.front().attributes.front().name, "axes");

    nert::Model empty_model(empty_graph, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult empty_result = empty_model.Run({ { "x", values } });
    ASSERT_EQ(empty_result.outputs.at("y").size(), 1U);
    EXPECT_NEAR(empty_result.outputs.at("y").front(), 300.0F, 1e-5F);
}

// Squeeze's pre-13 axes attribute stays in the graph; shape inference reads
// it in place. An empty list means squeeze every size-1 axis.
TEST_F(EngineLoader, HermeticLegacySqueezePreserved)
{
    std::vector<float> values;
    for (int i = 0; i < 8; ++i) {
        values.push_back(static_cast<float>(i));
    }

    std::string body;
    put_sub(body, 1, encode_node("sq0", "Squeeze", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {1})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 1, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("squeeze_axes_11.onnx", encode_model(body, 11)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    ASSERT_EQ(graph.nodes.front().inputs.size(), 1U);
    EXPECT_EQ(graph.nodes.front().schema_version, 11);
    ASSERT_EQ(graph.nodes.front().attributes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().attributes.front().name, "axes");
    EXPECT_EQ(graph.initializers.count("sq0/axes"), 0U);

    nert::Model model(graph, { { "x", { 2, 1, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult result = model.Run({ { "x", values } });
    ASSERT_EQ(result.outputs.at("y").size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(result.outputs.at("y")[i], values[i]);
    }

    std::string empty;
    put_sub(empty, 1, encode_node("sq_empty", "Squeeze", {"x"}, {"y"},
                        {encode_attribute_ints("axes", {})}));
    put_sub(empty, 11, encode_value_info("x", 1, {2, 1, 4}));
    put_sub(empty, 12, encode_value_info("y", 1, {2, 4}));
    const nert::Graph empty_graph = nert::LoadGraphFromFile(
        write_model_bytes("squeeze_empty_11.onnx", encode_model(empty, 11)));
    ASSERT_EQ(empty_graph.nodes.size(), 1U);
    EXPECT_EQ(empty_graph.nodes.front().inputs.size(), 1U);
    ASSERT_EQ(empty_graph.nodes.front().attributes.size(), 1U);
    EXPECT_EQ(empty_graph.nodes.front().attributes.front().name, "axes");
}

// Unsqueeze's required pre-13 axes attribute stays in the graph.
TEST_F(EngineLoader, HermeticLegacyUnsqueezePreserved)
{
    std::vector<float> values;
    for (int i = 0; i < 6; ++i) {
        values.push_back(static_cast<float>(i));
    }

    std::string body;
    put_sub(body, 1, encode_node("us0", "Unsqueeze", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {0})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("unsqueeze_axes_11.onnx", encode_model(body, 11)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    ASSERT_EQ(graph.nodes.front().inputs.size(), 1U);
    EXPECT_EQ(graph.nodes.front().schema_version, 11);
    ASSERT_EQ(graph.nodes.front().attributes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().attributes.front().name, "axes");
    EXPECT_EQ(graph.initializers.count("us0/axes"), 0U);

    nert::Model model(graph, { { "x", { 2, 3 } } }, nert::CpuDevice());
    const nert::InferenceResult result = model.Run({ { "x", values } });
    ASSERT_EQ(result.outputs.at("y").size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(result.outputs.at("y")[i], values[i]);
    }
}

// Split's pre-13 `split` attribute stays in the graph and is read in place.
TEST_F(EngineLoader, HermeticLegacySplitPreserved)
{
    std::vector<float> values;
    for (int i = 0; i < 24; ++i) {
        values.push_back(static_cast<float>(i));
    }

    std::string body;
    put_sub(body, 1, encode_node("sp0", "Split", {"x"}, {"a", "b"},
                      {encode_attribute_int("axis", 1),
                          encode_attribute_ints("split", {1, 2})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("a", 1, {2, 1, 4}));
    put_sub(body, 12, encode_value_info("b", 1, {2, 2, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("split_attr_11.onnx", encode_model(body, 11)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    ASSERT_EQ(graph.nodes.front().inputs.size(), 1U);
    EXPECT_EQ(graph.nodes.front().schema_version, 11);
    ASSERT_EQ(graph.nodes.front().attributes.size(), 2U);
    EXPECT_EQ(graph.initializers.count("sp0/split"), 0U);

    nert::Model model(graph, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult result = model.Run({ { "x", values } });
    std::vector<float> a;
    std::vector<float> b;
    for (int batch = 0; batch < 2; ++batch) {
        for (int c = 0; c < 4; ++c) {
            a.push_back(values[static_cast<size_t>((batch * 3 + 0) * 4 + c)]);
        }
        for (int part = 1; part < 3; ++part) {
            for (int c = 0; c < 4; ++c) {
                b.push_back(values[static_cast<size_t>((batch * 3 + part) * 4 + c)]);
            }
        }
    }
    ASSERT_EQ(result.outputs.at("a").size(), a.size());
    ASSERT_EQ(result.outputs.at("b").size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(result.outputs.at("a")[i], a[i]);
    }
    for (size_t i = 0; i < b.size(); ++i) {
        EXPECT_EQ(result.outputs.at("b")[i], b[i]);
    }
}

// The opset-11 Resize input layout is the same as 13+: X, roi, scales, sizes.
// NERT already accepts empty roi/scales and a sizes input, so no rewrite is
// added; this positive row proves the old declaration loads and runs.
TEST_F(EngineLoader, HermeticOpset11ResizeWithSizesLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer_empty_float("roi"));
    put_sub(body, 5, encode_initializer_empty_float("scales"));
    put_sub(body, 5, encode_initializer_int64("sizes", {4}, {1, 3, 8, 8}));
    put_sub(body, 1, encode_node("resize0", "Resize",
                      {"x", "roi", "scales", "sizes"}, {"y"},
                      {encode_attribute_string("mode", "nearest"),
                          encode_attribute_string("coordinate_transformation_mode", "asymmetric")}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 8, 8}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("resize_sizes_11.onnx", encode_model(body, 11)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Resize");
    EXPECT_EQ(graph.nodes.front().inputs.size(), 4U);

    std::vector<float> values;
    for (int i = 0; i < 48; ++i) {
        values.push_back(static_cast<float>(i));
    }
    nert::Model model(graph, { { "x", { 1, 3, 4, 4 } } }, nert::CpuDevice());
    const nert::InferenceResult result = model.Run({ { "x", values } });
    ASSERT_EQ(result.outputs.at("y").size(), 1U * 3 * 8 * 8);
    // asymmetric + nearest/round_prefer_floor maps output (2, 1) to input (1, 0).
    EXPECT_EQ(result.outputs.at("y")[static_cast<size_t>((0 * 8 + 2) * 8 + 1)], values[4]);
}

// Declarations above 18 are accepted when each resolved schema version is
// executable by the opset-18 kernels. Conv/Relu/Flatten/MatMul/Add all have
// stable or type-only schemas after 18.
TEST_F(EngineLoader, HermeticNewerStableOperators)
{
    auto load = [](int64_t opset) {
        std::string body;
        std::vector<float> conv_w(4 * 3 * 3 * 3);
        for (size_t i = 0; i < conv_w.size(); ++i) {
            conv_w[i] = 0.01F * static_cast<float>(static_cast<int>(i % 7) - 3);
        }
        std::vector<float> conv_b(4);
        for (size_t i = 0; i < conv_b.size(); ++i) {
            conv_b[i] = 0.1F * static_cast<float>(i);
        }
        std::vector<float> mm_w(256 * 2);
        for (size_t i = 0; i < mm_w.size(); ++i) {
            mm_w[i] = 0.01F * static_cast<float>(static_cast<int>(i % 5) - 2);
        }
        const std::vector<float> mm_b = {0.5F, -0.5F};

        put_sub(body, 5, encode_initializer("w", {4, 3, 3, 3}, conv_w));
        put_sub(body, 5, encode_initializer("b", {4}, conv_b));
        put_sub(body, 5, encode_initializer("w2", {256, 2}, mm_w));
        put_sub(body, 5, encode_initializer("b2", {2}, mm_b));
        put_sub(body, 1, encode_node("conv0", "Conv", {"x", "w", "b"}, {"c"},
                          {encode_attribute_ints("pads", {1, 1, 1, 1}),
                              encode_attribute_ints("strides", {1, 1})}));
        put_sub(body, 1, encode_node("relu0", "Relu", {"c"}, {"r"}, {}));
        put_sub(body, 1, encode_node("flat0", "Flatten", {"r"}, {"f"}, {}));
        put_sub(body, 1, encode_node("mm0", "MatMul", {"f", "w2"}, {"m"}, {}));
        put_sub(body, 1, encode_node("add0", "Add", {"m", "b2"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
        put_sub(body, 12, encode_value_info("y", 1, {1, 2}));
        return nert::LoadGraphFromFile(write_model_bytes(
            "newer_stable_" + std::to_string(opset) + ".onnx",
            encode_model(body, opset)));
    };

    std::vector<float> input;
    for (int i = 0; i < 192; ++i) {
        input.push_back(0.01F * static_cast<float>(static_cast<int>(i % 19) - 9));
    }

    const nert::Graph baseline = load(18);
    nert::Model baseline_model(baseline, { { "x", { 1, 3, 8, 8 } } }, nert::CpuDevice());
    const std::vector<float> expected =
        baseline_model.Run({ { "x", input } }).outputs.at("y");

    for (const int64_t opset : { int64_t{19}, int64_t{21}, int64_t{27} }) {
        const nert::Graph graph = load(opset);
        nert::Model model(graph, { { "x", { 1, 3, 8, 8 } } }, nert::CpuDevice());
        const std::vector<float> actual = model.Run({ { "x", input } }).outputs.at("y");
        ASSERT_EQ(actual.size(), expected.size()) << "opset " << opset;
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], 1e-6F)
                << "opset " << opset << " element " << i;
        }
    }
}

// A plain Cast carries only `to`; newer schema versions added optional
// attributes but the captured behavior is unchanged.
TEST_F(EngineLoader, HermeticNewerPlainCastAccepted)
{
    for (const int64_t opset : { int64_t{19}, int64_t{27} }) {
        std::string body;
        put_sub(body, 1, encode_node("cast0", "Cast", {"x"}, {"y"},
                          {encode_attribute_int("to", 7)}));
        put_sub(body, 11, encode_value_info("x", 1, {2}));
        put_sub(body, 12, encode_value_info("y", 7, {2}));

        const nert::Graph graph = nert::LoadGraphFromFile(
            write_model_bytes("newer_cast_" + std::to_string(opset) + ".onnx",
                encode_model(body, opset)));
        ASSERT_EQ(graph.nodes.size(), 1U) << "opset " << opset;
        EXPECT_EQ(graph.nodes.front().op_type, "Cast");
        ASSERT_EQ(graph.nodes.front().attributes.size(), 1U);
        EXPECT_EQ(graph.nodes.front().attributes.front().name, "to");
    }
}

// Pad's pre-19 modes remain executable after the schema adds `wrap`.
TEST_F(EngineLoader, HermeticNewerPadModesAccepted)
{
    for (const std::string& mode : { std::string("constant"),
             std::string("reflect"), std::string("edge") }) {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("pads", {4}, {0, 1, 0, 1}));
        put_sub(body, 1, encode_node("pad0", "Pad", {"x", "pads"}, {"y"},
                          {encode_attribute_string("mode", mode)}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 2}));
        put_sub(body, 12, encode_value_info("y", 1, {1, 4}));

        const nert::Graph graph = nert::LoadGraphFromFile(
            write_model_bytes("newer_pad_" + mode + ".onnx",
                encode_model(body, 19)));
        ASSERT_EQ(graph.nodes.size(), 1U) << mode;
        EXPECT_EQ(graph.nodes.front().op_type, "Pad");
    }
}

// AveragePool's pre-19 form has no `dilations`; that additive attribute is
// guarded, so the baseline form still loads.
TEST_F(EngineLoader, HermeticNewerAveragePoolWithoutDilationsAccepted)
{
    std::string body;
    put_sub(body, 1, encode_node("pool0", "AveragePool", {"x"}, {"y"},
                      {encode_attribute_ints("kernel_shape", {2, 2})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 2, 2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("newer_averagepool.onnx", encode_model(body, 19)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "AveragePool");
}

// A plain Range at 27 resolves to schema 27 with the additive `stash_type`
// absent; it is executable.
TEST_F(EngineLoader, HermeticNewerPlainRangeAccepted)
{
    std::string body;
    put_sub(body, 5, encode_initializer("start", {}, {0.0F}));
    put_sub(body, 5, encode_initializer("limit", {}, {3.0F}));
    put_sub(body, 5, encode_initializer("delta", {}, {1.0F}));
    put_sub(body, 1, encode_node("range0", "Range", {"start", "limit", "delta"},
                      {"y"}, {}));
    put_sub(body, 12, encode_value_info("y", 1, {3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("newer_range.onnx", encode_model(body, 27)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Range");
}

// GridSample at declared 19 resolves to schema 16, before the mode change.
TEST_F(EngineLoader, HermeticGridSampleAt19Accepted)
{
    std::string body;
    put_sub(body, 1, encode_node("grid0", "GridSample", {"x", "grid"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 2, 2}));
    put_sub(body, 11, encode_value_info("grid", 1, {1, 2, 2, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 2, 2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("gridsample_19.onnx", encode_model(body, 19)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "GridSample");
}

// ===========================================================================
// Phase A — per-node effective schema metadata.
// ===========================================================================

// Every source node carries the effective ONNX schema selected by the
// declaration: ReduceMean history is 1, 11, 13, 18. The graph records the
// declaration itself.
TEST_F(EngineLoader, HermeticNodeSchemaVersionTracksDeclaredOpset)
{
    auto load = [](int64_t declared) {
        std::string body;
        if (declared < 18) {
            put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x"}, {"y"},
                              {encode_attribute_ints("axes", {1})}));
        } else {
            put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
            put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x", "axes"},
                              {"y"}, {}));
        }
        put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));
        return nert::LoadGraphFromFile(write_model_bytes(
            "schema_meta_" + std::to_string(declared) + ".onnx",
            encode_model(body, declared)));
    };

    const std::vector<std::pair<int64_t, int64_t>> cases = {
        {11, 11}, {13, 13}, {18, 18}, {19, 18}, {27, 18}};
    for (const auto& entry : cases) {
        const nert::Graph graph = load(entry.first);
        EXPECT_EQ(graph.declared_opset, entry.first)
            << "declared " << entry.first;
        ASSERT_EQ(graph.nodes.size(), 1U) << "declared " << entry.first;
        EXPECT_EQ(graph.nodes.front().op_type, "ReduceMean");
        EXPECT_EQ(graph.nodes.front().schema_version, entry.second)
            << "declared " << entry.first;
        // The source form stays in the graph. Declarations before schema 18
        // keep the axes attribute and one input; declarations at or above 18
        // keep the axes input and no attribute.
        if (entry.second < 18) {
            EXPECT_EQ(graph.nodes.front().inputs.size(), 1U)
                << "declared " << entry.first;
            EXPECT_EQ(graph.nodes.front().attributes.size(), 1U)
                << "declared " << entry.first;
            EXPECT_EQ(graph.nodes.front().attributes.front().name, "axes")
                << "declared " << entry.first;
        } else {
            EXPECT_EQ(graph.nodes.front().inputs.size(), 2U)
                << "declared " << entry.first;
            EXPECT_TRUE(graph.nodes.front().attributes.empty())
                << "declared " << entry.first;
        }
    }
}

// A declared-11 Relu resolves to schema 6: the full-history catalog must
// select a version below the old opset-18 baseline.
TEST_F(EngineLoader, HermeticEffectiveSchemaBelowEleven)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("schema_relu_11.onnx", encode_model(body, 11)));
    EXPECT_EQ(graph.declared_opset, 11);
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Relu");
    EXPECT_EQ(graph.nodes.front().schema_version, 6);
}

// Add's effective schema at declared 27 is 14, not 27.
TEST_F(EngineLoader, HermeticNewerAddResolvesSchemaFourteen)
{
    std::string body;
    put_sub(body, 5, encode_initializer("b", {2}, {1.0F, 2.0F}));
    put_sub(body, 1, encode_node("add0", "Add", {"x", "b"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("schema_add_27.onnx", encode_model(body, 27)));
    EXPECT_EQ(graph.declared_opset, 27);
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Add");
    EXPECT_EQ(graph.nodes.front().schema_version, 14);
}

// Legacy Softmax stays a single source node with its schema metadata.
TEST_F(EngineLoader, HermeticLegacySoftmaxKeepsMetadata)
{
    std::string body;
    put_sub(body, 1, encode_node("sm0", "Softmax", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("schema_softmax_11.onnx", encode_model(body, 11)));
    EXPECT_EQ(graph.declared_opset, 11);
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "Softmax");
    EXPECT_EQ(graph.nodes.front().schema_version, 11);
}

// ===========================================================================
// EngineLoaderNegative.* — rejections whose messages must name the node and
// op type.
// ===========================================================================

class EngineLoaderNegative : public ::testing::Test {
};

// Unknown op: message must name the node AND the op type.

TEST_F(EngineLoaderNegative, UnknownOpNamesNodeAndOpType)
{
    std::string body;
    put_sub(body, 1, encode_node("bogus_node", "DepthWiseConv",
        {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 4}));

    const std::string message =
        load_error_message(encode_model(body), "unknown_op.onnx");
    EXPECT_NE(message.find("bogus_node"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("DepthWiseConv"), std::string::npos)
        << "message: " << message;
}

// Unsupported attribute: message must name the node AND the op type.

TEST_F(EngineLoaderNegative, UnsupportedAttributeNamesNodeAndOpType)
{
    // Relu accepts no attributes: any attribute is rejected with full
    // node/op context.
    std::string body;
    put_sub(body, 1, encode_node("relu_foo", "Relu", {"x"}, {"y"},
        {encode_attribute_name_only("leaky")}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 4}));

    const std::string message =
        load_error_message(encode_model(body), "bad_attr.onnx");
    EXPECT_NE(message.find("relu_foo"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Relu"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("leaky"), std::string::npos) << "message: " << message;
}

// Graph I/O widened (P3e): INT64 and BOOL graph inputs and outputs load with
// their declared dtype (RT-DETR's orig_target_sizes / labels, token masks).
TEST_F(EngineLoader, HermeticInt64AndBoolGraphIoAccepted)
{
    std::string body;
    put_sub(body, 1, encode_node("probe_node", "Identity",
        {"x_int"}, {"y_int"}, {}));
    put_sub(body, 1, encode_node("probe_bool", "Identity",
        {"x_bool"}, {"y_bool"}, {}));
    put_sub(body, 11, encode_value_info("x_int", 7 /*INT64*/, {2, 2}));
    put_sub(body, 11, encode_value_info("x_bool", 9 /*BOOL*/, {2, 2}));
    put_sub(body, 12, encode_value_info("y_int", 7 /*INT64*/, {2, 2}));
    put_sub(body, 12, encode_value_info("y_bool", 9 /*BOOL*/, {2, 2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("int_bool_io.onnx", encode_model(body)));
    ASSERT_EQ(graph.inputs.size(), 2U);
    EXPECT_EQ(graph.inputs[0].first, "x_int");
    EXPECT_EQ(graph.inputs[0].second.dtype, nert::DataType::Int64);
    EXPECT_EQ(graph.inputs[1].first, "x_bool");
    EXPECT_EQ(graph.inputs[1].second.dtype, nert::DataType::Bool);
    ASSERT_EQ(graph.outputs.size(), 2U);
    EXPECT_EQ(graph.outputs[0].first, "y_int");
    EXPECT_EQ(graph.outputs[0].second.dtype, nert::DataType::Int64);
    EXPECT_EQ(graph.outputs[1].first, "y_bool");
    EXPECT_EQ(graph.outputs[1].second.dtype, nert::DataType::Bool);
    EXPECT_EQ(graph.inputs[0].second.dims, (std::vector<int64_t>{2, 2}));
}

// Graph I/O outside FLOAT/INT64/BOOL (here INT32) is rejected by tensor name
// and element type, for inputs and outputs alike.
TEST_F(EngineLoaderNegative, UnsupportedGraphIoTypeRejected)
{
    {
        std::string body;
        put_sub(body, 1, encode_node("probe_node", "Relu",
            {"x_i32"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("x_i32", 6 /*INT32*/, {2, 2}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 2}));

        const std::string message =
            load_error_message(encode_model(body), "i32_in.onnx");
        EXPECT_NE(message.find("graph input"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("x_i32"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("INT32"), std::string::npos)
            << "message: " << message;
    }
    {
        std::string body;
        put_sub(body, 1, encode_node("probe_out", "Relu",
            {"x"}, {"y_i32"}, {}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 2}));
        put_sub(body, 12, encode_value_info("y_i32", 6 /*INT32*/, {2, 2}));

        const std::string message =
            load_error_message(encode_model(body), "i32_out.onnx");
        EXPECT_NE(message.find("graph output"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("y_i32"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("INT32"), std::string::npos)
            << "message: " << message;
    }
}

// A Constant of a dtype outside FLOAT/DOUBLE/INT32/INT64/BOOL is a load
// rejection naming the node, the op type, and the element type.
TEST_F(EngineLoaderNegative, ConstantOtherDtypeRejected)
{
    for (const auto& [file, elem, tag] :
        {std::tuple<std::string, int64_t, std::string>{
                 "const_u8.onnx", 2, "UINT8"},
            std::tuple<std::string, int64_t, std::string>{
                "const_f16.onnx", 10, "FLOAT16"}}) {
        // UINT8 is one raw byte per element; FLOAT16 is two. The test's point
        // is the dtype rejection, so the payload must have the right length.
        const std::string raw(static_cast<size_t>(elem == 10 ? 8 : 4), '\x01');
        std::string body;
        put_sub(body, 1,
            encode_constant_node("const_bad", "c_bad",
                encode_constant_tensor(elem, {4}, raw)));
        put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("x", 1, {4}));
        put_sub(body, 12, encode_value_info("y", 1, {4}));

        const std::string message =
            load_error_message(encode_model(body), file);
        EXPECT_NE(message.find("const_bad"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("Constant"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find(tag), std::string::npos)
            << "message: " << message;
    }
}

// An initializer dtype outside FLOAT/INT64 is rejected by name and type.

TEST_F(EngineLoaderNegative, NonFloatInitializerRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_float16("half_w", {2}, "\x00\x3c\x00\x40"));
    put_sub(body, 1, encode_node("relu0", "Relu", {"x", "half_w"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "fp16_init.onnx");
    EXPECT_NE(message.find("half_w"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("FLOAT16"), std::string::npos)
        << "message: " << message;
}

// A named (dim_param) or unknown input dimension loads as -1; the caller
// fixes it per load through the shapes handed to InferShapes.

TEST_F(EngineLoader, NamedDimensionLoadsAsSymbolic)
{
    std::string body;
    put_sub(body, 1, encode_node("dynamic_node", "Relu",
        {"x_dyn"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x_dyn", 1, {2, 2}, true));
    put_sub(body, 12, encode_value_info("y", 1, {2, 2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("dyn_in.onnx", encode_model(body)));
    ASSERT_EQ(graph.inputs.size(), 1U);
    EXPECT_EQ(graph.inputs.front().second.dims,
        (std::vector<std::int64_t>{-1, -1}));
}

// An explicit zero dim_value is rejected by name.

TEST_F(EngineLoaderNegative, ZeroDimensionRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x_zero"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x_zero", 1, {0, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "zero_dim.onnx");
    EXPECT_NE(message.find("x_zero"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("non-fixed"), std::string::npos)
        << "message: " << message;
}

// Missing file: the offending path itself appears in the message.

TEST_F(EngineLoaderNegative, MissingModelFileNamesPath)
{
    const std::string path =
        (std::filesystem::temp_directory_path() /
            "engine_loader_tests" / "no_such_model.onnx")
            .string();
    std::string message;
    try {
        nert::LoadGraphFromFile(path);
    } catch (const nert::ModelLoadException& e) {
        message = e.what();
    }
    EXPECT_FALSE(message.empty()) << "missing model file did not throw";
    EXPECT_NE(message.find(path), std::string::npos) << "message: " << message;
}

// Cast `to` outside {FLOAT, INT32, INT64, BOOL} rejects with node + op context.

TEST_F(EngineLoaderNegative, CastToOutsidePolicyRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("cast_bad", "Cast", {"x"}, {"y"},
        {encode_attribute_int("to", 2)})); // UINT8
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const std::string message =
        load_error_message(encode_model(body), "cast_to.onnx");
    EXPECT_NE(message.find("cast_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Cast"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("to"), std::string::npos) << "message: " << message;
}

// Resize outside the admitted policy (cubic mode, tf_crop_and_resize,
// antialiasing) rejects with node + op context.

TEST_F(EngineLoaderNegative, ResizeModeOutsidePolicyRejected)
{
    for (const auto& [file, attr] :
        {std::tuple<std::string, std::string>{"rs_mode.onnx",
             encode_attribute_string("mode", "cubic")},
            std::tuple<std::string, std::string>{"rs_aa.onnx",
                encode_attribute_int("antialias", 1)},
            std::tuple<std::string, std::string>{"rs_ctm.onnx",
                encode_attribute_string(
                    "coordinate_transformation_mode", "tf_crop_and_resize")}}) {
        std::string body;
        put_sub(body, 5,
            encode_initializer("scales", {4}, {1.0F, 1.0F, 2.0F, 2.0F}));
        put_sub(body, 1, encode_node("rs_bad", "Resize", {"x", "", "scales"},
                          {"y"}, {attr}));
        put_sub(body, 11, encode_value_info("x", 1, {1, 1, 4, 4}));
        put_sub(body, 12, encode_value_info("y", 1, {1, 1, 8, 8}));

        const std::string message =
            load_error_message(encode_model(body), file);
        EXPECT_NE(message.find("rs_bad"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("Resize"), std::string::npos)
            << "message: " << message;
    }
}

// Mod with fmod=1 (float remainder) rejects; the fixture pins fmod=0.

TEST_F(EngineLoaderNegative, ModFmodOutsidePolicyRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("mod_bad", "Mod", {"x", "x"}, {"y"},
        {encode_attribute_int("fmod", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {4}));
    put_sub(body, 12, encode_value_info("y", 1, {4}));

    const std::string message =
        load_error_message(encode_model(body), "mod_fmod.onnx");
    EXPECT_NE(message.find("mod_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Mod"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("fmod"), std::string::npos) << "message: " << message;
}

// TopK outside largest=1/sorted=1 rejects with node + op context.

TEST_F(EngineLoaderNegative, TopKPolicyOutsidePolicyRejected)
{
    const std::vector<std::string> largest_zero{encode_attribute_int("axis", -1),
        encode_attribute_int("largest", 0), encode_attribute_int("sorted", 1)};
    const std::vector<std::string> sorted_zero{encode_attribute_int("axis", -1),
        encode_attribute_int("largest", 1), encode_attribute_int("sorted", 0)};
    for (const auto& [file, attrs] :
        {std::tuple<std::string, std::vector<std::string>>{
             "tk_l.onnx", largest_zero},
            std::tuple<std::string, std::vector<std::string>>{
                "tk_s.onnx", sorted_zero}}) {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("k", {1}, {2}));
        put_sub(body, 1,
            encode_node("tk_bad", "TopK", {"x", "k"}, {"v", "i"}, attrs));
        put_sub(body, 11, encode_value_info("x", 1, {2, 5}));
        put_sub(body, 12, encode_value_info("v", 1, {2, 2}));

        const std::string message =
            load_error_message(encode_model(body), file);
        EXPECT_NE(message.find("tk_bad"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("TopK"), std::string::npos)
            << "message: " << message;
    }
}

// TopK whose K is not an Int64 constant is a load rejection, never a dynamic
// shape: here K is a graph input, so no initializer provides it.

TEST_F(EngineLoaderNegative, TopKNonConstantKRejectedAtLoad)
{
    std::string body;
    put_sub(body, 1, encode_node("tk_dyn", "TopK", {"x", "k"}, {"v", "i"},
                      {encode_attribute_int("axis", -1),
                          encode_attribute_int("largest", 1),
                          encode_attribute_int("sorted", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 5}));
    put_sub(body, 11, encode_value_info("k", 1, {1}));
    put_sub(body, 12, encode_value_info("v", 1, {2, 2}));

    const std::string message =
        load_error_message(encode_model(body), "topk_dyn_k.onnx");
    EXPECT_NE(message.find("tk_dyn"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("TopK"), std::string::npos) << "message: " << message;
}

// ConstantOfShape `value` (opset 18) may be a FLOAT, INT64, or BOOL
// one-element tensor, or absent (float32 zero). Converted from the former
// ConstantOfShapeFloatValueRejected: FLOAT is now accepted, and each tensor
// element type decodes into its own Attribute alternative.

TEST_F(EngineLoader, HermeticConstantOfShapeFloatBoolAndAbsentValueLoad)
{
    const std::vector<float> fill{2.5F};
    std::string body;
    put_sub(body, 5, encode_initializer_int64("cshape", {1}, {3}));
    put_sub(body, 1, encode_node("cos_f", "ConstantOfShape", {"cshape"},
                      {"ff"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(1 /*FLOAT*/, {1},
                              raw_bytes_float(fill)))}));
    put_sub(body, 1, encode_node("cos_b", "ConstantOfShape", {"cshape"},
                      {"fb"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(9 /*BOOL*/, {1},
                              std::string(1, '\x01')))}));
    put_sub(body, 1, encode_node("cos_d", "ConstantOfShape", {"cshape"},
                      {"fd"}, {}));
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("cos_values.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 4U);
    ASSERT_EQ(graph.nodes[0].attributes.size(), 1U);
    EXPECT_EQ(std::get<float>(graph.nodes[0].attributes.front().value), 2.5F);
    ASSERT_EQ(graph.nodes[1].attributes.size(), 1U);
    EXPECT_TRUE(std::get<bool>(graph.nodes[1].attributes.front().value));
    EXPECT_TRUE(graph.nodes[2].attributes.empty());
}

// InstanceNormalization loads with its float `epsilon` attribute.

TEST_F(EngineLoader, HermeticInstanceNormalizationEpsilonLoads)
{
    std::string body;
    put_sub(body, 5, encode_initializer("s", {2}, {1.0F, 2.0F}));
    put_sub(body, 5, encode_initializer("b", {2}, {0.0F, 0.5F}));
    put_sub(body, 1, encode_node("in0", "InstanceNormalization",
                      {"x", "s", "b"}, {"y"},
                      {encode_attribute_float("epsilon", 1e-3F)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 3}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("instancenorm.onnx", encode_model(body)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes[0].op_type, "InstanceNormalization");
    ASSERT_EQ(graph.nodes[0].attributes.size(), 1U);
    EXPECT_EQ(graph.nodes[0].attributes.front().name, "epsilon");
    EXPECT_EQ(std::get<float>(graph.nodes[0].attributes.front().value), 1e-3F);
}

// InstanceNormalization rejects an attribute outside its allowlist.

TEST_F(EngineLoaderNegative, InstanceNormalizationUnknownAttributeRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer("s", {2}, {1.0F, 1.0F}));
    put_sub(body, 5, encode_initializer("b", {2}, {0.0F, 0.0F}));
    put_sub(body, 1, encode_node("in_bad", "InstanceNormalization",
                      {"x", "s", "b"}, {"y"},
                      {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2, 3, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 2, 3, 3}));

    const std::string message =
        load_error_message(encode_model(body), "instancenorm_attr.onnx");
    EXPECT_NE(message.find("in_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("InstanceNormalization"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("axis"), std::string::npos) << "message: " << message;
}

// ConstantOfShape whose `value` is a DOUBLE tensor (outside FLOAT, INT64,
// BOOL) rejects with node + op context.

TEST_F(EngineLoaderNegative, ConstantOfShapeDoubleValueRejected)
{
    const double fill = 0.5;
    std::string body;
    put_sub(body, 5, encode_initializer_int64("cshape", {1}, {3}));
    put_sub(body, 1, encode_node("cos_bad", "ConstantOfShape", {"cshape"},
                      {"fill"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(11 /*DOUBLE*/, {1},
                              std::string(reinterpret_cast<const char*>(&fill),
                                  sizeof(fill))))}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "cos_double.onnx");
    EXPECT_NE(message.find("cos_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ConstantOfShape"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("DOUBLE"), std::string::npos)
        << "message: " << message;
}

// ConstantOfShape whose `value` tensor carries two elements rejects.

TEST_F(EngineLoaderNegative, ConstantOfShapeTwoElementValueRejected)
{
    const std::vector<float> fill{1.0F, 2.0F};
    std::string body;
    put_sub(body, 5, encode_initializer_int64("cshape", {1}, {3}));
    put_sub(body, 1, encode_node("cos_two", "ConstantOfShape", {"cshape"},
                      {"fill"},
                      {encode_attribute_tensor("value",
                          encode_constant_tensor(1 /*FLOAT*/, {2},
                              raw_bytes_float(fill)))}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "cos_two.onnx");
    EXPECT_NE(message.find("cos_two"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("exactly one element"), std::string::npos)
        << "message: " << message;
}

// Slice takes no attributes at opset 18 (starts/ends/axes/steps are inputs).

TEST_F(EngineLoaderNegative, SliceAttributeRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("starts", {1}, {0}));
    put_sub(body, 5, encode_initializer_int64("ends", {1}, {5}));
    put_sub(body, 1, encode_node("sl_bad", "Slice", {"x", "starts", "ends"},
                      {"y"}, {encode_attribute_int("axis", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 10}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 5}));

    const std::string message =
        load_error_message(encode_model(body), "slice_attr.onnx");
    EXPECT_NE(message.find("sl_bad"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Slice"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("axis"), std::string::npos) << "message: " << message;
}

// A no-attribute new op (Mul) still rejects any attribute with full context.

TEST_F(EngineLoaderNegative, UnsupportedAttributeOnNewOp)
{
    std::string body;
    put_sub(body, 1, encode_node("mul_foo", "Mul", {"x", "x"}, {"y"},
        {encode_attribute_name_only("alpha")}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body), "mul_attr.onnx");
    EXPECT_NE(message.find("mul_foo"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Mul"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("alpha"), std::string::npos) << "message: " << message;
}

// A negative dim_value in an intermediate value_info entry still rejects.

TEST_F(EngineLoaderNegative, ValueInfoNegativeDimensionRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));
    std::string dim_msg;
    put_varint_field(dim_msg, 1, static_cast<uint64_t>(int64_t{-3}));
    std::string shape;
    put_sub(shape, 1, dim_msg);
    std::string tensor_type;
    put_varint_field(tensor_type, 1, 7);
    put_sub(tensor_type, 2, shape);
    std::string type_proto;
    put_sub(type_proto, 1, tensor_type);
    std::string vi;
    put_length_delimited(vi, 1, "neg");
    put_sub(vi, 2, type_proto);
    put_sub(body, 13, vi);

    const std::string message =
        load_error_message(encode_model(body), "vi_neg_dim.onnx");
    EXPECT_NE(message.find("neg"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("non-fixed"), std::string::npos)
        << "message: " << message;
}

// Declared opsets below the floor reject at the model level. There is no
// upper ceiling; newer declarations are checked per node instead.
TEST_F(EngineLoaderNegative, OpsetBelowFloorRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body, 10), "opset_10.onnx");
    EXPECT_NE(message.find("10"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("start at 11"), std::string::npos)
        << "message: " << message;
}

// A model without an ONNX-domain opset import still fails.
TEST_F(EngineLoaderNegative, MissingOpsetDeclarationRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    std::string model;
    put_sub(model, 7, body);
    const std::string message =
        load_error_message(model, "missing_opset.onnx");
    EXPECT_NE(message.find("does not declare"), std::string::npos)
        << "message: " << message;
}

// A source node whose operator has no schema at or below the declaration has
// no effective schema. LayerNormalization first appears at schema 17.
TEST_F(EngineLoaderNegative, NoEffectiveSchemaAtDeclaredOpsetRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("ln0", "LayerNormalization", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4}));

    const std::string message =
        load_error_message(encode_model(body, 11), "layernorm_11.onnx");
    EXPECT_NE(message.find("ln0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("LayerNormalization"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 11"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("no effective schema"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("first schema is 17"), std::string::npos)
        << "message: " << message;
}

// At opset 18 the reduce axes are an input, so the old attribute is illegal.
TEST_F(EngineLoaderNegative, ReduceAxesAttributeAtOpset18Rejected)
{
    std::string body;
    put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {1})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const std::string message =
        load_error_message(encode_model(body, 18), "reduce_axes_18.onnx");
    EXPECT_NE(message.find("reduce0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ReduceMean"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("axes"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("declared opset 18"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 18"), std::string::npos)
        << "message: " << message;
}

// An axes input is not part of the opset-13..17 reduce schemas.
TEST_F(EngineLoaderNegative, ReduceAxesInputBeforeOpset18Rejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
    put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x", "axes"}, {"y"},
                      {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const std::string message =
        load_error_message(encode_model(body, 13), "reduce_axes_input_13.onnx");
    EXPECT_NE(message.find("reduce0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ReduceMean"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("axes"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("schema 18"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 13"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 13"), std::string::npos)
        << "message: " << message;
}

// An empty-string second operand is still an operand; it is illegal before
// schema 18, where the axes form is the attribute.
TEST_F(EngineLoaderNegative, ReduceEmptyAxesInputBeforeOpset18Rejected)
{
    std::string body;
    put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x", ""}, {"y"},
                      {encode_attribute_ints("axes", {1})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const std::string message = load_error_message(
        encode_model(body, 13), "reduce_empty_axes_input_13.onnx");
    EXPECT_NE(message.find("reduce0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ReduceMean"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("schema 18"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 13"), std::string::npos)
        << "message: " << message;
}

// The explicit default noop=0 on ReduceMax is valid and needs no capability
// rejection; only the nonzero identity form is a tracked gap.
TEST_F(EngineLoader, HermeticReduceMaxNoopZeroAtOpset18Accepted)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
    put_sub(body, 1, encode_node("rmax0", "ReduceMax", {"x", "axes"}, {"y"},
                      {encode_attribute_int("noop_with_empty_axes", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("reduce_max_noop0_18.onnx", encode_model(body, 18)));
    ASSERT_EQ(graph.nodes.size(), 1U);
    EXPECT_EQ(graph.nodes.front().op_type, "ReduceMax");
}

// ReduceMax schema 18 defines noop_with_empty_axes, but the identity behavior
// is not implemented yet; report it as a capability gap, not an unknown
// attribute.
TEST_F(EngineLoaderNegative, ReduceMaxNoopAtOpset18RejectedAsCapabilityGap)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
    put_sub(body, 1, encode_node("rmax0", "ReduceMax", {"x", "axes"}, {"y"},
                      {encode_attribute_int("noop_with_empty_axes", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const std::string message =
        load_error_message(encode_model(body, 18), "reduce_max_noop_18.onnx");
    EXPECT_NE(message.find("rmax0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ReduceMax"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("noop_with_empty_axes"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("not implemented"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 18"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 18"), std::string::npos)
        << "message: " << message;
}

// noop_with_empty_axes was added at opset 18; it cannot appear in an older
// reduce declaration.
TEST_F(EngineLoaderNegative, ReduceNoopBeforeOpset18Rejected)
{
    std::string body;
    put_sub(body, 1, encode_node("reduce0", "ReduceMean", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {1}),
                          encode_attribute_int("noop_with_empty_axes", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1, 4}));

    const std::string message =
        load_error_message(encode_model(body, 13), "reduce_noop_13.onnx");
    EXPECT_NE(message.find("reduce0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("noop_with_empty_axes"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("schema 18"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 13"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 13"), std::string::npos)
        << "message: " << message;
}

// Only the default ScatterND reduction is executable here.
TEST_F(EngineLoaderNegative, ScatterNDUnsupportedReductionRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("idx", {1, 1}, {0}));
    put_sub(body, 5, encode_initializer("upd", {1, 3}, {1.0F, 2.0F, 3.0F}));
    put_sub(body, 1, encode_node("scatter0", "ScatterND",
                      {"x", "idx", "upd"}, {"y"},
                      {encode_attribute_string("reduction", "add")}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const std::string message =
        load_error_message(encode_model(body, 18), "scatter_add.onnx");
    EXPECT_NE(message.find("scatter0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("ScatterND"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("add"), std::string::npos) << "message: " << message;
}

// The reduction attribute itself is not legal before opset 16.
TEST_F(EngineLoaderNegative, ScatterNDReductionBeforeOpset16Rejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("idx", {1, 1}, {0}));
    put_sub(body, 5, encode_initializer("upd", {1, 3}, {1.0F, 2.0F, 3.0F}));
    put_sub(body, 1, encode_node("scatter0", "ScatterND",
                      {"x", "idx", "upd"}, {"y"},
                      {encode_attribute_string("reduction", "none")}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const std::string message =
        load_error_message(encode_model(body, 13), "scatter_13.onnx");
    EXPECT_NE(message.find("reduction"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("schema 16"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 13"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 13"), std::string::npos)
        << "message: " << message;
}

// The old axes attributes are illegal from opset 13, where the same values
// are inputs. The allowlist admits them so the normalizer can name the form.
TEST_F(EngineLoaderNegative, LegacyAxesAttributeAtThirteenRejected)
{
    for (const std::string& op :
        { std::string("ReduceSum"), std::string("Squeeze"),
            std::string("Unsqueeze") }) {
        std::string body;
        put_sub(body, 1, encode_node("legacy0", op, {"x"}, {"y"},
                          {encode_attribute_ints("axes", {1})}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

        const std::string message =
            load_error_message(encode_model(body, 13), "legacy_axes_13.onnx");
        EXPECT_NE(message.find("legacy0"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find(op), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("axes"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("schema 13"), std::string::npos)
            << "message: " << message;
    }
}

// Split's split attribute is illegal from opset 13, where it is an input.
TEST_F(EngineLoaderNegative, LegacySplitAttributeAtThirteenRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("sp0", "Split", {"x"}, {"a", "b"},
                      {encode_attribute_int("axis", 1),
                          encode_attribute_ints("split", {1, 2})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("a", 1, {2, 1, 4}));
    put_sub(body, 12, encode_value_info("b", 1, {2, 2, 4}));

    const std::string message =
        load_error_message(encode_model(body, 13), "split_attr_13.onnx");
    EXPECT_NE(message.find("sp0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("split"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("schema 13"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 13"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 13"), std::string::npos)
        << "message: " << message;
}

// Opsets 11/12 use the attribute forms; an axes/split input is not legal
// until 13.
TEST_F(EngineLoaderNegative, LegacyAxesInputBeforeThirteenRejected)
{
    for (const std::string& op :
        { std::string("ReduceSum"), std::string("Squeeze"),
            std::string("Unsqueeze") }) {
        std::string body;
        put_sub(body, 5, encode_initializer_int64("axes", {1}, {1}));
        put_sub(body, 1, encode_node("legacy0", op, {"x", "axes"}, {"y"}, {}));
        put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
        put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

        const std::string message =
            load_error_message(encode_model(body, 11), "legacy_input_11.onnx");
        EXPECT_NE(message.find("legacy0"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("axes"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("requires schema 13"), std::string::npos)
            << "message: " << message;
    }
}

TEST_F(EngineLoaderNegative, LegacySplitInputBeforeThirteenRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("split", {2}, {1, 2}));
    put_sub(body, 1, encode_node("sp0", "Split", {"x", "split"}, {"a", "b"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("a", 1, {2, 1, 4}));
    put_sub(body, 12, encode_value_info("b", 1, {2, 2, 4}));

    const std::string message =
        load_error_message(encode_model(body, 11), "split_input_11.onnx");
    EXPECT_NE(message.find("sp0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("split"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("requires schema 13"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 11"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 11"), std::string::npos)
        << "message: " << message;
}

TEST_F(EngineLoaderNegative, LegacyUnsqueezeMissingAxesRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("us0", "Unsqueeze", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const std::string message =
        load_error_message(encode_model(body, 11), "unsqueeze_missing_11.onnx");
    EXPECT_NE(message.find("us0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("axes"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("required"), std::string::npos)
        << "message: " << message;
}

TEST_F(EngineLoaderNegative, LegacyUnsqueezeEmptyAxesRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("us0", "Unsqueeze", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3}));

    const std::string message =
        load_error_message(encode_model(body, 11), "unsqueeze_empty_11.onnx");
    EXPECT_NE(message.find("us0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("must not be empty"), std::string::npos)
        << "message: " << message;
}

TEST_F(EngineLoaderNegative, LegacySplitEmptyRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("sp0", "Split", {"x"}, {"a", "b"},
                      {encode_attribute_int("axis", 1),
                          encode_attribute_ints("split", {})}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("a", 1, {2, 1, 4}));
    put_sub(body, 12, encode_value_info("b", 1, {2, 2, 4}));

    const std::string message =
        load_error_message(encode_model(body, 11), "split_empty_11.onnx");
    EXPECT_NE(message.find("sp0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("must not be empty"), std::string::npos)
        << "message: " << message;
}

// noop_with_empty_axes was added at opset 13; it cannot appear in the older
// attribute form.
TEST_F(EngineLoaderNegative, LegacyReduceSumNoopBeforeThirteenRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("rs0", "ReduceSum", {"x"}, {"y"},
                      {encode_attribute_ints("axes", {1}),
                          encode_attribute_int("noop_with_empty_axes", 0)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 1}));

    const std::string message =
        load_error_message(encode_model(body, 11), "reduce_sum_noop_11.onnx");
    EXPECT_NE(message.find("rs0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("noop_with_empty_axes"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("schema 13"), std::string::npos)
        << "message: " << message;
}

TEST_F(EngineLoaderNegative, LegacySoftmaxAxisOutOfRangeRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("sm0", "Softmax", {"x"}, {"y"},
                      {encode_attribute_int("axis", 3)}));
    put_sub(body, 11, encode_value_info("x", 1, {2, 3, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {2, 3, 4}));

    const nert::Graph graph = nert::LoadGraphFromFile(
        write_model_bytes("softmax_axis_11.onnx", encode_model(body, 11)));
    try {
        nert::Model model(graph, { { "x", { 2, 3, 4 } } }, nert::CpuDevice());
        FAIL() << "expected shape inference to reject the Softmax axis";
    } catch (const nert::ModelLoadException& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("sm0"), std::string::npos)
            << "message: " << message;
        EXPECT_NE(message.find("out of range"), std::string::npos)
            << "message: " << message;
    }
}

// Cast's post-18 attributes change rounding/saturation semantics. A plain
// Cast is accepted (see the positive row); using either attribute fails with
// the effective schema version.
TEST_F(EngineLoaderNegative, NewerCastSaturateRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("cast0", "Cast", {"x"}, {"y"},
                      {encode_attribute_int("to", 7),
                          encode_attribute_int("saturate", 1)}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 7, {2}));

    const std::string message =
        load_error_message(encode_model(body, 19), "cast_saturate_19.onnx");
    EXPECT_NE(message.find("cast0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Cast"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("saturate"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("declared opset 19"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 19"), std::string::npos)
        << "message: " << message;
}

TEST_F(EngineLoaderNegative, NewerCastRoundModeRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("cast0", "Cast", {"x"}, {"y"},
                      {encode_attribute_int("to", 7),
                          encode_attribute_string("round_mode", "up")}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 7, {2}));

    const std::string message =
        load_error_message(encode_model(body, 24), "cast_round_24.onnx");
    EXPECT_NE(message.find("cast0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("round_mode"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 24"), std::string::npos)
        << "message: " << message;
}

TEST_F(EngineLoaderNegative, NewerPadWrapRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer_int64("pads", {4}, {0, 1, 0, 1}));
    put_sub(body, 1, encode_node("pad0", "Pad", {"x", "pads"}, {"y"},
                      {encode_attribute_string("mode", "wrap")}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 4}));

    const std::string message =
        load_error_message(encode_model(body, 19), "pad_wrap_19.onnx");
    EXPECT_NE(message.find("pad0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("wrap"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 19"), std::string::npos)
        << "message: " << message;
}

// AveragePool's dilations attribute only appears at schema 19; the kernel
// implements the undilated form.
TEST_F(EngineLoaderNegative, NewerAveragePoolDilationsRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("pool0", "AveragePool", {"x"}, {"y"},
                      {encode_attribute_ints("kernel_shape", {2, 2}),
                          encode_attribute_ints("dilations", {1, 1})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 4, 4}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 2, 2}));

    const std::string message =
        load_error_message(encode_model(body, 19), "averagepool_dil_19.onnx");
    EXPECT_NE(message.find("pool0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("dilations"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 19"), std::string::npos)
        << "message: " << message;
}

// Range's stash_type only appears at schema 27.
TEST_F(EngineLoaderNegative, NewerRangeStashTypeRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer("start", {}, {0.0F}));
    put_sub(body, 5, encode_initializer("limit", {}, {3.0F}));
    put_sub(body, 5, encode_initializer("delta", {}, {1.0F}));
    put_sub(body, 1, encode_node("range0", "Range", {"start", "limit", "delta"},
                      {"y"}, {encode_attribute_int("stash_type", 1)}));
    put_sub(body, 12, encode_value_info("y", 1, {3}));

    const std::string message =
        load_error_message(encode_model(body, 27), "range_stash_27.onnx");
    EXPECT_NE(message.find("range0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("stash_type"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 27"), std::string::npos)
        << "message: " << message;
}

// A historical schema that carries an unimplemented attribute is guarded by
// the full-history manifest. ConvTranspose.output_shape is legal at schema 11
// but has no kernel or shape-inference implementation.
TEST_F(EngineLoaderNegative, ConvTransposeOutputShapeRejectedByManifest)
{
    std::string body;
    put_sub(body, 5, encode_initializer("w", {1, 1, 2, 2}, {1.0F, 1.0F, 1.0F, 1.0F}));
    put_sub(body, 1, encode_node("ct0", "ConvTranspose", {"x", "w"}, {"y"},
                      {encode_attribute_ints("output_shape", {2, 2})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 2, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 3, 3}));

    const std::string message =
        load_error_message(encode_model(body, 11), "convtranspose_output_shape_11.onnx");
    EXPECT_NE(message.find("ct0"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("ConvTranspose"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("output_shape"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("declared opset 11"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("effective schema 11"), std::string::npos) << "message: " << message;
}

// Resize.axes is a schema-18 attribute but the kernel and shape inference do
// not consume it, so the manifest guards it instead of silently ignoring it.
TEST_F(EngineLoaderNegative, ResizeAxesAtSchemaEighteenRejected)
{
    std::string body;
    put_sub(body, 5, encode_initializer("scales", {4}, {1.0F, 1.0F, 1.0F, 1.0F}));
    put_sub(body, 1, encode_node("resize0", "Resize", {"x", "", "scales"}, {"y"},
                      {encode_attribute_ints("axes", {2, 3})}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 3, 8, 8}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 3, 8, 8}));

    const std::string message =
        load_error_message(encode_model(body, 18), "resize_axes_18.onnx");
    EXPECT_NE(message.find("resize0"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("Resize"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("axes"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("effective schema 18"), std::string::npos) << "message: " << message;
}

// Split.num_outputs is legal at schema 18 but is not implemented by the
// source-form Split path.
TEST_F(EngineLoaderNegative, SplitNumOutputsAtSchemaEighteenRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("split0", "Split", {"x"}, {"y0", "y1"},
                      {encode_attribute_int("axis", 1),
                          encode_attribute_int("num_outputs", 2)}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 4}));
    put_sub(body, 12, encode_value_info("y0", 1, {1, 2}));
    put_sub(body, 12, encode_value_info("y1", 1, {1, 2}));

    const std::string message =
        load_error_message(encode_model(body, 18), "split_num_outputs_18.onnx");
    EXPECT_NE(message.find("split0"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("Split"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("num_outputs"), std::string::npos) << "message: " << message;
    EXPECT_NE(message.find("effective schema 18"), std::string::npos) << "message: " << message;
}

// GridSample's schema-20 mode default/value change is a behavior change, so
// every GridSample node at effective 20+ is rejected until implemented.
TEST_F(EngineLoaderNegative, NewerGridSampleBehaviorChangeRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("grid0", "GridSample", {"x", "grid"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {1, 1, 2, 2}));
    put_sub(body, 11, encode_value_info("grid", 1, {1, 2, 2, 2}));
    put_sub(body, 12, encode_value_info("y", 1, {1, 1, 2, 2}));

    const std::string message =
        load_error_message(encode_model(body, 20), "gridsample_20.onnx");
    EXPECT_NE(message.find("grid0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("mode default"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("effective schema 20"), std::string::npos)
        << "message: " << message;
}

// Operators introduced after the implemented set keep the existing explicit
// unknown-operator failure even when the declaration is newer.
TEST_F(EngineLoaderNegative, NewerUnknownOperatorRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("mish0", "Mish", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 1, {2}));
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body, 19), "mish_19.onnx");
    EXPECT_NE(message.find("mish0"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("Mish"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("unsupported op_type"), std::string::npos)
        << "message: " << message;
}

// A post-18 dtype is rejected by the existing element-type gate, not by a
// kernel or a compatibility fallback.
TEST_F(EngineLoaderNegative, NewerUnsupportedDtypeRejected)
{
    std::string body;
    put_sub(body, 1, encode_node("relu0", "Relu", {"x"}, {"y"}, {}));
    put_sub(body, 11, encode_value_info("x", 17, {2})); // FLOAT8E4M3FN
    put_sub(body, 12, encode_value_info("y", 1, {2}));

    const std::string message =
        load_error_message(encode_model(body, 19), "float8_19.onnx");
    EXPECT_NE(message.find("x"), std::string::npos)
        << "message: " << message;
    EXPECT_NE(message.find("unsupported element type"), std::string::npos)
        << "message: " << message;
}
