// The OpenCL operator table (N1d pilot).
//
// One function-local static maps exact op-type strings to their kernels;
// unknown op types resolve to null so the executor rejects the node naming
// op + node. Rows: Relu/Add/Mul/Sub/Div/Sigmoid/Tanh/Softmax (first-party
// OpenCL C, runtime-compiled) plus Reshape (validated buffer copy),
// Transpose/Gather (indexed copies), Slice/Concat/Shape/Unsqueeze
// (movement rows), MaxPool/AveragePool/GlobalAveragePool (pools),
// Flatten/Gemm, Conv, MatMul/LayerNormalization, Expand/Where, and the
// OWLv2 family (Cast/ConstantOfShape/compare/And/IsNaN/Erf/Elu/ReduceL2/
// ArgMax/Einsum).

#include "nert/device.hpp"

#include <string>
#include <unordered_map>

namespace nert {
namespace opencl_kernels {

void Relu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Add(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Mul(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Sub(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Div(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Sigmoid(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Tanh(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Softmax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Reshape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Transpose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Gather(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Slice(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Concat(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Shape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Unsqueeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void MaxPool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void AveragePool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GlobalAveragePool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Flatten(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Gemm(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Conv(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void MatMul(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void LayerNormalization(const Node& node, const std::vector<TensorView>& inputs,
                        const std::vector<TensorView>& outputs);
void Expand(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Where(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Cast(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ConstantOfShape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Equal(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Greater(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void LessOrEqual(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void And(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void IsNaN(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Erf(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Elu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceL2(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ArgMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Einsum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

KernelFn FindOpenCLKernel(const std::string& op_type) {
    static const std::unordered_map<std::string, KernelFn> table = {
        {"Relu", &Relu},
        {"Add", &Add},
        {"Mul", &Mul},
        {"Sub", &Sub},
        {"Div", &Div},
        {"Sigmoid", &Sigmoid},
        {"Tanh", &Tanh},
        {"Softmax", &Softmax},
        {"Reshape", &Reshape},
        {"Transpose", &Transpose},
        {"Gather", &Gather},
        {"Slice", &Slice},
        {"Concat", &Concat},
        {"Shape", &Shape},
        {"Unsqueeze", &Unsqueeze},
        {"MaxPool", &MaxPool},
        {"AveragePool", &AveragePool},
        {"GlobalAveragePool", &GlobalAveragePool},
        {"Flatten", &Flatten},
        {"Gemm", &Gemm},
        {"Conv", &Conv},
        {"MatMul", &MatMul},
        {"LayerNormalization", &LayerNormalization},
        {"Expand", &Expand},
        {"Where", &Where},
        {"Cast", &Cast},
        {"ConstantOfShape", &ConstantOfShape},
        {"Equal", &Equal},
        {"Greater", &Greater},
        {"LessOrEqual", &LessOrEqual},
        {"And", &And},
        {"IsNaN", &IsNaN},
        {"Erf", &Erf},
        {"Elu", &Elu},
        {"ReduceL2", &ReduceL2},
        {"ArgMax", &ArgMax},
        {"Einsum", &Einsum},

    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace opencl_kernels
} // namespace nert
