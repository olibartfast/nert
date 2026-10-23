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
// ArgMax/Einsum), the extra math rows (Sqrt/Cos/Sin/Exp/Not/Less), the
// reduction rows (ReduceSum/ReduceMax), and the index-move/integer rows
// (Split/GatherElements/TopK/Mod/Tile), and the resampling rows
// (Resize/GridSample), and the normalization/range/pad rows
// (InstanceNormalization/Range/Pad), the second math rows
// (Clip/Log/Pow/Max/Min/Or), and the index/shape rows
// (Identity/Squeeze/CumSum/EyeLike/ScatterND/ReduceMean), and the
// transposed-convolution row (ConvTranspose).

#include "nert/device.hpp"

#include <string>
#include <unordered_map>

namespace nert {
namespace opencl_kernels {

void Abs(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Neg(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Floor(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GreaterOrEqual(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
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
void Sqrt(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Cos(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Sin(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Exp(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Not(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Less(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Split(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GatherElements(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void TopK(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Mod(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Tile(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GridSample(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void InstanceNormalization(const Node& node, const std::vector<TensorView>& inputs,
                           const std::vector<TensorView>& outputs);
void Range(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Pad(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Clip(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Log(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Pow(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Max(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Min(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Or(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Identity(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Squeeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void CumSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void EyeLike(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ScatterND(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceMean(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ConvTranspose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);

KernelFn FindOpenCLKernel(const std::string& op_type) {
    static const std::unordered_map<std::string, KernelFn> table = {
        {"Abs", &Abs},
        {"Neg", &Neg},
        {"Floor", &Floor},
        {"GreaterOrEqual", &GreaterOrEqual},
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
        {"Sqrt", &Sqrt},
        {"Cos", &Cos},
        {"Sin", &Sin},
        {"Exp", &Exp},
        {"Not", &Not},
        {"Less", &Less},
        {"ReduceSum", &ReduceSum},
        {"ReduceMax", &ReduceMax},
        {"Split", &Split},
        {"GatherElements", &GatherElements},
        {"TopK", &TopK},
        {"Mod", &Mod},
        {"Tile", &Tile},
        {"Resize", &Resize},
        {"GridSample", &GridSample},
        {"InstanceNormalization", &InstanceNormalization},
        {"Range", &Range},
        {"Pad", &Pad},
        {"Clip", &Clip},
        {"Log", &Log},
        {"Pow", &Pow},
        {"Max", &Max},
        {"Min", &Min},
        {"Or", &Or},
        {"Identity", &Identity},
        {"Squeeze", &Squeeze},
        {"CumSum", &CumSum},
        {"EyeLike", &EyeLike},
        {"ScatterND", &ScatterND},
        {"ReduceMean", &ReduceMean},
        {"ConvTranspose", &ConvTranspose},

    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace opencl_kernels
} // namespace nert
