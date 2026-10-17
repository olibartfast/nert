// The CUDA operator table (N1a pilot + N1b Conv + N1e first-party).
//
// One function-local static maps exact op-type strings to their kernels;
// unknown op types resolve to null so the executor rejects the node naming
// op + node. Rows: Gemm (cuBLAS),
// Relu/Add/Mul/Sub/Div/MaxPool/AveragePool/GlobalAveragePool/Sigmoid/Tanh/Sqrt/Erf/Softmax/
// Reshape/Unsqueeze/Flatten/Expand/Concat/Split/Slice/Transpose/Gather/GatherElements/MatMul/Cast/Equal/Where/
// Shape/ConstantOfShape/Mod/ReduceMax/ReduceSum/TopK/Resize/Sqrt/Erf/LayerNormalization/Sin/Cos/Exp/Elu/Log/Floor/
// IsNaN/Abs/Neg/Pow/Greater/Less/GreaterOrEqual/LessOrEqual/And/Or/Not/Max/Min/Clip/Tile/Identity/Squeeze/Range/
// EyeLike/Pad/ReduceMean/ReduceL2/ArgMax/CumSum/ScatterND (first-party, except MatMul via cuBLAS like Gemm), Conv
// (cuDNN, NERT_HAS_CUDNN builds only).

#include "nert/device.hpp"

#include <string>
#include <unordered_map>

namespace nert {
namespace cuda_kernels {

void Gemm(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Relu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Add(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Mul(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Sub(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Div(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void MaxPool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void AveragePool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GlobalAveragePool(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Sigmoid(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Tanh(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Sqrt(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Erf(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void LayerNormalization(const Node& node, const std::vector<TensorView>& inputs,
                        const std::vector<TensorView>& outputs);
void Sin(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Cos(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Exp(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Elu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Log(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Floor(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void IsNaN(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Abs(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Neg(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Pow(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Greater(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Less(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GreaterOrEqual(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void LessOrEqual(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void And(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Or(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Not(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Max(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Min(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Clip(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Identity(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Squeeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Tile(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Range(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void EyeLike(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Pad(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceMean(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceL2(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ArgMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void CumSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ScatterND(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Softmax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Reshape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Unsqueeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Flatten(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Expand(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Concat(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Split(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Slice(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Transpose(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Gather(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void GatherElements(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void MatMul(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Cast(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Equal(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Where(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Shape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ConstantOfShape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Mod(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceMax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void ReduceSum(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void TopK(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Resize(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
#ifdef NERT_HAS_CUDNN
#ifdef NERT_HAS_CUDNN_FRONTEND
void ConvFrontend(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
#else
void Conv(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
#endif
#endif

KernelFn FindCudaKernel(const std::string& op_type) {
    static const std::unordered_map<std::string, KernelFn> table = {
        {"Gemm", &Gemm},
        {"Relu", &Relu},
        {"Add", &Add},
        {"Mul", &Mul},
        {"Sub", &Sub},
        {"Div", &Div},
        {"MaxPool", &MaxPool},
        {"AveragePool", &AveragePool},
        {"GlobalAveragePool", &GlobalAveragePool},
        {"Sigmoid", &Sigmoid},
        {"Tanh", &Tanh},
        {"Sqrt", &Sqrt},
        {"Erf", &Erf},
        {"LayerNormalization", &LayerNormalization},
        {"Sin", &Sin},
        {"Cos", &Cos},
        {"Exp", &Exp},
        {"Elu", &Elu},
        {"Log", &Log},
        {"Floor", &Floor},
        {"IsNaN", &IsNaN},
        {"Abs", &Abs},
        {"Neg", &Neg},
        {"Pow", &Pow},
        {"Greater", &Greater},
        {"Less", &Less},
        {"GreaterOrEqual", &GreaterOrEqual},
        {"LessOrEqual", &LessOrEqual},
        {"And", &And},
        {"Or", &Or},
        {"Not", &Not},
        {"Max", &Max},
        {"Min", &Min},
        {"Clip", &Clip},
        {"Identity", &Identity},
        {"Squeeze", &Squeeze},
        {"Tile", &Tile},
        {"Range", &Range},
        {"EyeLike", &EyeLike},
        {"Pad", &Pad},
        {"ReduceMean", &ReduceMean},
        {"ReduceL2", &ReduceL2},
        {"ArgMax", &ArgMax},
        {"CumSum", &CumSum},
        {"ScatterND", &ScatterND},
        {"Softmax", &Softmax},
        {"Reshape", &Reshape},
        {"Unsqueeze", &Unsqueeze},
        {"Flatten", &Flatten},
        {"Expand", &Expand},
        {"Concat", &Concat},
        {"Split", &Split},
        {"Slice", &Slice},
        {"Transpose", &Transpose},
        {"Gather", &Gather},
        {"GatherElements", &GatherElements},
        {"MatMul", &MatMul},
        {"Cast", &Cast},
        {"Equal", &Equal},
        {"Where", &Where},
        {"Shape", &Shape},
        {"ConstantOfShape", &ConstantOfShape},
        {"Mod", &Mod},
        {"ReduceMax", &ReduceMax},
        {"ReduceSum", &ReduceSum},
        {"TopK", &TopK},
        {"Resize", &Resize},
#ifdef NERT_HAS_CUDNN
#ifdef NERT_HAS_CUDNN_FRONTEND
        {"Conv", &ConvFrontend},
#else
        {"Conv", &Conv},
#endif
#endif
    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace cuda_kernels
} // namespace nert
