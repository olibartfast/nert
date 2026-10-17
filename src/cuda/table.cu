// The CUDA operator table (N1a pilot + N1b Conv + N1e first-party).
//
// One function-local static maps exact op-type strings to their kernels;
// unknown op types resolve to null so the executor rejects the node naming
// op + node. Rows: Gemm (cuBLAS), Relu/Add/Mul/Sub/Div/MaxPool/AveragePool/Sigmoid/Tanh/Softmax/
// Reshape/Unsqueeze/Flatten/Expand (first-party), Conv (cuDNN, NERT_HAS_CUDNN builds only).

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
void Sigmoid(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Tanh(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Softmax(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Reshape(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Unsqueeze(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Flatten(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Expand(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
#ifdef NERT_HAS_CUDNN
void Conv(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
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
        {"Sigmoid", &Sigmoid},
        {"Tanh", &Tanh},
        {"Softmax", &Softmax},
        {"Reshape", &Reshape},
        {"Unsqueeze", &Unsqueeze},
        {"Flatten", &Flatten},
        {"Expand", &Expand},
#ifdef NERT_HAS_CUDNN
        {"Conv", &Conv},
#endif
    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace cuda_kernels
} // namespace nert
