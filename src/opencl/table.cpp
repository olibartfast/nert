// The OpenCL operator table (N1d pilot).
//
// One function-local static maps exact op-type strings to their kernels;
// unknown op types resolve to null so the executor rejects the node naming
// op + node. Rows: Relu/Add/Mul/Sub/Div/Sigmoid/Tanh/Softmax (first-party
// OpenCL C, runtime-compiled) plus Reshape (validated buffer copy).

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

KernelFn FindOpenCLKernel(const std::string& op_type) {
    static const std::unordered_map<std::string, KernelFn> table = {
        {"Relu", &Relu},       {"Add", &Add},   {"Mul", &Mul},         {"Sub", &Sub},         {"Div", &Div},
        {"Sigmoid", &Sigmoid}, {"Tanh", &Tanh}, {"Softmax", &Softmax}, {"Reshape", &Reshape},
    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace opencl_kernels
} // namespace nert
