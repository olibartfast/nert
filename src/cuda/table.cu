// The CUDA operator table (N1a pilot + N1b Conv).
//
// One function-local static maps exact op-type strings to their kernels;
// unknown op types resolve to null so the executor rejects the node naming
// op + node. Rows: Gemm (cuBLAS), Relu/Add (first-party), Conv (cuDNN,
// NERT_HAS_CUDNN builds only).

#include "nert/device.hpp"

#include <string>
#include <unordered_map>

namespace nert {
namespace cuda_kernels {

void Gemm(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Relu(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
void Add(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
#ifdef NERT_HAS_CUDNN
void Conv(const Node& node, const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs);
#endif

KernelFn FindCudaKernel(const std::string& op_type) {
    static const std::unordered_map<std::string, KernelFn> table = {
        {"Gemm", &Gemm},
        {"Relu", &Relu},
        {"Add", &Add},
#ifdef NERT_HAS_CUDNN
        {"Conv", &Conv},
#endif
    };
    const auto it = table.find(op_type);
    return it == table.end() ? nullptr : it->second;
}

} // namespace cuda_kernels
} // namespace nert
