// The CUDA device: allocator, transfers, and the kernel table behind the
// per-device seam.
//
// M1 scaffold grew into the N1a pilot table: the kernel lookup lives in
// src/cuda/table.cu (Gemm via cuBLAS, first-party Relu/Add). Only the C++17
// standard library plus the CUDA runtime API is used here.

#include "nert/device.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <string>

namespace nert {
namespace cuda_kernels {
KernelFn FindCudaKernel(const std::string& op_type);
} // namespace cuda_kernels
namespace {

// CUDA errors carry the call site: every runtime call is checked, and a
// failure throws naming the API call and the numeric error.
void CheckCuda(cudaError_t status, const char* what, const std::string& context) {
    if (status != cudaSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + cudaGetErrorString(status));
    }
}

// The planner lays tensors out on 64-byte boundaries; cudaMalloc meets at
// least that alignment.
class CudaAllocator final : public Allocator {
  public:
    // Returns device storage of at least `bytes` bytes, or null when `bytes`
    // is not positive or the allocation fails. The memory is not zeroed.
    void* allocate(int64_t bytes) override {
        if (bytes <= 0) {
            return nullptr;
        }
        void* buffer = nullptr;
        if (cudaMalloc(&buffer, static_cast<std::size_t>(bytes)) != cudaSuccess) {
            return nullptr;
        }
        return buffer;
    }

    // Frees a buffer from allocate(); a null pointer is a safe no-op.
    void release(void* buffer) override {
        if (buffer != nullptr) {
            cudaFree(buffer);
        }
    }
};

class CudaTransfer final : public Transfer {
  public:
    // Copies `bytes` bytes from the host into device memory.
    void host_to_device(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        CheckCuda(cudaMemcpy(dst, src, static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice), "cudaMemcpy",
                  "CudaDevice");
    }

    // Copies `bytes` bytes from device memory back to the host.
    void device_to_host(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        CheckCuda(cudaMemcpy(dst, src, static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost), "cudaMemcpy",
                  "CudaDevice");
    }
};

// The operator lookup for the CUDA device. It forwards to the CUDA kernel
// table in src/cuda/table.cu; ops without a CUDA row resolve to null so the
// executor rejects the node naming op + node.
class CudaKernelTable final : public KernelTable {
  public:
    KernelFn find(const std::string& op_type) const override { return cuda_kernels::FindCudaKernel(op_type); }
};

class CudaDeviceImpl final : public Device {
  public:
    const char* name() const override { return "cuda"; }
    Allocator& allocator() override { return allocator_; }
    Transfer& transfer() override { return transfer_; }
    const KernelTable& kernels() const override { return kernels_; }
    bool is_host() const override { return false; }

  private:
    CudaAllocator allocator_;
    CudaTransfer transfer_;
    CudaKernelTable kernels_;
};

} // namespace

const Device& CudaDevice() {
    // One process-lifetime instance; the reference is stable across calls.
    static CudaDeviceImpl device;
    return device;
}

} // namespace nert
