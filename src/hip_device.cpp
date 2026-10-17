// The HIP device: allocator, transfers, and an (M1-empty) kernel table
// behind the per-device seam.
//
// N1c HIP-on-NVIDIA slice: the HIP runtime API with the NVIDIA backend
// (__HIP_PLATFORM_NVIDIA__ maps hip* onto CUDA; this file compiles with
// nvcc). Raw HIP runtime API only; RAII locally. Construction touches the
// driver and throws naming the cause when no GPU is usable — presence
// probes catch that and SKIP.

#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>
#include <string>
#include <vector>

namespace nert {
namespace hip_kernels {
// M1: empty — every op resolves to null. M2+ rows land in src/hip/.
KernelFn FindHIPKernel(const std::string& /*op_type*/) { return nullptr; }
} // namespace hip_kernels
namespace {

void CheckHip(hipError_t status, const char* what, const std::string& context) {
    if (status != hipSuccess) {
        throw InferenceException(context + ": " + what + " failed: " + hipGetErrorString(status));
    }
}

// Touches the driver: no usable HIP device → throws naming the cause, so
// probes turn it into SKIP. A throw retries on the next call (function
// statics re-run after an exception), so a later GPU appearance recovers.
void EnsureDevice() {
    static bool ready = [] {
        int count = 0;
        CheckHip(hipGetDeviceCount(&count), "hipGetDeviceCount", "HipDevice");
        if (count <= 0) {
            throw InferenceException("HipDevice: no HIP device found");
        }
        CheckHip(hipSetDevice(0), "hipSetDevice", "HipDevice");
        return true;
    }();
    (void)ready;
}

class HIPAllocator final : public Allocator {
  public:
    // Returns device storage of at least `bytes` bytes, or null when `bytes`
    // is not positive or the allocation fails. The memory is not zeroed.
    void* allocate(int64_t bytes) override {
        if (bytes <= 0) {
            return nullptr;
        }
        EnsureDevice();
        void* buffer = nullptr;
        if (hipMalloc(&buffer, static_cast<std::size_t>(bytes)) != hipSuccess) {
            return nullptr;
        }
        return buffer;
    }

    // Frees a buffer from allocate(); a null pointer is a safe no-op.
    void release(void* buffer) override {
        if (buffer != nullptr) {
            (void)hipFree(buffer);
        }
    }
};

class HIPTransfer final : public Transfer {
  public:
    // Copies `bytes` bytes from the host into device memory (blocking).
    void host_to_device(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        EnsureDevice();
        CheckHip(hipMemcpy(dst, src, static_cast<std::size_t>(bytes), hipMemcpyHostToDevice), "hipMemcpy", "HipDevice");
    }

    // Copies `bytes` bytes from device memory back to the host (blocking).
    void device_to_host(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        EnsureDevice();
        CheckHip(hipMemcpy(dst, src, static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost), "hipMemcpy", "HipDevice");
    }
};

// The operator lookup for the HIP device. M1: empty — every op resolves
// to null so the executor rejects the node naming op + node. M2+ rows land
// in src/hip/ (hipified from src/cuda/, hand-reviewed).
class HIPKernelTable final : public KernelTable {
  public:
    KernelFn find(const std::string& op_type) const override { return hip_kernels::FindHIPKernel(op_type); }
};

class HIPDeviceImpl final : public Device {
  public:
    const char* name() const override { return "hip"; }
    Allocator& allocator() override { return allocator_; }
    Transfer& transfer() override { return transfer_; }
    const KernelTable& kernels() const override { return kernels_; }
    bool is_host() const override { return false; }

  private:
    HIPAllocator allocator_;
    HIPTransfer transfer_;
    HIPKernelTable kernels_;
};

} // namespace

const Device& HipDevice() {
    // Touches the driver at first call (throws when there is no GPU);
    // the reference is stable across calls.
    EnsureDevice();
    static HIPDeviceImpl device;
    return device;
}

} // namespace nert
