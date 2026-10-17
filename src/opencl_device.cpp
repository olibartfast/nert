// The OpenCL device: allocator, transfers, and an (M1-empty) kernel table
// behind the per-device seam.
//
// N1d portable backend, OpenCL-first (Vulkan needs a shader toolchain this
// box lacks; see the N1d packet). Raw OpenCL C API only; RAII locally.
// Construction selects the first GPU on the first platform that offers one
// and throws naming the cause when there is none — presence probes catch
// that and SKIP.

#include "nert/device.hpp"
#include "opencl/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_kernels {
KernelFn FindOpenCLKernel(const std::string& op_type);
} // namespace opencl_kernels
namespace {

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
    }
}

class OpenCLAllocator final : public Allocator {
  public:
    // Returns device storage of at least `bytes` bytes, or null when `bytes`
    // is not positive or the allocation fails. The memory is not zeroed.
    void* allocate(int64_t bytes) override {
        if (bytes <= 0) {
            return nullptr;
        }
        cl_int status = CL_SUCCESS;
        cl_mem buffer = clCreateBuffer(opencl_runtime::Context(), CL_MEM_READ_WRITE, static_cast<std::size_t>(bytes),
                                       nullptr, &status);
        if (status != CL_SUCCESS) {
            return nullptr;
        }
        return buffer;
    }

    // Frees a buffer from allocate(); a null pointer is a safe no-op.
    void release(void* buffer) override {
        if (buffer != nullptr) {
            clReleaseMemObject(reinterpret_cast<cl_mem>(buffer));
        }
    }
};

class OpenCLTransfer final : public Transfer {
  public:
    // Copies `bytes` bytes from the host into device memory (blocking).
    void host_to_device(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        CheckCl(clEnqueueWriteBuffer(opencl_runtime::Queue(), reinterpret_cast<cl_mem>(dst), CL_TRUE, 0,
                                     static_cast<std::size_t>(bytes), src, 0, nullptr, nullptr),
                "clEnqueueWriteBuffer", "OpenCLDevice");
    }

    // Copies `bytes` bytes from device memory back to the host (blocking).
    void device_to_host(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        CheckCl(clEnqueueReadBuffer(opencl_runtime::Queue(), reinterpret_cast<cl_mem>(const_cast<void*>(src)), CL_TRUE,
                                    0, static_cast<std::size_t>(bytes), dst, 0, nullptr, nullptr),
                "clEnqueueReadBuffer", "OpenCLDevice");
    }
};

// The operator lookup for the OpenCL device. It forwards to the OpenCL
// kernel table in src/opencl/table.cpp; ops without a row resolve to null
// so the executor rejects the node naming op + node.
class OpenCLKernelTable final : public KernelTable {
  public:
    KernelFn find(const std::string& op_type) const override { return opencl_kernels::FindOpenCLKernel(op_type); }
};

class OpenCLDeviceImpl final : public Device {
  public:
    const char* name() const override { return "opencl"; }
    Allocator& allocator() override { return allocator_; }
    Transfer& transfer() override { return transfer_; }
    const KernelTable& kernels() const override { return kernels_; }
    bool is_host() const override { return false; }

  private:
    OpenCLAllocator allocator_;
    OpenCLTransfer transfer_;
    OpenCLKernelTable kernels_;
};

} // namespace

const Device& OpenCLDevice() {
    // Touches the platform at first call (throws when there is none);
    // the reference is stable across calls.
    opencl_runtime::Context();
    static OpenCLDeviceImpl device;
    return device;
}

} // namespace nert
