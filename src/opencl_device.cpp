// The OpenCL device: allocator, transfers, and an (M1-empty) kernel table
// behind the per-device seam.
//
// N1d portable backend, OpenCL-first (Vulkan needs a shader toolchain this
// box lacks; see the N1d packet). Raw OpenCL C API only; RAII locally.
// Construction selects the first GPU on the first platform that offers one
// and throws naming the cause when there is none — presence probes catch
// that and SKIP.

#include "nert/device.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace {

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
    }
}

class OpenCLContext {
  public:
    OpenCLContext() {
        cl_uint platforms = 0;
        if (clGetPlatformIDs(0, nullptr, &platforms) != CL_SUCCESS || platforms == 0) {
            throw InferenceException("OpenCLDevice: no OpenCL platform found");
        }
        std::vector<cl_platform_id> ids(platforms);
        CheckCl(clGetPlatformIDs(platforms, ids.data(), nullptr), "clGetPlatformIDs", "OpenCLDevice");
        for (cl_platform_id platform : ids) {
            cl_uint devices = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices) != CL_SUCCESS || devices == 0) {
                continue;
            }
            std::vector<cl_device_id> list(devices);
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, devices, list.data(), nullptr) != CL_SUCCESS) {
                continue;
            }
            cl_int status = CL_SUCCESS;
            context_ = clCreateContext(nullptr, 1, &list[0], nullptr, nullptr, &status);
            if (status != CL_SUCCESS) {
                continue;
            }
            device_ = list[0];
            queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &status);
            if (status != CL_SUCCESS) {
                clReleaseContext(context_);
                context_ = nullptr;
                continue;
            }
            return;
        }
        throw InferenceException("OpenCLDevice: no OpenCL GPU device found");
    }

    ~OpenCLContext() {
        if (queue_ != nullptr) {
            clReleaseCommandQueue(queue_);
        }
        if (context_ != nullptr) {
            clReleaseContext(context_);
        }
    }

    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }

  private:
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue queue_ = nullptr;
};

OpenCLContext& SharedContext() {
    static OpenCLContext context;
    return context;
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
        cl_mem buffer = clCreateBuffer(SharedContext().context(), CL_MEM_READ_WRITE, static_cast<std::size_t>(bytes),
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
        CheckCl(clEnqueueWriteBuffer(SharedContext().queue(), reinterpret_cast<cl_mem>(dst), CL_TRUE, 0,
                                     static_cast<std::size_t>(bytes), src, 0, nullptr, nullptr),
                "clEnqueueWriteBuffer", "OpenCLDevice");
    }

    // Copies `bytes` bytes from device memory back to the host (blocking).
    void device_to_host(void* dst, const void* src, int64_t bytes) override {
        if (bytes == 0) {
            return;
        }
        CheckCl(clEnqueueReadBuffer(SharedContext().queue(), reinterpret_cast<cl_mem>(const_cast<void*>(src)), CL_TRUE,
                                    0, static_cast<std::size_t>(bytes), dst, 0, nullptr, nullptr),
                "clEnqueueReadBuffer", "OpenCLDevice");
    }
};

// The operator lookup for the OpenCL device. M1: empty — every op resolves
// to null so the executor rejects the node naming op + node. M2+ rows land
// in src/opencl/.
class OpenCLKernelTable final : public KernelTable {
  public:
    KernelFn find(const std::string& /*op_type*/) const override { return nullptr; }
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
    SharedContext();
    static OpenCLDeviceImpl device;
    return device;
}

} // namespace nert
