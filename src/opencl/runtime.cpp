// Shared OpenCL runtime (see runtime.hpp).

#include "runtime.hpp"

#include "nert/device.hpp"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_runtime {
namespace {

void CheckCl(cl_int status, const char* what, const std::string& context) {
    if (status != CL_SUCCESS) {
        throw InferenceException(context + ": " + what + " failed (OpenCL status " + std::to_string(status) + ")");
    }
}

class ContextHolder {
  public:
    ContextHolder() {
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

    ~ContextHolder() {
        for (const auto& entry : programs_) {
            clReleaseProgram(entry.second);
        }
        for (const auto& entry : kernels_) {
            clReleaseKernel(entry.second);
        }
        if (queue_ != nullptr) {
            clReleaseCommandQueue(queue_);
        }
        if (context_ != nullptr) {
            clReleaseContext(context_);
        }
    }

    ContextHolder(const ContextHolder&) = delete;
    ContextHolder& operator=(const ContextHolder&) = delete;

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }

    cl_kernel kernel_for(const std::string& op, const char* source, const char* name) {
        const auto cached = kernels_.find(op);
        if (cached != kernels_.end()) {
            return cached->second;
        }
        cl_int status = CL_SUCCESS;
        cl_program program = clCreateProgramWithSource(context_, 1, &source, nullptr, &status);
        CheckCl(status, "clCreateProgramWithSource", "OpenCL " + op);
        status = clBuildProgram(program, 1, &device_, "-cl-std=CL3.0", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t logBytes = 0;
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logBytes);
            std::string log(logBytes > 0 ? logBytes : 1, '\0');
            clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logBytes, log.data(), nullptr);
            clReleaseProgram(program);
            throw InferenceException("OpenCL " + op + ": clBuildProgram failed:\n" + log);
        }
        cl_kernel kernel = clCreateKernel(program, name, &status);
        CheckCl(status, "clCreateKernel", "OpenCL " + op);
        programs_[op] = program;
        kernels_[op] = kernel;
        return kernel;
    }

  private:
    cl_context context_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_command_queue queue_ = nullptr;
    std::map<std::string, cl_program> programs_;
    std::map<std::string, cl_kernel> kernels_;
};

ContextHolder& Shared() {
    static ContextHolder holder;
    return holder;
}

} // namespace

cl_context Context() { return Shared().context(); }
cl_command_queue Queue() { return Shared().queue(); }

cl_kernel KernelFor(const std::string& op, const char* source, const char* kernel) {
    return Shared().kernel_for(op, source, kernel);
}

} // namespace opencl_runtime
} // namespace nert
