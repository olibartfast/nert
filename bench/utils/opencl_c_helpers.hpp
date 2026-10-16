#pragma once
// OpenCL C API helpers for the bench harness.
//
// Translated (not pasted) from the gpu-playground reference
// `source/utils/opencl_c_helpers.h`; see ATTRIBUTION.md. Deltas: `.hpp`
// name, diagnostics throw `std::runtime_error` instead of print+abort, no C
// casts, `[[nodiscard]]` where trivial.

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nert_bench {

[[noreturn]] inline void ThrowClError(const char* file, int line, cl_int err) {
    throw std::runtime_error(std::string("OpenCL error ") + std::to_string(static_cast<int>(err)) + " at " + file +
                             ":" + std::to_string(line));
}

inline void ClCheck(const char* file, int line, cl_int err) {
    if (err != CL_SUCCESS) {
        ThrowClError(file, line, err);
    }
}

} // namespace nert_bench

#define NERT_BENCH_CL_CHECK(err) nert_bench::ClCheck(__FILE__, __LINE__, err)

// Backwards-compatible alias for harness code.
#define CL_CHECK NERT_BENCH_CL_CHECK

namespace nert_bench {

// First available GPU device with a context + command queue. Throws
// std::runtime_error when no GPU platform/device exists.
inline cl_device_id clSetupGPU(cl_context& ctx, cl_command_queue& queue) {
    cl_uint num_platforms = 0;
    ClCheck(__FILE__, __LINE__, clGetPlatformIDs(0, nullptr, &num_platforms));
    std::vector<cl_platform_id> platforms(static_cast<std::size_t>(num_platforms));
    ClCheck(__FILE__, __LINE__, clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    cl_device_id dev = nullptr;
    for (const auto platform : platforms) {
        cl_uint num_devices = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS) {
            continue;
        }
        if (num_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(static_cast<std::size_t>(num_devices));
        ClCheck(__FILE__, __LINE__, clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr));
        dev = devices.front();
        break;
    }
    if (dev == nullptr) {
        throw std::runtime_error("clSetupGPU: no OpenCL GPU device found");
    }

    cl_int err = CL_SUCCESS;
    ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    ClCheck(__FILE__, __LINE__, err);
    queue = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &err);
    ClCheck(__FILE__, __LINE__, err);
    return dev;
}

// Build a program from source; the full build log is attached to the thrown
// error when compilation fails.
inline cl_program clBuildFromSource(cl_context ctx, cl_device_id dev, const char* src, const char* options = nullptr) {
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    ClCheck(__FILE__, __LINE__, err);
    if (clBuildProgram(prog, 1, &dev, options, nullptr, nullptr) != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        clReleaseProgram(prog);
        throw std::runtime_error(std::string("OpenCL build error:\n") + log);
    }
    return prog;
}

inline void clTeardown(cl_context ctx, cl_command_queue queue, cl_program prog, cl_kernel kernel) {
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
}

// Preferred work-group size aligned to the kernel's preferred multiple,
// clamped to the kernel maximum and the `target` hint (default 256).
[[nodiscard]] inline std::size_t clPreferredLocalSize(cl_kernel kernel, cl_device_id dev, std::size_t target = 256) {
    std::size_t preferred_mult = 0;
    std::size_t max_wgs = 0;
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(std::size_t),
                             &preferred_mult, nullptr);
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(std::size_t), &max_wgs, nullptr);
    if (preferred_mult == 0) {
        preferred_mult = 64;
    }
    std::size_t local = (target / preferred_mult) * preferred_mult;
    if (local == 0) {
        local = preferred_mult;
    }
    if (max_wgs > 0 && local > max_wgs) {
        local = max_wgs;
    }
    return local;
}

} // namespace nert_bench
