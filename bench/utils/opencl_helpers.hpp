#pragma once
// OpenCL C++ wrapper helpers for the bench harness.
//
// Translated (not pasted) from the gpu-playground reference
// `source/utils/opencl_helpers.h`; see ATTRIBUTION.md. Deltas: `.hpp` name,
// `clppBuildProgram` throws with the build log instead of print+abort, no
// `using namespace` directives.

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace nert_bench {

// First available GPU device. Throws std::runtime_error when none exists.
inline cl::Device clppGetGPUDevice() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        try {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        } catch (const cl::Error&) {
            continue;
        }
        if (!devices.empty()) {
            return devices.front();
        }
    }
    throw std::runtime_error("clppGetGPUDevice: no OpenCL GPU device found");
}

// Compile a program from source; the build log is attached to the thrown
// error (instead of print+abort). Extra compiler options (e.g.
// "-D TILE_SIZE=16") go through `options`.
inline cl::Program clppBuildProgram(const cl::Context& ctx, const cl::Device& dev, const std::string& src,
                                    const char* options = nullptr) {
    cl::Program prog(ctx, src);
    try {
        prog.build({dev}, options);
    } catch (const cl::Error&) {
        const std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
        throw std::runtime_error(std::string("OpenCL build error:\n") + log);
    }
    return prog;
}

// Preferred work-group size aligned to the kernel's preferred multiple,
// clamped to the kernel maximum and the `target` hint (default 256).
[[nodiscard]] inline std::size_t clppPreferredLocalSize(const cl::Kernel& kernel, const cl::Device& dev,
                                                        std::size_t target = 256) {
    const std::size_t preferred_mult = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev);
    const std::size_t max_wgs = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev);
    const std::size_t mult = (preferred_mult == 0) ? 64 : preferred_mult;
    std::size_t local = (target / mult) * mult;
    if (local == 0) {
        local = mult;
    }
    if (max_wgs > 0 && local > max_wgs) {
        local = max_wgs;
    }
    return local;
}

} // namespace nert_bench
