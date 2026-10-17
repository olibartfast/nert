// Shared OpenCL runtime: context/queue access plus a per-op program cache.
//
// Kernels are embedded OpenCL C sources compiled once per process by
// clBuildProgram; a build failure throws naming the op with the compiler
// log. Single-threaded executor use only (no locking, like the tables).

#pragma once

#include <string>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace nert {
namespace opencl_runtime {

// The process-lifetime context + queue (throws naming the cause when no
// OpenCL GPU platform exists).
cl_context Context();
cl_command_queue Queue();

// The named kernel from the given source, compiled and cached per op.
cl_kernel KernelFor(const std::string& op, const char* source, const char* kernel);

} // namespace opencl_runtime
} // namespace nert
