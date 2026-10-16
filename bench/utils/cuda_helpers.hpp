#pragma once
// CUDA error-checking + wall-clock helpers for the bench harness.
//
// Translated (not pasted) from the gpu-playground reference
// `source/utils/cuda_helpers.h`; see ATTRIBUTION.md. Deltas: `.hpp` name,
// `<iostream>`-free diagnostics via exceptions, `CUDA_CHECK` throws
// `std::runtime_error` with file/line instead of print+abort, `getTime()`
// uses `<chrono>` instead of `gettimeofday`.

#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace nert_bench {

[[noreturn]] inline void ThrowCudaError(const char* call, const char* file, int line, cudaError_t err) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) + ": " + call + ": " +
                             cudaGetErrorString(err));
}

} // namespace nert_bench

#define NERT_BENCH_CUDA_CHECK(call)                                                                                    \
    do {                                                                                                               \
        const cudaError_t nert_bench_err__ = (call);                                                                   \
        if (nert_bench_err__ != cudaSuccess) {                                                                         \
            nert_bench::ThrowCudaError(#call, __FILE__, __LINE__, nert_bench_err__);                                   \
        }                                                                                                              \
    } while (0)

// Backwards-compatible alias for harness code.
#define CUDA_CHECK NERT_BENCH_CUDA_CHECK

namespace nert_bench {

// Wall-clock seconds, steady clock. Replaces the reference `getTime()`.
[[nodiscard]] inline double getTime() noexcept {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

} // namespace nert_bench
