#pragma once
// Bench device-backend API for GEMM (CUDA path).
//
// Declares the host-pointer wrapper the harness calls. In G2 this translation
// unit is a CPU fallback (`gemmGPU` throws); G3 replaces `cuda/gemm.cpp` with
// the tiled CUDA implementation behind the same signature.

#include "cpu/gemm.hpp"

namespace nert_bench {

// Same contract as gemmCpu, executed on the device: copies inputs H2D,
// computes, copies C back D2H (blocking, so the caller's timing captures the
// full end-to-end cost). Throws std::runtime_error when the device backend is
// not enabled or no device is present.
void gemmGPU(const float* h_A, const float* h_B, float* h_C, float alpha, float beta, int M, int N, int K);

} // namespace nert_bench
