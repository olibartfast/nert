#pragma once
// Bench device-backend API for GEMM (OpenCL C API path, G4).
//
// Same `gemmGPU` contract as the CUDA path: blocking H2D/compute/D2H so the
// harness timing captures end-to-end cost. Throws on setup/build failure or
// when no OpenCL GPU device exists (harness reports SKIP).

#include "cpu/gemm.hpp"

namespace nert_bench {

void gemmGPU(const float* h_A, const float* h_B, float* h_C, float alpha, float beta, int M, int N, int K);

} // namespace nert_bench
