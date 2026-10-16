#pragma once
// Bench device-backend API for GEMM (OpenCL C++ wrapper path, G4).
//
// Same `gemmGPU` contract as the other backends. Throws on setup/build
// failure or when no OpenCL GPU device exists (harness reports SKIP).

#include "cpu/gemm.hpp"

namespace nert_bench {

void gemmGPU(const float* h_A, const float* h_B, float* h_C, float alpha, float beta, int M, int N, int K);

} // namespace nert_bench
