#pragma once
// Bench device-backend API for MatMul (OpenCL C API path).
//
// Same contract as the CUDA path. Throws on setup/build failure or when no
// OpenCL GPU device exists (harness reports SKIP).

#include "cpu/matmul.hpp"

namespace nert_bench {

void matmulGPU(const float* h_A, const float* h_B, float* h_C, int batch, int batch_a, int batch_b, int M, int N,
               int K);

} // namespace nert_bench
