#pragma once
// Bench device-backend API for MatMul (CUDA path).
//
// Same contract as matmulCpu, executed on the device: blocking H2D/compute/
// D2H per batch item so the caller's timing captures end-to-end cost. Throws
// std::runtime_error when no device is present (harness reports SKIP).

#include "cpu/matmul.hpp"

namespace nert_bench {

void matmulGPU(const float* h_A, const float* h_B, float* h_C, int batch, int batch_a, int batch_b, int M, int N,
               int K);

} // namespace nert_bench
