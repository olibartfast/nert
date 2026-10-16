#pragma once
// Bench-local CPU oracle for MatMul: C = A @ B with broadcast batch dims.
//
// Shapes: A [..., M, K], B [..., K, N], C [..., M, N] where the batch part
// broadcasts (`batch_a`/`batch_b` of 1 reuses the single input batch).
// Duplicated from the runtime semantics instead of including the
// engine-private `Kernels.hpp` ([D-2] of the base packet). Double
// accumulation, one rounding per output; intentionally naive.

namespace nert_bench {

void matmulCpu(const float* A, const float* B, float* C, int batch, int batch_a, int batch_b, int M, int N, int K);

} // namespace nert_bench
