#pragma once
// Bench-local CPU oracle for GEMM: C = alpha * A * B + beta * C (FP32).
//
// Deliberately duplicated from the runtime `src/kernels/Gemm.cpp` instead of
// including the engine-private `Kernels.hpp`: the bench harness must not
// depend on engine internals ([D-2]). Like the runtime oracle it accumulates
// in double and rounds once per output; it is intentionally naive and is the
// correctness reference device kernels are measured against, never a
// performance claim.

namespace nert_bench {

void gemmCpu(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K);

} // namespace nert_bench
