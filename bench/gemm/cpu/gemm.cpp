// Bench-local CPU oracle for GEMM. Naive triple loop, float32 data, double
// accumulation with a single rounding per output (mirrors the runtime
// oracle's [D-15] policy). Single-threaded; do not optimize.

#include "cpu/gemm.hpp"

namespace nert_bench {

void gemmCpu(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K) {
    const double alpha_d = static_cast<double>(alpha);
    const double beta_d = static_cast<double>(beta);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            }
            const double c_old = static_cast<double>(C[i * N + j]);
            C[i * N + j] = static_cast<float>(alpha_d * sum + beta_d * c_old);
        }
    }
}

} // namespace nert_bench
