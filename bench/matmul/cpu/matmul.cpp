// Bench-local CPU oracle for MatMul. Naive triple loop per batch item,
// float32 data, double accumulation with a single rounding per output.
// Single-threaded; do not optimize.

#include "cpu/matmul.hpp"

#include <cstddef>

namespace nert_bench {

void matmulCpu(const float* A, const float* B, float* C, int batch, int batch_a, int batch_b, int M, int N, int K) {
    const auto item_a = static_cast<std::size_t>(M * K);
    const auto item_b = static_cast<std::size_t>(K * N);
    const auto item_c = static_cast<std::size_t>(M * N);
    for (int b = 0; b < batch; ++b) {
        const float* a = A + (batch_a == 1 ? 0 : static_cast<std::size_t>(b) * item_a);
        const float* bh = B + (batch_b == 1 ? 0 : static_cast<std::size_t>(b) * item_b);
        float* c = C + static_cast<std::size_t>(b) * item_c;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < K; ++k) {
                    sum += static_cast<double>(a[i * K + k]) * static_cast<double>(bh[k * N + j]);
                }
                c[i * N + j] = static_cast<float>(sum);
            }
        }
    }
}

} // namespace nert_bench
