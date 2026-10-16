// Tiled MatMul CUDA implementation for the bench harness.
//
// C = A @ B per batch item (plain, no alpha/beta). Batches loop on the host
// with one 2D tiled launch each; the 2D tile is the same TILE 16 pattern as
// bench/gemm (kept duplicated per [D-1]: the shared kernel header arrives
// with the N1 consumption redesign). Compiled `LANGUAGE CUDA`.

#include "cuda/matmul.hpp"

#include "cuda_helpers.hpp"

#include <cstddef>
#include <memory>

namespace nert_bench {
namespace {

constexpr int kTile = 16;

__global__ void matmulTiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float as[kTile][kTile];
    __shared__ float bs[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    const int num_tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTile + tx;
        as[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        const int b_row = t * kTile + ty;
        bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum += as[ty][k] * bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

struct CudaFree {
    void operator()(void* p) const noexcept {
        if (p != nullptr) {
            cudaFree(p);
        }
    }
};

using DevicePtr = std::unique_ptr<void, CudaFree>;

DevicePtr allocBytes(std::size_t bytes) {
    void* p = nullptr;
    NERT_BENCH_CUDA_CHECK(cudaMalloc(&p, bytes));
    return DevicePtr(p);
}

} // namespace

void matmulGPU(const float* h_A, const float* h_B, float* h_C, int batch, int batch_a, int batch_b, int M, int N,
               int K) {
    const auto item_a = static_cast<std::size_t>(M * K);
    const auto item_b = static_cast<std::size_t>(K * N);
    const auto item_c = static_cast<std::size_t>(M * N);
    const DevicePtr d_a = allocBytes(item_a * sizeof(float));
    const DevicePtr d_b = allocBytes(item_b * sizeof(float));
    const DevicePtr d_c = allocBytes(item_c * sizeof(float));

    const dim3 block(static_cast<unsigned>(kTile), static_cast<unsigned>(kTile));
    const dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile), static_cast<unsigned>((M + kTile - 1) / kTile));
    for (int b = 0; b < batch; ++b) {
        const float* a = h_A + (batch_a == 1 ? 0 : static_cast<std::size_t>(b) * item_a);
        const float* bh = h_B + (batch_b == 1 ? 0 : static_cast<std::size_t>(b) * item_b);
        float* c = h_C + static_cast<std::size_t>(b) * item_c;
        NERT_BENCH_CUDA_CHECK(cudaMemcpy(d_a.get(), a, item_a * sizeof(float), cudaMemcpyHostToDevice));
        NERT_BENCH_CUDA_CHECK(cudaMemcpy(d_b.get(), bh, item_b * sizeof(float), cudaMemcpyHostToDevice));
        matmulTiled<<<grid, block>>>(static_cast<const float*>(d_a.get()), static_cast<const float*>(d_b.get()),
                                     static_cast<float*>(d_c.get()), M, N, K);
        NERT_BENCH_CUDA_CHECK(cudaGetLastError());
        NERT_BENCH_CUDA_CHECK(cudaMemcpy(c, d_c.get(), item_c * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

} // namespace nert_bench
