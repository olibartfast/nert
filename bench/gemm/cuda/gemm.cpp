// Tiled GEMM CUDA implementation for the bench harness (G3).
//
// Computes C = alpha * A * B + beta * C, A [MxK], B [KxN], C [MxN], FP32.
// Algorithm translated (not pasted) from the gpu-playground gemm reference;
// host code is modern C++17 (RAII device buffers, throwing checks).
// Compiled `LANGUAGE CUDA` (see bench/gemm/CMakeLists.txt) so the root
// project keeps its CMake 3.16 / C++-only floor.

#include "cuda/gemm.hpp"

#include "cuda_helpers.hpp"

#include <cstddef>
#include <memory>

namespace nert_bench {
namespace {

constexpr int kTile = 16;

__global__ void gemmTiled(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K) {
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
        const float c_old = C[row * N + col];
        C[row * N + col] = alpha * sum + beta * c_old;
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

void gemmGPU(const float* h_A, const float* h_B, float* h_C, float alpha, float beta, int M, int N, int K) {
    const auto bytes_a = static_cast<std::size_t>(M * K) * sizeof(float);
    const auto bytes_b = static_cast<std::size_t>(K * N) * sizeof(float);
    const auto bytes_c = static_cast<std::size_t>(M * N) * sizeof(float);
    const DevicePtr d_a = allocBytes(bytes_a);
    const DevicePtr d_b = allocBytes(bytes_b);
    const DevicePtr d_c = allocBytes(bytes_c);

    NERT_BENCH_CUDA_CHECK(cudaMemcpy(d_a.get(), h_A, bytes_a, cudaMemcpyHostToDevice));
    NERT_BENCH_CUDA_CHECK(cudaMemcpy(d_b.get(), h_B, bytes_b, cudaMemcpyHostToDevice));
    NERT_BENCH_CUDA_CHECK(cudaMemcpy(d_c.get(), h_C, bytes_c, cudaMemcpyHostToDevice));

    const dim3 block(static_cast<unsigned>(kTile), static_cast<unsigned>(kTile));
    const dim3 grid(static_cast<unsigned>((N + kTile - 1) / kTile), static_cast<unsigned>((M + kTile - 1) / kTile));
    gemmTiled<<<grid, block>>>(static_cast<const float*>(d_a.get()), static_cast<const float*>(d_b.get()),
                               static_cast<float*>(d_c.get()), alpha, beta, M, N, K);
    NERT_BENCH_CUDA_CHECK(cudaGetLastError());
    // Blocking copy: the caller's end-to-end timing captures the full cost.
    NERT_BENCH_CUDA_CHECK(cudaMemcpy(h_C, d_c.get(), bytes_c, cudaMemcpyDeviceToHost));
}

} // namespace nert_bench
