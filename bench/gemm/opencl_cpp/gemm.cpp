// Tiled GEMM OpenCL C++ wrapper implementation for the bench harness (G4).
//
// Same kernel algorithm as the C API path; the C++ wrapper (`cl::Buffer`,
// `cl::KernelFunctor` via `setArg` + `cl::EnqueueArgs`) removes the manual
// retain/release bookkeeping — `cl::` RAII handles it. Exceptions
// (`cl::Error`, `std::runtime_error`) propagate to the harness SKIP path.

#include "opencl_cpp/gemm.hpp"

#include "opencl_helpers.hpp"

#include <cstddef>
#include <vector>

namespace nert_bench {
namespace {

constexpr int kTile = 16;

const std::string kKernelSource = R"(
__kernel void gemmTiled(__global const float* A, __global const float* B, __global float* C, float alpha, float beta,
                        int M, int N, int K) {
    __local float As[TILE_SIZE][TILE_SIZE];
    __local float Bs[TILE_SIZE][TILE_SIZE];

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int row = get_group_id(1) * TILE_SIZE + ty;
    int col = get_group_id(0) * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        int bRow = t * TILE_SIZE + ty;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        float c_old = C[row * N + col];
        C[row * N + col] = alpha * sum + beta * c_old;
    }
}
)";

} // namespace

void gemmGPU(const float* h_A, const float* h_B, float* h_C, float alpha, float beta, int M, int N, int K) {
    const cl::Device dev = clppGetGPUDevice();
    const cl::Context ctx(dev);
    const cl::Program prog = clppBuildProgram(ctx, dev, kKernelSource, "-D TILE_SIZE=16");
    cl::Kernel kernel(prog, "gemmTiled");
    cl::CommandQueue queue(ctx, dev);

    const auto bytes_a = static_cast<std::size_t>(M * K) * sizeof(float);
    const auto bytes_b = static_cast<std::size_t>(K * N) * sizeof(float);
    const auto bytes_c = static_cast<std::size_t>(M * N) * sizeof(float);
    const cl::Buffer buf_a(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_a, const_cast<float*>(h_A));
    const cl::Buffer buf_b(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_b, const_cast<float*>(h_B));
    const cl::Buffer buf_c(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes_c, h_C);

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_b);
    kernel.setArg(2, buf_c);
    kernel.setArg(3, alpha);
    kernel.setArg(4, beta);
    kernel.setArg(5, M);
    kernel.setArg(6, N);
    kernel.setArg(7, K);

    const cl::NDRange local(kTile, kTile);
    const cl::NDRange global(static_cast<std::size_t>((N + kTile - 1) / kTile) * kTile,
                             static_cast<std::size_t>((M + kTile - 1) / kTile) * kTile);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    // Blocking read: the caller's timing captures the full end-to-end cost.
    queue.enqueueReadBuffer(buf_c, CL_TRUE, 0, bytes_c, h_C);
}

} // namespace nert_bench
