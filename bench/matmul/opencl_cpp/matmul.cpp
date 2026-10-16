// Tiled MatMul OpenCL C++ wrapper implementation for the bench harness.
//
// Same per-batch host loop as the C API path; `cl::` RAII removes the manual
// bookkeeping. Kernel duplicated per [D-1] (shared header arrives with the
// N1 redesign).

#include "opencl_cpp/matmul.hpp"

#include "opencl_helpers.hpp"

#include <cstddef>
#include <string>

namespace nert_bench {
namespace {

constexpr int kTile = 16;

const std::string kKernelSource = R"(
__kernel void matmulTiled(__global const float* A, __global const float* B, __global float* C, int M, int N, int K) {
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
        C[row * N + col] = sum;
    }
}
)";

} // namespace

void matmulGPU(const float* h_A, const float* h_B, float* h_C, int batch, int batch_a, int batch_b, int M, int N,
               int K) {
    const cl::Device dev = clppGetGPUDevice();
    const cl::Context ctx(dev);
    const cl::Program prog = clppBuildProgram(ctx, dev, kKernelSource, "-D TILE_SIZE=16");
    cl::Kernel kernel(prog, "matmulTiled");
    cl::CommandQueue queue(ctx, dev);

    const auto item_a = static_cast<std::size_t>(M * K);
    const auto item_b = static_cast<std::size_t>(K * N);
    const auto item_c = static_cast<std::size_t>(M * N);
    const cl::Buffer buf_a(ctx, CL_MEM_READ_ONLY, item_a * sizeof(float));
    const cl::Buffer buf_b(ctx, CL_MEM_READ_ONLY, item_b * sizeof(float));
    const cl::Buffer buf_c(ctx, CL_MEM_READ_WRITE, item_c * sizeof(float));

    const cl::NDRange local(kTile, kTile);
    const cl::NDRange global(static_cast<std::size_t>((N + kTile - 1) / kTile) * kTile,
                             static_cast<std::size_t>((M + kTile - 1) / kTile) * kTile);
    for (int b = 0; b < batch; ++b) {
        const float* a = h_A + (batch_a == 1 ? 0 : static_cast<std::size_t>(b) * item_a);
        const float* bh = h_B + (batch_b == 1 ? 0 : static_cast<std::size_t>(b) * item_b);
        float* c = h_C + static_cast<std::size_t>(b) * item_c;
        queue.enqueueWriteBuffer(buf_a, CL_TRUE, 0, item_a * sizeof(float), a);
        queue.enqueueWriteBuffer(buf_b, CL_TRUE, 0, item_b * sizeof(float), bh);
        kernel.setArg(0, buf_a);
        kernel.setArg(1, buf_b);
        kernel.setArg(2, buf_c);
        kernel.setArg(3, M);
        kernel.setArg(4, N);
        kernel.setArg(5, K);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.enqueueReadBuffer(buf_c, CL_TRUE, 0, item_c * sizeof(float), c);
    }
}

} // namespace nert_bench
