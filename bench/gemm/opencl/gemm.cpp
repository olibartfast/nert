// Tiled GEMM OpenCL C API implementation for the bench harness (G4).
//
// Kernel algorithm translated (not pasted) from the gpu-playground gemm
// reference; host code is modern C++17 with RAII release of every CL object
// and throwing checks. TILE_SIZE is baked via the build option, as in the
// reference.

#include "opencl/gemm.hpp"

#include "opencl_c_helpers.hpp"

#include <cstddef>
#include <memory>

namespace nert_bench {
namespace {

constexpr int kTile = 16;

const char* kKernelSource = R"(
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

struct ClReleaser {
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program prog = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem buf_a = nullptr;
    cl_mem buf_b = nullptr;
    cl_mem buf_c = nullptr;
    ~ClReleaser() {
        if (buf_c != nullptr) {
            clReleaseMemObject(buf_c);
        }
        if (buf_b != nullptr) {
            clReleaseMemObject(buf_b);
        }
        if (buf_a != nullptr) {
            clReleaseMemObject(buf_a);
        }
        if (kernel != nullptr) {
            clReleaseKernel(kernel);
        }
        if (prog != nullptr) {
            clReleaseProgram(prog);
        }
        if (queue != nullptr) {
            clReleaseCommandQueue(queue);
        }
        if (ctx != nullptr) {
            clReleaseContext(ctx);
        }
    }
};

cl_mem makeBuffer(cl_context ctx, cl_mem_flags flags, std::size_t bytes, const void* host) {
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(ctx, flags | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void*>(host), &err);
    nert_bench::ClCheck(__FILE__, __LINE__, err);
    return buf;
}

} // namespace

void gemmGPU(const float* h_A, const float* h_B, float* h_C, float alpha, float beta, int M, int N, int K) {
    ClReleaser r;
    cl_device_id dev = clSetupGPU(r.ctx, r.queue);
    r.prog = clBuildFromSource(r.ctx, dev, kKernelSource, "-D TILE_SIZE=16");
    cl_int err = CL_SUCCESS;
    r.kernel = clCreateKernel(r.prog, "gemmTiled", &err);
    nert_bench::ClCheck(__FILE__, __LINE__, err);

    const auto bytes_a = static_cast<std::size_t>(M * K) * sizeof(float);
    const auto bytes_b = static_cast<std::size_t>(K * N) * sizeof(float);
    const auto bytes_c = static_cast<std::size_t>(M * N) * sizeof(float);
    r.buf_a = makeBuffer(r.ctx, CL_MEM_READ_ONLY, bytes_a, h_A);
    r.buf_b = makeBuffer(r.ctx, CL_MEM_READ_ONLY, bytes_b, h_B);
    r.buf_c = makeBuffer(r.ctx, CL_MEM_READ_WRITE, bytes_c, h_C);

    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 0, sizeof(cl_mem), &r.buf_a));
    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 1, sizeof(cl_mem), &r.buf_b));
    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 2, sizeof(cl_mem), &r.buf_c));
    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 3, sizeof(float), &alpha));
    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 4, sizeof(float), &beta));
    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 5, sizeof(int), &M));
    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 6, sizeof(int), &N));
    NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 7, sizeof(int), &K));

    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] = {static_cast<std::size_t>((N + kTile - 1) / kTile) * local[0],
                                   static_cast<std::size_t>((M + kTile - 1) / kTile) * local[1]};
    NERT_BENCH_CL_CHECK(clEnqueueNDRangeKernel(r.queue, r.kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
    // Blocking read: the caller's timing captures the full end-to-end cost.
    NERT_BENCH_CL_CHECK(clEnqueueReadBuffer(r.queue, r.buf_c, CL_TRUE, 0, bytes_c, h_C, 0, nullptr, nullptr));
}

} // namespace nert_bench
