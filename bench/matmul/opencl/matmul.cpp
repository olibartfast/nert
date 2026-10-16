// Tiled MatMul OpenCL C API implementation for the bench harness.
//
// Same per-batch host loop as the CUDA path; kernel duplicated per [D-1]
// (shared header arrives with the N1 redesign). RAII release of every CL
// object, throwing checks.

#include "opencl/matmul.hpp"

#include "opencl_c_helpers.hpp"

#include <cstddef>

namespace nert_bench {
namespace {

constexpr int kTile = 16;

const char* kKernelSource = R"(
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

void matmulGPU(const float* h_A, const float* h_B, float* h_C, int batch, int batch_a, int batch_b, int M, int N,
               int K) {
    ClReleaser r;
    cl_device_id dev = clSetupGPU(r.ctx, r.queue);
    r.prog = clBuildFromSource(r.ctx, dev, kKernelSource, "-D TILE_SIZE=16");
    cl_int err = CL_SUCCESS;
    r.kernel = clCreateKernel(r.prog, "matmulTiled", &err);
    nert_bench::ClCheck(__FILE__, __LINE__, err);

    const auto item_a = static_cast<std::size_t>(M * K);
    const auto item_b = static_cast<std::size_t>(K * N);
    const auto item_c = static_cast<std::size_t>(M * N);
    r.buf_a = makeBuffer(r.ctx, CL_MEM_READ_ONLY, item_a * sizeof(float), h_A);
    r.buf_b = makeBuffer(r.ctx, CL_MEM_READ_ONLY, item_b * sizeof(float), h_B);
    r.buf_c = makeBuffer(r.ctx, CL_MEM_READ_WRITE, item_c * sizeof(float), h_C);

    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] = {static_cast<std::size_t>((N + kTile - 1) / kTile) * local[0],
                                   static_cast<std::size_t>((M + kTile - 1) / kTile) * local[1]};
    for (int b = 0; b < batch; ++b) {
        const float* a = h_A + (batch_a == 1 ? 0 : static_cast<std::size_t>(b) * item_a);
        const float* bh = h_B + (batch_b == 1 ? 0 : static_cast<std::size_t>(b) * item_b);
        float* c = h_C + static_cast<std::size_t>(b) * item_c;
        NERT_BENCH_CL_CHECK(
            clEnqueueWriteBuffer(r.queue, r.buf_a, CL_TRUE, 0, item_a * sizeof(float), a, 0, nullptr, nullptr));
        NERT_BENCH_CL_CHECK(
            clEnqueueWriteBuffer(r.queue, r.buf_b, CL_TRUE, 0, item_b * sizeof(float), bh, 0, nullptr, nullptr));
        NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 0, sizeof(cl_mem), &r.buf_a));
        NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 1, sizeof(cl_mem), &r.buf_b));
        NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 2, sizeof(cl_mem), &r.buf_c));
        NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 3, sizeof(int), &M));
        NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 4, sizeof(int), &N));
        NERT_BENCH_CL_CHECK(clSetKernelArg(r.kernel, 5, sizeof(int), &K));
        NERT_BENCH_CL_CHECK(clEnqueueNDRangeKernel(r.queue, r.kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
        NERT_BENCH_CL_CHECK(
            clEnqueueReadBuffer(r.queue, r.buf_c, CL_TRUE, 0, item_c * sizeof(float), c, 0, nullptr, nullptr));
    }
}

} // namespace nert_bench
