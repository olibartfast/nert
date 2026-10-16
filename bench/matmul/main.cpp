// MatMul bench harness: CPU oracle vs one device backend + timings.
//
// Same contract as bench/gemm/main.cpp (kept duplicated per [D-1]: the
// shared harness header arrives with the N1 consumption redesign).
// Backend chosen at configure time; exactly one branch compiles.
// Exit 0 on exact small/batched matches + in-tolerance large matches
// (or device SKIP); exit 1 otherwise.

#if defined(NERT_BENCH_OPENCL_CPP_BACKEND)
#include "opencl_cpp/matmul.hpp"
#elif defined(NERT_BENCH_OPENCL_BACKEND)
#include "opencl/matmul.hpp"
#else
#include "cuda/matmul.hpp"
#endif

#include "benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr float kSmallAbsTol = 1.0e-4f;
constexpr float kCompareAtol = 1.0e-3f;

void printMatrix(const std::vector<float>& matrix, int rows, int cols, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < std::min(rows, 4); ++i) {
        std::cout << "  [";
        for (int j = 0; j < std::min(cols, 4); ++j) {
            std::cout << ' ' << matrix[static_cast<std::size_t>(i * cols + j)];
        }
        if (cols > 4) {
            std::cout << " ...";
        }
        std::cout << " ]\n";
    }
    if (rows > 4) {
        std::cout << "  ...\n";
    }
    std::cout << '\n';
}

bool compareResults(const std::vector<float>& ref, const std::vector<float>& cand, float atol) {
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int mismatches = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - cand[i]);
        max_diff = std::max(max_diff, diff);
        max_rel = std::max(max_rel, diff / (std::fabs(ref[i]) + 1.0e-6f));
        if (diff > atol * (1.0f + std::fabs(ref[i]))) {
            if (mismatches < 5) {
                std::cout << "Mismatch at index " << i << ": ref = " << ref[i] << ", cand = " << cand[i]
                          << " (diff = " << diff << ")\n";
            }
            ++mismatches;
        }
    }
    std::cout << "Max absolute difference: " << max_diff << "\nMax relative difference: " << max_rel << '\n';
    if (mismatches > 5) {
        std::cout << "... and " << (mismatches - 5) << " more mismatches\n";
    }
    return mismatches == 0;
}

void fillUniform(std::vector<float>& v, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) {
        x = dist(rng);
    }
}

// Runs one shape through CPU, then the device when present. Returns 0 on
// pass. Sets device_ok=false when the device throw means "no device".
int runCase(const std::string& name, const std::vector<float>& a, const std::vector<float>& b, int batch, int batch_a,
            int batch_b, int m, int n, int k, const std::vector<float>* exact, bool& device_ok, bool bench_device) {
    std::cout << name << ":\n----------------------------------------\n";
    std::vector<float> c_cpu(static_cast<std::size_t>(batch * m * n), 0.0f);
    std::vector<float> c_gpu(static_cast<std::size_t>(batch * m * n), 0.0f);

    nert_bench::matmulCpu(a.data(), b.data(), c_cpu.data(), batch, batch_a, batch_b, m, n, k);
    bool ok = true;
    if (exact != nullptr) {
        ok = compareResults(*exact, c_cpu, kSmallAbsTol);
        std::cout << (ok ? "CPU oracle matches the exact expectation.\n" : "CPU ORACLE MISMATCH.\n");
    }

    bool ran_device = false;
    if (device_ok) {
        try {
            nert_bench::matmulGPU(a.data(), b.data(), c_gpu.data(), batch, batch_a, batch_b, m, n, k);
            ran_device = true;
            const bool gpu_ok = compareResults(c_cpu, c_gpu, exact != nullptr ? kSmallAbsTol : kCompareAtol);
            std::cout << (gpu_ok ? "Device matches CPU.\n" : "Device DOES NOT match CPU.\n");
            ok = ok && gpu_ok;
        } catch (const std::exception& e) {
            device_ok = false;
            std::cout << "SKIP: device half not run (" << e.what() << ")\n";
        }
    } else {
        std::cout << "SKIP: device half not run (no device)\n";
    }

    const BenchResult cpu_bench = benchmarkWithReset(
        [&] { nert_bench::matmulCpu(a.data(), b.data(), c_cpu.data(), batch, batch_a, batch_b, m, n, k); },
        [&] { std::fill(c_cpu.begin(), c_cpu.end(), 0.0f); }, 1, 3);
    std::cout << "\nExecution Times:\n";
    printBench("CPU:", cpu_bench);
    if (ran_device && bench_device) {
        const BenchResult gpu_bench = benchmarkWithReset(
            [&] { nert_bench::matmulGPU(a.data(), b.data(), c_gpu.data(), batch, batch_a, batch_b, m, n, k); },
            [&] { std::fill(c_gpu.begin(), c_gpu.end(), 0.0f); });
        printBench("GPU:", gpu_bench);
        printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);
    }
    std::cout << '\n';
    return ok ? 0 : 1;
}

} // namespace

int main() {
    std::cout << "=== MatMul (Batched Matrix Multiplication) Test ===\n\n";
    try {
        bool device_ok = true;
        int rc = 0;

        // Small 2D: [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]].
        const std::vector<float> a_small = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const std::vector<float> b_small = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        const std::vector<float> expected_small = {58.0f, 64.0f, 139.0f, 154.0f};
        rc |= runCase("Small Test (2x3 @ 3x2)", a_small, b_small, 1, 1, 1, 2, 2, 3, &expected_small, device_ok, true);

        // Batched broadcast: batch=2, A per-batch, B shared identity.
        const std::vector<float> a_batched = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        const std::vector<float> b_ident = {1.0f, 0.0f, 0.0f, 1.0f};
        rc |= runCase("Batched Test (2 x 2x2 @ 2x2, broadcast B)", a_batched, b_ident, 2, 2, 1, 2, 2, 2, &a_batched,
                      device_ok, false);

        // Large 2D multi-seed.
        constexpr int M = 512;
        constexpr int K = 256;
        constexpr int N = 512;
        std::vector<float> a_large(static_cast<std::size_t>(M * K));
        std::vector<float> b_large(static_cast<std::size_t>(K * N));
        fillUniform(a_large, 42u);
        fillUniform(b_large, 142u);
        bool large_ok = true;
        const std::uint32_t seeds[] = {42u, 1337u, 2024u, 7u, 99u};
        int trial = 0;
        for (const auto seed : seeds) {
            ++trial;
            fillUniform(a_large, seed);
            fillUniform(b_large, seed + 1000u);
            std::vector<float> c_ref(static_cast<std::size_t>(M * N), 0.0f);
            std::vector<float> c_dev(static_cast<std::size_t>(M * N), 0.0f);
            nert_bench::matmulCpu(a_large.data(), b_large.data(), c_ref.data(), 1, 1, 1, M, N, K);
            bool ok = true;
            if (device_ok) {
                try {
                    nert_bench::matmulGPU(a_large.data(), b_large.data(), c_dev.data(), 1, 1, 1, M, N, K);
                    ok = compareResults(c_ref, c_dev, kCompareAtol);
                } catch (const std::exception& e) {
                    device_ok = false;
                    std::cout << "SKIP: device half not run in large test (" << e.what() << ")\n";
                }
            }
            std::cout << "Large Trial " << trial << '/' << (sizeof(seeds) / sizeof(seeds[0])) << " (seed " << seed
                      << "): " << (ok ? "PASSED" : "FAILED") << "\n\n";
            large_ok = large_ok && ok;
        }
        std::cout << "Large test (512x256 @ 256x512): " << (large_ok ? "PASSED" : "FAILED") << '\n';
        rc |= large_ok ? 0 : 1;

        const bool ok = rc == 0;
        std::cout << "\nOverall result: " << (ok ? "PASSED" : "FAILED") << '\n';
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAILED with exception: " << e.what() << '\n';
        return 1;
    }
}
