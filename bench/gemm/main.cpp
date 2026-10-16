// GEMM bench harness: CPU oracle vs one device backend + timings.
//
// Backend is chosen at configure time; exactly one branch compiles:
//   NERT_BENCH_OPENCL_CPP_BACKEND -> opencl_cpp/gemm.hpp (G4)
//   NERT_BENCH_OPENCL_BACKEND     -> opencl/gemm.hpp      (G4)
//   default                       -> cuda/gemm.hpp         (G2 fallback throws, G3 real CUDA)
//
// Exit 0 when the CPU oracle matches the exact small-test expectation and
// every device comparison either matches within tolerance or reports SKIP
// (no usable device); exit 1 otherwise. Modern C++17: vectors, iostream,
// <random>, C++ casts; no printf, no new/delete.

#if defined(NERT_BENCH_OPENCL_CPP_BACKEND)
#include "opencl_cpp/gemm.hpp"
#elif defined(NERT_BENCH_OPENCL_BACKEND)
#include "opencl/gemm.hpp"
#else
#include "cuda/gemm.hpp"
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
    const int show_rows = std::min(rows, 4);
    const int show_cols = std::min(cols, 4);
    for (int i = 0; i < show_rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < show_cols; ++j) {
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

// Scaled elementwise comparison: |a-b| <= atol * (1 + |ref|).
bool compareResults(const std::vector<float>& ref, const std::vector<float>& cand, float atol) {
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int mismatches = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - cand[i]);
        const float rel = diff / (std::fabs(ref[i]) + 1.0e-6f);
        max_diff = std::max(max_diff, diff);
        max_rel = std::max(max_rel, rel);
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

int runSmall(bool& device_ok) {
    std::cout << "Small Test (2x3 * 3x2):\n----------------------------------------\n";
    constexpr int M = 2;
    constexpr int K = 3;
    constexpr int N = 2;
    constexpr float kAlpha = 1.0f;
    constexpr float kBeta = 0.0f;
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> b = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    const std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};
    std::vector<float> c_cpu(M * N, 0.0f);
    std::vector<float> c_gpu(M * N, 0.0f);

    printMatrix(a, M, K, "Matrix A (2x3)");
    printMatrix(b, K, N, "Matrix B (3x2)");

    nert_bench::gemmCpu(a.data(), b.data(), c_cpu.data(), kAlpha, kBeta, M, N, K);
    printMatrix(c_cpu, M, N, "Matrix C (CPU Result)");
    const bool cpu_ok = compareResults(expected, c_cpu, kSmallAbsTol);
    std::cout << (cpu_ok ? "CPU oracle matches the exact expectation.\n" : "CPU ORACLE MISMATCH.\n");

    bool gpu_ok = false;
    try {
        nert_bench::gemmGPU(a.data(), b.data(), c_gpu.data(), kAlpha, kBeta, M, N, K);
        device_ok = true;
        printMatrix(c_gpu, M, N, "Matrix C (GPU Result)");
        gpu_ok = compareResults(c_cpu, c_gpu, kCompareAtol);
        std::cout << (gpu_ok ? "Device matches CPU.\n" : "Device DOES NOT match CPU.\n");
    } catch (const std::exception& e) {
        device_ok = false;
        std::cout << "SKIP: device half not run (" << e.what() << ")\n";
    }

    const BenchResult cpu_bench =
        benchmarkWithReset([&] { nert_bench::gemmCpu(a.data(), b.data(), c_cpu.data(), kAlpha, kBeta, M, N, K); },
                           [&] { std::fill(c_cpu.begin(), c_cpu.end(), 0.0f); });
    std::cout << "\nExecution Times:\n";
    printBench("CPU:", cpu_bench);
    if (device_ok) {
        const BenchResult gpu_bench =
            benchmarkWithReset([&] { nert_bench::gemmGPU(a.data(), b.data(), c_gpu.data(), kAlpha, kBeta, M, N, K); },
                               [&] { std::fill(c_gpu.begin(), c_gpu.end(), 0.0f); });
        printBench("GPU:", gpu_bench);
        printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);
    }
    std::cout << "\n========================================\n\n";
    return (cpu_ok && (gpu_ok || !device_ok)) ? 0 : 1;
}

int runLarge(bool device_ok) {
    std::cout << "Large Test (512x256 * 256x512):\n----------------------------------------\n";
    constexpr int M = 512;
    constexpr int K = 256;
    constexpr int N = 512;
    constexpr float kAlpha = 1.5f;
    constexpr float kBeta = 0.5f;
    std::cout << "M=" << M << ", K=" << K << ", N=" << N << "\na = " << kAlpha << ", b = " << kBeta << "\n\n";

    std::vector<float> a(static_cast<std::size_t>(M * K));
    std::vector<float> b(static_cast<std::size_t>(K * N));
    std::vector<float> c_cpu(static_cast<std::size_t>(M * N));
    std::vector<float> c_gpu(static_cast<std::size_t>(M * N));
    std::vector<float> c_seed(static_cast<std::size_t>(M * N));

    const std::uint32_t seeds[] = {42u, 1337u, 2024u, 7u, 99u};
    bool large_ok = true;
    int trial = 0;
    for (const auto seed : seeds) {
        ++trial;
        fillUniform(a, seed);
        fillUniform(b, seed + 1000u);
        fillUniform(c_seed, seed + 2000u);
        c_cpu = c_seed;
        c_gpu = c_seed;
        nert_bench::gemmCpu(a.data(), b.data(), c_cpu.data(), kAlpha, kBeta, M, N, K);
        bool ok = true;
        if (device_ok) {
            try {
                nert_bench::gemmGPU(a.data(), b.data(), c_gpu.data(), kAlpha, kBeta, M, N, K);
                ok = compareResults(c_cpu, c_gpu, kCompareAtol);
            } catch (const std::exception& e) {
                std::cout << "SKIP: device half not run in large test (" << e.what() << ")\n";
                device_ok = false;
            }
        }
        std::cout << "Trial " << trial << '/' << (sizeof(seeds) / sizeof(seeds[0])) << " (seed " << seed
                  << "): " << (ok ? "PASSED" : "FAILED") << "\n\n";
        large_ok = large_ok && ok;
    }
    printMatrix(c_cpu, M, N, "Matrix C (CPU Result) - First 4x4");
    if (device_ok) {
        printMatrix(c_gpu, M, N, "Matrix C (GPU Result) - First 4x4");
    }
    std::cout << "Large test: " << (large_ok ? "PASSED" : "FAILED") << '\n';

    const BenchResult cpu_bench =
        benchmarkWithReset([&] { nert_bench::gemmCpu(a.data(), b.data(), c_cpu.data(), kAlpha, kBeta, M, N, K); },
                           [&] { c_cpu = c_seed; }, 1, 3);
    std::cout << "\nExecution Times:\n";
    printBench("CPU:", cpu_bench);
    if (device_ok) {
        const BenchResult gpu_bench =
            benchmarkWithReset([&] { nert_bench::gemmGPU(a.data(), b.data(), c_gpu.data(), kAlpha, kBeta, M, N, K); },
                               [&] { c_gpu = c_seed; });
        printBench("GPU:", gpu_bench);
        printSpeedup("Speedup (CPU/GPU):", cpu_bench, gpu_bench);
    }
    return large_ok ? 0 : 1;
}

} // namespace

int main() {
    std::cout << "=== GEMM (General Matrix Multiplication) Test ===\n\n";
    try {
        bool device_ok = false;
        const int small = runSmall(device_ok);
        const int large = runLarge(device_ok);
        const bool ok = (small == 0) && (large == 0);
        std::cout << "\nOverall result: " << (ok ? "PASSED" : "FAILED") << '\n';
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "FAILED with exception: " << e.what() << '\n';
        return 1;
    }
}
