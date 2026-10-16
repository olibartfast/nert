// Compile-time + value check for the vendored bench utility headers.
// Registered as the ctest case `bench_utils_headers`. Includes every
// vendored header but executes no device code, so it passes on CPU-only
// runners with the SDKs installed.

#include "benchmark.hpp"
#include "benchmark_helpers.hpp"
#include "cuda_helpers.hpp"
#include "opencl_c_helpers.hpp"
#include "opencl_helpers.hpp"

#include <cmath>
#include <iostream>
#include <type_traits>

namespace {

bool nearly_equal(double a, double b) { return std::fabs(a - b) <= 1.0e-9; }

} // namespace

int main() {
    const BenchResult summary = summarize({1.0, 2.0, 3.0});
    if (!nearly_equal(summary.median_ms, 2.0)) {
        std::cerr << "bench_utils_headers: summarize median mismatch: expected 2.0, got " << summary.median_ms << '\n';
        return 1;
    }

    const double ratio = gpu_benchmark::speedup(2.0, 1.0);
    if (!nearly_equal(ratio, 2.0)) {
        std::cerr << "bench_utils_headers: speedup mismatch: expected 2.0, got " << ratio << '\n';
        return 1;
    }

    const double gflops = gpu_benchmark::giga_operations_per_second(1.0e9, 1000.0);
    if (!nearly_equal(gflops, 1.0)) {
        std::cerr << "bench_utils_headers: gflops mismatch: expected 1.0, got " << gflops << '\n';
        return 1;
    }

    // SDK headers are referenced (not executed): static checks that the
    // throwing helpers exist with the expected signatures. Any missing symbol
    // fails at compile time, which is the point of this TU.
    static_assert(std::is_same_v<decltype(&nert_bench::getTime), double (*)() noexcept>);
    static_assert(std::is_same_v<decltype(&nert_bench::clTeardown),
                                 void (*)(cl_context, cl_command_queue, cl_program, cl_kernel)>);

    std::cout << "bench_utils_headers: ok\n";
    return 0;
}
