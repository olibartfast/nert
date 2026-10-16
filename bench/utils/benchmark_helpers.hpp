#pragma once
// Derived throughput/speedup numbers for bench reports.
//
// Translated (not copied) from the gpu-playground reference
// (source/utils/benchmark_helpers.h); see bench/utils/ATTRIBUTION.md once it
// lands.
//
// Converts a latency (normally a BenchResult::median_ms from benchmark.hpp)
// into a rate: throughput, bandwidth, or a speedup ratio. Timing itself lives
// in benchmark.hpp (benchmark(), benchmarkWithReset(), benchmarkDevice(),
// benchmarkDeviceWithReset()); this file only derives reportable numbers from
// the milliseconds those produce.

namespace gpu_benchmark {

[[nodiscard]] constexpr double giga_operations_per_second(double operations, double milliseconds) noexcept {
    return operations / (milliseconds * 1.0e6);
}

[[nodiscard]] constexpr double gigabytes_per_second(double bytes, double milliseconds) noexcept {
    return bytes / (milliseconds * 1.0e6);
}

[[nodiscard]] constexpr double million_items_per_second(double items, double milliseconds) noexcept {
    return items / (milliseconds * 1.0e3);
}

[[nodiscard]] constexpr double speedup(double baseline_milliseconds, double candidate_milliseconds) noexcept {
    return baseline_milliseconds / candidate_milliseconds;
}

} // namespace gpu_benchmark
