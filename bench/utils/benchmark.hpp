#pragma once
// Timing helpers for the standalone bench harness.
//
// Translated (not copied) from the gpu-playground reference
// (source/utils/benchmark.h); see bench/utils/ATTRIBUTION.md once it lands.
//
// A single timed call says little: the first invocation pays one-off costs
// (context creation, program build, first-touch faults, clock ramp-up) that
// can dwarf the measured work. Harnesses therefore run a few untimed warm-up
// calls, then report statistics over several timed iterations so run-to-run
// noise is visible instead of hidden.
//
// Sync contract: a timed callable must block until its device work has
// actually completed before returning (a blocking device-to-host copy
// counts). Otherwise the timing only captures launch overhead, not the work
// itself. This applies to `fn` passed to benchmark()/benchmarkWithReset() and
// to the end-to-end timing around `fn` in benchmarkDevice()/
// benchmarkDeviceWithReset(); the device-reported value returned by `fn` in
// the latter two is assumed to already reflect completed device work (e.g.
// from CUDA events).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

// Default protocol: discard kBenchWarmup untimed calls, then time
// kBenchIterations.
constexpr int kBenchWarmup = 3;
constexpr int kBenchIterations = 10;

struct BenchResult {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    int warmup = 0;
    int iterations = 0;
};

// Reduces already-collected timing samples (milliseconds, warm-up already
// excluded) to a BenchResult. Order-independent: `samples` is sorted
// internally, so callers may pass them in any order. `warmup` is metadata
// only, recording how many untimed warm-up calls preceded these samples; it
// does not affect the statistics below. Median is the middle element for an
// odd count and the mean of the two middle elements for an even count.
// Stddev is the sample stddev (n-1), or 0 for a single sample. An empty
// `samples` yields a zeroed BenchResult (iterations == 0), with `warmup`
// still recorded.
[[nodiscard]] inline BenchResult summarize(std::vector<double> samples, int warmup = 0) {
    BenchResult result;
    result.warmup = warmup;

    const int iterations = static_cast<int>(samples.size());
    if (iterations == 0)
        return result;
    result.iterations = iterations;

    std::sort(samples.begin(), samples.end());
    result.min_ms = samples.front();
    result.max_ms = samples.back();
    result.median_ms =
        (iterations % 2 == 1) ? samples[iterations / 2] : 0.5 * (samples[iterations / 2 - 1] + samples[iterations / 2]);

    double sum = 0.0;
    for (double s : samples)
        sum += s;
    result.mean_ms = sum / iterations;

    double sq = 0.0;
    for (double s : samples)
        sq += (s - result.mean_ms) * (s - result.mean_ms);
    result.stddev_ms = (iterations > 1) ? std::sqrt(sq / (iterations - 1)) : 0.0;

    return result;
}

// `reset` runs before every call (warm-up and timed alike) and is never timed.
// Use it for kernels whose output buffer is also an input, e.g. GEMM's
// beta * C term, so each iteration measures the same work on the same
// starting state.
template <typename Fn, typename Reset>
[[nodiscard]] BenchResult benchmarkWithReset(Fn&& fn, Reset&& reset, int warmup = kBenchWarmup,
                                             int iterations = kBenchIterations) {
    if (iterations < 1)
        iterations = 1;
    if (warmup < 0)
        warmup = 0;

    for (int i = 0; i < warmup; i++) {
        reset();
        fn();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; i++) {
        reset();
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    return summarize(std::move(samples), warmup);
}

template <typename Fn>
[[nodiscard]] BenchResult benchmark(Fn&& fn, int warmup = kBenchWarmup, int iterations = kBenchIterations) {
    return benchmarkWithReset(fn, [] {}, warmup, iterations);
}

// Pairs host-side end-to-end timing (steady_clock around the whole call) with
// device-reported timing for the same calls, so a harness can report both
// kernel-only cost and the surrounding overhead (allocation, transfer,
// launch) in one pass. See benchmarkDeviceWithReset.
struct DeviceBenchResult {
    BenchResult end_to_end;
    BenchResult device;
};

// Like benchmarkWithReset, but `fn` returns that call's device-measured time
// in milliseconds (e.g. from CUDA events), convertible to double. `reset`
// runs before every call (warm-up and timed alike) and is never timed.
// Warm-up calls are discarded from both the end_to_end and device series.
// Defaults and clamping match benchmarkWithReset.
template <typename Fn, typename Reset>
[[nodiscard]] DeviceBenchResult benchmarkDeviceWithReset(Fn&& fn, Reset&& reset, int warmup = kBenchWarmup,
                                                         int iterations = kBenchIterations) {
    if (iterations < 1)
        iterations = 1;
    if (warmup < 0)
        warmup = 0;

    for (int i = 0; i < warmup; i++) {
        reset();
        fn();
    }

    std::vector<double> end_to_end_samples;
    std::vector<double> device_samples;
    end_to_end_samples.reserve(static_cast<std::size_t>(iterations));
    device_samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; i++) {
        reset();
        const auto t0 = std::chrono::steady_clock::now();
        const double device_ms = static_cast<double>(fn());
        const auto t1 = std::chrono::steady_clock::now();
        end_to_end_samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        device_samples.push_back(device_ms);
    }

    DeviceBenchResult result;
    result.end_to_end = summarize(std::move(end_to_end_samples), warmup);
    result.device = summarize(std::move(device_samples), warmup);
    return result;
}

template <typename Fn>
[[nodiscard]] DeviceBenchResult benchmarkDevice(Fn&& fn, int warmup = kBenchWarmup, int iterations = kBenchIterations) {
    return benchmarkDeviceWithReset(fn, [] {}, warmup, iterations);
}

// Human-readable one-line summary; the exact text is informational only.
inline void printBench(const char* label, const BenchResult& r) {
    std::cout << std::left << std::setw(22) << (label != nullptr ? label : "") << " median " << std::fixed
              << std::setprecision(4) << std::setw(9) << r.median_ms << " ms | mean " << std::setw(9) << r.mean_ms
              << " ms +/- " << r.stddev_ms << " | min " << std::setw(9) << r.min_ms << " | max " << std::setw(9)
              << r.max_ms << "  (" << r.warmup << " warm-up + " << r.iterations << " timed)\n";
}

// Compare on medians: the mean is the metric an outlier iteration can move
// most.
inline void printSpeedup(const char* label, const BenchResult& baseline, const BenchResult& candidate) {
    std::cout << std::left << std::setw(22) << (label != nullptr ? label : "");
    if (candidate.median_ms > 0.0) {
        std::cout << ' ' << std::fixed << std::setprecision(2) << baseline.median_ms / candidate.median_ms
                  << "x (median baseline / median candidate)\n";
    } else {
        std::cout << " n/a (candidate median is 0)\n";
    }
}
