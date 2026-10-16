# Requirements — bench/matmul (second playground-layout kernel)

Date: 2026-10-16. Branch: `feature/bench-matmul` (stacked on
`feature/bench-playground-layout`, PR nert#1). Base packet:
`specs/2026-10-16-kernel-bench-harness/` (harness contract, utils, flags —
all reused unchanged). Constitution: mission/tech-stack/roadmap as in the
base packet.

## Intent

Prove the `bench/` layout generalizes beyond the pilot: add `matmul`
(`C = A·B`, batched) following the exact `bench/gemm/` pattern, so adding
kernel N+1 is mechanical.

## Scope

In:

- [R-1] `bench/matmul/` mirrors `bench/gemm/`: `main.cpp`, `cpu/`,
  `cuda/`, `opencl/`, `opencl_cpp/`, `CMakeLists.txt`; target
  `nert_bench_matmul`, test `bench_matmul`; built whenever bench is `ON`
  (same rule as gemm — no per-kernel flag in this packet).
- [R-2] Same harness contract as gemm: 3-way `#ifdef` include, small exact
  2D test + small exact batched test (batch=2) + large 2D multi-seed (5)
  tolerance test (`1e-3·(1+|ref|)`), `benchmarkWithReset`, `std::cout`
  report, exit `0`/`1`, device-half `SKIP` + exit `0` when no device.
- [R-3] Bench-local CPU oracle with double accumulation, one rounding per
  output; supports 2D plus broadcast batch dims (same-batch and
  batch-vs-no-batch), matching the runtime MatMul's batch semantics for the
  covered shapes. Duplicated, not including engine-private `Kernels.hpp`
  ([D-2] of the base packet).
- [R-4] Device kernels: tiled 2D kernel per backend (same TILE 16 pattern as
  gemm); batches looped on the host (one launch per batch item). No
  alpha/beta — plain `C = A·B`.
- [R-5] TDD: red (acceptance lists `bench_matmul`, fails without the files)
  → green on all three backend legs on the RTX 3060 runner.
- [R-6] Modern C++17, `.hpp`, translated-not-pasted, same as the base packet
  ([R-7]/[C-5]/[C-6] carry over).

Out:

- Other kernels; per-kernel `NERT_BENCH_ENABLE_*` flags (single rule stands);
  fused/epilogue variants; N1 wiring; any `src/`/`include/`/`test/` change.

## Constraints

- [C-1] CUDA path compiles `LANGUAGE CUDA` per-file; bench floor CMake
  >= 3.17 stands; runtime untouched and stdlib-only.
- [C-2] No shared harness library in this packet — duplication of the small
  `main.cpp` pattern is accepted (dedupe only when the third kernel lands).

## Decisions

- [D-1] (2026-10-16, maintainer): bench-phase duplication (per-kernel
  `main.cpp`, per-backend kernel copies, bench-local oracles duplicated from
  `src/kernels/`) is accepted as scaffolding. When N1 consumption starts,
  it starts with a redesign that removes it: one shared harness header, one
  oracle source of truth, and kernel-table adapters — not adapters pasted
  per kernel. Resolves [Q-1] as deferred-to-N1.

## Open questions

- [Q-1] Factor the common harness (compare/bench/report) into
  `bench/utils/harness.hpp` once matmul proves the pattern twice? Owner:
  maintainer.
