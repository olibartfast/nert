# Plan — Bench every kernel, test every kernel, duplicate nothing

Small slices, each reviewable on its own. Every slice ends with green
legs and a ledger row.

## Slice 1 — Delete the legacy harnesses

- Remove `bench/gemm/`, `bench/matmul/`, and `bench/utils/` (the last
  one is only consumed by the first two; `bench/runtime` links `nert`
  alone). Drop the now-unused `NERT_BENCH_USE_OPENCL*` flags and update
  `bench/CMakeLists.txt` and `bench/README.md`.
- Validation: default OFF build untouched (8/8), CUDA bench leg still
  builds and runs, format clean.

## Slice 2 — One bench driver for all devices

- Teach `bench/runtime` to enumerate the CUDA, OpenCL, and HIP devices
  behind one fixture list. A device that is not compiled in (flag OFF)
  or not present at run time is skipped, never failed.
- Validation: one bench binary per enabled device runs every fixture;
  numbers recorded in `bench/README.md`.

## Slice 3 — One shared test-helper header

- Extract the duplicated upload/download/view/node helpers from the
  OpenCL and HIP kernel tests into a shared header; both suites use it.
  Net line count must go down.
- Validation: OpenCL leg and HIP leg still green, format clean.

## Slice 4 — Coverage audit

- Walk the three kernel tables row by row; check off unit test +
  bench fixture per row in `validation.md`. Missing rows become new
  roadmap entries with owners (N1c/N1d slice or new packet).
