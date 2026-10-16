# Plan — bench/matmul

Branch: `feature/bench-matmul` (stacked on `feature/bench-playground-layout`).
Note: requirements were written before code; this plan and `validation.md`
were reconstructed after implementation from the base packet's contract —
procedural deviation, recorded honestly. The work itself followed TDD
(red configure without `bench/matmul/`, green per backend leg on the RTX
3060 runner).

## M1 — Oracle + harness + CUDA (one slice)

- `cpu/matmul.{hpp,cpp}` (batched-broadcast oracle, double accum),
  `cuda/matmul.{hpp,cpp}` (tiled 2D kernel, per-batch host loop,
  `LANGUAGE CUDA`), `main.cpp` (small exact + batched-broadcast exact +
  large 5-seed + benches), `CMakeLists.txt` (backend switch cloned from
  gemm), `bench/CMakeLists.txt` += matmul.
- Green on the CUDA leg first (device present on the runner).

## M2 — OpenCL C + C++ ports

- `opencl/matmul.{hpp,cpp}` (C API, RAII releaser), `opencl_cpp/`
  (wrapper, `cl::` RAII). Same kernel algorithm, same `matmulGPU`
  signature. Green on both OpenCL legs.

## M3 — Close-out

- `format.sh --check` (bench scope), full-matrix `ctest -R bench_` 3/3 on
  all three legs (gemm regression included), validation evidence, commit,
  push, stacked PR onto `feature/bench-playground-layout`.
- No `src/`/`include/`/`test/` changes. Duplication accepted per [D-1].
