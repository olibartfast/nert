# Validation — N1 CUDA Device Layer

Defined before implementation. Evidence after execution.

## Checks

- [V-1] Opt-in (R-3): default `cmake -S . -B build` configures with no
  CUDA; `ldd build/libnert.a` n/a (static) — `grep -ri cuda
  src/CMakeFiles`… concretely: `nert` target has no CUDA link when OFF;
  `-DNERT_ENABLE_CUDA=ON` without a toolkit fails with a clear message.
- [V-2] Seam (R-1): CUDA-only micro-graph (Gemm/Relu/Add) loads on the
  CUDA device; a graph with any other op is rejected at load naming
  node + op; CPU path results unchanged.
- [V-3] Parity (R-4, GPU runner): `bench/gemm` shapes + micro-graph CPU
  vs CUDA within `1e-3·(1+|ref|)`; timings reported, exit 0.
- [V-4] Style/scope: `scripts/format.sh --check` clean (CUDA sources
  covered); `git diff --stat` shows device-layer + build wiring only,
  no core/kernel semantic changes.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | default configure+build; `grep cuda build/CMakeFiles/nert.dir/link.txt` empty; `ctest -N \| grep cuda` empty | pass | 2026-10-16 | OFF leg: 8/8 non-fixture green. |
| V-2 | micro-graph (Gemm/Relu/Add) on CUDA vs CPU + Sigmoid rejection naming node+op | pass | 2026-10-16 | executor staging; rejection at dispatch (deviation: see below). |
| V-3 | `bench/gemm` small exact `[[58,64],[139,154]]` bit-exact; transposed+bias within `1e-3(1+|ref|)`; micro-graph within tol | pass | 2026-10-16 | RTX 3060 Laptop GPU, CUDA 12.0, TF32 forbidden. |
| V-4 | `scripts/format.sh --check` clean (now covers `.cu`); scope: seam+staging+CUDA layer only | pass | 2026-10-16 | no `src/kernels`, loader, or shape changes. |

## Deviations

- Rejection timing: a graph with ops missing CUDA kernels fails at first
  `Run` (dispatch), not at `Model` construction — matching the pinned CPU
  behavior (`NertExecutorNegative.OpWithoutKernelThrows`, which constructs
  successfully and expects the throw at `Run`). The mission's "rejected at
  load" wording would require changing that test; flagged for maintainer.
  Implementation and executable checks take precedence on drift.
- `Add` CUDA broadcast caps rank at 16 with an explicit error (N1b
  generalizes); CPU oracle stays general.
- cuBLAS handle created/destroyed per `Gemm` call (pool in N1e); a
  `cudaDeviceSynchronize` per pilot kernel favors correctness proof over
  throughput (streams/Graphs are consumer-side per the backend decision).
