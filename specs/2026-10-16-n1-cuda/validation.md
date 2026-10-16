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
| — | — | — | — | _N1a implementation not started; needs GPU runner._ |
