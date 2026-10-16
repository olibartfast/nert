# Validation — bench/ playground layout

Defined before implementation (2026-10-16). Evidence after execution.
Trace: R-1→V-1, R-2→V-2/V-5, R-3→V-3, R-4→V-4/V-5, R-5→V-1/V-6, R-7→V-7.

## Checks

- [V-1] Opt-in (R-5): default `cmake -S . -B build` configures;
  `ctest -N --test-dir build | grep bench` empty. With
  `-DNERT_BUILD_BENCH=ON`, `ctest -N` lists `bench_utils_headers` (+
  `bench_gemm` once G2 lands).
- [V-2] Harness (R-2): `cmake --build build --target nert_bench_gemm` then
  `./build/bench/gemm/nert_bench_gemm` prints `[[58,64],[139,154]]`, 5/5
  seeds, timings + speedup, exit `0`; a temporary local corruption of one
  CPU element flips exit to `1` (not committed).
- [V-3] Utils (R-3): `ctest -R bench_utils_headers` passes;
  `grep -rn '\.h"' bench/` empty; `ATTRIBUTION.md` names source + deltas.
- [V-4] Parity (R-4): `ctest -R bench_gemm` passes CPU-only (`SKIP:` GPU
  half); on GPU runners CPU vs device within `1e-3·(1+|ref|)`.
- [V-5] Matrix (R-2/R-4): default, `NERT_BENCH_USE_OPENCL=ON`,
  `NERT_BENCH_USE_OPENCL_CPP=ON` each configure+build+test; both flags
  together fail at configure with a clear message. Bench `ON` requires both
  SDKs at configure (all helpers compile-checked together); which device
  runs is per flag, and every device half runtime-SKIPs without a GPU
  (CPU-only default build with SDKs stays green).
- [V-6] No-regression (C-1/C-2/C-4): default configure+build+ctest green
  (ResNet fixture path unchanged); `git status --porcelain` shows only
  `bench/`, root hook, `specs/2026-10-16-kernel-bench-harness/`;
  `nert` target links stdlib only (`ldd` has no CUDA/OpenCL).
- [V-7] Style (R-7): `grep -rnE 'printf|fprintf|new float|delete\[|malloc|free\(|\([a-z_]+ ?\*\)' bench/ --include=*.cpp --include=*.hpp`
  empty (excl. comments); `scripts/format.sh --check` passes.

## Evidence (to fill)

- Date / runner / GPU: 2026-10-16 / CPU-only container / no GPU (nvcc present, no device).
- V-1: PASS (G1a): default OFF configure OK, 0 bench tests; `-DNERT_BUILD_BENCH=ON` lists `bench_utils_headers`.
- V-2: pending (G2).
- V-3: DONE (G1b): `ctest -R bench_utils_headers` 1/1 passed with all five
  headers included; `grep -rn '\.h"' bench/` empty; `ATTRIBUTION.md` records
  source + per-file deltas.
- V-4: pending (G2).
- V-5: pending (G3/G4).
- V-6: PASS (G1a, partial): default configure+existing tests untouched (only root option hook added); full ctest at close-out.
- V-7: PASS (G1a files): `clang-format-18 --dry-run --Werror` clean on the 3 bench C++ files.

## Acceptance (bench enabled)

```bash
cmake -S . -B build -DNERT_BUILD_BENCH=ON && cmake --build build --target nert_bench_gemm \
&& ctest --test-dir build -R 'bench_' --output-on-failure
```
