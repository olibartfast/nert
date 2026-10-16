# Validation — N1f Runtime-Backed Test/Bench

Defined before implementation. Bars: OFF untouched, format clean, no
`src/kernels`/loader/shape/executor diffs.

## Checks

- [V-1] M1: after each migrated test file, ON matrix
  (`nert_cuda|nert_kernels|nert_device|nert_constant|nert_loader_negative|
  nert_shapes_negative|nert_plan_negative|nert_executor_negative`) green;
  final M1 identical pass count to pre-M1 (no test added/removed).
- [V-2] M2: `bench_runtime` on the RTX 3060 prints the timing table, all
  executed cases PASSED within `1e-3·(1+|ref|)`, exit 0.
- [V-3] `SKIP:` contract: driver with no CUDA row / no device reports
  SKIP and exits 0 (evidence: Conv case on the cuDNN-less host leg).
- [V-4] OFF build 8/8; `scripts/format.sh --check` clean; `bench/README.md`
  documents the runtime driver + the legacy-harness follow-up note.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | M1 per-file migration, ON 13/13 after each of the 5 files | pass | 2026-10-16 | net -209 test lines; zero tests added/removed. |
| V-2 | `bench_runtime`: 11 PASSED / 1 skipped, exit 0 (table in bench/README) | pass | 2026-10-16 | RTX 3060; worst maxdiff 1.05e-05 (Gemm), Sigmoid/Tanh 1.19e-07. |
| V-3 | Conv case `SKIP: no CUDA row` on cuDNN-less host, exit 0 | pass | 2026-10-16 | same run as V-2. |
| V-4 | OFF 8/8 + `format --check` clean | pass | 2026-10-16 | no src/kernels/loader/shape/executor diffs. |
