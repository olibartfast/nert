# Validation — N1e via cuDNN Frontend

Defined before implementation. Bar: numerics unchanged (tolerance
`1e-3·(1+|ref|)`, target exact vs the legacy row), OFF default
untouched, format clean, CUDA container leg green.

## Checks

- [V-A1] Slice A: Conv parity exact on ResNet + YOLO fixtures; bench
  row before/after (same fixture); CUDA + OFF legs green.
- [V-B1] Slice B: frontend Conv bit-exact vs CPU oracle and vs the
  legacy row on both shapes; stride/pad variants covered; no-row
  fallback tested when the frontend dir is absent.
- [V-B2] Slice B: default build (OFF) has no frontend include or link;
  CUDA-without-frontend still builds and passes.
- [V-C1] Slice C: fused Conv+Bias+Relu matches composed legacy rows;
  bench proves the fused win.
- [V-G] Format clean; no `src/kernels/`, loader, shape, executor diffs.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | pending | — | — |
