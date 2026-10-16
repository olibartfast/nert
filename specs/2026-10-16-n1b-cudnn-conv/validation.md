# Validation — N1b cuDNN Convolution

Defined before implementation. Evidence after execution.

## Checks

- [V-1] M1: `docker exec nert-gpu nvidia-smi` shows the RTX 3060;
  `nvcc --version` 12.x; `cudnn.h` present; host `git status` shows
  only `.devcontainer/`.
- [V-2] Parity (R-3, in-container GPU): hand-computed conv cases +
  multi-seed sweep CPU vs CUDA within `1e-3·(1+|ref|)`; exit 0.
- [V-3] No-regression: host OFF leg 8/8 non-fixture + no CUDA in link;
  container ON leg full suite; `scripts/format.sh --check` clean.
- [V-4] Scope: no `src/kernels`, loader, or shape diffs; cuDNN links
  PRIVATE; OFF closure stdlib-only.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | — | — | _to fill_ |
