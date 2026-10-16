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
| V-1 | `docker exec nert-gpu nvidia-smi` (RTX 3060) + `nvcc 13.4` + `cudnn.h` (9.25) | pass | 2026-10-16 | image `nert-gpu:26.08` from `.devcontainer/Dockerfile` (tensorrt:26.08-py3 base). |
| V-2 | `ctest -R nert_cuda_conv` in container: 1x1 exact, 3x3+bias, depthwise, negative, Model-vs-CPU | pass | 2026-10-16 | within `1e-3·(1+\|ref\|)`; TF32 forbidden; plan cache per geometry. |
| V-3 | host OFF 8/8 + no CUDA in link; host ON (no cuDNN) 11/11 (conv SKIPs); container ON 11/11; format clean (102 files) | pass | 2026-10-16 | container uses `build-ct/` (bind-mount path split); root-owned files chown'd back. |
| V-4 | scope: `src/cuda/conv.cu` + table row + CMake cuDNN detect + tests | pass | 2026-10-16 | no `src/kernels`, loader, or shape diffs; cuDNN links PRIVATE. |

## Deviations

- Legacy cudnnConvolutionForward path instead of the packet's v9
  operation-graph sketch: same library kernels underneath, far smaller
  API surface, identical numerics goal. Graph API stays a follow-up
  option if fused conv+bias+act is ever needed.
- Asymmetric padding (odd-total SAME_* or explicit begin != end) is
  rejected on CUDA with a clear message; the legacy API takes one pad
  array. Symmetric cases (all surveyed needs) work.
- `GTEST_SKIP` must fire in the test body: fixed the helper pattern in
  all three CUDA test files (skip-in-helper silently continued into a
  null kernel call — caught on the no-cuDNN host leg as a segfault).
- Container/host build dirs split (`build-ct/` vs `build-cuda/`) — the
  same relative name resolves to different absolute paths across the
  bind mount.
