# Validation — N1c HIP on NVIDIA

Defined before implementation. Bars: OFF untouched, format clean, no
`src/kernels`/loader/shape/executor diffs.

## Checks

- [V-1] M1: device names itself, empty table, 1 KiB round trip on the
  RTX 3060 via the HIP NVIDIA backend; OFF 8/8; format clean.
- [V-2] M2+: per-op parity within `1e-3·(1+|ref|)` on the RTX 3060.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| M4 | Sigmoid/Tanh/Softmax HIP (hipified, reviewed, TDD red-green per op) vs CPU (tolerance bar, axis + stability rows) + OFF 8/8 + HIP-leg 10/10 + format | pass | 2026-10-17 | Double-accumulated softmax like CUDA; no tolerance deviation needed. |
| M5 | Reshape HIP (hipified from src/cuda/reshape.cu, reviewed: namespace/API names only) vs CPU (flat-order [2,3]->[3,2], shape {0,-1}, int64 copy, count-mismatch negative) + table row; HIP gtest 19/19 + OFF 12/12 + format | pass | 2026-10-18 | Same host-side validation as CPU/CUDA; hipMemcpy movement, no warp assumptions. |
| M3 | Mul/Sub/Div HIP (hipified, reviewed, GPU-agnostic audit clean) vs CPU (float bcast, int64, IEEE div, div-by-zero throws) + OFF 8/8 + HIP-leg 10/10 + format | pass | 2026-10-17 | RTX 3060; Div-int64 device-flag/host-report via portable `atomicExch`. |
| M2 | Relu/Add HIP (hipified, reviewed) vs CPU (exact, int64 bcast, Gemm permanent probe) + OFF 8/8 + HIP-leg 10/10 + format | pass | 2026-10-17 | Translation is API-names only; `<<<>>>` kept, nvcc-compiled. |
| M1 | HipDevice names/seam/empty-table/1KiB round trip on RTX 3060 via HIP NVIDIA backend + OFF 8/8 + HIP-leg 9/9 + format | pass | 2026-10-17 | nvcc 12.0 + hip-dev 6.2 headers (rootless scratch); deprecated-CUDA warnings come from AMD headers, not our code. |
| — | — | pending | — | — |
