# Validation — N1c HIP on NVIDIA

Defined before implementation. Bars: OFF untouched, format clean, no
`src/kernels`/loader/shape/executor diffs.

## Checks

- [V-1] M1: device names itself, empty table, 1 KiB round trip on the
  RTX 3060 via the HIP NVIDIA backend; OFF 8/8; format clean.
- [V-2] M2+: per-op parity within `1e-3·(1+|ref|)` on the RTX 3060.
- [V-3] M6: the 68-row first-party table builds and resolves through the HIP
  NVIDIA backend; the adapted CUDA first-party suites pass; library-backed rows
  (`Gemm`, `MatMul`, `Conv`, `ConvTranspose`) remain null so the executor
  rejects them explicitly.
- [V-4] M7: the 72-row first-party table builds and resolves through the HIP
  NVIDIA backend; the four compute rows meet the CPU oracle (parity +
  negatives); the OFF and OpenCL suites stay green; format clean.

## Evidence

Main integration revalidation (2026-10-23): reused the four compute kernels
and 14 parity/negative tests from OpenCode `0279876`. Clean Release build:
`cmake -S . -B /tmp/nert-hip-complete -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_HIP=ON -DHIP_INCLUDE_DIR=/tmp/opencode/rocm/opt/rocm-6.2.0/include`,
then `cmake --build /tmp/nert-hip-complete --parallel 4` and
`ctest --test-dir /tmp/nert-hip-complete --output-on-failure` passed 15/15
on RTX 3060 Laptop GPU. Scoped clang-format-18 checks passed. Earlier
OpenCL/OFF evidence below is inherited, not rerun for this integration.
Subsequent native AMD validation passed on MI300X; full-model AMD and
persistent CI gates remain pending. Reproduction:
`../2026-10-23-amd-devcloud/runbook.md`.

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| M4 | Sigmoid/Tanh/Softmax HIP (hipified, reviewed, TDD red-green per op) vs CPU (tolerance bar, axis + stability rows) + OFF 8/8 + HIP-leg 10/10 + format | pass | 2026-10-17 | Double-accumulated softmax like CUDA; no tolerance deviation needed. |
| M5 | Reshape HIP (hipified from src/cuda/reshape.cu, reviewed: namespace/API names only) vs CPU (flat-order [2,3]->[3,2], shape {0,-1}, int64 copy, count-mismatch negative) + table row; HIP gtest 19/19 + OFF 12/12 + format | pass | 2026-10-18 | Same host-side validation as CPU/CUDA; hipMemcpy movement, no warp assumptions. |
| M3 | Mul/Sub/Div HIP (hipified, reviewed, GPU-agnostic audit clean) vs CPU (float bcast, int64, IEEE div, div-by-zero throws) + OFF 8/8 + HIP-leg 10/10 + format | pass | 2026-10-17 | RTX 3060; Div-int64 device-flag/host-report via portable `atomicExch`. |
| M2 | Relu/Add HIP (hipified, reviewed) vs CPU (exact, int64 bcast, Gemm permanent probe) + OFF 8/8 + HIP-leg 10/10 + format | pass | 2026-10-17 | Translation is API-names only; `<<<>>>` kept, nvcc-compiled. |
| M1 | HipDevice names/seam/empty-table/1KiB round trip on RTX 3060 via HIP NVIDIA backend + OFF 8/8 + HIP-leg 9/9 + format | pass | 2026-10-17 | nvcc 12.0 + hip-dev 6.2 headers (rootless scratch); deprecated-CUDA warnings come from AMD headers, not our code. |
| M6 | 59 additional hipified files; 68-row first-party table; `build-hip` on RTX 3060 via NVIDIA HIP; `ctest --test-dir build-hip` 15/15; adapted CUDA first-party suites 149/149; format 298 | pass | 2026-10-22 | `Gemm`, `MatMul`, `Conv`, `ConvTranspose` intentionally absent; AMD/ROCm half still pending. |
| M7 | 4 first-party compute rows (`src/hip/gemm.cu`, `mat_mul.cu`, `conv.cu`, `conv_transpose.cu`: naive, double-accum, no library); 72-row table probe; `build-hip` on RTX 3060 via NVIDIA HIP `ctest` 15/15; focused `nert_hip_kernels_test` Gemm/MatMul/Conv filter 14/14; `build-ocl` 29/29; OFF `build` 13/13; `scripts/format.sh --check` 302 files | pass | 2026-10-23 | No rocBLAS/hipBLAS/MIOpen dependency anywhere; only the AMD-backend build still needs an AMD runner. |
