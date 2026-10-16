# Plan — N1b cuDNN Convolution

Branch `wip/nert-n1b` from main. All CUDA work happens inside the repo
`.devcontainer` (bind-mounted, so code syncs to the host automatically);
commits happen on the host.

## M1 — Devcontainer (host-side, CPU-verifiable)

- `.devcontainer/Dockerfile` from `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`
  + cmake/g++/gtest/clang-format-18/python3; `devcontainer.json` mounts the
  repo, passes `--gpus all`.
- Verify: container `nvcc --version`, `nvidia-smi`, cuDNN header present;
  host tree untouched except `.devcontainer/`.

## M2 — cuDNN Conv kernel (in-container, GPU runner)

- `src/cuda/conv.cu`: attribute mirror of `src/kernels/conv.cpp`
  (groups/strides/dilations/pads/auto_pad/bias), v9 graph build +
  `cudnnGetPlan` heuristics with per-shape fallback autotune, plan cache
  keyed by (dims, attrs), workspace via `CudaDevice` allocator (or
  `cudaMalloc` with release after run — allocator keeps one owner).
- Table entry; no loader/shape/executor changes expected (staging covers
  it); loader policy already pins the accepted subset.

## M3 — Parity + close-out (in-container, results recorded on host)

- `test/cuda_conv_test.cpp` (CUDA-only): hand-computed 1x1/3x3,
  stride-2, grouped, padded, bias/no-bias vs CPU oracle within
  `1e-3·(1+|ref|)` + multi-seed sweep; SKIP without GPU.
- OFF leg on host (8/8, no CUDA in link), ON leg in container,
  format clean, merge on host, compact (container kept for N1c–e,
  build dirs removed).
