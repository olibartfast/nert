# Plan — N1 CUDA Device Layer

Branch `wip/nert-n1a` from main. Thin phases; needs a GPU runner with
CUDA toolkit + cuBLAS for green runs (CPU-only boxes verify the OFF
path + SKIP contract only).

## N1a-M1 — Device scaffold + build wiring (CPU-verifiable)

- `src/cuda_device.cpp` (+ private header): `CudaDevice` implementing
  the device-seam interface (allocator via `cudaMalloc`, H2D/D2H,
  kernel table returning null for unported ops → clean load rejection).
- Root `CMakeLists.txt`: `NERT_ENABLE_CUDA` (OFF default) pulling
  `find_package(CUDAToolkit)` + cuBLAS only when ON; CUDA language
  enabled per-target, never at root scope. Default build has no CUDA
  in the link closure (`ldd` check).
- `bench/`-style SKIP: CUDA tests exit 0 + `SKIP:` without a GPU.

## N1a-M2 — Pilot kernels (GPU runner)

- `Gemm` via cuBLAS (`cublasSgemm` path, alpha/beta/bias handling per
  runtime semantics), `Relu`/`Add` first-party elementwise kernels
  (`.cu`, one thread per element, broadcast indexing mirroring the CPU
  oracles).
- `cuda_kernels` table entries; loader device check accepts
  Gemm/Relu/Add graphs on CUDA, rejects the rest with node + op.

## N1a-M3 — Parity + close-out

- Parity harness: `bench/gemm` shapes + micro-graph, CPU vs CUDA
  within `1e-3·(1+|ref|)`; evidence in `validation.md` with runner/GPU.
- Roadmap N1 → In progress (N1a) with N1b–e queued; CHANGELOG entry;
  format clean; merge; compact.

## Later (queued, same packet family)

- N1b cuDNN conv, N1c HIP hipify, N1d Vulkan/OpenCL portable, N1e
  pre/post fusion. Each gets its own dated packet slice before code.
