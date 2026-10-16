# Requirements — N1 CUDA Device Layer (hybrid, libraries-first)

Date: 2026-10-16. Roadmap: Phase N1 (was Planned — now specified).
Base: `main` post-Phase-5 (`include/nert/`, namespace `nert`, 0.2.0).
Constitution: mission/tech-stack/roadmap.

## Vendor decision [Q-1-vendor] (2026-10-16, maintainer)

Hybrid, leaning heavily on libraries:

- **NVIDIA servers:** CUDA with cuDNN for convolutions and
  cuBLAS/cuBLASLt for GEMMs. cuDNN v9 frontend/graph API: heuristics +
  per-shape autotuning, engine plans cached at startup. NHWC FP16/BF16
  tensor-core paths. System throughput from dynamic batching across
  cameras, multiple CUDA streams, CUDA Graphs.
- **AMD ROCm-supported discrete:** HIP, via hipify from the CUDA code.
- **Portable fallback (Mali/Adreno/Intel iGPU/unsupported AMD):**
  Vulkan compute first (cooperative matrix on RDNA3 and NVIDIA),
  OpenCL as fallback — one backend covering everything.
- **First-party kernels only where they pay:** device pre-processing
  fuse (remap/undistort/resize/normalize/pack), post
  (decode/threshold/NMS/argmax on GPU, no full-res D2H), elementwise /
  activation fusion outside cuDNN graph patterns, odd ops (custom
  attention, deformable sampling). CUTLASS as middle ground for fused
  GEMM/conv with custom epilogue.
- **Cost accepted:** packaging (100s MB, CUDA/driver pins; cuDNN 9
  split sub-libraries to ship only what's used).
- **Open:** what pushes away from TensorRT (dynamic shapes?
  preprocessing control? plugin pain?) — confirm before committing the
  serving stack around N1.

## In Scope (N1a — CUDA pilot)

- [R-1] CUDA device layer behind the existing device seam
  (`include/nert/device.hpp`): allocator, transfer, op-keyed kernel
  table. One device per graph at load; no per-op fallback (mission
  holds). CPU path untouched and still the correctness oracle.
- [R-2] Pilot coverage: `Gemm` via cuBLAS + `Relu`/`Add` first-party
  elementwise (proves both library and first-party paths through the
  seam). Everything else rejects CUDA at load until ported.
- [R-3] Opt-in build: CUDA toolkit + cuBLAS required only when the CUDA
  device is enabled; default configure/build/test unchanged, runtime
  stays stdlib-only when CUDA is off.
- [R-4] Parity: CPU oracle vs CUDA within tolerance on the `bench/gemm`
  shapes + a Gemm/Relu/Add micro-graph, evidenced on a GPU runner.

## Out of Scope

- cuDNN conv integration (N1b), HIP hipify port (N1c), Vulkan/OpenCL
  portable backend (N1d), pre/post fusion kernels (N1e), TensorRT
  decision, serving/batching/Graphs machinery (consumer-side).

## Constraints

- Device-agnostic core untouched: new device = new kernel table, not a
  refactor (tech-stack architecture).
- `bench/` tiled kernels are the measurement reference, not pasted
  adapters (bench [D-1]): shared harness header + oracle source of
  truth + kernel-table adapters.
