# Plan — OpenCL pools

Work in the main tree. Forbidden: `bench/`,
`specs/2026-10-17-cudnn-frontend-spike/`,
`specs/2026-10-18-memory-footprint/`, `docs/`, `.github/`.
`implementer` dispatch unavailable (harness); planner implements
directly.

Single phase (one pattern, three ops). Port
`src/cuda/max_pool.cu`, `average_pool.cu`,
`global_average_pool.cu` to OpenCL C: host-side attribute/pad/shape
validation copied verbatim, compute in per-output-element kernels
with a packed `PoolParams` struct arg (Add precedent for by-value
struct args). GAP is one (n,c)-row kernel with double accumulation.

Bars: hand-computed + oracle-compare + negatives per op; OFF +
CUDA-ON + OpenCL-ON legs green; format clean; diff audit =
packet paths only.

## Validation

Evidence in `specs/2026-10-18-opencl-pools/validation.md`.
Roadmap + CHANGELOG sync on landing (planner-owned).
