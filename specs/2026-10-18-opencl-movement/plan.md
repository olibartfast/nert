# Plan — OpenCL shared movement rows

Work in the main tree. Forbidden: `bench/`,
`specs/2026-10-17-cudnn-frontend-spike/`, `specs/2026-10-18-memory-footprint/`,
`docs/`, `.github/`. `implementer` dispatch unavailable (harness);
planner implements directly.

## Phase A — Transpose + Gather (gather family)

Pattern: CUDA `transpose.cu`/`gather.cu` logic ported to OpenCL C
(per-output thread, packed `long` strides, D2H-downloaded params).
`Transpose`: perm attr default reverse. `Gather`: axis default 0,
negative index wrap. Tests in `test/opencl_kernels_test.cpp`.

## Phase B — Slice + Concat (movement)

`Slice`: starts/ends/axes/steps D2H downloads, wrap+clamp, per-output
gather kernel. `Concat`: axis wrap + rank agreement, per-(input,
outer-row) `clEnqueueCopyBuffer` D2D (ConcatCopy precedent, no
compute kernel). Tests beside Phase A.

## Phase C — Shape + Unsqueeze (shape family)

`Shape`: int64 out, start/end attr clamp, tiny fill kernel.
`Unsqueeze`: axes D2H, wrap/sorted-unique/range checks, buffer copy
(Squeeze precedent). Tests beside Phase A.

## Per-phase bars

Hand-computed + oracle-compare + negatives; targeted OpenCL tests
green on the RTX 3060 (NVIDIA platform); final acceptance once at
end: OFF all-green + CUDA-ON all-green + OpenCL-ON all-green +
format clean + diff audit = packet paths only.

## Validation

Evidence in `specs/2026-10-18-opencl-movement/validation.md`.
Roadmap + CHANGELOG sync on landing (planner-owned).
