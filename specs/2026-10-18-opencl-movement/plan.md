# Plan — OpenCL Movement Operators

Work in the main tree. Do not modify `bench/`,
`specs/2026-10-17-cudnn-frontend-spike/`,
`specs/2026-10-18-memory-footprint/`, `docs/`, or `.github/`.

## Transpose and Gather

Port the logic of `src/cuda/transpose.cu` and `src/cuda/gather.cu` to OpenCL C.
Use one work item per output element, packed `long` strides, and parameter
tensors downloaded to the host where needed.

`Transpose` reads the `perm` attribute and defaults to reverse order. `Gather`
defaults its axis to 0 and wraps negative indices. Tests live in
`test/opencl_kernels_test.cpp`.

## Slice and Concat

`Slice` downloads `starts`, `ends`, `axes`, and `steps` to the host, applies
the oracle's wrapping and clamping rules, and gathers each output element.

`Concat` wraps the axis, checks rank agreement, and copies contiguous blocks
with `clEnqueueCopyBuffer` for each input and outer row. It needs no compute
kernel. Add the bool data path decided during review, and preserve the
zero-extent behavior required by real graphs.

## Shape and Unsqueeze

`Shape` writes the selected dimensions as int64 using a small fill kernel and
the `start`/`end` attributes.

`Unsqueeze` downloads the axes, applies wrapping, sorting, duplicate checks,
and range checks, then copies the input buffer. Shape inference owns the output
shape. Tests live beside the earlier phases.

## Acceptance

Each operator needs a hand-computed case, a CPU-oracle comparison, and negative
cases. While implementing, run the targeted OpenCL test target on the RTX 3060
(NVIDIA platform). At the end, run the full matrix:

- default build: full CTest suite;
- CUDA build: full CTest suite;
- OpenCL build: full CTest suite;
- format check;
- diff audit limited to the expected specification and source paths.

## Validation

Evidence is recorded in [`validation.md`](validation.md). Update the roadmap
and changelog when the work lands.
