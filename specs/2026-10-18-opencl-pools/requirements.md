# OpenCL Pooling Operators

Date: 2026-10-18.

## Purpose

This specification adds the three pooling operators that ResNet-18 and the
detection models need on OpenCL: `MaxPool`, `AveragePool`, and
`GlobalAveragePool`. Together with `Flatten` and `Gemm`, they fill the
non-`Conv` gaps for ResNet-18. `Conv` remains the dependency that blocks every
full model on the portable backends.

## Scope

Implement the operators as first-party OpenCL C in `src/opencl/`, compiled at
runtime through `opencl_runtime`. Each is a direct port of the verified CUDA
row, with the same behavior and validation. Add table rows, target sources, and
tests in `test/opencl_kernels_test.cpp`. Tests need a hand-computed case, a
CPU-oracle comparison, and negatives; they skip when no OpenCL platform exists.

## Behavior

**`MaxPool`** requires `kernel_shape`. It supports dilations, `auto_pad` values
`NOTSET`, `VALID`, and `SAME_*`, `ceil_mode`, and negative infinity for
out-of-bounds samples. Only the primary output is written.

**`AveragePool`** fixes dilations at `{1, 1}`, matching the CPU oracle. It
supports `count_include_pad`, including the divisor rule for a window that
`ceil_mode` extends past the input.

**`GlobalAveragePool`** accepts rank 2 or higher and collapses the spatial
dimensions to one. It accumulates in double precision and rounds once per
output. Because OpenCL does not require double-precision floating point
(fp64), the implementation checks `CL_DEVICE_DOUBLE_FP_CONFIG` and raises a
named error when the device cannot provide it.

## Boundaries

Out of scope: HIP mirrors (this host has no hipify toolchain, as recorded in
`Q-1`), OpenCL `Conv`, `Flatten`, and `Gemm`, bench fixtures, loader and
shape-inference changes, executor changes, and Neuriplo model runs.

The default build remains standard-library-only. OpenCL is private and opt-in.
A graph uses one device and has no per-operator fallback. Packed parameters
retain the rank cap of 16, and kernels synchronize with `clFinish`, following
the existing `Relu` row.

## Decisions

No open questions remain. The maintainer approved the three-pool scope on
2026-10-18.
