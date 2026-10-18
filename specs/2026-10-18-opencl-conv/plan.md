# Plan — OpenCL Conv

Single phase. Host validation copied from `src/kernels/conv.cpp`
(rank/weight/group/stride/dilation/pad/output/bias, SAME_* resolution).
One kernel: output index decodes `(n, m, od, oh, ow)`, loops
`kc × kd × kh × kw` with bounds checks, accumulates in float32 and
stores once; params in a by-value struct (pools/Gemm precedent). The
rank lift uses unit leading axes exactly like the oracle.

Bars: hand-computed + oracle-compare + negatives; OFF/CUDA/OpenCL legs
green; format clean; diff audit.

## Validation

Evidence in `validation.md`.
