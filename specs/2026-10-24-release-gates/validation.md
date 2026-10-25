# Validation

Defined before implementation; results updated after execution.

- [x] PASS: Split loader, shape and execution contracts, including old-schema/
  non-positive loader attributes, conflicting sizes, uneven chunks, invalid
  dim2/count3 and dim6/count4 partitions, and preserved ModelLoadException.
- [x] PASS: Sign float32/int64 CPU/device parity, integer extremes, signed
  zero, infinities, NaN propagation, and invalid-shape rejection.
- [x] PASS: rebuilt CPU 14/14, CUDA/cuDNN 27/27, OpenCL 32/32 suites;
  HIP/NVIDIA selected 3/3 suites, including existing kernels and shared
  CPU/HIP release regressions. CUDA missing-row test now executes with cuDNN.
- [x] PASS: generated schema catalog (290 entries), formatting, strict checker
  equality/mismatch/NaN/extra-byte smoke checks, exact regeneration of all 23
  frozen input/reference files, and preserved commands/hashes.
- [x] APPROVED DEVIATION: five available representatives, all 15/15 local
  device task checks pass. Seven strict raw passes and eight preserved raw
  failures under the user-approved retained-detection criterion; see results.
- [x] APPROVED DEFERRAL: sixth placeholder has no distinct upstream model;
  named keypoint sizes would reuse Preview. Preview remains covered by entry 7.
- [x] PASS: independent read-only reviewer sign-off in review.md; final preserved-evidence acceptance uses `python3 specs/2026-10-24-release-gates/acceptance.py` (result in evidence/acceptance.json).
- EXCLUDED by user: native AMD full-model catalog gates.
- EXCLUDED by user: persistent AMD cloud CI.

Recovered session: ses_e95444623ffegPuDxHToNSWm7S. Base: 95027c3.
No release is published; future releases must follow GitFlow.
