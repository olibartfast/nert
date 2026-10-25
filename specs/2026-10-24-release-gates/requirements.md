# Release gate closure

- Implement Split schema-18 num_outputs, including uneven final chunks, on all four backends without changing legacy behavior. Reject conflicting sizes, invalid counts and schema misuse.
- Implement dtype-preserving Sign for float32 and int64 on CPU/CUDA/OpenCL/HIP, including integer extremes and special floating values.
- Provision and gate the remaining available catalog entries against frozen ORT references; preserve artifacts, hashes, commands and failures. The user approved deferring the unavailable keypoint-size placeholder and fixture-specific retained-detection criteria on 2026-10-24.
- Native AMD ROCm catalog testing and AMD cloud CI are excluded from this work by the user. Preserve them as open release requirements; do not count NVIDIA HIP as native AMD evidence.
- Do not weaken existing tolerances, count skipped jobs as passes, or publish a release without complete evidence.
