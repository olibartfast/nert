# HIP review — OpenCode candidate 0279876

Reviewed the first-party compute rows (`Gemm`, `MatMul`, `Conv`,
`ConvTranspose`), HIP table registration and CMake/test integration, alongside
the version-aware changes inspected in the earlier review.

## Findings

- P2 — `test/CMakeLists.txt:385` on the candidate: only
  `hip_kernels_test.cpp` is registered for HIP kernel parity. Its 28 cases
  cover the pilot and four compute rows, while the table advertises 72
  operators. The adapted CUDA suites cited as 149 passing cases in the HIP
  validation document are not registered as reproducible repository tests.
  Commit and register those adaptations (or backend-parameterize the shared
  suites), particularly for indexing, reductions, bool movement and legacy
  schema forms. The new HIP CI job exposes this existing coverage boundary;
  a green result does not validate every table row.

## Evidence and limits

- RTX 3060 Laptop GPU: existing candidate `build-hip` binaries,
  `ctest --test-dir build-hip -R '^nert_hip_' --output-on-failure`: 2/2 pass.
- Direct `nert_hip_kernels_test`: 28/28 pass, no skips; JUnit saved locally
  at `/tmp/nert-hip-review.xml`.
- These are existing build artifacts, not an independent clean HIP rebuild.
- AMD/ROCm execution remains unrun. CMake currently selects CUDA and defines
  `__HIP_PLATFORM_NVIDIA__`; AMD support also needs explicit build wiring.
- No additional blocking defect established in the inspected compute rows.
  This bounded review does not establish correctness of every hipified row.

## CI addition

`.github/workflows/gpu.yml` adds separate CUDA and HIP/NVIDIA jobs, selected
by a matrix. Device suites must exist and execute without GoogleTest skips.
See `GPU_CI.md` for runner provisioning and activation. The workflow is
prepared in the main checkout; it has not been published or dispatched.
