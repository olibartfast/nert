# Validation — N1c HIP Port (hipify)

Defined before implementation. Native AMD build and focused device parity
have now run on MI300X; full-model and persistent CI gates remain open.

## Checks

Mandatory 0.2.0 release gate, added by maintainer decision on 2026-10-23.
AMD Developer Cloud (AMD's vendor developer program, not a neocloud) is
the intended hardware source; access and a VM are
not provisioned by this specification. Current evidence is recorded below.
NVIDIA HIP runs cannot substitute for any AMD execution evidence.

- [x] Native AMD backend configures without nvcc/CUDA toolkit and builds
  all 72 registered HIP rows with the ROCm toolchain.
- [x] Every registered HIP device/parity test executes on a physical AMD
  GPU, with zero skips and tolerance `1e-3*(1+abs(reference))`.
- [ ] Every in-scope catalog entry has AMD full-model reference parity;
  preserve model/input/reference hashes and documented task-specific bars
  and deviations. Existing three-device PASS counts do not satisfy this.
- [x] Record base commit plus tested source snapshot hashes, GPU model, HIP/compiler/driver versions,
  build command and CTest/model logs in the validation evidence.
- [ ] CI enforces the AMD leg; an absent runner or skipped job is not PASS.

- [V-1] hipify translation committed as reviewable sources in `src/hip/`
  (no build-time translation step).
- [V-2] `NERT_ENABLE_HIP=ON` configures/builds with `hipcc` on the AMD
  runner; OFF and CUDA legs unaffected.
- [V-3] `nert_hip_*` parity within `1e-3·(1+|ref|)` on the AMD runner.
- [V-4] Scope: no `src/kernels`/loader/shape diffs; format clean.

## Evidence

2026-10-23: AMD Developer Cloud MI300X VF (`gfx942`), native Clang HIP
build without nvcc; 2/2 HIP CTest suites, 4 device + 28 kernel tests with zero
skips; shipped-runtime CPU/HIP benchmark 49/49 with zero skips/failures.
Install and independent consumer smoke passed. This does not prove all
schema modes or full-model catalog parity. Source hashes, commands, versions
and log locations: [AMD runbook](../2026-10-23-amd-devcloud/runbook.md).

Follow-up: full registered AMD CTest suite 15/15, zero skips, 8.01 seconds;
ResNet-18 opset 19, catalog YOLOv8n and Depth-Anything-V2-Small full-model
raw outputs match freshly generated ORT CPU references at the standard
scaled tolerance, twice per instance, with zero elements over tolerance.
Other AMD model legs remain UNRUN; the all-catalog checkbox stays open.

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | blocked | 2026-10-16 | no AMD GPU / ROCm / hipify here or in nert-gpu:26.08 |
| — | NVIDIA first-party translation (sibling `2026-10-17-n1c-hip-nvidia`) | pass | 2026-10-22 | 68 hipified rows validated on the RTX 3060 through the NVIDIA HIP backend; the AMD `HIP_PLATFORM=amd` half and rocBLAS/MIOpen rows remain blocked on AMD hardware. |
