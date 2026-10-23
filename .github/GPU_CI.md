# GPU CI

`gpu.yml` runs independent CUDA and HIP/NVIDIA jobs on trusted pushes and
manual dispatch. Enable the repository variable `NERT_GPU_CI=true` after
registering a Linux x64 runner with labels `self-hosted, linux, x64,
nert-nvidia`.

The runner needs an NVIDIA GPU and driver, nvcc/CUDA toolkit, CMake 3.20+,
a C++17 compiler, GoogleTest/GoogleMock development packages, Python 3,
and HIP NVIDIA headers discoverable through `HIP_PATH` (or `/opt/rocm`).
cuDNN is optional and its absence does not constitute a cuDNN validation.
Jobs build in Release mode, execute all registered device suites, reject
missing or skipped device tests, and upload JUnit results and the CTest log.

The NVIDIA HIP job selects the default `NERT_HIP_PLATFORM=nvidia`. The AMD
job selects `NERT_HIP_PLATFORM=amd`, which uses CMake's native HIP language
(CMake 3.21+) and `hip::host` from ROCm, without nvcc. Register a separate
trusted Linux x64 AMD GPU runner with label `nert-amd` and set repository
variable `NERT_AMD_GPU_CI=true`. The runner needs ROCm under `/opt/rocm`,
GoogleTest/GoogleMock, CMake, a C++17 compiler and Python 3. CMake detects the
GPU architecture; `CMAKE_HIP_ARCHITECTURES=gfx942` can pin MI300X manually.
An unconfigured/skipped AMD job does not satisfy the mandatory AMD release
gate. These device suites alone do not close the full-model catalog gate.

GPU jobs are disabled until the
runner and repository variable are configured; a skipped job is not GPU proof.
