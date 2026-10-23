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

The HIP job validates the NVIDIA HIP backend. The current CMake configuration
requires CUDA and defines `__HIP_PLATFORM_NVIDIA__`; an AMD runner alone
does not enable ROCm. AMD build wiring and hardware validation remain open.

Apply these files to the OpenCode candidate branch along with its kernel
changes before dispatching that branch. GPU jobs are disabled until the
runner and repository variable are configured; a skipped job is not GPU proof.
