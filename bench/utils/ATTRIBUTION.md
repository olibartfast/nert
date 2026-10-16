# Attribution — bench/utils vendors

Source: `https://github.com/olibartfast/gpu-playground`, directory
`source/utils/` (`benchmark.h`, `benchmark_helpers.h`, `cuda_helpers.h`,
`opencl_c_helpers.h`, `opencl_helpers.h`, `CMakeLists.txt`), vendored
2026-10-16 for the `bench/` measurement harness
(`specs/2026-10-16-kernel-bench-harness/`).

The algorithms and harness semantics are the reference; every file was
translated into modern C++17 (not pasted). Per-file deltas:

| NERT file (`bench/utils/`) | Playground source | Deltas |
| --- | --- | --- |
| `benchmark.hpp` | `benchmark.h` | `.h`→`.hpp`; `printf`→`<iostream>`/`printBench`; same `summarize`/`benchmark*` semantics |
| `benchmark_helpers.hpp` | `benchmark_helpers.h` | `.h`→`.hpp`; `noexcept`+`[[nodiscard]]`; same converters |
| `cuda_helpers.hpp` | `cuda_helpers.h` | `.h`→`.hpp`; `CUDA_CHECK` throws `std::runtime_error` (file:line) instead of print+`abort`; `getTime()` via `<chrono>` instead of `gettimeofday` |
| `opencl_c_helpers.hpp` | `opencl_c_helpers.h` | `.h`→`.hpp`; `CL_CHECK`/setup/build throw instead of print+`abort`; helpers live in `nert_bench` namespace |
| `opencl_helpers.hpp` | `opencl_helpers.h` | `.h`→`.hpp`; `clppBuildProgram` throws with the build log instead of print+`abort` |

CMake interface targets are renamed with the `nert_bench_` prefix
(`nert_bench_cuda_helpers`, `nert_bench_opencl_helpers`,
`nert_bench_opencl_cpp_helpers`) and carry their own SDK links, so the
vendored names never collide when both trees coexist.
