# AMD Developer Cloud — reproducible HIP validation

Date: 2026-10-23. Base commit: `d906f7f`, with the local HIP compute,
OpenCL parity and AMD CMake changes. This is validation of a working-tree
snapshot, not a claim that those changes are committed or released.

## Access and VM

1. Sign in to the AMD AI Developer Program / AMD Developer Cloud in your
   own browser. Login and cloud terms stay with the account owner.
2. Add an existing **public** SSH key, e.g. `~/.ssh/id_ed25519.pub`.
   Never send passwords or private keys or commit them into the repository.
3. Select a ROCm development image with an AMD GPU. This run used one
   **AMD Instinct MI300X VF**, architecture **gfx942**. Other AMD GPUs
   and ROCm releases are not validated by this run.
4. Copy the SSH connection shown in the console. The tested VM used root
   and the default SSH port. Substitute your own user and host below:

   ```bash
   ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 root@<VM_IP>
   ```

   The first attempt without `IdentitiesOnly=yes` failed with "Too many
   authentication failures" because the SSH agent offered other identities.
   Selecting the intended key fixed access.

Official guidance: [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
and [getting started](https://www.amd.com/en/developer/resources/technical-articles/2025/how-to-get-started-on-the-amd-developer-cloud-.html).

## Toolchain preflight

On the VM:

```bash
id
uname -r
/opt/rocm/bin/hipcc --version
/opt/rocm/bin/rocminfo
cmake --version
command -v g++
command -v nvcc || true
```

Observed: CMake 3.28.3, GNU C++ 13.3.0, HIP 7.16.26385,
AMD Clang 24.0.0git (`d6f6cb691863e9258034ab48c1dfce21689d5606`),
kernel `6.8.0-146-generic`; `/opt/rocm` resolves into the image's
`core-10.1` installation. No nvcc was found. This development image has no
`/opt/rocm/.info/version`, so no separate ROCm release number is inferred.

The VM lacked GoogleTest/GoogleMock development packages:

```bash
apt-get update -qq
apt-get install -y libgtest-dev libgmock-dev
```

APT reported a missing signing key for the image's AMD stable repository.
No signing checks were disabled or trust settings changed. The Ubuntu
GoogleTest/GoogleMock installation succeeded using the available indexes.

## Build changes

Previously `NERT_ENABLE_HIP=ON` always enabled CMake CUDA and defined
`__HIP_PLATFORM_NVIDIA__`. It could not build on an AMD-only host.

The new cache choice is `NERT_HIP_PLATFORM=amd|nvidia` (default `nvidia`).
AMD requires CMake 3.21+, enables the native HIP language, compiles
`src/hip/*.cu` as HIP, links `hip::host` privately, and uses HIP C++17.
NVIDIA retains nvcc and the existing HIP-header path. An invalid platform
fails at configure time. Installed AMD package configs resolve `hip` before
loading the exported NERT target.

The public device API stays toolkit-free. The four compute kernels
(`Gemm`, `MatMul`, `Conv`, `ConvTranspose`) are first-party implementations;
neither backend needs hipBLAS, rocBLAS or MIOpen for these rows.

## Transfer a bounded source snapshot

From the local repository:

```bash
tar -czf /tmp/nert-amd-source.tar.gz \
  CMakeLists.txt versions.env include src test tools
sha256sum /tmp/nert-amd-source.tar.gz
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 root@<VM_IP> \
  'mktemp -d /tmp/nert-amd-validation.XXXXXX'
scp -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
  /tmp/nert-amd-source.tar.gz root@<VM_IP>:<REMOTE_DIR>/source.tar.gz
scp -r -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 \
  bench root@<VM_IP>:<REMOTE_DIR>/
```

This transfers repository sources, not `.git`, account credentials, SSH
keys, sibling repositories, or model caches. Keep and compare the archive
hash on both hosts. The initial tested archive hash was
`6061e34aa702f80eab0c83ecc0c0f2c84d58f87cfa8e5bcaaf8ebbc36952bd84`.
Afterwards `CMakeLists.txt` was updated only to add AMD installed-package
dependency discovery and the package was rebuilt. Its tested final hash is
`7d396a95decc347a61f9b2331a0a74a0ccf1b1cf49ebc2c83020f190ddaed7de`.
The benchmark sources were transferred separately without modification.

## Native AMD configure and build

Inside the remote directory:

```bash
sha256sum source.tar.gz
tar -xzf source.tar.gz
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNERT_ENABLE_HIP=ON -DNERT_HIP_PLATFORM=amd \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build --parallel 8 > build.log 2>&1
ldd build/test/nert_hip_kernels_test
```

The link resolved `libamdhip64.so.7` and AMD ROCm dependencies, not CUDA.
The architecture pin is specific to MI300X; select the architecture reported
by `rocminfo` on other hardware.

## Device tests and shipped-kernel parity

```bash
ctest --test-dir build -R '^nert_hip_' --no-tests=error \
  --output-on-failure --output-junit hip-results.xml
./build/test/nert_hip_device_test --gtest_output=xml:hip-device.xml
./build/test/nert_hip_kernels_test --gtest_output=xml:hip-kernels.xml
cmake -S . -B build -DNERT_BUILD_BENCH=ON
cmake --build build --target nert_bench_runtime --parallel 8
./build/bench/runtime/nert_bench_runtime > bench-results.log 2>&1
```

Results: 2/2 HIP CTest suites pass; 4/4 device and 28/28 kernel GoogleTest
cases pass with no skips. The table probe checks all 72 names. The runtime
benchmark reports **49 passed, 0 skipped, 0 failed**, comparing the shipped
HIP kernels with the CPU oracle at `1e-3*(1+abs(reference))`, exact for
integer/bool outputs. These fixtures are not exhaustive testing of all 72
operators or all their schema modes. The 8 hermetic executor tests were
also run and passed; they are CPU/fake-device tests, not AMD model gates.

Evidence hashes:

| Artifact | SHA-256 |
| --- | --- |
| `bench-results.log` | `3de490530b933d9261efb05eba9f70318e8dd544150fc2231e7ce7c77035060c` |
| `hip-device.xml` | `699a4f96cfab8832d5d6a453f05fe9da7437f0ded0062473c8cabc26111e86c3` |
| `hip-kernels.xml` | `aeb744864ade069638f069729921d48bbdeda5b97985ffe8e7e94148e87efd78` |

Logs were retrieved locally under `/tmp/nert-amd-evidence/`; the raw
benchmark output is preserved in [bench-results.txt](bench-results.txt).
One log-transfer connection was refused temporarily; retrying SSH succeeded.
Keep source,
hardware/toolchain metadata and logs together before removing the VM;
`/tmp` is not durable artifact storage.

## Install and downstream consumer

The run also installed into a temporary prefix, configured an independent
C++17 consumer with `find_package(nert CONFIG REQUIRED)`, linked
`nert::nert`, and called `HipDevice()` plus the `ConvTranspose` table lookup.
The build and execution succeeded and printed `hip`.

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<REMOTE_DIR>/install
cmake --build build --parallel 8
cmake --install build
cmake -S consumer -B consumer-build \
  -DCMAKE_PREFIX_PATH='<REMOTE_DIR>/install;/opt/rocm'
cmake --build consumer-build
./consumer-build/consumer
```

## Local regression checks and release state

Follow-up: the full AMD build's registered CTest suite passed **15/15 in
8.01 seconds**, including the benchmark, with no skips. This VM has no
development-only Python ONNX package, so the optional generated-table check
was not registered there; that check passed locally. The ResNet fixture was
transferred from the existing local export, not downloaded again.

Three complete models were then compared with newly generated ONNX Runtime
1.27.0 CPUExecutionProvider references, using the exact frozen inputs.
Each HIP model instance ran twice. Every output was finite and every element
met `1e-3*(1+abs(reference))`; classification/detection/depth postprocessing
is not part of this raw-output gate.

| Model | Dispatched nodes | Output elements | Over tolerance (both runs) | Max absolute difference | Run 1 / run 2 |
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet-18, declared opset 19 | 49 | 1,000 | 0 | 2.86102e-06 | 8.60 / 6.85 ms |
| YOLOv8n, opset 18 | 260 | 705,600 | 0 | 0.000686646 | 91.08 / 57.37 ms |
| Depth-Anything-V2-Small, opset 18 | 553 | 268,324 | 0 | 9.60827e-05 | 130.21 / 96.46 ms |

The manual runner is committed as `tools/hip_model_check.cpp`; it deliberately
supports only static single-input/single-output float32 models and rejects
non-finite values or an incorrectly sized input/reference. It is not an
automatic all-catalog test runner. Compile and run from the source directory:

```bash
/opt/rocm/bin/hipcc -std=c++17 -O2 -I include \
  tools/hip_model_check.cpp -Lbuild -lnert -o build/hip_model_check
timeout 120 ./build/hip_model_check model.onnx input.bin ort-reference.bin
```

Use `-Lbuild -lnert`, not a positional `build/libnert.a` on this image:
hipcc's implicit `-x hip` otherwise parsed the archive as source. The first
manual compile failed for that reason; the corrected link command succeeded.
Each model check had a 120-second timeout. The observed inference times
exclude construction (80–98 ms), transfer of source/model files and ORT
reference generation.

Reference generation used local ONNX Runtime, not software installed on the
VM. For each model, run `InferenceSession(model, providers=['CPUExecutionProvider'])`,
load the frozen input using `numpy.fromfile(path, dtype=numpy.float32)` and
reshape to the input shape; save `session.run(None, {input_name: input})[0]`
with `.astype(numpy.float32).tofile(reference_path)`. YOLO and depth references
were byte-identical to their older frozen references. The unpaired older
ResNet reference differed and was not used; only the regenerated reference
for the exact selected input/model is the basis of the reported gate.

Exact tested file hashes:

| File | SHA-256 |
| --- | --- |
| runner | `db841c9c4a31e7f6a9a968f2f6cedddd6704f54494a225436da687c401c52eb2` |
| ResNet-18 model | `81db52082b05aed3e09b86332e55b931d6bd136432a16c29669437d599b89988` |
| ResNet external weights | `30b1aa4bdeba9751b30d868ef749ccb8c7fcb1be491e854eeca3f52369a00267` |
| ResNet input | `f58703a9ac8ecda4f75937f08872426f72aad262382ce5540cb4809f09a17608` |
| ResNet raw ORT reference | `05c86e5712fa2237bee71ac799ad094349ed7422214d881adbc83ee527b30d67` |
| YOLOv8n model | `fc34eff309da7fbd723c597f9621bac0b8e55695cbd102f6807c18bda325edf5` |
| YOLOv8n input | `8c220027a27d62268fa1da2c52f8a0ce820d57b7a86b723062f648457128d017` |
| YOLOv8n raw ORT reference | `1d5b1963ebdc8fe777ef1d1397feaf77c03021845a43246277a4dcc7ca39c1b6` |
| Depth model | `497acfffe67bffb5a00cc2aeaa11c91f033266385134ab12ebb26c36b9748492` |
| Depth input | `6a50426dda0294930cce1398dbeaad33a8cc3563a6d04faf3102464a95d378c4` |
| Depth raw ORT reference | `c8e898628d5eb91ccb7b296f4f0ec81f4cb13343a68bcb254f8bed90a711c0f7` |

Raw per-model logs and the full-suite output are preserved beside this
runbook. ResNet-18 opset 19 is an additional compatibility gate, not evidence
for a different catalog artifact. The other catalog AMD legs are still UNRUN.

- Clean OpenCL Release build on RTX 3060: 30/30 CTest suites pass, including
  new `Abs`, `Neg`, `Floor`, `GreaterOrEqual` tests (float special values,
  scalar input, int64 precision, broadcasting, invalid input rejection).
- HIP/NVIDIA reconfigure/build after AMD wiring: both HIP suites pass.
- Scoped clang-format-18, YAML parse and `git diff --check` pass.
- CI now contains a separate AMD job. It needs a trusted runner labelled
  `nert-amd` and repository variable `NERT_AMD_GPU_CI=true`; neither is
  registered/activated by this run. A skipped CI job is not validation.
- **The mandatory AMD release gate is not closed.** Full-model AMD legs
  for the catalog, complete operator-family test coverage and active CI
  evidence remain outstanding. Six in-scope catalog groups also still lack
  exports. No tag or release was created.
- At the owner's request, `sync; shutdown -h now` was sent via SSH after all
  logs were saved locally; SSH returned success. Cloud-resource deletion is
  not verified. The owner was advised to delete the droplet in the console,
  since power-off alone does not stop billing. No cloud API deletion was
  performed by this run.
