# Validation — Newer-than-18 opsets via verified compatibility

Defined before implementation. Implemented and validated on 2026-10-22.
The proof is a mix of generated-table freshness, loader behavior, and a real
model gate. A check is not evidence until it was run and recorded below.

## Automated Checks

Run from the repository root unless noted.

```text
python3 tools/gen_opset_compat.py --check
cmake -S . -B build -DNERT_FIXTURE="$PWD/fixtures/resnet18.onnx"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
scripts/format.sh --check
```

- [x] [V-1] Table reproducibility belongs to [D-2]/[T-3]:
  `python3 tools/gen_opset_compat.py --check` exits 0 and leaves the tree
  unchanged; the optional `nert_opset_table` CTest runs the same check when
  Python and ONNX are present. A deliberate edit fails the check until the
  generated file is refreshed.
- [x] [V-2] Stable newer declarations belong to [R-1]/[R-2]/[R-3]:
  synthetic opset-19, -21, and -27 models using `Conv`/`Relu`/`Flatten`/
  `MatMul`/`Add` load and produce the same outputs as their opset-18
  equivalent.
- [x] [V-3] Effective-version resolution belongs to [R-2]/[D-3]:
  a declared-27 model using `Add` (resolved schema 14) is accepted, not
  rejected by the declared number.
- [x] [V-4] Cast guards belong to [R-3]/[R-4]:
  plain `Cast` at declared 19 and 27 is accepted; `saturate` at 19+ and
  `round_mode` at 24+ are rejected naming the node, operator, declared
  opset, and effective version.
- [x] [V-5] Pad guard belongs to [R-4]:
  `mode="wrap"` is rejected at 19+; `constant`, `reflect`, and `edge` still
  load.
- [x] [V-6] Unsupported capabilities belong to [R-5]/[R-6]:
  `Mish` is rejected by the unknown-operator path; a FLOAT8 graph input is
  rejected by the element-type gate without reaching a kernel.
- [x] [V-7] No kernel changes belong to [R-7]:
  `git diff --stat` shows no changes under `src/kernels/`, `src/cuda/`,
  `src/opencl/`, `src/hip/`, `src/shape_inference.cpp`, or
  `src/executor.cpp`; the table and loader/gate are the runtime changes.
- [x] [V-8] Opset-18 regression belongs to [R-7]:
  OFF suite 13/13 (including the table-freshness CTest); CUDA + OpenCL
  suite 41/41 with the ResNet fixture; format clean. HIP is unavailable on
  this host.
- [x] [V-9] Real newer-than-18 model gate belongs to [R-8]:
  a native torchvision ResNet-18 export at declared 19 matches ONNX Runtime
  1.27.0 within `1e-3·(1+|ref|)` with zero elements over on CPU; OpenCL also
  passes; CUDA is recorded as a per-op gap because the host build has no
  cuDNN Conv row. A small native opset-19 graph exercises CUDA/OpenCL.
- [x] [V-10] Docs belong to [R-9]:
  README, tech-stack `Q-1`, coverage note, changelog, and roadmap all state
  the verified-compatibility policy and its limits.
- [x] [V-11] Generator-discovered guards belong to [R-4]:
  `AveragePool.dilations` (19+) and `Range.stash_type` (27+) are rejected
  when present and accepted when absent; `GridSample` at effective 20+ is
  rejected as a behavior change, while declared 19 resolves to schema 16 and
  loads.

## Manual Checks

- [x] [M-1] A developer can read the README policy and predict that a
  declared-21 model using only stable operators will load.
- [x] [M-2] A rejection for `Cast.saturate` says what is missing and points
  to the tracked follow-up; it does not suggest simply converting the model.
- [x] [M-3] A future opset with an unknown schema fails with a clear
  version/capability message, not a crash or a wrong result.

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `python3 tools/gen_opset_compat.py --check`; `ctest -R nert_opset_table`; deliberate appended edit then restore | pass | 2026-10-22 | 137 entries from onnx 1.22.0. The edited table failed the check; restoring the generated file passed; the CTest passed. |
| V-2 | `HermeticNewerStableOperators` at declared 19/21/27 vs 18 | pass | 2026-10-22 | Same inputs, outputs within `1e-6`; all loader tests 96/96. |
| V-3 | Table row `Add` since 14; declared-27 composite graph accepted | pass | 2026-10-22 | Effective schema is lower than the declaration, as required. |
| V-4 | `HermeticNewerPlainCastAccepted`, `NewerCastSaturateRejected`, `NewerCastRoundModeRejected` | pass | 2026-10-22 | Errors name `declared opset 19/24` and `effective schema 19/24`. |
| V-5 | `HermeticNewerPadModesAccepted`, `NewerPadWrapRejected` | pass | 2026-10-22 | `constant`/`reflect`/`edge` load; `wrap` names the value and effective schema. |
| V-6 | `NewerUnknownOperatorRejected`, `NewerUnsupportedDtypeRejected` | pass | 2026-10-22 | `Mish` uses the existing unsupported-op message; FLOAT8 uses the element-type gate. |
| V-7 | `git diff --stat` | pass | 2026-10-22 | Runtime changes: `src/model_loader.cpp`, `src/opset_compat.*`, generated `.inc`, CMake, tests, docs; no kernels/shape/executor. |
| V-8 | OFF `ctest` 13/13; CUDA+OpenCL `ctest --test-dir build-gpu` 41/41; `scripts/format.sh --check` 238 files | pass | 2026-10-22 | Host CUDA 12.0, RTX 3060, no cuDNN; OpenCL on the same GPU. HIP unavailable. The table-freshness CTest is present in both builds. |
| V-9 | Native torchvision ResNet-18 opset-19 export vs ORT 1.27.0: CPU max abs 2.86102294921875e-06, 0/1000; OpenCL max abs 5.7220458984375e-06, 0/1000; top-5 identical. Small native 19 graph: CPU max 4.47e-08, CUDA max 1.19e-07, OpenCL 0.0, 0/4 each | pass | 2026-10-22 | 49-node ResNet export, ops `Add/Conv/Gemm/MatMul/MaxPool/ReduceMean/Relu/Reshape`; external-data weights. CUDA ResNet leg skipped because the host build has no cuDNN Conv; the small graph covers the CUDA compatibility path. |
| V-10 | README, tech-stack Q-1, coverage note, CHANGELOG, roadmap | pass | 2026-10-22 | Policy says 11+, no upper ceiling, per-node effective-schema verification, guards and follow-ups listed. |
| V-11 | `NewerAveragePoolDilationsRejected`, `HermeticNewerAveragePoolWithoutDilationsAccepted`, `NewerRangeStashTypeRejected`, `HermeticNewerPlainRangeAccepted`, `NewerGridSampleBehaviorChangeRejected`, `HermeticGridSampleAt19Accepted` | pass | 2026-10-22 | Generator-discovered post-18 changes are guarded; absent additive attributes remain loadable. |

## Deviations

- The hand audit in the packet missed `AveragePool.dilations` (19+),
  `Range.stash_type` (27+), and the `GridSample` mode default/value change
  at 20. The generator caught them; the requirements, guards, tests, and
  roadmap follow-ups were updated in this slice.
- `Pad`’s new `mode="wrap"` value is a reviewed generator override because
  ONNX does not machine-encode string value domains in schema metadata.
- `GridSample` 20+ is rejected as a whole behavior change rather than
  attempting to accept a subset of explicit modes; the kernel still
  implements the schema-16 names/default.
- The CUDA ResNet leg is a per-op gap on this host: the build has no cuDNN,
  so `Conv` has no CUDA row. OpenCL runs the full ResNet; a small native
  opset-19 graph covers CUDA.
- The generator is development-only and requires the pinned `onnx` package.
  When Python and ONNX are available, `test/CMakeLists.txt` registers the
  `nert_opset_table` CTest, so CI and development `ctest` runs enforce table
  freshness without adding a Python runtime dependency to NERT.

## Definition of Done (integration)

- [x] Every automated and manual check executed and recorded.
- [x] Evidence is traceable to [R-1]–[R-9].
- [x] The generated table is reproducible and reviewed.
- [x] Rejected capabilities have linked follow-up slices in the roadmap.
- [x] Spec, code, roadmap, changelog, and constitution agree.
