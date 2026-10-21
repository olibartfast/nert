# Validation — Opset range (≤18) via load-time canonicalization

Defined before implementation. Phase A (declared opsets 13–18) was validated
on 2026-10-21; Phase B (11–12) on 2026-10-22. The supported range is now
**11–18**. Each requirement maps to an observable check; a check is not
evidence until the command or walkthrough was actually run and the result is
recorded below.

## Automated Checks

Run from the repository root unless noted.

```text
cmake -S . -B build -DNERT_FIXTURE="$PWD/fixtures/resnet18.onnx"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
scripts/format.sh --check
```

- [x] [V-1] Range acceptance belongs to [R-1]/[R-6]:
  synthetic opset-11, -12, -13, -14, -16, -17, and -18 models load; opset-10
  and -19 fail and each message names the declared version and `11..18`.
- [x] [V-2] Reduction canonicalization belongs to [R-2]/[R-3]/[R-7]:
  an opset-13 `ReduceMean`/`ReduceL2`/`ReduceMax` with `axes` produces a
  two-input node with an Int64 initializer and the same output as the
  opset-18 input form; the name is stable and collision-free.
- [x] [V-3] ScatterND canonicalization belongs to [R-3]:
  `reduction="none"` loads at 16 and 18 and matches the no-reduction row;
  `"add"` is rejected naming the node and operator; the attribute is
  rejected before opset 16.
- [x] [V-4] Version legality belongs to [R-4]:
  old attributes are rejected from opset 13; old inputs are rejected before
  13; `noop_with_empty_axes` is rejected before 13; legacy `Softmax` rejects
  unknown input rank and an out-of-range axis; empty `Unsqueeze` axes and
  empty `Split` split are rejected.
- [x] [V-5] No unintended runtime change belongs to [R-2]/[D-1]:
  the diff contains no changes under `src/kernels/`, `src/cuda/`,
  `src/opencl/`, `src/hip/`, `src/shape_inference.cpp`, or
  `src/executor.cpp`.
- [x] [V-6] Opset-18 regression belongs to [R-8]:
  OFF suite 12/12; CUDA + OpenCL suite 40/40 with the ResNet fixture;
  format clean. HIP is unavailable on this host (no HIP headers).
- [x] [V-7] Real-model gate, Phase A, belongs to [R-3]/[R-6]:
  a real-weight ResNet-18 rewritten to valid opset-13 and opset-17 syntax
  matches ONNX Runtime 1.27.0 within `1e-3·(1+|ref|)` on every output
  element, and matches the opset-18 NERT result exactly.
- [x] [V-8] Phase B gate belongs to [R-5]:
  an opset-11 and an opset-12 model using old `Softmax`, `ReduceSum`,
  `Squeeze`, `Unsqueeze`, and `Split` forms match ONNX Runtime; a
  real-weight opset-11 ResNet-18 with a legacy `Softmax` on an intermediate
  matches too; opset 10 still fails with the conversion message.
- [x] [V-9] Docs belong to [R-8]:
  `README.md`, `specs/tech-stack.md`, `CHANGELOG.md`, and the roadmap state
  the same supported range.

## Manual Checks

- [x] [M-1] A developer following the README can tell whether their model
  is in range and what to do below it, including the legacy-`Softmax`
  `value_info` requirement, without reading the loader source.
- [x] [M-2] A failure for opset 10/19, an old attribute at 13, a legacy
  `Softmax` rank/axis problem, or an empty `Unsqueeze`/`Split` list includes
  the node, operator, declared opset, and the reason.
- [x] [M-3] Re-running the loader on the same model produces the same
  injected initializer names and the same expanded `Softmax` graph
  (determinism).

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-1 | `nert_loader_test --gtest_filter='EngineLoader.*:EngineLoaderNegative.*'` with `NERT_FIXTURE=fixtures/resnet18.onnx`: 82/82; `ctest -R nert_loader`: 2/2 | pass | 2026-10-22 | Synthetic 11/12/13/14/16/17/18; 10 and 19 reject with `11..18`; missing declaration rejects. |
| V-2 | Canonicalization tests for ReduceMean/ReduceL2/ReduceMax; collision; empty/invalid axes; `HermeticOpset13ReduceMeanMatchesOpset18` | pass | 2026-10-21 | `reduce0/axes`; collision yields `reduce0/axes_1`; empty axes canonicalizes to an omitted input (reduce-all, mean 11.5); duplicate/out-of-range axes fail shape inference; 13-vs-18 output diff 0.0. |
| V-3 | ScatterND `none` at 16 and 18; `add` rejected; attribute rejected at 13 | pass | 2026-10-21 | Attribute dropped; other reductions name the node/op. |
| V-4 | `Legacy*` rejection tests plus the Phase A version-legality tests | pass | 2026-10-22 | Old attributes at 13, old inputs at 11, `noop_with_empty_axes` before 13, missing/empty Unsqueeze axes, empty Split, unknown Softmax rank, and out-of-range Softmax axis all fail with the node/op/version in the message. |
| V-5 | `git diff --stat` | pass | 2026-10-22 | Runtime changes are `src/model_loader.cpp`, `src/opset_normalize.*`, CMake, tests, docs; no kernel/shape/executor files. |
| V-6 | OFF: `ctest` 12/12. CUDA+OpenCL: `ctest --test-dir build-gpu` 40/40. `scripts/format.sh --check` 236 files | pass | 2026-10-22 | Host CUDA 12.0, RTX 3060; OpenCL on the same GPU. HIP unavailable (no `hip/hip_runtime_api.h`). |
| V-7 | ResNet-18 real weights, deterministic input; ORT 1.27.0 vs NERT 13/17: max abs 2.86102294921875e-06, 0/1000 over tolerance; NERT 13/17 vs NERT 18: 0.0 | pass | 2026-10-21 | The ONNX version converter lacks a Relu 14→13 adapter, so the gate rewrote the declaration and the ReduceMean/Reshape syntax to valid 13/17 forms; weights and graph are the provisioned ResNet-18. |
| V-8 | Opset-11/12 legacy-ops model vs ORT 1.27.0: max abs 1.1920929e-07, 0/24. ResNet-18 real weights at opset 11 + legacy Softmax on an intermediate: max abs 8.19563866e-08, 0/1000, top-5 classes identical. Small chain CPU vs CUDA: 0.0; CPU vs OpenCL: 1.1920929e-07 | pass | 2026-10-22 | The small graph exercises `Softmax` (old default axis 1), `ReduceSum`, `Squeeze`, `Unsqueeze`, `Split`, `Concat`, and `Add` at both 11 and 12. The ResNet declaration/syntax was rewritten to valid opset-11 forms (ReduceMean axes attribute, no `allowzero`) and a legacy Softmax appended; weights unchanged. Opset 10 still rejects with `11..18`. |
| V-9 | README, tech-stack Q-1, coverage note, CHANGELOG, roadmap | pass | 2026-10-22 | Policy now says 11–18; >18 remains planned. |

## Deviations

- Legacy `Softmax` uses an inserted `Shape` node to supply the original
  shape to the final `Reshape`, rather than injecting a static shape
  initializer. This keeps intermediate-input Softmax working without
  requiring a fully static shape constant at load; only the input rank is
  required from the model's shape metadata (`value_info` for intermediates).
  Missing rank fails fast rather than guessing.
- The opset-11 `Resize` input layout is the same as 13+ (`X, roi, scales,
  sizes`); NERT already accepted empty roi/scales plus a sizes input, so no
  rewrite was added. A positive opset-11 Resize test records the discovery.
- The loader CTest filters were `NertLoader.*`/`NertLoaderNegative.*` while
  the suites are `EngineLoader.*`/`EngineLoaderNegative.*`; the filters were
  corrected so the loader suite actually runs in CTest/CI.
- The stale `ConstantOtherDtypeRejected` case tested INT32 (which the loader
  supports) and a malformed payload; it now tests UINT8 and FLOAT16 with
  correctly sized payloads.
- The Phase A real-model gate used a rewritten declaration/syntax, not an
  exporter’s native opset-13 file, because `onnx.version_converter` cannot
  cross the Relu 14→13 schema boundary. The computation and weights are
  unchanged.
- The loader folds `Constant` nodes during graph decoding (the separate
  `FoldConstants` API is not on the load path), so the normalizer runs after
  that fold rather than before it. This also makes generated axes names
  avoid folded initializer names. The packet wording was corrected to match
  the code; the observable order relative to shape inference is unchanged.
- HIP was not run: the host has no HIP headers. The legacy-`Softmax`
  expansion uses `Shape`/`Flatten`/`Reshape`, which the HIP pilot table does
  not yet include; existing HIP rows are unaffected.

## Definition of Done (integration)

- [x] Every Phase A and Phase B automated and manual check executed and
      recorded.
- [x] Evidence is traceable to [R-1]–[R-8].
- [x] Spec, code, roadmap, changelog, and constitution agree on 11–18.
- [x] The >18 sibling packet can build on the shared opset plumbing.
