# Validation — Newer-than-18 opsets via verified compatibility

Defined before implementation. The proof is a mix of generated-table
freshness, loader behavior, and a real model gate. A check is not evidence
until it was run and recorded below.

## Automated Checks

Run from the repository root unless noted.

```text
python3 tools/gen_opset_compat.py --check
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
scripts/format.sh --check
```

- [ ] [V-1] Table reproducibility belongs to [D-2]/[T-3]:
  `python3 tools/gen_opset_compat.py --check` exits 0 and leaves the tree
  unchanged. A deliberate schema-table edit fails the check until the
  generated file is refreshed.
- [ ] [V-2] Stable newer declarations belong to [R-1]/[R-2]/[R-3]:
  synthetic opset-19, -21, and -27 models using only `Add`/`Conv`/`Relu`/
  `MatMul` load and produce the same outputs as their opset-18 equivalent.
- [ ] [V-3] Effective-version resolution belongs to [R-2]/[D-3]:
  a declared-27 model using `Add` (resolved schema 14) is accepted, not
  rejected by the declared number.
- [ ] [V-4] Cast guards belong to [R-3]/[R-4]:
  plain `Cast` at declared 27 is accepted; `saturate` at 19+ and
  `round_mode` at 24+ are rejected naming the node, operator, and effective
  version.
- [ ] [V-5] Pad guard belongs to [R-4]:
  `mode="wrap"` is rejected at 19+; `constant`, `reflect`, and `edge` still
  work.
- [ ] [V-6] Unsupported capabilities belong to [R-5]/[R-6]:
  an operator introduced after the implemented set is rejected by the
  unknown-operator path; float8/int4 or other unsupported dtypes are
  rejected without reaching a kernel.
- [ ] [V-7] No kernel changes belong to [R-7]:
  `git diff --stat` shows no changes under `src/kernels/`, `src/cuda/`,
  `src/opencl/`, `src/hip/`, `src/shape_inference.cpp`, or
  `src/executor.cpp`; the table and loader are the only runtime changes.
- [ ] [V-8] Opset-18 regression belongs to [R-7]:
  OFF suite 12/12; CUDA container suite with the ResNet fixture green;
  OpenCL and HIP device suites green; format clean.
- [ ] [V-9] Real newer-than-18 model gate belongs to [R-8]:
  native export at declared 19+ matches ONNX Runtime 1.27.0 within
  `1e-3·(1+|ref|)` with zero elements over tolerance on the CPU leg; record
  CUDA/OpenCL legs or the per-op gap.
- [ ] [V-10] Docs belong to [R-9]:
  README, tech-stack `Q-1`, coverage note, changelog, and roadmap all state
  the verified-compatibility policy and its limits.

## Manual Checks

- [ ] [M-1] A developer can read the README policy and predict whether a
  declared-21 model using only stable operators will load.
- [ ] [M-2] A rejection for `Cast.saturate` says what is missing and points
  to the follow-up slice; it does not suggest simply converting the model.
- [ ] [M-3] A model declaring a future opset with an unknown schema fails
  with a clear version/capability message, not a crash or a wrong result.

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | pending | — | — |

## Deviations

- None yet. Any value-domain or behavior change found by the real gate is
  recorded here and fed back into the generator and guards before landing.

## Definition of Done (integration)

- [ ] Every automated and manual check executed and recorded.
- [ ] Evidence is traceable to [R-1]–[R-9].
- [ ] The generated table is reproducible and reviewed.
- [ ] Rejected capabilities have linked follow-up slices in the roadmap.
- [ ] Spec, code, roadmap, changelog, and constitution agree.
