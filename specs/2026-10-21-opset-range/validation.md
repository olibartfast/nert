# Validation — Opset range (≤18) via load-time canonicalization

Defined before implementation. Each requirement maps to an observable
check. A check is not evidence until the command or walkthrough was actually
run and the result recorded below.

## Automated Checks

Run from the repository root unless noted.

```text
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
scripts/format.sh --check
```

- [ ] [V-1] Range acceptance belongs to [R-1]/[R-6]:
  synthetic opset-13, -17, and -18 models load; opset-12 fails before
  Phase B and opset-19 fails always. Each failure message names the declared
  version and the supported range.
- [ ] [V-2] Reduction canonicalization belongs to [R-2]/[R-3]/[R-7]:
  an opset-13/17 `ReduceMean`/`ReduceL2`/`ReduceMax` with `axes` produces the
  same graph shape and output as the opset-18 input form; the initializer
  name is stable and collision-free.
- [ ] [V-3] ScatterND canonicalization belongs to [R-3]:
  `reduction="none"` loads and matches the no-reduction row;
  `"add"`/`"mul"` are rejected naming the node and operator.
- [ ] [V-4] Version legality belongs to [R-4]:
  `axes` on an opset-18 reduction is rejected; the old attribute is accepted
  only where the normalizer rewrites it.
- [ ] [V-5] No unintended runtime change belongs to [R-2]/[D-1]:
  `git diff --stat` shows no changes under `src/kernels/`, `src/cuda/`,
  `src/opencl/`, `src/hip/`, `src/shape_inference.cpp`, or
  `src/executor.cpp` for Phase A.
- [ ] [V-6] Opset-18 regression belongs to [R-8]:
  OFF suite 12/12; CUDA container suite with the ResNet fixture green;
  OpenCL and HIP device suites green; format clean.
- [ ] [V-7] Real-model gate, Phase A, belongs to [R-3]/[R-6]:
  one opset-13 and one opset-17 model match ONNX Runtime within
  `1e-3·(1+|ref|)` on every output element.
- [ ] [V-8] Phase B gate belongs to [R-5]:
  an opset-11 or -12 model using old `Softmax`, `ReduceSum`, `Squeeze`,
  `Unsqueeze`, and `Split` forms matches ONNX Runtime; opset-10 still fails
  with the conversion message.
- [ ] [V-9] Docs belong to [R-8]:
  `README.md`, `specs/tech-stack.md`, `CHANGELOG.md`, and the roadmap state
  the same supported range.

## Manual Checks

- [ ] [M-1] A developer following the README can tell whether their model
  is in range and what to do below it, without reading the loader source.
- [ ] [M-2] A failure for opset 12 or 19 includes the node, operator,
  declared opset, and supported range.
- [ ] [M-3] Re-running the loader on the same model produces the same
  injected initializer name and graph (determinism).

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| — | — | pending | — | — |

## Deviations

- None yet. Record any discovered 13–17/11–12 form difference here and update
  requirements, plan, and this file in the same branch.

## Definition of Done (integration)

- [ ] Every automated and manual check executed and recorded.
- [ ] Evidence is traceable to [R-1]–[R-8].
- [ ] Spec, code, roadmap, changelog, and constitution agree.
- [ ] The >18 sibling packet can build on the shared opset plumbing.
