# Validation — Version-aware opsets

Defined before implementation. A check is not evidence until it is run and
recorded below. Each phase uses the same full scoreboard; targeted checks are
listed for the phase under change.

## Automated checks

From the repository root:

```sh
python3 tools/gen_opset_compat.py --check
cmake -S . -B build -DNERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx
cmake --build build --parallel "$(nproc)"
NERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx \
  ctest --test-dir build --output-on-failure
scripts/format.sh --check
```

The fixture path is host-specific. Use the provisioned ResNet-18 export that
matches the existing model gate (the pinned local reference is
`resnet18_op19.onnx`, 44 initializers, `ReduceMean`/`Reshape`). The older
`resnet18_export/resnet18.onnx` is a different export and fails fixture-content
assertions.

## Phase A — schema metadata and full-history catalog

- [x] [V-1] Catalog coverage belongs to [R-1]/[R-2]/[T-4]: the generated
  table contains every schema version selected by declarations 11, 13, 18,
  19, and 27 for the implemented operators; `--check` exits 0 and produces
  no diff. Verified 2026-10-22: 72 operators, 288 entries, every ONNX history
  version present, no diff.
- [x] [V-2] Node metadata belongs to [R-1]/[T-5]: a declared-13 `ReduceMean`
  node carries schema 13; a declared-18 node carries schema 18; a declared-27
  `Add` carries schema 14; `Graph.declared_opset` equals the declaration.
  Verified 2026-10-22 by the focused loader tests.
- [x] [V-3] Missing effective schema belongs to [R-6]/[T-7]: a node with no
  schema at or below the declaration fails with the node, operator, declared
  opset, and the missing effective schema. Verified 2026-10-22 with
  `LayerNormalization` at declared 11.
- [x] [V-4] Behavior preservation belongs to [R-8]: all existing loader,
  shape, planner, executor, kernel, and model tests pass unchanged except for
  the documented rejection of an operator introduced after the declaration;
  the ResNet-18 gate still loads and executes. Verified 2026-10-22 with
  `NERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx`: 13/13 CTest entries
  passed, including `nert_loader` (104/104), `nert_executor`, and
  `nert_opset_table`. Environment note:
  `/tmp/opencode/resnet18_export/resnet18.onnx` is a different export
  (`GlobalAveragePool`/`Flatten`, 42 initializers) and is not the pinned
  reference (`ReduceMean`/`Reshape`, 44 initializers). The earlier failed
  scoreboard was a fixture-selection failure, not a code failure.
- [x] [V-5] Transitional honesty belongs to [D-5]: the normalizer and its
  tests still exist and pass in Phase A; no documentation claims the
  canonical target is removed yet. Verified 2026-10-22: the normalizer
  sources are untouched and its negative/expansion tests pass.

## Phase B — reduce family

- [x] [V-6] Reduce forms belong to [R-4]/[R-7]: a declared-13
  `ReduceMean`/`ReduceL2`/`ReduceMax` with `axes` executes the old form
  without an inserted axes initializer and matches the declared-18 result.
  Verified 2026-10-22: the loader keeps one input plus the `axes` attribute
  and creates no `reduce0/axes*` initializer; CPU kernel, shape-inference,
  CUDA, and OpenCL tests execute both forms and agree. Empty present axes
  inputs are covered: `ReduceMean` with noop identity, omitted/Unknown axes,
  and `ReduceMax` reducing all axes. Existing schema-0 / input-form tests
  still pass.
- [x] [V-7] Loader legality belongs to [R-6]: axes as an input before 18,
  axes as an attribute at 18+, and `noop_with_empty_axes` before 18 are
  rejected with node/operator/declared/effective context. The pre-18 empty
  string second operand is rejected too. `ReduceMax` schema 18+
  `noop_with_empty_axes` is reported as an explicit capability gap (the
  identity behavior is a tracked follow-up), not as an unknown attribute.
  Verified 2026-10-22 by the five `EngineLoaderNegative` reduce-form tests;
  messages name the node, operator, offending form, declared opset 13 (or
  18), and effective schema 13 (or 18). `nert_loader` passes 104/104; the explicit `noop_with_empty_axes=0` form is accepted.
- [x] [V-8] Device parity belongs to [R-7]: CPU and every enabled GPU device
  that has the reduce row produce matching results for both forms; missing
  HIP rows are recorded as gaps, not passes. Verified 2026-10-22: CUDA
  `ReduceMean`/`ReduceL2`/`ReduceMax` attribute-form tests match the CPU
  oracle; OpenCL `ReduceMean` (`index_extra`), `ReduceL2` (`owlv2_rows`), and
  `ReduceMax` (`reduce_extra`) do the same. HIP is a recorded gap: this build
  has HIP off, and `src/hip/` has no `ReduceMean`/`ReduceL2`/`ReduceMax` row
  (only add/div/mul/relu/reshape/sigmoid/softmax/sub/tanh), so parity cannot
  be claimed there.

## Phase C — axes, Split, and ScatterND

- [ ] [V-9] Old attribute forms for `ReduceSum`, `Squeeze`, `Unsqueeze`, and
  `Split` execute in place and match their input-form equivalents.
- [ ] [V-10] `ScatterND.reduction="none"` remains accepted from schema 16;
  other values and pre-16 use are rejected.
- [ ] [V-11] `Squeeze` / `Unsqueeze` kernels remain layout-only; only shape
  inference resolves axes.

## Phase D — legacy Softmax and normalizer removal

- [ ] [V-12] Legacy Softmax belongs to [R-5]/[R-7]: declared-11/12 Softmax
  with default axis 1 and explicit axes matches ONNX Runtime or the frozen
  pre-refactor output within the existing tolerance; no rank metadata is
  required for an intermediate input.
- [ ] [V-13] The direct kernel uses `outer = prod(prefix)`,
  `dim = prod(suffix)`, `inner = 1`; CPU/CUDA/OpenCL/HIP agree where the row
  exists.
- [ ] [V-14] Normalizer removal belongs to [R-3]/[T-19]: no
  `opset_normalize` source, build entry, include, call, or test remains.
- [ ] [V-15] No global opset constant belongs to [R-3]: no
  `BASELINE_OPSET` or `opset-18 IR` policy remains in runtime or generator.

## Phase E — catalog/docs cleanup

- [ ] [V-16] Catalog policy belongs to [R-2]/[D-4]: the table resolves the
  effective schema and gates implemented capabilities without a baseline-18
  comparison; freshness still fails after a deliberate edit.
- [ ] [V-17] Documentation belongs to [R-8]: README, tech-stack Q-1, roadmap,
  coverage, and CHANGELOG describe per-node effective schemas and the
  remaining rejected post-18 capabilities.
- [x] [V-18] Formatting and warnings: `scripts/format.sh --check` exits 0;
  the build uses the existing `-Wall -Wextra -Wpedantic` policy. Verified
  2026-10-22: 239 files formatted; build warnings are pre-existing test
  warnings, not new production warnings.

## Evidence table

| Check | Command | Result | Date | Notes |
|---|---|---|---|---|
| V-1 | `python3 tools/gen_opset_compat.py --check` | pass | 2026-10-22 | `opset compatibility table is current (288 entries)`; every ONNX history version is present for all 72 implemented operators. |
| V-2 | `NERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx ./build/test/nert_loader_test --gtest_filter='EngineLoader.HermeticNodeSchemaVersionTracksDeclaredOpset:EngineLoader.HermeticEffectiveSchemaBelowEleven:EngineLoader.HermeticNewerAddResolvesSchemaFourteen:EngineLoader.HermeticLegacySoftmaxExpansionKeepsMetadata'` | pass (4/4) | 2026-10-22 | ReduceMean declared 11/13/18/19/27 resolves to 11/13/18/18/18; declared-11 Relu resolves to 6; declared-27 `Add` resolves to 14; `Graph.declared_opset` matches each declaration. The ReduceMean test also records the Phase A post-normalization form. |
| V-3 | `NERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx ./build/test/nert_loader_test --gtest_filter='EngineLoaderNegative.NoEffectiveSchemaAtDeclaredOpsetRejected'` | pass (1/1) | 2026-10-22 | `LayerNormalization` at declared 11 names `ln0`, `LayerNormalization`, `declared opset 11`, `no effective schema`, and `first schema is 17`. |
| V-4 | `cmake -S . -B build -DNERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx && cmake --build build --parallel "$(nproc)" && NERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx ctest --test-dir build --output-on-failure` | pass (13/13) | 2026-10-22 | `nert_loader` 104/104, `nert_executor`, and `nert_opset_table` included. Environment note: the older `/tmp/opencode/resnet18_export/resnet18.onnx` is a different export (GlobalAveragePool/Flatten, 42 initializers) and fails fixture-content assertions in the loader and shape suites; it is not the pinned reference. |
| V-5 | `git diff --stat`; `NERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx ./build/test/nert_loader_test --gtest_color=no` | pass (104/104) | 2026-10-22 | `src/opset_normalize.{hpp,cpp}` are unmodified; the legacy Softmax expansion and all negative normalizer tests pass. |
| V-6 | CPU `ctest`; loader `EngineLoader.Hermetic*Reduce*`; kernel old/new/empty form tests; shape `Reduce*AttributeFormDeclaredThirteen` and `ReduceMeanEmptyAxesNoopIdentity` | pass | 2026-10-22 | `nert_loader` 104/104; CPU reduce tests 20/20 in the filtered reduce suite; shape reduce tests 8/8; old form keeps one input + `axes` attr and creates no `reduce0/axes*` initializer; empty/noop cases pass. |
| V-7 | `nert_loader_negative` reduce-form tests | pass | 2026-10-22 | Five tests: pre-18 input, empty-string input, attribute at 18, noop before 18, and ReduceMax noop at 18 as a capability gap; each names node/op/declared/effective. |
| V-8 | `ctest --test-dir build-gpu -R 'nert_cuda_misc|nert_cuda_kernels|nert_opencl_kernels|nert_opencl_reduce_extra|nert_opencl_index_extra|nert_opencl_owlv2_rows'` | pass (6/6) | 2026-10-22 | CUDA Mean/L2/Max and OpenCL Mean/L2/Max attribute forms match the CPU oracle. HIP has no reduce rows; recorded gap. |
| V-18 | `scripts/format.sh --check` | pass | 2026-10-22 | 239 files formatted; no new production warnings. |
