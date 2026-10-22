# Plan — Version-aware opsets

Work in an isolated branch from the clean `wip/nert-opset-newer` HEAD. Each
phase is independently runnable and reviewable. Do not ship the transitional
state after Phase A: the normalizer is still authoritative there.

## Scoreboard — phase A

- **Command (repo-relative)**:
  ```sh
  cmake -S . -B build -DNERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx \
    && cmake --build build --parallel "$(nproc)" \
    && NERT_FIXTURE=/tmp/opencode/resnet18_op19.onnx \
       ctest --test-dir build --output-on-failure
  ```
  The fixture path is host-specific. In another environment, point
  `NERT_FIXTURE` at the provisioned ResNet-18 export used by the existing
  model gate; do not use an unrelated export. The pinned reference is
  `resnet18_op19.onnx` (44 initializers, `ReduceMean`/`Reshape`), not the
  older `resnet18_export/resnet18.onnx` (42 initializers,
  `GlobalAveragePool`/`Flatten`).
- **Adjudicates**: full build; all 13 CTest entries including
  `nert_opset_table`; existing ResNet-18 loader/executor gates; new
  effective-schema tests.
- **Run per attempt**: exactly once, as the final action.
- **Recorded with**: attempt exit status in the run ledger.

## Group 0 — Baseline

- [T-1] Confirm clean starting tree and record `git rev-parse HEAD`.
- [T-2] Capture the current full-suite result before changes.

## Group 1 — Schema metadata and full-history catalog (Phase A)

- [T-3] Add `Graph.declared_opset` and `Node.schema_version` to
  `include/nert/graph.hpp`, with defaults that keep aggregate construction
  source-compatible.
- [T-4] Extend `tools/gen_opset_compat.py` to emit every schema version in
  each implemented operator's history, including versions below 11, because
  a declared-11 model can resolve an operator to an older effective schema.
  Keep the generated table deterministic and committed. Do not mark
  pre-baseline entries as “equivalent to 18”; label them as historical
  schema entries that the resolver can select. Keep the baseline-at-18 entry
  classified for the existing above-18 gate.
- [T-5] Add a schema resolver that returns the effective entry for every
  node decoded from the model, for all accepted declarations. The loader sets
  `graph.declared_opset` and `node.schema_version` after decoding the node.
  Nodes synthesized later by the remaining normalizer may keep `0` in
  Phase A; source nodes must be populated.
- [T-6] Keep `NormalizeGraphForOpset` and the current canonical behavior in
  Phase A. Schema metadata is informational until Phase B removes the first
  rewrite. One deliberate strictness change is in scope: a source node whose
  operator has no schema at or below the declaration (for example,
  `LayerNormalization` at declared 11) is rejected as ONNX-invalid.
- [T-7] Add focused tests:
  - declared 11, 13, 18, 19, and 27 models expose the expected effective
    schema on each node;
  - a declared-27 model with `Add` resolves to schema 14, not 27;
  - a node whose operator has no schema entry at or below the declaration
    fails with node, operator, declared opset, and no-effective-schema;
  - `Graph.declared_opset` matches the model declaration.
- [T-8] Regenerate the table and run the freshness check.

## Group 2 — Reduce family (Phase B)

- [T-9] Remove the reduce-axis rewrite from `opset_normalize.cpp` and move
  the corresponding legality checks to the schema-aware loader. Do this
  before any consumer reads `schema_version` for form selection, because the
  Phase A metadata names the source schema while the node still carries the
  rewritten form.
- [T-10] Add shared `ReduceAxesForm` / `ResolveReduceAxes` semantics:
  attribute before schema 18 for `ReduceMean` / `ReduceL2` / `ReduceMax`;
  input from schema 18. Keep `ReduceSum` on its schema-13 boundary for the
  next group.
- [T-11] Update shape inference and the CPU/CUDA/OpenCL kernels that consume
  axes. HIP has no reduce rows; record the gap.
- [T-12] Replace the canonicalization tests with old-form execution tests and
  keep the illegal-form negative tests. The Phase A form assertion must be
  replaced by an assertion that the source form stays in the graph.

## Group 3 — Axes, Split, and ScatterND (Phase C)

- [T-13] Add schema-aware resolvers for `ReduceSum`, `Squeeze`,
  `Unsqueeze`, `Split`, and `ScatterND.reduction`.
- [T-14] Update shape inference and kernels (squeeze/unsqueeze are
  shape-inference-only; split needs sizes in CPU/CUDA/OpenCL).
- [T-15] Remove those rewrites and legality checks from the normalizer.
- [T-16] Update tests to keep the old node form and verify the same values.

## Group 4 — Legacy Softmax and normalizer removal (Phase D)

- [T-17] Implement legacy Softmax as a direct kernel mode:
  `outer = prod(dims[0..axis))`, `dim = prod(dims[axis..rank))`, `inner = 1`.
  Share the mode resolver across CPU, CUDA, OpenCL, and HIP.
- [T-18] Remove the `Shape → Flatten → Softmax → Reshape` expansion and the
  static-rank requirement.
- [T-19] Delete `src/opset_normalize.{hpp,cpp}`, its build entry, and its
  include/call sites. Remove `NormalizeGraphForOpset` tests.
- [T-20] Add model-backed comparisons for opset 11/12 Softmax against ONNX
  Runtime or the frozen pre-refactor output.

## Group 5 — Catalog and docs cleanup (Phase E)

- [T-21] Rename the compatibility seam to a schema catalog if the diff is
  still reviewable; otherwise leave names and document the semantic change.
- [T-22] Remove `BASELINE_OPSET = 18` from the generator and classify deltas
  from the full history. Add the implementation manifest required by the
  runtime acceptance gate.
- [T-23] Update `README.md`, `specs/tech-stack.md` `Q-1`, roadmap, coverage,
  and `CHANGELOG.md`.
- [T-24] Turn post-18 guards into version-aware capability slices only when
  their kernels land; do not silently accept them.

## Integration checkpoints

- After Phase A: full suite green; no unintended behavior change (the
  documented rejection of an operator introduced after the declaration is
  expected); table freshness green.
- After Phase B and C: old-form tests execute the source node form; no
  reduce/axes/split canonicalization remains.
- After Phase D: no normalizer file or call; full suite and model gates green.
- After Phase E: catalog docs and policy match the code; no global opset
  constant remains.

## Rollback

Keep the normalizer in the tree until Phase D. Each phase that removes a
rewrite can be reverted independently. Do not merge a phase with a red
scoreboard.
