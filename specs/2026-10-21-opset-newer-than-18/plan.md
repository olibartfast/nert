# Plan — Newer-than-18 opsets via verified compatibility

Work in its own worktree (`wip/nert-opset-newer`) after
`specs/2026-10-21-opset-range/` Phase A is merged: this packet reuses the
declared-opset parse, the normalizer seam, and the exact-18 test baseline.
The table is generated and reviewed before the loader accepts anything new.

## Group 0 — Baseline and index

- [T-1] Add this packet and its dependency to `specs/roadmap.md`; link the
  two opset packets. Deliverable: roadmap shows the order.
- [T-2] Confirm the range packet landed: synthetic opset-13/18 models load,
  opset-19 fails, and the full opset-18 suite is green. If not, stop.
  - Checks: `nert_loader` focused tests plus the full suite.

## Group 1 — Compatibility table

- [T-3] Write `tools/gen_opset_compat.py` (development only). For every
  operator in `src/kernels/cpu_kernels.cpp` it:
  - reads ONNX schema history (`onnx.defs`);
  - records the schema versions and their input names/optionality,
    attribute names/types/defaults, and documentation-level value domains;
  - compares each version at or after the opset-18 schema with the 18
    schema;
  - classifies it as `equivalent`, `additive-attributes`, `value-domain`,
    `behavior-change`, or `newer-operator`;
  - emits a deterministic header/`.inc` with the table and a generation
    comment.
  - Deliverable: the generator prints the table and fails loudly on an
    operator it cannot classify.
- [T-4] Review the generated table by hand against the 2026-10-21 audit.
  Expected first result: all implemented operators `equivalent` except
  `Cast` (`additive-attributes`: `saturate`, `round_mode`) and `Pad`
  (`value-domain`: `wrap`); `Resize`/`ReduceMax` post-18 entries are
  `equivalent` for the implemented attribute set.
  - Checks: the generated table matches the audit; unexplained differences
    stop the slice.
- [T-5] Commit the table under `src/` (internal, not a public header) and
  document the regeneration command in the file and in
  `specs/tech-stack.md`.
  - Deliverable: `python3 tools/gen_opset_compat.py --check` exits 0 and
    produces no diff.

## Group 2 — Loader resolver and guards

- [T-6] Resolve each node's effective schema version from the table. Bind
  the declaration from `opset_import` once per model; no per-node parser.
- [T-7] Apply the compatibility classes from [R-3]:
  - `equivalent`: accept;
  - `additive-attributes`: accept only if none of the added attributes is
    present (the allowlist already rejects them; keep the explicit check so
    the error names the version);
  - `value-domain`: accept only for the known-safe values; otherwise reject;
  - `behavior-change` / `newer-operator`: reject with the node, operator,
    and effective version.
- [T-8] Add the [R-4] guards:
  - `Cast`: reject `saturate` and `round_mode` when present; plain `to`
    stays accepted;
  - `Pad`: reject `mode="wrap"`; accept `constant`, `reflect`, `edge`;
  - structure the guard table so a future value-domain discovery is one
    entry, not a loader diff.
- [T-9] Error messages include the declared opset, effective version, and
  the capability that is missing. A model that declares 27 but uses only
  stable operators must not mention 27 as an error.
- [T-10] Unit tests:
  - declared 19/21/27 with `Add`/`Conv`/`Relu`/`MatMul` accepted;
  - declared 19 with `Cast.saturate` rejected;
  - declared 24 with `Cast.round_mode` rejected;
  - declared 19 with `Pad.wrap` rejected, with the other modes accepted;
  - a post-18 operator (for example `Mish` or `BitwiseAnd`) rejected as
    unknown;
  - an unsupported dtype still rejected;
  - a table-freshness test: regenerating yields no diff (or a CMake check
    equivalent).

## Group 3 — Real model gate

- [T-11] Provision a model exported natively at opset 19+ that uses only
  operators the table marks compatible. Compare NERT CPU against ONNX
  Runtime 1.27.0 for the same weights, fixture, and inputs.
  - Checks: all outputs within `1e-3·(1+|ref|)`, zero elements over;
    record export/reference hashes and commands in `validation.md`.
- [T-12] Run the model on CUDA and OpenCL as well if all operators have
  rows; otherwise record the per-op gap. Do not add a kernel row in this
  packet.

## Group 4 — Integration and docs

- [T-13] Update `README.md` “Opset policy” with the verified-compatibility
  rule and the conversion guidance for unverified newer features.
- [T-14] Update `specs/tech-stack.md` `Q-1`, the task-models coverage note,
  `CHANGELOG.md`, and the roadmap status.
- [T-15] Execute every check in `validation.md` on the merged tree; record
  evidence and deviations.
- [T-16] File follow-up slices for any rejected-but-wanted capability
  (`Cast.saturate`, `Cast.round_mode`, `Pad.wrap`, new operators). Link them
  from the roadmap; do not smuggle them into this packet.

## Notes

- The table is the only new source of truth. Do not hand-edit generated
  entries; change the generator or add a reviewed override with a comment.
- Keep the runtime table internal. This packet must not add a public API
  surface.
- If a real newer model exposes a value-domain change the generator missed,
  stop, add the guard and a test, and replan this packet before accepting
  the model.
