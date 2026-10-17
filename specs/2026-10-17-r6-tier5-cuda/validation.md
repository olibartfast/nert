# Validation — R6 + open-vocab Tier-5 rows on CUDA

Defined before implementation. Tolerance: `ExpectNear` bench rule
unless a check states exact. SKIP without a device is recorded.

## Checks

- [V-A] Unary float suite: hand values per op (`sin`/`cos` known angles,
  `exp(1)≈e`, Elu alpha path + negative branch, `log(e)=1`, floor of
  negatives, IsNaN true/false set) CPU vs CUDA; count-mismatch +
  non-float negatives.
- [V-B] Abs/Neg int64 + float paths; Pow broadcast incl. negative base
  integer exponent spot; negatives (dtype, broadcast).
- [V-C] Each comparison: float + int64 + broadcast + true/false mix;
  And/Or/Not truth tables incl. broadcast; negatives (dtype mix,
  non-bool out, broadcast mismatch).
- [V-D] Max/Min 3-input broadcast fold vs oracle incl. int64;
  Clip default/attrs-only-fallback... (min/max inputs present, absent,
  one-sided); negatives (broadcast, dtype).
- [V-E] Identity copy dtypes; Squeeze with/without axes + negative
  (non-1 dim); Tile repeats incl. zero-repeat edge per oracle; Range
  float + int + negative-delta; EyeLike; Pad constant (+ oracle modes
  as implemented); negatives per op.
- [V-F] ReduceMean vs oracle (keepdims/axes/noop); ReduceL2;
  ArgMax axis/keepdims/tie policy + int64 out; CumSum
  exclusive/reverse; ScatterND; negatives per op.
- [V-G1] OFF leg full `ctest` green. [V-G2] CUDA-ON leg full `ctest`
  green (RTX 3060). [V-G3] format clean. [V-G4] diff audit: packet
  paths only.

## Acceptance (full pair, once at the end)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel "$(nproc)" && ctest --test-dir build --output-on-failure && ./scripts/format.sh --check
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DNERT_ENABLE_CUDA=ON && cmake --build build-cuda --parallel "$(nproc)" && ctest --test-dir build-cuda --output-on-failure
```

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-A | `nert_cuda_activations_test` Sin/Cos/Exp/Elu/Log/Floor/IsNaN (7 exact + 3 negatives) | pass 10/10 new | 2026-10-17 | RTX 3060; suite now 20/20 with norm-corridor rows |
| V-B | Abs/Neg (float+int64) + Pow broadcast + 2 negatives | pass 5/5 new | 2026-10-17 | Suite 25/25 |
| V-C | Greater/Less/GOE/LOE + And/Or/Not (truth tables, broadcast, int64) + 2 negatives | pass 8/8 new | 2026-10-17 | Equal-template; sed-derivation damage found by diff audit and repaired before build |
| V-D | Max 3-input fold + Min int64 + Clip bounds/identity + 2 negatives | pass 5/5 new | 2026-10-17 | Hand-arithmetic slip in test expectation fixed (oracle was right) |
| V-E | `nert_cuda_move_test` Identity/Squeeze/Tile/Range/EyeLike/Pad-modes + 3 negatives | pass 9/9 | 2026-10-17 | New target; Pad covers constant/reflect/edge |
| V-F | ReduceMean/ReduceL2/ArgMax-ties/CumSum-modes/ScatterND + 4 negatives | pass 9/9 | 2026-10-17 | Misc suite 46/46 |
| V-G1 | OFF leg full `ctest` | pass 12/12 | 2026-10-17 | Fixture via `/tmp/opencode/resnet18_export` fallback |
| V-G2 | CUDA-ON leg full `ctest` | pass 23/23 | 2026-10-17 | RTX 3060 host leg |
| V-G3 | `scripts/format.sh --check` | pass | 2026-10-17 | `--fix` touched new tests only |
| V-G4 | Diff audit | pass | 2026-10-17 | Packet paths only; pre-existing dirty (`bench/`, spike spec) untouched |

## Deviations

- `implementer` subagent dispatch unavailable again (harness
  `tool_choice` error); planner implemented directly against this packet.
- Mechanical derivation (`sed`) of Less/GOE/LOE/Or from Greater/And
  caused collateral edits (comparison operators in stride loops,
  broadcast `&&`, messages); caught by per-file `diff` audit and
  repaired line-scoped before compiling. Lesson recorded: derive with
  name-only swaps + line-scoped operator edits, then diff every file.
- Pad `constant_value` first read dereferenced device memory on the
  host; caught on re-read before building and fixed to D2H.
- Max test hand-arithmetic wrong (`{4,5,4,2}` vs oracle `{4,2,5,5}`);
  oracle trusted, test fixed.
