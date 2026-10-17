# Requirements — R6 + open-vocab Tier-5 rows on CUDA

Date: 2026-10-17. The RF-DETR inventory (task-models V-2) left R5/R6;
the norm corridor (R1–R3 + ReduceSum) landed in
`specs/2026-10-17-rf-detr-norm-cuda/`. This packet closes R6a (`Sin`,
`Cos`, `Exp`, 11 nodes) + R6b (`Tile`, `Not`, `Greater`, `Less`, `And`,
9 nodes) and the Tier-5 CUDA-missing set (`Or`, `LessOrEqual`,
`GreaterOrEqual`, `ReduceL2`, `Max`, `Min`, `ArgMax`, `CumSum`,
`EyeLike`, `ScatterND`, `Elu`, `Pow`, `Abs`, `Neg`, `Log`, `Clip`,
`Floor`, `Identity`, `Squeeze`, `Range`, `Pad`, `ReduceMean`, `IsNaN`;
23 ops) — 31 kernels total. Same model + same plan on CUDA, matching
the CPU oracle within tolerance.

## In scope

- 31 first-party CUDA kernels under `src/cuda/`, table rows in
  `src/cuda/table.cu`, `target_sources` entries, tests beside the
  existing CUDA suites (activations/misc/norm + one new data-movement
  file if warranted). Per-op: hand-computed values, CPU-oracle vs CUDA
  compare at the bench tolerance, negatives. GPU-gated `GTEST_SKIP`.
- CPU oracle policies mirrored exactly (dtype rules below; attribute
  defaults; shape checks; node+op error messages). No new policy.

## Out of scope

- `GridSample` (R5, sampler — own slice), `Einsum` (stays CPU-only),
  HIP/OpenCL mirrors (queued behind Q-1 hipify/hand-write split),
  `bench/` fixtures, loader/shape/executor changes, Neuriplo parity runs.

## Oracle policies (binding)

- Unary float32-only, count match: `Sin` (`sinf`), `Cos` (`cosf`),
  `Exp` (`expf`), `Elu` (alpha float attr), `Log` (`logf`),
  `Floor` (`floorf`). `IsNaN`: float32 in, bool out.
- `Abs`/`Neg`: float32 + int64, shared dtype.
- `Pow`: binary float32 (NumPy broadcast like `Add`).
- Comparisons `Greater`/`Less`/`GreaterOrEqual`/`LessOrEqual`:
  float32+int64 shared dtype, bool out, NumPy broadcast (`Equal`
  precedent). `And`/`Or`: bool+bool→bool broadcast. `Not`: bool→bool.
- `Max`/`Min`: 1+ inputs, one shared float32/int64 dtype, full
  N-way broadcast validated upfront; pairwise fold in input order
  (exact, NaN order identical to the oracle). `Clip`: float32,
  optional scalar min/max *inputs* (Unknown = absent), D2H scalar read.
- `Tile` (repeats rank-1 int64), `Squeeze` (f32/i64/bool + optional
  axes), `Identity` (f32/i64/bool copy), `Range` (scalar start/limit/
  delta, float + int paths), `EyeLike` (f32/i64/bool), `Pad` (constant
  mode + supported modes per oracle — read `pad.cpp` first).
- `ReduceMean` (float32-only + axes/keepdims/noop per oracle),
  `ReduceL2` (float32, oracle epsilon/axes policy),
  `ArgMax` (f32/i64 in, int64 out, axis/keepdims/select_last_index),
  `CumSum` (f32/i64, scalar int64 axis input, exclusive/reverse),
  `ScatterND` (f32/i64/bool data + int64 indices; copy-then-scatter).

## Constraints (traced to constitution)

- stdlib-only default; CUDA PRIVATE. One device per graph, no fallback.
- Naive readable kernels, internal sync; double accumulation for float
  stats/sums; reduction/broadcast rank cap 16 (established pilot limit).

## Open questions

- Q-1 stays as decided (CUDA canonical; HIP via hipify; OpenCL
  hand-written). This packet writes CUDA rows only.
- V-3 (OWLv2/Grounding-DINO parity + GPU gaps) is still pending
  Neuriplo-side; this slice is ordered by the RF-DETR node counts +
  Tier-5 family grouping, not by V-3.
