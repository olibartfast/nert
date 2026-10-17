# Plan — R6 + open-vocab Tier-5 rows on CUDA

Work in the main tree (maintainer standing direction). Forbidden paths:
`README.md` (landed), `bench/` (unrelated dirty work),
`specs/2026-10-17-cudnn-frontend-spike/` (untracked, чужое), `.github/`.
Delegation harness `implementer` errored last run — planner implements
directly (deviation recorded in validation.md).

Six sequential phases (shared `table.cu` + `CMakeLists.txt` force order):

## Phase A — unary float (activations file)

`Sin`, `Cos`, `Exp`, `Elu`, `Log`, `Floor` (Sqrt/Erf pattern, device
math `sinf`/`cosf`/`expf`/alpha-form/`logf`/`floorf`) + `IsNaN`
(float→bool). Tests appended to `test/cuda_activations_test.cpp`.

## Phase B — Abs/Neg/Pow (activations file)

`Abs`/`Neg` templated float+int64 (one thread/element); `Pow` binary
float32 NumPy broadcast (`Equal` stride template, `powf`). Tests in
`test/cuda_activations_test.cpp`.

## Phase C — comparisons + logic (misc file)

`Greater`/`Less`/`GreaterOrEqual`/`LessOrEqual` (copy `equal.cu`,
swap operator), `And`/`Or` (bool inputs), `Not` (unary bool).
Tests in `test/cuda_misc_test.cpp`.

## Phase D — Max/Min/Clip (misc file)

`Max`/`Min`: full N-way broadcast validation upfront, then
broadcast-copy input0 → out (Expand precedent) + pairwise fold in
input order. `Clip`: D2H scalar min/max reads (Unknown = absent).
Tests in `test/cuda_misc_test.cpp`.

## Phase E — data movement (new `test/cuda_move_test.cpp` + target)

`Identity` (typed D2D copy), `Squeeze` (axes download, dim-1 checks),
`Tile` (repeats download + tiled copy), `Range` (scalar D2H ×3,
float+int paths), `EyeLike` (fill kernel), `Pad` (constant path first;
other oracle modes only if `pad.cpp` shows them trivial).

## Phase F — reductions + scatter (misc file)

`ReduceMean` (copy `reduce_sum.cu`, divide by redCount in double),
`ReduceL2` (oracle epsilon policy), `ArgMax` (per-output scan along
axis, int64 out, tie policy per oracle), `CumSum` (per-output scan
with exclusive/reverse), `ScatterND` (D2D copy data→out, then
per-update scatter). Tests in `test/cuda_misc_test.cpp`.

## Per-phase bars

Hand-computed + oracle-compare + negatives; targeted target green;
final acceptance (below) once per phase at most, full pair once at end:
OFF 12/12 (now more), CUDA-ON all green, format clean, diff audit =
packet paths only.

## Validation

Evidence lands in
`specs/2026-10-17-r6-tier5-cuda/validation.md` — one table.
Roadmap + CHANGELOG sync on landing (planner-owned).
