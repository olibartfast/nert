# Validation — N1e via cuDNN Frontend

Defined before implementation. Bar: numerics unchanged (tolerance
`1e-3·(1+|ref|)`, target exact vs the legacy row), OFF default
untouched, format clean, CUDA container leg green.

## Checks

- [V-A1] Slice A: Conv parity exact on ResNet + YOLO fixtures; bench
  row before/after (same fixture); CUDA + OFF legs green.
- [V-B1] Slice B: frontend Conv bit-exact vs CPU oracle and vs the
  legacy row on both shapes; stride/pad variants covered; no-row
  fallback tested when the frontend dir is absent.
- [V-B2] Slice B: default build (OFF) has no frontend include or link;
  CUDA-without-frontend still builds and passes.
- [V-C1] Slice C: in-graph bias matches the sliced row on the
  padded/grouped/bias fixtures; bench bias fixture before/after; same
  tolerance (target exact).
- [V-G] Format clean; no `src/kernels/`, loader, shape, executor diffs.

## Evidence

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| V-A1 | Slice A: container `nert-gpu:26.08`, CUDA+bench build; Conv bench fixture before 5.87 ms / after 0.98 ms (5.99x), maxdiff unchanged 3.81e-06; full ctest 23/23 with fixture; OFF 12/12; format | pass | 2026-10-18 | One process handle + cached descriptors/algo/workspace per geometry; numerics bit-identical to the pre-slice row. |
| V-B1 | Slice B: frontend build (`NERT_CUDNN_FRONTEND_DIR`, v1.31.0); ctest 23/23 with fixture; Conv bench 0.96 ms, maxdiff 3.81e-06 (unchanged); pointwise/padded/grouped/model-level conv rows all pass | pass | 2026-10-18 | Table row switches to `ConvFrontend`; no perf claim beyond A (same cuDNN kernels) — adoption is for slice C fusion. |
| V-B2 | Slice B fallback: configure without `NERT_CUDNN_FRONTEND_DIR` builds the legacy row; conv test 1/1 (full suite covered by A); OFF 12/12; format | pass | 2026-10-18 | Default build never sees frontend headers or `nvrtc`. |
| V-B3 | Docker revalidation, 2026-10-23: `nert-gpu:26.08` (CUDA 13.4.59, cuDNN 9.25.0), frontend v1.31.0 mounted at `/frontend/include`; configure with `-DNERT_ENABLE_CUDA=ON -DNERT_CUDNN_FRONTEND_DIR=/frontend/include`; full CTest 24/24 (including `nert_cuda_conv` 14/14); ResNet-18 gate 49 nodes, top-1/top-5 identical, max abs 2.4399e-03, 22/1000 elements over scaled 1e-3, mean 5.2826e-04 | pass with recorded deviation (R-6 class) | 2026-10-23 | `CMakeFiles/nert.dir/flags.make` contains `-DNERT_HAS_CUDNN_FRONTEND`; `nm` shows `nert::cuda_kernels::ConvFrontend`; CPU NERT vs fresh ORT reference is 1.67e-06 max, 0/1000 over tolerance. The old scratch `ort_resnet_op19.bin` was not used because it does not correspond to this model/input pair; the fresh reference is `ort_resnet_op19_current.bin`. |
## Docker reproduction (2026-10-23)

The NVIDIA image carries CUDA 13.4.59 and cuDNN 9.25.0. The pinned cuDNN
Frontend v1.31.0 headers are mounted read-only; nothing is installed into the
host toolchain.

```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -v /tmp/opencode/frontend:/frontend:ro \
  -v /tmp/opencode:/scratch:ro \
  -w /workspace --user "$(id -u):$(id -g)" -e HOME=/tmp \
  nert-gpu:26.08 bash -lc '
    cmake -S . -B build-cudnn-frontend \
      -DCMAKE_BUILD_TYPE=Release \
      -DNERT_ENABLE_CUDA=ON \
      -DNERT_CUDNN_FRONTEND_DIR=/frontend/include && \
    cmake --build build-cudnn-frontend --parallel "$(nproc)" && \
    NERT_FIXTURE=/scratch/resnet18_op19.onnx \
      ctest --test-dir build-cudnn-frontend --output-on-failure
  '
```

`CMakeFiles/nert.dir/flags.make` must contain
`-DNERT_HAS_CUDNN_FRONTEND`; `nm build-cudnn-frontend/libnert.a | grep
ConvFrontend` must resolve the symbol. Without the frontend mount, configure
falls back to the legacy `cudnnConvolutionForward` row.
