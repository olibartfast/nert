FROM nvcr.io/nvidia/tensorrt:26.08-py3

# GPU dev environment for NERT Phase N1 (cuBLAS/cuDNN work) on the 26.08
# stack (CUDA 13.x, cuDNN 9, TensorRT 11.2). The host stays toolkit-free;
# this image carries nvcc, cuBLAS, and cuDNN. The repo bind-mounts at /work
# so code syncs to the host automatically — commits happen on the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgtest-dev \
        libgmock-dev \
        python3-pip \
        clang-format-18 \
        libcudnn-frontend-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
