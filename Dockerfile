# nnssl image for CSCS Clariden (GH200 / arm64)
# Lives at the nnssl-private repo root; the repo IS the build context.
# Build ON an aarch64 node (GH200) with podman so the image is arm64, then
# enroot import to a .sqsh. Lean by design: torch + nnssl only; the pretrained
# checkpoint, plan, and FOMO300K data live on scratch and are mounted at runtime.
#   cd /iopsstor/scratch/cscs/petermcgor/nnssl-private
#   podman build -t nnssl:1.0 .
FROM docker.io/nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Python + git (git is needed for the nnssl git dependency) + minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git build-essential \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch for CUDA 13.0, aarch64 wheel from the cu130 index.
# Pinned to match the pretraining environment (torch 2.14.1+cu130).
# If this exact version is unavailable for arm64, drop "==2.14.1" to take the
# latest cu130 build; the patch version does not affect loading the checkpoint.
RUN pip3 install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# nnssl fork: the repo root is the build context (.dockerignore keeps the
# 3.6 GB data out). Pulls remaining deps incl. dynamic-network-architectures
# (from git), so the build node needs network for this layer.
COPY . /opt/nnssl
RUN pip3 install --no-cache-dir /opt/nnssl

# Smoke test: fail the build if the preprocess entrypoint cannot import.
RUN python3 -c "from nnssl.experiment_planning.plan_and_preprocess_entrypoints import preprocess_entry; print('nnssl import OK')"

WORKDIR /workspace
CMD ["/bin/bash"]
