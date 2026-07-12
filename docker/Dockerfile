# syntax=docker/dockerfile:1
#
# GraspQP -- lightweight image (DEFAULT).
#
# Uses ONLY the WARP SDF backend + pytorch_kinematics. Both install as prebuilt wheels /
# pure Python, so this image needs NO CUDA toolkit (nvcc) and does NO source compilation --
# a CUDA *runtime* base is sufficient. WARP is the default backend (SDF_BACKEND=WARP).
#
# Build (context = repo root):
#   docker build -f docker/Dockerfile -t graspqp:warp .
# Run:
#   docker run --rm --gpus all -it graspqp:warp
#
# For the non-default TorchSDF / Kaolin backends (which DO require nvcc), use
# docker/Dockerfile.torchsdf instead.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SDF_BACKEND=WARP

# git: to pip-install pytorch_kinematics from git+https. libgl/glib: trimesh + plotly rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/graspqp
COPY graspqp/ ./graspqp/
COPY scripts/ ./scripts/

# Lightweight install: base deps (numpy/trimesh/...) + WARP + pytorch_kinematics + viz/log/opt.
# No TorchSDF / pytorch3d / Kaolin => no nvcc, no compilation.
RUN pip install --no-cache-dir --no-build-isolation -e "./graspqp[lite]"

# Sanity check: WARP + graspqp import cleanly.
RUN python -c "import warp, graspqp; print('GraspQP lightweight image (WARP backend) ready')"

CMD ["bash"]
