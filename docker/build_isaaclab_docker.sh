#!/bin/bash
# Build the graspqp_isaaclab image: the GraspQP stack on top of the Isaac Lab base image.
#
# Prerequisite: the `isaac-lab-base` image must already exist. Build it first from an Isaac Lab
# checkout (Isaac Sim 5.1 / Isaac Lab 2.3 recommended), e.g. in the Isaac Lab repo:
#     ./docker/container.py start base      # builds & tags `isaac-lab-base`
#
# By default this builds a lightweight **WARP-only** image (no CUDA toolkit / pytorch3d /
# TorchSDF / Kaolin — no nvcc compilation, ~2 min). To also install the compiled backends
# (needed only for SDF_BACKEND=TORCHSDF|KAOLIN), set WITH_COMPILED_BACKENDS=1:
#     WITH_COMPILED_BACKENDS=1 ./docker/build_isaaclab_docker.sh
#
# The compiled deps must match the base image's PyTorch (defaults target Isaac Sim 5.1 ->
# torch 2.7 + cu128). For an Isaac Sim 4.5 base (torch 2.5.1 + cu118) also pass:
#     CUDA_PKG=cuda-toolkit-11-8 CUDA_HOME_DIR=/usr/local/cuda-11.8 \
#     KAOLIN_INDEX=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html \
#     TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9'
#
# Usage (run from anywhere; context is the repo root):
#     ./docker/build_isaaclab_docker.sh [ISAACLAB_BASE_IMAGE] [IMAGE_TAG]
# Defaults: ISAACLAB_BASE_IMAGE=isaac-lab-base   IMAGE_TAG=graspqp_isaaclab
set -e

ISAACLAB_BASE_IMAGE="${1:-isaac-lab-base}"
IMAGE_TAG="${2:-graspqp_isaaclab}"
WITH_COMPILED_BACKENDS="${WITH_COMPILED_BACKENDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ "${WITH_COMPILED_BACKENDS}" = "1" ]; then
    echo "[build_isaaclab_docker] Building '${IMAGE_TAG}' WITH compiled backends (pytorch3d/TorchSDF/kaolin)"
else
    echo "[build_isaaclab_docker] Building '${IMAGE_TAG}' (WARP-only; set WITH_COMPILED_BACKENDS=1 for compiled backends)"
fi

DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile.isaaclab \
    -t "${IMAGE_TAG}" \
    --build-arg ISAACLAB_BASE_IMAGE_ARG="${ISAACLAB_BASE_IMAGE}" \
    --build-arg WITH_COMPILED_BACKENDS="${WITH_COMPILED_BACKENDS}" \
    ${CUDA_PKG:+--build-arg CUDA_PKG="${CUDA_PKG}"} \
    ${CUDA_HOME_DIR:+--build-arg CUDA_HOME_DIR="${CUDA_HOME_DIR}"} \
    ${KAOLIN_INDEX:+--build-arg KAOLIN_INDEX="${KAOLIN_INDEX}"} \
    ${TORCH_CUDA_ARCH_LIST:+--build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"} \
    "${REPO_ROOT}"

echo "[build_isaaclab_docker] Built image '${IMAGE_TAG}'."
echo "  DexEvolve builds on top of this (BASE_IMAGE=${IMAGE_TAG})."
