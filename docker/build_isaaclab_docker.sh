#!/bin/bash
# Build the graspqp_isaaclab image: the full GraspQP stack on top of the Isaac Lab base image.
#
# Prerequisite: the `isaac-lab-base` image must already exist. Build it first from an Isaac Lab
# checkout (Isaac Sim >= 4.5), e.g. in the Isaac Lab repo:
#     ./docker/container.py start base      # builds & tags `isaac-lab-base`
#
# The CUDA toolkit / kaolin / pytorch3d versions are matched to the base image's PyTorch build
# via Dockerfile.isaaclab's build args (defaults target Isaac Sim 4.5 -> torch 2.5.1 + cu118).
# For a torch 2.7 / cu128 base, pass e.g.:
#     CUDA_PKG=cuda-toolkit-12-8 CUDA_HOME_DIR=/usr/local/cuda-12.8 \
#     KAOLIN_INDEX=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html \
#     TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' ./docker/build_isaaclab_docker.sh
#
# Usage (run from anywhere; context is the repo root):
#     ./docker/build_isaaclab_docker.sh [ISAACLAB_BASE_IMAGE] [IMAGE_TAG]
# Defaults: ISAACLAB_BASE_IMAGE=isaac-lab-base   IMAGE_TAG=graspqp_isaaclab
set -e

ISAACLAB_BASE_IMAGE="${1:-isaac-lab-base}"
IMAGE_TAG="${2:-graspqp_isaaclab}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[build_isaaclab_docker] Building '${IMAGE_TAG}' on top of '${ISAACLAB_BASE_IMAGE}'"
DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile.isaaclab \
    -t "${IMAGE_TAG}" \
    --build-arg ISAACLAB_BASE_IMAGE_ARG="${ISAACLAB_BASE_IMAGE}" \
    ${CUDA_PKG:+--build-arg CUDA_PKG="${CUDA_PKG}"} \
    ${CUDA_HOME_DIR:+--build-arg CUDA_HOME_DIR="${CUDA_HOME_DIR}"} \
    ${KAOLIN_INDEX:+--build-arg KAOLIN_INDEX="${KAOLIN_INDEX}"} \
    ${TORCH_CUDA_ARCH_LIST:+--build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"} \
    "${REPO_ROOT}"

echo "[build_isaaclab_docker] Built image '${IMAGE_TAG}'."
echo "  DexEvolve builds on top of this (BASE_IMAGE=${IMAGE_TAG})."
