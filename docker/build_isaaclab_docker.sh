#!/bin/bash

# Download dependencies for CUDA 12.8 if not existing
if [ ! -f "cuda-ubuntu2204.pin" ]; then
    echo "Downloading cuda-ubuntu2204.pin"
    curl -fsSL -o cuda-ubuntu2204.pin https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    curl -fsSL -o cuda-repo.deb https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
fi
# Build the docker image
docker compose --env-file .env.base --file docker-compose.yaml build graspqp_isaaclab
