ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

COPY docker/cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
COPY docker/cuda-repo.deb /tmp/cuda-repo.deb

RUN dpkg -i /tmp/cuda-repo.deb && \
    cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    rm -f /tmp/cuda-repo.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}

# Install Kaolin
RUN --mount=type=cache,target=/root/.cache/pip \
    bash -i -c "source ${HOME}/.bashrc && \
    pip install --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html  kaolin==0.18.0"

# Install PyTorch3D
RUN --mount=type=cache,target=/root/.cache/pip \
    bash -i -c "source ${HOME}/.bashrc && \
    TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' FORCE_CUDA=1 python -m pip install --no-build-isolation  git+https://github.com/facebookresearch/pytorch3d.git@stable"


COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Install GraspQP Dependencies
RUN --mount=type=cache,target=/root/.cache/pip bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/thirdparty/pytorch_kinematics && \
    pip install -e . && \
    cd  ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/thirdparty/TorchSDF && pip install -e . --no-build-isolation"

# Install GraspQP and GraspQP Isaac Lab
RUN --mount=type=cache,target=/root/.cache/pip bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/graspqp_isaaclab/src/ && \
    pip install -e . && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/graspqp/ && \
    pip install -e ."


# For singularity use
RUN mkdir -p /data

WORKDIR /workspace

RUN --mount=type=cache,target=/root/.cache/pip bash -i -c "source ${HOME}/.bashrc && \
    pip install git+https://github.com/renezurbruegg/TorchSDF --no-build-isolation"

# # make working directory as the Isaac Lab directory
# # this is the default directory when the container is run
