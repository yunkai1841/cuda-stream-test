FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3-pip \
    sudo \
    bash-completion \
    clang-format \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    meson \
    ninja

# use $USER instead of $USERNAME
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to non-root user
USER $USERNAME

# use bash
SHELL ["/bin/bash", "-c"]

# Set default working directory
WORKDIR /workspace