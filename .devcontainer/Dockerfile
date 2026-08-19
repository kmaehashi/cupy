# CuPy Development Container
# Provides a complete development environment with CUDA Toolkit 13.x, Python, compilers, and build tools

ARG CUDA_VERSION=13.0
FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install basic utilities and build tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    g++ \
    gcc \
    make \
    cmake \
    pkg-config \
    ca-certificates \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install sccache (Rust-based distributed compiler cache)
ARG SCCACHE_VERSION=0.9.0
RUN wget https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz \
    && tar xzf sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz \
    && mv sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl/sccache /usr/local/bin/ \
    && chmod +x /usr/local/bin/sccache \
    && rm -rf sccache-v${SCCACHE_VERSION}-x86_64-unknown-linux-musl* \
    && sccache --version

# Configure ccache
ENV PATH=/usr/lib/ccache:$PATH
RUN ccache --max-size=5G && ccache --set-config=compression=true

# Set up CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

# Install CuPy build dependencies
RUN pip install --no-cache-dir \
    "setuptools>=77" \
    "Cython>=3.1,<3.2" \
    "numpy>=2.0" \
    wheel

# Install development and testing tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-xdist \
    mock \
    pre-commit \
    ruff \
    mypy \
    autopep8 \
    cython-lint

# Create non-root user for development
ARG USERNAME=cupy
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Configure ccache and sccache for user
RUN mkdir -p /home/$USERNAME/.ccache /home/$USERNAME/.cache/sccache \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/.ccache /home/$USERNAME/.cache

# Set up Git for container (symbolic link support)
RUN git config --global core.symlinks true

# Switch to non-root user
USER $USERNAME

# Set working directory
WORKDIR /workspace

# Set environment variables for the user
ENV NVCC="ccache nvcc"
ENV CC="ccache gcc"
ENV CXX="ccache g++"
ENV CCACHE_DIR=/home/$USERNAME/.ccache
ENV SCCACHE_DIR=/home/$USERNAME/.cache/sccache

# Add helpful message for user
RUN echo 'echo "CuPy Development Environment"' >> /home/$USERNAME/.bashrc \
    && echo 'echo "CUDA Version: $CUDA_PATH"' >> /home/$USERNAME/.bashrc \
    && echo 'echo "Python Version: $(python --version)"' >> /home/$USERNAME/.bashrc \
    && echo 'echo ""' >> /home/$USERNAME/.bashrc \
    && echo 'echo "Quick Start:"' >> /home/$USERNAME/.bashrc \
    && echo 'echo "  pip install --no-build-isolation -e .  # Install CuPy in editable mode"' >> /home/$USERNAME/.bashrc \
    && echo 'echo "  pre-commit run -a                       # Run linters"' >> /home/$USERNAME/.bashrc \
    && echo 'echo "  python -m pytest tests/cupy_tests       # Run tests"' >> /home/$USERNAME/.bashrc \
    && echo 'echo ""' >> /home/$USERNAME/.bashrc

CMD ["/bin/bash"]
