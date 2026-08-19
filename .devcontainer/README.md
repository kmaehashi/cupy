# CuPy Development Container

This devcontainer provides a complete development environment for CuPy with:

- **CUDA Toolkit 13.0** (configurable)
- **Python 3.11** with development headers
- **Host compilers**: gcc, g++, make, cmake
- **Build caches**: ccache and sccache for faster rebuilds
- **Pre-configured tools**: pytest, pre-commit, ruff, mypy, and more

## Usage

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU access
- [Visual Studio Code](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (recommended)

### Getting Started with VS Code

1. Open the CuPy repository in VS Code
2. Press `F1` and select "Dev Containers: Reopen in Container"
3. VS Code will build the container and set up the environment
4. Once ready, the terminal will show helpful quick start commands

### Getting Started without VS Code

1. Build the container:
   ```bash
   cd .devcontainer
   docker build -t cupy-dev .
   ```

2. Run the container with GPU access:
   ```bash
   docker run --gpus all -it -v $(pwd)/..:/workspace cupy-dev
   ```

### Quick Start Commands

Once inside the container:

```bash
# Install CuPy in editable mode
pip install --no-build-isolation -e .

# Run linters
pre-commit run -a

# Run tests
python -m pytest tests/cupy_tests
```

## Configuration

### Changing CUDA Version

Edit the `CUDA_VERSION` argument in `devcontainer.json`:

```json
"args": {
    "CUDA_VERSION": "12.6"  // Change to desired version
}
```

Supported CUDA versions match those available in [NVIDIA's CUDA Docker images](https://hub.docker.com/r/nvidia/cuda).

### Environment Variables

The following environment variables are pre-configured:

- `NVCC="ccache nvcc"` - Use ccache for NVCC compilation
- `CC="ccache gcc"` - Use ccache for C compilation  
- `CXX="ccache g++"` - Use ccache for C++ compilation
- `CUDA_PATH=/usr/local/cuda`
- `CCACHE_DIR=/home/cupy/.ccache`
- `SCCACHE_DIR=/home/cupy/.cache/sccache`

### Installed VS Code Extensions

- Python language support and IntelliSense
- Ruff linter
- Black formatter
- Jupyter notebook support

## Features

### Build Caching

The container includes both **ccache** and **sccache** to significantly speed up rebuilds:

- **ccache**: Caches compilation outputs locally (max 5GB)
- **sccache**: Rust-based cache with support for distributed caching

### Non-Root User

The container runs as a non-root user (`cupy`) for security and to avoid file permission issues with the host system.

### GPU Access

The container is configured to access all available GPUs via `--gpus=all` flag.

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:
```bash
nvidia-smi  # Should show your GPU
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### Build Fails

Try rebuilding without cache:
```bash
docker build --no-cache -t cupy-dev .
```

### Permission Issues

The container uses UID/GID 1000 by default. If your user has a different UID, rebuild with:
```bash
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t cupy-dev .
```

## Additional Resources

- [Dev Containers documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [CuPy contribution guide](https://docs.cupy.dev/en/stable/contribution.html)
- [AGENTS.md](../AGENTS.md) - Comprehensive developer guide
