# CuPy Developer Guide for AI Coding Agents

This document provides instructions for AI coding agents contributing to CuPy. This information is also useful for human developers.

## Development Environment Setup

### Prerequisites

- **Python**: 3.10 or later (3.10 / 3.11 / 3.12 / 3.13 / 3.14)
- **CUDA Toolkit**: 12.0 or later (tested with 12.x and 13.x)
  - NVIDIA GPU with Compute Capability 3.0 or higher
  - Compatible NVIDIA driver
- **Build Tools**:
  - C++ compiler (g++, clang, or MSVC)
  - Git (with symbolic link support on Windows)
  
### Installation Steps

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/cupy/cupy.git
   cd cupy
   ```

2. **Install build dependencies:**
   ```bash
   pip install -U setuptools pip
   pip install "setuptools>=77" wheel "Cython>=3.1,<3.2" numpy
   ```

3. **Install pre-commit for linting:**
   ```bash
   pip install pre-commit
   pre-commit install  # Optional: auto-run on each commit
   ```

4. **Install CuPy in editable mode:**
   ```bash
   pip install --no-build-isolation -e .
   ```
   
   The `--no-build-isolation` flag enables incremental compilation, reusing object files from previous builds.

5. **Install testing dependencies:**
   ```bash
   pip install pytest mock
   ```

### Optimizing Build Performance

- **Use ccache** to speed up rebuilds:
  ```bash
  export NVCC='ccache nvcc'
  ```

- **Limit target architectures** to reduce build time:
  ```bash
  export CUPY_NVCC_GENERATE_CODE=arch=compute_70,code=sm_70
  ```
  Replace with your GPU's compute capability.

- **Clean build artifacts** when modifying `.pxd` files:
  ```bash
  git clean -fdx
  ```

### Windows Development

On Windows (outside WSL), enable symbolic links before cloning:
```bash
git config --global core.symlinks true
```
Also activate Developer Mode in Windows settings.

## Source Tree Organization

### Main Source Directories

- **`cupy/`**: Core CuPy library implementing NumPy-compatible APIs
  - `cupy/_core/`: Core array operations and kernel infrastructure
  - `cupy/cuda/`: CUDA-specific functionality (Runtime, CUBLAS, etc.)
  - `cupy/linalg/`: Linear algebra routines
  - `cupy/random/`: Random number generation
  - `cupy/fft/`: Fast Fourier Transform
  
- **`cupyx/`**: CuPy extensions (SciPy-compatible APIs and experimental features)
  - `cupyx/scipy/`: SciPy-compatible implementations
  
- **`cupy_backends/`**: Backend support for different GPU platforms

- **`tests/`**: Test suite organized in parallel with source structure
  - `tests/cupy_tests/`: Tests for core CuPy (mirrors `cupy/` structure)
  - `tests/cupyx_tests/`: Tests for CuPy extensions (mirrors `cupyx/` structure)
  - `tests/install_tests/`: Installation and import tests
  - `tests/example_tests/`: Tests for example scripts

- **`docs/`**: Documentation source files (reStructuredText)
  - `docs/source/`: Documentation source
  - `docs/source/contribution.rst`: Contribution guidelines
  - `docs/source/install.rst`: Installation guide

- **`examples/`**: Example scripts demonstrating CuPy usage

### Key Configuration Files

- **`pyproject.toml`**: Project metadata and build configuration
- **`.pre-commit-config.yaml`**: Linting and formatting hooks
- **`setup.py`**: Build script for Cython extensions
- **`.github/CONTRIBUTING.md`**: Points to online contribution guide

## Building and Running Tests

### Initial Build

Build CuPy in-place before running tests:
```bash
pip install --no-build-isolation -e .
```

**Important**: When modifying `.pxd` files, clean and rebuild:
```bash
git clean -fdx
pip install "setuptools>=77" wheel "Cython>=3.1,<3.2" numpy
pip install --no-build-isolation -e .
```

### Running Tests

Run all tests:
```bash
python -m pytest
```

Run tests for a specific module:
```bash
python -m pytest tests/cupy_tests             # Core CuPy tests
python -m pytest tests/cupyx_tests            # CuPy extensions tests
python -m pytest tests/cupy_tests/linalg_tests  # Linear algebra tests
```

Run a specific test file:
```bash
python -m pytest tests/cupy_tests/core_tests/test_ndarray.py
```

### Test Options

- **Skip slow tests:**
  ```bash
  python -m pytest -m 'not slow'
  ```

- **Limit GPU usage** (for multi-GPU tests with limited hardware):
  ```bash
  export CUPY_TEST_GPU_LIMIT=1
  python -m pytest
  ```

- **Verbose output:**
  ```bash
  python -m pytest -v
  ```

### Test Naming Conventions

- Test directories end with `_tests` suffix
- Test files start with `test_` prefix
- For module `cupy.x.y.z`, tests are at `tests/cupy_tests/x_tests/y_tests/test_z.py`

### Writing Tests

- All test classes inherit from `unittest.TestCase`
- Use `assert` statements instead of `self.assertEqual()`
- Use `pytest.raises()` instead of `self.assertRaises()`
- Use `@testing.multi_gpu(n)` decorator for multi-GPU tests
- Use `@testing.slow` decorator for time-consuming tests

Example:
```python
import unittest
from cupy import testing

class TestMyFunc(unittest.TestCase):
    def test_basic(self):
        result = my_func()
        assert result == expected

    @testing.slow
    def test_slow_operation(self):
        # Time-consuming test
        pass
```

## Linting and Code Style

### Running Pre-commit Checks

Check all files:
```bash
pre-commit run -a
```

Check staged files only:
```bash
pre-commit run
```

### Linters Used

Pre-commit runs the following tools (see `.pre-commit-config.yaml`):

1. **autopep8**: PEP8 formatting
2. **ruff**: Fast Python linter (errors, style, imports)
3. **cython-lint**: Cython code linting
4. **mypy**: Type checking for Python code
5. **Standard hooks**: YAML validation, trailing whitespace, etc.

### Coding Guidelines

- Follow PEP8 and OpenStack Style Guidelines
- Line length: 79 characters (enforced by ruff)
- **Do not use relative imports** - always use absolute imports
- **Import only modules**, not symbols (except in tests/examples)
- Organize imports: standard library, third-party, internal
- **Do not use shortcut symbols** in library code (e.g., use `cupy.cuda.device.Device` not `cupy.cuda.Device`)
- For Cython code:
  - Use `void*` for Cython-only pointers
  - Use `intptr_t` for pointers exposed to Python
  - Use `size_t` for memory sizes
  - Use `ptrdiff_t` for memory offsets

## Submitting Changes

### Branch Strategy

- **`main` branch**: Development of alpha, beta, and RC versions
- **`vN` branches**: Stable version maintenance (e.g., `v14`)
- Submit PRs to `main` branch by default
- Core developers may backport to stable branches if applicable

### Pull Request Guidelines

1. **Before writing code:**
   - Check the appropriate target branch (usually `main`)
   - For new features/API changes, open an issue for design discussion first

2. **In your PR:**
   - Write clear explanation of what and how you fixed
   - Include unit tests for your changes
   - Update documentation if changing/adding APIs
   - Ensure pre-commit checks pass
   - Wait for CI checks (GitHub Actions runs CPU tests automatically)

3. **Work-in-progress PRs:**
   - Prefix title with `[WIP]` for incomplete work
   - Useful for early feedback and design discussions

4. **Backport PRs** (to stable `vN` branch):
   - Prefix title with `[backport]`
   - Reference original PR: "This is a backport of #XXXX"
   - Explain motivation for backporting

### For Copilot AI Coding Agents

**When creating a pull request for the upstream CuPy repository:**

After your changes are complete and this PR is ready, include the following in your final summary:

```
To submit this change to the upstream CuPy repository, please create a PR at:
https://github.com/cupy/cupy/compare/main...<your-branch>

Or use the GitHub CLI:
gh pr create --repo cupy/cupy --base main --head <your-branch>
```

## Building Documentation

Install documentation dependencies:
```bash
pip install -r docs/requirements.txt
```

Build HTML documentation:
```bash
cd docs
make html
```

View the documentation at `docs/build/html/index.html`.

**Note**: Docstrings are collected from the installed CuPy module, so install changes before building docs:
```bash
pip install --no-build-isolation -e .
cd docs
make html
```

## Additional Resources

- **Online contribution guide**: https://docs.cupy.dev/en/stable/contribution.html
- **Installation guide**: https://docs.cupy.dev/en/stable/install.html
- **API Reference**: https://docs.cupy.dev/en/stable/reference/
- **GitHub Repository**: https://github.com/cupy/cupy
- **Issues**: https://github.com/cupy/cupy/issues
- **Gitter Chat**: https://gitter.im/cupy/community
- **User Forum**: https://groups.google.com/forum/#!forum/cupy

## Quick Reference

### Common Commands

```bash
# Setup
git clone --recursive https://github.com/cupy/cupy.git
cd cupy
pip install -U setuptools pip
pip install "setuptools>=77" wheel "Cython>=3.1,<3.2" numpy
pip install --no-build-isolation -e .
pip install pytest mock pre-commit
pre-commit install

# Development
pre-commit run -a              # Run all linters
python -m pytest               # Run all tests
python -m pytest tests/cupy_tests  # Run core tests
pip install --no-build-isolation -e .  # Rebuild after changes

# Documentation
pip install -r docs/requirements.txt
cd docs && make html

# Clean rebuild (after .pxd changes)
git clean -fdx
pip install "setuptools>=77" wheel "Cython>=3.1,<3.2" numpy
pip install --no-build-isolation -e .
```

### Environment Variables

- `CUDA_PATH`: Specify CUDA installation directory
- `NVCC='ccache nvcc'`: Use ccache for faster rebuilds
- `CUPY_NVCC_GENERATE_CODE`: Limit target GPU architectures
- `CUPY_TEST_GPU_LIMIT`: Limit number of GPUs for testing
- `LD_LIBRARY_PATH`: May need to include `$CUDA_PATH/lib64`

## Troubleshooting

### Build Issues

- **Ensure latest setuptools/pip**: `pip install -U setuptools pip`
- **Clean build**: `git clean -fdx` then reinstall build dependencies
- **Check CUDA path**: Set `CUDA_PATH` environment variable
- **Check symbolic links on Windows**: Enable in Git and Windows Developer Mode

### Runtime Issues

- **NVRTC compilation errors**: Ensure CUDA runtime headers are available
  - For pip: `pip install "nvidia-cuda-runtime-cu12==12.X.*"`
  - For apt: `sudo apt install cuda-cudart-dev-12-X`
- **Library not found**: Add to `LD_LIBRARY_PATH`: `export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH`

### Test Issues

- **Multi-GPU tests failing**: Set `CUPY_TEST_GPU_LIMIT=1`
- **Slow tests**: Run with `-m 'not slow'` to skip
- **Flaky numerical tests**: May be ignored during review if unrelated to changes
