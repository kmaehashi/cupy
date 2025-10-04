# GitHub Copilot Instructions for CuPy

CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing with Python. This document provides guidance for GitHub Copilot when contributing to the CuPy repository.

## Project Overview

CuPy acts as a drop-in replacement for NumPy/SciPy, enabling existing code to run on NVIDIA CUDA or AMD ROCm platforms with minimal changes. The project maintains high compatibility with NumPy's API while providing GPU acceleration.

## Core Development Principles

### API Compatibility
- **NumPy/SciPy Compatibility**: All functions must maintain API compatibility with their NumPy/SciPy counterparts
- **Parameter Consistency**: Function signatures, parameter names, and behavior should match NumPy/SciPy exactly
- **Return Types**: Return the same types and shapes as NumPy equivalents
- **Error Handling**: Raise the same exceptions with similar messages as NumPy/SciPy

### GPU Programming Considerations
- **Memory Management**: Be mindful of GPU memory allocation and deallocation
- **Device Context**: Understand CUDA device context and multi-GPU scenarios
- **Performance**: Optimize for GPU execution patterns (parallel operations, memory coalescing)
- **Backend Support**: Consider both CUDA and ROCm backend compatibility

## Code Style and Standards

### Python Code Style
- Follow **PEP 8** and OpenStack Style Guidelines
- Use `pre-commit` for automatic style checking
- Run `pre-commit run -a` before submitting code
- Organize imports into three sections: standard libraries, third-party libraries, internal imports

### Import Guidelines
- **No relative imports** - use absolute imports only
- **No shortcut symbols** in library implementation (e.g., don't use `cupy.cuda.Device`, use `cupy.cuda.device.Device`)
- Shortcut symbols are allowed in `tests/` and `examples/` directories only

### Cython Guidelines
- For new Cython files, follow pointer type guidelines:
  - Use `void*` for pointers only used within Cython
  - Use `intptr_t` for pointers exposed to Python
  - Use `size_t` for memory sizes
  - Use `ptrdiff_t` for memory offsets

## Testing Requirements

### Test Structure
- All test classes must inherit from `unittest.TestCase`
- Place tests in appropriate subdirectories under `tests/`
- Use `assert` statements instead of `self.assert*` methods
- Use `with pytest.raises(...):` instead of `with self.assertRaises(...)`

### GPU Testing
- Use `@testing.multi_gpu(N)` decorator for multi-GPU tests
- Use `@testing.slow` decorator for time-consuming tests
- Consider both CPU and GPU test scenarios
- Test with different array dtypes and shapes

### Test Categories
```python
import unittest
from cupy import testing

class TestMyFunc(unittest.TestCase):
    @testing.multi_gpu(2)  # for multi-GPU tests
    def test_my_two_gpu_func(self):
        ...
    
    @testing.slow  # for slow tests
    def test_my_slow_func(self):
        ...
```

## Documentation Standards

### API Documentation
- Include comprehensive docstrings following NumPy documentation style
- Reference corresponding NumPy/SciPy functions when applicable
- Document any limitations or differences from NumPy/SciPy behavior
- Include examples demonstrating usage

### Code Comments
- Add comments for complex GPU-specific optimizations
- Explain non-obvious memory management decisions
- Document kernel launch configurations and performance considerations

## Development Workflow

### Before Making Changes
1. Build Cython files: `pip install --no-build-isolation -e .`
2. Run existing tests to ensure clean starting state
3. Check coding style with `pre-commit run -a`

### Branch Selection
- Use `main` branch for most contributions
- Follow versioned branch guidelines for specific releases
- Never commit directly to protected branches

### Pull Request Guidelines
- Include unit tests for new functionality
- Update documentation for API changes
- Ensure backward compatibility unless explicitly breaking
- Test on both CUDA and ROCm if applicable
- Include performance benchmarks for optimization changes

## Common Patterns

### Array Creation
```python
# Good: Use CuPy's array creation functions
import cupy as cp
x = cp.array([1, 2, 3])
y = cp.zeros((3, 3))

# Ensure proper device placement
with cp.cuda.Device(0):
    z = cp.ones((2, 2))
```

### Memory Management
```python
# Good: Explicit memory management when needed
import cupy as cp
mempool = cp.get_default_memory_pool()
with mempool:
    # Operations that need controlled memory usage
    result = cp.matmul(a, b)
```

### Error Handling
```python
# Good: Match NumPy error behavior
if not isinstance(arr, cp.ndarray):
    raise TypeError("Input must be a CuPy array")

if arr.ndim != 2:
    raise ValueError("Input must be 2-dimensional")
```

## What to Avoid

### Anti-patterns
- Don't create CPU-GPU copies unnecessarily
- Don't ignore device context in multi-GPU scenarios
- Don't break NumPy API compatibility without strong justification
- Don't add dependencies without careful consideration
- Don't use deprecated NumPy features
- Don't submit AI-generated content without thorough review and testing

### Performance Anti-patterns
- Avoid frequent host-device transfers
- Don't launch kernels with poor occupancy
- Avoid creating temporary arrays unnecessarily
- Don't ignore memory alignment and coalescing

## Community Guidelines

### Issue Reporting
- Only report bugs that impact real-world usage
- Provide minimal reproducible examples
- Include system information (GPU, CUDA version, etc.)
- Avoid hypothetical or pedantic scenarios

### Code Quality
- Ensure changes are production-ready
- Include comprehensive error handling
- Consider edge cases and boundary conditions
- Maintain high code coverage with meaningful tests

## Build and CI

### Local Development
- Use `pre-commit` hooks for automated checking
- Run relevant test suites before submitting
- Build documentation locally to verify changes
- Test with multiple Python versions if possible

### Continuous Integration
- GitHub Actions performs automated style and basic testing
- PFN CI handles GPU testing for CUDA on Linux/Windows
- Self-hosted CI tests AMD ROCm compatibility
- All checks must pass before merge consideration

This document should be regularly updated as the project evolves and new best practices emerge.