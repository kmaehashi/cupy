"""
Example demonstrating the use of custom cache backends for CuPy kernels.

This example shows how to:
1. Use the default DiskCacheBackend
2. Create and use a GCPStorageCacheBackend
3. Implement a simple custom cache backend

Note: This is an experimental feature. The API may change in future releases.
"""

import os
import tempfile
import warnings


def example_disk_backend():
    """Example using the built-in disk cache backend."""
    print("\n=== Example 1: Disk Cache Backend ===")
    
    from cupy.cuda._compiler_cache import DiskKernelCacheBackend
    import hashlib
    
    # Create a disk cache backend with a custom directory
    cache_dir = tempfile.mkdtemp(prefix='cupy_cache_example_')
    backend = DiskKernelCacheBackend(cache_dir=cache_dir)
    
    print(f"Cache directory: {cache_dir}")
    
    # Demonstrate the backend interface
    test_name = "test_kernel.cubin"
    cubin_data = b"compiled_kernel_binary_data"
    # Create proper hash (SHA1 of the cubin data, hex-encoded)
    cubin_hash = hashlib.sha1(cubin_data).hexdigest().encode('ascii')
    test_data = cubin_hash + cubin_data
    
    # Save data
    print(f"Saving test data to cache...")
    backend.save(test_name, test_data)
    
    # Check existence
    print(f"Checking if '{test_name}' exists: {backend.exists(test_name)}")
    
    # Load data
    loaded_data = backend.load(test_name)
    print(f"Loaded data matches saved data: {loaded_data == test_data}")
    
    print(f"\nCache directory contains: {os.listdir(cache_dir)}")


def example_gcp_backend():
    """Example using the GCP Storage cache backend."""
    print("\n=== Example 2: GCP Storage Cache Backend ===")
    
    try:
        from examples.kernel_cache_backends.gcp_storage_backend import (
            GCPStorageCacheBackend
        )
    except ImportError as e:
        print(f"Failed to import GCP backend: {e}")
        return
    
    import hashlib
    
    # Create a GCP cache backend
    # Note: This requires a valid GCP bucket and credentials
    bucket_name = os.environ.get('CUPY_GCP_BUCKET', 'cupy-kernel-cache-example')
    
    print(f"Creating GCP backend with bucket: {bucket_name}")
    print("Note: This requires valid GCP credentials and an existing bucket.")
    
    try:
        backend = GCPStorageCacheBackend(
            bucket_name=bucket_name,
            local_cache_dir=tempfile.mkdtemp(prefix='cupy_gcp_cache_'),
            prefix='examples/'
        )
        
        # The backend automatically falls back to local cache if GCP is unavailable
        test_name = "example_kernel.cubin"
        cubin_data = b"another_compiled_kernel"
        # Create proper hash (SHA1 of the cubin data, hex-encoded)
        cubin_hash = hashlib.sha1(cubin_data).hexdigest().encode('ascii')
        test_data = cubin_hash + cubin_data
        
        print(f"Saving test data...")
        backend.save(test_name, test_data)
        
        print(f"Checking if '{test_name}' exists: {backend.exists(test_name)}")
        
        loaded_data = backend.load(test_name)
        print(f"Loaded data matches saved data: {loaded_data == test_data}")
        
    except Exception as e:
        print(f"Error with GCP backend: {e}")
        print("The backend will fall back to local disk cache.")


def example_custom_backend():
    """Example implementing a simple in-memory cache backend."""
    print("\n=== Example 3: Custom In-Memory Cache Backend ===")
    
    from cupy.cuda._compiler_cache import KernelCacheBackend
    import hashlib
    
    class InMemoryCacheBackend(KernelCacheBackend):
        """Simple in-memory cache backend for demonstration."""
        
        def __init__(self):
            self._cache = {}
        
        def load(self, name):
            return self._cache.get(name)
        
        def save(self, name, data):
            self._cache[name] = data
    
    # Create and use the in-memory backend
    backend = InMemoryCacheBackend()
    
    test_name = "memory_kernel.cubin"
    cubin_data = b"in_memory_kernel"
    # Create proper hash (SHA1 of the cubin data, hex-encoded)
    cubin_hash = hashlib.sha1(cubin_data).hexdigest().encode('ascii')
    test_data = cubin_hash + cubin_data
    
    print("Saving test data to in-memory cache...")
    backend.save(test_name, test_data)
    
    print(f"Loaded data matches saved data: {backend.load(test_name) == test_data}")


def example_with_real_kernel():
    """Example using a cache backend with a real CuPy kernel compilation."""
    print("\n=== Example 4: Real Kernel Compilation with Custom Backend ===")
    
    try:
        import cupy
        from cupy.cuda import compiler
        from cupy.cuda._compiler_cache import DiskKernelCacheBackend
        from cupy.cuda.compiler import _set_kernel_cache_backend
    except ImportError as e:
        print(f"CuPy is not installed: {e}")
        return
    
    # Create a custom cache backend
    cache_dir = tempfile.mkdtemp(prefix='cupy_real_kernel_cache_')
    backend = DiskKernelCacheBackend(cache_dir=cache_dir)
    
    print(f"Using cache directory: {cache_dir}")
    
    # Set the global cache backend
    _set_kernel_cache_backend(backend)
    
    # Simple kernel source
    kernel_source = r'''
    extern "C" __global__
    void my_kernel(float* out, const float* in, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            out[tid] = in[tid] * 2.0f;
        }
    }
    '''
    
    print("Compiling kernel with custom cache backend...")
    print("Note: Using _set_kernel_cache_backend to change the global backend.")
    
    try:
        # Compile with the custom backend (now uses global backend)
        module = compiler._compile_module_with_cache(
            kernel_source,
            options=()
        )
        
        print("Kernel compiled successfully!")
        print(f"Cache directory now contains: {os.listdir(cache_dir)}")
        
        # Try to compile again - should use cache
        print("\nCompiling the same kernel again (should use cache)...")
        module2 = compiler._compile_module_with_cache(
            kernel_source,
            options=()
        )
        print("Second compilation completed (used cached version).")
        
    except Exception as e:
        print(f"Error during compilation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all examples."""
    print("=" * 70)
    print("CuPy Kernel Cache Backend Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the experimental cache backend API.")
    print("The API may change in future releases.")
    
    # Run examples
    example_disk_backend()
    example_gcp_backend()
    example_custom_backend()
    
    # Check if CUDA is available before running real kernel example
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            example_with_real_kernel()
        else:
            print("\n=== Example 4: Skipped (No CUDA device available) ===")
    except Exception:
        print("\n=== Example 4: Skipped (CuPy not available or no CUDA) ===")
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()
