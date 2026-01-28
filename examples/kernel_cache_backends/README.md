# Kernel Cache Backends

This directory contains **EXPERIMENTAL** cache backend implementations for CuPy's kernel compilation cache.

## Overview

CuPy compiles CUDA kernels at runtime and caches the compiled binaries to avoid recompilation. By default, CuPy uses a disk-based cache stored in `~/.cupy/kernel_cache`. This works well for single machines but can be inefficient in distributed environments.

The cache backend abstraction allows you to plug in alternative storage backends for the kernel cache, enabling shared caches across multiple machines.

## ⚠️ Experimental Status

**WARNING**: The cache backend API is experimental and may change in future releases. These examples are provided for advanced users who want to experiment with distributed caching solutions.

## Available Backends

### DiskCacheBackend (Built-in)

The default disk-based cache backend. Stores compiled kernels in a local directory.

```python
from cupy.cuda.compiler import DiskCacheBackend

backend = DiskCacheBackend('/path/to/cache')
```

### GCPStorageCacheBackend (Example)

Stores compiled kernels in Google Cloud Storage (GCS) for sharing across distributed environments. Downloaded files are persisted to local disk for faster future access.

**Requirements:**
- `google-cloud-storage` package: `pip install google-cloud-storage`
- GCP credentials configured

**Usage:**
```python
from examples.kernel_cache_backends.gcp_storage_backend import GCPStorageCacheBackend

backend = GCPStorageCacheBackend(
    bucket_name='my-team-kernel-cache',
    local_cache_dir='/tmp/cupy_cache',
    prefix='prod/kernels/'
)
```

## Creating Custom Backends

To create a custom cache backend, inherit from `cupy.cuda.compiler.CacheBackend` and implement three methods:

```python
from cupy.cuda.compiler import CacheBackend

class MyCustomBackend(CacheBackend):
    def load(self, name):
        """
        Load a cached kernel binary.
        
        Args:
            name (str): The cache key (filename) for the compiled kernel.
            
        Returns:
            bytes or None: The cached binary data if found and valid,
                None otherwise.
        """
        # Your implementation here
        pass
    
    def save(self, name, data):
        """
        Save a compiled kernel binary to cache.
        
        Args:
            name (str): The cache key (filename) for the compiled kernel.
            data (bytes): The binary data to cache (hash + cubin).
        """
        # Your implementation here
        pass
    
    def exists(self, name):
        """
        Check if a cached kernel binary exists.
        
        Args:
            name (str): The cache key (filename) for the compiled kernel.
            
        Returns:
            bool: True if the cache entry exists, False otherwise.
        """
        # Your implementation here
        pass
```

### Best Practices

1. **Inherit from DiskCacheBackend**: If your backend needs local persistence, inherit from `DiskCacheBackend` instead of `CacheBackend` to get disk caching for free.

2. **Hash Validation**: The cache data includes a hash prefix for integrity checking. Make sure your backend preserves the entire data blob.

3. **Error Handling**: Implement graceful fallbacks for network or storage failures. The GCP backend example shows how to fall back to local cache.

4. **Thread Safety**: Your backend should be safe to use from multiple threads, as CuPy may compile kernels concurrently.

## Using a Custom Backend

Currently, custom backends need to be passed programmatically when calling compilation functions. The API for this is still being designed.

For now, you can experiment by:

1. Importing the backend class
2. Modifying your code to pass the backend to `_compile_module_with_cache` via the `cache_backend` parameter

Example:
```python
import cupy
from cupy.cuda import compiler
from examples.kernel_cache_backends.gcp_storage_backend import GCPStorageCacheBackend

# Create custom backend
backend = GCPStorageCacheBackend(bucket_name='my-cache-bucket')

# Use it in compilation (requires modifying CuPy internals)
# This is for experimentation only
```

## Ideas for Other Backends

Here are some ideas for other cache backends you could implement:

- **AWS S3**: Similar to GCS backend, using boto3
- **Azure Blob Storage**: Using azure-storage-blob
- **Redis**: For in-memory distributed caching
- **NFS/Shared Filesystem**: For cluster environments with shared storage
- **HTTP Cache**: For read-only caching from a web server

## Contributing

If you develop a useful cache backend, consider contributing it back to the CuPy project! Please open an issue or pull request on the [CuPy GitHub repository](https://github.com/cupy/cupy).

## License

These examples are part of CuPy and are distributed under the same license as CuPy (MIT License).
