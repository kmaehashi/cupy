"""
GCP Cloud Storage backed kernel cache for CuPy.

This module provides an **EXPERIMENTAL** cache backend that stores compiled
kernel binaries in Google Cloud Platform (GCP) Cloud Storage. This is useful
in distributed environments where multiple machines need to share a common
kernel cache.

**WARNING**: This is an experimental feature and the API may change in future
releases.

Requirements:
    - google-cloud-storage: Install with `pip install google-cloud-storage`
    - GCP credentials configured (via GOOGLE_APPLICATION_CREDENTIALS or
      gcloud auth application-default login)

Usage:
    ```python
    from cupy.cuda.compiler import _set_kernel_cache_backend
    from examples.kernel_cache_backends.gcp_storage_backend import (
        GCPStorageCacheBackend
    )

    # Create the GCP storage backend
    backend = GCPStorageCacheBackend(
        bucket_name='my-cupy-kernel-cache',
        local_cache_dir='/tmp/cupy_cache',
        prefix='team/kernels/'
    )
    
    # Set it as the global cache backend
    _set_kernel_cache_backend(backend)
    ```

Features:
    - Downloads compiled kernels from GCP Cloud Storage
    - Persists downloaded files to local disk for faster future access
    - Uploads newly compiled kernels to GCP Cloud Storage for sharing
    - Falls back to local disk cache if GCP is unavailable
"""

from __future__ import annotations

import os
import warnings

from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions

# Import the base class from cupy
from cupy.cuda.compiler import DiskKernelCacheBackend


class GCPStorageCacheBackend(DiskKernelCacheBackend):
    """
    **EXPERIMENTAL** GCP Cloud Storage backed cache backend.

    This cache backend stores compiled kernel binaries in Google Cloud Storage
    for sharing across distributed environments. Downloaded files are persisted
    to local disk for faster future access.

    Args:
        bucket_name (str): Name of the GCS bucket to use for cache storage.
        local_cache_dir (str | None, optional): Local directory to cache
            downloaded files. Defaults to ~/.cupy/kernel_cache.
        prefix (str): Prefix to use for all cache keys in GCS.
        project (str | None, optional): GCP project ID. If None, uses the
            default project from credentials.

    Attributes:
        bucket_name (str): The GCS bucket name.
        prefix (str): The GCS key prefix for cache entries.

    Example:
        >>> backend = GCPStorageCacheBackend(
        ...     bucket_name='my-team-kernel-cache',
        ...     local_cache_dir='/tmp/cupy_cache',
        ...     prefix='prod/kernels/'
        ... )
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str,
        local_cache_dir: str | None = None,
        project: str | None = None,
    ) -> None:
        """Initialize the GCP Storage cache backend."""
        # Initialize the parent disk cache
        super().__init__(local_cache_dir)

        self.bucket_name = bucket_name
        self.prefix = prefix
        self._gcp_enabled = True

        try:
            # Initialize GCS client
            self._client = storage.Client(project=project)
            self._bucket = self._client.bucket(bucket_name)

            # Verify bucket exists and is accessible
            if not self._bucket.exists():
                warnings.warn(
                    f"GCS bucket '{bucket_name}' does not exist or is not accessible. "
                    "Will only use local disk cache.",
                    RuntimeWarning
                )
                self._gcp_enabled = False
        except Exception as e:
            warnings.warn(
                f"Failed to initialize GCS client: {e}. "
                "Will only use local disk cache.",
                RuntimeWarning
            )
            self._gcp_enabled = False
            self._bucket = None

    def load(self, name: str) -> bytes | None:
        """
        Load a cached kernel binary.

        First checks local disk cache, then falls back to GCS if not found.
        Downloaded files are persisted to local disk for future use.

        Args:
            name (str): The cache key (filename) for the compiled kernel.

        Returns:
            bytes or None: The cubin binary data (without hash prefix) if
                found and valid, None otherwise.
        """
        # First, try to load from local disk cache
        cubin = super().load(name)
        if cubin is not None:
            return cubin

        # If not in local cache and GCP is enabled, try to download from GCS
        if not self._gcp_enabled or self._bucket is None:
            return None

        try:
            gcs_key = self.prefix + name
            blob = self._bucket.blob(gcs_key)

            if blob.exists():
                # Download from GCS (this includes hash + cubin)
                data = blob.download_as_bytes()

                # Parse and validate before persisting
                # Note: super().save() will add hash, so we need to extract
                # cubin from downloaded data first
                from cupy.cuda._compiler_cache import _hash_length, _hash_hexdigest
                
                if len(data) < _hash_length:
                    return None
                
                hash_stored = data[:_hash_length]
                cubin = data[_hash_length:]
                cubin_hash = _hash_hexdigest(cubin).encode('ascii')
                
                if hash_stored != cubin_hash:
                    # Hash mismatch, corrupted cache
                    return None

                # Persist to local disk for future use
                # We pass empty string for source as we don't have it
                super().save(name, cubin, '')

                return cubin
        except gcp_exceptions.GoogleAPIError as e:
            warnings.warn(
                f"Failed to download from GCS: {e}. Using local cache only.",
                RuntimeWarning
            )
        except Exception as e:
            warnings.warn(
                f"Unexpected error downloading from GCS: {e}",
                RuntimeWarning
            )

        return None

    def save(self, name: str, cubin: bytes, source: str) -> None:
        """
        Save a compiled kernel binary to cache.

        Saves to both local disk and GCS (if enabled).

        Args:
            name (str): The cache key (filename) for the compiled kernel.
            cubin (bytes): The compiled kernel binary data.
            source (str): The CUDA source code.
        """
        # Always save to local disk first
        super().save(name, cubin, source)

        # If GCP is not enabled, early return
        if not self._gcp_enabled or self._bucket is None:
            return

        try:
            gcs_key = self.prefix + name
            blob = self._bucket.blob(gcs_key)

            # Need to upload with hash prefix for compatibility
            from cupy.cuda._compiler_cache import _hash_hexdigest
            cubin_hash = _hash_hexdigest(cubin).encode('ascii')
            data = cubin_hash + cubin

            # Upload to GCS
            blob.upload_from_string(data)
        except gcp_exceptions.GoogleAPIError as e:
            warnings.warn(
                f"Failed to upload to GCS: {e}. Kernel is cached locally only.",
                RuntimeWarning
            )
        except Exception as e:
            warnings.warn(
                f"Unexpected error uploading to GCS: {e}",
                RuntimeWarning
            )
