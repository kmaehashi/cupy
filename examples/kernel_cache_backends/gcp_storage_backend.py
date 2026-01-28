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
    from cupy.cuda.compiler import CacheBackend
    import cupy

    # Create the GCP storage backend
    backend = GCPStorageCacheBackend(
        bucket_name='my-cupy-kernel-cache',
        local_cache_dir='/tmp/cupy_cache'
    )

    # Use it for kernel compilation
    # Note: You'll need to modify cupy internals to pass this backend
    # This is just an example of how it could be used
    ```

Features:
    - Downloads compiled kernels from GCP Cloud Storage
    - Persists downloaded files to local disk for faster future access
    - Uploads newly compiled kernels to GCP Cloud Storage for sharing
    - Falls back to local disk cache if GCP is unavailable
"""

import os
import warnings

from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions

# Import the base class from cupy
from cupy.cuda.compiler import KernelCacheBackend, DiskKernelCacheBackend


class GCPStorageCacheBackend(DiskKernelCacheBackend):
    """
    **EXPERIMENTAL** GCP Cloud Storage backed cache backend.

    This cache backend stores compiled kernel binaries in Google Cloud Storage
    for sharing across distributed environments. Downloaded files are persisted
    to local disk for faster future access.

    Args:
        bucket_name (str): Name of the GCS bucket to use for cache storage.
        local_cache_dir (str, optional): Local directory to cache downloaded
            files. Defaults to ~/.cupy/kernel_cache.
        prefix (str, optional): Prefix to use for all cache keys in GCS.
            Defaults to 'cupy_kernels/'.
        project (str, optional): GCP project ID. If None, uses the default
            project from credentials.

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

    def __init__(self, bucket_name, local_cache_dir=None, prefix='cupy_kernels/',
                 project=None):
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

    def _get_gcs_key(self, name):
        """Get the full GCS key for a cache entry.

        Args:
            name (str): The cache entry name (filename).

        Returns:
            str: The full GCS key with prefix.
        """
        return self.prefix + name

    def load(self, name):
        """
        Load a cached kernel binary.

        First checks local disk cache, then falls back to GCS if not found.
        Downloaded files are persisted to local disk for future use.

        Args:
            name (str): The cache key (filename) for the compiled kernel.

        Returns:
            bytes or None: The cached binary data if found and valid,
                None otherwise.
        """
        # First, try to load from local disk cache
        data = super().load(name)
        if data is not None:
            return data

        # If not in local cache and GCP is enabled, try to download from GCS
        if self._gcp_enabled and self._bucket is not None:
            try:
                gcs_key = self._get_gcs_key(name)
                blob = self._bucket.blob(gcs_key)

                if blob.exists():
                    # Download from GCS
                    data = blob.download_as_bytes()

                    # Persist to local disk for future use
                    super().save(name, data)

                    return data
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

    def save(self, name, data):
        """
        Save a compiled kernel binary to cache.

        Saves to both local disk and GCS (if enabled).

        Args:
            name (str): The cache key (filename) for the compiled kernel.
            data (bytes): The binary data to cache (hash + cubin).
        """
        # Always save to local disk first
        super().save(name, data)

        # If GCP is enabled, also upload to GCS
        if self._gcp_enabled and self._bucket is not None:
            try:
                gcs_key = self._get_gcs_key(name)
                blob = self._bucket.blob(gcs_key)

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
