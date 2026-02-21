from __future__ import annotations

import hashlib
import os
import tempfile
import unittest

from cupy.cuda._compiler_cache import (
    DiskKernelCacheBackend,
    KernelCacheBackend,
    _hash_hexdigest,
    _hash_length,
    _default_cache_dir,
)


class TestHashFunction(unittest.TestCase):
    """Tests for the hash utility function."""

    def test_hash_hexdigest(self):
        """Test that hash function produces correct SHA1 hex digest."""
        test_data = b'test_data'
        expected_hash = hashlib.sha1(
            test_data, usedforsecurity=False).hexdigest()
        result = _hash_hexdigest(test_data)
        self.assertEqual(result, expected_hash)
        self.assertIsInstance(result, str)

    def test_hash_hexdigest_empty(self):
        """Test hash of empty bytes."""
        result = _hash_hexdigest(b'')
        expected = hashlib.sha1(b'', usedforsecurity=False).hexdigest()
        self.assertEqual(result, expected)

    def test_hash_length_constant(self):
        """Test that _hash_length is correct for SHA1 (40 hex chars)."""
        self.assertEqual(_hash_length, 40)

    def test_hash_hexdigest_length(self):
        """Test that hash output length matches _hash_length."""
        test_data = b'some random data for testing'
        result = _hash_hexdigest(test_data)
        self.assertEqual(len(result), _hash_length)


class TestKernelCacheBackendInterface(unittest.TestCase):
    """Tests for KernelCacheBackend abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that KernelCacheBackend cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            KernelCacheBackend()

    def test_must_implement_load(self):
        """Test that subclasses must implement load method."""
        class IncompleteBackend(KernelCacheBackend):
            def save(self, name: str, cubin: bytes, source: str) -> None:
                pass

        with self.assertRaises(TypeError):
            IncompleteBackend()

    def test_must_implement_save(self):
        """Test that subclasses must implement save method."""
        class IncompleteBackend(KernelCacheBackend):
            def load(self, name: str) -> bytes | None:
                return None

        with self.assertRaises(TypeError):
            IncompleteBackend()

    def test_complete_implementation(self):
        """Test that complete implementation can be instantiated."""
        class CompleteBackend(KernelCacheBackend):
            def load(self, name: str) -> bytes | None:
                return None

            def save(self, name: str, cubin: bytes, source: str) -> None:
                pass

        backend = CompleteBackend()
        self.assertIsNotNone(backend)
        self.assertIsNone(backend.load('test'))
        backend.save('test', b'data', 'source')  # Should not raise


class TestDiskKernelCacheBackend(unittest.TestCase):
    """Tests for DiskKernelCacheBackend implementation."""

    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory."""
        # Save original env var
        original_env = os.environ.get('CUPY_CACHE_DIR')
        try:
            # Unset the environment variable
            if 'CUPY_CACHE_DIR' in os.environ:
                del os.environ['CUPY_CACHE_DIR']

            backend = DiskKernelCacheBackend()
            self.assertTrue(
                backend._cache_dir.endswith('.cupy/kernel_cache'))
        finally:
            # Restore original env var
            if original_env is not None:
                os.environ['CUPY_CACHE_DIR'] = original_env

    def test_init_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, 'custom_cache')
            backend = DiskKernelCacheBackend(cache_dir=cache_dir)
            self.assertEqual(backend._cache_dir, cache_dir)
            self.assertTrue(os.path.isdir(cache_dir))

    def test_init_env_var_cache_dir(self):
        """Test initialization with CUPY_CACHE_DIR environment variable."""
        original_env = os.environ.get('CUPY_CACHE_DIR')
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                env_cache_dir = os.path.join(tmpdir, 'env_cache')
                os.environ['CUPY_CACHE_DIR'] = env_cache_dir

                backend = DiskKernelCacheBackend()
                self.assertEqual(backend._cache_dir, env_cache_dir)
                self.assertTrue(os.path.isdir(env_cache_dir))
        finally:
            if original_env is not None:
                os.environ['CUPY_CACHE_DIR'] = original_env
            elif 'CUPY_CACHE_DIR' in os.environ:
                del os.environ['CUPY_CACHE_DIR']

    def test_save_and_load(self):
        """Test basic save and load operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'test_kernel.cubin'
            cubin = b'compiled_kernel_binary'
            source = 'extern "C" __global__ void test() {}'

            # Save the kernel
            backend.save(name, cubin, source)

            # Load it back
            loaded_cubin = backend.load(name)
            self.assertEqual(loaded_cubin, cubin)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            result = backend.load('nonexistent.cubin')
            self.assertIsNone(result)

    def test_load_file_too_short(self):
        """Test loading a file that's too short to contain a hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            # Write a file with less than _hash_length bytes
            name = 'short.cubin'
            path = os.path.join(tmpdir, name)
            with open(path, 'wb') as f:
                f.write(b'too_short')

            result = backend.load(name)
            self.assertIsNone(result)

    def test_load_corrupted_hash(self):
        """Test that corrupted cache files are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'corrupted.cubin'
            path = os.path.join(tmpdir, name)

            # Write file with wrong hash
            cubin = b'kernel_data'
            wrong_hash = b'0' * _hash_length  # Wrong hash
            with open(path, 'wb') as f:
                f.write(wrong_hash + cubin)

            # Load should return None due to hash mismatch
            result = backend.load(name)
            self.assertIsNone(result)

    def test_save_creates_file_with_hash(self):
        """Test that save creates a file with hash prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'test.cubin'
            cubin = b'test_binary_data'
            source = '__global__ void test() {}'

            backend.save(name, cubin, source)

            # Read the file directly and verify structure
            path = os.path.join(tmpdir, name)
            with open(path, 'rb') as f:
                file_data = f.read()

            # Check that file contains hash + cubin
            self.assertGreater(len(file_data), _hash_length)

            stored_hash = file_data[:_hash_length]
            stored_cubin = file_data[_hash_length:]

            # Verify the stored cubin matches
            self.assertEqual(stored_cubin, cubin)

            # Verify the hash is correct
            expected_hash = _hash_hexdigest(cubin).encode('ascii')
            self.assertEqual(stored_hash, expected_hash)

    def test_save_overwrites_existing_file(self):
        """Test that save overwrites an existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'test.cubin'
            cubin1 = b'first_version'
            cubin2 = b'second_version'
            source = '__global__ void test() {}'

            # Save first version
            backend.save(name, cubin1, source)
            loaded1 = backend.load(name)
            self.assertEqual(loaded1, cubin1)

            # Save second version (overwrite)
            backend.save(name, cubin2, source)
            loaded2 = backend.load(name)
            self.assertEqual(loaded2, cubin2)

    def test_save_source_file_when_env_var_set(self):
        """Test that .cu source file is saved when env var is set."""
        original_env = os.environ.get('CUPY_CACHE_SAVE_CUDA_SOURCE')
        try:
            os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'

            with tempfile.TemporaryDirectory() as tmpdir:
                backend = DiskKernelCacheBackend(cache_dir=tmpdir)

                name = 'test.cubin'
                cubin = b'compiled_data'
                source = '__global__ void test() { /* code */ }'

                backend.save(name, cubin, source)

                # Check that .cu file was created
                cu_path = os.path.join(tmpdir, name + '.cu')
                self.assertTrue(os.path.exists(cu_path))

                # Verify source content
                with open(cu_path) as f:
                    saved_source = f.read()
                self.assertEqual(saved_source, source)
        finally:
            if original_env is not None:
                os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = original_env
            elif 'CUPY_CACHE_SAVE_CUDA_SOURCE' in os.environ:
                del os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE']

    def test_save_source_file_not_saved_by_default(self):
        """Test that .cu source file is not saved by default."""
        original_env = os.environ.get('CUPY_CACHE_SAVE_CUDA_SOURCE')
        try:
            # Ensure env var is not set
            if 'CUPY_CACHE_SAVE_CUDA_SOURCE' in os.environ:
                del os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE']

            with tempfile.TemporaryDirectory() as tmpdir:
                backend = DiskKernelCacheBackend(cache_dir=tmpdir)

                name = 'test.cubin'
                cubin = b'compiled_data'
                source = '__global__ void test() {}'

                backend.save(name, cubin, source)

                # Check that .cu file was NOT created
                cu_path = os.path.join(tmpdir, name + '.cu')
                self.assertFalse(os.path.exists(cu_path))
        finally:
            if original_env is not None:
                os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = original_env

    def test_multiple_kernels(self):
        """Test saving and loading multiple different kernels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            # Save multiple kernels
            kernels = {
                'kernel1.cubin': b'binary_data_1',
                'kernel2.cubin': b'binary_data_2',
                'kernel3.cubin': b'binary_data_3',
            }

            for name, cubin in kernels.items():
                backend.save(name, cubin, 'source')

            # Load and verify each kernel
            for name, expected_cubin in kernels.items():
                loaded_cubin = backend.load(name)
                self.assertEqual(loaded_cubin, expected_cubin)

    def test_save_empty_cubin(self):
        """Test saving and loading empty cubin data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'empty.cubin'
            cubin = b''
            source = '// empty'

            backend.save(name, cubin, source)
            loaded = backend.load(name)
            self.assertEqual(loaded, cubin)

    def test_save_large_cubin(self):
        """Test saving and loading large cubin data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = DiskKernelCacheBackend(cache_dir=tmpdir)

            name = 'large.cubin'
            # Create large binary data (1 MB)
            cubin = b'x' * (1024 * 1024)
            source = '// large kernel'

            backend.save(name, cubin, source)
            loaded = backend.load(name)
            self.assertEqual(loaded, cubin)
            self.assertEqual(len(loaded), len(cubin))

    def test_cache_dir_with_special_chars(self):
        """Test cache directory with special characters in name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, 'cache-dir_with.special@chars')
            backend = DiskKernelCacheBackend(cache_dir=cache_dir)

            self.assertTrue(os.path.isdir(cache_dir))

            name = 'test.cubin'
            cubin = b'data'
            backend.save(name, cubin, 'source')
            loaded = backend.load(name)
            self.assertEqual(loaded, cubin)


class TestDefaultCacheDir(unittest.TestCase):
    """Tests for default cache directory constant."""

    def test_default_cache_dir_format(self):
        """Test that default cache dir has expected format."""
        self.assertIn('.cupy', _default_cache_dir)
        self.assertIn('kernel_cache', _default_cache_dir)

    def test_default_cache_dir_expanded(self):
        """Test that default cache dir is expanded."""
        # Should not contain ~ after expansion
        self.assertNotIn('~', _default_cache_dir)
        # Should be an absolute path
        self.assertTrue(os.path.isabs(_default_cache_dir))
