from __future__ import annotations

import os
import pickle
import unittest
from unittest import mock

import cupy
from cupy.cuda import compiler


def cuda_version():
    return cupy.cuda.runtime.runtimeGetVersion()


@unittest.skipIf(cupy.cuda.runtime.is_hip, 'CUDA specific tests')
class TestNvrtcArch(unittest.TestCase):
    def setUp(self):
        cupy.clear_memo()  # _get_arch result is cached

    def _check_get_arch(self, device_cc, expected_arch):
        with mock.patch('cupy.cuda.device.Device') as device_class:
            device_class.return_value.compute_capability = device_cc
            assert compiler._get_arch() == expected_arch
        cupy.clear_memo()  # _get_arch result is cached

    @unittest.skipUnless(
        9000 <= cuda_version() < 13000, 'Requires CUDA 9.x-12.x')
    def test_get_arch_cuda9(self):
        self._check_get_arch('62', '62')  # Tegra
        self._check_get_arch('70', '70')
        self._check_get_arch('72', '72')  # Tegra

    @unittest.skipUnless(10010 <= cuda_version(),
                         'Requires CUDA 10.1 or later')
    def test_get_arch_cuda101(self):
        self._check_get_arch('75', '75')

    @unittest.skipUnless(11000 <= cuda_version(),
                         'Requires CUDA 11.0 or later')
    def test_get_arch_cuda11(self):
        self._check_get_arch('80', '80')

    @unittest.skipUnless(12080 <= cuda_version(),
                         'Requires CUDA 12.8 or later')
    def test_get_arch_cuda128(self):
        self._check_get_arch('100', '100')
        self._check_get_arch('120', '120')

    def _compile(self, arch):
        compiler.compile_using_nvrtc('', arch=arch)

    @unittest.skipUnless(
        9000 <= cuda_version() < 13000, 'Requires CUDA 9.x-12.x')
    def test_compile_cuda9(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_72` as it is for Tegra.)
        self._compile('70')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, '73')

    @unittest.skipUnless(10010 <= cuda_version() < 11000,
                         'Requires CUDA 10.1 or 10.2')
    def test_compile_cuda101(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        # (Do not test `compute_72` as it is for Tegra.)
        self._compile('75')

        # It should fail. (compute_80 is not supported until CUDA 11)
        self.assertRaises(
            compiler.CompileException, self._compile, '80')

    @unittest.skipUnless(11000 <= cuda_version(),
                         'Requires CUDA 11.0 or later')
    def test_compile_cuda11(self):
        # This test is intended to detect specification change in NVRTC API.

        # It should not fail.
        self._compile('80')

        # It should fail.
        self.assertRaises(
            compiler.CompileException, self._compile, '83')


class TestNvrtcStderr(unittest.TestCase):

    @unittest.skipIf(cupy.cuda.runtime.is_hip,
                     'HIPRTC has different error message')
    def test1(self):
        # An error message contains the file name `kern.cu`
        with self.assertRaisesRegex(compiler.CompileException, 'kern.cu'):
            compiler.compile_using_nvrtc('a')

    @unittest.skipIf(not cupy.cuda.runtime.is_hip,
                     'NVRTC has different error message')
    def test2(self):
        with self.assertRaises(compiler.CompileException) as e:
            compiler.compile_using_nvrtc('a')
            assert "unknown type name 'a'" in e


class TestIsValidKernelName(unittest.TestCase):

    def test_valid(self):
        assert compiler.is_valid_kernel_name('valid_name_1')

    def test_empty(self):
        assert not compiler.is_valid_kernel_name('')

    def test_start_with_digit(self):
        assert not compiler.is_valid_kernel_name('0_invalid')

    def test_new_line(self):
        assert not compiler.is_valid_kernel_name('invalid\nname')

    def test_symbol(self):
        assert not compiler.is_valid_kernel_name('invalid$name')

    def test_space(self):
        assert not compiler.is_valid_kernel_name('invalid name')


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = compiler.CompileException('msg', 'fn.cu', 'fn', ('-ftz=true',))
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


class TestCompileWithCache:
    def test_compile_module_with_cache(self):
        compiler._compile_module_with_cache('__device__ void func() {}')


class TestCacheBackend(unittest.TestCase):
    """Tests for cache backend abstraction."""

    def test_disk_cache_backend_save_and_load(self):
        """Test basic save and load operations."""
        import tempfile
        import os
        import hashlib
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = compiler.DiskKernelCacheBackend(cache_dir=tmpdir)
            
            # Test data with proper hash
            name = 'test_kernel.cubin'
            cubin_data = b'kernel_binary_data'
            cubin_hash = hashlib.sha1(cubin_data).hexdigest().encode('ascii')
            test_data = cubin_hash + cubin_data
            
            # Save data
            backend.save(name, test_data)
            
            # Load data
            loaded_data = backend.load(name)
            self.assertEqual(loaded_data, test_data)
            
            # Verify file was created
            cache_path = os.path.join(tmpdir, name)
            self.assertTrue(os.path.exists(cache_path))

    def test_disk_cache_backend_load_nonexistent(self):
        """Test loading a non-existent cache entry."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = compiler.DiskKernelCacheBackend(cache_dir=tmpdir)
            
            # Try to load non-existent file
            result = backend.load('nonexistent.cubin')
            self.assertIsNone(result)

    def test_disk_cache_backend_hash_validation(self):
        """Test that corrupted cache data is rejected."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = compiler.DiskKernelCacheBackend(cache_dir=tmpdir)
            
            name = 'test_kernel.cubin'
            # Create corrupted data (wrong hash)
            corrupted_data = b'wrong_hash_1234567890123456789012345678' + b'kernel_data'
            
            # Write corrupted data directly to file
            cache_path = os.path.join(tmpdir, name)
            with open(cache_path, 'wb') as f:
                f.write(corrupted_data)
            
            # Load should return None due to hash mismatch
            result = backend.load(name)
            self.assertIsNone(result)

    def test_disk_cache_backend_default_directory(self):
        """Test that default cache directory is used when none specified."""
        backend = compiler.DiskKernelCacheBackend()
        
        # Should use the default cache directory
        expected_default = os.environ.get('CUPY_CACHE_DIR', 
                                         os.path.expanduser('~/.cupy/kernel_cache'))
        self.assertEqual(backend.get_cache_dir(), expected_default)

    def test_compile_with_custom_backend(self):
        """Test compilation with a custom cache backend."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = compiler.DiskKernelCacheBackend(cache_dir=tmpdir)
            
            # Set the global backend
            original_backend = compiler._kernel_cache_backend
            compiler._set_kernel_cache_backend(backend)
            
            try:
                # Simple kernel source
                source = '__device__ void test_func() {}'
                
                # First compilation - should create cache entry
                result1 = compiler._compile_module_with_cache(source)
                self.assertIsNotNone(result1)
                
                # Check that something was cached
                cache_files = os.listdir(tmpdir)
                self.assertGreater(len(cache_files), 0)
                
                # Second compilation - should use cache
                result2 = compiler._compile_module_with_cache(source)
                self.assertIsNotNone(result2)
            finally:
                # Restore the original backend
                compiler._set_kernel_cache_backend(original_backend)

    def test_cache_backend_interface(self):
        """Test that KernelCacheBackend is an abstract interface."""
        # KernelCacheBackend is an ABC and should not be instantiable directly
        with self.assertRaises(TypeError):
            compiler.KernelCacheBackend()
        
        # Create a concrete implementation for testing
        class TestBackend(compiler.KernelCacheBackend):
            def load(self, name):
                return None
            
            def save(self, name, data):
                pass
        
        # Concrete implementation should be instantiable
        backend = TestBackend()
        self.assertIsNone(backend.load('test'))
        backend.save('test', b'data')  # Should not raise
