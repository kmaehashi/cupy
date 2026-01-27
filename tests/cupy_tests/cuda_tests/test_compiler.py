from __future__ import annotations

import json
import os
import pickle
import tempfile
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


class TestCacheDebug(unittest.TestCase):
    """Tests for the cache debug feature."""

    def setUp(self):
        # Save original environment
        self.original_env = os.environ.get('CUPY_CACHE_DEBUG')
        # Reset the tracker for each test
        compiler._cache_tracker.reset()

    def tearDown(self):
        # Restore original environment
        if self.original_env is None:
            os.environ.pop('CUPY_CACHE_DEBUG', None)
        else:
            os.environ['CUPY_CACHE_DEBUG'] = self.original_env
        # Reset the tracker after each test
        compiler._cache_tracker.reset()

    def test_cache_debug_disabled_by_default(self):
        """Test that cache debug is disabled by default (zero overhead)."""
        os.environ.pop('CUPY_CACHE_DEBUG', None)
        compiler._cache_tracker.initialize()
        assert compiler._cache_tracker._mode is None

        # Recording should be no-op
        compiler._cache_tracker.record_hit('key1')
        compiler._cache_tracker.record_miss('key2')
        assert compiler._cache_tracker._hit_count == 0
        assert compiler._cache_tracker._miss_count == 0

    def test_cache_debug_stats_mode(self):
        """Test stats mode that only tracks hit/miss counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'stats.json')
            os.environ['CUPY_CACHE_DEBUG'] = f'stats:{output_file}'

            # Initialize and record some hits/misses
            compiler._cache_tracker.initialize()
            assert compiler._cache_tracker._mode == 'stats'

            compiler._cache_tracker.record_hit('key1', 'src1')
            compiler._cache_tracker.record_hit('key2', 'src2')
            compiler._cache_tracker.record_miss('key3', 'src3')

            # Write results
            compiler._cache_tracker._write_results()

            # Verify output file
            assert os.path.exists(output_file)
            with open(output_file) as f:
                data = json.load(f)

            assert data['mode'] == 'stats'
            assert data['cache_hits'] == 2
            assert data['cache_misses'] == 1
            assert data['total_lookups'] == 3
            assert abs(data['hit_ratio'] - 2/3) < 0.001
            # Stats mode should not include records
            assert 'records' not in data

    def test_cache_debug_debug_mode(self):
        """Test debug mode that records full details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'debug.json')
            os.environ['CUPY_CACHE_DEBUG'] = f'debug:{output_file}'

            # Initialize and record some hits/misses
            compiler._cache_tracker.initialize()
            assert compiler._cache_tracker._mode == 'debug'

            compiler._cache_tracker.record_hit('hash1', 'source1')
            compiler._cache_tracker.record_miss('hash2', 'source2')
            compiler._cache_tracker.record_hit('hash3', 'source3')

            # Write results
            compiler._cache_tracker._write_results()

            # Verify output file
            assert os.path.exists(output_file)
            with open(output_file) as f:
                data = json.load(f)

            assert data['mode'] == 'debug'
            assert data['summary']['cache_hits'] == 2
            assert data['summary']['cache_misses'] == 1
            assert data['summary']['total_lookups'] == 3
            assert abs(data['summary']['hit_ratio'] - 2/3) < 0.001

            # Debug mode should include records
            assert 'records' in data
            assert len(data['records']) == 3
            assert data['records'][0]['type'] == 'hit'
            assert data['records'][0]['hashed_key'] == 'hash1'
            assert data['records'][0]['cache_key'] == 'source1'
            assert data['records'][1]['type'] == 'miss'
            assert data['records'][1]['hashed_key'] == 'hash2'
            assert data['records'][2]['type'] == 'hit'

    def test_cache_debug_default_path(self):
        """Test that default path is used when only mode is specified."""
        os.environ['CUPY_CACHE_DEBUG'] = 'stats'
        compiler._cache_tracker.initialize()
        assert compiler._cache_tracker._mode == 'stats'
        assert compiler._cache_tracker._output_path == 'cupy_cache_debug.json'

    def test_cache_debug_invalid_mode(self):
        """Test that invalid mode triggers a warning."""
        os.environ['CUPY_CACHE_DEBUG'] = 'invalid_mode'
        with self.assertWarns(RuntimeWarning):
            compiler._cache_tracker.initialize()
        assert compiler._cache_tracker._mode is None

    def test_cache_debug_with_actual_compilation(self):
        """Test cache debug with actual compilation to ensure integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, 'cache')
            output_file = os.path.join(tmpdir, 'debug.json')
            os.environ['CUPY_CACHE_DEBUG'] = f'debug:{output_file}'

            # Reset tracker to ensure fresh state
            compiler._cache_tracker.reset()

            # First compilation - should be a cache miss
            source = '__global__ void test_kernel() {}'
            compiler._compile_module_with_cache(
                source, cache_dir=cache_dir)

            # Second compilation with same source - should be a cache hit
            compiler._compile_module_with_cache(
                source, cache_dir=cache_dir)

            # Write results
            compiler._cache_tracker._write_results()

            # Verify results
            assert os.path.exists(output_file)
            with open(output_file) as f:
                data = json.load(f)

            assert data['mode'] == 'debug'
            # At least first compilation
            assert data['summary']['cache_misses'] >= 1
            # At least second compilation
            assert data['summary']['cache_hits'] >= 1
            assert len(data['records']) >= 2
