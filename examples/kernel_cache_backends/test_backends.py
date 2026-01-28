"""
Simple test script for cache backends that doesn't require CuPy installation.

This script tests the cache backend interface independently of CuPy's
compilation system.
"""

import os
import sys
import tempfile

# Add the parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class MockCacheBackend:
    """Mock CacheBackend for testing without CuPy."""
    
    def load(self, name):
        raise NotImplementedError
    
    def save(self, name, data):
        raise NotImplementedError
    
    def exists(self, name):
        raise NotImplementedError


def test_gcp_backend_without_credentials():
    """Test that GCP backend gracefully handles missing credentials."""
    print("Testing GCP backend without credentials...")
    
    try:
        from examples.kernel_cache_backends.gcp_storage_backend import (
            GCPStorageCacheBackend
        )
    except ImportError as e:
        print(f"  Failed to import GCP backend: {e}")
        return False
    
    import hashlib
    
    # Create backend with a non-existent bucket (should fall back to local cache)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            backend = GCPStorageCacheBackend(
                bucket_name='test-bucket-does-not-exist',
                local_cache_dir=tmpdir
            )
            
            # Test basic operations (should use local cache)
            test_name = 'test.cubin'
            cubin_data = b'compiled_kernel_binary_data'
            
            # Create proper test data with valid hash
            cubin_hash = hashlib.sha1(cubin_data).hexdigest().encode('ascii')
            test_data = cubin_hash + cubin_data
            
            # Save
            backend.save(test_name, test_data)
            print("  ✓ Save operation completed")
            
            # Check existence
            if backend.exists(test_name):
                print("  ✓ Exists check passed")
            else:
                print("  ✗ Exists check failed")
                return False
            
            # Load
            loaded = backend.load(test_name)
            if loaded == test_data:
                print("  ✓ Load operation passed")
            else:
                print("  ✗ Load operation failed")
                print(f"    Expected: {test_data[:50]}...")
                print(f"    Got: {loaded[:50] if loaded else None}...")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_mock_backend_interface():
    """Test that the backend interface is properly defined."""
    print("Testing backend interface...")
    
    # Create a concrete implementation for testing
    class TestBackend(MockCacheBackend):
        def load(self, name):
            return None
        
        def save(self, name, data):
            pass
        
        def exists(self, name):
            return False
    
    backend = TestBackend()
    
    # Test that methods work correctly
    result = backend.load('test')
    if result is None:
        print("  ✓ load() returns None")
    else:
        print("  ✗ load() should return None")
        return False
    
    try:
        backend.save('test', b'data')
        print("  ✓ save() completes without error")
    except Exception as e:
        print(f"  ✗ save() raised exception: {e}")
        return False
    
    if not backend.exists('test'):
        print("  ✓ exists() returns False")
    else:
        print("  ✗ exists() should return False")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Cache Backend Interface Tests")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Interface
    print("Test 1: Backend Interface")
    results.append(("Interface", test_mock_backend_interface()))
    print()
    
    # Test 2: GCP Backend
    print("Test 2: GCP Backend Fallback")
    results.append(("GCP Backend", test_gcp_backend_without_credentials()))
    print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
