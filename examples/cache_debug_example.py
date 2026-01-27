#!/usr/bin/env python
"""
Example demonstrating the CUPY_CACHE_DEBUG feature.

This script shows how to use the cache debug feature to diagnose
cache hit/miss behavior in CuPy applications.
"""

import os
import json
import tempfile

# This example would work if CuPy is fully built and available
# For now, it serves as documentation

print("=" * 70)
print("CuPy Cache Debug Feature - Example Usage")
print("=" * 70)

print("""
The CUPY_CACHE_DEBUG feature helps diagnose compilation cache behavior.

USAGE:
------

1. Stats Mode (lightweight - only hit/miss counts):
   
   $ export CUPY_CACHE_DEBUG=stats:/path/to/stats.json
   $ python your_cupy_script.py
   
   Output (stats.json):
   {
     "mode": "stats",
     "cache_hits": 150,
     "cache_misses": 50,
     "total_lookups": 200,
     "hit_ratio": 0.75
   }

2. Debug Mode (detailed - includes cache keys):
   
   $ export CUPY_CACHE_DEBUG=debug:/path/to/debug.json
   $ python your_cupy_script.py
   
   Output (debug.json):
   {
     "mode": "debug",
     "summary": {
       "cache_hits": 150,
       "cache_misses": 50,
       "total_lookups": 200,
       "hit_ratio": 0.75
     },
     "records": [
       {
         "type": "miss",
         "hashed_key": "abc123def456...",
         "cache_key": "((arch, options, ...), base, source, ...)"
       },
       {
         "type": "hit",
         "hashed_key": "def456ghi789...",
         "cache_key": "((arch, options, ...), base, source, ...)"
       },
       ...
     ]
   }

3. Default Path (if you don't specify a path):
   
   $ export CUPY_CACHE_DEBUG=stats
   $ python your_cupy_script.py
   # Creates cupy_cache_debug.json in current directory

USE CASES:
----------

1. CI/CD Performance Diagnosis:
   - Check if cache is working across CI runs
   - Identify unexpected cache misses
   - Measure cache hit ratio improvements

2. Development:
   - Verify cache key generation is working
   - Debug why certain kernels aren't cached
   - Understand cache behavior changes after code modifications

3. Production Monitoring:
   - Use stats mode to track cache efficiency
   - Minimal overhead suitable for production use

ZERO OVERHEAD:
--------------
When CUPY_CACHE_DEBUG is not set, this feature has zero overhead.
The tracking code only executes when explicitly enabled.

""")

# Example of what the output would look like
print("\nEXAMPLE OUTPUT:")
print("-" * 70)

example_stats = {
    "mode": "stats",
    "cache_hits": 42,
    "cache_misses": 8,
    "total_lookups": 50,
    "hit_ratio": 0.84
}

print("\nStats mode output:")
print(json.dumps(example_stats, indent=2))

example_debug = {
    "mode": "debug",
    "summary": {
        "cache_hits": 3,
        "cache_misses": 2,
        "total_lookups": 5,
        "hit_ratio": 0.6
    },
    "records": [
        {
            "type": "miss",
            "hashed_key": "a1b2c3d4e5f6.cubin",
            "cache_key": "((70, ('-ftz=true',), ...), ..., '__global__ void kernel1() {}', ...)"
        },
        {
            "type": "hit",
            "hashed_key": "b2c3d4e5f6a1.cubin",
            "cache_key": "((70, ('-ftz=true',), ...), ..., '__global__ void kernel2() {}', ...)"
        },
        {
            "type": "miss",
            "hashed_key": "c3d4e5f6a1b2.cubin",
            "cache_key": "((70, ('-ftz=true',), ...), ..., '__global__ void kernel3() {}', ...)"
        }
    ]
}

print("\nDebug mode output (truncated):")
print(json.dumps(example_debug, indent=2))

print("\n" + "=" * 70)
