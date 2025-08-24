#!/usr/bin/env python3
"""Quick GPU functionality test"""

try:
    import cupy as cp
    print("✅ GPU Support Available")
    print(f"   Device: {cp.cuda.runtime.getDeviceName(0)}")
    print(f"   Memory: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")
    
    # Quick GPU test
    x = cp.array([1, 2, 3])
    y = x * 2
    print(f"   Test: {x} * 2 = {y}")
    print("✅ GPU operations working")
    
except ImportError:
    print("⚠️  CuPy not installed - CPU only mode")
    print("   Run: pip install cupy-cuda11x")
except Exception as e:
    print(f"❌ GPU error: {e}")