"""
stocha Tutorial 1: Random Number Generation
============================================

Demonstrates basic RNG usage, reproducibility, and performance benchmarks.
"""

import time

import numpy as np
import stocha

# --- 1. Basic RNG Usage ---
print("=" * 50)
print("1. Basic RNG Usage")
print("=" * 50)

rng = stocha.RNG(seed=42)
print(f"RNG: {rng}")

u = rng.uniform(size=10)
print(f"uniform(10): min={u.min():.4f}, max={u.max():.4f}")

z = rng.standard_normal(size=100_000)
print(f"standard_normal(100k): mean={z.mean():.4f}, std={z.std():.4f}")

x = rng.normal(size=100_000, loc=5.0, scale=2.0)
print(f"normal(100k, loc=5, scale=2): mean={x.mean():.4f}, std={x.std():.4f}")

# --- 2. Reproducibility ---
print("\n" + "=" * 50)
print("2. Reproducibility (same seed → same output)")
print("=" * 50)

rng1 = stocha.RNG(seed=123)
rng2 = stocha.RNG(seed=123)
s1 = rng1.normal(size=5)
s2 = rng2.normal(size=5)
print(f"rng1: {s1}")
print(f"rng2: {s2}")
print(f"Identical: {np.allclose(s1, s2)}")

# --- 3. State Serialization ---
print("\n" + "=" * 50)
print("3. State Serialization (checkpointing)")
print("=" * 50)

rng = stocha.RNG(seed=999)
state_json = rng.save_state()
print(f"Saved state: {state_json}")

# --- 4. Algorithm Selection ---
print("\n" + "=" * 50)
print("4. Algorithm Selection")
print("=" * 50)

rng_pcg = stocha.RNG(seed=42, algorithm="pcg64dxsm")
print(f"PCG64DXSM (default): {rng_pcg}")

# --- 5. Performance Benchmark ---
print("\n" + "=" * 50)
print("5. Performance Benchmark")
print("=" * 50)

rng = stocha.RNG(seed=0)
for n in [10_000, 100_000, 1_000_000]:
    t0 = time.time()
    _ = rng.standard_normal(size=n)
    elapsed = time.time() - t0
    rate = n / elapsed / 1e6
    print(f"standard_normal({n:>10,}): {elapsed*1000:.1f}ms  ({rate:.1f}M samples/s)")

print("\n✅ All tests passed")
