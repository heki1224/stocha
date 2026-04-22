"""
stocha Tutorial 3: Quasi-Random Number Generation
==================================================

Demonstrates Sobol and Halton low-discrepancy sequences and their
advantages over pseudo-random Monte Carlo for numerical integration.

Financial background:
    - Quasi-Monte Carlo (QMC) achieves O(1/N) convergence vs O(1/sqrt(N))
      for standard MC, reducing sample requirements by orders of magnitude
      for smooth, low-to-moderate dimensional integrands.
    - Sobol sequences (Joe & Kuo 2008) are the standard choice for
      financial simulation (option pricing, CVA, scenario generation).
    - Halton sequences use consecutive prime bases; simpler but slightly
      inferior to Sobol for high dimensions.
"""

import math
import time

import numpy as np
import stocha

# --- 1. Sobol Sequence ---
print("=" * 55)
print("1. Sobol Low-Discrepancy Sequence")
print("=" * 55)

pts = stocha.sobol(dim=2, n_samples=1024)
print(f"Shape: {pts.shape}")
print(f"Range: [{pts.min():.6f}, {pts.max():.6f}]  (expected [0, 1))")
print(f"Mean per dim: {pts.mean(axis=0)}")  # should be ~0.5 each
print(f"Std  per dim: {pts.std(axis=0)}")   # should be ~0.289 (Uniform[0,1])

# --- 2. Halton Sequence ---
print("\n" + "=" * 55)
print("2. Halton Low-Discrepancy Sequence")
print("=" * 55)

pts_h = stocha.halton(dim=2, n_samples=1024)
print(f"Shape: {pts_h.shape}")
print(f"Range: [{pts_h.min():.6f}, {pts_h.max():.6f}]  (expected (0, 1))")
print(f"Mean per dim: {pts_h.mean(axis=0)}")
print(f"Std  per dim: {pts_h.std(axis=0)}")

# skip parameter: useful for randomized QMC or avoiding burn-in
pts_h_skip = stocha.halton(dim=2, n_samples=100, skip=1000)
print(f"Halton with skip=1000, shape: {pts_h_skip.shape}")

# --- 3. Convergence Comparison: Estimating Pi ---
print("\n" + "=" * 55)
print("3. Convergence Comparison: Estimating Pi")
print("   (fraction of points inside unit circle * 4)")
print("=" * 55)

rng = stocha.RNG(seed=42)

print(f"\n{'N':>10}  {'MC error':>12}  {'Sobol error':>12}  {'Halton error':>12}")
print("-" * 52)

for n in [64, 256, 1024, 4096, 16384, 65536]:
    # Standard pseudo-random MC
    u_mc = rng.uniform(size=n * 2).reshape(n, 2)
    pi_mc = 4.0 * np.sum(u_mc[:, 0]**2 + u_mc[:, 1]**2 <= 1.0) / n
    err_mc = abs(pi_mc - math.pi)

    # Sobol QMC
    u_sb = stocha.sobol(dim=2, n_samples=n)
    pi_sb = 4.0 * np.sum(u_sb[:, 0]**2 + u_sb[:, 1]**2 <= 1.0) / n
    err_sb = abs(pi_sb - math.pi)

    # Halton QMC
    u_ht = stocha.halton(dim=2, n_samples=n)
    pi_ht = 4.0 * np.sum(u_ht[:, 0]**2 + u_ht[:, 1]**2 <= 1.0) / n
    err_ht = abs(pi_ht - math.pi)

    print(f"{n:>10,}  {err_mc:>12.6f}  {err_sb:>12.6f}  {err_ht:>12.6f}")

print(f"\nTrue pi = {math.pi:.6f}")
print("QMC sequences typically converge faster than pseudo-random MC.")

# --- 4. High-Dimensional Sobol ---
print("\n" + "=" * 55)
print("4. High-Dimensional Sobol (up to 1000 dims)")
print("=" * 55)

pts_hd = stocha.sobol(dim=100, n_samples=2048)
print(f"100-dim Sobol shape: {pts_hd.shape}")
print(f"Per-dimension mean range: [{pts_hd.mean(axis=0).min():.4f}, "
      f"{pts_hd.mean(axis=0).max():.4f}]  (all should be ~0.5)")

# --- 5. Throughput Benchmark ---
print("\n" + "=" * 55)
print("5. Throughput Benchmark")
print("=" * 55)

print("\nSobol:")
for dim, n in [(2, 65536), (10, 65536), (50, 16384), (100, 8192)]:
    t0 = time.time()
    _ = stocha.sobol(dim=dim, n_samples=n)
    elapsed = time.time() - t0
    total = dim * n
    print(f"  dim={dim:>4}, n={n:>6,}: {elapsed*1000:6.1f}ms  "
          f"({total/elapsed/1e6:.1f}M values/s)")

print("\nHalton:")
for dim, n in [(2, 65536), (10, 65536), (30, 16384), (40, 8192)]:
    t0 = time.time()
    _ = stocha.halton(dim=dim, n_samples=n)
    elapsed = time.time() - t0
    total = dim * n
    print(f"  dim={dim:>4}, n={n:>6,}: {elapsed*1000:6.1f}ms  "
          f"({total/elapsed/1e6:.1f}M values/s)")

print("\n✅ Quasi-random generation complete")
