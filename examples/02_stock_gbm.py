"""
stocha Tutorial 2: Stock Price Simulation with GBM
===================================================

Demonstrates Geometric Brownian Motion (GBM) path generation,
European call option pricing via Monte Carlo, and antithetic variates.

Financial background:
    - Black-Scholes model
    - Risk-neutral pricing
    - Monte Carlo expectation estimation
"""

import math
import time

import numpy as np
import stocha

# --- 1. GBM Path Generation ---
print("=" * 55)
print("1. GBM Path Generation")
print("=" * 55)

S0 = 100.0    # Initial stock price
mu = 0.05     # Drift (5% p.a.)
sigma = 0.20  # Volatility (20% p.a.)
T = 1.0       # 1 year
steps = 252   # Daily steps
n_paths = 10_000

t0 = time.time()
paths = stocha.gbm(s0=S0, mu=mu, sigma=sigma, t=T, steps=steps, n_paths=n_paths, seed=42)
elapsed = time.time() - t0

print(f"n_paths={n_paths:,}, steps={steps}")
print(f"Output shape: {paths.shape}")
print(f"Generation time: {elapsed*1000:.1f}ms")
print(f"Initial price (mean): {paths[:, 0].mean():.4f}")
print(f"Terminal price: mean={paths[:, -1].mean():.4f}, std={paths[:, -1].std():.4f}")

# --- 2. Comparison with Theoretical Mean ---
print("\n" + "=" * 55)
print("2. Theoretical Check: E[S(T)] = S0 * exp(mu * T)")
print("=" * 55)

expected_mean = S0 * math.exp(mu * T)
simulated_mean = paths[:, -1].mean()
rel_err = abs(simulated_mean - expected_mean) / expected_mean

print(f"Theoretical E[S(T)] = {expected_mean:.4f}")
print(f"Simulated mean      = {simulated_mean:.4f}")
print(f"Relative error      = {rel_err:.4f} ({rel_err*100:.2f}%)")

# --- 3. European Call Option Pricing ---
print("\n" + "=" * 55)
print("3. European Call Option Pricing (Monte Carlo)")
print("=" * 55)

K = 105.0   # Strike price
r = 0.02    # Risk-free rate

# Paths under risk-neutral measure (drift = r)
paths_rn = stocha.gbm(s0=S0, mu=r, sigma=sigma, t=T, steps=steps, n_paths=100_000, seed=0)

payoffs = np.maximum(paths_rn[:, -1] - K, 0.0)
mc_price = math.exp(-r * T) * payoffs.mean()
mc_stderr = payoffs.std() / math.sqrt(len(payoffs))

# Black-Scholes analytical price
d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

bs_price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

print(f"Strike K={K}, Risk-free rate r={r}")
print(f"Monte Carlo: {mc_price:.4f} ± {mc_stderr:.4f}  (95% CI: ±{1.96*mc_stderr:.4f})")
print(f"Black-Scholes: {bs_price:.4f}")
print(f"Difference: {abs(mc_price - bs_price):.4f}")

# --- 4. Antithetic Variates ---
print("\n" + "=" * 55)
print("4. Antithetic Variates for Variance Reduction")
print("=" * 55)

n_compare = 10_000

paths_std = stocha.gbm(s0=S0, mu=r, sigma=sigma, t=T, steps=steps, n_paths=n_compare, seed=0)
payoffs_std = np.maximum(paths_std[:, -1] - K, 0.0)
price_std = math.exp(-r * T) * payoffs_std.mean()
stderr_std = payoffs_std.std() / math.sqrt(n_compare)

paths_anti = stocha.gbm(
    s0=S0, mu=r, sigma=sigma, t=T, steps=steps,
    n_paths=n_compare, seed=0, antithetic=True
)
payoffs_anti = np.maximum(paths_anti[:, -1] - K, 0.0)
price_anti = math.exp(-r * T) * payoffs_anti.mean()
stderr_anti = payoffs_anti.std() / math.sqrt(n_compare)

print(f"Standard MC:       price={price_std:.4f}, stderr={stderr_std:.4f}")
print(f"Antithetic MC:     price={price_anti:.4f}, stderr={stderr_anti:.4f}")
print(f"Black-Scholes:     {bs_price:.4f}")

# --- 5. Throughput Benchmark ---
print("\n" + "=" * 55)
print("5. Throughput Benchmark")
print("=" * 55)

for n in [1_000, 10_000, 100_000]:
    t0 = time.time()
    _ = stocha.gbm(s0=S0, mu=mu, sigma=sigma, t=T, steps=252, n_paths=n, seed=42)
    elapsed = time.time() - t0
    rate = n / elapsed
    print(f"n_paths={n:>8,}: {elapsed*1000:6.1f}ms  ({rate:>10,.0f} paths/s)")

print("\n✅ GBM simulation complete")
