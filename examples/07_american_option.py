"""
stocha Tutorial 7: American Option Pricing via LSMC
====================================================

Demonstrates Longstaff-Schwartz Monte Carlo (LSMC) for American option pricing.

Financial background:
    European options can only be exercised at expiry. American options allow
    early exercise at any time — making them more valuable (early exercise
    premium) but analytically intractable for most models.

    Longstaff-Schwartz (2001) LSMC algorithm:
        1. Simulate N risk-neutral GBM paths.
        2. At each time step (backward induction), regress the discounted
           future cash flows against basis functions of the current price
           (polynomial in S) to estimate the continuation value.
        3. Exercise early if the intrinsic value > continuation value estimate.
        4. Price = discounted average of optimal exercise cash flows.

    stocha uses polynomial basis (degree 1–4) with QR decomposition
    (faer library) for the least-squares regression step.

    Early Exercise Premium = American price − European (Black-Scholes) price.
    For puts: early exercise becomes optimal deep in-the-money, as the
    time value of money on the strike outweighs the insurance value of waiting.
"""

import math
import time

import numpy as np
import stocha

# Option parameters
S0    = 100.0   # Initial stock price
K     = 100.0   # Strike (ATM)
r     = 0.05    # Risk-free rate
sigma = 0.20    # Volatility
T     = 1.0     # 1 year
steps = 50      # Exercise opportunities

# --- 1. LSMC American Put Pricing ---
print("=" * 60)
print("1. LSMC American Put Option Pricing")
print("=" * 60)

n_paths = 100_000
t0 = time.time()
price, stderr = stocha.lsmc_american_option(
    s0=S0, k=K, r=r, sigma=sigma, t=T,
    steps=steps, n_paths=n_paths, is_put=True, poly_degree=3, seed=42,
)
elapsed = time.time() - t0

print(f"S0={S0}, K={K}, r={r:.0%}, sigma={sigma:.0%}, T={T}y")
print(f"n_paths={n_paths:,}, steps={steps}, poly_degree=3")
print(f"American put: {price:.4f} ± {stderr:.4f}  (95% CI: ±{1.96*stderr:.4f})")
print(f"Pricing time: {elapsed*1000:.1f}ms")

# --- 2. European Black-Scholes Reference ---
print("\n" + "=" * 60)
print("2. European Put (Black-Scholes) vs American Put")
print("=" * 60)

d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

bs_put = K * math.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)

early_exercise_premium = price - bs_put

print(f"European put (Black-Scholes): {bs_put:.4f}")
print(f"American put (LSMC):          {price:.4f} ± {stderr:.4f}")
print(f"Early Exercise Premium:       {early_exercise_premium:.4f}  "
      f"({early_exercise_premium/bs_put*100:.2f}% of European price)")
print("\nAmerican put >= European put (by no-arbitrage); positive premium confirms correctness.")

# --- 3. American Call vs European Call ---
print("\n" + "=" * 60)
print("3. American Call (no dividends — early exercise not optimal)")
print("=" * 60)

price_call, stderr_call = stocha.lsmc_american_option(
    s0=S0, k=K, r=r, sigma=sigma, t=T,
    steps=steps, n_paths=n_paths, is_put=False, poly_degree=3, seed=42,
)

bs_call = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
eep_call = price_call - bs_call

print(f"European call (Black-Scholes): {bs_call:.4f}")
print(f"American call (LSMC):          {price_call:.4f} ± {stderr_call:.4f}")
print(f"Early Exercise Premium:        {eep_call:.4f}  (should be ~0)")
print("(No dividends: early exercise of calls is never optimal — Merton 1973)")

# --- 4. Polynomial Degree Sensitivity ---
print("\n" + "=" * 60)
print("4. Polynomial Degree Sensitivity")
print("   (higher degree = richer basis for continuation value)")
print("=" * 60)

print(f"\n{'Degree':>8}  {'Price':>10}  {'Std Err':>10}  {'vs BS':>10}")
print("-" * 44)
for deg in [1, 2, 3, 4]:
    p, se = stocha.lsmc_american_option(
        s0=S0, k=K, r=r, sigma=sigma, t=T,
        steps=steps, n_paths=50_000, is_put=True,
        poly_degree=deg, seed=42,
    )
    print(f"{deg:>8}  {p:>10.4f}  {se:>10.4f}  {p-bs_put:>+10.4f}")
print(f"{'(BS ref)':>8}  {bs_put:>10.4f}")

# --- 5. Moneyness Sensitivity ---
print("\n" + "=" * 60)
print("5. American Put Price vs Moneyness")
print("   (S/K ratio; deep ITM → larger early exercise premium)")
print("=" * 60)

print(f"\n{'S0':>8}  {'S/K':>6}  {'Eur BS':>10}  {'LSMC':>10}  {'EEP':>10}")
print("-" * 50)
for s0_test in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
    d1_t = (math.log(s0_test/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2_t = d1_t - sigma * math.sqrt(T)
    bs_t = K * math.exp(-r*T) * norm_cdf(-d2_t) - s0_test * norm_cdf(-d1_t)

    p_t, _ = stocha.lsmc_american_option(
        s0=s0_test, k=K, r=r, sigma=sigma, t=T,
        steps=steps, n_paths=50_000, is_put=True, poly_degree=3, seed=42,
    )
    eep_t = p_t - bs_t
    print(f"{s0_test:>8.0f}  {s0_test/K:>6.2f}  {bs_t:>10.4f}  {p_t:>10.4f}  {eep_t:>+10.4f}")

print("\n(Deep ITM puts show the largest EEP — interest on strike makes early exercise valuable)")

# --- 6. Throughput Benchmark ---
print("\n" + "=" * 60)
print("6. Throughput Benchmark")
print("=" * 60)

for n in [10_000, 50_000, 100_000]:
    t0 = time.time()
    _ = stocha.lsmc_american_option(
        s0=S0, k=K, r=r, sigma=sigma, t=T,
        steps=steps, n_paths=n, is_put=True, poly_degree=3, seed=42,
    )
    elapsed = time.time() - t0
    print(f"  n_paths={n:>8,}: {elapsed*1000:6.1f}ms  ({n/elapsed:>10,.0f} paths/s)")

print("\n✅ American option pricing (LSMC) complete")
