"""
stocha Tutorial 6: Interest Rate Models and Volatility Smile
============================================================

Demonstrates the Hull-White 1-factor short-rate model and the SABR
implied volatility model used in fixed-income and swaption markets.

Financial background:
    Hull-White (1990):
        dr(t) = (theta - a*r(t))*dt + sigma*dW(t)
        A mean-reverting Gaussian short-rate model. Long-run mean = theta/a.
        stocha uses exact simulation (zero discretization bias) via the
        Gaussian conditional distribution of r(t+dt) | r(t).

    SABR (Hagan et al. 2002):
        dF = alpha * F^beta * dW
        dalpha = nu * alpha * dZ
        corr(dW, dZ) = rho
        The SABR model produces an analytic approximation for the Black
        implied volatility across strikes, capturing the volatility smile
        (or skew) observed in swaption and cap/floor markets.
        Shifted SABR (shift > 0) handles negative forward rates.
"""

import math
import time

import numpy as np
import stocha

# ──────────────────────────────────────────────
# Hull-White parameters
# ──────────────────────────────────────────────
r0    = 0.05   # Initial short rate (5%)
a     = 0.10   # Mean-reversion speed
lr_mean = 0.05 # Target long-run mean rate (5%)
theta = a * lr_mean  # Hull-White theta = a * long_run_mean = 0.005
sigma_hw = 0.01  # Rate volatility (1%)
T     = 10.0   # 10-year simulation
steps = 120    # Monthly steps (10y * 12)
n_paths = 50_000

# --- 1. Hull-White Path Generation ---
print("=" * 60)
print("1. Hull-White Short-Rate Model — Path Generation")
print("=" * 60)

t0 = time.time()
rates = stocha.hull_white(
    r0=r0, a=a, theta=theta, sigma=sigma_hw,
    t=T, steps=steps, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"n_paths={n_paths:,}, steps={steps} (monthly over {T:.0f}y)")
print(f"Output shape: {rates.shape}")
print(f"Generation time: {elapsed*1000:.1f}ms")
print(f"Initial rate (col 0): mean={rates[:, 0].mean():.4f}  (= r0={r0})")
print(f"Terminal rate:        mean={rates[:, -1].mean():.4f}")

# --- 2. Theoretical vs Simulated Mean Reversion ---
print("\n" + "=" * 60)
print("2. Theoretical vs Simulated E[r(t)]")
print("   E[r(t)] = theta/a + (r0 - theta/a) * exp(-a*t)")
print("=" * 60)

times = np.linspace(0, T, steps + 1)
print(f"\n{'t (years)':>10}  {'Theory E[r]':>14}  {'Simulated mean':>14}  {'Error':>10}")
print("-" * 54)
for i in [0, 12, 24, 60, 120]:  # 0, 1, 2, 5, 10 years
    t_val = times[i]
    theory = lr_mean + (r0 - lr_mean) * math.exp(-a * t_val)
    simulated = float(rates[:, i].mean())
    err = abs(simulated - theory)
    print(f"{t_val:>10.1f}  {theory:>14.6f}  {simulated:>14.6f}  {err:>10.6f}")

# Confirm mean-reversion: spread narrows toward long-run mean
print(f"\nLong-run mean (theta/a) = {lr_mean:.4f}")
print(f"Rate dispersion at t=0:  std={rates[:, 0].std():.4f}")
print(f"Rate dispersion at t=5y: std={rates[:, 60].std():.4f}")
print(f"Rate dispersion at t=10y:std={rates[:, 120].std():.4f}")

# --- 3. SABR Volatility Smile ---
print("\n" + "=" * 60)
print("3. SABR Volatility Smile")
print("   Black implied vol across strikes for a 5Y swaption")
print("=" * 60)

F     = 0.05   # ATM forward swap rate
T_sab = 5.0    # Swaption expiry
alpha = 0.20   # Initial vol (SABR)
beta  = 0.5    # CEV exponent (0.5 = square-root process, common in rates)
rho   = -0.30  # Negative skew (rates often have negative vol-rate correlation)
nu    = 0.40   # Vol-of-vol

strikes = [0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.09, 0.10]

print(f"\nForward F={F:.0%}, T={T_sab}y, alpha={alpha}, beta={beta}, "
      f"rho={rho}, nu={nu}")
print(f"\n{'Strike K':>10}  {'Moneyness':>10}  {'Impl. Vol (SABR)':>18}")
print("-" * 44)

for k in strikes:
    iv = stocha.sabr_implied_vol(f=F, k=k, t=T_sab,
                                  alpha=alpha, beta=beta, rho=rho, nu=nu)
    moneyness = k / F
    marker = " ← ATM" if abs(k - F) < 1e-10 else ""
    print(f"{k:>10.2%}  {moneyness:>10.2f}x  {iv:>18.4%}{marker}")

# --- 4. Shifted SABR (Negative Rates) ---
print("\n" + "=" * 60)
print("4. Shifted SABR — Negative Rate Support")
print("   shift=2% allows strikes down to -2%")
print("=" * 60)

shift = 0.02  # 2% shift: all rates shifted up by shift
F_neg = 0.00  # Forward at 0% (near-zero rate environment)

strikes_neg = [-0.015, -0.01, -0.005, 0.00, 0.005, 0.01, 0.02, 0.03]

print(f"\nForward F={F_neg:.1%} (near zero), shift={shift:.0%}")
print(f"\n{'Strike K':>10}  {'Impl. Vol (Shifted SABR)':>26}")
print("-" * 40)

for k in strikes_neg:
    if k + shift <= 0:
        print(f"{k:>10.2%}  {'below shift bound':>26}")
        continue
    iv = stocha.sabr_implied_vol(f=F_neg, k=k, t=T_sab,
                                  alpha=alpha, beta=beta, rho=rho, nu=nu,
                                  shift=shift)
    print(f"{k:>10.2%}  {iv:>26.4%}")

# --- 5. Throughput Benchmark ---
print("\n" + "=" * 60)
print("5. Throughput Benchmark")
print("=" * 60)

print("\nHull-White:")
for n in [1_000, 10_000, 50_000]:
    t0 = time.time()
    _ = stocha.hull_white(r0=r0, a=a, theta=theta, sigma=sigma_hw,
                          t=T, steps=steps, n_paths=n, seed=0)
    elapsed = time.time() - t0
    print(f"  n_paths={n:>7,}: {elapsed*1000:6.1f}ms  ({n/elapsed:>10,.0f} paths/s)")

print("\nSABR (single strike, scalar output):")
n_eval = 10_000
t0 = time.time()
for _ in range(n_eval):
    stocha.sabr_implied_vol(f=F, k=F, t=T_sab, alpha=alpha, beta=beta, rho=rho, nu=nu)
elapsed = time.time() - t0
print(f"  {n_eval:,} evaluations: {elapsed*1000:.1f}ms  ({n_eval/elapsed:,.0f} evals/s)")

print("\n✅ Interest rate models complete")
