"""
stocha Tutorial 5: Risk Measures and Copulas
=============================================

Demonstrates Value-at-Risk (VaR), Conditional VaR (CVaR / Expected
Shortfall), and multivariate dependence modeling via copulas.

Financial background:
    VaR and CVaR:
        VaR(alpha) = quantile loss at confidence level alpha.
        CVaR(alpha) = E[loss | loss > VaR(alpha)] — average loss in the
        worst (1-alpha) fraction. CVaR is coherent (convex, sub-additive)
        and is the preferred risk measure under Basel III / FRTB.

    Copulas:
        A copula decouples marginal distributions from joint dependence.
        The Gaussian copula models linear correlation but has NO tail
        dependence — joint extreme losses are no more likely than independent.
        The Student-t copula adds tail dependence (lambda_L = lambda_U > 0),
        capturing the empirical finding that assets crash together.
        The 2008 crisis exposed the danger of using Gaussian copulas for
        CDO tranches (the "formula that killed Wall Street").
"""

import math
import time

import numpy as np
import stocha

# --- 1. Portfolio Return Simulation ---
print("=" * 60)
print("1. Portfolio Return Simulation (2-asset GBM)")
print("=" * 60)

S0     = 100.0
r      = 0.05
sigma1 = 0.20   # Asset 1 volatility
sigma2 = 0.30   # Asset 2 volatility
T      = 1.0 / 252  # 1 trading day horizon
steps  = 1
n_paths = 100_000

paths1 = stocha.gbm(s0=S0, mu=r, sigma=sigma1, t=T, steps=steps, n_paths=n_paths, seed=1)
paths2 = stocha.gbm(s0=S0, mu=r, sigma=sigma2, t=T, steps=steps, n_paths=n_paths, seed=2)

ret1 = paths1[:, -1] / paths1[:, 0] - 1.0
ret2 = paths2[:, -1] / paths2[:, 0] - 1.0

# Equal-weight portfolio
port_returns = 0.5 * ret1 + 0.5 * ret2

print(f"n_paths={n_paths:,}, horizon=1 trading day")
print(f"Asset 1: mean={ret1.mean():.6f}, std={ret1.std():.4f}")
print(f"Asset 2: mean={ret2.mean():.6f}, std={ret2.std():.4f}")
print(f"Portfolio: mean={port_returns.mean():.6f}, std={port_returns.std():.4f}")

# --- 2. Value-at-Risk and CVaR ---
print("\n" + "=" * 60)
print("2. Value-at-Risk (VaR) and Conditional VaR (CVaR)")
print("   Losses are positive; both measures are loss magnitudes.")
print("=" * 60)

print(f"\n{'Confidence':>12}  {'VaR':>10}  {'CVaR':>10}  {'VaR/CVaR ratio':>14}")
print("-" * 54)
for conf in [0.90, 0.95, 0.99, 0.999]:
    var, cvar = stocha.var_cvar(port_returns, confidence=conf)
    print(f"{conf:>12.1%}  {var:>10.6f}  {cvar:>10.6f}  {var/cvar:>14.4f}")

# Theoretical normal VaR for comparison
port_std = port_returns.std()

def norm_ppf(p: float) -> float:
    """Inverse normal CDF via rational approximation (Beasley-Springer-Moro)."""
    if p < 0.5:
        return -norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - (c[0] + c[1]*t + c[2]*t**2) / (1 + d[0]*t + d[1]*t**2 + d[2]*t**3)

print(f"\nNormal approximation (std={port_std:.4f}):")
for conf in [0.95, 0.99]:
    var_norm = -port_returns.mean() + port_std * norm_ppf(conf)
    var_sim, cvar_sim = stocha.var_cvar(port_returns, confidence=conf)
    print(f"  {conf:.0%}: Normal VaR={var_norm:.6f}, Simulated VaR={var_sim:.6f}")

# --- 3. Gaussian Copula ---
print("\n" + "=" * 60)
print("3. Gaussian Copula Sampling")
print("=" * 60)

corr = np.array([[1.0, 0.8],
                 [0.8, 1.0]])
n_cop = 100_000

t0 = time.time()
u_gauss = stocha.gaussian_copula(corr, n_samples=n_cop, seed=42)
elapsed = time.time() - t0

print(f"Correlation matrix rho=0.8, n_samples={n_cop:,}")
print(f"Generation time: {elapsed*1000:.1f}ms")
print(f"Output shape: {u_gauss.shape}, range=[{u_gauss.min():.4f}, {u_gauss.max():.4f}]")

# Spearman rank correlation (should be close to sin(pi/2 * rho))
from_rank = np.corrcoef(u_gauss[:, 0].argsort().argsort(),
                        u_gauss[:, 1].argsort().argsort())[0, 1]
spearman_theory = (6 / math.pi) * math.asin(0.8 / 2)
print(f"Spearman rank corr: simulated={from_rank:.4f}, "
      f"theory≈{spearman_theory:.4f}")

# --- 4. Student-t Copula ---
print("\n" + "=" * 60)
print("4. Student-t Copula Sampling (heavy tails, tail dependence)")
print("=" * 60)

nu = 4.0  # Degrees of freedom (lower = heavier tails)

t0 = time.time()
u_t = stocha.student_t_copula(corr, nu=nu, n_samples=n_cop, seed=42)
elapsed = time.time() - t0

print(f"Correlation matrix rho=0.8, nu={nu}, n_samples={n_cop:,}")
print(f"Generation time: {elapsed*1000:.1f}ms")
print(f"Output shape: {u_t.shape}, range=[{u_t.min():.4f}, {u_t.max():.4f}]")

# --- 5. Tail Dependence: Gaussian vs Student-t ---
print("\n" + "=" * 60)
print("5. Tail Dependence Comparison: Gaussian vs Student-t")
print("   (joint probability that both variables exceed the 99th percentile)")
print("=" * 60)

for threshold in [0.95, 0.99, 0.999]:
    # Joint exceedance: both u1 > threshold AND u2 > threshold
    joint_gauss = np.mean((u_gauss[:, 0] > threshold) & (u_gauss[:, 1] > threshold))
    joint_t     = np.mean((u_t[:, 0] > threshold)     & (u_t[:, 1] > threshold))
    # Under independence: (1-threshold)^2
    independent  = (1 - threshold)**2
    print(f"  P(U1>{threshold:.1%}, U2>{threshold:.1%}): "
          f"Gaussian={joint_gauss:.6f}, "
          f"Student-t={joint_t:.6f}, "
          f"Independent={independent:.6f}")

print("\n(Student-t copula shows more joint extremes than Gaussian — tail dependence)")

# --- 6. Throughput Benchmark ---
print("\n" + "=" * 60)
print("6. Throughput Benchmark")
print("=" * 60)

for name, fn in [
    ("Gaussian copula (2D)", lambda n: stocha.gaussian_copula(corr, n_samples=n, seed=0)),
    ("Student-t copula (2D)", lambda n: stocha.student_t_copula(corr, nu=nu, n_samples=n, seed=0)),
]:
    print(f"\n{name}:")
    for n in [10_000, 100_000, 500_000]:
        t0 = time.time()
        _ = fn(n)
        elapsed = time.time() - t0
        print(f"  n={n:>8,}: {elapsed*1000:6.1f}ms  ({n/elapsed/1e6:.2f}M samples/s)")

print("\n✅ Risk measures and copulas complete")
