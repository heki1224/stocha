"""
stocha Tutorial 4: Stochastic Volatility and Jump-Diffusion Models
==================================================================

Demonstrates the Heston stochastic volatility model and the Merton
jump-diffusion model as extensions to standard GBM.

Financial background:
    - Black-Scholes assumes constant volatility — empirically false.
      Real markets exhibit volatility clustering and the "vol smile".
    - Heston (1993): volatility v(t) is itself a mean-reverting stochastic
      process (CIR). Parameters: kappa (speed), theta (long-run var),
      xi/volvol (vol-of-vol), rho (asset-vol correlation, usually negative).
    - Merton (1976): adds a compound Poisson jump process to GBM, capturing
      sudden price gaps (earnings surprises, macro shocks). Results in fatter
      tails than log-normal.

stocha uses:
    - Heston: Euler-Maruyama with Full Truncation (FT) scheme
      (prevents negative variance from causing errors between steps)
    - Merton: lognormal jump sizes with drift compensator to preserve
      the risk-neutral martingale property E[S(T)] = S0 * exp(mu * T)
"""

import math
import time

import numpy as np
import stocha

S0    = 100.0   # Initial stock price
mu    = 0.05    # Drift (risk-free rate under risk-neutral measure)
sigma = 0.20    # GBM baseline volatility
T     = 1.0     # 1 year
steps = 252     # Daily steps
K     = 100.0   # Strike (ATM)
r     = mu      # risk-free rate = drift

# --- 1. Heston Path Generation ---
print("=" * 60)
print("1. Heston Stochastic Volatility — Path Generation")
print("=" * 60)

v0    = 0.04   # Initial variance (sigma0 = 20%)
kappa = 2.0    # Mean-reversion speed
theta = 0.04   # Long-run variance (long-run vol = 20%)
xi    = 0.3    # Vol-of-vol
rho   = -0.7   # Asset–vol correlation (negative = leverage effect)
n_paths = 50_000

t0 = time.time()
paths = stocha.heston(
    s0=S0, v0=v0, mu=mu,
    kappa=kappa, theta=theta, xi=xi, rho=rho,
    t=T, steps=steps, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"n_paths={n_paths:,}, steps={steps}")
print(f"Output shape: {paths.shape}")
print(f"Generation time: {elapsed*1000:.1f}ms")
print(f"Terminal price: mean={paths[:, -1].mean():.4f}, "
      f"std={paths[:, -1].std():.4f}")

feller = 2 * kappa * theta / xi**2
print(f"Feller condition: 2*kappa*theta/xi^2 = {feller:.2f}  "
      f"({'satisfied' if feller > 1 else 'NOT satisfied — variance may touch 0'})")

# --- 2. Heston European Call Pricing ---
print("\n" + "=" * 60)
print("2. Heston European Call Pricing (Monte Carlo)")
print("=" * 60)

payoffs = np.maximum(paths[:, -1] - K, 0.0)
mc_price = math.exp(-r * T) * payoffs.mean()
mc_stderr = payoffs.std() / math.sqrt(n_paths)

# GBM (constant-vol) Black-Scholes reference
d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

bs_price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

print(f"Strike K={K}, r={r}")
print(f"Heston MC:      {mc_price:.4f} ± {mc_stderr:.4f}")
print(f"Black-Scholes:  {bs_price:.4f}  (constant vol={sigma:.0%})")
print(f"Difference:     {abs(mc_price - bs_price):.4f}")
print("(Heston price differs due to vol smile / skew from rho < 0)")

# --- 3. Merton Jump-Diffusion Path Generation ---
print("\n" + "=" * 60)
print("3. Merton Jump-Diffusion — Path Generation")
print("=" * 60)

lambda_ = 1.0    # 1 jump per year on average
mu_j    = -0.05  # Mean log-jump (negative → downward jumps)
sigma_j = 0.10   # Log-jump std dev

t0 = time.time()
paths_jump = stocha.merton_jump_diffusion(
    s0=S0, mu=mu, sigma=sigma,
    lambda_=lambda_, mu_j=mu_j, sigma_j=sigma_j,
    t=T, steps=steps, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"n_paths={n_paths:,}, lambda={lambda_}, mu_j={mu_j}, sigma_j={sigma_j}")
print(f"Output shape: {paths_jump.shape}")
print(f"Generation time: {elapsed*1000:.1f}ms")

# --- 4. Tail Comparison: GBM vs Merton ---
print("\n" + "=" * 60)
print("4. Tail Distribution: GBM vs Merton Jump-Diffusion")
print("   (log-returns of terminal price)")
print("=" * 60)

paths_gbm = stocha.gbm(
    s0=S0, mu=mu, sigma=sigma, t=T, steps=steps,
    n_paths=n_paths, seed=42,
)

log_ret_gbm  = np.log(paths_gbm[:, -1] / S0)
log_ret_jump = np.log(paths_jump[:, -1] / S0)

def skewness(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    return float(((x - m)**3).mean() / s**3)

def kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    return float(((x - m)**4).mean() / s**4)

print(f"\n{'Metric':<22} {'GBM':>12} {'Merton Jump':>12}")
print("-" * 48)
print(f"{'Mean log-return':<22} {log_ret_gbm.mean():>12.4f} {log_ret_jump.mean():>12.4f}")
print(f"{'Std  log-return':<22} {log_ret_gbm.std():>12.4f} {log_ret_jump.std():>12.4f}")
print(f"{'Skewness':<22} {skewness(log_ret_gbm):>12.4f} {skewness(log_ret_jump):>12.4f}")
print(f"{'Excess kurtosis':<22} {kurtosis(log_ret_gbm)-3:>12.4f} {kurtosis(log_ret_jump)-3:>12.4f}")

# Tail probabilities
for pct in [1, 5]:
    q_gbm  = float(np.percentile(log_ret_gbm, pct))
    q_jump = float(np.percentile(log_ret_jump, pct))
    print(f"{'VaR '+str(pct)+'% (log-ret)':<22} {q_gbm:>12.4f} {q_jump:>12.4f}")

print("\n(Merton: heavier left tail → higher excess kurtosis, more negative skewness)")

# --- 5. Throughput Benchmark ---
print("\n" + "=" * 60)
print("5. Throughput Benchmark")
print("=" * 60)

models = [
    ("Heston", lambda n: stocha.heston(
        s0=S0, v0=v0, mu=mu, kappa=kappa, theta=theta,
        xi=xi, rho=rho, t=T, steps=steps, n_paths=n, seed=0,
    )),
    ("Merton", lambda n: stocha.merton_jump_diffusion(
        s0=S0, mu=mu, sigma=sigma, lambda_=lambda_,
        mu_j=mu_j, sigma_j=sigma_j, t=T, steps=steps, n_paths=n, seed=0,
    )),
]

for name, fn in models:
    print(f"\n{name}:")
    for n in [1_000, 10_000, 50_000]:
        t0 = time.time()
        _ = fn(n)
        elapsed = time.time() - t0
        print(f"  n_paths={n:>7,}: {elapsed*1000:6.1f}ms  ({n/elapsed:>10,.0f} paths/s)")

print("\n✅ Stochastic volatility and jump-diffusion complete")
