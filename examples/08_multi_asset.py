"""
stocha Tutorial 8: Multi-Asset Correlated Simulation
=====================================================

Demonstrates correlated multi-asset GBM via Cholesky decomposition,
portfolio-level risk analysis, and correlation structure verification.

Financial background:
    In practice, asset returns are correlated. A portfolio of tech stocks
    moves together; equities and bonds often exhibit negative correlation.
    Ignoring cross-asset correlation underestimates tail risk — the 2008
    crisis showed that correlations spike during market stress.

    The Cholesky approach:
        Given a correlation matrix Sigma, decompose Sigma = L L^T.
        Draw independent Z ~ N(0, I_n), then epsilon = L Z has the
        desired correlation structure. Each asset evolves as:
            S_i(t+dt) = S_i(t) * exp((mu_i - 0.5*sigma_i^2)*dt
                                      + sigma_i * sqrt(dt) * epsilon_i)
"""

import math
import time

import numpy as np
import stocha

# --- 1. Basic Multi-Asset Simulation ---
print("=" * 60)
print("1. Correlated 3-Asset GBM Simulation")
print("=" * 60)

s0 = [100.0, 50.0, 200.0]
mu = [0.05, 0.08, 0.03]
sigma = [0.20, 0.30, 0.15]
corr = np.array([
    [1.00, 0.60, 0.30],
    [0.60, 1.00, 0.50],
    [0.30, 0.50, 1.00],
])
n_paths = 50_000

t0 = time.time()
paths = stocha.multi_gbm(
    s0=s0, mu=mu, sigma=sigma, corr=corr,
    t=1.0, steps=252, n_paths=n_paths, seed=42,
)
elapsed = time.time() - t0

print(f"Shape: {paths.shape}  (n_paths, steps+1, n_assets)")
print(f"Generation time: {elapsed*1000:.1f}ms")
print(f"All prices positive: {(paths > 0).all()}")
print(f"\nInitial prices: {paths[0, 0, :]}")
print(f"Terminal means:  {paths[:, -1, :].mean(axis=0)}")
for i in range(3):
    expected = s0[i] * math.exp(mu[i])
    actual = paths[:, -1, i].mean()
    print(f"  Asset {i}: E[S(T)]={expected:.2f}, simulated={actual:.2f}, "
          f"rel_err={abs(actual-expected)/expected:.4f}")

# --- 2. Correlation Structure Verification ---
print("\n" + "=" * 60)
print("2. Correlation Structure Verification")
print("   (log-return sample correlation vs input matrix)")
print("=" * 60)

log_ret = np.log(paths[:, 1:, :] / paths[:, :-1, :])
flat = log_ret.reshape(-1, 3)
sample_corr = np.corrcoef(flat.T)

print(f"\nInput correlation matrix:")
for row in corr:
    print(f"  [{', '.join(f'{v:6.2f}' for v in row)}]")

print(f"\nSample correlation (from {flat.shape[0]:,} log-returns):")
for row in sample_corr:
    print(f"  [{', '.join(f'{v:6.4f}' for v in row)}]")

max_err = np.max(np.abs(sample_corr - corr))
print(f"\nMax absolute error: {max_err:.6f}")

# --- 3. Portfolio VaR / CVaR ---
print("\n" + "=" * 60)
print("3. Portfolio VaR / CVaR (Equal-Weight Portfolio)")
print("=" * 60)

weights = np.array([1/3, 1/3, 1/3])
initial_value = (np.array(s0) * weights).sum()
terminal_values = (paths[:, -1, :] * weights).sum(axis=1)
portfolio_returns = terminal_values / initial_value - 1.0

print(f"Initial portfolio value: {initial_value:.2f}")
print(f"Terminal mean: {terminal_values.mean():.2f}")
print(f"Portfolio return: mean={portfolio_returns.mean():.4f}, "
      f"std={portfolio_returns.std():.4f}")

print(f"\n{'Confidence':>12}  {'VaR':>10}  {'CVaR':>10}")
print("-" * 38)
for conf in [0.90, 0.95, 0.99]:
    var, cvar = stocha.var_cvar(portfolio_returns, confidence=conf)
    print(f"{conf:>12.0%}  {var:>10.4f}  {cvar:>10.4f}")

# --- 4. Antithetic Variates ---
print("\n" + "=" * 60)
print("4. Antithetic Variates: Variance Reduction")
print("=" * 60)

kw = dict(s0=s0, mu=mu, sigma=sigma, corr=corr,
          t=1.0, steps=252, n_paths=n_paths, seed=42)

plain = stocha.multi_gbm(**kw, antithetic=False)
anti = stocha.multi_gbm(**kw, antithetic=True)

for i in range(3):
    std_plain = plain[:, -1, i].std()
    std_anti = anti[:, -1, i].std()
    reduction = (1 - std_anti / std_plain) * 100
    print(f"Asset {i}: plain std={std_plain:.4f}, antithetic std={std_anti:.4f}, "
          f"reduction={reduction:.1f}%")

port_plain = (plain[:, -1, :] * weights).sum(axis=1)
port_anti = (anti[:, -1, :] * weights).sum(axis=1)
reduction = (1 - port_anti.std() / port_plain.std()) * 100
print(f"Portfolio: plain std={port_plain.std():.4f}, antithetic std={port_anti.std():.4f}, "
      f"reduction={reduction:.1f}%")

# --- 5. Throughput Benchmark ---
print("\n" + "=" * 60)
print("5. Throughput Benchmark")
print("=" * 60)

for n_assets, label in [(2, "2-asset"), (3, "3-asset"), (5, "5-asset")]:
    c = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            c[i, j] = c[j, i] = 0.5
    t0 = time.time()
    _ = stocha.multi_gbm(
        s0=[100.0]*n_assets, mu=[0.05]*n_assets, sigma=[0.2]*n_assets,
        corr=c, t=1.0, steps=252, n_paths=100_000, seed=0,
    )
    elapsed = time.time() - t0
    total_steps = 100_000 * 252 * n_assets
    print(f"  {label}: {elapsed*1000:.0f}ms  ({total_steps/elapsed/1e6:.1f}M steps/s)")

print("\n✅ Multi-asset correlated simulation complete")
