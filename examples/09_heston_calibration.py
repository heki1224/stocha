"""
stocha Tutorial 9: Heston Analytical Pricing and Calibration
============================================================

Demonstrates the COS method for fast Heston option pricing and
Levenberg-Marquardt calibration of the Heston stochastic volatility model.

Financial background:
    The Heston (1993) model:
        dS = r*S*dt + sqrt(V)*S*dW_S
        dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
        corr(dW_S, dW_V) = rho

    Unlike Black-Scholes, Heston captures the volatility smile/skew
    through stochastic variance. The model has a semi-analytical
    characteristic function, enabling fast option pricing via the
    COS method (Fang & Oosterlee 2008) without Monte Carlo simulation.

    Calibration fits (v0, kappa, theta, xi, rho) to observed market
    option prices using a Projected Levenberg-Marquardt optimizer
    with Vega-weighted residuals.

Topics covered:
    1. COS method pricing — analytical Heston call/put prices
    2. Implied volatility smile from Heston prices
    3. Calibration — recover parameters from synthetic market data
    4. Multi-maturity calibration — term structure of vol surface
"""

import math
import time

import numpy as np
import stocha


# ──────────────────────────────────────────────
# Heston model parameters (typical equity index)
# ──────────────────────────────────────────────
S0    = 100.0
V0    = 0.04    # Initial variance (20% vol)
R     = 0.05    # Risk-free rate
KAPPA = 2.0     # Mean-reversion speed
THETA = 0.04    # Long-run variance
XI    = 0.3     # Vol-of-vol
RHO   = -0.7    # Spot-vol correlation (negative = leverage effect)
T     = 1.0     # 1-year maturity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. COS Method Pricing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("1. Heston COS Method — Analytical Option Pricing")
print("=" * 60)

strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)

t0 = time.time()
call_prices = stocha.heston_price(
    strikes=strikes, is_call=[True] * len(strikes),
    s0=S0, v0=V0, r=R, kappa=KAPPA, theta=THETA, xi=XI, rho=RHO, t=T,
)
put_prices = stocha.heston_price(
    strikes=strikes, is_call=[False] * len(strikes),
    s0=S0, v0=V0, r=R, kappa=KAPPA, theta=THETA, xi=XI, rho=RHO, t=T,
)
elapsed = time.time() - t0

print(f"\nParameters: S0={S0}, V0={V0}, r={R}, kappa={KAPPA}, theta={THETA}")
print(f"            xi={XI}, rho={RHO}, T={T}")
print(f"Pricing time: {elapsed*1000:.1f} ms ({len(strikes)} strikes x 2)")
print(f"\n{'Strike':>8} {'Call':>10} {'Put':>10} {'Parity err':>12}")
print("-" * 44)
for i, k in enumerate(strikes):
    parity_err = call_prices[i] - put_prices[i] - S0 + k * math.exp(-R * T)
    print(f"{k:8.0f} {call_prices[i]:10.4f} {put_prices[i]:10.4f} {parity_err:12.2e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Implied Volatility Smile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("2. Heston Implied Volatility Smile")
print("=" * 60)


def bs_iv_newton(price, s, k, r, t, is_call, tol=1e-10):
    """Newton's method to invert BS price to implied vol."""
    def _ncdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    sigma = 0.2
    for _ in range(100):
        d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        if is_call:
            bs = s * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
        else:
            bs = k * math.exp(-r * t) * _ncdf(-d2) - s * _ncdf(-d1)
        vega = s * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * math.sqrt(t)
        if vega < 1e-20:
            break
        diff = bs - price
        if abs(diff) < tol:
            break
        sigma -= diff / vega
        sigma = max(sigma, 0.001)
    return sigma


print(f"\n{'Strike':>8} {'Call Price':>12} {'Implied Vol':>12}")
print("-" * 36)
for i, k in enumerate(strikes):
    iv = bs_iv_newton(call_prices[i], S0, k, R, T, True)
    print(f"{k:8.0f} {call_prices[i]:12.4f} {iv:12.4f}")

print("\nNote: Negative rho (-0.7) produces a downward-sloping skew")
print("      (lower strikes have higher implied vol — leverage effect).")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Calibration — Single Maturity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("3. Heston Calibration — Single Maturity")
print("=" * 60)

# Generate synthetic "market" prices with known params
true_params = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
calib_strikes = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
market_prices = stocha.heston_price(
    strikes=calib_strikes, is_call=[True] * 7,
    s0=S0, r=R, t=T, n_cos=256, **true_params,
)

t0 = time.time()
result = stocha.heston_calibrate(
    strikes=calib_strikes,
    maturities=np.full(7, T),
    market_prices=market_prices,
    is_call=[True] * 7,
    s0=S0, r=R,
)
calib_time = time.time() - t0

print(f"\nCalibration time: {calib_time*1000:.0f} ms")
print(f"Converged: {result['converged']}, Iterations: {result['iterations']}")
print(f"RMSE: {result['rmse']:.2e}")
print(f"Feller condition (2κθ > ξ²): {result['feller_satisfied']}")
print(f"\n{'Param':>8} {'True':>10} {'Calibrated':>12} {'Error':>10}")
print("-" * 44)
for p in ['v0', 'kappa', 'theta', 'xi', 'rho']:
    err = abs(result[p] - true_params[p])
    print(f"{p:>8} {true_params[p]:10.4f} {result[p]:12.6f} {err:10.2e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Multi-Maturity Calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("4. Heston Calibration — Multi-Maturity (Vol Surface)")
print("=" * 60)

true2 = dict(v0=0.06, kappa=1.5, theta=0.06, xi=0.5, rho=-0.8)
maturities_set = [0.25, 0.5, 1.0]
strikes_per_mat = [90.0, 95.0, 100.0, 105.0, 110.0]

all_strikes = []
all_mats = []
all_prices = []
all_calls = []

for tau in maturities_set:
    ks = np.array(strikes_per_mat, dtype=float)
    ps = stocha.heston_price(
        strikes=ks, is_call=[True] * len(ks),
        s0=S0, r=R, t=tau, n_cos=256, **true2,
    )
    all_strikes.extend(strikes_per_mat)
    all_mats.extend([tau] * len(ks))
    all_prices.extend(ps.tolist())
    all_calls.extend([True] * len(ks))

t0 = time.time()
result2 = stocha.heston_calibrate(
    strikes=np.array(all_strikes),
    maturities=np.array(all_mats),
    market_prices=np.array(all_prices),
    is_call=all_calls,
    s0=S0, r=R, max_iter=300,
)
calib_time2 = time.time() - t0

print(f"\nData: {len(all_strikes)} options across maturities {maturities_set}")
print(f"Calibration time: {calib_time2*1000:.0f} ms")
print(f"Converged: {result2['converged']}, Iterations: {result2['iterations']}")
print(f"RMSE: {result2['rmse']:.2e}")
print(f"Feller condition: {result2['feller_satisfied']}")
print(f"\n{'Param':>8} {'True':>10} {'Calibrated':>12} {'Error':>10}")
print("-" * 44)
for p in ['v0', 'kappa', 'theta', 'xi', 'rho']:
    err = abs(result2[p] - true2[p])
    print(f"{p:>8} {true2[p]:10.4f} {result2[p]:12.6f} {err:10.2e}")

print("\n" + "=" * 60)
print("Tutorial complete.")
print("=" * 60)
