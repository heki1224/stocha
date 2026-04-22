# stocha

> High-performance random number and financial simulation library for Python, powered by Rust.

[![PyPI](https://img.shields.io/pypi/v/stocha)](https://pypi.org/project/stocha/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[日本語版 README](README.ja.md)

## Features

- **Fast PRNG**: PCG64DXSM (NumPy default algorithm)
- **Quasi-random sequences**: Sobol (Joe & Kuo 2008) and Halton sequences
- **Stochastic models**: GBM, Heston, Merton Jump-Diffusion, Hull-White
- **Risk metrics**: VaR and CVaR (Expected Shortfall)
- **Copulas**: Gaussian and Student-t copulas for multivariate dependence
- **Volatility**: SABR implied volatility (Hagan 2002) with negative-rate support
- **Option pricing**: Longstaff-Schwartz LSMC for American options
- **Parallel**: Rayon-powered path generation
- **Reproducible**: block-split RNG streams guarantee identical results across thread counts

## Installation

**PyPI (coming soon):**

```bash
pip install stocha
```

**From source (current):**

```bash
git clone https://github.com/heki1224/stocha.git
cd stocha
pip install maturin
maturin develop --release
```

## Quick Start

```python
import stocha
import numpy as np

# ── Random number generation ──────────────────────────────────────────────
rng = stocha.RNG(seed=42)
samples = rng.normal(size=10_000, loc=0.0, scale=1.0)

# ── GBM stock price simulation ────────────────────────────────────────────
paths = stocha.gbm(
    s0=100.0, mu=0.05, sigma=0.20,
    t=1.0, steps=252, n_paths=100_000, seed=42,
)
# paths.shape == (100_000, 253)

# ── VaR / CVaR ────────────────────────────────────────────────────────────
returns = paths[:, -1] / paths[:, 0] - 1
var, cvar = stocha.var_cvar(returns, confidence=0.95)
print(f"95% VaR={var:.4f}  CVaR={cvar:.4f}")

# ── Gaussian copula ───────────────────────────────────────────────────────
corr = np.array([[1.0, 0.8], [0.8, 1.0]])
u = stocha.gaussian_copula(corr, n_samples=10_000)
# u.shape == (10_000, 2),  values in (0, 1)

# ── Student-t copula (heavier joint tails) ────────────────────────────────
u_t = stocha.student_t_copula(corr, nu=5.0, n_samples=10_000)

# ── Hull-White short-rate model ───────────────────────────────────────────
rates = stocha.hull_white(
    r0=0.05, a=0.1, theta=0.005, sigma=0.01,
    t=1.0, steps=252, n_paths=10_000,
)
# rates.shape == (10_000, 253)

# ── SABR implied volatility ───────────────────────────────────────────────
iv = stocha.sabr_implied_vol(
    f=0.05, k=0.05, t=1.0,
    alpha=0.20, beta=0.5, rho=-0.3, nu=0.4,
)
print(f"SABR ATM implied vol: {iv:.4f}")

# ── American option via LSMC ──────────────────────────────────────────────
price, std_err = stocha.lsmc_american_option(
    s0=100.0, k=100.0, r=0.05, sigma=0.20,
    t=1.0, steps=50, n_paths=50_000,
)
print(f"American put: {price:.4f} ± {std_err:.4f}")
```

## API Reference

### Random Number Generation

| Function / Class | Description |
|---|---|
| `RNG(seed)` | PCG64DXSM PRNG; `seed` accepts integers up to 128-bit |
| `RNG.normal(size, loc, scale)` | Sample from N(loc, scale²) |
| `RNG.uniform(size)` | Sample from Uniform[0, 1) |
| `RNG.save_state()` | Serialize seed to JSON string (records seed only, not stream position) |
| `RNG.from_state(json)` | Restore RNG from JSON produced by `save_state`; equivalent to `RNG(seed=original_seed)` |
| `sobol(dim, n_samples)` | Sobol low-discrepancy sequence (Joe & Kuo 2008) |
| `halton(dim, n_samples, skip)` | Halton low-discrepancy sequence |

### Stochastic Price Models

| Function | Description |
|---|---|
| `gbm(s0, mu, sigma, t, steps, n_paths, ...)` | Geometric Brownian Motion (Euler-Maruyama, Rayon-parallel) |
| `heston(s0, v0, mu, kappa, theta, xi, rho, ...)` | Heston stochastic volatility (Full Truncation scheme) |
| `merton_jump_diffusion(s0, mu, sigma, lambda_, ...)` | Merton Jump-Diffusion with lognormal jumps |
| `hull_white(r0, a, theta, sigma, t, steps, n_paths)` | Hull-White 1-factor short rate (Exact Simulation) |

### Risk & Derivatives

| Function | Description |
|---|---|
| `var_cvar(returns, confidence)` | Value-at-Risk and Conditional VaR |
| `gaussian_copula(corr, n_samples)` | Gaussian copula samples |
| `student_t_copula(corr, nu, n_samples)` | Student-t copula samples |
| `sabr_implied_vol(f, k, t, alpha, beta, rho, nu, shift)` | SABR Black implied volatility |
| `lsmc_american_option(s0, k, r, sigma, t, steps, n_paths, ...)` | American option price via Longstaff-Schwartz LSMC |

## Performance (Apple M-series, release build)

| Operation | Throughput |
|---|---|
| Normal sampling | ~155M samples/sec |
| GBM (252 steps) | ~680k paths/sec |

## Tutorials

| File | Contents |
|---|---|
| `examples/01_basic_rng.py` | RNG basics, reproducibility, performance benchmark |
| `examples/02_stock_gbm.py` | GBM path simulation, European option pricing, antithetic variates |
| `examples/03_quasi_random.py` | Sobol/Halton sequences, QMC vs MC convergence comparison |
| `examples/04_stochastic_vol.py` | Heston stochastic volatility, Merton jump-diffusion |
| `examples/05_risk_copula.py` | VaR/CVaR, Gaussian/Student-t copula tail dependence |
| `examples/06_interest_rate.py` | Hull-White short-rate model, SABR volatility smile |
| `examples/07_american_option.py` | LSMC American option pricing, early exercise premium |

Each example has a Japanese counterpart (`*.ja.py`).

## Roadmap

| Version | Features |
|---|---|
| **v0.1** | PCG64DXSM, normal distribution, GBM, antithetic variates |
| **v0.2** | Sobol (Joe & Kuo 2008), Halton, Heston model, Merton Jump-Diffusion |
| **v0.3** | VaR/CVaR, Gaussian/Student-t copula, Hull-White, SABR, LSMC |
| **v1.0** | Ziggurat sampler, DLPack zero-copy, calibration utilities |

## License

Licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Shigeki Yamato
