# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-04-22

### Fixed

- **LSMC**: panic on `n_paths=1` due to division by `(n-1)=0` in std_err calculation;
  now returns `std_err=0.0` for `n=1`, and Python binding rejects `n_paths < 2` with `ValueError`
- **Hull-White**: RNG stream initialization switched from `seed.wrapping_add(i)` to
  `advance(i * block_size)` (consistent with GBM/Heston/Merton); improves statistical
  independence between paths
- **Gaussian copula**: added validation that correlation matrix diagonal elements equal 1.0
  and that the matrix is symmetric (tolerance 1e-10); previously silent wrong results
- **SABR**: removed dead variable `gamma1`; ATM detection changed from absolute-difference
  to `|log(F/K)| < 1e-8` (scale-invariant, more stable for small strikes)
- **Cargo.toml**: corrected license field from `MIT OR Apache-2.0` to `MIT`

## [0.3.0] - 2026-04-22

### Added

- **VaR / CVaR** (`stocha.var_cvar(returns, confidence)`): Sort-based Value-at-Risk
  and Conditional VaR (Expected Shortfall) from a return series; zero-copy NumPy input
- **Gaussian copula** (`stocha.gaussian_copula(corr, n_samples)`): Cholesky decomposition
  + normal CDF transform; models joint dependence with Gaussian tail behavior
- **Student-t copula** (`stocha.student_t_copula(corr, nu, n_samples)`): Same structure
  with chi-squared scaling for heavier joint tails; custom Gamma sampler (Marsaglia-Tsang)
  with no external distribution library
- **Hull-White 1-factor** (`stocha.hull_white(r0, a, theta, sigma, t, steps, n_paths)`):
  Short-rate model with Exact Simulation (zero discretization bias); Rayon-parallel
- **SABR implied vol** (`stocha.sabr_implied_vol(f, k, t, alpha, beta, rho, nu, shift=0.0)`):
  Hagan et al. (2002) Black implied volatility formula; `shift` parameter for negative-rate
  support (Shifted SABR); pure analytic computation, no simulation
- **LSMC American option** (`stocha.lsmc_american_option(s0, k, r, sigma, t, steps, n_paths, ...)`):
  Longstaff-Schwartz backward induction with faer QR least-squares; polynomial basis
  normalized by spot mean/std to avoid multicollinearity; returns `(price, std_err)`

### Dependencies

- `faer = "0.24"` — pure-Rust linear algebra (QR solver for LSMC regression)

## [0.2.0] - 2026-04-22

### Added

- **Sobol sequence** (`stocha.sobol(dim, n_samples)`): Joe & Kuo 2008 direction numbers
  via `sobol-qmc 2.5.1`; f64 output, up to 1,000 dimensions (21,201 with `EXTENDED` params)
- **Halton sequence** (`stocha.halton(dim, n_samples, skip=0)`): van der Corput
  construction with consecutive prime bases; up to 40 dimensions, `skip` parameter
  for warm-up
- **Heston model** (`stocha.heston(...)`): Stochastic volatility with Euler-Maruyama
  and Full Truncation (FT) scheme; correlated Brownians via Cholesky decomposition
- **Merton Jump-Diffusion** (`stocha.merton_jump_diffusion(...)`): Lognormal jump sizes
  with Poisson arrivals (Bernoulli approximation); martingale compensator applied

## [0.1.0] - 2026-04-22

### Added

- **PCG64DXSM PRNG** (`RNG` class): NumPy-compatible default algorithm with reproducible
  block-split streams for parallel path generation
- **Normal distribution sampling**: Marsaglia polar method via `RNG.normal()`
  and `stocha.standard_normal()`
- **Geometric Brownian Motion** (`stocha.gbm()`): Euler-Maruyama discretization
  with Rayon-parallel path generation
- **Antithetic variates**: Built-in variance reduction via `antithetic=True` parameter
- **Reproducibility guarantee**: `advance(i * block_size)` block splitting ensures
  identical results regardless of thread count
- **State serialization**: Seed-based save/load via `RNG.save_state()`
- MIT License

### Performance (Apple M-series, release build)

| Operation | Throughput |
|---|---|
| `standard_normal(10M samples)` | ~155M samples/sec |
| `gbm(n_paths=100k, steps=252)` | ~680k paths/sec |

[Unreleased]: https://github.com/heki1224/stocha/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/heki1224/stocha/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/heki1224/stocha/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/heki1224/stocha/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/heki1224/stocha/releases/tag/v0.1.0
