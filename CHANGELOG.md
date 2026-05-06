# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.0] - 2026-05-06

### Added

- **`ssvi_calibrate`**: Fit SSVI (Surface SVI) parameters `(η, γ, ρ)` to
  observed total implied variance data. Uses Projected Levenberg-Marquardt
  with analytical Jacobian. The SSVI parameterization (Gatheral & Jacquier)
  guarantees calendar-arbitrage-free surfaces by construction.
- **`ssvi_implied_vol`**: Compute implied volatilities from calibrated SSVI
  parameters for arbitrary strikes and maturities.
- **`ssvi_local_vol`**: Compute the Dupire local volatility surface from SSVI
  parameters using closed-form partial derivatives (∂w/∂k, ∂²w/∂k², ∂w/∂θ).
  No numerical finite differences — numerically stable even in the tails.
- **Continuous dividend yield (`q`)**: Added to `gbm()` and `bs_price_div()`.
  Drift becomes `(μ - q)` in GBM; Black-Scholes uses discounted spot
  `S·exp(-qT)`. Backward compatible (default `q=0`).

## [1.5.0] - 2026-05-01

### Added

- **`heston_price`**: Analytical European option pricing under the Heston model
  using the COS method (Fang & Oosterlee 2008). Employs the Albrecher et al.
  (2007) characteristic function formulation that avoids complex logarithm
  branch-cut discontinuities. Truncation range from cumulants with hard-limit
  safety bounds. Supports vectorized multi-strike pricing with shared CF cache.
  Default N=160 COS terms; configurable via `n_cos` parameter.
- **`heston_calibrate`**: Fit Heston parameters `(v0, kappa, theta, xi, rho)` to
  observed option prices. Uses a Projected Levenberg-Marquardt optimizer with
  Vega-weighted price residuals (avoids IV solver overhead). Central-difference
  Jacobian with automatic switch to forward/backward difference near parameter
  boundaries. Rayon-parallel Jacobian computation across parameters. Heuristic
  initial guess from ATM implied volatility. Feller condition reported as a
  diagnostic flag (soft constraint). Returns dict with fitted parameters, RMSE,
  iteration count, convergence status, and Feller flag.

### Dependencies

- `num-complex`: added `0.4` for complex arithmetic in characteristic function

## [1.4.0] - 2026-04-30

### Added

- **`greeks_fd`**: Compute Monte Carlo Greeks via bump-and-revalue finite
  difference. Supports GBM, Heston, and Merton models. Computes delta, gamma,
  vega, theta, and rho using central differences (forward difference for theta).
  All bump scenarios share the same random seed (Common Random Numbers) for
  variance reduction. Scenario deduplication avoids redundant simulations when
  computing multiple Greeks. Built-in payoffs (call/put) execute entirely in
  Rust with no GIL overhead; custom Python callables use a vectorized design
  with a single GIL acquisition per scenario. Adaptive bump size defaults to
  1% of parameter value.
- **`greeks_pathwise`**: Compute Monte Carlo Greeks via pathwise (IPA) method
  for GBM European options. Supports delta and vega. More accurate than
  bump-and-revalue for continuous payoffs — requires only a single simulation
  run. Uses the analytical sensitivity of the GBM terminal price with respect
  to spot and volatility.

## [1.3.0] - 2026-04-29

### Added

- **`multi_gbm`**: Simulate correlated multi-asset GBM paths. Uses Cholesky
  decomposition of the correlation matrix to generate correlated Brownian
  increments, then applies the log-Euler scheme independently to each asset.
  Returns a 3-D array of shape `(n_paths, steps + 1, n_assets)` optimized for
  portfolio-level analytics (VaR, time-series aggregation). Supports antithetic
  variates for variance reduction. Rayon-parallel over paths with block-split
  RNG streams for full reproducibility.
- `examples/08_multi_asset.py` (and `.ja.py`): multi-asset correlated
  simulation tutorial — portfolio VaR, correlation verification, antithetic
  variates comparison.

## [1.2.0] - 2026-04-28

### Changed

- **`RNG.save_state()`**: Now serializes the full internal RNG state (position +
  seed), enabling mid-stream checkpointing. Previously recorded seed only.
  `from_state()` accepts both the new full-state format and the legacy
  `{"seed": N}` format for backward compatibility.

### Added

- **Heston QE scheme**: `heston(..., scheme="qe")` selects the Andersen (2008)
  Quadratic Exponential discretization with martingale-corrected log-price
  update. More accurate than Euler-FT with fewer time steps, especially when
  the Feller condition is violated. Default remains `"euler"` for backward
  compatibility.

### Dependencies

- `rand_pcg`: added `serde` feature for RNG state serialization
- `serde_json`: added `arbitrary_precision` feature for u128 round-trip

## [1.1.0] - 2026-04-27

### Added

- **`sabr_calibrate`**: Fit SABR parameters `(α, ρ, ν)` to an observed Black
  implied-vol smile. The ATM α is recovered exactly by 1-D Brent root-finding
  on the Hagan ATM formula; `(ρ, ν)` are then fit by a Projected
  Levenberg-Marquardt loop with central-difference Jacobian and step
  clipping for box constraints. β is held fixed (industry standard).
  Supports negative rates via the same `shift` parameter as
  `sabr_implied_vol`. Returns a dict with `alpha`, `rho`, `nu`, `rmse`,
  `iterations`, `converged`.
- `examples/06_interest_rate.py` (and `.ja.py`): added a calibration demo
  that recovers `(α, ρ, ν)` from a synthetic smile.

## [1.0.0] - 2026-04-24

### Changed

- **Ziggurat sampler**: Normal distribution sampling switched from Marsaglia polar
  method to Ziggurat algorithm (N=256 rectangles). ~3× faster than previous
  implementation; now matches NumPy throughput (~300M samples/sec on Apple M-series).
- **GBM throughput**: Geometric Brownian Motion path generation improved from ~1.6×
  to ~3.0× faster than NumPy (benefiting from the Ziggurat normal sampler).

### Added

- `scripts/gen_ziggurat_table.py`: Ziggurat table generator using mpmath
  (50-digit precision) for reproducible constant generation.

## [0.3.3] - 2026-04-24

### Security

- **F-1 (HIGH) GIL-released ArrayView data race**: `gaussian_copula` and `student_t_copula`
  previously passed a raw `ArrayView2` borrow of the Python-managed `corr` array into
  `py.detach()`. A concurrent Python thread could have mutated the array while the GIL was
  released, causing undefined behavior. Fixed by calling `.to_owned()` before entering
  `py.detach`, so Rust holds an independent copy during parallel execution.
- **F-2 (HIGH) NaN/Inf input causes process abort in `var_cvar`**: `sort_unstable_by` with
  `.partial_cmp().unwrap()` panicked on NaN inputs, crashing the entire Python process via
  PyO3. Fixed by adding an `is_finite()` pre-check that returns `ValueError` for any NaN or
  infinite value, and switching the sort comparator to `total_cmp` as a secondary safeguard.

## [0.3.2] - 2026-04-22

### Added

- **`RNG.from_state(json)`**: class method to restore an RNG from a JSON string
  produced by `save_state`; equivalent to `RNG(seed=original_seed)`
- **`RNG` seed accepts 128-bit integers**: `seed` parameter upgraded from `u64` to
  `u128`; full 128-bit seed space accessible from Python
- **Tests**: LSMC call option, Merton `lambda=0` → GBM equivalence, Heston Feller
  condition violation stability, Hull-White negative initial rate, VaR with all-positive
  returns

### Fixed

- **`RNG.save_state`**: docstring now clearly states that only the seed is recorded
  (not the full internal state); restoring replays from the beginning, not from the
  current position
- **CVaR definition**: inline comment and docstring clarified that CVaR is computed
  as `E[Loss | Loss >= VaR]` (inclusive boundary); applicable only to discrete samples
- **Code quality**: `into_py_array2` helper eliminates 7 duplicate `Array2` conversion
  patterns in `src/lib.rs`; `#[allow(dead_code)]` narrowed from impl-block scope to
  individual unused methods; `norm_ppf` doc/allow attribute order corrected

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

[Unreleased]: https://github.com/heki1224/stocha/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/heki1224/stocha/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/heki1224/stocha/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/heki1224/stocha/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/heki1224/stocha/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/heki1224/stocha/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/heki1224/stocha/compare/v0.3.3...v1.0.0
[0.3.3]: https://github.com/heki1224/stocha/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/heki1224/stocha/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/heki1224/stocha/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/heki1224/stocha/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/heki1224/stocha/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/heki1224/stocha/releases/tag/v0.1.0
