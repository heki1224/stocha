# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/heki1224/stocha/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/heki1224/stocha/releases/tag/v0.1.0
