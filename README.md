# stocha

> High-performance random number and financial simulation library for Python, powered by Rust.

[![PyPI](https://img.shields.io/pypi/v/stocha)](https://pypi.org/project/stocha/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[日本語版 README](README.ja.md)

## Features

- **Fast PRNG**: PCG64DXSM (NumPy default algorithm), Xoshiro256++, MT19937
- **Normal distribution**: Marsaglia polar method (Ziggurat planned for v0.2)
- **GBM simulation**: Euler-Maruyama with Rayon parallel path generation
- **Antithetic variates**: built-in variance reduction
- **Reproducible**: block-split RNG streams guarantee identical results across thread counts
- **PyTorch / JAX ready**: DLPack zero-copy tensor output (v0.2)

## Installation

```bash
pip install stocha
```

## Quick Start

```python
import stocha

# Random number generation
rng = stocha.RNG(seed=42)
samples = rng.normal(size=10_000, loc=0.0, scale=1.0)

# GBM stock price simulation
paths = stocha.gbm(
    s0=100.0,   # initial price
    mu=0.05,    # drift (annual)
    sigma=0.20, # volatility (annual)
    t=1.0,      # 1 year
    steps=252,  # daily steps
    n_paths=100_000,
    seed=42,
)
# paths.shape == (100_000, 253)
```

## Performance (Apple M-series, release build)

| Operation | Throughput |
|---|---|
| Normal sampling | ~155M samples/sec |
| GBM (252 steps) | ~680k paths/sec |

## Roadmap

| Version | Features |
|---|---|
| **v0.1** | PCG64DXSM, normal distribution, GBM, antithetic variates |
| **v0.2** | Sobol sequence (Joe & Kuo 2008 + Owen scrambling), Halton, Heston model, Jump-Diffusion |
| **v1.0** | VaR/CVaR, Gaussian/t-Copula, Hull-White, SABR, LSMC, DLPack zero-copy |

## License

Licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Shigeki Yamato
