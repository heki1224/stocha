"""
stocha - High-performance random number and financial simulation library.

Provides fast random/quasi-random number generation and stochastic financial
models backed by a Rust core with Rayon parallel processing.

Example:
    >>> import stocha
    >>> rng = stocha.RNG(seed=42)
    >>> samples = rng.normal(size=1000, loc=0.0, scale=1.0)
    >>> paths = stocha.gbm(s0=100.0, mu=0.05, sigma=0.2, t=1.0, steps=252, n_paths=10000)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from stocha._stocha import RNG as _RNG
from stocha._stocha import gbm as _gbm
from stocha._stocha import __version__

__all__ = ["RNG", "gbm", "__version__"]


class RNG:
    """High-speed pseudo-random number generator using PCG64DXSM.

    Same algorithm as NumPy's default RNG. Supports jump-ahead for
    independent parallel streams and JSON state serialization.

    Args:
        seed:      Integer seed for reproducible results (default 0).
        algorithm: RNG algorithm. Only ``"pcg64dxsm"`` is supported in Phase 1.

    Example:
        >>> rng = RNG(seed=42)
        >>> rng.normal(size=5)
        array([...])
    """

    def __init__(
        self,
        seed: int = 0,
        algorithm: Literal["pcg64dxsm"] = "pcg64dxsm",
    ) -> None:
        self._inner = _RNG(seed=seed, algorithm=algorithm)

    @property
    def seed(self) -> int:
        """The seed used to initialize this RNG."""
        return self._inner.seed

    def uniform(self, size: int = 1) -> np.ndarray:
        """Sample from Uniform[0, 1).

        Args:
            size: Number of samples.

        Returns:
            NumPy array of shape ``(size,)``.
        """
        return self._inner.uniform(size)

    def standard_normal(self, size: int = 1) -> np.ndarray:
        """Sample from the standard normal distribution N(0, 1).

        Uses the Marsaglia polar method (Ziggurat planned for Phase 2).

        Args:
            size: Number of samples.

        Returns:
            NumPy array of shape ``(size,)``.
        """
        return self._inner.standard_normal(size)

    def normal(
        self,
        size: int = 1,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Sample from N(loc, scale^2).

        Args:
            size:  Number of samples.
            loc:   Mean (default ``0.0``).
            scale: Standard deviation, must be positive (default ``1.0``).

        Returns:
            NumPy array of shape ``(size,)``.
        """
        return self._inner.normal(size=size, loc=loc, scale=scale)

    def save_state(self) -> str:
        """Serialize the current RNG state to a JSON string.

        Can be used for checkpointing and audit trails.
        """
        return self._inner.save_state()

    def __repr__(self) -> str:
        return f"RNG(seed={self.seed}, algorithm='pcg64dxsm')"


def gbm(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate Geometric Brownian Motion (GBM) paths.

    Applies Euler-Maruyama discretization with Rayon parallel path generation.
    Each path uses an independent RNG stream (block splitting) for reproducibility.

    Args:
        s0:         Initial asset price (must be > 0).
        mu:         Drift rate (annualized).
        sigma:      Volatility (annualized, must be > 0).
        t:          Time to maturity in years (must be > 0).
        steps:      Number of time steps (e.g. 252 for daily, 1-year).
        n_paths:    Number of simulation paths to generate.
        seed:       Random seed for reproducibility (default ``42``).
        antithetic: If ``True``, use antithetic variates for variance reduction.

    Returns:
        NumPy array of shape ``(n_paths, steps + 1)``.
        Each row is one simulated price path.
        Column 0 is the initial price ``s0``.
        Column ``steps`` is the terminal price.

    Example:
        >>> paths = gbm(s0=100.0, mu=0.05, sigma=0.2, t=1.0, steps=252, n_paths=10000)
        >>> paths.shape
        (10000, 253)
        >>> paths[:, 0].mean()
        100.0
    """
    return _gbm(
        s0=s0,
        mu=mu,
        sigma=sigma,
        t=t,
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
