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
from stocha._stocha import sobol as _sobol
from stocha._stocha import halton as _halton
from stocha._stocha import heston as _heston
from stocha._stocha import merton_jump_diffusion as _merton_jump_diffusion
from stocha._stocha import __version__

__all__ = [
    "RNG",
    "gbm",
    "sobol",
    "halton",
    "heston",
    "merton_jump_diffusion",
    "__version__",
]


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


def sobol(dim: int, n_samples: int) -> np.ndarray:
    """Generate a Sobol low-discrepancy sequence.

    Uses Joe & Kuo 2008 direction numbers with up to 1,000 dimensions.

    Args:
        dim:       Number of dimensions (1–1000).
        n_samples: Number of sample points.

    Returns:
        NumPy array of shape ``(n_samples, dim)`` with values in [0, 1).

    Example:
        >>> pts = sobol(dim=2, n_samples=1024)
        >>> pts.shape
        (1024, 2)
    """
    return _sobol(dim=dim, n_samples=n_samples)


def halton(dim: int, n_samples: int, skip: int = 0) -> np.ndarray:
    """Generate a Halton low-discrepancy sequence.

    Uses consecutive primes (2, 3, 5, ...) as the base for each dimension.
    Supports up to 40 dimensions.

    Args:
        dim:       Number of dimensions (1–40).
        n_samples: Number of sample points.
        skip:      Number of initial elements to skip (default ``0``).

    Returns:
        NumPy array of shape ``(n_samples, dim)`` with values in (0, 1).

    Example:
        >>> pts = halton(dim=2, n_samples=1000)
        >>> pts.shape
        (1000, 2)
    """
    return _halton(dim=dim, n_samples=n_samples, skip=skip)


def heston(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate Heston stochastic volatility paths.

    Uses Euler-Maruyama with the Full Truncation (FT) scheme so that
    the variance process v(t) can go negative between steps (reducing
    discretization bias near the boundary).

    Args:
        s0:     Initial asset price (must be > 0).
        v0:     Initial variance (not volatility; v0 = sigma0**2, e.g. 0.04).
        mu:     Drift rate (annualized).
        kappa:  Mean-reversion speed of variance (must be > 0).
        theta:  Long-run mean variance (must be > 0).
        xi:     Volatility of variance / vol-of-vol (must be > 0).
        rho:    Correlation between asset and variance Brownians (in [-1, 1]).
        t:      Time to maturity in years (must be > 0).
        steps:  Number of time steps (e.g. 252 for daily over 1 year).
        n_paths: Number of simulation paths.
        seed:   Random seed (default ``42``).

    Returns:
        NumPy array of shape ``(n_paths, steps + 1)``.
        Column 0 is the initial price ``s0``.

    Example:
        >>> paths = heston(s0=100, v0=0.04, mu=0.05, kappa=2.0,
        ...                theta=0.04, xi=0.3, rho=-0.7,
        ...                t=1.0, steps=252, n_paths=10000)
        >>> paths.shape
        (10000, 253)
    """
    return _heston(
        s0=s0, v0=v0, mu=mu, kappa=kappa, theta=theta,
        xi=xi, rho=rho, t=t, steps=steps, n_paths=n_paths, seed=seed,
    )


def merton_jump_diffusion(
    s0: float,
    mu: float,
    sigma: float,
    lambda_: float,
    mu_j: float,
    sigma_j: float,
    t: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate Merton Jump-Diffusion paths.

    Model: dS = (mu - lambda*m_bar)*S*dt + sigma*S*dW + S*(J-1)*dN
    where J = exp(mu_j + sigma_j*Z) is a lognormal jump size and
    the compensator ensures E[S(T)] = S0 * exp(mu * T).

    Args:
        s0:      Initial asset price (must be > 0).
        mu:      Drift rate (annualized).
        sigma:   Diffusion volatility (annualized, must be > 0).
        lambda_: Jump intensity: average jumps per year (must be >= 0).
        mu_j:    Mean of the log-jump size.
        sigma_j: Std dev of the log-jump size (must be >= 0).
        t:       Time to maturity in years (must be > 0).
        steps:   Number of time steps.
        n_paths: Number of simulation paths.
        seed:    Random seed (default ``42``).

    Returns:
        NumPy array of shape ``(n_paths, steps + 1)``.
        Column 0 is the initial price ``s0``.

    Example:
        >>> paths = merton_jump_diffusion(
        ...     s0=100, mu=0.05, sigma=0.2,
        ...     lambda_=1.0, mu_j=-0.05, sigma_j=0.1,
        ...     t=1.0, steps=252, n_paths=10000)
        >>> paths.shape
        (10000, 253)
    """
    return _merton_jump_diffusion(
        s0=s0, mu=mu, sigma=sigma, lambda_=lambda_,
        mu_j=mu_j, sigma_j=sigma_j, t=t, steps=steps,
        n_paths=n_paths, seed=seed,
    )
