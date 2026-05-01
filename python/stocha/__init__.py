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
from stocha._stocha import var_cvar as _var_cvar
from stocha._stocha import gaussian_copula as _gaussian_copula
from stocha._stocha import student_t_copula as _student_t_copula
from stocha._stocha import hull_white as _hull_white
from stocha._stocha import sabr_implied_vol as _sabr_implied_vol
from stocha._stocha import sabr_calibrate as _sabr_calibrate
from stocha._stocha import multi_gbm as _multi_gbm
from stocha._stocha import lsmc_american_option as _lsmc_american_option
from stocha._stocha import greeks_fd as _greeks_fd
from stocha._stocha import greeks_pathwise as _greeks_pathwise
from stocha._stocha import heston_price as _heston_price
from stocha._stocha import heston_calibrate as _heston_calibrate
from stocha._stocha import __version__

__all__ = [
    "RNG",
    "gbm",
    "sobol",
    "halton",
    "heston",
    "merton_jump_diffusion",
    "var_cvar",
    "gaussian_copula",
    "student_t_copula",
    "hull_white",
    "sabr_implied_vol",
    "sabr_calibrate",
    "multi_gbm",
    "lsmc_american_option",
    "greeks_fd",
    "greeks_pathwise",
    "heston_price",
    "heston_calibrate",
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
        """Serialize the full RNG state to a JSON string.

        Captures the exact internal position, enabling mid-stream
        checkpointing. Restoring via :meth:`from_state` resumes the
        sequence from the saved position.

        Returns:
            JSON string containing the full generator state.
        """
        return self._inner.save_state()

    @classmethod
    def from_state(cls, json: str) -> "RNG":
        """Restore an RNG from a JSON string produced by :meth:`save_state`.

        Accepts both full-state (v1.2+) and legacy seed-only format.
        Full-state restores the exact position; seed-only restarts
        from the beginning.

        Args:
            json: JSON string as returned by :meth:`save_state`.

        Returns:
            New ``RNG`` instance.
        """
        inner = _RNG.from_state(json)
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

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
    scheme: Literal["euler", "qe"] = "euler",
) -> np.ndarray:
    """Simulate Heston stochastic volatility paths.

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
        scheme: Discretization scheme (default ``"euler"``).
                ``"euler"``: Full Truncation Euler-Maruyama.
                ``"qe"``: Andersen (2008) Quadratic Exponential with
                martingale correction. More accurate with fewer steps.

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
        scheme=scheme,
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


def var_cvar(
    returns: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    """Compute Value-at-Risk and Conditional VaR (Expected Shortfall).

    Losses are defined as *negative* returns. Both outputs are positive
    (they represent loss magnitudes, not signed P&L).

    Args:
        returns:    1-D NumPy array of portfolio returns.
        confidence: Confidence level in (0, 1). E.g. ``0.95`` for 95% VaR.

    Returns:
        ``(var, cvar)`` where both values are positive loss magnitudes.

    Example:
        >>> paths = gbm(s0=100.0, mu=0.05, sigma=0.2, t=1.0, steps=252, n_paths=10000)
        >>> returns = paths[:, -1] / paths[:, 0] - 1
        >>> var, cvar = var_cvar(returns, confidence=0.95)
    """
    return _var_cvar(np.asarray(returns, dtype=np.float64), confidence)


def gaussian_copula(
    corr: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample from a Gaussian copula with a given correlation matrix.

    Args:
        corr:      2-D NumPy array of shape ``(dim, dim)``.
                   Must be a valid positive-definite correlation matrix.
        n_samples: Number of samples to draw.
        seed:      Random seed (default ``42``).

    Returns:
        NumPy array of shape ``(n_samples, dim)`` with values in ``(0, 1)``.

    Example:
        >>> import numpy as np
        >>> corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        >>> u = gaussian_copula(corr, n_samples=1000)
    """
    return _gaussian_copula(
        np.asarray(corr, dtype=np.float64), n_samples=n_samples, seed=seed
    )


def student_t_copula(
    corr: np.ndarray,
    nu: float,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample from a Student-t copula with a given correlation matrix.

    Heavier tails than the Gaussian copula — captures joint extreme events.

    Args:
        corr:      2-D NumPy array of shape ``(dim, dim)``.
                   Must be a valid positive-definite correlation matrix.
        nu:        Degrees of freedom (must be > 2 for finite variance).
        n_samples: Number of samples to draw.
        seed:      Random seed (default ``42``).

    Returns:
        NumPy array of shape ``(n_samples, dim)`` with values in ``(0, 1)``.
    """
    return _student_t_copula(
        np.asarray(corr, dtype=np.float64), nu=nu, n_samples=n_samples, seed=seed
    )


def hull_white(
    r0: float,
    a: float,
    theta: float,
    sigma: float,
    t: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate Hull-White 1-factor short-rate paths via Exact Simulation.

    Model: ``dr = (theta - a*r)*dt + sigma*dW``

    Uses the exact Gaussian transition (zero discretization bias).
    The long-run mean rate is ``theta / a``.

    Args:
        r0:      Initial short rate.
        a:       Mean-reversion speed (must be > 0).
        theta:   Drift constant equal to ``a * long_run_mean``.
        sigma:   Volatility of the short rate (must be > 0).
        t:       Time horizon in years (must be > 0).
        steps:   Number of time steps.
        n_paths: Number of simulation paths.
        seed:    Random seed (default ``42``).

    Returns:
        NumPy array of shape ``(n_paths, steps + 1)``.
        Column 0 is the initial rate ``r0``.

    Example:
        >>> rates = hull_white(r0=0.05, a=0.1, theta=0.005,
        ...                    sigma=0.01, t=1.0, steps=252, n_paths=10000)
        >>> rates.shape
        (10000, 253)
    """
    return _hull_white(
        r0=r0, a=a, theta=theta, sigma=sigma,
        t=t, steps=steps, n_paths=n_paths, seed=seed,
    )


def sabr_implied_vol(
    f: float,
    k: float,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> float:
    """Compute the Black implied volatility using the SABR model.

    Uses the Hagan et al. (2002) approximation formula.
    Supports negative rates via the Shifted SABR approach.

    Args:
        f:     Forward price or rate.
        k:     Strike price or rate.
        t:     Time to expiry in years (must be > 0).
        alpha: Initial volatility (must be > 0).
        beta:  CEV exponent in [0, 1]. ``1`` = lognormal, ``0`` = normal.
        rho:   Correlation between forward and vol Brownians (in (-1, 1)).
        nu:    Vol-of-vol (must be >= 0).
        shift: Shift for negative-rate support (default ``0.0``).

    Returns:
        Black (lognormal) implied volatility as a float.

    Example:
        >>> iv = sabr_implied_vol(f=0.05, k=0.05, t=1.0,
        ...                       alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
    """
    return _sabr_implied_vol(f=f, k=k, t=t, alpha=alpha, beta=beta,
                             rho=rho, nu=nu, shift=shift)


def sabr_calibrate(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    f: float,
    t: float,
    beta: float = 0.5,
    shift: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict:
    """Calibrate SABR parameters (alpha, rho, nu) to an observed IV smile.

    Beta is held fixed (industry standard). The ATM alpha is recovered exactly
    by 1-D Brent root-finding on the Hagan ATM formula; (rho, nu) are then
    fit by a Projected Levenberg-Marquardt loop.

    Args:
        strikes:     1-D array of strikes K_i.
        market_vols: 1-D array of observed Black implied vols.
        f:           Forward price or rate.
        t:           Time to expiry in years (must be > 0).
        beta:        CEV exponent in [0, 1] (default ``0.5``).
        shift:       Shift for negative-rate support (default ``0.0``).
        max_iter:    Maximum LM iterations (default ``100``).
        tol:         Convergence tolerance (default ``1e-10``).

    Returns:
        Dict with keys ``alpha``, ``rho``, ``nu``, ``rmse``, ``iterations``,
        ``converged``.

    Example:
        >>> import numpy as np
        >>> strikes = np.array([0.04, 0.045, 0.05, 0.055, 0.06])
        >>> vols = np.array([0.25, 0.22, 0.20, 0.19, 0.185])
        >>> r = sabr_calibrate(strikes, vols, f=0.05, t=1.0, beta=0.5)
    """
    return _sabr_calibrate(
        np.asarray(strikes, dtype=np.float64),
        np.asarray(market_vols, dtype=np.float64),
        f=f, t=t, beta=beta, shift=shift,
        max_iter=max_iter, tol=tol,
    )


def multi_gbm(
    s0: list[float],
    mu: list[float],
    sigma: list[float],
    corr: np.ndarray,
    t: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate correlated multi-asset GBM paths.

    Uses Cholesky decomposition of the correlation matrix to generate
    correlated Brownian increments, then applies the log-Euler scheme
    independently to each asset.

    Args:
        s0:         List of initial asset prices (all must be > 0).
        mu:         List of drift rates (annualized), one per asset.
        sigma:      List of volatilities (annualized, all must be > 0).
        corr:       Correlation matrix of shape ``(n_assets, n_assets)``.
                    Must be symmetric and positive definite.
        t:          Time to maturity in years (must be > 0).
        steps:      Number of time steps (e.g. 252 for daily, 1-year).
        n_paths:    Number of simulation paths.
        seed:       Random seed (default ``42``).
        antithetic: Use antithetic variates for variance reduction (default ``False``).

    Returns:
        NumPy array of shape ``(n_paths, steps + 1, n_assets)``.
        ``result[:, 0, :]`` contains the initial prices.
        ``result[:, -1, :]`` contains the terminal prices.

    Example:
        >>> import numpy as np
        >>> corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        >>> paths = multi_gbm(s0=[100.0, 50.0], mu=[0.05, 0.08],
        ...                   sigma=[0.2, 0.3], corr=corr,
        ...                   t=1.0, steps=252, n_paths=10000)
        >>> paths.shape
        (10000, 253, 2)
    """
    return _multi_gbm(
        s0=list(s0),
        mu=list(mu),
        sigma=list(sigma),
        corr=np.asarray(corr, dtype=np.float64),
        t=t,
        steps=steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )


def lsmc_american_option(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
    n_paths: int,
    is_put: bool = True,
    poly_degree: int = 3,
    seed: int = 42,
) -> tuple[float, float]:
    """Price an American option via Longstaff-Schwartz Monte Carlo (LSMC).

    Simulates GBM paths under the risk-neutral measure, then applies backward
    induction with polynomial least-squares regression (QR decomposition) to
    determine the optimal early-exercise boundary.

    Args:
        s0:         Initial asset price (must be > 0).
        k:          Strike price (must be > 0).
        r:          Risk-free rate (annualized).
        sigma:      Volatility (annualized, must be > 0).
        t:          Time to maturity in years (must be > 0).
        steps:      Number of exercise opportunities (time steps).
        n_paths:    Number of simulation paths.
        is_put:     ``True`` for put option, ``False`` for call (default ``True``).
        poly_degree: Polynomial degree for basis functions (1–4, default ``3``).
        seed:       Random seed (default ``42``).

    Returns:
        ``(price, std_err)`` tuple.

    Example:
        >>> price, err = lsmc_american_option(
        ...     s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
        ...     steps=50, n_paths=50000)
    """
    return _lsmc_american_option(
        s0=s0, k=k, r=r, sigma=sigma, t=t, steps=steps,
        n_paths=n_paths, is_put=is_put, poly_degree=poly_degree, seed=seed,
    )


def greeks_fd(
    model: str,
    params: dict,
    payoff: "str | Callable",
    strike: float,
    n_paths: int,
    n_steps: int,
    greeks: list[str],
    seed: int = 42,
    bump_size: float = 0.01,
) -> dict[str, float]:
    """Compute Monte Carlo Greeks via bump-and-revalue (finite difference).

    All bump scenarios use the same random seed (Common Random Numbers)
    for variance reduction.

    Args:
        model:     Model name: ``"gbm"``, ``"heston"``, or ``"merton"``.
        params:    Dict of model parameters.

                   - GBM: ``s0, r, sigma, t``
                   - Heston: ``s0, v0, r, kappa, theta, xi, rho, t``
                   - Merton: ``s0, r, sigma, lambda_, mu_j, sigma_j, t``
        payoff:    ``"call"``, ``"put"``, or a callable ``f(terminals) -> values``
                   where ``terminals`` is a 1-D NumPy array of terminal prices.
        strike:    Strike price (used by built-in payoffs).
        n_paths:   Number of simulation paths.
        n_steps:   Number of time steps.
        greeks:    List of Greeks: ``"delta"``, ``"gamma"``, ``"vega"``,
                   ``"theta"``, ``"rho"``.
        seed:      Random seed (default ``42``).
        bump_size: Relative bump size (default ``0.01`` = 1%).

    Returns:
        Dict mapping Greek names to float values.

    Example:
        >>> r = greeks_fd(
        ...     model="gbm",
        ...     params={"s0": 100, "r": 0.05, "sigma": 0.2, "t": 1.0},
        ...     payoff="call", strike=100.0,
        ...     n_paths=100_000, n_steps=252,
        ...     greeks=["delta", "vega"])
    """
    return _greeks_fd(
        model=model, params=params, payoff=payoff, strike=strike,
        n_paths=n_paths, n_steps=n_steps, greeks=greeks,
        seed=seed, bump_size=bump_size,
    )


def greeks_pathwise(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    strike: float,
    is_call: bool,
    n_paths: int,
    n_steps: int,
    greeks: list[str],
    seed: int = 42,
) -> dict[str, float]:
    """Compute Monte Carlo Greeks via pathwise (IPA) method (GBM only).

    More accurate than bump-and-revalue for continuous payoffs (European
    call/put). Only requires a single simulation run.

    Args:
        s0:      Initial asset price (must be > 0).
        r:       Risk-free rate (annualized).
        sigma:   Volatility (annualized, must be > 0).
        t:       Time to maturity in years (must be > 0).
        strike:  Strike price.
        is_call: ``True`` for call, ``False`` for put.
        n_paths: Number of simulation paths.
        n_steps: Number of time steps.
        greeks:  List of Greeks: ``"delta"`` and/or ``"vega"``.
        seed:    Random seed (default ``42``).

    Returns:
        Dict mapping Greek names to float values.

    Example:
        >>> r = greeks_pathwise(
        ...     s0=100.0, r=0.05, sigma=0.2, t=1.0,
        ...     strike=100.0, is_call=True,
        ...     n_paths=100_000, n_steps=252,
        ...     greeks=["delta", "vega"])
    """
    return _greeks_pathwise(
        s0=s0, r=r, sigma=sigma, t=t,
        strike=strike, is_call=is_call,
        n_paths=n_paths, n_steps=n_steps,
        greeks=greeks, seed=seed,
    )


def heston_price(
    strikes: np.ndarray,
    is_call: list[bool],
    s0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_cos: int = 160,
) -> np.ndarray:
    """Price European options under the Heston model using the COS method.

    Uses the Fang & Oosterlee (2008) COS expansion with the Albrecher (2007)
    characteristic function (branch-cut safe).

    Args:
        strikes:  1-D array of strike prices.
        is_call:  List of booleans (True for call, False for put).
        s0:       Spot price (must be > 0).
        v0:       Initial variance (must be > 0).
        r:        Risk-free rate (annualized).
        kappa:    Mean-reversion speed (must be > 0).
        theta:    Long-run variance (must be > 0).
        xi:       Vol-of-vol (must be > 0).
        rho:      Correlation in (-1, 1).
        t:        Time to maturity in years (must be > 0).
        n_cos:    Number of COS expansion terms (default 160).

    Returns:
        1-D NumPy array of option prices.

    Example:
        >>> import numpy as np
        >>> prices = heston_price(
        ...     strikes=np.array([90.0, 100.0, 110.0]),
        ...     is_call=[True, True, True],
        ...     s0=100.0, v0=0.04, r=0.05,
        ...     kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, t=1.0)
    """
    return _heston_price(
        np.asarray(strikes, dtype=np.float64),
        list(is_call),
        s0=s0, v0=v0, r=r, kappa=kappa, theta=theta,
        xi=xi, rho=rho, t=t, n_cos=n_cos,
    )


def heston_calibrate(
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_prices: np.ndarray,
    is_call: list[bool],
    s0: float,
    r: float,
    max_iter: int = 200,
    tol: float = 1e-8,
    n_cos: int = 160,
) -> dict:
    """Calibrate Heston parameters to market option prices.

    Fits ``(v0, kappa, theta, xi, rho)`` using Projected Levenberg-Marquardt
    with Vega-weighted price residuals. COS method repricing ensures speed.

    Args:
        strikes:       1-D array of strike prices.
        maturities:    1-D array of times to maturity (years).
        market_prices: 1-D array of observed option prices.
        is_call:       List of booleans (True for call).
        s0:            Spot price (must be > 0).
        r:             Risk-free rate.
        max_iter:      Max LM iterations (default 200).
        tol:           Convergence tolerance (default 1e-8).
        n_cos:         COS terms (default 160).

    Returns:
        Dict with ``v0``, ``kappa``, ``theta``, ``xi``, ``rho``, ``rmse``,
        ``iterations``, ``converged``, ``feller_satisfied``.

    Example:
        >>> r = heston_calibrate(
        ...     strikes=np.array([90, 95, 100, 105, 110]),
        ...     maturities=np.array([1.0]*5),
        ...     market_prices=prices,
        ...     is_call=[True]*5,
        ...     s0=100.0, r=0.05)
    """
    return _heston_calibrate(
        np.asarray(strikes, dtype=np.float64),
        np.asarray(maturities, dtype=np.float64),
        np.asarray(market_prices, dtype=np.float64),
        list(is_call),
        s0=s0, r=r, max_iter=max_iter, tol=tol, n_cos=n_cos,
    )
