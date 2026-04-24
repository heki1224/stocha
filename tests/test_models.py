"""
Tests for stocha stochastic models: GBM, Heston, Merton, Hull-White, SABR, LSMC.

Covers: src/finance/
Run with: pytest tests/ -v
"""

import math
import pytest
import numpy as np
import stocha


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bs_put(s, k, r, sigma, t):
    """Black-Scholes European put price."""
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------

class TestGBM:
    def test_output_shape(self):
        paths = stocha.gbm(s0=100.0, mu=0.05, sigma=0.2,
                           t=1.0, steps=252, n_paths=1_000, seed=0)
        assert paths.shape == (1_000, 253)
        assert paths.dtype == np.float64

    def test_initial_price(self):
        paths = stocha.gbm(s0=50.0, mu=0.0, sigma=0.1,
                           t=1.0, steps=10, n_paths=100, seed=0)
        np.testing.assert_array_equal(paths[:, 0], 50.0)

    def test_reproducibility(self):
        kw = dict(s0=100.0, mu=0.05, sigma=0.2, t=1.0, steps=252,
                  n_paths=1_000, seed=42)
        np.testing.assert_array_equal(stocha.gbm(**kw), stocha.gbm(**kw))

    def test_expected_terminal_price(self):
        # E[S(T)] = S0 * exp(mu * T)
        paths = stocha.gbm(s0=100.0, mu=0.05, sigma=0.2,
                           t=1.0, steps=252, n_paths=100_000, seed=0)
        mean_terminal = paths[:, -1].mean()
        expected = 100.0 * math.exp(0.05)
        assert pytest.approx(mean_terminal, rel=5e-3) == expected

    def test_positive_prices(self):
        paths = stocha.gbm(s0=100.0, mu=0.05, sigma=0.3,
                           t=1.0, steps=252, n_paths=10_000, seed=0)
        assert (paths > 0).all()

    def test_antithetic_reduces_variance(self):
        kw = dict(s0=100.0, mu=0.05, sigma=0.2, t=1.0, steps=252,
                  n_paths=50_000, seed=42)
        plain = stocha.gbm(**kw, antithetic=False)
        anti = stocha.gbm(**kw, antithetic=True)
        assert anti[:, -1].std() < plain[:, -1].std()

    def test_negative_s0(self):
        with pytest.raises((ValueError, Exception)):
            stocha.gbm(s0=-1.0, mu=0.05, sigma=0.2,
                       t=1.0, steps=10, n_paths=10, seed=0)

    def test_zero_sigma(self):
        with pytest.raises((ValueError, Exception)):
            stocha.gbm(s0=100.0, mu=0.05, sigma=0.0,
                       t=1.0, steps=10, n_paths=5, seed=0)

    def test_negative_t(self):
        with pytest.raises((ValueError, Exception)):
            stocha.gbm(s0=100.0, mu=0.05, sigma=0.2,
                       t=-1.0, steps=10, n_paths=10, seed=0)


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------

class TestHeston:
    def _kw(self, n_paths=1000):
        return dict(
            s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04,
            xi=0.3, rho=-0.7, t=1.0, steps=252, n_paths=n_paths, seed=42
        )

    def test_output_shape(self):
        paths = stocha.heston(**self._kw())
        assert paths.shape == (1000, 253)
        assert paths.dtype == np.float64

    def test_initial_price(self):
        paths = stocha.heston(**self._kw())
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_positive_prices(self):
        paths = stocha.heston(**self._kw())
        assert (paths > 0).all()

    def test_reproducibility(self):
        kw = self._kw()
        np.testing.assert_array_equal(stocha.heston(**kw), stocha.heston(**kw))

    def test_expected_terminal_price(self):
        # E[S(T)] = S0 * exp(mu * T) under Heston
        paths = stocha.heston(**self._kw(n_paths=100_000))
        mean_terminal = paths[:, -1].mean()
        expected = 100.0 * math.exp(0.05)
        assert pytest.approx(mean_terminal, rel=2e-2) == expected

    def test_invalid_s0(self):
        kw = self._kw()
        kw["s0"] = -1.0
        with pytest.raises((ValueError, Exception)):
            stocha.heston(**kw)

    def test_invalid_rho(self):
        kw = self._kw()
        kw["rho"] = 2.0
        with pytest.raises((ValueError, Exception)):
            stocha.heston(**kw)


# ---------------------------------------------------------------------------
# Merton Jump-Diffusion
# ---------------------------------------------------------------------------

class TestMertonJumpDiffusion:
    def _kw(self, n_paths=1000):
        return dict(
            s0=100.0, mu=0.05, sigma=0.2,
            lambda_=1.0, mu_j=-0.05, sigma_j=0.1,
            t=1.0, steps=252, n_paths=n_paths, seed=42
        )

    def test_output_shape(self):
        paths = stocha.merton_jump_diffusion(**self._kw())
        assert paths.shape == (1000, 253)
        assert paths.dtype == np.float64

    def test_initial_price(self):
        paths = stocha.merton_jump_diffusion(**self._kw())
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_positive_prices(self):
        paths = stocha.merton_jump_diffusion(**self._kw())
        assert (paths > 0).all()

    def test_reproducibility(self):
        kw = self._kw()
        np.testing.assert_array_equal(
            stocha.merton_jump_diffusion(**kw),
            stocha.merton_jump_diffusion(**kw),
        )

    def test_expected_terminal_price(self):
        # E[S(T)] = S0 * exp(mu * T) due to Merton compensator
        paths = stocha.merton_jump_diffusion(**self._kw(n_paths=200_000))
        mean_terminal = paths[:, -1].mean()
        expected = 100.0 * math.exp(0.05)
        assert pytest.approx(mean_terminal, rel=2e-2) == expected

    def test_invalid_sigma(self):
        kw = self._kw()
        kw["sigma"] = 0.0
        with pytest.raises((ValueError, Exception)):
            stocha.merton_jump_diffusion(**kw)

    def test_invalid_s0(self):
        kw = self._kw()
        kw["s0"] = -1.0
        with pytest.raises((ValueError, Exception)):
            stocha.merton_jump_diffusion(**kw)


# ---------------------------------------------------------------------------
# Hull-White
# ---------------------------------------------------------------------------

class TestHullWhite:
    def test_shape(self):
        rates = stocha.hull_white(
            r0=0.05, a=0.1, theta=0.005, sigma=0.01, t=1.0, steps=12, n_paths=100
        )
        assert rates.shape == (100, 13)

    def test_initial_rate(self):
        rates = stocha.hull_white(
            r0=0.05, a=0.1, theta=0.005, sigma=0.01, t=1.0, steps=12, n_paths=500
        )
        np.testing.assert_allclose(rates[:, 0], 0.05)

    def test_mean_reversion(self):
        # With large a and small noise, terminal rate should be close to theta/a.
        long_run = 0.05
        a = 1.0
        theta = a * long_run
        rates = stocha.hull_white(
            r0=0.10, a=a, theta=theta, sigma=0.001,
            t=10.0, steps=100, n_paths=2000, seed=0
        )
        terminal_mean = rates[:, -1].mean()
        assert abs(terminal_mean - long_run) < 0.01, \
            f"terminal_mean={terminal_mean:.4f}, expected≈{long_run}"

    def test_invalid_a(self):
        with pytest.raises(Exception):
            stocha.hull_white(r0=0.05, a=0.0, theta=0.005, sigma=0.01,
                              t=1.0, steps=12, n_paths=10)

    def test_invalid_sigma(self):
        with pytest.raises(Exception):
            stocha.hull_white(r0=0.05, a=0.1, theta=0.005, sigma=0.0,
                              t=1.0, steps=12, n_paths=10)


# ---------------------------------------------------------------------------
# SABR Implied Volatility
# ---------------------------------------------------------------------------

class TestSabrImpliedVol:
    def test_atm_vol_positive(self):
        iv = stocha.sabr_implied_vol(f=0.05, k=0.05, t=1.0,
                                     alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
        assert iv > 0

    def test_vol_smile_negative_rho(self):
        # Negative rho → negatively skewed smile: low-strike vol > high-strike vol.
        atm = stocha.sabr_implied_vol(f=0.20, k=0.20, t=1.0,
                                      alpha=0.04, beta=0.5, rho=-0.5, nu=0.3)
        low_strike = stocha.sabr_implied_vol(f=0.20, k=0.18, t=1.0,
                                             alpha=0.04, beta=0.5, rho=-0.5, nu=0.3)
        high_strike = stocha.sabr_implied_vol(f=0.20, k=0.22, t=1.0,
                                              alpha=0.04, beta=0.5, rho=-0.5, nu=0.3)
        assert low_strike > high_strike, \
            f"low={low_strike:.4f}, atm={atm:.4f}, high={high_strike:.4f}"

    def test_beta_zero_is_normal(self):
        iv = stocha.sabr_implied_vol(f=0.05, k=0.05, t=1.0,
                                     alpha=0.005, beta=0.0, rho=0.0, nu=0.0)
        assert iv > 0

    def test_shifted_sabr_negative_rate(self):
        iv = stocha.sabr_implied_vol(f=-0.005, k=-0.01, t=1.0,
                                     alpha=0.003, beta=0.5, rho=-0.3, nu=0.3,
                                     shift=0.03)
        assert iv > 0

    def test_invalid_alpha(self):
        with pytest.raises(Exception):
            stocha.sabr_implied_vol(f=0.05, k=0.05, t=1.0,
                                    alpha=-0.1, beta=0.5, rho=0.0, nu=0.3)

    def test_reproducibility(self):
        iv1 = stocha.sabr_implied_vol(f=0.05, k=0.05, t=1.0,
                                      alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
        iv2 = stocha.sabr_implied_vol(f=0.05, k=0.05, t=1.0,
                                      alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
        assert iv1 == iv2


# ---------------------------------------------------------------------------
# LSMC American Option
# ---------------------------------------------------------------------------

class TestLsmcAmericanOption:
    def test_shape(self):
        price, std_err = stocha.lsmc_american_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
            steps=10, n_paths=1000
        )
        assert isinstance(price, float)
        assert isinstance(std_err, float)
        assert price > 0
        assert std_err > 0

    def test_american_put_ge_european(self):
        # American put >= European put (early exercise premium).
        price, std_err = stocha.lsmc_american_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
            steps=50, n_paths=30_000, seed=0
        )
        european = bs_put(100.0, 100.0, 0.05, 0.20, 1.0)
        assert price >= european - 3 * std_err, \
            f"American={price:.4f}, European={european:.4f}"

    def test_american_put_accuracy(self):
        # Known benchmark: ATM put S=K=100, r=5%, σ=20%, T=1y ≈ $6.07.
        price, std_err = stocha.lsmc_american_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
            steps=50, n_paths=50_000, seed=42
        )
        assert abs(price - 6.07) < 0.30, f"price={price:.4f}"

    def test_deep_itm_put(self):
        price, std_err = stocha.lsmc_american_option(
            s0=80.0, k=100.0, r=0.01, sigma=0.10, t=1.0,
            steps=50, n_paths=10_000
        )
        assert price > 15.0, f"price={price:.4f}"

    def test_deep_otm_put_near_zero(self):
        price, std_err = stocha.lsmc_american_option(
            s0=150.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
            steps=50, n_paths=10_000
        )
        assert price < 2.0, f"price={price:.4f}"

    def test_call_option(self):
        price, std_err = stocha.lsmc_american_option(
            s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
            steps=50, n_paths=20_000, is_put=False
        )
        assert price > 0

    def test_invalid_s0(self):
        with pytest.raises(Exception):
            stocha.lsmc_american_option(s0=-1.0, k=100.0, r=0.05,
                                        sigma=0.20, t=1.0, steps=10, n_paths=100)

    def test_invalid_poly_degree(self):
        with pytest.raises(Exception):
            stocha.lsmc_american_option(s0=100.0, k=100.0, r=0.05,
                                        sigma=0.20, t=1.0, steps=10,
                                        n_paths=100, poly_degree=0)

    def test_reproducibility(self):
        kwargs = dict(s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
                      steps=50, n_paths=5_000, seed=7)
        p1, e1 = stocha.lsmc_american_option(**kwargs)
        p2, e2 = stocha.lsmc_american_option(**kwargs)
        assert p1 == p2
