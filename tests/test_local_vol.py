"""Tests for SSVI surface, Dupire local vol, and dividend support (v1.6.0)."""

import numpy as np
import pytest

import stocha


class TestDividendGBM:
    """GBM with continuous dividend yield."""

    def test_dividend_reduces_drift(self):
        """q > 0 should lower the expected terminal price."""
        params = dict(s0=100.0, mu=0.05, sigma=0.2, t=1.0, steps=252, n_paths=50000, seed=0)
        paths_no_div = stocha.gbm(**params)
        paths_div = stocha.gbm(**params, q=0.03)
        mean_no_div = paths_no_div[:, -1].mean()
        mean_div = paths_div[:, -1].mean()
        assert mean_div < mean_no_div

    def test_dividend_expected_value(self):
        """E[S(T)] = S0 * exp((mu - q) * T) under GBM with dividends."""
        s0, mu, q, t = 100.0, 0.08, 0.03, 1.0
        paths = stocha.gbm(s0=s0, mu=mu, sigma=0.2, t=t, steps=252, n_paths=100000, seed=7, q=q)
        mean_terminal = paths[:, -1].mean()
        expected = s0 * np.exp((mu - q) * t)
        rel_err = abs(mean_terminal - expected) / expected
        assert rel_err < 0.02, f"rel_err={rel_err:.4f}"

    def test_q_zero_backward_compatible(self):
        """q=0 should produce identical results to the old API."""
        params = dict(s0=100.0, mu=0.05, sigma=0.2, t=1.0, steps=50, n_paths=100, seed=42)
        paths_default = stocha.gbm(**params)
        paths_explicit = stocha.gbm(**params, q=0.0)
        np.testing.assert_array_equal(paths_default, paths_explicit)


class TestSSVICalibrate:
    """SSVI calibration tests."""

    def test_roundtrip(self):
        """Calibrate to synthetic SSVI data and recover parameters."""
        true_eta, true_gamma, true_rho = 1.2, 0.4, -0.4
        thetas = [0.01, 0.02, 0.04, 0.06, 0.09]
        ks = np.linspace(-0.3, 0.3, 7)

        log_m, theta_v, market_w = [], [], []
        for th in thetas:
            for k in ks:
                log_m.append(k)
                theta_v.append(th)
                phi = true_eta / (th**true_gamma * (1 + th)**(1 - true_gamma))
                pk_rho = phi * k + true_rho
                w = 0.5 * th * (1 + true_rho * phi * k + np.sqrt(pk_rho**2 + 1 - true_rho**2))
                market_w.append(w)

        result = stocha.ssvi_calibrate(
            np.array(log_m), np.array(theta_v), np.array(market_w),
            max_iter=200, tol=1e-12,
        )
        assert result["converged"]
        assert result["rmse"] < 1e-6
        assert abs(result["eta"] - true_eta) < 0.02
        assert abs(result["gamma"] - true_gamma) < 0.02
        assert abs(result["rho"] - true_rho) < 0.02

    def test_min_data_points(self):
        """Should reject fewer than 3 data points."""
        with pytest.raises(ValueError):
            stocha.ssvi_calibrate(
                np.array([0.0, 0.1]),
                np.array([0.04, 0.04]),
                np.array([0.04, 0.045]),
            )


class TestSSVIImpliedVol:
    """SSVI implied vol computation."""

    def test_atm_vol(self):
        """At k=0, implied vol = sqrt(theta / T)."""
        theta, t = 0.04, 1.0
        vols = stocha.ssvi_implied_vol(
            np.array([0.0]), theta=theta, t=t, eta=1.0, gamma=0.5, rho=-0.3,
        )
        expected = np.sqrt(theta / t)
        np.testing.assert_allclose(vols[0], expected, rtol=1e-10)

    def test_smile_shape(self):
        """With negative rho, left wing should be higher (skew)."""
        ks = np.linspace(-0.3, 0.3, 7)
        vols = stocha.ssvi_implied_vol(
            ks, theta=0.04, t=1.0, eta=1.0, gamma=0.5, rho=-0.5,
        )
        assert vols[0] > vols[-1], "Negative rho should produce downward skew"

    def test_all_positive(self):
        """All implied vols should be positive."""
        ks = np.linspace(-0.5, 0.5, 20)
        vols = stocha.ssvi_implied_vol(
            ks, theta=0.04, t=1.0, eta=1.5, gamma=0.5, rho=-0.3,
        )
        assert np.all(vols > 0)


class TestSSVILocalVol:
    """Dupire local volatility from SSVI surface."""

    def test_output_shape(self):
        """Output shape should be (n_slices, n_strikes)."""
        ks = np.linspace(-0.3, 0.3, 30)
        thetas = np.array([0.01, 0.02, 0.04, 0.06])
        ts = np.array([0.25, 0.5, 1.0, 1.5])
        lv = stocha.ssvi_local_vol(ks, thetas, ts, eta=1.0, gamma=0.5, rho=-0.3)
        assert lv.shape == (4, 30)

    def test_all_positive(self):
        """Local vol should be positive everywhere for well-behaved parameters."""
        ks = np.linspace(-0.2, 0.2, 20)
        thetas = np.array([0.01, 0.02, 0.04, 0.06, 0.09])
        ts = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
        lv = stocha.ssvi_local_vol(ks, thetas, ts, eta=0.5, gamma=0.5, rho=-0.3)
        assert np.all(lv > 0), f"Negative local vol found: min={lv.min()}"

    def test_atm_local_vol_reasonable(self):
        """ATM local vol should be in a reasonable range (10%–50% for typical params)."""
        ks = np.array([0.0])
        thetas = np.array([0.04])
        ts = np.array([1.0])
        lv = stocha.ssvi_local_vol(ks, thetas, ts, eta=0.8, gamma=0.5, rho=-0.3)
        assert 0.05 < lv[0, 0] < 0.50, f"ATM local vol = {lv[0, 0]}"
