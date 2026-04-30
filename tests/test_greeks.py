"""Tests for Monte Carlo Greeks (v1.4)."""

import math

import numpy as np
import pytest

import stocha


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_d1(s, k, r, sigma, t):
    return (math.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))


def bs_d2(s, k, r, sigma, t):
    return bs_d1(s, k, r, sigma, t) - sigma * math.sqrt(t)


def bs_call_delta(s, k, r, sigma, t):
    return norm_cdf(bs_d1(s, k, r, sigma, t))


def bs_put_delta(s, k, r, sigma, t):
    return bs_call_delta(s, k, r, sigma, t) - 1.0


def bs_gamma(s, k, r, sigma, t):
    d1 = bs_d1(s, k, r, sigma, t)
    return norm_pdf(d1) / (s * sigma * math.sqrt(t))


def bs_vega(s, k, r, sigma, t):
    d1 = bs_d1(s, k, r, sigma, t)
    return s * norm_pdf(d1) * math.sqrt(t)


def bs_call_theta(s, k, r, sigma, t):
    d1 = bs_d1(s, k, r, sigma, t)
    d2 = bs_d2(s, k, r, sigma, t)
    term1 = -s * norm_pdf(d1) * sigma / (2.0 * math.sqrt(t))
    term2 = -r * k * math.exp(-r * t) * norm_cdf(d2)
    return term1 + term2


def bs_call_rho(s, k, r, sigma, t):
    d2 = bs_d2(s, k, r, sigma, t)
    return k * t * math.exp(-r * t) * norm_cdf(d2)


# --- Parameters ---
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
GBM_PARAMS = {"s0": S0, "r": R, "sigma": SIGMA, "t": T}
N_PATHS = 200_000
N_STEPS = 252
SEED = 42


class TestGreeksFdGBM:
    """Bump-and-revalue Greeks for GBM vs Black-Scholes closed-form."""

    def test_delta_call(self):
        r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        expected = bs_call_delta(S0, K, R, SIGMA, T)
        assert abs(r["delta"] - expected) < 0.02, f"delta={r['delta']}, expected={expected}"

    def test_delta_put(self):
        r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="put", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        expected = bs_put_delta(S0, K, R, SIGMA, T)
        assert abs(r["delta"] - expected) < 0.02, f"delta={r['delta']}, expected={expected}"

    def test_gamma(self):
        r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["gamma"], seed=SEED,
        )
        expected = bs_gamma(S0, K, R, SIGMA, T)
        assert abs(r["gamma"] - expected) < 0.005, f"gamma={r['gamma']}, expected={expected}"

    def test_vega(self):
        r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["vega"], seed=SEED,
        )
        expected = bs_vega(S0, K, R, SIGMA, T)
        assert abs(r["vega"] - expected) < 2.0, f"vega={r['vega']}, expected={expected}"

    def test_theta(self):
        r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["theta"], seed=SEED,
        )
        expected = bs_call_theta(S0, K, R, SIGMA, T)
        assert abs(r["theta"] - expected) < 1.0, f"theta={r['theta']}, expected={expected}"

    def test_rho(self):
        r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["rho"], seed=SEED,
        )
        expected = bs_call_rho(S0, K, R, SIGMA, T)
        assert abs(r["rho"] - expected) < 2.0, f"rho={r['rho']}, expected={expected}"

    def test_all_greeks_at_once(self):
        r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS,
            greeks=["delta", "gamma", "vega", "theta", "rho"], seed=SEED,
        )
        assert set(r.keys()) == {"delta", "gamma", "vega", "theta", "rho"}
        assert 0 < r["delta"] < 1
        assert r["gamma"] > 0
        assert r["vega"] > 0
        assert r["theta"] < 0
        assert r["rho"] > 0


class TestGreeksPathwise:
    """Pathwise (IPA) Greeks for GBM."""

    def test_delta_call(self):
        r = stocha.greeks_pathwise(
            s0=S0, r=R, sigma=SIGMA, t=T, strike=K, is_call=True,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        expected = bs_call_delta(S0, K, R, SIGMA, T)
        assert abs(r["delta"] - expected) < 0.015, f"delta={r['delta']}, expected={expected}"

    def test_delta_put(self):
        r = stocha.greeks_pathwise(
            s0=S0, r=R, sigma=SIGMA, t=T, strike=K, is_call=False,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        expected = bs_put_delta(S0, K, R, SIGMA, T)
        assert abs(r["delta"] - expected) < 0.015, f"delta={r['delta']}, expected={expected}"

    def test_vega_call(self):
        r = stocha.greeks_pathwise(
            s0=S0, r=R, sigma=SIGMA, t=T, strike=K, is_call=True,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["vega"], seed=SEED,
        )
        expected = bs_vega(S0, K, R, SIGMA, T)
        assert abs(r["vega"] - expected) < 1.5, f"vega={r['vega']}, expected={expected}"

    def test_unsupported_greek_raises(self):
        with pytest.raises(ValueError, match="only supports"):
            stocha.greeks_pathwise(
                s0=S0, r=R, sigma=SIGMA, t=T, strike=K, is_call=True,
                n_paths=1000, n_steps=10, greeks=["gamma"],
            )


class TestGreeksConsistency:
    """Cross-method consistency checks."""

    def test_pathwise_vs_fd_delta(self):
        fd = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        pw = stocha.greeks_pathwise(
            s0=S0, r=R, sigma=SIGMA, t=T, strike=K, is_call=True,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        assert abs(fd["delta"] - pw["delta"]) < 0.03

    def test_put_call_delta_parity(self):
        call_r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        put_r = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="put", strike=K,
            n_paths=N_PATHS, n_steps=N_STEPS, greeks=["delta"], seed=SEED,
        )
        diff = call_r["delta"] - put_r["delta"]
        assert abs(diff - 1.0) < 0.03, f"call_delta - put_delta = {diff}, expected ~1.0"


class TestGreeksCRN:
    """Common Random Numbers: reproducibility."""

    def test_reproducibility(self):
        r1 = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=10_000, n_steps=50, greeks=["delta", "vega"], seed=123,
        )
        r2 = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=10_000, n_steps=50, greeks=["delta", "vega"], seed=123,
        )
        assert r1["delta"] == r2["delta"]
        assert r1["vega"] == r2["vega"]


class TestGreeksCustomPayoff:
    """Custom Python callable payoff."""

    def test_custom_matches_builtin(self):
        custom_call = lambda terminals: np.maximum(terminals - K, 0.0)
        r_builtin = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
            n_paths=50_000, n_steps=50, greeks=["delta"], seed=SEED,
        )
        r_custom = stocha.greeks_fd(
            model="gbm", params=GBM_PARAMS, payoff=custom_call, strike=K,
            n_paths=50_000, n_steps=50, greeks=["delta"], seed=SEED,
        )
        assert abs(r_builtin["delta"] - r_custom["delta"]) < 1e-10


class TestGreeksHestonMerton:
    """Smoke tests for non-GBM models."""

    def test_heston_delta_call(self):
        params = {
            "s0": 100.0, "v0": 0.04, "r": 0.05,
            "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7, "t": 1.0,
        }
        r = stocha.greeks_fd(
            model="heston", params=params, payoff="call", strike=100.0,
            n_paths=50_000, n_steps=100, greeks=["delta"], seed=SEED,
        )
        assert 0 < r["delta"] < 1, f"heston delta={r['delta']}"

    def test_merton_delta_call(self):
        params = {
            "s0": 100.0, "r": 0.05, "sigma": 0.2,
            "lambda_": 1.0, "mu_j": -0.05, "sigma_j": 0.1, "t": 1.0,
        }
        r = stocha.greeks_fd(
            model="merton", params=params, payoff="call", strike=100.0,
            n_paths=50_000, n_steps=100, greeks=["delta"], seed=SEED,
        )
        assert 0 < r["delta"] < 1, f"merton delta={r['delta']}"


class TestGreeksValidation:
    """Input validation."""

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            stocha.greeks_fd(
                model="invalid", params=GBM_PARAMS, payoff="call", strike=K,
                n_paths=1000, n_steps=10, greeks=["delta"],
            )

    def test_invalid_greek(self):
        with pytest.raises(ValueError, match="Unknown greek"):
            stocha.greeks_fd(
                model="gbm", params=GBM_PARAMS, payoff="call", strike=K,
                n_paths=1000, n_steps=10, greeks=["invalid"],
            )

    def test_invalid_payoff(self):
        with pytest.raises(ValueError, match="Unknown payoff"):
            stocha.greeks_fd(
                model="gbm", params=GBM_PARAMS, payoff="binary", strike=K,
                n_paths=1000, n_steps=10, greeks=["delta"],
            )

    def test_pathwise_negative_s0(self):
        with pytest.raises(ValueError, match="s0 must be positive"):
            stocha.greeks_pathwise(
                s0=-1.0, r=R, sigma=SIGMA, t=T, strike=K, is_call=True,
                n_paths=1000, n_steps=10, greeks=["delta"],
            )
