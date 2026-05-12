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

    def test_invalid_scheme(self):
        kw = self._kw()
        with pytest.raises((ValueError, Exception)):
            stocha.heston(**kw, scheme="bad")


class TestHestonQE:
    def _kw(self, n_paths=1000):
        return dict(
            s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04,
            xi=0.3, rho=-0.7, t=1.0, steps=252, n_paths=n_paths,
            seed=42, scheme="qe",
        )

    def test_output_shape(self):
        paths = stocha.heston(**self._kw())
        assert paths.shape == (1000, 253)

    def test_positive_prices(self):
        paths = stocha.heston(**self._kw())
        assert (paths > 0).all()

    def test_reproducibility(self):
        kw = self._kw()
        np.testing.assert_array_equal(stocha.heston(**kw), stocha.heston(**kw))

    def test_expected_terminal_price(self):
        paths = stocha.heston(**self._kw(n_paths=100_000))
        mean_terminal = paths[:, -1].mean()
        expected = 100.0 * math.exp(0.05)
        assert pytest.approx(mean_terminal, rel=2e-2) == expected

    def test_feller_violated_stability(self):
        paths = stocha.heston(
            s0=100.0, v0=0.04, mu=0.05, kappa=1.0, theta=0.04,
            xi=0.3, rho=-0.7, t=1.0, steps=252, n_paths=5000,
            seed=77, scheme="qe",
        )
        assert paths.shape == (5000, 253)
        assert (paths > 0).all()

    def test_euler_default_backward_compat(self):
        kw = dict(
            s0=100.0, v0=0.04, mu=0.05, kappa=2.0, theta=0.04,
            xi=0.3, rho=-0.7, t=1.0, steps=50, n_paths=10, seed=42,
        )
        default_paths = stocha.heston(**kw)
        euler_paths = stocha.heston(**kw, scheme="euler")
        np.testing.assert_array_equal(default_paths, euler_paths)


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
# Merton MJD European call accuracy audit
# (closed form vs Monte Carlo, plus closed-form internal sanity checks)
# ---------------------------------------------------------------------------

def _bs_call(S, K, r, sigma, T):
    if sigma <= 0 or T <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S * Phi(d1) - K * math.exp(-r * T) * Phi(d2)


def _merton_call_price(S, K, r, sigma, T, lam, mu_j, sigma_j):
    """Merton (1976) European call: Poisson-weighted BS sum in log space.

    Truncation: continue until cumulative Poisson weight exceeds 1 - 1e-12
    (and at least 20 terms have been summed).
    """
    if lam <= 0:
        return _bs_call(S, K, r, sigma, T)
    m_bar = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    lam_prime = lam * (1.0 + m_bar)
    lam_T = lam_prime * T
    log_lamT = math.log(lam_T)
    price = 0.0
    cum_w = 0.0
    for n in range(200):
        log_w = -lam_T + n * log_lamT - math.lgamma(n + 1)
        w = math.exp(log_w)
        var_n = sigma * sigma + n * sigma_j * sigma_j / T
        r_n = r - lam * m_bar + n * (mu_j + 0.5 * sigma_j * sigma_j) / T
        price += w * _bs_call(S, K, r_n, math.sqrt(var_n), T)
        cum_w += w
        if n >= 20 and cum_w > 1.0 - 1e-12:
            break
    return price


def _merton_put_price(S, K, r, sigma, T, lam, mu_j, sigma_j):
    # Put-call parity: C - P = S - K exp(-rT) (Merton compensator preserves it).
    return _merton_call_price(S, K, r, sigma, T, lam, mu_j, sigma_j) \
        - S + K * math.exp(-r * T)


def _mc_merton_call(S, K, r, sigma, T, lam, mu_j, sigma_j, steps, n_paths, seed=42):
    paths = stocha.merton_jump_diffusion(
        s0=S, mu=r, sigma=sigma, lambda_=lam, mu_j=mu_j, sigma_j=sigma_j,
        t=T, steps=steps, n_paths=n_paths, seed=seed,
    )
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)
    return float(math.exp(-r * T) * payoff.mean())


class TestMertonMjdEuropeanCall:
    """Accuracy audit: Merton MJD MC call price vs the Poisson-weighted BS sum.

    Tolerances are observation-based (see
    .artifacts/2026-05-12-accuracy-audit-merton-mjd/observations.md): measured
    max abs_err = 1.77e-01 and max rel_err = 3.5% at n_paths=200_000. With
    atol=2.5e-01 / rtol=4e-02 we keep ~1.4x margin over the worst case while
    retaining detection power for ≥5% coefficient bugs.

    Heavy-tailed log-normal payoffs make MC abs_err = 2–4 × SE typical; steps
    cannot fix this (Bernoulli bias < SE here), so n_paths must do the work.
    """

    ATOL = 2.5e-01
    RTOL = 4e-02
    N_PATHS = 200_000

    @pytest.mark.parametrize(
        "label,S,K,r,sigma,T,lam,mu_j,sigma_j",
        [
            ("standard",          100, 100, 0.05, 0.20, 1.0,   1.0, -0.05, 0.10),
            ("low_lam",           100, 100, 0.05, 0.20, 1.0,   0.5, -0.05, 0.10),
            ("high_lam",          100, 100, 0.05, 0.20, 1.0,   5.0, -0.05, 0.10),
            ("negative_mu_j_big", 100, 100, 0.05, 0.20, 1.0,   1.0, -0.20, 0.15),
            ("positive_mu_j",     100, 100, 0.05, 0.20, 1.0,   1.0,  0.10, 0.10),
            ("deep_otm_K140",     100, 140, 0.05, 0.20, 1.0,   1.0, -0.05, 0.10),
            ("itm_K70",           100,  70, 0.05, 0.20, 1.0,   1.0, -0.05, 0.10),
            ("short_T_0p25",      100, 100, 0.05, 0.20, 0.25,  1.0, -0.05, 0.10),
            ("long_T_2p0",        100, 100, 0.05, 0.20, 2.0,   1.0, -0.05, 0.10),
            ("high_sigma_j",      100, 100, 0.05, 0.20, 1.0,   1.0, -0.05, 0.30),
            ("crash_regime",      100, 100, 0.05, 0.15, 1.0,   2.0, -0.30, 0.20),
        ],
    )
    def test_mc_call_matches_analytic(
        self, label, S, K, r, sigma, T, lam, mu_j, sigma_j
    ):
        # Pick steps so lambda*dt <= 0.005 (Bernoulli approx accuracy floor).
        steps = max(252, int(math.ceil(lam * T / 0.005))) if lam > 0 else 252
        analytic = _merton_call_price(S, K, r, sigma, T, lam, mu_j, sigma_j)
        c_mc = _mc_merton_call(S, K, r, sigma, T, lam, mu_j, sigma_j, steps, self.N_PATHS)
        assert c_mc == pytest.approx(analytic, abs=self.ATOL, rel=self.RTOL), (
            f"[{label}] C_mc={c_mc:.6f}, C_analytic={analytic:.6f}, "
            f"abs_err={abs(c_mc - analytic):.3e}"
        )

    def test_lambda_zero_matches_black_scholes(self):
        # With no jumps, the Poisson-weighted sum collapses to a single n=0
        # term and must equal Black-Scholes exactly. This validates the
        # analytic formula coefficients independently of MC noise.
        S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        merton = _merton_call_price(S, K, r, sigma, T, 0.0, 0.0, 0.0)
        bs = _bs_call(S, K, r, sigma, T)
        assert merton == pytest.approx(bs, rel=1e-12)

    @pytest.mark.parametrize(
        "S,K,r,sigma,T,lam,mu_j,sigma_j",
        [
            (100, 100, 0.05, 0.20, 1.0, 1.0, -0.05, 0.10),
            (100, 140, 0.05, 0.20, 1.0, 1.0, -0.05, 0.10),
            (100,  70, 0.05, 0.20, 1.0, 1.0, -0.05, 0.10),
            (100, 100, 0.05, 0.15, 1.0, 2.0, -0.30, 0.20),
        ],
    )
    def test_put_call_parity_analytic(
        self, S, K, r, sigma, T, lam, mu_j, sigma_j
    ):
        # Compensator preserves risk-neutral parity: C - P = S - K exp(-rT).
        c = _merton_call_price(S, K, r, sigma, T, lam, mu_j, sigma_j)
        p = _merton_put_price(S, K, r, sigma, T, lam, mu_j, sigma_j)
        forward = S - K * math.exp(-r * T)
        assert (c - p) == pytest.approx(forward, abs=1e-12)


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
# Hull-White ZCB accuracy audit (Vasicek analytic vs MC)
# ---------------------------------------------------------------------------

def _vasicek_zcb_price(r0, a, theta, sigma, T):
    """Analytic ZCB P(0,T) for dr = (theta - a r) dt + sigma dW (theta const).

    Long-run mean b = theta / a; closed form is Vasicek.
        B(T) = (1 - exp(-a T)) / a
        A(T) = exp((b - sigma^2/(2 a^2))(B(T) - T) - sigma^2 B(T)^2 / (4 a))
        P    = A(T) exp(-B(T) r0)
    """
    B = (1.0 - math.exp(-a * T)) / a
    b = theta / a
    A = math.exp(
        (b - sigma * sigma / (2.0 * a * a)) * (B - T)
        - sigma * sigma * B * B / (4.0 * a)
    )
    return A * math.exp(-B * r0)


def _mc_zcb_price(r0, a, theta, sigma, T, steps, n_paths, seed=42):
    """MC ZCB via trapezoidal discount over Hull-White paths."""
    paths = stocha.hull_white(
        r0=r0, a=a, theta=theta, sigma=sigma,
        t=T, steps=steps, n_paths=n_paths, seed=seed,
    )
    times = np.linspace(0.0, T, steps + 1)
    integrals = np.trapezoid(paths, x=times, axis=1)
    discount = np.exp(-integrals)
    return float(discount.mean())


class TestHullWhiteZcb:
    """Accuracy audit: stocha.hull_white MC pricing of P(0,T) vs Vasicek closed form.

    Tolerances are observation-based (see
    .artifacts/2026-05-12-accuracy-audit-hull-white-zcb/observations.md):
    measured max abs_err = 7.0e-04, max rel_err = 8.2e-04 at n_paths=50k.
    Tolerances atol=1.5e-03 / rtol=2e-03 give roughly 2x margin while keeping
    coefficient-bug detection power.
    """

    # MC vs analytic Vasicek tolerances (observation-based, ~2x margin).
    ATOL = 1.5e-03
    RTOL = 2e-03

    # Standard simulation budget for the audit: n_paths=50k, dt ≈ 0.02.
    N_PATHS = 50_000

    @pytest.mark.parametrize(
        "label,r0,a,theta,sigma,T",
        [
            ("standard_1y",    0.03,  0.10, 0.005,  0.010,  1.0),
            ("standard_5y",    0.03,  0.10, 0.005,  0.010,  5.0),
            ("standard_10y",   0.03,  0.10, 0.005,  0.010, 10.0),
            ("low_a",          0.03,  0.05, 0.0025, 0.010,  5.0),
            ("high_a",         0.03,  0.50, 0.025,  0.010,  5.0),
            ("low_vol",        0.03,  0.10, 0.005,  0.003,  5.0),
            ("high_vol",       0.03,  0.10, 0.005,  0.020,  5.0),
            ("negative_r0",   -0.01,  0.10, 0.005,  0.010,  5.0),
            ("r0_above_b",     0.08,  0.10, 0.002,  0.010,  5.0),
            ("r0_below_b",     0.01,  0.10, 0.008,  0.010,  5.0),
        ],
    )
    def test_zcb_mc_matches_analytic(self, label, r0, a, theta, sigma, T):
        steps = max(50, int(252 * T / 5.0))  # keep dt ≈ 0.02 across maturities
        analytic = _vasicek_zcb_price(r0, a, theta, sigma, T)
        p_mc = _mc_zcb_price(r0, a, theta, sigma, T, steps, self.N_PATHS)
        assert p_mc == pytest.approx(analytic, abs=self.ATOL, rel=self.RTOL), (
            f"[{label}] P_mc={p_mc:.6f}, P_analytic={analytic:.6f}, "
            f"abs_err={abs(p_mc - analytic):.3e}"
        )

    def test_zcb_sigma_zero_matches_deterministic_ode(self):
        # σ = 0 → r(t) = b + (r0 - b) exp(-a t); the closed form must equal
        # exp(-∫r dt) computed analytically (no MC noise). Strict tolerance
        # validates the Vasicek formula itself, independent of MC.
        r0, a, theta, sigma, T = 0.03, 0.10, 0.005, 0.0, 5.0
        b = theta / a
        int_r = b * T + (r0 - b) * (1.0 - math.exp(-a * T)) / a
        det_discount = math.exp(-int_r)
        analytic = _vasicek_zcb_price(r0, a, theta, sigma, T)
        assert analytic == pytest.approx(det_discount, rel=1e-12)


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
# SABR Calibration
# ---------------------------------------------------------------------------

class TestSabrCalibrate:
    def _smile(self, f, t, alpha, beta, rho, nu, shift=0.0, n=7, width=0.3):
        ks = f + (f + shift) * np.linspace(-width, width, n)
        vols = np.array([
            stocha.sabr_implied_vol(f=f, k=k, t=t, alpha=alpha, beta=beta,
                                    rho=rho, nu=nu, shift=shift)
            for k in ks
        ])
        return ks, vols

    def test_round_trip_lognormal(self):
        # β=1: synthetic smile → recover (α, ρ, ν).
        f, t = 0.05, 1.0
        ks, vols = self._smile(f, t, alpha=0.20, beta=1.0, rho=-0.3, nu=0.4)
        r = stocha.sabr_calibrate(ks, vols, f=f, t=t, beta=1.0)
        assert r["converged"]
        assert r["rmse"] < 1e-6
        assert abs(r["alpha"] - 0.20) / 0.20 < 0.01
        assert abs(r["rho"] - (-0.3)) < 0.02
        assert abs(r["nu"] - 0.4) < 0.02

    def test_round_trip_beta_half(self):
        f, t = 0.05, 1.0
        ks, vols = self._smile(f, t, alpha=0.025, beta=0.5, rho=-0.5, nu=0.5)
        r = stocha.sabr_calibrate(ks, vols, f=f, t=t, beta=0.5)
        assert r["converged"]
        assert r["rmse"] < 1e-6
        assert abs(r["alpha"] - 0.025) / 0.025 < 0.01
        assert abs(r["rho"] - (-0.5)) < 0.02
        assert abs(r["nu"] - 0.5) < 0.02

    def test_round_trip_shifted(self):
        f, t, sh = -0.005, 1.0, 0.03
        ks, vols = self._smile(f, t, alpha=0.003, beta=0.5,
                               rho=-0.3, nu=0.3, shift=sh)
        r = stocha.sabr_calibrate(ks, vols, f=f, t=t, beta=0.5, shift=sh)
        assert r["converged"]
        assert r["rmse"] < 1e-5
        assert abs(r["rho"] - (-0.3)) < 0.05
        assert abs(r["nu"] - 0.3) < 0.05

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            stocha.sabr_calibrate(np.array([0.04, 0.05, 0.06]),
                                  np.array([0.2, 0.2]),
                                  f=0.05, t=1.0)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            stocha.sabr_calibrate(np.array([0.04, 0.05]),
                                  np.array([0.2, 0.2]),
                                  f=0.05, t=1.0)

    def test_nan_input(self):
        with pytest.raises(ValueError):
            stocha.sabr_calibrate(np.array([0.04, 0.05, 0.06]),
                                  np.array([0.2, np.nan, 0.18]),
                                  f=0.05, t=1.0)

    def test_shift_invalid(self):
        # F + shift = 0 → invalid.
        with pytest.raises(ValueError):
            stocha.sabr_calibrate(np.array([0.01, 0.02, 0.03]),
                                  np.array([0.2, 0.2, 0.2]),
                                  f=-0.03, t=1.0, shift=0.03)

    def test_deterministic(self):
        f, t = 0.05, 1.0
        ks, vols = self._smile(f, t, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
        r1 = stocha.sabr_calibrate(ks, vols, f=f, t=t, beta=0.5)
        r2 = stocha.sabr_calibrate(ks, vols, f=f, t=t, beta=0.5)
        assert r1 == r2


# ---------------------------------------------------------------------------
# LSMC American Option
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-Asset GBM
# ---------------------------------------------------------------------------

class TestMultiGBM:
    def _corr2(self):
        return np.array([[1.0, 0.6], [0.6, 1.0]])

    def _corr3(self):
        return np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])

    def test_output_shape(self):
        paths = stocha.multi_gbm(
            s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
            corr=self._corr2(), t=1.0, steps=252, n_paths=1000, seed=42,
        )
        assert paths.shape == (1000, 253, 2)
        assert paths.dtype == np.float64

    def test_three_assets(self):
        paths = stocha.multi_gbm(
            s0=[100.0, 50.0, 200.0], mu=[0.05, 0.08, 0.03],
            sigma=[0.2, 0.3, 0.15], corr=self._corr3(),
            t=1.0, steps=252, n_paths=500, seed=0,
        )
        assert paths.shape == (500, 253, 3)

    def test_initial_prices(self):
        s0 = [100.0, 50.0, 200.0]
        paths = stocha.multi_gbm(
            s0=s0, mu=[0.05, 0.08, 0.03], sigma=[0.2, 0.3, 0.15],
            corr=self._corr3(), t=1.0, steps=10, n_paths=100, seed=0,
        )
        for i, s in enumerate(s0):
            np.testing.assert_array_equal(paths[:, 0, i], s)

    def test_positive_prices(self):
        paths = stocha.multi_gbm(
            s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
            corr=self._corr2(), t=1.0, steps=252, n_paths=10_000, seed=0,
        )
        assert (paths > 0).all()

    def test_reproducibility(self):
        kw = dict(s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
                  corr=self._corr2(), t=1.0, steps=252, n_paths=1000, seed=42)
        np.testing.assert_array_equal(stocha.multi_gbm(**kw), stocha.multi_gbm(**kw))

    def test_expected_terminal_prices(self):
        s0 = [100.0, 50.0]
        mu = [0.05, 0.08]
        paths = stocha.multi_gbm(
            s0=s0, mu=mu, sigma=[0.2, 0.3],
            corr=self._corr2(), t=1.0, steps=252, n_paths=100_000, seed=0,
        )
        for i in range(2):
            mean_terminal = paths[:, -1, i].mean()
            expected = s0[i] * math.exp(mu[i])
            rel_err = abs(mean_terminal - expected) / expected
            assert rel_err < 0.02, f"asset {i}: rel_err={rel_err:.4f}"

    def test_correlation_structure(self):
        paths = stocha.multi_gbm(
            s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
            corr=self._corr2(), t=1.0, steps=252, n_paths=50_000, seed=0,
        )
        # Log-returns across all steps
        log_ret = np.log(paths[:, 1:, :] / paths[:, :-1, :])
        # Flatten to (n_paths * steps, n_assets) for correlation estimation
        flat = log_ret.reshape(-1, 2)
        sample_corr = np.corrcoef(flat.T)[0, 1]
        assert abs(sample_corr - 0.6) < 0.05, f"sample_corr={sample_corr:.4f}"

    def test_antithetic_shape(self):
        paths = stocha.multi_gbm(
            s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
            corr=self._corr2(), t=1.0, steps=252, n_paths=1000,
            seed=42, antithetic=True,
        )
        assert paths.shape == (1000, 253, 2)

    def test_antithetic_reduces_variance(self):
        kw = dict(s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
                  corr=self._corr2(), t=1.0, steps=252, n_paths=50_000, seed=42)
        plain = stocha.multi_gbm(**kw, antithetic=False)
        anti = stocha.multi_gbm(**kw, antithetic=True)
        # Portfolio value: equal weight
        plain_port = plain[:, -1, :].sum(axis=1)
        anti_port = anti[:, -1, :].sum(axis=1)
        assert anti_port.std() < plain_port.std()

    def test_uncorrelated_assets(self):
        corr = np.eye(2)
        paths = stocha.multi_gbm(
            s0=[100.0, 100.0], mu=[0.05, 0.05], sigma=[0.2, 0.2],
            corr=corr, t=1.0, steps=252, n_paths=50_000, seed=0,
        )
        log_ret = np.log(paths[:, 1:, :] / paths[:, :-1, :])
        flat = log_ret.reshape(-1, 2)
        sample_corr = np.corrcoef(flat.T)[0, 1]
        assert abs(sample_corr) < 0.02, f"sample_corr={sample_corr:.4f}"

    def test_invalid_length_mismatch(self):
        with pytest.raises((ValueError, Exception)):
            stocha.multi_gbm(s0=[100.0], mu=[0.05, 0.08], sigma=[0.2],
                              corr=np.eye(1), t=1.0, steps=10, n_paths=10)

    def test_invalid_s0_negative(self):
        with pytest.raises((ValueError, Exception)):
            stocha.multi_gbm(s0=[-1.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
                              corr=self._corr2(), t=1.0, steps=10, n_paths=10)

    def test_invalid_corr_not_pd(self):
        bad_corr = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises((ValueError, Exception)):
            stocha.multi_gbm(s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
                              corr=bad_corr, t=1.0, steps=10, n_paths=10)

    def test_invalid_corr_shape(self):
        with pytest.raises((ValueError, Exception)):
            stocha.multi_gbm(s0=[100.0, 50.0], mu=[0.05, 0.08], sigma=[0.2, 0.3],
                              corr=np.eye(3), t=1.0, steps=10, n_paths=10)


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


# ---------------------------------------------------------------------------
# Heston COS Pricing
# ---------------------------------------------------------------------------

class TestHestonPrice:
    """Tests for heston_price (COS method analytical pricing)."""

    PARAMS = dict(s0=100.0, v0=0.04, r=0.05,
                  kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, t=1.0)

    def test_atm_call_reasonable(self):
        prices = stocha.heston_price(
            strikes=np.array([100.0]), is_call=[True], **self.PARAMS,
        )
        assert 5.0 < prices[0] < 20.0, f"ATM call={prices[0]:.4f}"

    def test_call_monotone_decreasing(self):
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
        prices = stocha.heston_price(
            strikes=strikes, is_call=[True]*7, **self.PARAMS,
        )
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i+1], \
                f"K={strikes[i]}: {prices[i]:.4f} < K={strikes[i+1]}: {prices[i+1]:.4f}"

    def test_put_monotone_increasing(self):
        strikes = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
        prices = stocha.heston_price(
            strikes=strikes, is_call=[False]*7, **self.PARAMS,
        )
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i+1], \
                f"K={strikes[i]}: {prices[i]:.4f} > K={strikes[i+1]}: {prices[i+1]:.4f}"

    def test_put_call_parity(self):
        strikes = np.array([90.0, 100.0, 110.0])
        calls = stocha.heston_price(
            strikes=strikes, is_call=[True]*3, **self.PARAMS,
        )
        puts = stocha.heston_price(
            strikes=strikes, is_call=[False]*3, **self.PARAMS,
        )
        s0 = self.PARAMS["s0"]
        r = self.PARAMS["r"]
        t = self.PARAMS["t"]
        parity = calls - puts - s0 + strikes * np.exp(-r * t)
        np.testing.assert_allclose(parity, 0.0, atol=1e-6)

    def test_bs_limit(self):
        """When xi→0 and rho=0, Heston→BS with sigma=sqrt(v0)."""
        prices = stocha.heston_price(
            strikes=np.array([100.0]), is_call=[True],
            s0=100.0, v0=0.04, r=0.05,
            kappa=2.0, theta=0.04, xi=1e-6, rho=0.0, t=1.0,
            n_cos=256,
        )
        sigma = 0.2  # sqrt(0.04)
        d1 = (math.log(100/100) + (0.05 + 0.5*sigma**2)) / sigma
        d2 = d1 - sigma
        ncdf = lambda x: 0.5*(1 + math.erf(x / math.sqrt(2)))
        bs_call = 100*ncdf(d1) - 100*math.exp(-0.05)*ncdf(d2)
        rel_err = abs(prices[0] - bs_call) / bs_call
        assert rel_err < 1e-3, f"COS={prices[0]:.6f}, BS={bs_call:.6f}, err={rel_err:.2e}"

    def test_positive_prices(self):
        for k in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
            for call in [True, False]:
                p = stocha.heston_price(
                    strikes=np.array([k]), is_call=[call], **self.PARAMS,
                )
                assert p[0] >= 0, f"Negative price K={k} call={call}: {p[0]}"

    def test_n_cos_convergence(self):
        """Increasing N should converge."""
        results = []
        for n in [64, 128, 256]:
            p = stocha.heston_price(
                strikes=np.array([100.0]), is_call=[True],
                **self.PARAMS, n_cos=n,
            )
            results.append(p[0])
        # Difference between N=128 and N=256 should be much smaller
        # than between N=64 and N=128.
        diff_low = abs(results[1] - results[0])
        diff_high = abs(results[2] - results[1])
        assert diff_high < diff_low, \
            f"Not converging: Δ(64,128)={diff_low:.2e}, Δ(128,256)={diff_high:.2e}"

    def test_reproducibility(self):
        kw = dict(strikes=np.array([100.0]), is_call=[True], **self.PARAMS)
        np.testing.assert_array_equal(
            stocha.heston_price(**kw), stocha.heston_price(**kw))

    def test_invalid_v0(self):
        with pytest.raises(ValueError):
            stocha.heston_price(strikes=np.array([100.0]), is_call=[True],
                                s0=100.0, v0=-0.01, r=0.05,
                                kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, t=1.0)

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            stocha.heston_price(strikes=np.array([100.0]), is_call=[True],
                                s0=100.0, v0=0.04, r=0.05,
                                kappa=2.0, theta=0.04, xi=0.3, rho=-1.5, t=1.0)


# ---------------------------------------------------------------------------
# Heston Calibration
# ---------------------------------------------------------------------------

class TestHestonCalibrate:
    """Tests for heston_calibrate (Projected LM with COS repricing)."""

    def _synthetic(self, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                   s0=100.0, r=0.05, t=1.0, n_strikes=7):
        strikes = np.linspace(85, 115, n_strikes)
        is_call = [True] * n_strikes
        prices = stocha.heston_price(
            strikes=strikes, is_call=is_call,
            s0=s0, v0=v0, r=r,
            kappa=kappa, theta=theta, xi=xi, rho=rho, t=t,
            n_cos=256,
        )
        return strikes, np.full(n_strikes, t), prices, is_call

    def test_round_trip(self):
        true = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        strikes, mats, prices, is_call = self._synthetic(**true)
        result = stocha.heston_calibrate(
            strikes=strikes, maturities=mats,
            market_prices=prices, is_call=is_call,
            s0=100.0, r=0.05,
        )
        assert result["converged"]
        assert result["rmse"] < 1e-4
        for param, val in true.items():
            assert abs(result[param] - val) / max(abs(val), 1e-6) < 0.05, \
                f"{param}: {result[param]:.6f} vs {val}"

    def test_high_vol_of_vol(self):
        true = dict(v0=0.06, kappa=1.5, theta=0.06, xi=0.5, rho=-0.8)
        strikes, mats, prices, is_call = self._synthetic(**true)
        result = stocha.heston_calibrate(
            strikes=strikes, maturities=mats,
            market_prices=prices, is_call=is_call,
            s0=100.0, r=0.05, max_iter=300,
        )
        assert result["rmse"] < 0.01
        assert abs(result["rho"] - (-0.8)) < 0.1

    def test_multi_maturity(self):
        true = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        s0, r = 100.0, 0.05
        strikes = np.array([90, 95, 100, 105, 110, 90, 100, 110], dtype=float)
        mats = np.array([0.5]*5 + [1.0]*3)
        is_call = [True]*8
        prices = []
        for i in range(len(strikes)):
            p = stocha.heston_price(
                strikes=np.array([strikes[i]]), is_call=[True],
                s0=s0, r=r, t=mats[i], n_cos=256, **true,
            )
            prices.append(p[0])
        prices = np.array(prices)
        result = stocha.heston_calibrate(
            strikes=strikes, maturities=mats,
            market_prices=prices, is_call=is_call,
            s0=s0, r=r,
        )
        assert result["rmse"] < 0.01

    def test_feller_flag(self):
        # Params that violate Feller: 2*kappa*theta < xi^2
        true = dict(v0=0.04, kappa=0.5, theta=0.04, xi=0.8, rho=-0.5)
        strikes, mats, prices, is_call = self._synthetic(**true)
        result = stocha.heston_calibrate(
            strikes=strikes, maturities=mats,
            market_prices=prices, is_call=is_call,
            s0=100.0, r=0.05, max_iter=300,
        )
        assert not result["feller_satisfied"]

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            stocha.heston_calibrate(
                strikes=np.array([100.0, 105.0]),
                maturities=np.array([1.0]),
                market_prices=np.array([10.0, 8.0]),
                is_call=[True, True],
                s0=100.0, r=0.05,
            )

    def test_too_few_observations(self):
        with pytest.raises(ValueError):
            stocha.heston_calibrate(
                strikes=np.array([100.0]),
                maturities=np.array([1.0]),
                market_prices=np.array([10.0]),
                is_call=[True],
                s0=100.0, r=0.05,
            )

    def test_deterministic(self):
        strikes, mats, prices, is_call = self._synthetic()
        r1 = stocha.heston_calibrate(
            strikes=strikes, maturities=mats,
            market_prices=prices, is_call=is_call,
            s0=100.0, r=0.05,
        )
        r2 = stocha.heston_calibrate(
            strikes=strikes, maturities=mats,
            market_prices=prices, is_call=is_call,
            s0=100.0, r=0.05,
        )
        assert r1 == r2
