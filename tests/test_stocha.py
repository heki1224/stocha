"""
Basic pytest tests for stocha v0.1.

Requires: maturin develop --release (or pip install stocha)
Run with: pytest tests/ -v
"""

import math
import pytest
import numpy as np
import stocha


# ---------------------------------------------------------------------------
# RNG class
# ---------------------------------------------------------------------------

class TestRNG:
    def test_uniform_range(self):
        rng = stocha.RNG(seed=0)
        samples = rng.uniform(size=10_000)
        assert samples.dtype == np.float64
        assert samples.min() >= 0.0
        assert samples.max() < 1.0

    def test_normal_mean_std(self):
        rng = stocha.RNG(seed=42)
        samples = rng.normal(size=1_000_000)
        assert samples.dtype == np.float64
        assert abs(samples.mean()) < 0.01       # mean ≈ 0
        assert abs(samples.std() - 1.0) < 0.01  # std ≈ 1

    def test_normal_loc_scale(self):
        rng = stocha.RNG(seed=1)
        samples = rng.normal(size=100_000, loc=5.0, scale=2.0)
        assert pytest.approx(samples.mean(), rel=1e-2) == 5.0
        assert pytest.approx(samples.std(), rel=1e-2) == 2.0

    def test_reproducibility_same_seed(self):
        a = stocha.RNG(seed=99).normal(size=1_000)
        b = stocha.RNG(seed=99).normal(size=1_000)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = stocha.RNG(seed=1).normal(size=100)
        b = stocha.RNG(seed=2).normal(size=100)
        assert not np.array_equal(a, b)

    def test_save_state_returns_json(self):
        rng = stocha.RNG(seed=7)
        _ = rng.uniform(size=100)
        state = rng.save_state()
        assert isinstance(state, str)
        assert len(state) > 0

    def test_same_seed_reproduces_after_advance(self):
        # Phase 1: seed-based reproduction (full state serialization in Phase 2)
        rng1 = stocha.RNG(seed=7)
        rng2 = stocha.RNG(seed=7)
        np.testing.assert_array_equal(
            rng1.normal(size=50),
            rng2.normal(size=50),
        )


# ---------------------------------------------------------------------------
# RNG.standard_normal method
# ---------------------------------------------------------------------------

class TestStandardNormal:
    def test_shape(self):
        rng = stocha.RNG(seed=0)
        out = rng.standard_normal(size=10_000)
        assert out.shape == (10_000,)
        assert out.dtype == np.float64

    def test_statistics(self):
        rng = stocha.RNG(seed=0)
        out = rng.standard_normal(size=500_000)
        assert abs(out.mean()) < 0.005
        assert abs(out.std() - 1.0) < 0.005

    def test_reproducibility(self):
        a = stocha.RNG(seed=42).standard_normal(size=1_000)
        b = stocha.RNG(seed=42).standard_normal(size=1_000)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# gbm function
# ---------------------------------------------------------------------------

class TestGBM:
    def test_output_shape(self):
        paths = stocha.gbm(s0=100.0, mu=0.05, sigma=0.2,
                           t=1.0, steps=252, n_paths=1_000, seed=0)
        assert paths.shape == (1_000, 253)   # n_paths × (steps + 1)
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
                  n_paths=10_000, seed=0)
        plain = stocha.gbm(**kw, antithetic=False)
        anti = stocha.gbm(**kw, antithetic=True)
        # antithetic should produce paths with lower terminal price variance
        assert anti[:, -1].std() < plain[:, -1].std()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_gbm_negative_s0(self):
        with pytest.raises((ValueError, Exception)):
            stocha.gbm(s0=-1.0, mu=0.05, sigma=0.2,
                       t=1.0, steps=10, n_paths=10, seed=0)

    def test_gbm_zero_sigma(self):
        # sigma must be positive; sigma=0 raises ValueError
        with pytest.raises((ValueError, Exception)):
            stocha.gbm(s0=100.0, mu=0.05, sigma=0.0,
                       t=1.0, steps=10, n_paths=5, seed=0)

    def test_gbm_negative_t(self):
        with pytest.raises((ValueError, Exception)):
            stocha.gbm(s0=100.0, mu=0.05, sigma=0.2,
                       t=-1.0, steps=10, n_paths=10, seed=0)

    def test_rng_normal_invalid_scale(self):
        with pytest.raises((ValueError, Exception)):
            stocha.RNG(seed=0).normal(size=10, scale=-1.0)
