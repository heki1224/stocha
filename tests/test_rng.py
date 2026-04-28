"""
Tests for stocha random number generation: RNG, Sobol, Halton.

Covers: src/prng/, src/qrng/
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from scipy import stats
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
        assert abs(samples.mean()) < 0.01
        assert abs(samples.std() - 1.0) < 0.01

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

    def test_from_state_roundtrip(self):
        rng = stocha.RNG(seed=123)
        state = rng.save_state()
        restored = stocha.RNG.from_state(state)
        np.testing.assert_array_equal(
            restored.normal(size=50), stocha.RNG(seed=123).normal(size=50)
        )

    def test_full_state_mid_stream_roundtrip(self):
        rng = stocha.RNG(seed=42)
        _ = rng.normal(size=500)
        state = rng.save_state()
        expected = rng.normal(size=100)
        restored = stocha.RNG.from_state(state)
        np.testing.assert_array_equal(restored.normal(size=100), expected)

    def test_legacy_seed_only_format(self):
        restored = stocha.RNG.from_state('{"seed":42}')
        fresh = stocha.RNG(seed=42)
        np.testing.assert_array_equal(
            restored.normal(size=50), fresh.normal(size=50)
        )

    def test_from_state_invalid_json(self):
        with pytest.raises(Exception):
            stocha.RNG.from_state("not-valid-json")

    def test_rng_normal_invalid_scale(self):
        with pytest.raises((ValueError, Exception)):
            stocha.RNG(seed=0).normal(size=10, scale=-1.0)


# ---------------------------------------------------------------------------
# RNG.standard_normal
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
        assert abs(out.std() - 1.0) < 0.01

    def test_ks_normality(self):
        """Kolmogorov-Smirnov test: Ziggurat output vs N(0,1)."""
        rng = stocha.RNG(seed=123)
        samples = rng.standard_normal(size=100_000)
        stat, p = stats.kstest(samples, "norm")
        assert p > 0.01, f"KS test failed: stat={stat:.6f}, p={p:.6f}"

    def test_reproducibility(self):
        a = stocha.RNG(seed=42).standard_normal(size=1_000)
        b = stocha.RNG(seed=42).standard_normal(size=1_000)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# sobol function
# ---------------------------------------------------------------------------

class TestSobol:
    def test_shape(self):
        pts = stocha.sobol(dim=4, n_samples=256)
        assert pts.shape == (256, 4)
        assert pts.dtype == np.float64

    def test_values_in_unit_interval(self):
        pts = stocha.sobol(dim=3, n_samples=1000)
        assert (pts >= 0.0).all()
        assert (pts < 1.0).all()

    def test_deterministic(self):
        a = stocha.sobol(dim=2, n_samples=64)
        b = stocha.sobol(dim=2, n_samples=64)
        np.testing.assert_array_equal(a, b)

    def test_low_discrepancy_uniformity(self):
        pts = stocha.sobol(dim=3, n_samples=1024)
        for d in range(3):
            assert abs(pts[:, d].mean() - 0.5) < 0.05

    def test_invalid_dim(self):
        with pytest.raises((ValueError, Exception)):
            stocha.sobol(dim=0, n_samples=10)


# ---------------------------------------------------------------------------
# halton function
# ---------------------------------------------------------------------------

class TestHalton:
    def test_shape(self):
        pts = stocha.halton(dim=3, n_samples=500)
        assert pts.shape == (500, 3)
        assert pts.dtype == np.float64

    def test_values_in_unit_interval(self):
        pts = stocha.halton(dim=4, n_samples=1000)
        assert (pts > 0.0).all()
        assert (pts < 1.0).all()

    def test_deterministic(self):
        a = stocha.halton(dim=2, n_samples=100)
        b = stocha.halton(dim=2, n_samples=100)
        np.testing.assert_array_equal(a, b)

    def test_skip_consistency(self):
        full = stocha.halton(dim=2, n_samples=20, skip=0)
        skipped = stocha.halton(dim=2, n_samples=10, skip=10)
        np.testing.assert_array_almost_equal(full[10:, :], skipped)

    def test_invalid_dim(self):
        with pytest.raises((ValueError, Exception)):
            stocha.halton(dim=0, n_samples=10)
