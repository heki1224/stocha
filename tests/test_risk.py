"""
Tests for stocha risk metrics and copulas: VaR/CVaR, Gaussian copula, Student-t copula.

Covers: src/risk/, src/copula/
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import stocha


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

class TestVarCvar:
    def test_output_types(self):
        returns = np.random.default_rng(0).normal(0, 0.01, 1000)
        var, cvar = stocha.var_cvar(returns, 0.95)
        assert isinstance(var, float)
        assert isinstance(cvar, float)

    def test_cvar_ge_var(self):
        returns = np.random.default_rng(1).normal(0, 0.01, 10000)
        var, cvar = stocha.var_cvar(returns, 0.95)
        assert cvar >= var

    def test_known_distribution(self):
        # For N(0, sigma), 95th-percentile loss ≈ 1.645 * sigma.
        sigma = 0.01
        returns = np.random.default_rng(42).normal(0, sigma, 100_000)
        var, cvar = stocha.var_cvar(returns, 0.95)
        expected_var = 1.645 * sigma
        assert abs(var - expected_var) < 0.002, f"VaR={var:.4f}, expected≈{expected_var:.4f}"

    def test_var_monotone_in_confidence(self):
        returns = np.random.default_rng(0).normal(0, 0.01, 10000)
        var90, _ = stocha.var_cvar(returns, 0.90)
        var95, _ = stocha.var_cvar(returns, 0.95)
        var99, _ = stocha.var_cvar(returns, 0.99)
        assert var90 <= var95 <= var99

    def test_invalid_confidence(self):
        returns = np.ones(100)
        with pytest.raises(Exception):
            stocha.var_cvar(returns, 1.0)
        with pytest.raises(Exception):
            stocha.var_cvar(returns, 0.0)

    def test_nan_input_raises(self):
        returns = np.array([0.01, float("nan"), -0.02])
        with pytest.raises(ValueError, match="NaN or Inf"):
            stocha.var_cvar(returns, 0.95)

    def test_inf_input_raises(self):
        returns = np.array([0.01, float("inf"), -0.02])
        with pytest.raises(ValueError, match="NaN or Inf"):
            stocha.var_cvar(returns, 0.95)


# ---------------------------------------------------------------------------
# Gaussian Copula
# ---------------------------------------------------------------------------

class TestGaussianCopula:
    def test_shape(self):
        corr = np.eye(3)
        u = stocha.gaussian_copula(corr, n_samples=500)
        assert u.shape == (500, 3)

    def test_values_in_unit_interval(self):
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        u = stocha.gaussian_copula(corr, n_samples=2000)
        assert np.all(u > 0) and np.all(u < 1)

    def test_correlation_preserved(self):
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        u = stocha.gaussian_copula(corr, n_samples=5000, seed=0)
        rho = np.corrcoef(u[:, 0], u[:, 1])[0, 1]
        assert rho > 0.7, f"Pearson rho={rho:.3f}"

    def test_not_positive_definite(self):
        corr = np.array([[1.0, 1.1], [1.1, 1.0]])
        with pytest.raises(Exception):
            stocha.gaussian_copula(corr, n_samples=100)

    def test_reproducibility(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        u1 = stocha.gaussian_copula(corr, n_samples=100, seed=7)
        u2 = stocha.gaussian_copula(corr, n_samples=100, seed=7)
        np.testing.assert_array_equal(u1, u2)


# ---------------------------------------------------------------------------
# Student-t Copula
# ---------------------------------------------------------------------------

class TestStudentTCopula:
    def test_shape(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        u = stocha.student_t_copula(corr, nu=5.0, n_samples=500)
        assert u.shape == (500, 2)

    def test_values_in_unit_interval(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        u = stocha.student_t_copula(corr, nu=5.0, n_samples=2000)
        assert np.all(u > 0) and np.all(u < 1)

    def test_nu_must_be_gt_2(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(Exception):
            stocha.student_t_copula(corr, nu=2.0, n_samples=100)

    def test_tail_dependence_heavier_than_gaussian(self):
        # Student-t copula (low nu) has heavier joint tails than Gaussian copula.
        # With rho=0.8 and nu=3, joint upper-tail exceedances should be
        # significantly more frequent than in the Gaussian copula.
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        n = 50_000
        threshold = 0.95

        u_t = stocha.student_t_copula(corr, nu=3.0, n_samples=n, seed=0)
        u_g = stocha.gaussian_copula(corr, n_samples=n, seed=0)

        joint_t = np.mean((u_t[:, 0] > threshold) & (u_t[:, 1] > threshold))
        joint_g = np.mean((u_g[:, 0] > threshold) & (u_g[:, 1] > threshold))

        assert joint_t > joint_g, \
            f"Student-t joint tail={joint_t:.4f} should exceed Gaussian={joint_g:.4f}"

    def test_reproducibility(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        u1 = stocha.student_t_copula(corr, nu=5.0, n_samples=100, seed=3)
        u2 = stocha.student_t_copula(corr, nu=5.0, n_samples=100, seed=3)
        np.testing.assert_array_equal(u1, u2)
