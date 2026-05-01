mod copula;
mod dist;
mod finance;
mod prng;
mod qrng;
mod risk;

use copula::{gaussian_copula_samples, student_t_copula_samples};
use dist::NormalSampler;
use finance::gbm::{gbm_paths, GbmParams};
use finance::greeks::{
    compute_bump, compute_greeks_from_prices, greeks_fd_core, greeks_pathwise_core,
    required_scenarios, BumpDir, Greek, ModelSpec, Payoff, ScenarioKey,
};
use finance::heston::{heston_paths_with_scheme, HestonParams, HestonScheme};
use finance::heston_calibration::calibrate as heston_calibrate_core;
use finance::heston_cos::heston_cos_price_vec;
use finance::hull_white::{hull_white_paths, HullWhiteParams};
use finance::jump_diffusion::{merton_paths, MertonParams};
use finance::lsmc::{lsmc_american_option as lsmc_price, LsmcParams};
use finance::multi_gbm::{multi_gbm_paths, MultiGbmParams};
use finance::sabr::sabr_implied_vol as sabr_vol;
use finance::sabr_calibration::calibrate as sabr_calibrate_core;
use ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use prng::Pcg64Dxsm;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use qrng::{halton_sequence, sobol_sequence};
use risk::var_cvar as compute_var_cvar;
use std::collections::HashMap;

/// Convert an owned `Array2<f64>` into a Python NumPy array.
fn into_py_array2<'py>(arr: Array2<f64>, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = [arr.shape()[0], arr.shape()[1]];
    let flat: Vec<f64> = arr.into_raw_vec_and_offset().0;
    let out = Array2::from_shape_vec(shape, flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

fn into_py_array3<'py>(arr: Array3<f64>, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let shape = [arr.shape()[0], arr.shape()[1], arr.shape()[2]];
    let flat: Vec<f64> = arr.into_raw_vec_and_offset().0;
    let out = Array3::from_shape_vec(shape, flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

/// Python-accessible random number generator backed by PCG64DXSM.
#[pyclass]
struct RNG {
    inner: Pcg64Dxsm,
    seed: u128,
}

#[pymethods]
impl RNG {
    /// Create a new RNG.
    ///
    /// Args:
    ///     seed:      Integer seed for reproducibility (default 0). Accepts up to 128-bit integers.
    ///     algorithm: Algorithm name. Currently only "pcg64dxsm" is supported.
    #[new]
    #[pyo3(signature = (seed=0, algorithm="pcg64dxsm"))]
    fn new(seed: u128, algorithm: &str) -> PyResult<Self> {
        match algorithm {
            "pcg64dxsm" | "default" => Ok(RNG {
                inner: Pcg64Dxsm::new(seed),
                seed,
            }),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown algorithm: '{}'. Supported: 'pcg64dxsm'",
                other
            ))),
        }
    }

    /// The seed used to initialize this RNG.
    #[getter]
    fn seed(&self) -> u128 {
        self.seed
    }

    /// Generate `size` samples from Uniform[0, 1) as a NumPy array.
    fn uniform<'py>(&mut self, py: Python<'py>, size: usize) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = py.detach(|| (0..size).map(|_| self.inner.next_f64()).collect());
        v.into_pyarray(py)
    }

    /// Generate `size` samples from N(0, 1) as a NumPy array (Marsaglia polar method).
    fn standard_normal<'py>(&mut self, py: Python<'py>, size: usize) -> Bound<'py, PyArray1<f64>> {
        let mut buf = vec![0.0f64; size];
        py.detach(|| NormalSampler::sample_into(&mut self.inner, &mut buf));
        buf.into_pyarray(py)
    }

    /// Generate `size` samples from N(loc, scale^2) as a NumPy array.
    ///
    /// Args:
    ///     size:  Number of samples.
    ///     loc:   Mean (default 0.0).
    ///     scale: Standard deviation, must be positive (default 1.0).
    #[pyo3(signature = (size, loc=0.0, scale=1.0))]
    fn normal<'py>(
        &mut self,
        py: Python<'py>,
        size: usize,
        loc: f64,
        scale: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if scale <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "scale must be positive",
            ));
        }
        let mut buf = vec![0.0f64; size];
        py.detach(|| {
            for v in buf.iter_mut() {
                *v = NormalSampler::sample_scaled(&mut self.inner, loc, scale);
            }
        });
        Ok(buf.into_pyarray(py))
    }

    /// Serialize the full RNG state to a JSON string.
    ///
    /// Captures the exact internal position of the generator, enabling
    /// mid-stream checkpointing. Restoring via ``from_state`` resumes
    /// the sequence from the saved position.
    ///
    /// Returns:
    ///     JSON string containing the full generator state.
    fn save_state(&self) -> String {
        self.inner.save_state()
    }

    /// Restore an RNG from a JSON string produced by :meth:`save_state`.
    ///
    /// Accepts both the full-state format (v1.2+) and the legacy seed-only
    /// format (``{"seed": N}``). Full-state restores the exact position;
    /// seed-only restarts from the beginning.
    ///
    /// Args:
    ///     json: JSON string as returned by :meth:`save_state`.
    ///
    /// Returns:
    ///     New ``RNG`` instance.
    #[staticmethod]
    fn from_state(json: &str) -> PyResult<RNG> {
        let inner =
            Pcg64Dxsm::from_state(json).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let seed = inner.seed();
        Ok(RNG { inner, seed })
    }

    fn __repr__(&self) -> String {
        format!("RNG(seed={}, algorithm='pcg64dxsm')", self.seed)
    }
}

/// Simulate Geometric Brownian Motion paths in parallel.
///
/// Uses Euler-Maruyama discretization with Rayon parallel path generation.
/// Each path receives an independent RNG stream (block splitting), ensuring
/// reproducibility regardless of thread count.
///
/// Args:
///     s0:         Initial asset price (must be > 0).
///     mu:         Drift rate (annualized).
///     sigma:      Volatility (annualized, must be > 0).
///     t:          Time to maturity in years (must be > 0).
///     steps:      Number of time steps.
///     n_paths:    Number of simulation paths.
///     seed:       Random seed (default 42).
///     antithetic: Use antithetic variates for variance reduction (default False).
///
/// Returns:
///     NumPy array of shape (n_paths, steps + 1).
///     Column 0 is the initial price; column `steps` is the terminal price.
#[pyfunction]
#[pyo3(signature = (s0, mu, sigma, t, steps, n_paths, seed=42, antithetic=false))]
fn gbm<'py>(
    py: Python<'py>,
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    n_paths: usize,
    seed: u64,
    antithetic: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if s0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0 must be positive",
        ));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sigma must be positive",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }

    let params = GbmParams {
        s0,
        mu,
        sigma,
        t,
        steps,
        n_paths,
        antithetic,
    };
    let result = py.detach(|| gbm_paths(&params, seed as u128));
    into_py_array2(result, py)
}

/// Generate a Sobol low-discrepancy sequence using Joe & Kuo 2008 direction numbers.
///
/// Args:
///     dim:      Number of dimensions (1–1000).
///     n_samples: Number of sample points to generate.
///
/// Returns:
///     NumPy array of shape ``(n_samples, dim)`` with values in [0, 1).
#[pyfunction]
fn sobol<'py>(
    py: Python<'py>,
    dim: usize,
    n_samples: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if dim == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dim must be at least 1",
        ));
    }
    let arr = py
        .detach(|| sobol_sequence(dim, n_samples))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    into_py_array2(arr, py)
}

/// Generate a Halton low-discrepancy sequence.
///
/// Args:
///     dim:      Number of dimensions (1–40).
///     n_samples: Number of sample points to generate.
///     skip:     Number of initial elements to skip (default 0).
///
/// Returns:
///     NumPy array of shape ``(n_samples, dim)`` with values in (0, 1).
#[pyfunction]
#[pyo3(name = "halton", signature = (dim, n_samples, skip=0))]
fn halton_seq<'py>(
    py: Python<'py>,
    dim: usize,
    n_samples: usize,
    skip: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if dim == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dim must be at least 1",
        ));
    }
    let arr = py
        .detach(|| halton_sequence(dim, n_samples, skip))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    into_py_array2(arr, py)
}

/// Simulate Heston stochastic volatility paths.
///
/// Args:
///     s0:     Initial asset price (must be > 0).
///     v0:     Initial variance (not volatility; e.g. 0.04 for 20% vol).
///     mu:     Drift rate (annualized).
///     kappa:  Mean-reversion speed of variance (must be > 0).
///     theta:  Long-run mean of variance (must be > 0).
///     xi:     Volatility of variance / vol-of-vol (must be > 0).
///     rho:    Correlation between asset and variance Brownians (in [-1, 1]).
///     t:      Time to maturity in years (must be > 0).
///     steps:  Number of time steps.
///     n_paths: Number of simulation paths.
///     seed:   Random seed (default 42).
///     scheme: Discretization scheme: ``"euler"`` (Full Truncation) or ``"qe"``
///             (Andersen 2008 Quadratic Exponential with martingale correction).
///             Default ``"euler"``.
///
/// Returns:
///     NumPy array of shape ``(n_paths, steps + 1)``.
#[pyfunction]
#[pyo3(signature = (s0, v0, mu, kappa, theta, xi, rho, t, steps, n_paths, seed=42, scheme="euler"))]
fn heston<'py>(
    py: Python<'py>,
    s0: f64,
    v0: f64,
    mu: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    t: f64,
    steps: usize,
    n_paths: usize,
    seed: u64,
    scheme: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if s0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0 must be positive",
        ));
    }
    if v0 < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "v0 must be non-negative",
        ));
    }
    if kappa <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "kappa must be positive",
        ));
    }
    if theta <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must be positive",
        ));
    }
    if xi <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xi must be positive",
        ));
    }
    if rho < -1.0 || rho > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rho must be in [-1, 1]",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }
    let heston_scheme = match scheme {
        "euler" => HestonScheme::Euler,
        "qe" => HestonScheme::QE,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "scheme must be 'euler' or 'qe'",
            ))
        }
    };

    let params = HestonParams {
        s0,
        v0,
        mu,
        kappa,
        theta,
        xi,
        rho,
        t,
        steps,
        n_paths,
    };
    let result = py.detach(|| heston_paths_with_scheme(&params, seed as u128, heston_scheme));
    into_py_array2(result, py)
}

/// Simulate Merton Jump-Diffusion paths.
///
/// Model:  dS = (mu - lambda*m_bar)*S*dt + sigma*S*dW + S*(J-1)*dN
/// where J = exp(mu_j + sigma_j * Z) is a lognormal jump size and
/// m_bar = exp(mu_j + 0.5*sigma_j^2) - 1 is the mean jump return.
///
/// Args:
///     s0:      Initial asset price (must be > 0).
///     mu:      Drift rate (annualized).
///     sigma:   Diffusion volatility (annualized, must be > 0).
///     lambda_: Jump intensity (average jumps per year, must be >= 0).
///     mu_j:    Mean of log-jump size.
///     sigma_j: Std dev of log-jump size (must be >= 0).
///     t:       Time to maturity in years (must be > 0).
///     steps:   Number of time steps.
///     n_paths: Number of simulation paths.
///     seed:    Random seed (default 42).
///
/// Returns:
///     NumPy array of shape ``(n_paths, steps + 1)``.
#[pyfunction]
#[pyo3(signature = (s0, mu, sigma, lambda_, mu_j, sigma_j, t, steps, n_paths, seed=42))]
fn merton_jump_diffusion<'py>(
    py: Python<'py>,
    s0: f64,
    mu: f64,
    sigma: f64,
    lambda_: f64,
    mu_j: f64,
    sigma_j: f64,
    t: f64,
    steps: usize,
    n_paths: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if s0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0 must be positive",
        ));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sigma must be positive",
        ));
    }
    if lambda_ < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lambda_ must be non-negative",
        ));
    }
    if sigma_j < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sigma_j must be non-negative",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }

    let params = MertonParams {
        s0,
        mu,
        sigma,
        lambda: lambda_,
        mu_j,
        sigma_j,
        t,
        steps,
        n_paths,
    };
    let result = py.detach(|| merton_paths(&params, seed as u128));
    into_py_array2(result, py)
}

/// Compute Value-at-Risk and Conditional VaR (Expected Shortfall) from a return series.
///
/// Losses are defined as *negative* returns. Both outputs are positive (loss magnitudes).
///
/// Args:
///     returns:    1-D NumPy array of portfolio returns.
///     confidence: Confidence level in (0, 1), e.g. ``0.95`` for 95% VaR.
///
/// Returns:
///     ``(var, cvar)`` tuple of floats.
///
/// Example:
///     >>> returns = paths[:, -1] / paths[:, 0] - 1
///     >>> var, cvar = stocha.var_cvar(returns, confidence=0.95)
#[pyfunction]
fn var_cvar<'py>(returns: PyReadonlyArray1<'py, f64>, confidence: f64) -> PyResult<(f64, f64)> {
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "confidence must be in (0, 1)",
        ));
    }
    let arr = returns.as_array();
    if arr.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "returns must not be empty",
        ));
    }
    let slice: Vec<f64> = arr.iter().copied().collect();
    compute_var_cvar(&slice, confidence).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Sample from a Gaussian copula with a given correlation matrix.
///
/// Args:
///     corr:      2-D NumPy array of shape ``(dim, dim)``, a valid correlation matrix.
///     n_samples: Number of samples to draw.
///     seed:      Random seed (default ``42``).
///
/// Returns:
///     NumPy array of shape ``(n_samples, dim)`` with values in ``(0, 1)``.
///
/// Example:
///     >>> import numpy as np
///     >>> corr = np.array([[1.0, 0.8], [0.8, 1.0]])
///     >>> u = stocha.gaussian_copula(corr, n_samples=1000)
#[pyfunction]
#[pyo3(signature = (corr, n_samples, seed=42))]
fn gaussian_copula<'py>(
    py: Python<'py>,
    corr: PyReadonlyArray2<'py, f64>,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let owned = corr.as_array().to_owned();
    let result = py
        .detach(|| gaussian_copula_samples(owned.view(), n_samples, seed as u128))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    into_py_array2(result, py)
}

/// Sample from a Student-t copula with a given correlation matrix.
///
/// Args:
///     corr:      2-D NumPy array of shape ``(dim, dim)``, a valid correlation matrix.
///     nu:        Degrees of freedom (must be > 2 for finite variance).
///     n_samples: Number of samples to draw.
///     seed:      Random seed (default ``42``).
///
/// Returns:
///     NumPy array of shape ``(n_samples, dim)`` with values in ``(0, 1)``.
#[pyfunction]
#[pyo3(signature = (corr, nu, n_samples, seed=42))]
fn student_t_copula<'py>(
    py: Python<'py>,
    corr: PyReadonlyArray2<'py, f64>,
    nu: f64,
    n_samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if nu <= 2.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("nu must be > 2"));
    }
    let owned = corr.as_array().to_owned();
    let result = py
        .detach(|| student_t_copula_samples(owned.view(), nu, n_samples, seed as u128))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    into_py_array2(result, py)
}

/// Simulate Hull-White 1-factor short-rate paths via Exact Simulation.
///
/// Model: dr = (theta - a*r)*dt + sigma*dW
///
/// Uses the exact Gaussian transition (zero discretization bias).
/// The long-run mean rate is ``theta / a``.
///
/// Args:
///     r0:      Initial short rate.
///     a:       Mean-reversion speed (must be > 0).
///     theta:   Product ``a * long_run_mean`` (i.e. the drift constant).
///     sigma:   Volatility of the short rate (must be > 0).
///     t:       Time horizon in years (must be > 0).
///     steps:   Number of time steps.
///     n_paths: Number of simulation paths.
///     seed:    Random seed (default ``42``).
///
/// Returns:
///     NumPy array of shape ``(n_paths, steps + 1)``.
///     Column 0 is the initial rate ``r0``.
///
/// Example:
///     >>> rates = stocha.hull_white(r0=0.05, a=0.1, theta=0.005, sigma=0.01,
///     ...                           t=1.0, steps=252, n_paths=10000)
#[pyfunction]
#[pyo3(signature = (r0, a, theta, sigma, t, steps, n_paths, seed=42))]
fn hull_white<'py>(
    py: Python<'py>,
    r0: f64,
    a: f64,
    theta: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    n_paths: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if a <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a must be positive",
        ));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sigma must be positive",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }

    let params = HullWhiteParams {
        r0,
        a,
        theta,
        sigma,
        t,
        steps,
        n_paths,
    };
    let result = py.detach(|| hull_white_paths(&params, seed as u128));
    into_py_array2(result, py)
}

/// Compute the Black implied volatility using the SABR model (Hagan et al. 2002).
///
/// Supports negative rates via the Shifted SABR approach (add ``shift`` to F and K).
///
/// Args:
///     f:     Forward price or rate.
///     k:     Strike price or rate.
///     t:     Time to expiry in years (must be > 0).
///     alpha: Initial volatility level (must be > 0).
///     beta:  CEV exponent in [0, 1].  ``beta=1`` → lognormal, ``beta=0`` → normal.
///     rho:   Correlation between forward and vol Brownians (in (-1, 1)).
///     nu:    Vol-of-vol (must be >= 0).
///     shift: Shift for negative-rate support (default ``0.0``).
///
/// Returns:
///     Black (lognormal) implied volatility as a float.
///
/// Example:
///     >>> iv = stocha.sabr_implied_vol(f=0.05, k=0.05, t=1.0,
///     ...                              alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
#[pyfunction]
#[pyo3(signature = (f, k, t, alpha, beta, rho, nu, shift=0.0))]
fn sabr_implied_vol<'py>(
    f: f64,
    k: f64,
    t: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    shift: f64,
) -> PyResult<f64> {
    sabr_vol(f, k, t, alpha, beta, rho, nu, shift)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

/// Calibrate SABR parameters (alpha, rho, nu) to an observed implied-vol smile.
///
/// Beta is held fixed (typically 0.5). The ATM alpha is recovered exactly by a
/// 1-D root-find on the Hagan ATM formula; (rho, nu) are then fit by a
/// Projected Levenberg-Marquardt loop with central-difference Jacobian.
///
/// Args:
///     strikes:      1-D NumPy array of strikes K_i (must straddle the forward).
///     market_vols:  1-D NumPy array of observed Black implied vols.
///     f:            Forward price or rate.
///     t:            Time to expiry in years (must be > 0).
///     beta:         CEV exponent in [0, 1] (default ``0.5``).
///     shift:        Shift for negative-rate support (default ``0.0``).
///     max_iter:     Maximum LM iterations (default ``100``).
///     tol:          Convergence tolerance (default ``1e-10``).
///
/// Returns:
///     Dict with keys ``alpha``, ``rho``, ``nu``, ``rmse``, ``iterations``,
///     ``converged``.
///
/// Example:
///     >>> import numpy as np
///     >>> strikes = np.array([0.04, 0.045, 0.05, 0.055, 0.06])
///     >>> vols = np.array([0.25, 0.22, 0.20, 0.19, 0.185])
///     >>> r = stocha.sabr_calibrate(strikes, vols, f=0.05, t=1.0, beta=0.5)
///     >>> r["alpha"], r["rho"], r["nu"]
#[pyfunction]
#[pyo3(signature = (strikes, market_vols, f, t, beta=0.5, shift=0.0, max_iter=100, tol=1e-10))]
fn sabr_calibrate<'py>(
    py: Python<'py>,
    strikes: PyReadonlyArray1<'py, f64>,
    market_vols: PyReadonlyArray1<'py, f64>,
    f: f64,
    t: f64,
    beta: f64,
    shift: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let strikes_vec: Vec<f64> = strikes.as_array().iter().copied().collect();
    let vols_vec: Vec<f64> = market_vols.as_array().iter().copied().collect();
    let result = sabr_calibrate_core(&strikes_vec, &vols_vec, f, t, beta, shift, max_iter, tol)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("alpha", result.alpha)?;
    dict.set_item("rho", result.rho)?;
    dict.set_item("nu", result.nu)?;
    dict.set_item("rmse", result.rmse)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict)
}

/// Simulate correlated multi-asset GBM paths.
///
/// Uses Cholesky decomposition of the correlation matrix to generate
/// correlated Brownian increments across assets.
///
/// Args:
///     s0:         List of initial asset prices (all must be > 0).
///     mu:         List of drift rates (annualized), one per asset.
///     sigma:      List of volatilities (annualized, all must be > 0).
///     corr:       2-D correlation matrix of shape ``(n_assets, n_assets)``.
///     t:          Time to maturity in years (must be > 0).
///     steps:      Number of time steps.
///     n_paths:    Number of simulation paths.
///     seed:       Random seed (default ``42``).
///     antithetic: Use antithetic variates (default ``False``).
///
/// Returns:
///     NumPy array of shape ``(n_paths, steps + 1, n_assets)``.
#[pyfunction]
#[pyo3(signature = (s0, mu, sigma, corr, t, steps, n_paths, seed=42, antithetic=false))]
fn multi_gbm<'py>(
    py: Python<'py>,
    s0: Vec<f64>,
    mu: Vec<f64>,
    sigma: Vec<f64>,
    corr: PyReadonlyArray2<'py, f64>,
    t: f64,
    steps: usize,
    n_paths: usize,
    seed: u64,
    antithetic: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let n = s0.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0 must not be empty",
        ));
    }
    if mu.len() != n || sigma.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0, mu, sigma must have the same length",
        ));
    }
    if s0.iter().any(|&v| v <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all s0 must be positive",
        ));
    }
    if sigma.iter().any(|&v| v <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "all sigma must be positive",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }
    let corr_shape = corr.shape();
    if corr_shape[0] != n || corr_shape[1] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "corr must be ({}, {}), got ({}, {})",
            n, n, corr_shape[0], corr_shape[1]
        )));
    }

    let corr_owned = corr.as_array().to_owned();
    let params = MultiGbmParams {
        s0,
        mu,
        sigma,
        corr: corr_owned,
        t,
        steps,
        n_paths,
        antithetic,
    };
    let result = py
        .detach(|| multi_gbm_paths(&params, seed as u128))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    into_py_array3(result, py)
}

/// Price an American option via Longstaff-Schwartz Monte Carlo (LSMC).
///
/// Simulates GBM paths under the risk-neutral measure, then uses backward
/// induction with least-squares regression (polynomial basis, QR solver)
/// to determine optimal early-exercise boundaries.
///
/// Args:
///     s0:         Initial asset price (must be > 0).
///     k:          Strike price (must be > 0).
///     r:          Risk-free rate (annualized).
///     sigma:      Volatility (annualized, must be > 0).
///     t:          Time to maturity in years (must be > 0).
///     steps:      Number of exercise opportunities (time steps).
///     n_paths:    Number of simulation paths.
///     is_put:     ``True`` for put, ``False`` for call (default ``True``).
///     poly_degree: Polynomial degree for basis functions (1–4, default ``3``).
///     seed:       Random seed (default ``42``).
///
/// Returns:
///     ``(price, std_err)`` tuple.
///
/// Example:
///     >>> price, err = stocha.lsmc_american_option(
///     ...     s0=100.0, k=100.0, r=0.05, sigma=0.20, t=1.0,
///     ...     steps=50, n_paths=50000)
#[pyfunction]
#[pyo3(signature = (s0, k, r, sigma, t, steps, n_paths, is_put=true, poly_degree=3, seed=42))]
fn lsmc_american_option<'py>(
    py: Python<'py>,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    steps: usize,
    n_paths: usize,
    is_put: bool,
    poly_degree: usize,
    seed: u64,
) -> PyResult<(f64, f64)> {
    if s0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0 must be positive",
        ));
    }
    if k <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must be positive",
        ));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sigma must be positive",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }
    if n_paths < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_paths must be at least 2",
        ));
    }
    if poly_degree == 0 || poly_degree > 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "poly_degree must be in [1, 4]",
        ));
    }

    let params = LsmcParams {
        s0,
        k,
        r,
        sigma,
        t,
        steps,
        n_paths,
        is_put,
        poly_degree,
    };
    let (price, std_err) = py.detach(|| lsmc_price(&params, seed as u128));
    Ok((price, std_err))
}

fn parse_greeks(greeks: &Bound<'_, PyList>) -> PyResult<Vec<Greek>> {
    greeks
        .iter()
        .map(|item| {
            let s: String = item.extract()?;
            match s.as_str() {
                "delta" => Ok(Greek::Delta),
                "gamma" => Ok(Greek::Gamma),
                "vega" => Ok(Greek::Vega),
                "theta" => Ok(Greek::Theta),
                "rho" => Ok(Greek::Rho),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown greek: '{}'. Supported: delta, gamma, vega, theta, rho",
                    other
                ))),
            }
        })
        .collect()
}

fn greek_name(g: Greek) -> &'static str {
    match g {
        Greek::Delta => "delta",
        Greek::Gamma => "gamma",
        Greek::Vega => "vega",
        Greek::Theta => "theta",
        Greek::Rho => "rho",
    }
}

fn extract_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    dict.get_item(key)?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Missing required param: '{}'", key))
        })?
        .extract::<f64>()
}

fn build_model_spec(
    model: &str,
    params: &Bound<'_, PyDict>,
    n_paths: usize,
    n_steps: usize,
) -> PyResult<ModelSpec> {
    match model {
        "gbm" => {
            let s0 = extract_f64(params, "s0")?;
            let r = extract_f64(params, "r")?;
            let sigma = extract_f64(params, "sigma")?;
            let t = extract_f64(params, "t")?;
            Ok(ModelSpec::Gbm {
                s0,
                mu: r,
                sigma,
                t,
                steps: n_steps,
                n_paths,
            })
        }
        "heston" => {
            let s0 = extract_f64(params, "s0")?;
            let v0 = extract_f64(params, "v0")?;
            let r = extract_f64(params, "r")?;
            let kappa = extract_f64(params, "kappa")?;
            let theta = extract_f64(params, "theta")?;
            let xi = extract_f64(params, "xi")?;
            let rho = extract_f64(params, "rho")?;
            let t = extract_f64(params, "t")?;
            let scheme_str: String = params
                .get_item("scheme")?
                .map(|v| v.extract())
                .unwrap_or(Ok("euler".to_string()))?;
            let scheme = match scheme_str.as_str() {
                "euler" => HestonScheme::Euler,
                "qe" => HestonScheme::QE,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "scheme must be 'euler' or 'qe'",
                    ))
                }
            };
            Ok(ModelSpec::Heston {
                s0,
                v0,
                mu: r,
                kappa,
                theta,
                xi,
                rho,
                t,
                steps: n_steps,
                n_paths,
                scheme,
            })
        }
        "merton" => {
            let s0 = extract_f64(params, "s0")?;
            let r = extract_f64(params, "r")?;
            let sigma = extract_f64(params, "sigma")?;
            let lambda = extract_f64(params, "lambda_")?;
            let mu_j = extract_f64(params, "mu_j")?;
            let sigma_j = extract_f64(params, "sigma_j")?;
            let t = extract_f64(params, "t")?;
            Ok(ModelSpec::Merton {
                s0,
                mu: r,
                sigma,
                lambda,
                mu_j,
                sigma_j,
                t,
                steps: n_steps,
                n_paths,
            })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown model: '{}'. Supported: gbm, heston, merton",
            other
        ))),
    }
}

/// Compute Monte Carlo Greeks via bump-and-revalue (finite difference).
///
/// Supports built-in payoffs ("call" / "put") and custom Python callables.
/// All bump scenarios use the same random seed (CRN) for variance reduction.
///
/// Args:
///     model:     Model name: ``"gbm"``, ``"heston"``, or ``"merton"``.
///     params:    Dict of model parameters. GBM: ``s0, r, sigma, t``.
///                Heston: ``s0, v0, r, kappa, theta, xi, rho, t``.
///                Merton: ``s0, r, sigma, lambda_, mu_j, sigma_j, t``.
///     payoff:    ``"call"``, ``"put"``, or a Python callable ``f(terminals) -> values``.
///     strike:    Strike price (required for built-in payoffs, ignored for callable).
///     n_paths:   Number of simulation paths.
///     n_steps:   Number of time steps.
///     greeks:    List of Greeks to compute: ``"delta"``, ``"gamma"``, ``"vega"``,
///                ``"theta"``, ``"rho"``.
///     seed:      Random seed (default ``42``).
///     bump_size: Relative bump size (default ``0.01`` = 1%). Absolute bump is
///                ``max(1e-8, |param| * bump_size)``.
///
/// Returns:
///     Dict mapping Greek names to float values.
#[pyfunction]
#[pyo3(signature = (model, params, payoff, strike, n_paths, n_steps, greeks, seed=42, bump_size=0.01))]
fn greeks_fd<'py>(
    py: Python<'py>,
    model: &str,
    params: &Bound<'py, PyDict>,
    payoff: Py<PyAny>,
    strike: f64,
    n_paths: usize,
    n_steps: usize,
    greeks: &Bound<'py, PyList>,
    seed: u64,
    bump_size: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let greek_list = parse_greeks(greeks)?;
    let model_spec = build_model_spec(model, params, n_paths, n_steps)?;

    if let Ok(payoff_str) = payoff.extract::<String>(py) {
        let payoff_enum = match payoff_str.as_str() {
            "call" => Payoff::Call { strike },
            "put" => Payoff::Put { strike },
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown payoff: '{}'. Supported: 'call', 'put', or a callable",
                    other
                )))
            }
        };
        let results = py.detach(|| {
            greeks_fd_core(
                &model_spec,
                &payoff_enum,
                &greek_list,
                bump_size,
                seed as u128,
            )
        });
        let dict = PyDict::new(py);
        for (g, v) in results {
            dict.set_item(greek_name(g), v)?;
        }
        Ok(dict)
    } else if payoff.bind(py).is_callable() {
        let scenarios = required_scenarios(&greek_list);
        let mut bumps = HashMap::new();
        for &key in &scenarios {
            if let ScenarioKey::Bumped(param, _) = key {
                bumps
                    .entry(param)
                    .or_insert_with(|| compute_bump(model_spec.param_value(param), bump_size));
            }
        }

        let mut prices = HashMap::new();
        let seed128 = seed as u128;
        for &key in &scenarios {
            let m = match key {
                ScenarioKey::Base => model_spec.clone(),
                ScenarioKey::Bumped(param, dir) => {
                    let h = bumps[&param];
                    let delta = match dir {
                        BumpDir::Up => h,
                        BumpDir::Down => -h,
                    };
                    model_spec.with_bumped(param, delta)
                }
            };

            let r_rate = m.risk_free_rate();
            let mat = m.maturity();
            let terminals = py.detach(|| m.terminal_prices(seed128));

            let terminals_np = terminals.into_pyarray(py);
            let payoff_result = payoff.call1(py, (terminals_np,))?;
            let payoff_values: Vec<f64> = payoff_result.extract(py)?;

            let discount = (-r_rate * mat).exp();
            let price = discount * payoff_values.iter().sum::<f64>() / payoff_values.len() as f64;
            prices.insert(key, price);
        }

        let results = compute_greeks_from_prices(&greek_list, &prices, &bumps);
        let dict = PyDict::new(py);
        for (g, v) in results {
            dict.set_item(greek_name(g), v)?;
        }
        Ok(dict)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "payoff must be 'call', 'put', or a callable",
        ))
    }
}

/// Compute Monte Carlo Greeks via pathwise (IPA) method (GBM only).
///
/// More accurate than bump-and-revalue for continuous payoffs (European call/put).
/// Only requires a single simulation run. Supports Delta and Vega.
///
/// Args:
///     s0:      Initial asset price (must be > 0).
///     r:       Risk-free rate (annualized).
///     sigma:   Volatility (annualized, must be > 0).
///     t:       Time to maturity in years (must be > 0).
///     strike:  Strike price.
///     is_call: ``True`` for call, ``False`` for put.
///     n_paths: Number of simulation paths.
///     n_steps: Number of time steps.
///     greeks:  List of Greeks: ``"delta"`` and/or ``"vega"``.
///     seed:    Random seed (default ``42``).
///
/// Returns:
///     Dict mapping Greek names to float values.
#[pyfunction]
#[pyo3(signature = (s0, r, sigma, t, strike, is_call, n_paths, n_steps, greeks, seed=42))]
fn greeks_pathwise<'py>(
    py: Python<'py>,
    s0: f64,
    r: f64,
    sigma: f64,
    t: f64,
    strike: f64,
    is_call: bool,
    n_paths: usize,
    n_steps: usize,
    greeks: &Bound<'py, PyList>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    if s0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0 must be positive",
        ));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sigma must be positive",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if n_paths == 0 || n_steps == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_paths and n_steps must be positive",
        ));
    }

    let greek_list = parse_greeks(greeks)?;
    for &g in &greek_list {
        if g != Greek::Delta && g != Greek::Vega {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Pathwise method only supports 'delta' and 'vega', got '{}'",
                greek_name(g)
            )));
        }
    }

    let results = py.detach(|| {
        greeks_pathwise_core(
            s0,
            r,
            sigma,
            t,
            strike,
            is_call,
            n_paths,
            n_steps,
            &greek_list,
            seed as u128,
        )
    });
    let dict = PyDict::new(py);
    for (g, v) in results {
        dict.set_item(greek_name(g), v)?;
    }
    Ok(dict)
}

/// Price European options under the Heston model using the COS method.
///
/// Uses the Fang & Oosterlee (2008) COS expansion with the Albrecher (2007)
/// characteristic function formulation (branch-cut safe).
///
/// Args:
///     strikes:  1-D array of strike prices.
///     is_call:  1-D array of booleans (True for call, False for put).
///     s0:       Current spot price (must be > 0).
///     v0:       Initial variance (must be > 0).
///     r:        Risk-free rate (annualized).
///     kappa:    Mean-reversion speed (must be > 0).
///     theta:    Long-run variance (must be > 0).
///     xi:       Vol-of-vol (must be > 0).
///     rho:      Correlation in (-1, 1).
///     t:        Time to maturity in years (must be > 0).
///     n_cos:    Number of COS terms (default 160).
///
/// Returns:
///     1-D NumPy array of option prices, same length as ``strikes``.
#[pyfunction]
#[pyo3(signature = (strikes, is_call, s0, v0, r, kappa, theta, xi, rho, t, n_cos=160))]
fn heston_price<'py>(
    py: Python<'py>,
    strikes: PyReadonlyArray1<'py, f64>,
    is_call: &Bound<'py, PyList>,
    s0: f64,
    v0: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    t: f64,
    n_cos: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if s0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s0 must be positive",
        ));
    }
    if v0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "v0 must be positive",
        ));
    }
    if kappa <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "kappa must be positive",
        ));
    }
    if theta <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must be positive",
        ));
    }
    if xi <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xi must be positive",
        ));
    }
    if rho <= -1.0 || rho >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rho must be in (-1, 1)",
        ));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "t must be positive",
        ));
    }
    if n_cos == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cos must be positive",
        ));
    }

    let strikes_vec: Vec<f64> = strikes.as_array().iter().copied().collect();
    let is_call_vec: Vec<bool> = is_call
        .iter()
        .map(|item| item.extract::<bool>())
        .collect::<PyResult<Vec<bool>>>()?;

    if strikes_vec.len() != is_call_vec.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "strikes and is_call must have the same length",
        ));
    }

    let prices = py.detach(|| {
        heston_cos_price_vec(
            s0,
            v0,
            r,
            kappa,
            theta,
            xi,
            rho,
            t,
            &strikes_vec,
            &is_call_vec,
            n_cos,
        )
    });
    Ok(prices.into_pyarray(py))
}

/// Calibrate Heston model parameters to market option prices.
///
/// Fits ``(v0, kappa, theta, xi, rho)`` using a Projected Levenberg-Marquardt
/// optimizer with Vega-weighted price residuals. The COS method is used for
/// fast analytical repricing during calibration.
///
/// Args:
///     strikes:       1-D array of strike prices.
///     maturities:    1-D array of times to maturity (years).
///     market_prices: 1-D array of observed option prices.
///     is_call:       1-D array of booleans (True for call, False for put).
///     s0:            Current spot price (must be > 0).
///     r:             Risk-free rate (annualized).
///     max_iter:      Maximum LM iterations (default 200).
///     tol:           Convergence tolerance (default 1e-8).
///     n_cos:         COS terms for repricing (default 160).
///
/// Returns:
///     Dict: ``v0``, ``kappa``, ``theta``, ``xi``, ``rho``, ``rmse``,
///     ``iterations``, ``converged``, ``feller_satisfied``.
#[pyfunction]
#[pyo3(signature = (strikes, maturities, market_prices, is_call, s0, r, max_iter=200, tol=1e-8, n_cos=160))]
fn heston_calibrate<'py>(
    py: Python<'py>,
    strikes: PyReadonlyArray1<'py, f64>,
    maturities: PyReadonlyArray1<'py, f64>,
    market_prices: PyReadonlyArray1<'py, f64>,
    is_call: &Bound<'py, PyList>,
    s0: f64,
    r: f64,
    max_iter: usize,
    tol: f64,
    n_cos: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let strikes_vec: Vec<f64> = strikes.as_array().iter().copied().collect();
    let mat_vec: Vec<f64> = maturities.as_array().iter().copied().collect();
    let prices_vec: Vec<f64> = market_prices.as_array().iter().copied().collect();
    let is_call_vec: Vec<bool> = is_call
        .iter()
        .map(|item| item.extract::<bool>())
        .collect::<PyResult<Vec<bool>>>()?;

    let result = py
        .detach(|| {
            heston_calibrate_core(
                &strikes_vec,
                &mat_vec,
                &prices_vec,
                &is_call_vec,
                s0,
                r,
                max_iter,
                tol,
                n_cos,
            )
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let dict = PyDict::new(py);
    dict.set_item("v0", result.v0)?;
    dict.set_item("kappa", result.kappa)?;
    dict.set_item("theta", result.theta)?;
    dict.set_item("xi", result.xi)?;
    dict.set_item("rho", result.rho)?;
    dict.set_item("rmse", result.rmse)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("feller_satisfied", result.feller_satisfied)?;
    Ok(dict)
}

/// stocha: High-performance random number and financial simulation library.
#[pymodule]
fn _stocha(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RNG>()?;
    m.add_function(wrap_pyfunction!(gbm, m)?)?;
    m.add_function(wrap_pyfunction!(sobol, m)?)?;
    m.add_function(wrap_pyfunction!(halton_seq, m)?)?;
    m.add_function(wrap_pyfunction!(heston, m)?)?;
    m.add_function(wrap_pyfunction!(merton_jump_diffusion, m)?)?;
    m.add_function(wrap_pyfunction!(var_cvar, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_copula, m)?)?;
    m.add_function(wrap_pyfunction!(student_t_copula, m)?)?;
    m.add_function(wrap_pyfunction!(hull_white, m)?)?;
    m.add_function(wrap_pyfunction!(sabr_implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(sabr_calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(multi_gbm, m)?)?;
    m.add_function(wrap_pyfunction!(lsmc_american_option, m)?)?;
    m.add_function(wrap_pyfunction!(greeks_fd, m)?)?;
    m.add_function(wrap_pyfunction!(greeks_pathwise, m)?)?;
    m.add_function(wrap_pyfunction!(heston_price, m)?)?;
    m.add_function(wrap_pyfunction!(heston_calibrate, m)?)?;
    m.add("__version__", "1.5.0")?;
    Ok(())
}
