mod copula;
mod dist;
mod finance;
mod prng;
mod qrng;
mod risk;

use copula::{gaussian_copula_samples, student_t_copula_samples};
use dist::NormalSampler;
use finance::gbm::{gbm_paths, GbmParams};
use finance::heston::{heston_paths, HestonParams};
use finance::hull_white::{hull_white_paths, HullWhiteParams};
use finance::jump_diffusion::{merton_paths, MertonParams};
use finance::lsmc::{lsmc_american_option as lsmc_price, LsmcParams};
use finance::sabr::sabr_implied_vol as sabr_vol;
use qrng::{halton_sequence, sobol_sequence};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array2;
use prng::Pcg64Dxsm;
use risk::var_cvar as compute_var_cvar;
use pyo3::prelude::*;

/// Convert an owned `Array2<f64>` into a Python NumPy array.
fn into_py_array2<'py>(arr: Array2<f64>, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = [arr.shape()[0], arr.shape()[1]];
    let flat: Vec<f64> = arr.into_raw_vec_and_offset().0;
    let out = Array2::from_shape_vec(shape, flat)
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
    fn standard_normal<'py>(
        &mut self,
        py: Python<'py>,
        size: usize,
    ) -> Bound<'py, PyArray1<f64>> {
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

    /// Serialize the seed to a JSON string.
    ///
    /// **Important limitation**: records the *original seed only*, not the full
    /// internal generator state. Restoring via ``from_state`` reconstructs the RNG
    /// from scratch (equivalent to ``RNG(seed=original_seed)``), which replays the
    /// sequence **from the beginning** — not from the position at which ``save_state``
    /// was called. Suitable for audit trails and fixed-seed reproducibility; not for
    /// mid-stream checkpointing.
    ///
    /// Returns:
    ///     JSON string, e.g. ``'{"seed":42}'``.
    fn save_state(&self) -> String {
        self.inner.save_state()
    }

    /// Restore an RNG from a JSON string produced by :meth:`save_state`.
    ///
    /// The restored RNG is identical to ``RNG(seed=original_seed)`` — it starts
    /// from the beginning of the sequence regardless of how far the original RNG
    /// had advanced. See :meth:`save_state` for the full limitation description.
    ///
    /// Args:
    ///     json: JSON string as returned by :meth:`save_state`.
    ///
    /// Returns:
    ///     New ``RNG`` instance seeded from the recorded value.
    #[staticmethod]
    fn from_state(json: &str) -> PyResult<RNG> {
        let inner = Pcg64Dxsm::from_state(json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
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

    let params = GbmParams { s0, mu, sigma, t, steps, n_paths, antithetic };
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
        return Err(pyo3::exceptions::PyValueError::new_err("dim must be at least 1"));
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
        return Err(pyo3::exceptions::PyValueError::new_err("dim must be at least 1"));
    }
    let arr = py
        .detach(|| halton_sequence(dim, n_samples, skip))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    into_py_array2(arr, py)
}

/// Simulate Heston stochastic volatility paths.
///
/// Uses Euler-Maruyama discretization with the Full Truncation (FT) scheme to
/// handle negative variance states without bias.
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
///
/// Returns:
///     NumPy array of shape ``(n_paths, steps + 1)``.
#[pyfunction]
#[pyo3(signature = (s0, v0, mu, kappa, theta, xi, rho, t, steps, n_paths, seed=42))]
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
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if s0 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("s0 must be positive"));
    }
    if v0 < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("v0 must be non-negative"));
    }
    if kappa <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kappa must be positive"));
    }
    if theta <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("theta must be positive"));
    }
    if xi <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("xi must be positive"));
    }
    if rho < -1.0 || rho > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("rho must be in [-1, 1]"));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("t must be positive"));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }

    let params = HestonParams { s0, v0, mu, kappa, theta, xi, rho, t, steps, n_paths };
    let result = py.detach(|| heston_paths(&params, seed as u128));
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
        return Err(pyo3::exceptions::PyValueError::new_err("s0 must be positive"));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("sigma must be positive"));
    }
    if lambda_ < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("lambda_ must be non-negative"));
    }
    if sigma_j < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("sigma_j must be non-negative"));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("t must be positive"));
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
fn var_cvar<'py>(
    returns: PyReadonlyArray1<'py, f64>,
    confidence: f64,
) -> PyResult<(f64, f64)> {
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
    Ok(compute_var_cvar(&slice, confidence))
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
    let arr = corr.as_array();
    let result = py
        .detach(|| gaussian_copula_samples(arr, n_samples, seed as u128))
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
    let arr = corr.as_array();
    let result = py
        .detach(|| student_t_copula_samples(arr, nu, n_samples, seed as u128))
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
        return Err(pyo3::exceptions::PyValueError::new_err("a must be positive"));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("sigma must be positive"));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("t must be positive"));
    }
    if steps == 0 || n_paths == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "steps and n_paths must be positive",
        ));
    }

    let params = HullWhiteParams { r0, a, theta, sigma, t, steps, n_paths };
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
        return Err(pyo3::exceptions::PyValueError::new_err("s0 must be positive"));
    }
    if k <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("k must be positive"));
    }
    if sigma <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("sigma must be positive"));
    }
    if t <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("t must be positive"));
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

    let params = LsmcParams { s0, k, r, sigma, t, steps, n_paths, is_put, poly_degree };
    let (price, std_err) = py.detach(|| lsmc_price(&params, seed as u128));
    Ok((price, std_err))
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
    m.add_function(wrap_pyfunction!(lsmc_american_option, m)?)?;
    m.add("__version__", "0.3.2")?;
    Ok(())
}
