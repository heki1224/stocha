mod dist;
mod finance;
mod prng;
mod qrng;

use dist::NormalSampler;
use finance::gbm::{gbm_paths, GbmParams};
use finance::heston::{heston_paths, HestonParams};
use finance::jump_diffusion::{merton_paths, MertonParams};
use qrng::{halton_sequence, sobol_sequence};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use prng::Pcg64Dxsm;
use pyo3::prelude::*;

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
    ///     seed:      Integer seed for reproducibility (default 0).
    ///     algorithm: Algorithm name. Currently only "pcg64dxsm" is supported.
    #[new]
    #[pyo3(signature = (seed=0, algorithm="pcg64dxsm"))]
    fn new(seed: u64, algorithm: &str) -> PyResult<Self> {
        match algorithm {
            "pcg64dxsm" | "default" => Ok(RNG {
                inner: Pcg64Dxsm::new(seed as u128),
                seed: seed as u128,
            }),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown algorithm: '{}'. Supported: 'pcg64dxsm'",
                other
            ))),
        }
    }

    /// The seed used to initialize this RNG.
    #[getter]
    fn seed(&self) -> u64 {
        self.seed as u64
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

    /// Serialize the current RNG state to a JSON string for checkpointing.
    fn save_state(&self) -> String {
        self.inner.save_state()
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

    let shape = [result.shape()[0], result.shape()[1]];
    let flat: Vec<f64> = result.into_raw_vec_and_offset().0;
    let arr = numpy::ndarray::Array2::from_shape_vec(shape, flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
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

    let shape = [arr.shape()[0], arr.shape()[1]];
    let flat: Vec<f64> = arr.into_raw_vec_and_offset().0;
    let out = numpy::ndarray::Array2::from_shape_vec(shape, flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
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

    let shape = [arr.shape()[0], arr.shape()[1]];
    let flat: Vec<f64> = arr.into_raw_vec_and_offset().0;
    let out = numpy::ndarray::Array2::from_shape_vec(shape, flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
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

    let shape = [result.shape()[0], result.shape()[1]];
    let flat: Vec<f64> = result.into_raw_vec_and_offset().0;
    let arr = numpy::ndarray::Array2::from_shape_vec(shape, flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
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

    let shape = [result.shape()[0], result.shape()[1]];
    let flat: Vec<f64> = result.into_raw_vec_and_offset().0;
    let arr = numpy::ndarray::Array2::from_shape_vec(shape, flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
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
    m.add("__version__", "0.2.0")?;
    Ok(())
}
