mod dist;
mod finance;
mod prng;

use dist::NormalSampler;
use finance::gbm::{gbm_paths, GbmParams};
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

/// stocha: High-performance random number and financial simulation library.
#[pymodule]
fn _stocha(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RNG>()?;
    m.add_function(wrap_pyfunction!(gbm, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
