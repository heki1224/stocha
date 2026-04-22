use crate::dist::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::Array2;
use rayon::prelude::*;

/// Parameters for a GBM simulation.
#[derive(Debug, Clone)]
pub struct GbmParams {
    /// Initial asset price.
    pub s0: f64,
    /// Drift (annualized).
    pub mu: f64,
    /// Volatility (annualized).
    pub sigma: f64,
    /// Time to maturity in years.
    pub t: f64,
    /// Number of time steps.
    pub steps: usize,
    /// Number of simulation paths.
    pub n_paths: usize,
    /// Enable antithetic variates variance reduction.
    pub antithetic: bool,
}

/// Simulate GBM paths in parallel using Euler-Maruyama discretization.
///
/// Discretization scheme:
///   S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
///
/// Each path is assigned an independent RNG stream via block splitting:
/// path `i` advances the base RNG by `i * block_size` steps before sampling.
/// This guarantees reproducibility regardless of thread scheduling.
///
/// When `antithetic=true`, a mirrored path using `-Z` is generated for each
/// base path and appended to the output, reducing estimator variance.
///
/// # Returns
/// An `Array2<f64>` of shape `(n_paths, steps + 1)`.
/// Column 0 is the initial price `s0`; column `steps` is the terminal price.
pub fn gbm_paths(params: &GbmParams, seed: u128) -> Array2<f64> {
    let dt = params.t / params.steps as f64;
    let drift = (params.mu - 0.5 * params.sigma * params.sigma) * dt;
    let diffusion = params.sigma * dt.sqrt();

    // When using antithetic variates, generate half the paths then mirror them.
    let base_paths = if params.antithetic {
        (params.n_paths + 1) / 2
    } else {
        params.n_paths
    };

    // Block size: conservative upper bound on RNG consumption per path.
    let block_size: u128 = (params.steps as u128 + 1024) * 2;

    let simulate = |path_idx: usize, negate_z: bool| -> Vec<f64> {
        let mut rng = Pcg64Dxsm::new(seed);
        rng.advance(path_idx as u128 * block_size);

        let mut path = Vec::with_capacity(params.steps + 1);
        path.push(params.s0);
        let mut s = params.s0;

        for _ in 0..params.steps {
            let z = if negate_z {
                -NormalSampler::sample(&mut rng)
            } else {
                NormalSampler::sample(&mut rng)
            };
            s *= (drift + diffusion * z).exp();
            path.push(s);
        }
        path
    };

    // Generate base paths in parallel.
    let base: Vec<Vec<f64>> = (0..base_paths)
        .into_par_iter()
        .map(|i| simulate(i, false))
        .collect();

    if params.antithetic {
        // Generate antithetic (negated Z) paths in parallel.
        let anti: Vec<Vec<f64>> = (0..base_paths)
            .into_par_iter()
            .map(|i| simulate(i, true))
            .collect();

        let total = (base.len() + anti.len()).min(params.n_paths);
        let mut result = Array2::<f64>::zeros((total, params.steps + 1));
        for (i, path) in base.iter().chain(anti.iter()).take(total).enumerate() {
            for (j, &v) in path.iter().enumerate() {
                result[[i, j]] = v;
            }
        }
        result
    } else {
        let mut result = Array2::<f64>::zeros((params.n_paths, params.steps + 1));
        for (i, path) in base.iter().enumerate() {
            for (j, &v) in path.iter().enumerate() {
                result[[i, j]] = v;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> GbmParams {
        GbmParams {
            s0: 100.0,
            mu: 0.05,
            sigma: 0.2,
            t: 1.0,
            steps: 252,
            n_paths: 1000,
            antithetic: false,
        }
    }

    #[test]
    fn test_output_shape() {
        let paths = gbm_paths(&default_params(), 42);
        assert_eq!(paths.shape(), &[1000, 253]);
    }

    #[test]
    fn test_initial_price_equals_s0() {
        let params = default_params();
        let paths = gbm_paths(&params, 42);
        for i in 0..params.n_paths {
            assert!((paths[[i, 0]] - params.s0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_all_prices_positive() {
        let paths = gbm_paths(&default_params(), 42);
        assert!(paths.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_reproducibility() {
        let params = default_params();
        assert_eq!(gbm_paths(&params, 42), gbm_paths(&params, 42));
    }

    #[test]
    fn test_antithetic_shape() {
        let mut params = default_params();
        params.antithetic = true;
        let paths = gbm_paths(&params, 42);
        assert_eq!(paths.shape()[0], params.n_paths);
    }

    #[test]
    fn test_expected_terminal_price() {
        // E[S(T)] = S0 * exp(mu * T) under GBM
        let params = GbmParams {
            n_paths: 100_000,
            ..default_params()
        };
        let paths = gbm_paths(&params, 0);
        let mean_terminal = paths.column(params.steps).mean().unwrap();
        let expected = params.s0 * (params.mu * params.t).exp();
        let rel_err = (mean_terminal - expected).abs() / expected;
        assert!(rel_err < 0.02, "rel_err={:.4}", rel_err);
    }
}
