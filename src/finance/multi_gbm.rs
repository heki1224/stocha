use crate::copula::gaussian::cholesky;
use crate::dist::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::{Array2, Array3};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct MultiGbmParams {
    pub s0: Vec<f64>,
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
    pub corr: Array2<f64>,
    pub t: f64,
    pub steps: usize,
    pub n_paths: usize,
    pub antithetic: bool,
}

/// Simulate correlated multi-asset GBM paths via Cholesky decomposition.
///
/// Returns Array3<f64> of shape (n_paths, steps+1, n_assets).
pub fn multi_gbm_paths(params: &MultiGbmParams, seed: u128) -> Result<Array3<f64>, String> {
    let n_assets = params.s0.len();
    let dt = params.t / params.steps as f64;
    let sqrt_dt = dt.sqrt();

    let l = cholesky(params.corr.view())?;

    let drift: Vec<f64> = (0..n_assets)
        .map(|i| (params.mu[i] - 0.5 * params.sigma[i] * params.sigma[i]) * dt)
        .collect();
    let diffusion: Vec<f64> = (0..n_assets).map(|i| params.sigma[i] * sqrt_dt).collect();

    let base_paths = if params.antithetic {
        (params.n_paths + 1) / 2
    } else {
        params.n_paths
    };

    let block_size: u128 = (params.steps as u128 * n_assets as u128 + 1024) * 2;

    let simulate = |path_idx: usize, negate: bool| -> Vec<f64> {
        let mut rng = Pcg64Dxsm::new(seed);
        rng.advance(path_idx as u128 * block_size);

        // (steps+1) * n_assets, row-major: [step0_asset0, step0_asset1, ..., step1_asset0, ...]
        let mut path = Vec::with_capacity((params.steps + 1) * n_assets);

        // Initial prices
        for i in 0..n_assets {
            path.push(params.s0[i]);
        }

        let mut z = vec![0.0f64; n_assets];
        let mut s: Vec<f64> = params.s0.clone();

        for _ in 0..params.steps {
            // Independent N(0,1)
            NormalSampler::sample_into(&mut rng, &mut z);
            if negate {
                for v in z.iter_mut() {
                    *v = -*v;
                }
            }

            // Correlated increments via Cholesky: epsilon = L * z
            for i in 0..n_assets {
                let mut eps = 0.0;
                for k in 0..=i {
                    eps += l[[i, k]] * z[k];
                }
                s[i] *= (drift[i] + diffusion[i] * eps).exp();
                path.push(s[i]);
            }
        }
        path
    };

    let base: Vec<Vec<f64>> = (0..base_paths)
        .into_par_iter()
        .map(|i| simulate(i, false))
        .collect();

    let total = if params.antithetic {
        let anti: Vec<Vec<f64>> = (0..base_paths)
            .into_par_iter()
            .map(|i| simulate(i, true))
            .collect();

        let total = (base.len() + anti.len()).min(params.n_paths);
        let mut result = Array3::<f64>::zeros([total, params.steps + 1, n_assets]);
        for (p, flat) in base.iter().chain(anti.iter()).take(total).enumerate() {
            for step in 0..=params.steps {
                for a in 0..n_assets {
                    result[[p, step, a]] = flat[step * n_assets + a];
                }
            }
        }
        return Ok(result);
    } else {
        params.n_paths
    };

    let mut result = Array3::<f64>::zeros([total, params.steps + 1, n_assets]);
    for (p, flat) in base.iter().enumerate() {
        for step in 0..=params.steps {
            for a in 0..n_assets {
                result[[p, step, a]] = flat[step * n_assets + a];
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn default_params() -> MultiGbmParams {
        MultiGbmParams {
            s0: vec![100.0, 50.0],
            mu: vec![0.05, 0.08],
            sigma: vec![0.2, 0.3],
            corr: array![[1.0, 0.6], [0.6, 1.0]],
            t: 1.0,
            steps: 252,
            n_paths: 1000,
            antithetic: false,
        }
    }

    #[test]
    fn test_output_shape() {
        let paths = multi_gbm_paths(&default_params(), 42).unwrap();
        assert_eq!(paths.shape(), &[1000, 253, 2]);
    }

    #[test]
    fn test_initial_prices() {
        let params = default_params();
        let paths = multi_gbm_paths(&params, 42).unwrap();
        for p in 0..params.n_paths {
            assert!((paths[[p, 0, 0]] - 100.0).abs() < 1e-10);
            assert!((paths[[p, 0, 1]] - 50.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_all_positive() {
        let paths = multi_gbm_paths(&default_params(), 42).unwrap();
        assert!(paths.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_reproducibility() {
        let params = default_params();
        let a = multi_gbm_paths(&params, 42).unwrap();
        let b = multi_gbm_paths(&params, 42).unwrap();
        assert_eq!(a, b);
    }
}
