use crate::dist::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::Array2;
use rayon::prelude::*;

/// Parameters for a Merton Jump-Diffusion simulation.
#[derive(Debug, Clone)]
pub struct MertonParams {
    /// Initial asset price.
    pub s0: f64,
    /// Drift rate (annualized).
    pub mu: f64,
    /// Diffusion volatility (annualized, must be > 0).
    pub sigma: f64,
    /// Jump intensity: average number of jumps per year.
    pub lambda: f64,
    /// Mean of the log-jump size: log(J) ~ N(mu_j, sigma_j^2).
    pub mu_j: f64,
    /// Standard deviation of the log-jump size.
    pub sigma_j: f64,
    /// Time to maturity in years.
    pub t: f64,
    /// Number of time steps.
    pub steps: usize,
    /// Number of simulation paths.
    pub n_paths: usize,
}

/// Simulate Merton Jump-Diffusion paths.
///
/// Model (Merton 1976):
///   dS = (mu - lambda * m_bar) * S * dt + sigma * S * dW + S * (J - 1) * dN
///
/// where:
///   J = exp(mu_j + sigma_j * Z_j)   — lognormal jump size
///   m_bar = exp(mu_j + 0.5 * sigma_j^2) - 1   — mean jump return
///   dN ~ Poisson(lambda * dt)       — jump arrival (Bernoulli approximation)
///
/// The compensator `-lambda * m_bar * dt` ensures martingale property.
///
/// Jump arrival is approximated as Bernoulli with P(jump) = lambda * dt,
/// which is accurate for lambda * dt << 1 (typical regime: lambda ~ 1-5,
/// dt = 1/252 gives lambda*dt ~ 0.004–0.02).
///
/// # Returns
/// An `Array2<f64>` of shape `(n_paths, steps + 1)`.
/// Column 0 is the initial price `s0`; column `steps` is the terminal price.
pub fn merton_paths(params: &MertonParams, seed: u128) -> Array2<f64> {
    let dt = params.t / params.steps as f64;
    let sqrt_dt = dt.sqrt();

    // Martingale correction: mean jump return
    let m_bar = (params.mu_j + 0.5 * params.sigma_j * params.sigma_j).exp() - 1.0;

    // Diffusion drift with jump compensation
    let drift = (params.mu - 0.5 * params.sigma * params.sigma - params.lambda * m_bar) * dt;
    let diffusion = params.sigma * sqrt_dt;

    // Bernoulli jump probability per step
    let jump_prob = params.lambda * dt;

    // Per step: up to 3 RNG calls (1 uniform for jump test, 1 normal for diffusion,
    // 1 normal for jump size when triggered). Conservative block size.
    let block_size: u128 = (params.steps as u128 + 1024) * 8;

    let simulate = |path_idx: usize| -> Vec<f64> {
        let mut rng = Pcg64Dxsm::new(seed);
        rng.advance(path_idx as u128 * block_size);

        let mut path = Vec::with_capacity(params.steps + 1);
        path.push(params.s0);
        let mut s = params.s0;

        for _ in 0..params.steps {
            let z = NormalSampler::sample(&mut rng);
            let diffusion_return = drift + diffusion * z;

            // Bernoulli jump: one uniform draw per step regardless of outcome
            let u = rng.next_f64();
            let jump_return = if u < jump_prob {
                let z_j = NormalSampler::sample(&mut rng);
                params.mu_j + params.sigma_j * z_j
            } else {
                0.0
            };

            s *= (diffusion_return + jump_return).exp();
            path.push(s);
        }
        path
    };

    let paths: Vec<Vec<f64>> = (0..params.n_paths)
        .into_par_iter()
        .map(|i| simulate(i))
        .collect();

    let mut result = Array2::<f64>::zeros((params.n_paths, params.steps + 1));
    for (i, path) in paths.iter().enumerate() {
        for (j, &v) in path.iter().enumerate() {
            result[[i, j]] = v;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> MertonParams {
        MertonParams {
            s0: 100.0,
            mu: 0.05,
            sigma: 0.2,
            lambda: 1.0,
            mu_j: -0.05,
            sigma_j: 0.1,
            t: 1.0,
            steps: 252,
            n_paths: 1000,
        }
    }

    #[test]
    fn test_output_shape() {
        let paths = merton_paths(&default_params(), 42);
        assert_eq!(paths.shape(), &[1000, 253]);
    }

    #[test]
    fn test_initial_price_equals_s0() {
        let params = default_params();
        let paths = merton_paths(&params, 42);
        for i in 0..params.n_paths {
            assert!((paths[[i, 0]] - params.s0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_positive_prices() {
        let paths = merton_paths(&default_params(), 42);
        assert!(paths.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_reproducibility() {
        let params = default_params();
        assert_eq!(merton_paths(&params, 42), merton_paths(&params, 42));
    }

    #[test]
    fn test_expected_terminal_price() {
        // Under Merton, E[S(T)] = S0 * exp(mu * T) due to compensator
        let params = MertonParams {
            n_paths: 200_000,
            ..default_params()
        };
        let paths = merton_paths(&params, 0);
        let mean_terminal = paths.column(params.steps).mean().unwrap();
        let expected = params.s0 * (params.mu * params.t).exp();
        let rel_err = (mean_terminal - expected).abs() / expected;
        assert!(rel_err < 0.02, "rel_err={:.4}", rel_err);
    }
}
