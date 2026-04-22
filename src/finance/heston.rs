use crate::dist::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::Array2;
use rayon::prelude::*;

/// Parameters for a Heston stochastic volatility simulation.
#[derive(Debug, Clone)]
pub struct HestonParams {
    /// Initial asset price.
    pub s0: f64,
    /// Initial variance (not volatility; v0 = sigma0^2).
    pub v0: f64,
    /// Drift rate (annualized).
    pub mu: f64,
    /// Mean-reversion speed of variance.
    pub kappa: f64,
    /// Long-run mean of variance.
    pub theta: f64,
    /// Volatility of variance (vol-of-vol).
    pub xi: f64,
    /// Correlation between asset and variance Brownian motions.
    pub rho: f64,
    /// Time to maturity in years.
    pub t: f64,
    /// Number of time steps.
    pub steps: usize,
    /// Number of simulation paths.
    pub n_paths: usize,
}

/// Simulate Heston stochastic volatility paths using the Full Truncation (FT) scheme.
///
/// Discretization (Euler-Maruyama with Full Truncation):
///   v+ = max(v, 0)          — used only for drift/diffusion, v itself may go negative
///   v(t+dt) = v + kappa*(theta - v+)*dt + xi*sqrt(v+)*sqrt(dt)*dW2
///   S(t+dt) = S * exp((mu - 0.5*v+)*dt + sqrt(v+)*sqrt(dt)*dW1)
///   dW2 = rho*dW1 + sqrt(1 - rho^2)*dZ   (correlated Brownians)
///
/// Using Absorption (max(v,0)) for the state update would introduce positive bias
/// near the boundary. FT avoids this by preserving the sign of v between steps.
///
/// # Returns
/// An `Array2<f64>` of shape `(n_paths, steps + 1)`.
/// Column 0 is the initial price `s0`; column `steps` is the terminal price.
pub fn heston_paths(params: &HestonParams, seed: u128) -> Array2<f64> {
    let dt = params.t / params.steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_comp = (1.0 - params.rho * params.rho).sqrt();

    // Two normals per step (dW1, dZ). Conservative block size with buffer.
    let block_size: u128 = (params.steps as u128 + 1024) * 6;

    let simulate = |path_idx: usize| -> Vec<f64> {
        let mut rng = Pcg64Dxsm::new(seed);
        rng.advance(path_idx as u128 * block_size);

        let mut path = Vec::with_capacity(params.steps + 1);
        path.push(params.s0);
        let mut s = params.s0;
        let mut v = params.v0;

        for _ in 0..params.steps {
            let dw1 = NormalSampler::sample(&mut rng);
            let dz = NormalSampler::sample(&mut rng);
            // Correlated Brownian for variance
            let dw2 = params.rho * dw1 + rho_comp * dz;

            // Full Truncation: v_plus used for all calculations, but v state is unrestricted
            let v_plus = v.max(0.0);
            let sqrt_v = v_plus.sqrt();

            // Update variance state (may go negative — intentional in FT scheme)
            v += params.kappa * (params.theta - v_plus) * dt
                + params.xi * sqrt_v * sqrt_dt * dw2;

            // Update stock price using log-Euler scheme
            s *= ((params.mu - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * dw1).exp();
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

    fn default_params() -> HestonParams {
        HestonParams {
            s0: 100.0,
            v0: 0.04, // sigma0 = 0.2
            mu: 0.05,
            kappa: 2.0,
            theta: 0.04,
            xi: 0.3,
            rho: -0.7,
            t: 1.0,
            steps: 252,
            n_paths: 1000,
        }
    }

    #[test]
    fn test_output_shape() {
        let paths = heston_paths(&default_params(), 42);
        assert_eq!(paths.shape(), &[1000, 253]);
    }

    #[test]
    fn test_initial_price_equals_s0() {
        let params = default_params();
        let paths = heston_paths(&params, 42);
        for i in 0..params.n_paths {
            assert!((paths[[i, 0]] - params.s0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reproducibility() {
        let params = default_params();
        assert_eq!(heston_paths(&params, 42), heston_paths(&params, 42));
    }

    #[test]
    fn test_positive_prices() {
        let paths = heston_paths(&default_params(), 42);
        assert!(paths.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_expected_terminal_price() {
        // Under Heston, E[S(T)] = S0 * exp(mu * T) (same as risk-neutral drift)
        let params = HestonParams {
            n_paths: 100_000,
            ..default_params()
        };
        let paths = heston_paths(&params, 0);
        let mean_terminal = paths.column(params.steps).mean().unwrap();
        let expected = params.s0 * (params.mu * params.t).exp();
        let rel_err = (mean_terminal - expected).abs() / expected;
        assert!(rel_err < 0.02, "rel_err={:.4}", rel_err);
    }

    #[test]
    fn test_feller_condition_violated_stability() {
        // 2*kappa*theta < xi^2 → variance process hits zero; Full Truncation should
        // keep the simulation numerically stable (no panic, all prices positive).
        let params = HestonParams {
            kappa: 1.0,
            theta: 0.04,  // 2*kappa*theta = 0.08 < xi^2 = 0.09 → Feller violated
            xi: 0.3,
            n_paths: 5_000,
            ..default_params()
        };
        let paths = heston_paths(&params, 77);
        assert_eq!(paths.shape(), &[5_000, 253]);
        assert!(paths.iter().all(|&v| v > 0.0), "negative price under Feller violation");
    }
}
