use crate::dist::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::Array2;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct HestonParams {
    pub s0: f64,
    pub v0: f64,
    pub mu: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    pub t: f64,
    pub steps: usize,
    pub n_paths: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HestonScheme {
    Euler,
    QE,
}

/// Simulate Heston paths using the Full Truncation (FT) Euler scheme.
#[cfg(test)]
pub fn heston_paths(params: &HestonParams, seed: u128) -> Array2<f64> {
    heston_euler(params, seed)
}

/// Simulate Heston paths with a chosen scheme.
pub fn heston_paths_with_scheme(
    params: &HestonParams,
    seed: u128,
    scheme: HestonScheme,
) -> Array2<f64> {
    match scheme {
        HestonScheme::Euler => heston_euler(params, seed),
        HestonScheme::QE => heston_qe(params, seed),
    }
}

fn heston_euler(params: &HestonParams, seed: u128) -> Array2<f64> {
    let dt = params.t / params.steps as f64;
    let sqrt_dt = dt.sqrt();
    let rho_comp = (1.0 - params.rho * params.rho).sqrt();
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
            let dw2 = params.rho * dw1 + rho_comp * dz;

            let v_plus = v.max(0.0);
            let sqrt_v = v_plus.sqrt();

            v += params.kappa * (params.theta - v_plus) * dt
                + params.xi * sqrt_v * sqrt_dt * dw2;

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

const PSI_C: f64 = 1.5;

fn heston_qe(params: &HestonParams, seed: u128) -> Array2<f64> {
    let dt = params.t / params.steps as f64;
    let e_kdt = (-params.kappa * dt).exp();
    let k1 = params.kappa * params.theta;

    // QE uses 2 uniforms + 1 normal per step (variance sampling + asset update)
    let block_size: u128 = (params.steps as u128 + 1024) * 6;

    let simulate = |path_idx: usize| -> Vec<f64> {
        let mut rng = Pcg64Dxsm::new(seed);
        rng.advance(path_idx as u128 * block_size);

        let mut path = Vec::with_capacity(params.steps + 1);
        path.push(params.s0);
        let mut s = params.s0;
        let mut v = params.v0;

        for _ in 0..params.steps {
            let v_plus = v.max(0.0);

            // Conditional mean and variance of V(t+dt) given V(t)
            let m = k1 * (1.0 - e_kdt) / params.kappa + v_plus * e_kdt;
            let m = m.max(1e-15);
            let s2 = v_plus * params.xi * params.xi * e_kdt
                * (1.0 - e_kdt) / params.kappa
                + k1 * params.xi * params.xi
                    * (1.0 - e_kdt).powi(2)
                    / (2.0 * params.kappa * params.kappa);
            let psi = s2 / (m * m);

            let v_next = if psi <= PSI_C {
                // Quadratic branch: V ~ a*(b + Z)^2
                let b2 = 2.0 / psi - 1.0
                    + (2.0 / psi * (2.0 / psi - 1.0)).sqrt();
                let b2 = b2.max(0.0);
                let a = m / (1.0 + b2);
                let b = b2.sqrt();
                let z = NormalSampler::sample(&mut rng);
                a * (b + z) * (b + z)
            } else {
                // Exponential branch: V ~ mixed point-mass at 0 + exponential
                let p = (psi - 1.0) / (psi + 1.0);
                let beta = (1.0 - p) / m;
                let u = rng.next_f64();
                if u <= p {
                    0.0
                } else {
                    ((1.0 - p) / (1.0 - u)).ln() / beta
                }
            };

            let v_next = v_next.max(0.0);

            // Andersen (2008) log-price update
            let z1 = NormalSampler::sample(&mut rng);
            let kappa_rho_xi = params.kappa * params.rho / params.xi;
            let gamma1 = 0.5;
            let gamma2 = 0.5;
            let k0_s = (params.mu - kappa_rho_xi * params.theta) * dt;
            let k1_s = gamma1 * dt * (kappa_rho_xi - 0.5) - params.rho / params.xi;
            let k2_s = gamma2 * dt * (kappa_rho_xi - 0.5) + params.rho / params.xi;
            let k3_s = gamma1 * dt * (1.0 - params.rho * params.rho);
            let k4_s = gamma2 * dt * (1.0 - params.rho * params.rho);

            let var_term = (k3_s * v_plus + k4_s * v_next).max(0.0);
            let log_s = s.ln() + k0_s + k1_s * v_plus + k2_s * v_next
                + var_term.sqrt() * z1;

            s = log_s.clamp(-500.0, 500.0).exp();
            v = v_next;
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
        for (j, &val) in path.iter().enumerate() {
            result[[i, j]] = val;
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
            v0: 0.04,
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
        let params = HestonParams {
            kappa: 1.0,
            theta: 0.04,
            xi: 0.3,
            n_paths: 5_000,
            ..default_params()
        };
        let paths = heston_paths(&params, 77);
        assert_eq!(paths.shape(), &[5_000, 253]);
        assert!(paths.iter().all(|&v| v > 0.0), "negative price under Feller violation");
    }

    // QE tests
    #[test]
    fn test_qe_output_shape() {
        let paths = heston_paths_with_scheme(&default_params(), 42, HestonScheme::QE);
        assert_eq!(paths.shape(), &[1000, 253]);
    }

    #[test]
    fn test_qe_reproducibility() {
        let params = default_params();
        let a = heston_paths_with_scheme(&params, 42, HestonScheme::QE);
        let b = heston_paths_with_scheme(&params, 42, HestonScheme::QE);
        assert_eq!(a, b);
    }

    #[test]
    fn test_qe_positive_prices() {
        let paths = heston_paths_with_scheme(&default_params(), 42, HestonScheme::QE);
        assert!(paths.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_qe_expected_terminal_price() {
        let params = HestonParams {
            n_paths: 100_000,
            ..default_params()
        };
        let paths = heston_paths_with_scheme(&params, 0, HestonScheme::QE);
        let mean_terminal = paths.column(params.steps).mean().unwrap();
        let expected = params.s0 * (params.mu * params.t).exp();
        let rel_err = (mean_terminal - expected).abs() / expected;
        assert!(rel_err < 0.02, "QE rel_err={:.4}", rel_err);
    }

    #[test]
    fn test_qe_feller_violated_stability() {
        let params = HestonParams {
            kappa: 1.0,
            theta: 0.04,
            xi: 0.3,
            n_paths: 5_000,
            ..default_params()
        };
        let paths = heston_paths_with_scheme(&params, 77, HestonScheme::QE);
        assert_eq!(paths.shape(), &[5_000, 253]);
        assert!(paths.iter().all(|&v| v > 0.0));
    }
}
