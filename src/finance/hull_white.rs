use crate::dist::normal::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::Array2;
use rayon::prelude::*;

pub struct HullWhiteParams {
    pub r0: f64,
    pub a: f64,     // Mean-reversion speed.
    pub theta: f64, // Long-run mean rate (constant; theta/a in Vasicek notation).
    pub sigma: f64, // Volatility of the short rate.
    pub t: f64,
    pub steps: usize,
    pub n_paths: usize,
}

/// Simulate Hull-White 1-factor short-rate paths via Exact Simulation.
///
/// The model is: dr = (theta - a*r)*dt + sigma*dW
///
/// Exact transition (no discretization bias):
///   r(t+dt) = r(t)*exp(-a*dt) + (theta/a)*(1 - exp(-a*dt))
///              + sigma*sqrt((1 - exp(-2a*dt))/(2a)) * Z
///
/// Returns array of shape `(n_paths, steps + 1)` where column 0 = r0.
pub fn hull_white_paths(params: &HullWhiteParams, seed: u128) -> Array2<f64> {
    let dt = params.t / params.steps as f64;
    let e = (-params.a * dt).exp();
    let mean_reversion = (params.theta / params.a) * (1.0 - e);
    let vol = params.sigma * ((1.0 - e * e) / (2.0 * params.a)).sqrt();

    let n = params.n_paths;
    let steps = params.steps;
    let mut flat = vec![0.0f64; n * (steps + 1)];

    // Parallelise over paths; each path gets an independent RNG stream.
    flat.par_chunks_mut(steps + 1)
        .enumerate()
        .for_each(|(i, row)| {
            let path_seed = seed.wrapping_add(i as u128);
            let mut rng = Pcg64Dxsm::new(path_seed);

            row[0] = params.r0;
            for s in 0..steps {
                let z = NormalSampler::sample(&mut rng);
                row[s + 1] = row[s] * e + mean_reversion + vol * z;
            }
        });

    Array2::from_shape_vec([n, steps + 1], flat).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hull_white_shape() {
        let params = HullWhiteParams {
            r0: 0.05, a: 0.1, theta: 0.05, sigma: 0.01,
            t: 1.0, steps: 12, n_paths: 100,
        };
        let paths = hull_white_paths(&params, 42);
        assert_eq!(paths.shape(), [100, 13]);
        assert!((paths[[0, 0]] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_hull_white_mean_reversion() {
        // With a=1, theta=0.05, r0=0.10 → should drift toward theta/a = 0.05.
        let params = HullWhiteParams {
            r0: 0.10, a: 1.0, theta: 0.05, sigma: 0.001,
            t: 10.0, steps: 100, n_paths: 1000,
        };
        let paths = hull_white_paths(&params, 99);
        let terminal_mean = paths.column(100).mean().unwrap();
        assert!((terminal_mean - 0.05).abs() < 0.01, "mean={}", terminal_mean);
    }
}
