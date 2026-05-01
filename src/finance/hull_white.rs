use crate::dist::normal::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::Array2;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct HullWhiteParams {
    pub r0: f64,
    pub a: f64,     // Mean-reversion speed.
    pub theta: f64, // Drift constant = a × long_run_mean_rate (i.e. dr = (theta - a*r)*dt).
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

    // Block splitting: each path advances from the same seed by i * block_size steps.
    // 1 normal per step; Marsaglia polar uses ~2 u64 draws on average + buffer.
    let block_size: u128 = (steps as u128 + 1024) * 4;

    // Parallelise over paths; each path gets an independent RNG stream.
    flat.par_chunks_mut(steps + 1)
        .enumerate()
        .for_each(|(i, row)| {
            let mut rng = Pcg64Dxsm::new(seed);
            rng.advance(i as u128 * block_size);

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
            r0: 0.05,
            a: 0.1,
            theta: 0.05,
            sigma: 0.01,
            t: 1.0,
            steps: 12,
            n_paths: 100,
        };
        let paths = hull_white_paths(&params, 42);
        assert_eq!(paths.shape(), [100, 13]);
        assert!((paths[[0, 0]] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_hull_white_mean_reversion() {
        // With a=1, theta=0.05, r0=0.10 → should drift toward theta/a = 0.05.
        let params = HullWhiteParams {
            r0: 0.10,
            a: 1.0,
            theta: 0.05,
            sigma: 0.001,
            t: 10.0,
            steps: 100,
            n_paths: 1000,
        };
        let paths = hull_white_paths(&params, 99);
        let terminal_mean = paths.column(100).mean().unwrap();
        assert!(
            (terminal_mean - 0.05).abs() < 0.01,
            "mean={}",
            terminal_mean
        );
    }

    #[test]
    fn test_hull_white_negative_initial_rate() {
        // Negative initial rate (e.g. EUR/CHF negative rate environment).
        // Exact simulation handles r0 < 0 without special treatment.
        let params = HullWhiteParams {
            r0: -0.01,
            a: 0.5,
            theta: 0.02,
            sigma: 0.005,
            t: 5.0,
            steps: 60,
            n_paths: 500,
        };
        let paths = hull_white_paths(&params, 7);
        assert_eq!(paths.shape(), [500, 61]);
        // First column should equal r0.
        for i in 0..500 {
            assert!((paths[[i, 0]] - (-0.01)).abs() < 1e-12);
        }
        // Terminal mean should drift toward theta/a = 0.04.
        let terminal_mean = paths.column(60).mean().unwrap();
        assert!(
            terminal_mean > -0.05 && terminal_mean < 0.10,
            "terminal_mean={} unexpectedly far from long-run mean",
            terminal_mean
        );
    }
}
