use crate::prng::Pcg64Dxsm;

/// Normal distribution sampler using the Marsaglia polar method.
///
/// A rejection-based improvement of Box-Muller that avoids trigonometric
/// functions and produces exact normal samples.
///
/// Note: Ziggurat method is planned for Phase 2 to further improve throughput.
pub struct NormalSampler;

impl NormalSampler {
    /// Draw one sample from the standard normal distribution N(0, 1).
    pub fn sample(rng: &mut Pcg64Dxsm) -> f64 {
        loop {
            let u = rng.next_f64() * 2.0 - 1.0;
            let v = rng.next_f64() * 2.0 - 1.0;
            let s = u * u + v * v;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            return u * factor;
        }
    }

    /// Fill a mutable slice with standard normal samples.
    ///
    /// Generates two samples per iteration using the polar method for efficiency.
    pub fn sample_into(rng: &mut Pcg64Dxsm, buf: &mut [f64]) {
        let mut i = 0;
        while i < buf.len() {
            loop {
                let u = rng.next_f64() * 2.0 - 1.0;
                let v = rng.next_f64() * 2.0 - 1.0;
                let s = u * u + v * v;
                if s >= 1.0 || s == 0.0 {
                    continue;
                }
                let factor = (-2.0 * s.ln() / s).sqrt();
                buf[i] = u * factor;
                i += 1;
                if i < buf.len() {
                    buf[i] = v * factor;
                    i += 1;
                }
                break;
            }
        }
    }

    /// Draw one sample from N(mu, sigma^2).
    pub fn sample_scaled(rng: &mut Pcg64Dxsm, mu: f64, sigma: f64) -> f64 {
        mu + sigma * Self::sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_normal_mean_and_variance() {
        let mut rng = Pcg64Dxsm::new(42);
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| NormalSampler::sample(&mut rng)).collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.05, "mean={}", mean);
        assert!((var - 1.0).abs() < 0.05, "var={}", var);
    }

    #[test]
    fn test_sample_into_fills_correctly() {
        let mut rng = Pcg64Dxsm::new(0);
        let mut buf = vec![0.0f64; 10_000];
        NormalSampler::sample_into(&mut rng, &mut buf);

        let mean = buf.iter().sum::<f64>() / buf.len() as f64;
        assert!(mean.abs() < 0.05, "mean={}", mean);
    }

    #[test]
    fn test_scaled_normal_mean() {
        let mut rng = Pcg64Dxsm::new(0);
        let n = 100_000;
        let mu = 3.0;
        let sigma = 2.0;
        let samples: Vec<f64> = (0..n)
            .map(|_| NormalSampler::sample_scaled(&mut rng, mu, sigma))
            .collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        assert!((mean - mu).abs() < 0.05, "mean={}", mean);
    }
}
