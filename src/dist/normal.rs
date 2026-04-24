use crate::dist::ziggurat_tables::{ZIGGURAT_R, ZIGGURAT_X, ZIGGURAT_Y};
use crate::prng::Pcg64Dxsm;

/// Normal distribution sampler using the Ziggurat method (N=256).
///
/// Marsaglia & Tsang (2000) with Doornik's fix: wedge rejection uses
/// an independent random number to avoid collision-test failures.
///
/// Bit layout of a single u64:
///   - bits [0..8)   : layer index (0..255)
///   - bit  8        : sign (0 = positive, 1 = negative)
///   - bits [11..64) : uniform x coordinate (53-bit mantissa)
pub struct NormalSampler;

impl NormalSampler {
    /// Draw one sample from the standard normal distribution N(0, 1).
    #[inline]
    pub fn sample(rng: &mut Pcg64Dxsm) -> f64 {
        loop {
            let u = rng.next_u64();
            let layer = (u & 0xFF) as usize;
            let sign = ((u >> 8) & 1) as f64 * -2.0 + 1.0; // 1.0 or -1.0
            let u_frac = (u >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let x = u_frac * ZIGGURAT_X[layer];

            // Fast path: x falls strictly inside the rectangle below
            if x < ZIGGURAT_X[layer + 1] {
                return sign * x;
            }

            // Tail: layer 0 requires special sampling
            if layer == 0 {
                let tail = Self::sample_tail(rng);
                return sign * tail;
            }

            // Wedge rejection: use a fresh random number (Doornik fix)
            let y = ZIGGURAT_Y[layer] + rng.next_f64() * (ZIGGURAT_Y[layer + 1] - ZIGGURAT_Y[layer]);
            if y < (-0.5 * x * x).exp() {
                return sign * x;
            }
        }
    }

    /// Tail sampler for |x| > r using the method of Marsaglia (1964).
    ///
    /// Generates x ~ f(x) for x > r by rejection sampling:
    ///   x = -ln(U1) / r
    ///   accept if -2 ln(U2) >= x^2
    /// Then return x + r.
    #[cold]
    #[inline(never)]
    fn sample_tail(rng: &mut Pcg64Dxsm) -> f64 {
        loop {
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            // Guard against ln(0)
            if u1 <= 0.0 || u2 <= 0.0 {
                continue;
            }
            let x = -u1.ln() / ZIGGURAT_R;
            if -2.0 * u2.ln() >= x * x {
                return x + ZIGGURAT_R;
            }
        }
    }

    /// Fill a mutable slice with standard normal samples.
    pub fn sample_into(rng: &mut Pcg64Dxsm, buf: &mut [f64]) {
        for slot in buf.iter_mut() {
            *slot = Self::sample(rng);
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
