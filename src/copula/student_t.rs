use crate::dist::normal::NormalSampler;
use crate::prng::Pcg64Dxsm;
use crate::copula::gaussian::cholesky;
use ndarray::{Array2, ArrayView2};

/// Student-t CDF via regularized incomplete beta function (Abramowitz & Stegun).
/// Maximum error ~1e-5 for |x| < 10, sufficient for copula transforms.
fn student_t_cdf(x: f64, nu: f64) -> f64 {
    let z = nu / (nu + x * x);
    let ibeta_half = regularized_incomplete_beta(z, nu / 2.0, 0.5);
    if x >= 0.0 {
        1.0 - 0.5 * ibeta_half
    } else {
        0.5 * ibeta_half
    }
}

/// Regularized incomplete beta I_x(a, b) via continued fraction (modified Lentz).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let log_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - log_beta).exp() / a;
    front * beta_cf(x, a, b)
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-7;
    const FPMIN: f64 = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = (1.0 - qab * x / qap).max(FPMIN);
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = (1.0 + aa * d).abs().max(FPMIN);
        c = (1.0 + aa / c).abs().max(FPMIN);
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = (1.0 + aa * d).abs().max(FPMIN);
        c = (1.0 + aa / c).abs().max(FPMIN);
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS { break; }
    }
    h
}

/// Log-gamma via Lanczos approximation (g=7, 9 coefficients).
fn lgamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let mut x = x;
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - lgamma(1.0 - x);
    }
    x -= 1.0;
    let mut acc = C[0];
    for i in 1..9 {
        acc += C[i] / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

/// Sample from Gamma(shape, scale=1) using Marsaglia-Tsang (2000) squeeze method.
/// Works for shape >= 1. For shape < 1 uses Ahrens-Dieter reduction.
fn sample_gamma(rng: &mut Pcg64Dxsm, shape: f64) -> f64 {
    if shape < 1.0 {
        // Reduction: Gamma(a) = Gamma(a+1) * U^(1/a)
        let u = rng.next_f64();
        return sample_gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = NormalSampler::sample(rng);
        let v_inner = 1.0 + c * x;
        if v_inner <= 0.0 { continue; }
        let v = v_inner * v_inner * v_inner;
        let u = rng.next_f64();
        // Squeeze test.
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample from chi-squared(nu) = Gamma(nu/2, scale=2).
fn sample_chi_sq(rng: &mut Pcg64Dxsm, nu: f64) -> f64 {
    2.0 * sample_gamma(rng, nu / 2.0)
}

/// Simulate samples from a Student-t copula.
///
/// Steps:
/// 1. Cholesky decompose Σ.
/// 2. Draw correlated N(0,1) vectors X = L × Z.
/// 3. Scale by √(ν/χ²(ν)) to get t-distributed marginals.
/// 4. Apply Student-t CDF → uniform [0, 1].
///
/// Returns array of shape `(n_samples, dim)` in (0, 1).
pub fn student_t_copula_samples(
    corr: ArrayView2<f64>,
    nu: f64,
    n_samples: usize,
    seed: u128,
) -> Result<Array2<f64>, String> {
    // Also validated in the Python binding, but retained here so the Rust API
    // is safe when called directly without going through PyO3.
    if nu <= 2.0 {
        return Err("nu must be > 2 for finite variance".into());
    }
    let dim = corr.nrows();
    if corr.ncols() != dim {
        return Err("correlation matrix must be square".into());
    }
    let l = cholesky(corr)?;
    let mut rng = Pcg64Dxsm::new(seed);
    let mut out = Array2::<f64>::zeros([n_samples, dim]);
    let mut z = vec![0.0f64; dim];

    for i in 0..n_samples {
        let chi_sq = sample_chi_sq(&mut rng, nu);
        let scale = (nu / chi_sq).sqrt();
        NormalSampler::sample_into(&mut rng, &mut z);
        for j in 0..dim {
            let mut xj = 0.0;
            for k in 0..=j {
                xj += l[[j, k]] * z[k];
            }
            out[[i, j]] = student_t_cdf(xj * scale, nu);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_student_t_cdf_symmetry() {
        let nu = 5.0;
        assert!((student_t_cdf(0.0, nu) - 0.5).abs() < 1e-6);
        assert!((student_t_cdf(1.0, nu) + student_t_cdf(-1.0, nu) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_student_t_copula_shape() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        let samples = student_t_copula_samples(corr.view(), 5.0, 500, 42).unwrap();
        assert_eq!(samples.shape(), [500, 2]);
        for &v in samples.iter() {
            assert!(v > 0.0 && v < 1.0, "out of (0,1): {}", v);
        }
    }

    #[test]
    fn test_sample_gamma_mean() {
        let mut rng = Pcg64Dxsm::new(0);
        let shape = 3.0;
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| sample_gamma(&mut rng, shape)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        assert!((mean - shape).abs() < 0.1, "mean={}", mean);
    }
}
