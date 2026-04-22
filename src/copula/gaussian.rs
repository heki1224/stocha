use crate::dist::norm_cdf;
use crate::dist::normal::NormalSampler;
use crate::prng::Pcg64Dxsm;
use ndarray::{Array2, ArrayView2};

/// Simulate samples from a Gaussian copula.
///
/// Steps:
/// 1. Cholesky decompose the correlation matrix Σ.
/// 2. Draw independent N(0,1) vectors Z.
/// 3. Compute correlated normals X = L × Z.
/// 4. Transform each marginal via Φ (normal CDF) → uniform [0,1].
///
/// Returns array of shape `(n_samples, dim)` in [0, 1].
pub fn gaussian_copula_samples(
    corr: ArrayView2<f64>,
    n_samples: usize,
    seed: u128,
) -> Result<Array2<f64>, String> {
    let dim = corr.nrows();
    if corr.ncols() != dim {
        return Err("correlation matrix must be square".into());
    }

    // Cholesky decomposition (lower triangular L such that L L^T = Σ).
    let l = cholesky(corr)?;

    let mut rng = Pcg64Dxsm::new(seed);
    let mut out = Array2::<f64>::zeros([n_samples, dim]);

    // Pre-allocate scratch buffer for independent normals.
    let mut z = vec![0.0f64; dim];

    for i in 0..n_samples {
        // Sample independent N(0,1).
        NormalSampler::sample_into(&mut rng, &mut z);
        // Multiply by L: x = L * z.
        for j in 0..dim {
            let mut xj = 0.0;
            for k in 0..=j {
                xj += l[[j, k]] * z[k];
            }
            // Transform to uniform via normal CDF.
            out[[i, j]] = norm_cdf(xj);
        }
    }

    Ok(out)
}

/// Lower-triangular Cholesky decomposition. Returns L s.t. L * L^T = A.
pub(crate) fn cholesky(a: ArrayView2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros([n, n]);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "matrix is not positive-definite (diagonal element {} = {})",
                        i, sum
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gaussian_copula_shape() {
        let corr = array![[1.0, 0.8], [0.8, 1.0]];
        let samples = gaussian_copula_samples(corr.view(), 1000, 42).unwrap();
        assert_eq!(samples.shape(), [1000, 2]);
        // Values must be in [0, 1].
        for &v in samples.iter() {
            assert!(v > 0.0 && v < 1.0, "out of range: {}", v);
        }
    }

    #[test]
    fn test_cholesky_identity() {
        let eye = array![[1.0, 0.0], [0.0, 1.0]];
        let l = cholesky(eye.view()).unwrap();
        assert!((l[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((l[[1, 1]] - 1.0).abs() < 1e-12);
    }
}
