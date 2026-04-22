use ndarray::Array2;
use sobol_qmc::{params::JoeKuoD6, Sobol};

/// Generate `n_samples` points from a Sobol low-discrepancy sequence.
///
/// Uses Joe & Kuo 2008 direction numbers (up to 1,000 dimensions with
/// `STANDARD`, up to 21,201 with `EXTENDED`).
///
/// # Returns
/// An `Array2<f64>` of shape `(n_samples, dim)` with values in [0, 1).
pub fn sobol_sequence(dim: usize, n_samples: usize) -> Result<Array2<f64>, String> {
    if dim == 0 {
        return Err("dim must be at least 1".to_string());
    }
    if n_samples == 0 {
        return Ok(Array2::<f64>::zeros((0, dim)));
    }

    let seq = Sobol::<f64>::new(dim, &JoeKuoD6::STANDARD).map_err(|e| e.to_string())?;
    let mut result = Array2::<f64>::zeros((n_samples, dim));
    for (i, point) in seq.take(n_samples).enumerate() {
        for (j, v) in point.iter().enumerate() {
            result[[i, j]] = *v;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let arr = sobol_sequence(3, 10).unwrap();
        assert_eq!(arr.shape(), &[10, 3]);
    }

    #[test]
    fn test_values_in_unit_interval() {
        let arr = sobol_sequence(4, 100).unwrap();
        assert!(arr.iter().all(|&v| v >= 0.0 && v < 1.0));
    }

    #[test]
    fn test_deterministic() {
        let a = sobol_sequence(2, 50).unwrap();
        let b = sobol_sequence(2, 50).unwrap();
        assert_eq!(a, b);
    }
}
