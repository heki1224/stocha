use ndarray::Array2;

/// First 40 primes (all fit in u8) to support up to 40 dimensions.
const PRIMES: [u8; 40] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
];

/// Maximum number of dimensions supported by the Halton generator.
pub const HALTON_MAX_DIM: usize = PRIMES.len();

/// Generate `n_samples` points from a Halton low-discrepancy sequence.
///
/// Uses consecutive primes (2, 3, 5, ...) as the base for each dimension.
/// Supports up to 40 dimensions.
///
/// # Arguments
/// * `dim`      — Number of dimensions (1 to 40).
/// * `n_samples`— Number of sample points to generate.
/// * `skip`     — Number of initial sequence elements to skip (warm-up).
///
/// # Returns
/// An `Array2<f64>` of shape `(n_samples, dim)` with values in (0, 1).
pub fn halton_sequence(dim: usize, n_samples: usize, skip: usize) -> Result<Array2<f64>, String> {
    if dim == 0 {
        return Err("dim must be at least 1".to_string());
    }
    if dim > HALTON_MAX_DIM {
        return Err(format!(
            "dim must be <= {} (number of supported prime bases), got {}",
            HALTON_MAX_DIM, dim
        ));
    }
    if n_samples == 0 {
        return Ok(Array2::<f64>::zeros((0, dim)));
    }

    let mut result = Array2::<f64>::zeros((n_samples, dim));
    for d in 0..dim {
        let seq: Vec<f64> = halton::Sequence::new(PRIMES[d])
            .skip(skip)
            .take(n_samples)
            .collect();
        for (i, v) in seq.iter().enumerate() {
            result[[i, d]] = *v;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let arr = halton_sequence(3, 10, 0).unwrap();
        assert_eq!(arr.shape(), &[10, 3]);
    }

    #[test]
    fn test_values_in_unit_interval() {
        let arr = halton_sequence(4, 100, 0).unwrap();
        assert!(arr.iter().all(|&v| v > 0.0 && v < 1.0));
    }

    #[test]
    fn test_skip_works() {
        let full = halton_sequence(2, 20, 0).unwrap();
        let skipped = halton_sequence(2, 10, 10).unwrap();
        // skipped[i] should equal full[10 + i]
        for i in 0..10 {
            assert!((full[[10 + i, 0]] - skipped[[i, 0]]).abs() < 1e-15);
            assert!((full[[10 + i, 1]] - skipped[[i, 1]]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_deterministic() {
        let a = halton_sequence(2, 50, 0).unwrap();
        let b = halton_sequence(2, 50, 0).unwrap();
        assert_eq!(a, b);
    }
}
