/// Standard normal CDF using Horner-form rational approximation.
/// Maximum absolute error < 7.5e-8 (Abramowitz & Stegun 26.2.17).
pub fn norm_cdf(x: f64) -> f64 {
    const P: f64 = 0.2316419;
    const B: [f64; 5] = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429];

    let t = 1.0 / (1.0 + P * x.abs());
    let poly = t * (B[0] + t * (B[1] + t * (B[2] + t * (B[3] + t * B[4]))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf_pos = 1.0 - pdf * poly;
    if x >= 0.0 { cdf_pos } else { 1.0 - cdf_pos }
}

/// Inverse standard normal CDF (probit) using rational approximation.
///
/// Based on Peter Acklam's algorithm. Max error < 1.15e-9.
#[allow(dead_code)]
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,  3.224671290700398e-01,
        2.445134137142996e+00,  3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (C[0] + q * (C[1] + q * (C[2] + q * (C[3] + q * (C[4] + q * C[5])))))
            / (1.0 + q * (D[0] + q * (D[1] + q * (D[2] + q * D[3]))))
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (A[0] + r * (A[1] + r * (A[2] + r * (A[3] + r * (A[4] + r * A[5])))))
            / (B[0] + r * (B[1] + r * (B[2] + r * (B[3] + r * (B[4] + r)))))
            * q
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(C[0] + q * (C[1] + q * (C[2] + q * (C[3] + q * (C[4] + q * C[5])))))
            / (1.0 + q * (D[0] + q * (D[1] + q * (D[2] + q * D[3]))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf_roundtrip() {
        for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let p = norm_cdf(x);
            let x2 = norm_ppf(p);
            assert!((x - x2).abs() < 1e-6, "roundtrip failed at x={}", x);
        }
    }

    #[test]
    fn test_norm_cdf_known() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.0) - 0.841344746).abs() < 1e-7);
        assert!((norm_cdf(-1.0) - 0.158655254).abs() < 1e-7);
    }
}
