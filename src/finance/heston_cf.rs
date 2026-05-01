use num_complex::Complex64;

/// Heston characteristic function using the Albrecher et al. (2007) formulation.
///
/// This "second formulation" avoids the branch-cut discontinuity of the
/// original Heston (1993) complex logarithm by ensuring the integrand
/// stays within the principal branch.
///
/// Returns φ(u) = E[exp(i·u·ln(S_T))]
pub fn heston_cf(
    u: Complex64,
    s0: f64,
    v0: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    tau: f64,
) -> Complex64 {
    let i = Complex64::i();
    let xi2 = xi * xi;

    let alpha = -0.5 * u * (u + i);
    let beta = Complex64::new(kappa, 0.0) - i * rho * xi * u;

    let d = (beta * beta - 2.0 * xi2 * alpha).sqrt();

    let g = (beta - d) / (beta + d);
    let exp_neg_d_tau = (-d * tau).exp();

    // Albrecher formulation: ln((1 - g·e^{-dτ}) / (1 - g)) computed as a single ln(A/B)
    let log_term = ((1.0 - g * exp_neg_d_tau) / (1.0 - g)).ln();

    let c = i * u * r * tau + (kappa * theta / xi2) * ((beta - d) * tau - 2.0 * log_term);

    let dd = (v0 / xi2) * (beta - d) * (1.0 - exp_neg_d_tau) / (1.0 - g * exp_neg_d_tau);

    (c + dd + i * u * s0.ln()).exp()
}

/// Compute the first four cumulants of log(S_T) under the Heston model.
/// Used for COS method truncation range.
/// Returns (c1, c2, c4).
pub fn heston_cumulants(
    s0: f64,
    v0: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    tau: f64,
) -> (f64, f64, f64) {
    let ekt = (-kappa * tau).exp();

    let c1 = s0.ln() + (r - 0.5 * theta) * tau + (1.0 - ekt) * (v0 - theta) / (2.0 * kappa);

    let xi2 = xi * xi;
    let kappa2 = kappa * kappa;
    let kappa3 = kappa2 * kappa;

    let c2 = (1.0 / (8.0 * kappa3))
        * (xi * tau * kappa * ekt * (v0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
            + kappa * rho * xi * (1.0 - ekt) * (16.0 * theta - 8.0 * v0)
            + 2.0 * theta * kappa * tau * (-4.0 * kappa * rho * xi + xi2 + 4.0 * kappa2)
            + xi2 * ((theta - 2.0 * v0) * ekt.powi(2) + theta * (6.0 * ekt - 7.0) + 2.0 * v0)
            + 8.0 * kappa2 * (v0 - theta) * (1.0 - ekt));

    // Fourth cumulant (simplified approximation for truncation range)
    let c4 = c2 * c2 * 0.5;

    (c1, c2.max(1e-16), c4.max(1e-32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cf_at_zero() {
        // φ(0) = 1 for any model
        let phi = heston_cf(
            Complex64::new(0.0, 0.0),
            100.0,
            0.04,
            0.05,
            2.0,
            0.04,
            0.3,
            -0.7,
            1.0,
        );
        assert!((phi.re - 1.0).abs() < 1e-12);
        assert!(phi.im.abs() < 1e-12);
    }

    #[test]
    fn test_cf_martingale() {
        // E[S_T] = S0·e^{rT} → φ(-i) = e^{rT}·S0... actually
        // φ(-i) = E[e^{ln S_T}] = E[S_T] = S0·e^{rT}
        let s0 = 100.0;
        let r = 0.05;
        let tau = 1.0;
        let phi = heston_cf(
            Complex64::new(0.0, -1.0),
            s0,
            0.04,
            r,
            2.0,
            0.04,
            0.3,
            -0.7,
            tau,
        );
        let expected = s0 * (r * tau).exp();
        assert!((phi.re - expected).abs() / expected < 1e-10);
        assert!(phi.im.abs() < 1e-10);
    }
}
