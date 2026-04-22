/// SABR model implied volatility using Hagan et al. (2002) approximation.
///
/// Model: dF = σ F^β dW₁,  dσ = α σ dW₂,  ⟨dW₁,dW₂⟩ = ρ dt
///
/// Negative rate support via the Shifted SABR approach:
/// Replace F → F + shift, K → K + shift before applying the formula.
///
/// Computes the Black (lognormal) implied volatility σ_B(F, K, T).
pub fn sabr_implied_vol(
    f: f64,       // Forward price / rate.
    k: f64,       // Strike.
    t: f64,       // Time to expiry (years).
    alpha: f64,   // Initial vol-of-vol (must be > 0).
    beta: f64,    // CEV exponent in [0, 1].
    rho: f64,     // Correlation between F and σ Brownians (in (-1, 1)).
    nu: f64,      // Vol-of-vol (must be >= 0).
    shift: f64,   // Shift for negative-rate support (default 0.0).
) -> Result<f64, String> {
    if alpha <= 0.0 {
        return Err("alpha must be positive".into());
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err("beta must be in [0, 1]".into());
    }
    if rho <= -1.0 || rho >= 1.0 {
        return Err("rho must be in (-1, 1)".into());
    }
    if nu < 0.0 {
        return Err("nu must be non-negative".into());
    }
    if t <= 0.0 {
        return Err("t must be positive".into());
    }

    let fs = f + shift;
    let ks = k + shift;

    if fs <= 0.0 || ks <= 0.0 {
        return Err(format!(
            "shifted forward ({}) and strike ({}) must be positive; increase shift",
            fs, ks
        ));
    }

    // At-the-money approximation when F ≈ K.
    if (fs - ks).abs() < 1e-10 * fs.abs().max(1e-10) {
        return Ok(sabr_atm(fs, t, alpha, beta, rho, nu));
    }

    // General case (Hagan 2002, eq. 2.17a–b).
    let fk_mid = (fs * ks).sqrt();
    let log_fk = (fs / ks).ln();

    // Leading term denominator.
    let one_minus_beta = 1.0 - beta;
    let fk_beta = fk_mid.powf(one_minus_beta);

    // z and χ(z) terms.
    let z = if nu.abs() < 1e-14 {
        // ν → 0 limit: z/χ(z) → 1.
        1.0
    } else {
        let z_raw = (nu / alpha) * fk_beta * log_fk;
        // χ(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]  (Hagan 2002 eq. A.3b)
        let chi = (((1.0 - 2.0 * rho * z_raw + z_raw * z_raw).sqrt() + z_raw - rho)
            / (1.0 - rho)).ln();
        if chi.abs() < 1e-14 { 1.0 } else { z_raw / chi }
    };

    // Correction terms.
    let log_fk_sq = log_fk * log_fk;
    let omβ2 = one_minus_beta * one_minus_beta;
    let gamma1 = 2.0 * beta - 1.0;
    let correction_num = 1.0
        + (omβ2 / 24.0) * log_fk_sq
        + (omβ2 * omβ2 / 1920.0) * log_fk_sq * log_fk_sq;

    let a_term = (omβ2 / 24.0) * alpha * alpha / (fk_beta * fk_beta);
    let b_term = 0.25 * rho * beta * nu * alpha / fk_beta;
    let c_term = (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu;
    let time_correction = 1.0 + (a_term + b_term + c_term) * t;

    // Compute the final implied vol.
    let sigma_b = alpha / (fk_beta * correction_num) * z * time_correction;

    if sigma_b <= 0.0 || sigma_b.is_nan() || sigma_b.is_infinite() {
        return Err(format!(
            "SABR produced invalid implied vol: {} (check parameters)",
            sigma_b
        ));
    }

    // Suppress unused variable warning.
    let _ = gamma1;

    Ok(sigma_b)
}

/// ATM SABR approximation (Hagan 2002, eq. 2.17a).
fn sabr_atm(f: f64, t: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> f64 {
    let one_minus_beta = 1.0 - beta;
    let f_beta = f.powf(one_minus_beta);

    let a_term = (one_minus_beta * one_minus_beta / 24.0) * alpha * alpha / (f_beta * f_beta);
    let b_term = 0.25 * rho * beta * nu * alpha / f_beta;
    let c_term = (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu;

    (alpha / f_beta) * (1.0 + (a_term + b_term + c_term) * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sabr_atm_beta_one() {
        // β=1 (lognormal): formula simplifies to approximately alpha.
        let sigma = sabr_implied_vol(0.05, 0.05, 1.0, 0.20, 1.0, -0.3, 0.4, 0.0).unwrap();
        // With beta=1, the leading term is alpha/F^0 = alpha.
        assert!(sigma > 0.0 && sigma < 1.0, "sigma={}", sigma);
    }

    #[test]
    fn test_sabr_shifted_negative_rate() {
        // Negative forward; shifted by 0.03 so shifted-F = 0.01 > 0.
        let sigma = sabr_implied_vol(-0.02, -0.03, 1.0, 0.20, 0.5, -0.3, 0.4, 0.03).unwrap();
        assert!(sigma > 0.0, "sigma={}", sigma);
    }

    #[test]
    fn test_sabr_otm_near_atm() {
        let atm = sabr_implied_vol(0.05, 0.05, 1.0, 0.20, 0.5, -0.3, 0.4, 0.0).unwrap();
        let otm = sabr_implied_vol(0.05, 0.06, 1.0, 0.20, 0.5, -0.3, 0.4, 0.0).unwrap();
        // OTM vol should be close to ATM for small strikes.
        assert!((atm - otm).abs() < 0.05, "atm={}, otm={}", atm, otm);
    }
}
