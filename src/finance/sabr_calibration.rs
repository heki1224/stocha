/// SABR calibration: fit (alpha, rho, nu) to an observed implied-vol smile.
///
/// Algorithm:
///   1. ATM alpha is recovered by 1-D Brent root-finding (reduces the search
///      to (rho, nu) only — Hagan ATM formula is monotone in alpha).
///   2. (rho, nu) are fit by a Projected Levenberg-Marquardt loop with
///      central-difference Jacobian. Box constraints are enforced by step
///      clipping (no logit transform — keeps the gradient well-behaved).
///   3. NaN/Inf in residuals trigger a large penalty so LM rolls the step back.
///
/// Beta is held fixed (industry standard: jointly identifying beta and rho
/// is ill-posed for typical smile data).
use super::sabr::sabr_implied_vol;

pub struct CalibrationResult {
    pub alpha: f64,
    pub rho: f64,
    pub nu: f64,
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

const RHO_BOUND: f64 = 0.999;
const NU_MIN: f64 = 1e-8;
const NU_MAX: f64 = 5.0;
const ALPHA_MIN: f64 = 1e-8;
const ALPHA_MAX: f64 = 5.0;

pub fn calibrate(
    strikes: &[f64],
    market_vols: &[f64],
    f: f64,
    t: f64,
    beta: f64,
    shift: f64,
    max_iter: usize,
    tol: f64,
) -> Result<CalibrationResult, String> {
    if strikes.len() != market_vols.len() {
        return Err("strikes and market_vols must have the same length".into());
    }
    if strikes.len() < 3 {
        return Err("at least 3 strike/vol pairs are required".into());
    }
    if t <= 0.0 {
        return Err("t must be positive".into());
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err("beta must be in [0, 1]".into());
    }
    if f + shift <= 0.0 {
        return Err(format!(
            "shifted forward (f + shift = {}) must be positive",
            f + shift
        ));
    }
    for (i, (k, v)) in strikes.iter().zip(market_vols.iter()).enumerate() {
        if !k.is_finite() || !v.is_finite() {
            return Err(format!("non-finite input at index {}", i));
        }
        if k + shift <= 0.0 {
            return Err(format!(
                "shifted strike at index {} (k + shift = {}) must be positive",
                i,
                k + shift
            ));
        }
        if *v <= 0.0 {
            return Err(format!("market_vols[{}] must be positive", i));
        }
    }

    // ATM proxy: closest strike to the forward.
    let (atm_idx, _) = strikes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (**a - f).abs().total_cmp(&(**b - f).abs()))
        .unwrap();
    let sigma_atm = market_vols[atm_idx];

    // Initial guesses: rho_0 = 0, nu_0 = 0.3 (classic warm start).
    let mut rho = 0.0_f64;
    let mut nu = 0.3_f64;
    let mut alpha = solve_alpha_atm(f, t, beta, rho, nu, shift, sigma_atm)
        .unwrap_or((sigma_atm * (f + shift).powf(1.0 - beta)).clamp(ALPHA_MIN, ALPHA_MAX));

    let n = strikes.len();
    let cost_at = |alpha: f64, rho: f64, nu: f64| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            let r = match sabr_implied_vol(f, strikes[i], t, alpha, beta, rho, nu, shift) {
                Ok(v) => v - market_vols[i],
                Err(_) => return f64::MAX / 4.0,
            };
            if !r.is_finite() {
                return f64::MAX / 4.0;
            }
            s += r * r;
        }
        0.5 * s
    };

    let mut cost = cost_at(alpha, rho, nu);
    let mut lambda = 1e-3_f64;
    let mut iterations = 0;
    let mut converged = false;

    let h_rho = 1e-6;
    let h_nu = 1e-6;

    for it in 0..max_iter {
        iterations = it + 1;

        // Build residual vector and Jacobian (n x 2) via central differences.
        let mut r_vec = vec![0.0_f64; n];
        let mut j_rho = vec![0.0_f64; n];
        let mut j_nu = vec![0.0_f64; n];

        for i in 0..n {
            r_vec[i] = match sabr_implied_vol(f, strikes[i], t, alpha, beta, rho, nu, shift) {
                Ok(v) => v - market_vols[i],
                Err(_) => {
                    return Err("Hagan formula failed at current parameters".into());
                }
            };
        }

        // ∂σ/∂ρ
        let rho_p = (rho + h_rho).min(RHO_BOUND);
        let rho_m = (rho - h_rho).max(-RHO_BOUND);
        let alpha_rp = solve_alpha_atm(f, t, beta, rho_p, nu, shift, sigma_atm).unwrap_or(alpha);
        let alpha_rm = solve_alpha_atm(f, t, beta, rho_m, nu, shift, sigma_atm).unwrap_or(alpha);
        for i in 0..n {
            let vp = sabr_implied_vol(f, strikes[i], t, alpha_rp, beta, rho_p, nu, shift)
                .unwrap_or(market_vols[i]);
            let vm = sabr_implied_vol(f, strikes[i], t, alpha_rm, beta, rho_m, nu, shift)
                .unwrap_or(market_vols[i]);
            j_rho[i] = (vp - vm) / (rho_p - rho_m);
        }

        // ∂σ/∂ν
        let nu_p = (nu + h_nu).min(NU_MAX);
        let nu_m = (nu - h_nu).max(NU_MIN);
        let alpha_np = solve_alpha_atm(f, t, beta, rho, nu_p, shift, sigma_atm).unwrap_or(alpha);
        let alpha_nm = solve_alpha_atm(f, t, beta, rho, nu_m, shift, sigma_atm).unwrap_or(alpha);
        for i in 0..n {
            let vp = sabr_implied_vol(f, strikes[i], t, alpha_np, beta, rho, nu_p, shift)
                .unwrap_or(market_vols[i]);
            let vm = sabr_implied_vol(f, strikes[i], t, alpha_nm, beta, rho, nu_m, shift)
                .unwrap_or(market_vols[i]);
            j_nu[i] = (vp - vm) / (nu_p - nu_m);
        }

        // Build 2x2 normal-equation matrix A = JᵀJ + λ·diag(JᵀJ); g = Jᵀr.
        let mut a00 = 0.0;
        let mut a01 = 0.0;
        let mut a11 = 0.0;
        let mut g0 = 0.0;
        let mut g1 = 0.0;
        for i in 0..n {
            a00 += j_rho[i] * j_rho[i];
            a01 += j_rho[i] * j_nu[i];
            a11 += j_nu[i] * j_nu[i];
            g0 += j_rho[i] * r_vec[i];
            g1 += j_nu[i] * r_vec[i];
        }
        let m00 = a00 * (1.0 + lambda);
        let m11 = a11 * (1.0 + lambda);
        let det = m00 * m11 - a01 * a01;
        if det.abs() < 1e-30 {
            // Degenerate Jacobian: bump damping and retry.
            lambda = (lambda * 10.0).min(1e12);
            if lambda >= 1e12 {
                break;
            }
            continue;
        }
        let dp_rho = -(m11 * g0 - a01 * g1) / det;
        let dp_nu = -(-a01 * g0 + m00 * g1) / det;

        let new_rho = (rho + dp_rho).clamp(-RHO_BOUND, RHO_BOUND);
        let new_nu = (nu + dp_nu).clamp(NU_MIN, NU_MAX);
        let new_alpha =
            solve_alpha_atm(f, t, beta, new_rho, new_nu, shift, sigma_atm).unwrap_or(alpha);
        let new_cost = cost_at(new_alpha, new_rho, new_nu);

        if new_cost < cost {
            let dcost = cost - new_cost;
            let dpinf = dp_rho.abs().max(dp_nu.abs());
            rho = new_rho;
            nu = new_nu;
            alpha = new_alpha;
            cost = new_cost;
            lambda = (lambda / 10.0).max(1e-12);
            if dcost < tol * (cost + tol) || dpinf < tol {
                converged = true;
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
    }

    let rmse = (2.0 * cost / n as f64).sqrt();

    Ok(CalibrationResult {
        alpha,
        rho,
        nu,
        rmse,
        iterations,
        converged,
    })
}

/// Solve σ_Hagan(F, F, T, α, β, ρ, ν, shift) = σ_atm for α via Brent's method.
fn solve_alpha_atm(
    f: f64,
    t: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    shift: f64,
    sigma_atm: f64,
) -> Result<f64, String> {
    let g = |a: f64| -> f64 {
        match sabr_implied_vol(f, f, t, a, beta, rho, nu, shift) {
            Ok(v) => v - sigma_atm,
            Err(_) => f64::NAN,
        }
    };
    let mut lo = ALPHA_MIN;
    let mut hi = ALPHA_MAX;
    let mut flo = g(lo);
    let mut fhi = g(hi);
    if !flo.is_finite() || !fhi.is_finite() {
        return Err("Hagan ATM evaluation produced NaN at bracket".into());
    }
    if flo * fhi > 0.0 {
        return Err("alpha bracket does not straddle the root".into());
    }

    // Brent's method (simplified — bisection + secant fallback is enough here).
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let fmid = g(mid);
        if !fmid.is_finite() {
            return Err("Hagan ATM evaluation produced NaN".into());
        }
        if fmid.abs() < 1e-14 || (hi - lo) < 1e-14 {
            return Ok(mid);
        }
        if flo * fmid < 0.0 {
            hi = mid;
            fhi = fmid;
        } else {
            lo = mid;
            flo = fmid;
        }
        let _ = fhi; // silence unused-write lints in some configs
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_atm_recovery() {
        // β=1: σ_ATM ≈ α (lognormal). Recover α from σ_ATM.
        let sigma_atm = sabr_implied_vol(0.05, 0.05, 1.0, 0.20, 1.0, -0.3, 0.4, 0.0).unwrap();
        let a = solve_alpha_atm(0.05, 1.0, 1.0, -0.3, 0.4, 0.0, sigma_atm).unwrap();
        assert!((a - 0.20).abs() < 1e-6, "got alpha={}", a);
    }

    #[test]
    fn test_round_trip_lognormal() {
        // Generate a synthetic smile with β=1 and recover (α, ρ, ν).
        let f = 0.05;
        let t = 1.0;
        let beta = 1.0;
        let alpha = 0.20;
        let rho = -0.3;
        let nu = 0.4;
        let strikes: Vec<f64> = (-3..=3).map(|i| f * (1.0 + 0.05 * i as f64)).collect();
        let vols: Vec<f64> = strikes
            .iter()
            .map(|&k| sabr_implied_vol(f, k, t, alpha, beta, rho, nu, 0.0).unwrap())
            .collect();
        let res = calibrate(&strikes, &vols, f, t, beta, 0.0, 200, 1e-12).unwrap();
        assert!(res.rmse < 1e-6, "rmse={}", res.rmse);
        assert!(
            (res.alpha - alpha).abs() / alpha < 0.01,
            "alpha={}",
            res.alpha
        );
        assert!((res.rho - rho).abs() < 0.05, "rho={}", res.rho);
        assert!((res.nu - nu).abs() < 0.05, "nu={}", res.nu);
    }
}
