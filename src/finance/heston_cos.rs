use num_complex::Complex64;
use std::f64::consts::PI;

use super::heston_cf::{heston_cf, heston_cumulants};

const DEFAULT_L: f64 = 12.0;
const MIN_RANGE_WIDTH: f64 = 0.5;
const MAX_RANGE_WIDTH: f64 = 50.0;

/// χ_k(c, d; a, b) = ∫_c^d e^y cos(kπ(y-a)/(b-a)) dy
fn chi_k(k: usize, c: f64, d: f64, a: f64, b: f64) -> f64 {
    let ba = b - a;
    if k == 0 {
        d.exp() - c.exp()
    } else {
        let w = k as f64 * PI / ba;
        let w2 = w * w;
        let denom = 1.0 + w2;
        let cos_d = (w * (d - a)).cos();
        let sin_d = (w * (d - a)).sin();
        let cos_c = (w * (c - a)).cos();
        let sin_c = (w * (c - a)).sin();
        (d.exp() * (cos_d + w * sin_d) - c.exp() * (cos_c + w * sin_c)) / denom
    }
}

/// ψ_k(c, d; a, b) = ∫_c^d cos(kπ(y-a)/(b-a)) dy
fn psi_k(k: usize, c: f64, d: f64, a: f64, b: f64) -> f64 {
    if k == 0 {
        d - c
    } else {
        let ba = b - a;
        let w = k as f64 * PI / ba;
        ((w * (d - a)).sin() - (w * (c - a)).sin()) / w
    }
}

/// Truncation range [a, b] for the density of ln(S_T).
fn truncation_range(
    s0: f64,
    v0: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    tau: f64,
) -> (f64, f64) {
    let (c1, c2, c4) = heston_cumulants(s0, v0, r, kappa, theta, xi, rho, tau);
    let width = DEFAULT_L * (c2 + c4.sqrt()).sqrt();
    let a = c1 - width;
    let b = c1 + width;

    let actual_width = b - a;
    if actual_width < MIN_RANGE_WIDTH {
        let mid = 0.5 * (a + b);
        (mid - MIN_RANGE_WIDTH * 0.5, mid + MIN_RANGE_WIDTH * 0.5)
    } else if actual_width > MAX_RANGE_WIDTH {
        let mid = 0.5 * (a + b);
        (mid - MAX_RANGE_WIDTH * 0.5, mid + MAX_RANGE_WIDTH * 0.5)
    } else {
        (a, b)
    }
}

/// Price European options using the COS method (Fang & Oosterlee 2008).
///
/// Works in the ln(S_T) space:
/// - Truncation range [a, b] is for the density of ln(S_T)
/// - CF cache is shared across all strikes
/// - Payoff coefficients use ln(K) as the exercise boundary
///
/// Call: V_k = (2/(b-a)) · [χ_k(ln K, b) - K·ψ_k(ln K, b)]
/// Put:  V_k = (2/(b-a)) · [-χ_k(a, ln K) + K·ψ_k(a, ln K)]
pub fn heston_cos_price_vec(
    s0: f64,
    v0: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    tau: f64,
    strikes: &[f64],
    is_call: &[bool],
    n_cos: usize,
) -> Vec<f64> {
    let (a, b) = truncation_range(s0, v0, r, kappa, theta, xi, rho, tau);
    let ba = b - a;
    let df = (-r * tau).exp();

    // CF of ln(S_T) at COS frequencies — shared across all strikes
    let cf_cache: Vec<Complex64> = (0..n_cos)
        .map(|k| {
            let u_k = k as f64 * PI / ba;
            let cf = heston_cf(
                Complex64::new(u_k, 0.0),
                s0,
                v0,
                r,
                kappa,
                theta,
                xi,
                rho,
                tau,
            );
            // Multiply by exp(-i·u_k·a) for COS shift
            cf * (Complex64::i() * u_k * (-a)).exp()
        })
        .collect();

    strikes
        .iter()
        .zip(is_call.iter())
        .map(|(&strike, &call)| {
            let ln_k = strike.ln();
            let mut sum = 0.0;

            for k in 0..n_cos {
                let vk = if call {
                    // V_k = (2/(b-a)) * (chi_k(ln K, b) - K * psi_k(ln K, b))
                    let c = chi_k(k, ln_k, b, a, b);
                    let p = psi_k(k, ln_k, b, a, b);
                    2.0 / ba * (c - strike * p)
                } else {
                    // V_k = (2/(b-a)) * (-chi_k(a, ln K) + K * psi_k(a, ln K))
                    let c = chi_k(k, a, ln_k, a, b);
                    let p = psi_k(k, a, ln_k, a, b);
                    2.0 / ba * (-c + strike * p)
                };

                let weight = if k == 0 { 0.5 } else { 1.0 };
                sum += weight * cf_cache[k].re * vk;
            }

            (df * sum).max(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heston_cos_price(
        s0: f64,
        v0: f64,
        r: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
        tau: f64,
        strike: f64,
        is_call: bool,
        n_cos: usize,
    ) -> f64 {
        heston_cos_price_vec(
            s0,
            v0,
            r,
            kappa,
            theta,
            xi,
            rho,
            tau,
            &[strike],
            &[is_call],
            n_cos,
        )[0]
    }

    #[test]
    fn test_cos_vs_bs_limit() {
        // When xi → 0, Heston → BS with sigma = sqrt(v0)
        let s0 = 100.0;
        let v0 = 0.04;
        let r = 0.05;
        let kappa = 2.0;
        let theta = 0.04;
        let xi = 1e-6;
        let rho = 0.0;
        let tau = 1.0;
        let k = 100.0;

        let call_cos = heston_cos_price(s0, v0, r, kappa, theta, xi, rho, tau, k, true, 256);
        let sigma = v0.sqrt();
        let call_bs = crate::finance::bs::bs_price(s0, k, r, tau, sigma, true);

        let rel_err = (call_cos - call_bs).abs() / call_bs;
        assert!(
            rel_err < 1e-3,
            "COS vs BS: rel_err={rel_err:.2e}, COS={call_cos:.6}, BS={call_bs:.6}"
        );
    }

    #[test]
    fn test_put_call_parity_cos() {
        let s0 = 100.0;
        let v0 = 0.04;
        let r = 0.05;
        let kappa = 2.0;
        let theta = 0.04;
        let xi = 0.3;
        let rho = -0.7;
        let tau = 1.0;
        let k = 100.0;

        let call = heston_cos_price(s0, v0, r, kappa, theta, xi, rho, tau, k, true, 256);
        let put = heston_cos_price(s0, v0, r, kappa, theta, xi, rho, tau, k, false, 256);
        let parity = call - put - s0 + k * (-r * tau).exp();
        assert!(
            parity.abs() < 0.05,
            "Put-call parity violation: {parity:.6} (call={call:.4}, put={put:.4})"
        );
    }

    #[test]
    fn test_cos_atm_call_reasonable() {
        // ATM call S0=100, K=100, r=5%, T=1, v0=0.04 (sigma≈20%)
        // Should be around 10-12 (BS ATM call ≈ 10.45)
        let price = heston_cos_price(
            100.0, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0, 100.0, true, 160,
        );
        assert!(
            price > 5.0 && price < 20.0,
            "ATM call price unreasonable: {price:.4}"
        );
    }

    #[test]
    fn test_cos_price_positive() {
        let s0 = 100.0;
        let v0 = 0.04;
        let r = 0.05;
        let kappa = 2.0;
        let theta = 0.04;
        let xi = 0.3;
        let rho = -0.7;
        let tau = 1.0;

        for &k in &[80.0, 90.0, 100.0, 110.0, 120.0] {
            let call = heston_cos_price(s0, v0, r, kappa, theta, xi, rho, tau, k, true, 160);
            let put = heston_cos_price(s0, v0, r, kappa, theta, xi, rho, tau, k, false, 160);
            assert!(call >= 0.0, "Negative call at K={k}: {call}");
            assert!(put >= 0.0, "Negative put at K={k}: {put}");
        }
    }
}
