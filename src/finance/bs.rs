use std::f64::consts::{FRAC_1_SQRT_2, PI};

pub(crate) fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

fn norm_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
}

/// Abramowitz & Stegun approximation (max error ~1.5e-7)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

pub fn bs_price(s: f64, k: f64, r: f64, t: f64, sigma: f64, is_call: bool) -> f64 {
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let df = (-r * t).exp();
    if is_call {
        s * norm_cdf(d1) - k * df * norm_cdf(d2)
    } else {
        k * df * norm_cdf(-d2) - s * norm_cdf(-d1)
    }
}

pub fn bs_vega(s: f64, k: f64, r: f64, t: f64, sigma: f64) -> f64 {
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    s * norm_pdf(d1) * t.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let t = 1.0;
        let sigma = 0.2;
        let call = bs_price(s, k, r, t, sigma, true);
        let put = bs_price(s, k, r, t, sigma, false);
        let parity = call - put - s + k * (-r * t).exp();
        assert!(parity.abs() < 1e-10);
    }

    #[test]
    fn test_bs_vega_positive() {
        let v = bs_vega(100.0, 100.0, 0.05, 1.0, 0.2);
        assert!(v > 0.0);
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.0) + norm_cdf(-1.0) - 1.0).abs() < 1e-7);
    }
}
