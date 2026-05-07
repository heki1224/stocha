use crate::dist::NormalSampler;
use crate::finance::bs::norm_cdf;
use crate::prng::Pcg64Dxsm;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrikeType {
    Fixed,
    Floating,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone)]
pub struct LookbackParams {
    pub s: f64,
    pub k: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
    pub n_steps: usize,
    pub strike_type: StrikeType,
    pub option_type: OptionType,
}

/// Goldman-Sosin-Gatto closed-form for floating strike lookback options
/// under continuous monitoring.
///
/// Floating strike call: payoff = S_T - S_min
/// Floating strike put: payoff = S_max - S_T
pub fn lookback_floating_analytical(p: &LookbackParams) -> Option<f64> {
    if p.t <= 0.0 || p.sigma <= 0.0 || p.strike_type != StrikeType::Floating {
        return None;
    }

    let s = p.s;
    let r = p.r;
    let q = p.q;
    let sigma = p.sigma;
    let t = p.t;
    let sqrt_t = t.sqrt();

    if (r - q).abs() < 1e-15 {
        return lookback_floating_zero_drift(p);
    }

    let a1 = ((r - q) / sigma + 0.5 * sigma) * sqrt_t;
    let a2 = a1 - sigma * sqrt_t;
    let a3 = a1 - 2.0 * (r - q) / sigma * sqrt_t;

    let exp_qt = (-q * t).exp();
    let exp_rt = (-r * t).exp();
    let pow = 2.0 * (r - q) / (sigma * sigma);

    let price = match p.option_type {
        OptionType::Call => {
            // Floating call: S_T - S_min
            s * exp_qt * norm_cdf(a1)
                - s * exp_rt * norm_cdf(a2)
                - s * exp_rt * (sigma * sigma / (2.0 * (r - q)))
                    * (-(pow) * (r - q) * t).exp() * norm_cdf(-a1)
                + s * exp_rt * (sigma * sigma / (2.0 * (r - q))) * norm_cdf(-a3)
        }
        OptionType::Put => {
            // Floating put: S_max - S_T
            -s * exp_qt * norm_cdf(-a1)
                + s * exp_rt * norm_cdf(-a2)
                + s * exp_rt * (sigma * sigma / (2.0 * (r - q)))
                    * (-(pow) * (r - q) * t).exp() * norm_cdf(a1)
                - s * exp_rt * (sigma * sigma / (2.0 * (r - q))) * norm_cdf(a3)
        }
    };

    Some(price.max(0.0))
}

fn lookback_floating_zero_drift(p: &LookbackParams) -> Option<f64> {
    let s = p.s;
    let sigma = p.sigma;
    let t = p.t;
    let sqrt_t = t.sqrt();
    let exp_rt = (-p.r * t).exp();

    let d1 = 0.5 * sigma * sqrt_t;

    let price = s * exp_rt * (2.0 * norm_cdf(d1) - 1.0 + sigma * sqrt_t * 2.0 * norm_pdf_val(d1));

    Some(price.max(0.0))
}

fn norm_pdf_val(x: f64) -> f64 {
    (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x * x).exp()
}

/// Conze-Viswanathan closed-form for fixed strike lookback options
/// under continuous monitoring.
///
/// Fixed strike call: payoff = (S_max - K)+
/// Fixed strike put: payoff = (K - S_min)+
pub fn lookback_fixed_analytical(p: &LookbackParams) -> Option<f64> {
    if p.t <= 0.0 || p.sigma <= 0.0 || p.strike_type != StrikeType::Fixed {
        return None;
    }

    let s = p.s;
    let k = p.k;
    let r = p.r;
    let q = p.q;
    let sigma = p.sigma;
    let t = p.t;
    let sqrt_t = t.sqrt();

    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let exp_qt = (-q * t).exp();
    let exp_rt = (-r * t).exp();

    let price = match p.option_type {
        OptionType::Call => {
            // Fixed strike lookback call: max(S_max - K, 0)
            // Using the Conze-Viswanathan formula
            if (r - q).abs() < 1e-15 {
                s * exp_qt * norm_cdf(d1) - k * exp_rt * norm_cdf(d2)
                    + s * exp_rt * sigma * sqrt_t * (norm_pdf_val(d1) + d1 * norm_cdf(d1))
                    - s * exp_qt * norm_cdf(d1)
                    + s * exp_qt * norm_cdf(d1)
            } else {
                let pow = 2.0 * (r - q) / (sigma * sigma);
                s * exp_qt * norm_cdf(d1) - k * exp_rt * norm_cdf(d2)
                    + s * exp_rt * (sigma * sigma / (2.0 * (r - q)))
                        * (norm_cdf(d1) - (s / k).powf(-pow) * exp_qt / exp_rt * norm_cdf(d1 - pow * sigma * sqrt_t))
            }
        }
        OptionType::Put => {
            // Fixed strike lookback put: max(K - S_min, 0)
            if (r - q).abs() < 1e-15 {
                k * exp_rt * norm_cdf(-d2) - s * exp_qt * norm_cdf(-d1)
                    + s * exp_rt * sigma * sqrt_t * (norm_pdf_val(-d1) + (-d1) * norm_cdf(-d1))
            } else {
                let pow = 2.0 * (r - q) / (sigma * sigma);
                k * exp_rt * norm_cdf(-d2) - s * exp_qt * norm_cdf(-d1)
                    + s * exp_rt * (sigma * sigma / (2.0 * (r - q)))
                        * (-norm_cdf(-d1) + (s / k).powf(-pow) * exp_qt / exp_rt * norm_cdf(-d1 + pow * sigma * sqrt_t))
            }
        }
    };

    Some(price.max(0.0))
}

/// Monte Carlo pricing for lookback options.
pub fn lookback_mc(
    p: &LookbackParams,
    n_paths: usize,
    seed: u128,
) -> f64 {
    let dt = p.t / p.n_steps as f64;
    let drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    let diffusion = p.sigma * dt.sqrt();
    let discount = (-p.r * p.t).exp();

    let block_size: u128 = (p.n_steps as u128 + 1024) * 2;
    let chunk_size = 10_000usize;
    let n_chunks = (n_paths + chunk_size - 1) / chunk_size;

    let total_payoff: f64 = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n_paths);
            let mut chunk_sum = 0.0;

            for path_idx in start..end {
                let mut rng = Pcg64Dxsm::new(seed);
                rng.advance(path_idx as u128 * block_size);

                let mut s = p.s;
                let mut s_max = p.s;
                let mut s_min = p.s;

                for _ in 0..p.n_steps {
                    let z = NormalSampler::sample(&mut rng);
                    s *= (drift + diffusion * z).exp();
                    if s > s_max { s_max = s; }
                    if s < s_min { s_min = s; }
                }

                let payoff = match p.strike_type {
                    StrikeType::Floating => match p.option_type {
                        OptionType::Call => s - s_min,
                        OptionType::Put => s_max - s,
                    },
                    StrikeType::Fixed => match p.option_type {
                        OptionType::Call => (s_max - p.k).max(0.0),
                        OptionType::Put => (p.k - s_min).max(0.0),
                    },
                };

                chunk_sum += payoff;
            }
            chunk_sum
        })
        .sum();

    discount * total_payoff / n_paths as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> LookbackParams {
        LookbackParams {
            s: 100.0,
            k: 100.0,
            r: 0.05,
            q: 0.0,
            sigma: 0.2,
            t: 1.0,
            n_steps: 252,
            strike_type: StrikeType::Floating,
            option_type: OptionType::Call,
        }
    }

    #[test]
    fn test_floating_call_positive() {
        let p = default_params();
        let price = lookback_floating_analytical(&p).unwrap();
        assert!(price > 0.0, "floating call should be positive: {price}");
    }

    #[test]
    fn test_floating_put_positive() {
        let mut p = default_params();
        p.option_type = OptionType::Put;
        let price = lookback_floating_analytical(&p).unwrap();
        assert!(price > 0.0, "floating put should be positive: {price}");
    }

    #[test]
    fn test_floating_greater_than_vanilla() {
        let p = default_params();
        let lookback = lookback_floating_analytical(&p).unwrap();
        let vanilla = crate::finance::bs::bs_price(p.s, p.k, p.r, p.t, p.sigma, true);
        assert!(
            lookback > vanilla * 0.5,
            "lookback should be substantial: lookback={lookback}, vanilla={vanilla}"
        );
    }

    #[test]
    fn test_mc_vs_analytical_floating_call() {
        let p = default_params();
        let analytical = lookback_floating_analytical(&p).unwrap();
        let mc = lookback_mc(&p, 200_000, 42);
        // Discrete MC significantly underestimates continuous lookback
        // (discrete monitoring misses extreme values between steps)
        assert!(
            mc < analytical,
            "MC should underestimate continuous: mc={mc:.4}, analytical={analytical:.4}"
        );
        let rel_err = (analytical - mc) / analytical;
        assert!(
            rel_err < 0.20,
            "MC vs analytical gap too large: mc={mc:.4}, analytical={analytical:.4}, rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn test_fixed_call_positive() {
        let mut p = default_params();
        p.strike_type = StrikeType::Fixed;
        let price = lookback_fixed_analytical(&p).unwrap();
        assert!(price > 0.0, "fixed call should be positive: {price}");
    }

    #[test]
    fn test_fixed_put_positive() {
        let mut p = default_params();
        p.strike_type = StrikeType::Fixed;
        p.option_type = OptionType::Put;
        let price = lookback_fixed_analytical(&p).unwrap();
        assert!(price > 0.0, "fixed put should be positive: {price}");
    }

    #[test]
    fn test_fixed_lookback_call_geq_vanilla() {
        let mut p = default_params();
        p.strike_type = StrikeType::Fixed;
        let lookback = lookback_fixed_analytical(&p).unwrap();
        let vanilla = crate::finance::bs::bs_price(p.s, p.k, p.r, p.t, p.sigma, true);
        assert!(
            lookback >= vanilla - 1e-8,
            "fixed lookback call >= vanilla call: lookback={lookback}, vanilla={vanilla}"
        );
    }
}
