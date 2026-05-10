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
    /// Historical running maximum since inception. `None` ≡ S (no seasoning).
    pub running_max: Option<f64>,
    /// Historical running minimum since inception. `None` ≡ S (no seasoning).
    pub running_min: Option<f64>,
}

fn norm_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x * x).exp()
}

/// d1(S, X) = (ln(S/X) + (b + σ²/2)T) / (σ√T) where b = r - q.
fn d1_fn(s: f64, x: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    ((s / x).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

/// Lookback-specific premium term T3(S, X). Common to all four formulas.
///
/// b = r - q ≠ 0:
///   T3 = S e^{-rT} (σ²/(2b)) [(S/X)^{-Y} N(-d1 + (2b/σ)√T) - e^{bT} N(-d1)]
/// b = 0 (L'Hôpital):
///   T3 = S e^{-rT} σ √T [n(d1) - d1·N(-d1)]
fn t3(s: f64, x: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = d1_fn(s, x, r, q, sigma, t);
    let b = r - q;
    let exp_rt = (-r * t).exp();

    if b.abs() < 1e-12 {
        s * exp_rt * sigma * sqrt_t * (norm_pdf(d1) - d1 * norm_cdf(-d1))
    } else {
        let y = 2.0 * b / (sigma * sigma);
        let arg2 = -d1 + (2.0 * b / sigma) * sqrt_t;
        let bracket = (s / x).powf(-y) * norm_cdf(arg2) - (b * t).exp() * norm_cdf(-d1);
        s * exp_rt * (sigma * sigma / (2.0 * b)) * bracket
    }
}

/// Goldman-Sosin-Gatto closed-form for floating-strike lookback options
/// under continuous monitoring, with optional running extremum (seasoning).
pub fn lookback_floating_analytical(p: &LookbackParams) -> Option<f64> {
    if p.t <= 0.0 || p.sigma <= 0.0 || p.strike_type != StrikeType::Floating {
        return None;
    }
    let exp_qt = (-p.q * p.t).exp();
    let exp_rt = (-p.r * p.t).exp();

    let price = match p.option_type {
        OptionType::Call => {
            // Floating call uses running min m (≤ S).
            let m = p.running_min.unwrap_or(p.s);
            if m > p.s {
                return None;
            }
            let d1 = d1_fn(p.s, m, p.r, p.q, p.sigma, p.t);
            let d2 = d1 - p.sigma * p.t.sqrt();
            p.s * exp_qt * norm_cdf(d1) - m * exp_rt * norm_cdf(d2) + t3(p.s, m, p.r, p.q, p.sigma, p.t)
        }
        OptionType::Put => {
            // Floating put uses running max M (≥ S).
            let big_m = p.running_max.unwrap_or(p.s);
            if big_m < p.s {
                return None;
            }
            let d1 = d1_fn(p.s, big_m, p.r, p.q, p.sigma, p.t);
            let d2 = d1 - p.sigma * p.t.sqrt();
            big_m * exp_rt * norm_cdf(-d2) - p.s * exp_qt * norm_cdf(-d1)
                + t3(p.s, big_m, p.r, p.q, p.sigma, p.t)
        }
    };

    Some(price.max(0.0))
}

/// Conze-Viswanathan closed-form for fixed-strike lookback options
/// under continuous monitoring, with optional running extremum (seasoning).
pub fn lookback_fixed_analytical(p: &LookbackParams) -> Option<f64> {
    if p.t <= 0.0 || p.sigma <= 0.0 || p.strike_type != StrikeType::Fixed || p.k <= 0.0 {
        return None;
    }
    let exp_qt = (-p.q * p.t).exp();
    let exp_rt = (-p.r * p.t).exp();

    let price = match p.option_type {
        OptionType::Call => {
            let big_m = p.running_max.unwrap_or(p.s);
            if big_m < p.s {
                return None;
            }
            if p.k > big_m {
                // K > M: standard fixed-strike call, X = K.
                let d1 = d1_fn(p.s, p.k, p.r, p.q, p.sigma, p.t);
                let d2 = d1 - p.sigma * p.t.sqrt();
                p.s * exp_qt * norm_cdf(d1) - p.k * exp_rt * norm_cdf(d2)
                    + t3(p.s, p.k, p.r, p.q, p.sigma, p.t)
            } else {
                // K ≤ M: deterministic intrinsic + lookback at strike M.
                let d1 = d1_fn(p.s, big_m, p.r, p.q, p.sigma, p.t);
                let d2 = d1 - p.sigma * p.t.sqrt();
                exp_rt * (big_m - p.k)
                    + p.s * exp_qt * norm_cdf(d1)
                    - big_m * exp_rt * norm_cdf(d2)
                    + t3(p.s, big_m, p.r, p.q, p.sigma, p.t)
            }
        }
        OptionType::Put => {
            let m = p.running_min.unwrap_or(p.s);
            if m > p.s {
                return None;
            }
            if p.k < m {
                let d1 = d1_fn(p.s, p.k, p.r, p.q, p.sigma, p.t);
                let d2 = d1 - p.sigma * p.t.sqrt();
                p.k * exp_rt * norm_cdf(-d2) - p.s * exp_qt * norm_cdf(-d1)
                    + t3(p.s, p.k, p.r, p.q, p.sigma, p.t)
            } else {
                let d1 = d1_fn(p.s, m, p.r, p.q, p.sigma, p.t);
                let d2 = d1 - p.sigma * p.t.sqrt();
                exp_rt * (p.k - m)
                    + m * exp_rt * norm_cdf(-d2)
                    - p.s * exp_qt * norm_cdf(-d1)
                    + t3(p.s, m, p.r, p.q, p.sigma, p.t)
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
    let n_chunks = n_paths.div_ceil(chunk_size);

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
                let mut s_max = p.running_max.unwrap_or(p.s).max(p.s);
                let mut s_min = p.running_min.unwrap_or(p.s).min(p.s);

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
            running_max: None,
            running_min: None,
        }
    }

    #[test]
    fn test_floating_call_atm_no_seasoning() {
        // Reference (Gemini hand-computed, Hull/Haug GSG): 17.2169.
        let p = default_params();
        let price = lookback_floating_analytical(&p).unwrap();
        assert!(
            (price - 17.2169).abs() < 1e-3,
            "floating call ATM should be ≈17.22: got {price}"
        );
    }

    #[test]
    fn test_floating_put_atm_no_seasoning() {
        // Reference: 12.3398.
        let mut p = default_params();
        p.option_type = OptionType::Put;
        let price = lookback_floating_analytical(&p).unwrap();
        assert!(
            (price - 12.3398).abs() < 1e-3,
            "floating put ATM should be ≈12.34: got {price}"
        );
    }

    #[test]
    fn test_fixed_call_atm_no_seasoning() {
        // ATM with K = M = S → reflection symmetry implies fixed call = floating call = 17.22.
        let mut p = default_params();
        p.strike_type = StrikeType::Fixed;
        let price = lookback_fixed_analytical(&p).unwrap();
        assert!(
            (price - 17.2169).abs() < 1e-3,
            "fixed call ATM should be ≈17.22: got {price}"
        );
    }

    #[test]
    fn test_fixed_put_atm_no_seasoning() {
        let mut p = default_params();
        p.strike_type = StrikeType::Fixed;
        p.option_type = OptionType::Put;
        let price = lookback_fixed_analytical(&p).unwrap();
        assert!(
            (price - 12.3398).abs() < 1e-3,
            "fixed put ATM should be ≈12.34: got {price}"
        );
    }

    #[test]
    fn test_floating_call_positive() {
        let p = default_params();
        let price = lookback_floating_analytical(&p).unwrap();
        assert!(price > 0.0);
    }

    #[test]
    fn test_floating_put_positive() {
        let mut p = default_params();
        p.option_type = OptionType::Put;
        let price = lookback_floating_analytical(&p).unwrap();
        assert!(price > 0.0);
    }

    #[test]
    fn test_floating_greater_than_vanilla() {
        let p = default_params();
        let lookback = lookback_floating_analytical(&p).unwrap();
        let vanilla = crate::finance::bs::bs_price(p.s, p.k, p.r, p.t, p.sigma, true);
        assert!(lookback > vanilla);
    }

    #[test]
    fn test_floating_seasoning_lower_min_increases_call() {
        let p_no = default_params();
        let mut p_low = p_no.clone();
        p_low.running_min = Some(80.0);
        let no = lookback_floating_analytical(&p_no).unwrap();
        let low = lookback_floating_analytical(&p_low).unwrap();
        assert!(low > no, "lower running_min ⇒ higher floating call: no={no}, low={low}");
    }

    #[test]
    fn test_floating_seasoning_higher_max_increases_put() {
        let mut p_no = default_params();
        p_no.option_type = OptionType::Put;
        let mut p_high = p_no.clone();
        p_high.running_max = Some(120.0);
        let no = lookback_floating_analytical(&p_no).unwrap();
        let high = lookback_floating_analytical(&p_high).unwrap();
        assert!(high > no);
    }

    #[test]
    fn test_floating_running_min_eq_s_matches_no_seasoning() {
        let p_no = default_params();
        let mut p_eq = p_no.clone();
        p_eq.running_min = Some(p_no.s);
        let no = lookback_floating_analytical(&p_no).unwrap();
        let eq = lookback_floating_analytical(&p_eq).unwrap();
        assert!((no - eq).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_call_seasoning_intrinsic() {
        // M=120, K=100: intrinsic ≥ (M-K)e^{-rT} ≈ 19.02.
        let mut p = default_params();
        p.strike_type = StrikeType::Fixed;
        p.running_max = Some(120.0);
        let price = lookback_fixed_analytical(&p).unwrap();
        let intrinsic = (120.0 - 100.0) * (-0.05f64).exp();
        assert!(price >= intrinsic - 1e-8);
    }

    #[test]
    fn test_mc_vs_analytical_floating_call() {
        // Discrete MC underestimates by BGK shift ≈ 0.5826 σ √(T/N) S.
        let p = default_params();
        let analytical = lookback_floating_analytical(&p).unwrap();
        let mc = lookback_mc(&p, 100_000, 42);
        assert!(mc < analytical);
        let bgk_gap = 0.5826 * p.sigma * (p.t / p.n_steps as f64).sqrt() * p.s;
        // MC should sit within ~3× BGK gap of analytical.
        assert!(
            (analytical - mc).abs() < 3.0 * bgk_gap,
            "gap too large: analytical={analytical}, mc={mc}, bgk_gap={bgk_gap}"
        );
    }

    #[test]
    fn test_fixed_lookback_call_geq_vanilla() {
        let mut p = default_params();
        p.strike_type = StrikeType::Fixed;
        let lookback = lookback_fixed_analytical(&p).unwrap();
        let vanilla = crate::finance::bs::bs_price(p.s, p.k, p.r, p.t, p.sigma, true);
        assert!(lookback >= vanilla - 1e-8);
    }
}
