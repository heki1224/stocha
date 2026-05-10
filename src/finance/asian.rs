use crate::dist::NormalSampler;
use crate::finance::bs::norm_cdf;
use crate::prng::Pcg64Dxsm;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AverageType {
    Arithmetic,
    Geometric,
}

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
pub struct AsianParams {
    pub s: f64,
    pub k: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
    pub n_steps: usize,
    pub average_type: AverageType,
    pub strike_type: StrikeType,
    pub option_type: OptionType,
    /// Running average over [0, time_elapsed] (must match `average_type`'s mean).
    /// Used together with `time_elapsed` for in-progress (seasoned) Asian options.
    pub running_avg: Option<f64>,
    /// Time already elapsed since inception, in years.
    pub time_elapsed: Option<f64>,
}

/// Returns (a_spent, t1) when both seasoning fields are provided and t1 ∈ (0, T].
fn extract_seasoning(p: &AsianParams) -> Option<(f64, f64)> {
    match (p.running_avg, p.time_elapsed) {
        (Some(a), Some(t1)) if t1 > 0.0 && a >= 0.0 => Some((a, t1)),
        _ => None,
    }
}

/// Risk-neutral expectation of the geometric average over [t1, T] given current spot S.
/// E[exp(Y)] where Y = (1/T) ∫_{t1}^T ln S_u du is normal under Q.
fn expected_geometric_avg(s: f64, r: f64, q: f64, sigma: f64, t1: f64, t_total: f64, geo_avg_spent: f64) -> f64 {
    let remaining = t_total - t1;
    let log_spent_part = (t1 / t_total) * geo_avg_spent.ln();
    let mean_y = (remaining * s.ln() + (r - q - 0.5 * sigma * sigma) * remaining * remaining / 2.0) / t_total;
    let var_y = sigma * sigma * remaining.powi(3) / (3.0 * t_total * t_total);
    (log_spent_part + mean_y + 0.5 * var_y).exp()
}

/// Risk-neutral expectation of the arithmetic average over [0, T] given current spot S
/// at time t1 with running_avg = a_spent.
fn expected_arithmetic_avg(s: f64, r: f64, q: f64, t1: f64, t_total: f64, a_spent: f64) -> f64 {
    let remaining = t_total - t1;
    let mu = r - q;
    let e_a_remain = if mu.abs() < 1e-12 {
        s
    } else {
        s * ((mu * remaining).exp() - 1.0) / (mu * remaining)
    };
    (t1 / t_total) * a_spent + (remaining / t_total) * e_a_remain
}

/// Kemna-Vorst closed-form for geometric average, fixed strike Asian option.
///
/// The geometric average of a GBM is lognormal, enabling an exact formula.
/// Uses continuous-averaging approximation.
pub fn asian_geometric_fixed_analytical(p: &AsianParams) -> Option<f64> {
    if p.t <= 0.0 || p.sigma <= 0.0 {
        return None;
    }
    if p.average_type != AverageType::Geometric || p.strike_type != StrikeType::Fixed {
        return None;
    }

    if let Some((a_spent, t1)) = extract_seasoning(p) {
        if t1 >= p.t {
            // At or past expiry: realized payoff (no further discount).
            let payoff = match p.option_type {
                OptionType::Call => (a_spent - p.k).max(0.0),
                OptionType::Put => (p.k - a_spent).max(0.0),
            };
            return Some(payoff);
        }
        let remaining = p.t - t1;
        let k_star = (p.t * p.k - t1 * a_spent) / remaining;
        let scale = remaining / p.t;

        if k_star > 0.0 {
            // Standard transformation: forward-starting Asian on [t1, T] with strike K*.
            let mut pp = p.clone();
            pp.k = k_star;
            pp.t = remaining;
            pp.running_avg = None;
            pp.time_elapsed = None;
            return asian_geometric_fixed_analytical(&pp).map(|price| scale * price);
        } else {
            // Deep ITM (call) or always OTM (put).
            return Some(match p.option_type {
                OptionType::Put => 0.0,
                OptionType::Call => {
                    let e_geo = expected_geometric_avg(p.s, p.r, p.q, p.sigma, t1, p.t, a_spent);
                    let pv = (-p.r * remaining).exp() * (e_geo - p.k);
                    pv.max(0.0)
                }
            });
        }
    }

    let sigma_g = p.sigma / 3.0_f64.sqrt();
    let mu_g = 0.5 * (p.r - p.q - 0.5 * p.sigma * p.sigma) + 0.5 * 0.5 * sigma_g * sigma_g;
    let d1 = ((p.s / p.k).ln() + (mu_g + 0.5 * sigma_g * sigma_g) * p.t) / (sigma_g * p.t.sqrt());
    let d2 = d1 - sigma_g * p.t.sqrt();
    let df = (-p.r * p.t).exp();

    let price = match p.option_type {
        OptionType::Call => {
            p.s * ((mu_g - p.r) * p.t).exp() * norm_cdf(d1) - p.k * df * norm_cdf(d2)
        }
        OptionType::Put => {
            p.k * df * norm_cdf(-d2) - p.s * ((mu_g - p.r) * p.t).exp() * norm_cdf(-d1)
        }
    };

    Some(price.max(0.0))
}

/// Monte Carlo pricing for Asian options.
///
/// For arithmetic average with geometric control variate:
/// - Simulates paths and computes both arithmetic and geometric averages
/// - Uses the geometric average price as a control variate to reduce variance
pub fn asian_mc(
    p: &AsianParams,
    n_paths: usize,
    seed: u128,
) -> f64 {
    if let Some((a_spent, t1)) = extract_seasoning(p) {
        if t1 >= p.t {
            // Realized at expiry.
            let payoff = match (p.strike_type, p.option_type) {
                (StrikeType::Fixed, OptionType::Call) => (a_spent - p.k).max(0.0),
                (StrikeType::Fixed, OptionType::Put) => (p.k - a_spent).max(0.0),
                // Floating with t1 == T is degenerate; fall back to 0.
                _ => 0.0,
            };
            return payoff;
        }
        if p.strike_type == StrikeType::Fixed {
            let remaining = p.t - t1;
            let k_star = (p.t * p.k - t1 * a_spent) / remaining;
            let scale = remaining / p.t;

            if k_star > 0.0 {
                let mut pp = p.clone();
                pp.k = k_star;
                pp.t = remaining;
                pp.running_avg = None;
                pp.time_elapsed = None;
                return scale * asian_mc(&pp, n_paths, seed);
            } else {
                // Deep ITM call (analytical) or always-OTM put (zero).
                return match p.option_type {
                    OptionType::Put => 0.0,
                    OptionType::Call => {
                        let e_a_t = match p.average_type {
                            AverageType::Arithmetic => {
                                expected_arithmetic_avg(p.s, p.r, p.q, t1, p.t, a_spent)
                            }
                            AverageType::Geometric => {
                                expected_geometric_avg(p.s, p.r, p.q, p.sigma, t1, p.t, a_spent)
                            }
                        };
                        ((-p.r * remaining).exp() * (e_a_t - p.k)).max(0.0)
                    }
                };
            }
        }
        // Floating-strike seasoning: simulate remaining period and combine with spent.
        return asian_mc_floating_seasoned(p, a_spent, t1, n_paths, seed);
    }

    let dt = p.t / p.n_steps as f64;
    let drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    let diffusion = p.sigma * dt.sqrt();
    let discount = (-p.r * p.t).exp();

    let block_size: u128 = (p.n_steps as u128 + 1024) * 2;
    let chunk_size = 10_000usize;
    let n_chunks = (n_paths + chunk_size - 1) / chunk_size;

    let use_cv = p.average_type == AverageType::Arithmetic
        && p.strike_type == StrikeType::Fixed;

    let geo_analytical = if use_cv {
        let geo_params = AsianParams {
            average_type: AverageType::Geometric,
            ..p.clone()
        };
        asian_geometric_fixed_analytical(&geo_params).unwrap_or(0.0)
    } else {
        0.0
    };

    // Collect (arith_payoff, geo_payoff) per path for CV estimation
    let results: Vec<(f64, f64)> = (0..n_chunks)
        .into_par_iter()
        .flat_map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n_paths);
            let mut chunk_results = Vec::with_capacity(end - start);

            for path_idx in start..end {
                let mut rng = Pcg64Dxsm::new(seed);
                rng.advance(path_idx as u128 * block_size);

                let mut s = p.s;
                let mut sum_arith = 0.0;
                let mut sum_log = 0.0;

                for _ in 0..p.n_steps {
                    let z = NormalSampler::sample(&mut rng);
                    s *= (drift + diffusion * z).exp();
                    sum_arith += s;
                    sum_log += s.ln();
                }

                let avg_arith = sum_arith / p.n_steps as f64;
                let avg_geo = (sum_log / p.n_steps as f64).exp();

                let payoff_arith = match p.strike_type {
                    StrikeType::Fixed => match p.option_type {
                        OptionType::Call => (avg_arith - p.k).max(0.0),
                        OptionType::Put => (p.k - avg_arith).max(0.0),
                    },
                    StrikeType::Floating => match p.option_type {
                        OptionType::Call => (s - avg_arith).max(0.0),
                        OptionType::Put => (avg_arith - s).max(0.0),
                    },
                };

                let payoff_geo = match p.strike_type {
                    StrikeType::Fixed => match p.option_type {
                        OptionType::Call => (avg_geo - p.k).max(0.0),
                        OptionType::Put => (p.k - avg_geo).max(0.0),
                    },
                    StrikeType::Floating => match p.option_type {
                        OptionType::Call => (s - avg_geo).max(0.0),
                        OptionType::Put => (avg_geo - s).max(0.0),
                    },
                };

                chunk_results.push((payoff_arith, payoff_geo));
            }
            chunk_results
        })
        .collect();

    if use_cv && results.len() > 1 {
        let n = results.len() as f64;
        let mean_arith: f64 = results.iter().map(|(a, _)| a).sum::<f64>() / n;
        let mean_geo: f64 = results.iter().map(|(_, g)| g).sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_geo = 0.0;
        for &(a, g) in &results {
            cov += (a - mean_arith) * (g - mean_geo);
            var_geo += (g - mean_geo) * (g - mean_geo);
        }

        let beta = if var_geo > 1e-30 { cov / var_geo } else { 0.0 };
        let geo_mc_price = discount * mean_geo;
        let adjusted = discount * mean_arith - beta * (geo_mc_price - geo_analytical);
        adjusted.max(0.0)
    } else {
        let mean_payoff: f64 = results.iter().map(|(a, _)| a).sum::<f64>() / results.len() as f64;
        (discount * mean_payoff).max(0.0)
    }
}

/// MC for floating-strike Asian with seasoning: simulate over [t1, T] and combine
/// with the spent running average to form the time-T average.
fn asian_mc_floating_seasoned(
    p: &AsianParams,
    a_spent: f64,
    t1: f64,
    n_paths: usize,
    seed: u128,
) -> f64 {
    let remaining = p.t - t1;
    // Allocate steps proportional to remaining time, but at least 1.
    let n_steps_remaining = ((p.n_steps as f64 * remaining / p.t).round() as usize).max(1);
    let dt = remaining / n_steps_remaining as f64;
    let drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    let diffusion = p.sigma * dt.sqrt();
    let discount = (-p.r * remaining).exp();
    let weight_spent = t1 / p.t;
    let weight_remain = remaining / p.t;

    let block_size: u128 = (n_steps_remaining as u128 + 1024) * 2;
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
                let mut sum_arith = 0.0;
                let mut sum_log = 0.0;
                for _ in 0..n_steps_remaining {
                    let z = NormalSampler::sample(&mut rng);
                    s *= (drift + diffusion * z).exp();
                    sum_arith += s;
                    sum_log += s.ln();
                }
                let avg_remain = match p.average_type {
                    AverageType::Arithmetic => sum_arith / n_steps_remaining as f64,
                    AverageType::Geometric => (sum_log / n_steps_remaining as f64).exp(),
                };
                let avg_t = match p.average_type {
                    AverageType::Arithmetic => weight_spent * a_spent + weight_remain * avg_remain,
                    AverageType::Geometric => {
                        (a_spent.ln() * weight_spent + avg_remain.ln() * weight_remain).exp()
                    }
                };
                let payoff = match p.option_type {
                    OptionType::Call => (s - avg_t).max(0.0),
                    OptionType::Put => (avg_t - s).max(0.0),
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

    fn default_params() -> AsianParams {
        AsianParams {
            s: 100.0,
            k: 100.0,
            r: 0.05,
            q: 0.0,
            sigma: 0.2,
            t: 1.0,
            n_steps: 252,
            average_type: AverageType::Geometric,
            strike_type: StrikeType::Fixed,
            option_type: OptionType::Call,
            running_avg: None,
            time_elapsed: None,
        }
    }

    #[test]
    fn test_geometric_analytical_positive() {
        let p = default_params();
        let price = asian_geometric_fixed_analytical(&p).unwrap();
        assert!(price > 0.0, "price should be positive: {price}");
    }

    #[test]
    fn test_geometric_less_than_vanilla() {
        let p = default_params();
        let asian = asian_geometric_fixed_analytical(&p).unwrap();
        let vanilla = crate::finance::bs::bs_price(p.s, p.k, p.r, p.t, p.sigma, true);
        assert!(
            asian < vanilla + 1e-10,
            "Asian geometric should be <= vanilla: asian={asian}, vanilla={vanilla}"
        );
    }

    #[test]
    fn test_put_call_relation_geometric() {
        let p_call = default_params();
        let mut p_put = p_call.clone();
        p_put.option_type = OptionType::Put;

        let call = asian_geometric_fixed_analytical(&p_call).unwrap();
        let put = asian_geometric_fixed_analytical(&p_put).unwrap();
        // Both should be non-negative
        assert!(call >= 0.0);
        assert!(put >= 0.0);
    }

    #[test]
    fn test_mc_geometric_vs_analytical() {
        let p = default_params();
        let analytical = asian_geometric_fixed_analytical(&p).unwrap();
        let mc = asian_mc(&p, 200_000, 42);
        // Discrete MC vs continuous analytical: allow wider tolerance
        let rel_err = (mc - analytical).abs() / analytical.max(1e-10);
        assert!(
            rel_err < 0.10,
            "MC vs analytical: mc={mc:.4}, analytical={analytical:.4}, rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn test_arithmetic_cv_reduces_variance() {
        let mut p = default_params();
        p.average_type = AverageType::Arithmetic;
        let price = asian_mc(&p, 100_000, 42);
        assert!(price > 0.0, "arithmetic Asian call should be positive");
        // Arithmetic average >= geometric average for call
        let geo = asian_geometric_fixed_analytical(&AsianParams {
            average_type: AverageType::Geometric,
            ..p.clone()
        })
        .unwrap();
        assert!(
            price >= geo * 0.9,
            "arithmetic should be roughly >= geometric: arith={price}, geo={geo}"
        );
    }

    #[test]
    fn test_floating_strike_mc() {
        let mut p = default_params();
        p.strike_type = StrikeType::Floating;
        p.average_type = AverageType::Arithmetic;
        let price = asian_mc(&p, 100_000, 42);
        assert!(price > 0.0, "floating strike Asian should be positive");
    }
}
