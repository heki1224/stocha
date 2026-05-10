use crate::dist::NormalSampler;
use crate::finance::bs::norm_cdf;
use crate::prng::Pcg64Dxsm;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrierDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarrierKind {
    In,
    Out,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone)]
pub struct BarrierParams {
    pub s: f64,
    pub k: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub t: f64,
    pub barrier: f64,
    pub direction: BarrierDirection,
    pub kind: BarrierKind,
    pub option_type: OptionType,
    /// Number of discrete monitoring dates. `None` = continuous monitoring.
    /// Applied only to `barrier_analytical` via the Broadie-Glasserman-Kou (1997)
    /// continuity correction; `barrier_mc` ignores it (paths are already discrete).
    pub n_monitoring: Option<u32>,
    /// Rebate amount R paid when the barrier condition triggers (KO hit, or KI not hit).
    /// 0.0 = no rebate.
    pub rebate: f64,
    /// If true and `kind=Out`, rebate is paid immediately at the first hit time τ;
    /// otherwise (KO or KI) it is paid at expiry T conditional on the relevant event.
    pub rebate_at_hit: bool,
}

/// Probability of hitting barrier H during [0, T] under GBM with drift r-q, vol σ.
fn hit_probability(s: f64, h: f64, r: f64, q: f64, sigma: f64, t: f64, dir: BarrierDirection) -> f64 {
    let nu = r - q - 0.5 * sigma * sigma;
    let mu = nu / (sigma * sigma);
    let sqrt_t = t.sqrt();
    let (h1, h2) = match dir {
        BarrierDirection::Up => {
            let a = (s / h).ln();
            ((a + nu * t) / (sigma * sqrt_t), (a - nu * t) / (sigma * sqrt_t))
        }
        BarrierDirection::Down => {
            let a = (h / s).ln();
            ((a - nu * t) / (sigma * sqrt_t), (a + nu * t) / (sigma * sqrt_t))
        }
    };
    let p = norm_cdf(h1) + (h / s).powf(2.0 * mu) * norm_cdf(h2);
    p.clamp(0.0, 1.0)
}

/// Present value of a rebate R paid at the first barrier-hit time τ ≤ T.
/// V = R · [(H/S)^{μ-λ} N(d1) + (H/S)^{μ+λ} N(d2)] under canonical GBM first-passage.
fn rebate_paid_at_hit(s: f64, h: f64, rebate: f64, r: f64, q: f64, sigma: f64, t: f64, dir: BarrierDirection) -> f64 {
    if rebate == 0.0 {
        return 0.0;
    }
    let nu = r - q - 0.5 * sigma * sigma;
    let sigma2 = sigma * sigma;
    let mu = nu / sigma2;
    let lambda = (mu * mu + 2.0 * r / sigma2).sqrt();
    let sqrt_t = t.sqrt();
    let log_ratio = match dir {
        BarrierDirection::Up => (h / s).ln(),    // > 0
        BarrierDirection::Down => (s / h).ln(),  // > 0
    };
    let d1 = (-log_ratio + lambda * sigma2 * t) / (sigma * sqrt_t);
    let d2 = (-log_ratio - lambda * sigma2 * t) / (sigma * sqrt_t);
    let ratio = h / s;
    rebate * (ratio.powf(mu - lambda) * norm_cdf(d1) + ratio.powf(mu + lambda) * norm_cdf(d2))
}

/// Present value of rebate R for the four (KO/KI × direction) variants.
/// Returns 0 if rebate is 0.
pub fn rebate_value(p: &BarrierParams) -> f64 {
    if p.rebate == 0.0 {
        return 0.0;
    }
    let already_hit = match p.direction {
        BarrierDirection::Up => p.s >= p.barrier,
        BarrierDirection::Down => p.s <= p.barrier,
    };

    match (p.kind, p.rebate_at_hit) {
        (BarrierKind::Out, true) => {
            if already_hit {
                p.rebate
            } else {
                rebate_paid_at_hit(p.s, p.barrier, p.rebate, p.r, p.q, p.sigma, p.t, p.direction)
            }
        }
        (BarrierKind::Out, false) => {
            let prob = if already_hit {
                1.0
            } else {
                hit_probability(p.s, p.barrier, p.r, p.q, p.sigma, p.t, p.direction)
            };
            p.rebate * (-p.r * p.t).exp() * prob
        }
        (BarrierKind::In, _) => {
            // Knock-in rebate: paid at expiry only if barrier was NOT hit. Once hit,
            // the option converts to vanilla and the rebate is extinguished.
            if already_hit {
                0.0
            } else {
                let prob = hit_probability(p.s, p.barrier, p.r, p.q, p.sigma, p.t, p.direction);
                p.rebate * (-p.r * p.t).exp() * (1.0 - prob)
            }
        }
    }
}

/// Broadie-Glasserman-Kou (1997) continuity-correction constant: -ζ(1/2)/√(2π) ≈ 0.5826.
pub const BGK_BETA: f64 = 0.5825971579390106;

/// Apply BGK shift to the barrier: H_continuous = H_discrete · exp(±β·σ·√Δt).
/// Up barriers shift outward (+), down barriers shift outward (-).
fn bgk_adjusted_barrier(h: f64, sigma: f64, t: f64, n: u32, dir: BarrierDirection) -> f64 {
    if n == 0 {
        return h;
    }
    let dt = t / n as f64;
    let shift = BGK_BETA * sigma * dt.sqrt();
    match dir {
        BarrierDirection::Up => h * shift.exp(),
        BarrierDirection::Down => h * (-shift).exp(),
    }
}

pub fn barrier_analytical(p: &BarrierParams) -> Option<f64> {
    let s = p.s;
    let k = p.k;
    let h = match p.n_monitoring {
        Some(n) if n > 0 => bgk_adjusted_barrier(p.barrier, p.sigma, p.t, n, p.direction),
        _ => p.barrier,
    };
    let r = p.r;
    let q = p.q;
    let sigma = p.sigma;
    let t = p.t;

    if t <= 0.0 || sigma <= 0.0 {
        return None;
    }

    match p.direction {
        BarrierDirection::Up => {
            if s >= h {
                return Some(knocked_value(p) + rebate_value(p));
            }
        }
        BarrierDirection::Down => {
            if s <= h {
                return Some(knocked_value(p) + rebate_value(p));
            }
        }
    }

    let sqrt_t = t.sqrt();
    let mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma);
    let d = |x: f64, y: f64| -> f64 {
        (x.ln() + y * sigma * sigma * t) / (sigma * sqrt_t)
    };

    let a = |phi: f64| -> f64 {
        let d1 = d(s / k, 1.0 + mu);
        let d2 = d1 - sigma * sqrt_t;
        phi * (s * ((-q * t).exp()) * norm_cdf(phi * d1) - k * ((-r * t).exp()) * norm_cdf(phi * d2))
    };

    let b = |phi: f64| -> f64 {
        let d1 = d(s / h, 1.0 + mu);
        let d2 = d1 - sigma * sqrt_t;
        phi * (s * ((-q * t).exp()) * norm_cdf(phi * d1) - k * ((-r * t).exp()) * norm_cdf(phi * d2))
    };

    let c = |phi: f64, eta: f64| -> f64 {
        let pow = (h / s).powf(2.0 * (1.0 + mu));
        let d1 = d(h * h / (s * k), 1.0 + mu);
        let d2 = d1 - sigma * sqrt_t;
        phi * (s * ((-q * t).exp()) * pow * norm_cdf(eta * d1) - k * ((-r * t).exp()) * (h / s).powf(2.0 * mu) * norm_cdf(eta * d2))
    };

    let dd = |phi: f64, eta: f64| -> f64 {
        let pow = (h / s).powf(2.0 * (1.0 + mu));
        let d1 = d(h / s, 1.0 + mu);
        let d2 = d1 - sigma * sqrt_t;
        phi * (s * ((-q * t).exp()) * pow * norm_cdf(eta * d1) - k * ((-r * t).exp()) * (h / s).powf(2.0 * mu) * norm_cdf(eta * d2))
    };

    let price = match (p.direction, p.kind, p.option_type) {
        // Down-and-out call (H < K)
        (BarrierDirection::Down, BarrierKind::Out, OptionType::Call) if h <= k => {
            a(1.0) - c(1.0, 1.0)
        }
        // Down-and-out call (H >= K)
        (BarrierDirection::Down, BarrierKind::Out, OptionType::Call) => {
            b(1.0) - dd(1.0, 1.0)
        }
        // Down-and-in call (H < K)
        (BarrierDirection::Down, BarrierKind::In, OptionType::Call) if h <= k => {
            c(1.0, 1.0)
        }
        // Down-and-in call (H >= K)
        (BarrierDirection::Down, BarrierKind::In, OptionType::Call) => {
            a(1.0) - b(1.0) + dd(1.0, 1.0)
        }
        // Up-and-out call (H > K)
        (BarrierDirection::Up, BarrierKind::Out, OptionType::Call) if h > k => {
            a(1.0) - b(1.0) + c(1.0, -1.0) - dd(1.0, -1.0)
        }
        // Up-and-out call (H <= K)
        (BarrierDirection::Up, BarrierKind::Out, OptionType::Call) => 0.0,
        // Up-and-in call (H > K)
        (BarrierDirection::Up, BarrierKind::In, OptionType::Call) if h > k => {
            b(1.0) - c(1.0, -1.0) + dd(1.0, -1.0)
        }
        // Up-and-in call (H <= K)
        (BarrierDirection::Up, BarrierKind::In, OptionType::Call) => a(1.0),

        // Down-and-out put (H < K)
        (BarrierDirection::Down, BarrierKind::Out, OptionType::Put) if h < k => {
            a(-1.0) - b(-1.0) + c(-1.0, 1.0) - dd(-1.0, 1.0)
        }
        // Down-and-out put (H >= K)
        (BarrierDirection::Down, BarrierKind::Out, OptionType::Put) => 0.0,
        // Down-and-in put (H < K)
        (BarrierDirection::Down, BarrierKind::In, OptionType::Put) if h < k => {
            b(-1.0) - c(-1.0, 1.0) + dd(-1.0, 1.0)
        }
        // Down-and-in put (H >= K)
        (BarrierDirection::Down, BarrierKind::In, OptionType::Put) => a(-1.0),

        // Up-and-out put (H >= K)
        (BarrierDirection::Up, BarrierKind::Out, OptionType::Put) if h >= k => {
            a(-1.0) - c(-1.0, -1.0)
        }
        // Up-and-out put (H < K)
        (BarrierDirection::Up, BarrierKind::Out, OptionType::Put) => {
            b(-1.0) - dd(-1.0, -1.0)
        }
        // Up-and-in put (H >= K)
        (BarrierDirection::Up, BarrierKind::In, OptionType::Put) if h >= k => {
            c(-1.0, -1.0)
        }
        // Up-and-in put (H < K)
        (BarrierDirection::Up, BarrierKind::In, OptionType::Put) => {
            a(-1.0) - b(-1.0) + dd(-1.0, -1.0)
        }
    };

    Some((price.max(0.0)) + rebate_value(p))
}

fn knocked_value(p: &BarrierParams) -> f64 {
    match p.kind {
        BarrierKind::Out => 0.0,
        BarrierKind::In => {
            // Already knocked in — value equals vanilla
            vanilla_price(p)
        }
    }
}

fn vanilla_price(p: &BarrierParams) -> f64 {
    use crate::finance::bs::bs_price_div;
    let is_call = matches!(p.option_type, OptionType::Call);
    bs_price_div(p.s, p.k, p.r, p.q, p.t, p.sigma, is_call)
}

pub fn barrier_mc(
    p: &BarrierParams,
    n_paths: usize,
    n_steps: usize,
    seed: u128,
) -> f64 {
    let dt = p.t / n_steps as f64;
    let drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    let diffusion = p.sigma * dt.sqrt();
    let discount = (-p.r * p.t).exp();

    let block_size: u128 = (n_steps as u128 + 1024) * 2;
    let chunk_size = 10_000usize;
    let n_chunks = n_paths.div_ceil(chunk_size);

    let already_hit = match p.direction {
        BarrierDirection::Up => p.s >= p.barrier,
        BarrierDirection::Down => p.s <= p.barrier,
    };

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
                let mut hit_step: i32 = -1;

                if already_hit {
                    hit_step = 0;
                } else {
                    for step in 1..=n_steps {
                        let z = NormalSampler::sample(&mut rng);
                        s *= (drift + diffusion * z).exp();
                        if check_barrier(s, p.barrier, p.direction) {
                            hit_step = step as i32;
                            break;
                        }
                    }
                }
                let knocked = hit_step >= 0;

                // Drive remaining path to T (only relevant if KI knocked, to determine S_T).
                if knocked && p.kind == BarrierKind::In && (hit_step as usize) < n_steps {
                    for _ in (hit_step as usize)..n_steps {
                        let z = NormalSampler::sample(&mut rng);
                        s *= (drift + diffusion * z).exp();
                    }
                } else if !knocked {
                    // continue path is unnecessary; s already at terminal
                }

                let option_payoff = match p.kind {
                    BarrierKind::Out => {
                        if knocked { 0.0 } else { intrinsic(s, p.k, p.option_type) }
                    }
                    BarrierKind::In => {
                        if knocked { intrinsic(s, p.k, p.option_type) } else { 0.0 }
                    }
                };
                // Path-level rebate contribution (added BEFORE outer discount; we
                // need to undo the global discount for paid_at_hit since timing varies).
                let rebate_pv = if p.rebate == 0.0 {
                    0.0
                } else {
                    rebate_path_pv(p, hit_step, n_steps, dt)
                };
                // option_payoff is at T (will get global discount); rebate_pv is already PV.
                chunk_sum += option_payoff + rebate_pv / discount;
            }
            chunk_sum
        })
        .sum();

    discount * total_payoff / n_paths as f64
}

/// Per-path rebate present value (already discounted from event time to t=0).
/// Returned in "to-be-discounted-by-T units" by the caller — we return the true PV
/// here and the caller divides by the global discount before adding to chunk_sum
/// (which is then multiplied by the global discount). Net: rebate_pv flows through
/// untouched.
fn rebate_path_pv(p: &BarrierParams, hit_step: i32, n_steps: usize, dt: f64) -> f64 {
    let knocked = hit_step >= 0;
    match (p.kind, p.rebate_at_hit) {
        (BarrierKind::Out, true) => {
            if knocked {
                let tau = hit_step as f64 * dt;
                p.rebate * (-p.r * tau).exp()
            } else {
                0.0
            }
        }
        (BarrierKind::Out, false) => {
            if knocked {
                p.rebate * (-p.r * (n_steps as f64 * dt)).exp()
            } else {
                0.0
            }
        }
        (BarrierKind::In, _) => {
            if knocked {
                0.0
            } else {
                p.rebate * (-p.r * (n_steps as f64 * dt)).exp()
            }
        }
    }
}

fn check_barrier(s: f64, barrier: f64, direction: BarrierDirection) -> bool {
    match direction {
        BarrierDirection::Up => s >= barrier,
        BarrierDirection::Down => s <= barrier,
    }
}

fn intrinsic(s: f64, k: f64, option_type: OptionType) -> f64 {
    match option_type {
        OptionType::Call => (s - k).max(0.0),
        OptionType::Put => (k - s).max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> BarrierParams {
        BarrierParams {
            s: 100.0,
            k: 100.0,
            r: 0.05,
            q: 0.0,
            sigma: 0.2,
            t: 1.0,
            barrier: 120.0,
            direction: BarrierDirection::Up,
            kind: BarrierKind::Out,
            option_type: OptionType::Call,
            n_monitoring: None,
            rebate: 0.0,
            rebate_at_hit: false,
        }
    }

    #[test]
    fn test_in_out_parity_up_call() {
        let p_out = default_params();
        let mut p_in = p_out.clone();
        p_in.kind = BarrierKind::In;

        let out = barrier_analytical(&p_out).unwrap();
        let inp = barrier_analytical(&p_in).unwrap();
        let van = vanilla_price(&p_out);

        assert!(
            (out + inp - van).abs() < 1e-8,
            "in-out parity failed: out={out}, in={inp}, vanilla={van}"
        );
    }

    #[test]
    fn test_in_out_parity_down_put() {
        let mut p_out = default_params();
        p_out.direction = BarrierDirection::Down;
        p_out.kind = BarrierKind::Out;
        p_out.option_type = OptionType::Put;
        p_out.barrier = 80.0;

        let mut p_in = p_out.clone();
        p_in.kind = BarrierKind::In;

        let out = barrier_analytical(&p_out).unwrap();
        let inp = barrier_analytical(&p_in).unwrap();
        let van = vanilla_price(&p_out);

        assert!(
            (out + inp - van).abs() < 1e-8,
            "in-out parity failed: out={out}, in={inp}, vanilla={van}"
        );
    }

    #[test]
    fn test_up_out_call_less_than_vanilla() {
        let p = default_params();
        let barrier = barrier_analytical(&p).unwrap();
        let van = vanilla_price(&p);
        assert!(barrier <= van + 1e-10);
    }

    #[test]
    fn test_mc_vs_analytical() {
        let p = default_params();
        let analytical = barrier_analytical(&p).unwrap();
        let mc = barrier_mc(&p, 200_000, 1000, 42);
        // Discrete MC vs continuous analytical: allow wider tolerance
        let rel_err = (mc - analytical).abs() / analytical.max(1e-10);
        assert!(
            rel_err < 0.15,
            "MC vs analytical: mc={mc:.4}, analytical={analytical:.4}, rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn test_knocked_spot_above_up_barrier() {
        let mut p = default_params();
        p.s = 125.0; // above barrier of 120
        let price = barrier_analytical(&p).unwrap();
        assert!(price.abs() < 1e-10, "up-and-out should be 0 when S >= H");
    }

    #[test]
    fn test_knocked_in_equals_vanilla() {
        let mut p = default_params();
        p.s = 125.0;
        p.kind = BarrierKind::In;
        let price = barrier_analytical(&p).unwrap();
        let van = vanilla_price(&p);
        assert!((price - van).abs() < 1e-10);
    }

    #[test]
    fn test_all_eight_types_non_negative() {
        for dir in [BarrierDirection::Up, BarrierDirection::Down] {
            for kind in [BarrierKind::In, BarrierKind::Out] {
                for opt in [OptionType::Call, OptionType::Put] {
                    let barrier = match dir {
                        BarrierDirection::Up => 120.0,
                        BarrierDirection::Down => 80.0,
                    };
                    let p = BarrierParams {
                        barrier,
                        direction: dir,
                        kind,
                        option_type: opt,
                        ..default_params()
                    };
                    let price = barrier_analytical(&p).unwrap();
                    assert!(
                        price >= -1e-10,
                        "Negative price for {dir:?}/{kind:?}/{opt:?}: {price}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_bgk_constant_value() {
        // Reference: Broadie-Glasserman-Kou 1997, β ≈ 0.5826
        assert!((BGK_BETA - 0.5826).abs() < 1e-3);
    }

    #[test]
    fn test_bgk_converges_to_continuous_for_large_n() {
        // BGK shift decays as O(1/√n); 1e-2 is a realistic tolerance at n=10^5.
        let p_cont = default_params();
        let mut p_disc = p_cont.clone();
        p_disc.n_monitoring = Some(100_000);
        let cont = barrier_analytical(&p_cont).unwrap();
        let disc = barrier_analytical(&p_disc).unwrap();
        assert!(
            (cont - disc).abs() < 1e-2,
            "BGK should converge: cont={cont}, disc={disc}"
        );
    }

    #[test]
    fn test_bgk_up_out_call_lower_than_continuous() {
        // Discrete monitoring of up-and-out: barrier shifts UP (away from S),
        // so KO probability decreases, hence price is HIGHER than continuous.
        let p_cont = default_params();
        let mut p_disc = p_cont.clone();
        p_disc.n_monitoring = Some(12); // monthly
        let cont = barrier_analytical(&p_cont).unwrap();
        let disc = barrier_analytical(&p_disc).unwrap();
        assert!(
            disc > cont,
            "Monthly-monitored UO call should price higher than continuous: cont={cont}, disc={disc}"
        );
    }

    #[test]
    fn test_bgk_down_out_put_lower_than_continuous() {
        // Same logic for down-and-out put: discrete monitoring → fewer KOs → higher price.
        let mut p_cont = default_params();
        p_cont.direction = BarrierDirection::Down;
        p_cont.option_type = OptionType::Put;
        p_cont.barrier = 80.0;
        let mut p_disc = p_cont.clone();
        p_disc.n_monitoring = Some(12);
        let cont = barrier_analytical(&p_cont).unwrap();
        let disc = barrier_analytical(&p_disc).unwrap();
        assert!(
            disc > cont,
            "Monthly-monitored DO put should price higher than continuous: cont={cont}, disc={disc}"
        );
    }

    #[test]
    fn test_bgk_mc_unaffected() {
        // MC is already discrete; n_monitoring should not affect it.
        let p_cont = default_params();
        let mut p_disc = p_cont.clone();
        p_disc.n_monitoring = Some(12);
        let mc_cont = barrier_mc(&p_cont, 50_000, 252, 7);
        let mc_disc = barrier_mc(&p_disc, 50_000, 252, 7);
        assert!(
            (mc_cont - mc_disc).abs() < 1e-12,
            "MC should ignore n_monitoring (got {mc_cont} vs {mc_disc})"
        );
    }

    #[test]
    fn test_in_out_parity_all_types() {
        for dir in [BarrierDirection::Up, BarrierDirection::Down] {
            for opt in [OptionType::Call, OptionType::Put] {
                let barrier = match dir {
                    BarrierDirection::Up => 120.0,
                    BarrierDirection::Down => 80.0,
                };
                let p_out = BarrierParams {
                    barrier,
                    direction: dir,
                    kind: BarrierKind::Out,
                    option_type: opt,
                    ..default_params()
                };
                let mut p_in = p_out.clone();
                p_in.kind = BarrierKind::In;

                let out = barrier_analytical(&p_out).unwrap();
                let inp = barrier_analytical(&p_in).unwrap();
                let van = vanilla_price(&p_out);

                assert!(
                    (out + inp - van).abs() < 1e-6,
                    "in-out parity failed for {dir:?}/{opt:?}: out={out}, in={inp}, van={van}"
                );
            }
        }
    }
}
