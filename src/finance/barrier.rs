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
}

pub fn barrier_analytical(p: &BarrierParams) -> Option<f64> {
    let s = p.s;
    let k = p.k;
    let h = p.barrier;
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
                return Some(knocked_value(p));
            }
        }
        BarrierDirection::Down => {
            if s <= h {
                return Some(knocked_value(p));
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

    Some(price.max(0.0))
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
                let mut knocked = false;

                for _ in 0..n_steps {
                    let z = NormalSampler::sample(&mut rng);
                    s *= (drift + diffusion * z).exp();

                    if check_barrier(s, p.barrier, p.direction) {
                        knocked = true;
                        break;
                    }
                }

                let payoff = match p.kind {
                    BarrierKind::Out => {
                        if knocked { 0.0 } else { intrinsic(s, p.k, p.option_type) }
                    }
                    BarrierKind::In => {
                        if knocked { intrinsic(s, p.k, p.option_type) } else { 0.0 }
                    }
                };
                chunk_sum += payoff;
            }
            chunk_sum
        })
        .sum();

    discount * total_payoff / n_paths as f64
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
