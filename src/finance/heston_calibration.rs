use super::bs::{bs_price, bs_vega};
use super::heston_cos::heston_cos_price_vec;
use rayon::prelude::*;

pub struct HestonCalibrationResult {
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
    pub feller_satisfied: bool,
}

const V0_MIN: f64 = 1e-6;
const V0_MAX: f64 = 3.0;
const KAPPA_MIN: f64 = 1e-4;
const KAPPA_MAX: f64 = 20.0;
const THETA_MIN: f64 = 1e-6;
const THETA_MAX: f64 = 3.0;
const XI_MIN: f64 = 1e-4;
const XI_MAX: f64 = 5.0;
const RHO_MIN: f64 = -0.999;
const RHO_MAX: f64 = 0.999;

const VEGA_FLOOR: f64 = 1e-4;
const N_PARAMS: usize = 5;

struct MarketData<'a> {
    strikes: &'a [f64],
    prices: &'a [f64],
    is_call: &'a [bool],
    maturities: &'a [f64],
    s0: f64,
    r: f64,
    weights: Vec<f64>,
}

fn clip(val: f64, lo: f64, hi: f64) -> f64 {
    val.max(lo).min(hi)
}

fn pack(v0: f64, kappa: f64, theta: f64, xi: f64, rho: f64) -> [f64; N_PARAMS] {
    [v0, kappa, theta, xi, rho]
}

fn clip_params(p: &mut [f64; N_PARAMS]) {
    p[0] = clip(p[0], V0_MIN, V0_MAX);
    p[1] = clip(p[1], KAPPA_MIN, KAPPA_MAX);
    p[2] = clip(p[2], THETA_MIN, THETA_MAX);
    p[3] = clip(p[3], XI_MIN, XI_MAX);
    p[4] = clip(p[4], RHO_MIN, RHO_MAX);
}

fn model_prices(p: &[f64; N_PARAMS], data: &MarketData, n_cos: usize) -> Vec<f64> {
    let (v0, kappa, theta, xi, rho) = (p[0], p[1], p[2], p[3], p[4]);

    // Group by maturity for efficient COS evaluation
    let mut results = vec![0.0_f64; data.strikes.len()];
    let mut mat_groups: std::collections::HashMap<u64, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &tau) in data.maturities.iter().enumerate() {
        let key = tau.to_bits();
        mat_groups.entry(key).or_default().push(i);
    }

    for (tau_bits, indices) in &mat_groups {
        let tau = f64::from_bits(*tau_bits);
        let ks: Vec<f64> = indices.iter().map(|&i| data.strikes[i]).collect();
        let calls: Vec<bool> = indices.iter().map(|&i| data.is_call[i]).collect();
        let prices = heston_cos_price_vec(
            data.s0, v0, data.r, kappa, theta, xi, rho, tau, &ks, &calls, n_cos,
        );
        for (j, &i) in indices.iter().enumerate() {
            results[i] = prices[j];
        }
    }
    results
}

fn weighted_residuals(p: &[f64; N_PARAMS], data: &MarketData, n_cos: usize) -> Vec<f64> {
    let mp = model_prices(p, data, n_cos);
    mp.iter()
        .zip(data.prices.iter())
        .zip(data.weights.iter())
        .map(|((&m, &mkt), &w)| {
            let r = (m - mkt) * w;
            if r.is_finite() {
                r
            } else {
                f64::MAX / 4.0
            }
        })
        .collect()
}

fn jacobian(p: &[f64; N_PARAMS], data: &MarketData, n_cos: usize) -> Vec<[f64; N_PARAMS]> {
    let n_obs = data.strikes.len();
    let bounds_lo = [V0_MIN, KAPPA_MIN, THETA_MIN, XI_MIN, RHO_MIN];
    let bounds_hi = [V0_MAX, KAPPA_MAX, THETA_MAX, XI_MAX, RHO_MAX];

    let base_residuals = weighted_residuals(p, data, n_cos);

    // Compute columns in parallel (one per parameter)
    let columns: Vec<Vec<f64>> = (0..N_PARAMS)
        .into_par_iter()
        .map(|j| {
            let h = (p[j].abs() * 0.01).max(1e-8);

            let at_lower = p[j] - h < bounds_lo[j];
            let at_upper = p[j] + h > bounds_hi[j];

            if at_lower && at_upper {
                vec![0.0; n_obs]
            } else if at_lower {
                // Forward difference
                let mut p_up = *p;
                p_up[j] = clip(p[j] + h, bounds_lo[j], bounds_hi[j]);
                let r_up = weighted_residuals(&p_up, data, n_cos);
                let actual_h = p_up[j] - p[j];
                r_up.iter()
                    .zip(base_residuals.iter())
                    .map(|(&u, &b)| (u - b) / actual_h)
                    .collect()
            } else if at_upper {
                // Backward difference
                let mut p_dn = *p;
                p_dn[j] = clip(p[j] - h, bounds_lo[j], bounds_hi[j]);
                let r_dn = weighted_residuals(&p_dn, data, n_cos);
                let actual_h = p[j] - p_dn[j];
                base_residuals
                    .iter()
                    .zip(r_dn.iter())
                    .map(|(&b, &d)| (b - d) / actual_h)
                    .collect()
            } else {
                // Central difference
                let mut p_up = *p;
                let mut p_dn = *p;
                p_up[j] = p[j] + h;
                p_dn[j] = p[j] - h;
                let r_up = weighted_residuals(&p_up, data, n_cos);
                let r_dn = weighted_residuals(&p_dn, data, n_cos);
                r_up.iter()
                    .zip(r_dn.iter())
                    .map(|(&u, &d)| (u - d) / (2.0 * h))
                    .collect()
            }
        })
        .collect();

    let mut jac = vec![[0.0; N_PARAMS]; n_obs];
    for j in 0..N_PARAMS {
        for i in 0..n_obs {
            jac[i][j] = columns[j][i];
        }
    }
    jac
}

fn solve_5x5(jtj: &[[f64; N_PARAMS]; N_PARAMS], jtr: &[f64; N_PARAMS]) -> [f64; N_PARAMS] {
    // Gaussian elimination with partial pivoting for 5x5
    let mut a = *jtj;
    let mut b = *jtr;

    for col in 0..N_PARAMS {
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..N_PARAMS {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        a.swap(col, max_row);
        b.swap(col, max_row);

        let pivot = a[col][col];
        if pivot.abs() < 1e-30 {
            return [0.0; N_PARAMS];
        }
        for row in (col + 1)..N_PARAMS {
            let factor = a[row][col] / pivot;
            for c in col..N_PARAMS {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0; N_PARAMS];
    for i in (0..N_PARAMS).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N_PARAMS {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    x
}

pub fn calibrate(
    strikes: &[f64],
    maturities: &[f64],
    market_prices: &[f64],
    is_call: &[bool],
    s0: f64,
    r: f64,
    max_iter: usize,
    tol: f64,
    n_cos: usize,
) -> Result<HestonCalibrationResult, String> {
    let n_obs = strikes.len();
    if n_obs != maturities.len() || n_obs != market_prices.len() || n_obs != is_call.len() {
        return Err("all input arrays must have the same length".into());
    }
    if n_obs < N_PARAMS {
        return Err(format!("at least {N_PARAMS} observations required"));
    }
    if s0 <= 0.0 {
        return Err("s0 must be positive".into());
    }

    for i in 0..n_obs {
        if !strikes[i].is_finite() || !maturities[i].is_finite() || !market_prices[i].is_finite() {
            return Err(format!("non-finite input at index {i}"));
        }
        if strikes[i] <= 0.0 || maturities[i] <= 0.0 || market_prices[i] < 0.0 {
            return Err(format!("invalid value at index {i}"));
        }
    }

    // Compute Vega weights: 1/max(bs_vega, VEGA_FLOOR)
    let weights: Vec<f64> = (0..n_obs)
        .map(|i| {
            let sigma_guess = bs_iv_from_price(
                market_prices[i],
                s0,
                strikes[i],
                r,
                maturities[i],
                is_call[i],
            );
            let v = bs_vega(s0, strikes[i], r, maturities[i], sigma_guess);
            1.0 / v.max(VEGA_FLOOR)
        })
        .collect();

    let data = MarketData {
        strikes,
        prices: market_prices,
        is_call,
        maturities,
        s0,
        r,
        weights,
    };

    // Heuristic initial guess
    let atm_idx = strikes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| ((**a - s0).abs()).partial_cmp(&((**b - s0).abs())).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let atm_sigma = bs_iv_from_price(
        market_prices[atm_idx],
        s0,
        strikes[atm_idx],
        r,
        maturities[atm_idx],
        is_call[atm_idx],
    );
    let v0_init = atm_sigma * atm_sigma;

    let mut params = pack(v0_init, 2.0, v0_init, 0.5, -0.7);
    clip_params(&mut params);

    let mut lambda = 1e-3;
    let mut residuals = weighted_residuals(&params, &data, n_cos);
    let mut cost: f64 = residuals.iter().map(|r| r * r).sum();
    let mut converged = false;
    let mut iter = 0;

    for it in 0..max_iter {
        iter = it + 1;
        let jac = jacobian(&params, &data, n_cos);

        // J^T J + λI
        let mut jtj = [[0.0; N_PARAMS]; N_PARAMS];
        let mut jtr = [0.0; N_PARAMS];
        for i in 0..n_obs {
            for a in 0..N_PARAMS {
                jtr[a] -= jac[i][a] * residuals[i];
                for b in 0..N_PARAMS {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }
        for a in 0..N_PARAMS {
            jtj[a][a] += lambda * (1.0 + jtj[a][a]);
        }

        let step = solve_5x5(&jtj, &jtr);

        if step.iter().all(|s| !s.is_finite()) {
            lambda *= 10.0;
            continue;
        }

        let mut new_params = params;
        for j in 0..N_PARAMS {
            new_params[j] += step[j];
        }
        clip_params(&mut new_params);

        let new_residuals = weighted_residuals(&new_params, &data, n_cos);
        let new_cost: f64 = new_residuals.iter().map(|r| r * r).sum();

        if new_cost < cost {
            params = new_params;
            residuals = new_residuals;
            let improvement = (cost - new_cost) / cost.max(1e-30);
            cost = new_cost;
            lambda *= 0.3;
            lambda = lambda.max(1e-10);

            let grad_norm = jtr.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
            let step_norm = step.iter().map(|s| s.abs()).fold(0.0_f64, f64::max);
            if (improvement < tol && grad_norm < tol) || step_norm < tol * 1e-2 || cost < tol * tol
            {
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

    let unweighted_residuals = model_prices(&params, &data, n_cos)
        .iter()
        .zip(market_prices.iter())
        .map(|(&m, &mkt)| m - mkt)
        .collect::<Vec<_>>();
    let rmse = (unweighted_residuals.iter().map(|r| r * r).sum::<f64>() / n_obs as f64).sqrt();

    let feller = 2.0 * params[1] * params[2] > params[3] * params[3];

    Ok(HestonCalibrationResult {
        v0: params[0],
        kappa: params[1],
        theta: params[2],
        xi: params[3],
        rho: params[4],
        rmse,
        iterations: iter,
        converged,
        feller_satisfied: feller,
    })
}

fn bs_iv_from_price(price: f64, s: f64, k: f64, r: f64, t: f64, is_call: bool) -> f64 {
    let mut sigma = 0.2;
    for _ in 0..50 {
        let p = bs_price(s, k, r, t, sigma, is_call);
        let v = bs_vega(s, k, r, t, sigma);
        if v < 1e-20 {
            break;
        }
        let diff = p - price;
        if diff.abs() < 1e-12 {
            break;
        }
        sigma -= diff / v;
        if sigma <= 0.001 {
            sigma = 0.001;
        }
        if sigma > 5.0 {
            sigma = 5.0;
        }
    }
    sigma.max(0.01)
}
