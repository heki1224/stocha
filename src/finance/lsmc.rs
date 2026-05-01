use crate::finance::gbm::{gbm_paths, GbmParams};
use faer::{prelude::*, Mat};

pub struct LsmcParams {
    pub s0: f64,
    pub k: f64,
    pub r: f64,
    pub sigma: f64,
    pub t: f64,
    pub steps: usize,
    pub n_paths: usize,
    pub is_put: bool,
    pub poly_degree: usize, // 1–4
}

/// Price an American option via Longstaff-Schwartz Monte Carlo (LSMC).
///
/// Algorithm:
/// 1. Simulate GBM paths under the risk-neutral measure.
/// 2. Compute intrinsic values at expiry.
/// 3. Backward induction: regress continuation values against polynomial
///    basis of ITM spot prices (QR via faer), compare with exercise value.
///
/// Returns `(price, std_err)`.
pub fn lsmc_american_option(params: &LsmcParams, seed: u128) -> (f64, f64) {
    let dt = params.t / params.steps as f64;
    let discount = (-params.r * dt).exp();

    let gbm_params = GbmParams {
        s0: params.s0,
        mu: params.r,
        sigma: params.sigma,
        t: params.t,
        steps: params.steps,
        n_paths: params.n_paths,
        antithetic: false,
    };
    let paths = gbm_paths(&gbm_params, seed);

    let n = params.n_paths;
    let steps = params.steps;

    // Initialize cash flows at expiry.
    let mut cashflow: Vec<f64> = (0..n)
        .map(|i| intrinsic(paths[[i, steps]], params.k, params.is_put))
        .collect();

    // Backward induction from step (steps-1) down to step 1.
    for step in (1..steps).rev() {
        // Discount stored cash flows one step.
        for cf in cashflow.iter_mut() {
            *cf *= discount;
        }

        let spot: Vec<f64> = (0..n).map(|i| paths[[i, step]]).collect();
        let itm: Vec<usize> = (0..n)
            .filter(|&i| intrinsic(spot[i], params.k, params.is_put) > 0.0)
            .collect();

        let ncols = params.poly_degree + 1;
        if itm.len() < ncols + 1 {
            continue;
        }

        // Normalize spot prices to improve numerical conditioning.
        let s_mean = itm.iter().map(|&i| spot[i]).sum::<f64>() / itm.len() as f64;
        let s_std = {
            let v = itm.iter().map(|&i| (spot[i] - s_mean).powi(2)).sum::<f64>() / itm.len() as f64;
            v.sqrt().max(1e-12)
        };

        let m = itm.len();

        // Build X (m × ncols) and y (m × 1) matrices using faer.
        let x_mat = Mat::<f64>::from_fn(m, ncols, |row, col| {
            let x = (spot[itm[row]] - s_mean) / s_std;
            x.powi(col as i32)
        });
        let y_mat = Mat::<f64>::from_fn(m, 1, |row, _| cashflow[itm[row]]);

        // Least-squares solve via QR decomposition.
        let qr = x_mat.qr();
        let sol = qr.solve_lstsq(&y_mat);

        // sol has shape (ncols, 1); read coefficients from the single column.
        let col0 = sol.col_as_slice(0);
        let coeffs: Vec<f64> = (0..ncols).map(|i| col0[i]).collect();

        // Exercise if intrinsic > estimated continuation.
        for &path_i in &itm {
            let x = (spot[path_i] - s_mean) / s_std;
            let continuation: f64 = (0..ncols).map(|col| coeffs[col] * x.powi(col as i32)).sum();
            let exercise = intrinsic(spot[path_i], params.k, params.is_put);
            if exercise > continuation.max(0.0) {
                cashflow[path_i] = exercise;
            }
        }
    }

    // Discount from step 1 to step 0.
    for cf in cashflow.iter_mut() {
        *cf *= discount;
    }

    let mean = cashflow.iter().sum::<f64>() / n as f64;
    let std_err = if n <= 1 {
        0.0
    } else {
        let var = cashflow.iter().map(|&c| (c - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        (var / n as f64).sqrt()
    };

    (mean, std_err)
}

fn intrinsic(s: f64, k: f64, is_put: bool) -> f64 {
    if is_put {
        (k - s).max(0.0)
    } else {
        (s - k).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::norm_cdf;

    fn bs_put(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
        let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
    }

    #[test]
    fn test_lsmc_put_reasonable() {
        let params = LsmcParams {
            s0: 100.0,
            k: 100.0,
            r: 0.05,
            sigma: 0.20,
            t: 1.0,
            steps: 50,
            n_paths: 20_000,
            is_put: true,
            poly_degree: 3,
        };
        let (price, std_err) = lsmc_american_option(&params, 42);
        let european = bs_put(100.0, 100.0, 0.05, 0.20, 1.0);
        // American put >= European put (early exercise premium).
        assert!(
            price >= european - 3.0 * std_err,
            "price={}, eu={}",
            price,
            european
        );
        assert!(price < european + 2.0, "price too high: {}", price);
        assert!(std_err < 0.10, "std_err too large: {}", std_err);
    }

    #[test]
    fn test_lsmc_call_reasonable() {
        // Deep ITM call: price should be close to intrinsic (S0 - K * exp(-r*T))
        let params = LsmcParams {
            s0: 110.0,
            k: 100.0,
            r: 0.05,
            sigma: 0.20,
            t: 1.0,
            steps: 50,
            n_paths: 20_000,
            is_put: false,
            poly_degree: 3,
        };
        let (price, std_err) = lsmc_american_option(&params, 42);
        // American call on non-dividend-paying stock == European call (no early exercise).
        // Price must be positive and std_err finite.
        assert!(price > 0.0, "call price must be positive: {}", price);
        assert!(
            std_err >= 0.0 && std_err.is_finite(),
            "std_err invalid: {}",
            std_err
        );
        // Rough sanity: call price < S0
        assert!(price < params.s0, "call price exceeds S0: {}", price);
    }
}
