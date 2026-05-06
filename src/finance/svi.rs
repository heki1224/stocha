use rayon::prelude::*;

/// SSVI (Surface SVI) parameterization of the total implied variance surface.
///
/// Total variance: w(k, θ) = (θ/2) * (1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + (1 - ρ²)))
/// where k = ln(K/F) is forward log-moneyness, θ = σ_ATM² · T is ATM total variance,
/// and φ(θ) = η / (θ^γ · (1 + θ)^(1-γ)) is the Heston-like power law.
///
/// Parameters: (η, γ, ρ) with η > 0, γ ∈ [0, 1], ρ ∈ (-1, 1).
#[derive(Debug, Clone)]
pub struct SsviParams {
    pub eta: f64,
    pub gamma: f64,
    pub rho: f64,
}

/// A single maturity slice: ATM total variance θ and time to expiry.
#[derive(Debug, Clone)]
pub struct SsviSlice {
    pub theta: f64,
    pub t: f64,
}

/// Result of SSVI calibration.
#[derive(Debug, Clone)]
pub struct SsviCalibResult {
    pub eta: f64,
    pub gamma: f64,
    pub rho: f64,
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl SsviParams {
    fn phi(&self, theta: f64) -> f64 {
        self.eta / (theta.powf(self.gamma) * (1.0 + theta).powf(1.0 - self.gamma))
    }

    /// Total implied variance w(k, θ).
    pub fn total_variance(&self, k: f64, theta: f64) -> f64 {
        let p = self.phi(theta);
        let pk_rho = p * k + self.rho;
        0.5 * theta * (1.0 + self.rho * p * k + (pk_rho * pk_rho + 1.0 - self.rho * self.rho).sqrt())
    }

    /// Implied volatility σ(K, T) = √(w(k, θ) / T).
    pub fn implied_vol(&self, k: f64, slice: &SsviSlice) -> f64 {
        let w = self.total_variance(k, slice.theta);
        (w / slice.t).sqrt()
    }

    /// ∂w/∂k (first derivative with respect to log-moneyness).
    pub fn dw_dk(&self, k: f64, theta: f64) -> f64 {
        let p = self.phi(theta);
        let pk_rho = p * k + self.rho;
        let disc = (pk_rho * pk_rho + 1.0 - self.rho * self.rho).sqrt();
        0.5 * theta * p * (self.rho + pk_rho / disc)
    }

    /// ∂²w/∂k² (second derivative with respect to log-moneyness).
    pub fn d2w_dk2(&self, k: f64, theta: f64) -> f64 {
        let p = self.phi(theta);
        let pk_rho = p * k + self.rho;
        let disc_sq = pk_rho * pk_rho + 1.0 - self.rho * self.rho;
        let disc = disc_sq.sqrt();
        0.5 * theta * p * p * (1.0 - self.rho * self.rho) / (disc * disc_sq)
    }

    /// ∂w/∂θ (partial derivative with respect to ATM total variance).
    /// Used for Dupire's formula in the time direction.
    pub fn dw_dtheta(&self, k: f64, theta: f64) -> f64 {
        let p = self.phi(theta);
        let dp = self.dphi_dtheta(theta);
        let pk_rho = p * k + self.rho;
        let disc = (pk_rho * pk_rho + 1.0 - self.rho * self.rho).sqrt();

        let term1 = 1.0 + self.rho * p * k + disc;
        let term2 = self.rho * dp * k + (pk_rho * dp * k) / disc;
        0.5 * (term1 + theta * term2)
    }

    fn dphi_dtheta(&self, theta: f64) -> f64 {
        let num = self.eta;
        let a = theta.powf(self.gamma);
        let b = (1.0 + theta).powf(1.0 - self.gamma);
        let denom = a * b;
        let da = self.gamma * theta.powf(self.gamma - 1.0);
        let db = (1.0 - self.gamma) * (1.0 + theta).powf(-self.gamma);
        -num * (da * b + a * db) / (denom * denom)
    }

    /// Check butterfly arbitrage condition: g(k) ≥ 0 where
    /// g(k) = (1 - k·w'/2w)² - w'/4·(1/w + 1/4) + w''/2
    /// (w' = ∂w/∂k, w'' = ∂²w/∂k²)
    pub fn butterfly_density(&self, k: f64, theta: f64) -> f64 {
        let w = self.total_variance(k, theta);
        if w <= 0.0 {
            return -1.0;
        }
        let wp = self.dw_dk(k, theta);
        let wpp = self.d2w_dk2(k, theta);
        let term1 = (1.0 - k * wp / (2.0 * w)).powi(2);
        let term2 = wp * wp / 4.0 * (1.0 / w + 0.25);
        term1 - term2 + 0.5 * wpp
    }
}

/// Dupire local variance from SSVI surface (analytical derivatives).
///
/// σ_loc²(k, T) = ∂w/∂T / g(k)
/// where g(k) = (1 - k·w'/(2w))² - (w')²/4·(1/w + 1/4) + w''/2
/// and ∂w/∂T = (∂w/∂θ)·(dθ/dT).
///
/// Returns local variance (σ_loc²). Caller takes sqrt for local vol.
pub fn dupire_local_var(
    params: &SsviParams,
    k: f64,
    theta: f64,
    dtheta_dt: f64,
) -> f64 {
    let g = params.butterfly_density(k, theta);
    if g <= 0.0 {
        return 0.0;
    }
    let dwdt = params.dw_dtheta(k, theta) * dtheta_dt;
    (dwdt / g).max(0.0)
}

/// Compute the full local volatility surface on a grid of (k, T) points.
///
/// `slices`: sorted by T, each with (θ_i, T_i).
/// `log_strikes`: grid of forward log-moneyness values.
///
/// Returns a 2D Vec (n_slices × n_strikes) of local volatilities.
pub fn local_vol_surface(
    params: &SsviParams,
    slices: &[SsviSlice],
    log_strikes: &[f64],
) -> Vec<Vec<f64>> {
    slices
        .par_iter()
        .enumerate()
        .map(|(i, slice)| {
            let dtheta_dt = if slices.len() == 1 {
                slice.theta / slice.t
            } else if i == 0 {
                (slices[1].theta - slice.theta) / (slices[1].t - slice.t)
            } else if i == slices.len() - 1 {
                (slice.theta - slices[i - 1].theta) / (slice.t - slices[i - 1].t)
            } else {
                (slices[i + 1].theta - slices[i - 1].theta)
                    / (slices[i + 1].t - slices[i - 1].t)
            };
            log_strikes
                .iter()
                .map(|&k| {
                    dupire_local_var(params, k, slice.theta, dtheta_dt).sqrt()
                })
                .collect()
        })
        .collect()
}

/// Calibrate SSVI parameters (η, γ, ρ) to market implied volatility data.
///
/// Input: a set of (k_i, θ_i, w_market_i) where w_market = σ_IV² · T.
/// Uses Levenberg-Marquardt with projected parameter bounds.
pub fn ssvi_calibrate(
    log_moneyness: &[f64],
    theta_values: &[f64],
    market_total_var: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<SsviCalibResult, String> {
    let n = log_moneyness.len();
    if n != theta_values.len() || n != market_total_var.len() {
        return Err("Input arrays must have the same length".to_string());
    }
    if n < 3 {
        return Err("Need at least 3 data points".to_string());
    }

    let mut eta = 1.0;
    let mut gamma = 0.5;
    let mut rho = -0.3;

    let mut lambda = 1e-3;
    let mut prev_sse = f64::MAX;

    let project = |e: &mut f64, g: &mut f64, r: &mut f64| {
        *e = e.clamp(0.01, 10.0);
        *g = g.clamp(0.01, 0.99);
        *r = r.clamp(-0.99, 0.99);
    };

    project(&mut eta, &mut gamma, &mut rho);

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        iterations = iter + 1;
        let params = SsviParams { eta, gamma, rho };

        let residuals: Vec<f64> = (0..n)
            .map(|i| params.total_variance(log_moneyness[i], theta_values[i]) - market_total_var[i])
            .collect();

        let sse: f64 = residuals.iter().map(|r| r * r).sum();

        if (prev_sse - sse).abs() < tol * tol && iter > 0 {
            converged = true;
            break;
        }
        prev_sse = sse;

        let h = 1e-6;
        let mut jac = vec![vec![0.0; 3]; n];
        for i in 0..n {
            let k = log_moneyness[i];
            let th = theta_values[i];
            let base = residuals[i] + market_total_var[i];

            let p_eta = SsviParams { eta: eta + h, gamma, rho };
            jac[i][0] = (p_eta.total_variance(k, th) - base) / h;

            let p_gamma = SsviParams { eta, gamma: gamma + h, rho };
            jac[i][1] = (p_gamma.total_variance(k, th) - base) / h;

            let p_rho = SsviParams { eta, gamma, rho: rho + h };
            jac[i][2] = (p_rho.total_variance(k, th) - base) / h;
        }

        // J^T J + λI
        let mut jtj = [[0.0f64; 3]; 3];
        let mut jtr = [0.0f64; 3];
        for i in 0..n {
            for a in 0..3 {
                jtr[a] += jac[i][a] * residuals[i];
                for b in 0..3 {
                    jtj[a][b] += jac[i][a] * jac[i][b];
                }
            }
        }
        for a in 0..3 {
            jtj[a][a] += lambda;
        }

        // Solve 3x3 system via Cramer's rule
        let delta = match solve_3x3(&jtj, &jtr) {
            Some(d) => d,
            None => break,
        };

        let mut new_eta = eta - delta[0];
        let mut new_gamma = gamma - delta[1];
        let mut new_rho = rho - delta[2];
        project(&mut new_eta, &mut new_gamma, &mut new_rho);

        let new_params = SsviParams { eta: new_eta, gamma: new_gamma, rho: new_rho };
        let new_sse: f64 = (0..n)
            .map(|i| {
                let r = new_params.total_variance(log_moneyness[i], theta_values[i]) - market_total_var[i];
                r * r
            })
            .sum();

        if new_sse < sse {
            eta = new_eta;
            gamma = new_gamma;
            rho = new_rho;
            lambda *= 0.5;
        } else {
            lambda *= 4.0;
        }
        lambda = lambda.clamp(1e-10, 1e6);
    }

    let rmse = (prev_sse / n as f64).sqrt();
    Ok(SsviCalibResult {
        eta,
        gamma,
        rho,
        rmse,
        iterations,
        converged,
    })
}

fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv_det = 1.0 / det;
    let x0 = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv_det;
    let x1 = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv_det;
    let x2 = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv_det;
    Some([x0, x1, x2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atm_total_variance() {
        let params = SsviParams { eta: 1.0, gamma: 0.5, rho: -0.3 };
        let theta = 0.04;
        let w = params.total_variance(0.0, theta);
        assert!((w - theta).abs() < 1e-12, "ATM total var should equal θ, got {}", w);
    }

    #[test]
    fn test_butterfly_positive_atm() {
        let params = SsviParams { eta: 0.5, gamma: 0.5, rho: -0.3 };
        let g = params.butterfly_density(0.0, 0.04);
        assert!(g > 0.0, "Butterfly density at ATM should be positive, got {}", g);
    }

    #[test]
    fn test_dw_dk_antisymmetric_rho_zero() {
        let params = SsviParams { eta: 1.0, gamma: 0.5, rho: 0.0 };
        let theta = 0.04;
        let dw_pos = params.dw_dk(0.1, theta);
        let dw_neg = params.dw_dk(-0.1, theta);
        assert!((dw_pos + dw_neg).abs() < 1e-12, "dw/dk should be antisymmetric when ρ=0");
    }

    #[test]
    fn test_d2w_dk2_positive() {
        let params = SsviParams { eta: 1.0, gamma: 0.5, rho: -0.3 };
        let wpp = params.d2w_dk2(0.0, 0.04);
        assert!(wpp > 0.0, "Second derivative should be positive (convexity)");
    }

    #[test]
    fn test_local_vol_positive() {
        let params = SsviParams { eta: 0.5, gamma: 0.5, rho: -0.3 };
        let lv = dupire_local_var(&params, 0.0, 0.04, 0.04);
        assert!(lv > 0.0, "Local variance should be positive");
    }

    #[test]
    fn test_ssvi_calibrate_roundtrip() {
        let true_params = SsviParams { eta: 1.2, gamma: 0.4, rho: -0.4 };
        let thetas = [0.01, 0.02, 0.04, 0.06, 0.09];
        let ks = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3];

        let mut log_m = Vec::new();
        let mut theta_v = Vec::new();
        let mut market_w = Vec::new();

        for &th in &thetas {
            for &k in &ks {
                log_m.push(k);
                theta_v.push(th);
                market_w.push(true_params.total_variance(k, th));
            }
        }

        let result = ssvi_calibrate(&log_m, &theta_v, &market_w, 200, 1e-12).unwrap();
        assert!(result.converged, "Should converge");
        assert!(result.rmse < 1e-6, "RMSE={}", result.rmse);
        assert!((result.eta - 1.2).abs() < 0.01, "eta={}", result.eta);
        assert!((result.gamma - 0.4).abs() < 0.01, "gamma={}", result.gamma);
        assert!((result.rho - -0.4).abs() < 0.01, "rho={}", result.rho);
    }
}
