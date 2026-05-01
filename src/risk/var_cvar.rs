/// Compute Value-at-Risk and Conditional VaR (Expected Shortfall) from a return series.
///
/// Losses are defined as negative returns. VaR is the loss exceeded with probability
/// `1 - confidence`. CVaR is the expected loss given that the loss exceeds VaR.
///
/// Returns `(var, cvar)` where both values are positive (represent loss magnitudes).
pub fn var_cvar(returns: &[f64], confidence: f64) -> Result<(f64, f64), String> {
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err("confidence must be in (0, 1)".into());
    }
    if returns.is_empty() {
        return Err("returns must not be empty".into());
    }
    if returns.iter().any(|r| !r.is_finite()) {
        return Err("returns must not contain NaN or Inf".into());
    }

    // Convert to losses (positive = loss).
    let mut losses: Vec<f64> = returns.iter().map(|&r| -r).collect();
    losses.sort_unstable_by(|a, b| a.total_cmp(b));

    let n = losses.len();
    // Index of the VaR threshold: the smallest loss exceeded with prob (1 - confidence).
    let idx = (confidence * n as f64).ceil() as usize;
    let idx = idx.min(n - 1);

    let var = losses[idx];

    // CVaR: mean of losses at or beyond the VaR threshold (losses[idx] == VaR is included).
    // This implements E[Loss | Loss >= VaR], the "inclusive" Expected Shortfall convention.
    // For continuous distributions the boundary point has measure zero; for discrete samples
    // (as here) the VaR point itself contributes to the average.
    let tail = &losses[idx..];
    let cvar = if tail.is_empty() {
        var
    } else {
        tail.iter().sum::<f64>() / tail.len() as f64
    };

    Ok((var, cvar))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_cvar_simple() {
        // Losses: -returns → sorted losses = [0.01,0.02,...,0.10]
        let returns: Vec<f64> = (1..=10).map(|i| -(i as f64 * 0.01)).collect();
        let (var, cvar) = var_cvar(&returns, 0.9).unwrap();
        // 90th percentile of losses: idx = ceil(0.9*10) = 9 → losses[9] = 0.10
        assert!((var - 0.10).abs() < 1e-10);
        // CVaR = mean of losses[9..] = 0.10
        assert!((cvar - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_var_cvar_all_positive_returns() {
        // All returns are gains → all losses are negative → VaR < 0 (no loss at confidence level).
        let returns: Vec<f64> = (1..=20).map(|i| i as f64 * 0.01).collect();
        let (var, cvar) = var_cvar(&returns, 0.95).unwrap();
        // losses = [-0.20, -0.19, ..., -0.01]; at 95% idx = ceil(19) = 19 → losses[19] = -0.01
        assert!(
            var < 0.0,
            "VaR should be negative when all returns are positive: {}",
            var
        );
        assert!(
            cvar <= var + 1e-10,
            "CVaR >= VaR must hold: var={}, cvar={}",
            var,
            cvar
        );
    }
}
