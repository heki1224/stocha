/// Compute Value-at-Risk and Conditional VaR (Expected Shortfall) from a return series.
///
/// Losses are defined as negative returns. VaR is the loss exceeded with probability
/// `1 - confidence`. CVaR is the expected loss given that the loss exceeds VaR.
///
/// Returns `(var, cvar)` where both values are positive (represent loss magnitudes).
pub fn var_cvar(returns: &[f64], confidence: f64) -> (f64, f64) {
    assert!(
        confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0, 1)"
    );
    assert!(!returns.is_empty(), "returns must not be empty");

    // Convert to losses (positive = loss).
    let mut losses: Vec<f64> = returns.iter().map(|&r| -r).collect();
    losses.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n = losses.len();
    // Index of the VaR threshold: the smallest loss exceeded with prob (1 - confidence).
    let idx = (confidence * n as f64).ceil() as usize;
    let idx = idx.min(n - 1);

    let var = losses[idx];

    // CVaR: mean of losses strictly beyond the VaR index.
    let tail = &losses[idx..];
    let cvar = if tail.is_empty() {
        var
    } else {
        tail.iter().sum::<f64>() / tail.len() as f64
    };

    (var, cvar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_cvar_simple() {
        // Losses: -returns → sorted losses = [0.01,0.02,...,0.10]
        let returns: Vec<f64> = (1..=10).map(|i| -(i as f64 * 0.01)).collect();
        let (var, cvar) = var_cvar(&returns, 0.9);
        // 90th percentile of losses: idx = ceil(0.9*10) = 9 → losses[9] = 0.10
        assert!((var - 0.10).abs() < 1e-10);
        // CVaR = mean of losses[9..] = 0.10
        assert!((cvar - 0.10).abs() < 1e-10);
    }
}
