use rand_pcg::rand_core::Rng;
use rand_pcg::Lcg128CmDxsm64;
use serde::{Deserialize, Serialize};

/// PCG64DXSM pseudo-random number generator.
///
/// Uses the same algorithm as NumPy's default RNG (PCG64DXSM).
/// Supports jump-ahead for independent parallel streams.
pub struct Pcg64Dxsm {
    rng: Lcg128CmDxsm64,
    /// Seed retained for reproducibility and state serialization.
    seed: u128,
}

/// Serializable RNG state for checkpointing and audit trails.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Pcg64DxsmState {
    pub seed: u128,
}

impl Pcg64Dxsm {
    /// Create a new RNG from a 128-bit seed.
    pub fn new(seed: u128) -> Self {
        // Derive state and stream from seed to ensure independence.
        let state = seed;
        let stream = seed.wrapping_add(0xda3e_39cb_94b9_5bdb_7ef5_5da5_c037_4f5a_u128);
        Self {
            rng: Lcg128CmDxsm64::new(state, stream),
            seed,
        }
    }

    /// Generate one u64 random value.
    #[allow(dead_code)]
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    /// Generate one f64 value in [0, 1) with 53-bit precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Advance the RNG state by `delta` steps (for parallel stream splitting).
    ///
    /// Block splitting: assign stream `i` by calling `advance(i * block_size)`.
    pub fn advance(&mut self, delta: u128) {
        self.rng.advance(delta);
    }

    /// Clone the internal RNG for stream forking.
    #[allow(dead_code)]
    pub fn fork(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            seed: self.seed,
        }
    }

    /// Return the original seed value.
    pub fn seed(&self) -> u128 {
        self.seed
    }

    /// Serialize the seed to a JSON string for reproducibility.
    ///
    /// **Limitation**: records the *original seed only*, not the full internal state.
    /// If the RNG has been advanced (e.g. via `advance()` or by generating samples),
    /// restoring from this snapshot will replay the sequence **from the beginning**,
    /// not from the current position. Use this for audit trails and exact reproducibility
    /// from a fixed starting point, not for mid-stream checkpointing.
    pub fn save_state(&self) -> String {
        let state = Pcg64DxsmState { seed: self.seed };
        serde_json::to_string(&state).unwrap_or_default()
    }

    /// Restore an RNG from a JSON string produced by `save_state`.
    ///
    /// Constructs a fresh RNG from the recorded seed. The restored RNG is equivalent
    /// to calling `Pcg64Dxsm::new(seed)` — it starts from the beginning of the
    /// sequence, regardless of how far the original RNG had advanced.
    pub fn from_state(json: &str) -> Result<Self, String> {
        let state: Pcg64DxsmState =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        Ok(Self::new(state.seed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_reproducibility() {
        let mut rng1 = Pcg64Dxsm::new(42);
        let mut rng2 = Pcg64Dxsm::new(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut rng1 = Pcg64Dxsm::new(42);
        let mut rng2 = Pcg64Dxsm::new(43);
        let v1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let v2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_f64_in_unit_interval() {
        let mut rng = Pcg64Dxsm::new(0);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_advance_produces_independent_streams() {
        let mut rng_a = Pcg64Dxsm::new(42);
        let mut rng_b = Pcg64Dxsm::new(42);
        rng_b.advance(1_000_000);

        let v_a: Vec<u64> = (0..10).map(|_| rng_a.next_u64()).collect();
        let v_b: Vec<u64> = (0..10).map(|_| rng_b.next_u64()).collect();
        assert_ne!(v_a, v_b);
    }

    #[test]
    fn test_save_restore_state() {
        let rng = Pcg64Dxsm::new(12345);
        let json = rng.save_state();
        let rng2 = Pcg64Dxsm::from_state(&json).unwrap();
        assert_eq!(rng.seed(), rng2.seed());
    }
}
