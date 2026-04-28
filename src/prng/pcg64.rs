use rand_pcg::rand_core::Rng;
use rand_pcg::Lcg128CmDxsm64;
use serde::{Deserialize, Serialize};

pub struct Pcg64Dxsm {
    rng: Lcg128CmDxsm64,
    seed: u128,
}

#[derive(Serialize, Deserialize)]
struct FullState {
    version: u8,
    rng: Lcg128CmDxsm64,
    seed: u128,
}

#[derive(Deserialize)]
struct SeedOnly {
    seed: u128,
}

impl Pcg64Dxsm {
    pub fn new(seed: u128) -> Self {
        let state = seed;
        let stream = seed.wrapping_add(0xda3e_39cb_94b9_5bdb_7ef5_5da5_c037_4f5a_u128);
        Self {
            rng: Lcg128CmDxsm64::new(state, stream),
            seed,
        }
    }

    #[allow(dead_code)]
    pub fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn advance(&mut self, delta: u128) {
        self.rng.advance(delta);
    }

    #[allow(dead_code)]
    pub fn fork(&self) -> Self {
        Self {
            rng: self.rng.clone(),
            seed: self.seed,
        }
    }

    pub fn seed(&self) -> u128 {
        self.seed
    }

    /// Serialize the full internal state (RNG position + seed) to JSON.
    pub fn save_state(&self) -> String {
        let state = FullState {
            version: 2,
            rng: self.rng.clone(),
            seed: self.seed,
        };
        serde_json::to_string(&state).unwrap_or_default()
    }

    /// Restore from JSON. Accepts both full-state (v2) and legacy seed-only format.
    pub fn from_state(json: &str) -> Result<Self, String> {
        if let Ok(full) = serde_json::from_str::<FullState>(json) {
            return Ok(Self {
                rng: full.rng,
                seed: full.seed,
            });
        }
        let seed_only: SeedOnly =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        Ok(Self::new(seed_only.seed))
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

    #[test]
    fn test_full_state_round_trip() {
        let mut rng = Pcg64Dxsm::new(42);
        for _ in 0..500 {
            rng.next_u64();
        }
        let json = rng.save_state();
        let mut rng2 = Pcg64Dxsm::from_state(&json).unwrap();
        let expected: Vec<u64> = (0..100).map(|_| rng.next_u64()).collect();
        let actual: Vec<u64> = (0..100).map(|_| rng2.next_u64()).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_legacy_seed_only_format() {
        let json = r#"{"seed":42}"#;
        let mut rng = Pcg64Dxsm::from_state(json).unwrap();
        let mut fresh = Pcg64Dxsm::new(42);
        let v1: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();
        let v2: Vec<u64> = (0..10).map(|_| fresh.next_u64()).collect();
        assert_eq!(v1, v2);
    }
}
