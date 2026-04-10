//! DecayingIntegral: the core Volterra second-kind integral primitive.
//!
//! Laboratory layer: pure math. Caller provides timestamps.

use serde::{Deserialize, Serialize};

/// A single Volterra second-kind integral with caller-provided timestamps.
///
/// Each integral decays independently: I(t+dt) = impulse + I(t) * exp(-lambda * dt)
/// where dt is measured from THIS integral's last update, not a shared clock.
///
/// The caller owns the time source. Pass seconds since an arbitrary epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayingIntegral {
    value: f64,
    lambda: f64,
    last_update: f64,
}

impl DecayingIntegral {
    /// Create a new integral with decay rate lambda, starting at time `now`.
    pub fn new(lambda: f64, now: f64) -> Self {
        Self {
            value: 0.0,
            lambda,
            last_update: now,
        }
    }

    /// Record an impulse at time `now`, applying exponential decay since last update.
    pub fn record(&mut self, impulse: f64, now: f64) {
        let dt = now - self.last_update;
        self.last_update = now;
        self.value = impulse + self.value * (-self.lambda * dt).exp();
    }

    /// Read the current decayed value at time `now` without mutating state.
    pub fn current_value(&self, now: f64) -> f64 {
        let dt = now - self.last_update;
        self.value * (-self.lambda * dt).exp()
    }

    /// The decay rate.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// A bank of N decaying integrals, one per pressure channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegralBank {
    integrals: Vec<DecayingIntegral>,
}

impl IntegralBank {
    /// Create a bank from a slice of decay rates, all starting at time `now`.
    pub fn from_lambdas(lambdas: &[f64], now: f64) -> Self {
        Self {
            integrals: lambdas
                .iter()
                .map(|&lam| DecayingIntegral::new(lam, now))
                .collect(),
        }
    }

    /// Number of channels.
    pub fn dim(&self) -> usize {
        self.integrals.len()
    }

    /// Record an impulse on channel `idx` at time `now`.
    pub fn record(&mut self, idx: usize, impulse: f64, now: f64) {
        self.integrals[idx].record(impulse, now);
    }

    /// Read the current decayed value of channel `idx` at time `now`.
    pub fn current_value(&self, idx: usize, now: f64) -> f64 {
        self.integrals[idx].current_value(now)
    }

    /// Read all current values as a vector.
    pub fn current_values(&self, now: f64) -> Vec<f64> {
        self.integrals.iter().map(|i| i.current_value(now)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_integral_starts_at_zero() {
        let i = DecayingIntegral::new(1.0, 0.0);
        assert_eq!(i.current_value(0.0), 0.0);
    }

    #[test]
    fn impulse_recorded() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(5.0, 0.0);
        assert!((i.current_value(0.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn exponential_decay() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(1.0, 0.0);
        // After 1 second with lambda=1: value = 1.0 * exp(-1) ~ 0.3679
        let v = i.current_value(1.0);
        assert!((v - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn accumulated_impulses_with_decay() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(1.0, 0.0);
        // At t=1, value = exp(-1). Record another impulse of 1.0.
        i.record(1.0, 1.0);
        // Now value = 1.0 + exp(-1) ~ 1.3679
        let expected = 1.0 + (-1.0_f64).exp();
        assert!((i.current_value(1.0) - expected).abs() < 1e-12);
    }

    #[test]
    fn current_value_does_not_mutate() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(1.0, 0.0);
        let v1 = i.current_value(1.0);
        let v2 = i.current_value(1.0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn integral_bank_basic() {
        let mut bank = IntegralBank::from_lambdas(&[1.0, 2.0], 0.0);
        assert_eq!(bank.dim(), 2);
        bank.record(0, 1.0, 0.0);
        bank.record(1, 1.0, 0.0);
        // Channel 1 decays faster (lambda=2)
        let v0 = bank.current_value(0, 1.0);
        let v1 = bank.current_value(1, 1.0);
        assert!(v0 > v1, "Channel with lower lambda should retain more value");
    }

    // ── Edge-case tests ─────────────────────────────────────────────

    #[test]
    fn negative_impulse() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(-3.0, 0.0);
        assert!((i.current_value(0.0) - (-3.0)).abs() < 1e-12);
        // Negative value still decays toward zero
        let v = i.current_value(1.0);
        assert!(v > -3.0, "negative value should decay toward zero");
        assert!(v < 0.0, "negative value should stay negative without positive impulse");
    }

    #[test]
    fn negative_impulse_mixed_with_positive() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(5.0, 0.0);
        i.record(-3.0, 0.0);
        // 5.0 * exp(0) + (-3.0) = 2.0
        assert!((i.current_value(0.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn very_large_dt_decays_to_near_zero() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(1e6, 0.0);
        // After dt=1000 with lambda=1, exp(-1000) is essentially 0
        let v = i.current_value(1000.0);
        assert!(v.abs() < 1e-100, "large dt should fully decay the value");
    }

    #[test]
    fn very_large_dt_in_record_decays_prior_value() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(100.0, 0.0);
        i.record(1.0, 1000.0);
        // Prior value fully decayed, only the new impulse remains
        assert!((i.current_value(1000.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn very_small_dt_preserves_value() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(10.0, 0.0);
        // dt = 1e-15: exp(-1e-15) ≈ 1
        let v = i.current_value(1e-15);
        assert!((v - 10.0).abs() < 1e-10, "tiny dt should barely decay");
    }

    #[test]
    fn very_small_dt_between_records() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(5.0, 0.0);
        i.record(3.0, 1e-15);
        // 5.0 * exp(-1e-15) + 3.0 ≈ 8.0
        assert!((i.current_value(1e-15) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn dt_zero_no_decay() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(7.0, 0.0);
        // current_value at same timestamp: no decay
        assert!((i.current_value(0.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn dt_zero_record_accumulates() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(3.0, 0.0);
        i.record(4.0, 0.0);
        // dt=0 means exp(0)=1, so value = 4.0 + 3.0 * 1.0 = 7.0
        assert!((i.current_value(0.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn rapid_alternating_impulses() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        // Alternate +1 / -1 at tiny intervals
        for k in 0..100 {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            i.record(sign, k as f64 * 0.001);
        }
        // The value should remain bounded and finite
        let v = i.current_value(0.1);
        assert!(v.is_finite(), "alternating impulses should produce finite value");
        assert!(v.abs() < 100.0, "alternating impulses should not blow up");
    }

    #[test]
    fn rapid_alternating_impulses_cancel() {
        let mut i = DecayingIntegral::new(0.0, 0.0);
        // With lambda=0 (no decay), equal +/- impulses cancel perfectly
        i.record(1.0, 0.0);
        i.record(-1.0, 0.0);
        assert!((i.current_value(0.0)).abs() < 1e-12);
    }

    #[test]
    fn record_and_current_value_same_timestamp() {
        let mut i = DecayingIntegral::new(1.0, 0.0);
        i.record(5.0, 10.0);
        // current_value at the same timestamp as the last record: no additional decay
        assert!((i.current_value(10.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn current_value_before_record_timestamp() {
        let mut i = DecayingIntegral::new(1.0, 10.0);
        i.record(1.0, 10.0);
        // Querying at t=9 (before last_update=10) gives exp(-lambda*(-1)) = exp(1)
        // This is the current behavior: negative dt causes growth in current_value
        let v = i.current_value(9.0);
        assert!((v - (1.0_f64).exp()).abs() < 1e-12);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// For any sequence of non-negative impulses, the value at t2 cannot exceed
        /// the value at t1 plus the sum of impulses recorded between t1 and t2.
        #[test]
        fn monotonic_decay_invariant(
            lambda in 0.01_f64..10.0,
            impulses in prop::collection::vec((0.0_f64..100.0, 0.0_f64..1.0), 1..50),
        ) {
            let mut integral = DecayingIntegral::new(lambda, 0.0);

            // Build sorted timestamps from relative deltas
            let mut timestamps: Vec<f64> = Vec::with_capacity(impulses.len());
            let mut t = 0.0;
            for &(_, dt) in &impulses {
                t += dt;
                timestamps.push(t);
            }

            // Record all impulses
            for (idx, &(imp, _)) in impulses.iter().enumerate() {
                integral.record(imp, timestamps[idx]);
            }

            // Check invariant between every pair of observation points after the
            // last recorded impulse (so no new impulses intervene).
            let t_final = *timestamps.last().unwrap();
            let t1 = t_final;
            let t2 = t_final + 1.0;

            let v1 = integral.current_value(t1);
            let v2 = integral.current_value(t2);

            // No impulses between t1 and t2, so v2 <= v1 (pure decay)
            prop_assert!(
                v2 <= v1 + 1e-10,
                "decay violated: v1={}, v2={}, lambda={}", v1, v2, lambda
            );
        }

        /// The full invariant: value at t2 <= value at t1 + sum of impulses between t1 and t2.
        #[test]
        fn bounded_growth_invariant(
            lambda in 0.01_f64..10.0,
            impulses in prop::collection::vec((0.0_f64..100.0, 0.001_f64..1.0), 2..30),
        ) {
            let mut integral = DecayingIntegral::new(lambda, 0.0);

            // Build monotonic timestamps
            let mut timestamps: Vec<f64> = Vec::with_capacity(impulses.len());
            let mut t = 0.0;
            for &(_, dt) in &impulses {
                t += dt;
                timestamps.push(t);
            }

            // Pick a split point: measure at midpoint, record rest, measure again
            let split = impulses.len() / 2;

            // Record first half
            for idx in 0..split {
                integral.record(impulses[idx].0, timestamps[idx]);
            }
            let t1 = timestamps[split - 1];
            let v1 = integral.current_value(t1);

            // Record second half
            let mut impulse_sum = 0.0;
            for idx in split..impulses.len() {
                integral.record(impulses[idx].0, timestamps[idx]);
                impulse_sum += impulses[idx].0;
            }
            let t2 = *timestamps.last().unwrap();
            let v2 = integral.current_value(t2);

            prop_assert!(
                v2 <= v1 + impulse_sum + 1e-10,
                "bounded growth violated: v1={}, v2={}, impulse_sum={}, lambda={}",
                v1, v2, impulse_sum, lambda
            );
        }
    }
}
