//! DecayingIntegral: the core Volterra second-kind integral primitive.
//!
//! Laboratory layer: pure math. Caller provides timestamps.

/// A single Volterra second-kind integral with caller-provided timestamps.
///
/// Each integral decays independently: I(t+dt) = impulse + I(t) * exp(-lambda * dt)
/// where dt is measured from THIS integral's last update, not a shared clock.
///
/// The caller owns the time source. Pass seconds since an arbitrary epoch.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
}
