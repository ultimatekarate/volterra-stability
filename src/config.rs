//! Configuration types for Volterra integral stability analysis.
//!
//! Dictionary layer: inert nouns. No IO, no computation beyond accessors.

use serde::{Deserialize, Serialize};

/// Configuration for a single pressure channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Human-readable name (e.g., "metabolic", "bandwidth").
    pub name: String,
    /// Decay rate λ. The kernel is K(t,s) = λ·e^{-λ(t-s)}.
    /// Related to half-life by: λ = ln(2) / half_life.
    pub lambda: f64,
    /// Critical threshold. The scaler σ(x) = max(0, 1 - x/critical).
    pub critical: f64,
}

/// Full system configuration: N channels with their decay rates and thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Per-channel configuration.
    pub channels: Vec<ChannelConfig>,
    /// Composite stress weights. Length must equal channels.len().
    /// Sum should be 1.0.
    pub stress_weights: Vec<f64>,
}

impl SystemConfig {
    /// Number of pressure channels.
    pub fn dim(&self) -> usize {
        self.channels.len()
    }

    /// Extract decay rates as a slice-compatible vector.
    pub fn lambdas(&self) -> Vec<f64> {
        self.channels.iter().map(|c| c.lambda).collect()
    }

    /// Extract critical thresholds as a vector.
    pub fn criticals(&self) -> Vec<f64> {
        self.channels.iter().map(|c| c.critical).collect()
    }

    /// Channel names for report formatting.
    pub fn names(&self) -> Vec<&str> {
        self.channels.iter().map(|c| c.name.as_str()).collect()
    }
}

/// Operating point: current integral magnitudes at which the system is linearized.
#[derive(Debug, Clone)]
pub struct OperatingPoint {
    pub vals: Vec<f64>,
}

impl OperatingPoint {
    /// All integrals at zero (system at rest).
    pub fn idle(dim: usize) -> Self {
        Self {
            vals: vec![0.0; dim],
        }
    }

    /// Each integral at a given fraction of its critical threshold.
    pub fn at_fraction(config: &SystemConfig, fraction: f64) -> Self {
        Self {
            vals: config
                .channels
                .iter()
                .map(|c| c.critical * fraction)
                .collect(),
        }
    }
}

/// Base impulse rates: the unthrottled rate at which events feed each integral.
/// Units are impulse/second.
#[derive(Debug, Clone)]
pub struct ImpulseRates {
    pub rates: Vec<f64>,
}

impl ImpulseRates {
    /// Construct from a slice. Length must match the system dimension.
    pub fn from_slice(rates: &[f64]) -> Self {
        Self {
            rates: rates.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_config_accessors() {
        let cfg = SystemConfig {
            channels: vec![
                ChannelConfig {
                    name: "a".into(),
                    lambda: 0.5,
                    critical: 10.0,
                },
                ChannelConfig {
                    name: "b".into(),
                    lambda: 1.0,
                    critical: 20.0,
                },
            ],
            stress_weights: vec![0.5, 0.5],
        };
        assert_eq!(cfg.dim(), 2);
        assert_eq!(cfg.lambdas(), vec![0.5, 1.0]);
        assert_eq!(cfg.criticals(), vec![10.0, 20.0]);
    }

    #[test]
    fn operating_point_at_fraction() {
        let cfg = SystemConfig {
            channels: vec![
                ChannelConfig {
                    name: "x".into(),
                    lambda: 1.0,
                    critical: 100.0,
                },
                ChannelConfig {
                    name: "y".into(),
                    lambda: 1.0,
                    critical: 50.0,
                },
            ],
            stress_weights: vec![0.5, 0.5],
        };
        let op = OperatingPoint::at_fraction(&cfg, 0.5);
        assert_eq!(op.vals, vec![50.0, 25.0]);
    }
}
