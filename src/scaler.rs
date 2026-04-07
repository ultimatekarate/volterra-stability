//! Scaler computation: dimensionless [0,1] pressure signals.
//!
//! Laboratory layer: pure math.

/// Linear scaler: sigma(x) = max(0, 1 - x/critical).
/// Returns 1.0 when x=0 (no pressure), 0.0 when x >= critical (saturated).
#[inline]
pub fn linear_scaler(value: f64, critical: f64) -> f64 {
    (1.0 - value / critical).max(0.0)
}

/// Derivative of the linear scaler with respect to value.
/// In the linear regime (value < critical): -1/critical.
/// In the saturated regime (value >= critical): 0.
#[inline]
pub fn dscaler(value: f64, critical: f64) -> f64 {
    if value < critical {
        -1.0 / critical
    } else {
        0.0
    }
}

/// Compute composite stress as a weighted sum of normalized integrals.
///
/// Each integral is divided by its critical threshold and clamped to [0, 1].
/// The weighted sum gives a single composite stress value.
pub fn composite_stress(values: &[f64], criticals: &[f64], weights: &[f64]) -> f64 {
    values
        .iter()
        .zip(criticals.iter())
        .zip(weights.iter())
        .map(|((v, c), w)| w * (v / c).min(1.0))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaler_at_zero() {
        assert!((linear_scaler(0.0, 10.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn scaler_at_half() {
        assert!((linear_scaler(5.0, 10.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn scaler_at_critical() {
        assert!((linear_scaler(10.0, 10.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn scaler_beyond_critical() {
        assert!((linear_scaler(15.0, 10.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn dscaler_linear_regime() {
        assert!((dscaler(5.0, 10.0) - (-0.1)).abs() < 1e-12);
    }

    #[test]
    fn dscaler_saturated_regime() {
        assert!((dscaler(10.0, 10.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn composite_stress_basic() {
        let stress = composite_stress(&[5.0, 10.0], &[10.0, 20.0], &[0.5, 0.5]);
        // 0.5 * (5/10) + 0.5 * (10/20) = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        assert!((stress - 0.5).abs() < 1e-12);
    }
}
