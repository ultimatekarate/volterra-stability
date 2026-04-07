//! Jacobian construction via the CouplingModel trait.
//!
//! Laboratory layer: delegates domain-specific coupling to the trait implementor.
//! This module provides the generic analysis entry point that builds a Jacobian
//! from any CouplingModel and feeds it to the eigenvalue/spectral machinery.

use nalgebra::DMatrix;

use crate::config::{ImpulseRates, OperatingPoint};
use crate::coupling::CouplingModel;
use crate::eigenvalues::analyze_stability;
use crate::report::StabilityReport;

/// Build the Jacobian and run stability analysis for a single scenario.
pub fn analyze_scenario(
    scenario: &str,
    model: &dyn CouplingModel,
    rates: &ImpulseRates,
    op: &OperatingPoint,
) -> StabilityReport {
    let jacobian = model.build_jacobian(rates, op);
    let scales = model.normalization_scales();
    let lyapunov_p = model.lyapunov_matrix();

    analyze_stability(
        scenario,
        &jacobian,
        Some(&scales),
        lyapunov_p.as_ref(),
    )
}

/// Build the Jacobian and run stability analysis across multiple scenarios.
pub fn analyze_scenarios(
    model: &dyn CouplingModel,
    scenarios: &[(&str, ImpulseRates, OperatingPoint)],
) -> Vec<StabilityReport> {
    scenarios
        .iter()
        .map(|(name, rates, op)| analyze_scenario(name, model, rates, op))
        .collect()
}

/// Build a raw Jacobian matrix from the coupling model.
///
/// Convenience wrapper for when you need the matrix directly
/// (e.g., for Dyson series or spectral analysis).
pub fn build_jacobian(
    model: &dyn CouplingModel,
    rates: &ImpulseRates,
    op: &OperatingPoint,
) -> DMatrix<f64> {
    model.build_jacobian(rates, op)
}
