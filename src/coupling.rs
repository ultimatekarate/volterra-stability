//! CouplingModel trait: domain-specific coupling structure.
//!
//! Laboratory layer: defines the interface that domain-specific code
//! must implement to use the stability analysis machinery.

use nalgebra::DMatrix;

use crate::config::{ImpulseRates, OperatingPoint, SystemConfig};

/// Domain-specific coupling model.
///
/// The coupling model defines how channels interact — which channels affect
/// which, and with what coefficients. This is the domain knowledge that
/// distinguishes one system from another.
///
/// The stability analysis machinery (eigenvalues, spectral gap, Dyson series)
/// operates on the Jacobian matrix produced by this trait.
pub trait CouplingModel {
    /// System configuration (channels, decay rates, thresholds).
    fn config(&self) -> &SystemConfig;

    /// Build the Jacobian at a given operating point and traffic regime.
    ///
    /// J[i,j] = df_i/dI_j - lambda_i * delta_{ij}
    /// where f_i is the effective impulse rate for channel i.
    fn build_jacobian(&self, rates: &ImpulseRates, op: &OperatingPoint) -> DMatrix<f64>;

    /// Normalization scales for contractivity check.
    /// Each integral divided by its scale yields a dimensionless [0,1] quantity.
    fn normalization_scales(&self) -> Vec<f64>;

    /// Lyapunov matrix P for contractivity proof, if available.
    /// V(x) = xt P x must decrease along trajectories for contractivity.
    fn lyapunov_matrix(&self) -> Option<DMatrix<f64>>;
}
