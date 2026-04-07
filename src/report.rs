//! Report types for stability analysis results.
//!
//! Dictionary layer: inert nouns. Structs that hold analysis results.

use nalgebra::{Complex, DMatrix};

/// Results of the eigenvalue stability analysis for a single scenario.
#[derive(Debug, Clone)]
pub struct StabilityReport {
    /// The label for this analysis scenario.
    pub scenario: String,
    /// Complex eigenvalues of the Jacobian.
    pub eigenvalues: Vec<Complex<f64>>,
    /// Maximum real part across all eigenvalues (must be < 0 for stability).
    pub max_real_part: f64,
    /// Spectral abscissa (same as max_real_part, standard control-theory term).
    pub spectral_abscissa: f64,
    /// True if and only if all eigenvalues have strictly negative real parts.
    pub is_stable: bool,
    /// Index of the dominant (least-damped) eigenvalue.
    pub dominant_mode_idx: usize,
    /// The Jacobian matrix itself, for inspection.
    pub jacobian: DMatrix<f64>,
    /// Whether the symmetric part (J+Jt)/2 is negative definite.
    pub jsym_negative_definite: bool,
    /// Maximum eigenvalue of the symmetric part (J+Jt)/2.
    pub jsym_max_eigenvalue: f64,
    /// Whether Q = PJ_n + J_ntP is negative definite under a Lyapunov matrix P.
    pub is_contractive: bool,
    /// Contractivity margin: -lambda_max(Q). Positive means contractive.
    pub contractivity_margin: f64,
}

/// Per-scenario spectral gap and eigenvector analysis.
#[derive(Debug, Clone)]
pub struct SpectralGapReport {
    /// Label for this analysis scenario.
    pub scenario: String,
    /// Eigenvalues sorted by |Re(lambda)| ascending (dominant / slowest mode first).
    pub eigenvalues_sorted: Vec<Complex<f64>>,
    /// Absolute spectral gap: |Re(lambda_dominant)|.
    pub spectral_gap_gamma1: f64,
    /// Modal gap: |Re(lambda_2)| - |Re(lambda_1)|.
    pub spectral_gap_gamma2: f64,
    /// Dimensionless spectral gap ratio gamma2/gamma1.
    pub spectral_gap_ratio: f64,
    /// Real eigenvector matrix V.
    pub eigenvector_matrix: DMatrix<f64>,
    /// Gram matrix Vt V. Identity iff eigenvectors are orthonormal.
    pub gram_matrix: DMatrix<f64>,
    /// Condition number kappa(V) = sigma_max(V)/sigma_min(V).
    pub eigenvector_condition_number: f64,
    /// Henrici departure from normality.
    pub henrici_departure: f64,
    /// Stability radius r(J) = min_omega sigma_min(i*omega*I - J).
    pub stability_radius: f64,
    /// Frequency omega* where the stability radius minimum is attained.
    pub stability_radius_omega: f64,
    /// Time after which exponential decay dominates transient growth.
    pub guaranteed_decay_time: f64,
    /// Frobenius norm of the Jacobian.
    pub jacobian_frobenius: f64,
}

/// Aggregate report across multiple operating scenarios.
#[derive(Debug, Clone)]
pub struct FullSpectralReport {
    /// Per-scenario results.
    pub scenarios: Vec<SpectralGapReport>,
    /// Human-readable combined robustness certificate.
    pub combined_certificate: String,
    /// Worst (smallest) gamma1 across all scenarios.
    pub worst_gamma1: f64,
    /// Worst (smallest) stability radius across all scenarios.
    pub worst_stability_radius: f64,
    /// Worst (largest) eigenvector condition number across all scenarios.
    pub worst_condition_number: f64,
    /// Maximal Lyapunov exponent from nonlinear partition analysis, if computed.
    pub lyapunov_mu1: Option<f64>,
}

/// Dyson series correction terms for the time-evolution operator.
#[derive(Debug, Clone)]
pub struct DysonTerms {
    /// Zeroth order: G0(T) -- the free propagator at the final time.
    pub zeroth: DMatrix<f64>,
    /// First-order correction.
    pub first: DMatrix<f64>,
    /// Second-order correction.
    pub second: DMatrix<f64>,
    /// Convergence ratio ||U2||_F / ||U1||_F. Must be < 1 for series convergence.
    pub convergence_ratio: f64,
}

/// A time-localized threat perturbation to the integral system.
#[derive(Debug, Clone)]
pub struct ThreatProfile {
    pub name: String,
    /// Onset time in seconds from t=0.
    pub onset: f64,
    /// Duration of the threat in seconds.
    pub duration: f64,
    /// Direct impulse injection rate (N-vector). Active during [onset, onset+duration].
    pub forcing: Vec<f64>,
    /// Optional perturbation to the Jacobian. When Some, the system matrix
    /// becomes J + V during the threat window.
    pub coupling_delta: Option<DMatrix<f64>>,
}

/// Time series of integral states.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
}

/// Impulse response metrics for a threat scenario.
#[derive(Debug, Clone)]
pub struct ImpulseResponseReport {
    pub scenario: String,
    /// Peak absolute value of each integral during/after the threat.
    pub peak_values: Vec<f64>,
    /// Time at which each integral reaches its peak.
    pub peak_times: Vec<f64>,
    /// Time for each integral to return to 10% of its peak.
    pub recovery_times: Vec<f64>,
    /// Full time series.
    pub time_series: TimeSeries,
}

/// Cascade analysis: compare sequential threats against each individually.
#[derive(Debug, Clone)]
pub struct CascadeReport {
    pub threat_a_alone: ImpulseResponseReport,
    pub threat_b_alone: ImpulseResponseReport,
    pub cascade: ImpulseResponseReport,
    /// peak(A->B) / max(peak(A), peak(B)) per integral.
    /// Values > 1.0 indicate dangerous compounding.
    pub compounding_factors: Vec<f64>,
}

/// Full Dyson analysis results.
#[derive(Debug, Clone)]
pub struct DysonAnalysisReport {
    pub threat_responses: Vec<ImpulseResponseReport>,
    pub cascade: CascadeReport,
    pub dyson_terms: DysonTerms,
    pub convergence_radius: f64,
}

/// Statistics from the adaptive integrator.
#[derive(Debug, Clone, Default)]
pub struct StepStatistics {
    /// Number of accepted steps.
    pub n_accepted: usize,
    /// Number of rejected steps.
    pub n_rejected: usize,
    /// Smallest dt actually used in an accepted step.
    pub dt_min_used: f64,
    /// Largest dt actually used in an accepted step.
    pub dt_max_used: f64,
    /// History of accepted step sizes.
    pub dt_history: Vec<f64>,
}

impl StepStatistics {
    pub fn new() -> Self {
        Self {
            dt_min_used: f64::INFINITY,
            dt_max_used: 0.0,
            ..Default::default()
        }
    }

    pub fn record_accepted(&mut self, dt: f64) {
        self.n_accepted += 1;
        self.dt_min_used = self.dt_min_used.min(dt);
        self.dt_max_used = self.dt_max_used.max(dt);
        self.dt_history.push(dt);
    }
}

/// Configuration for the adaptive Dormand-Prince RK4(5) integrator.
#[derive(Debug, Clone)]
pub struct AdaptiveStepConfig {
    /// Absolute tolerance for each component.
    pub atol: f64,
    /// Relative tolerance for each component.
    pub rtol: f64,
    /// Minimum allowed step size (seconds).
    pub dt_min: f64,
    /// Maximum allowed step size (seconds).
    pub dt_max: f64,
    /// Safety factor for step size controller (< 1.0).
    pub safety: f64,
    /// Maximum growth factor per step.
    pub max_growth: f64,
    /// Minimum shrink factor per step.
    pub min_shrink: f64,
}

impl Default for AdaptiveStepConfig {
    fn default() -> Self {
        Self {
            atol: 1e-8,
            rtol: 1e-6,
            dt_min: 1e-6,
            dt_max: 0.5,
            safety: 0.9,
            max_growth: 5.0,
            min_shrink: 0.2,
        }
    }
}
