//! Eigenvalue stability analysis.
//!
//! Laboratory layer: pure math. Given a Jacobian matrix, compute eigenvalues,
//! check stability, check contractivity.

use nalgebra::{Complex, DMatrix};

use crate::report::StabilityReport;

/// Compute eigenvalues and stability properties from a Jacobian matrix.
///
/// The `norm_scales` and `lyapunov_p` parameters enable the contractivity check.
/// If either is `None`, contractivity fields default to false/0.
pub fn analyze_stability(
    scenario: &str,
    jacobian: &DMatrix<f64>,
    norm_scales: Option<&[f64]>,
    lyapunov_p: Option<&DMatrix<f64>>,
) -> StabilityReport {
    let dim = jacobian.nrows();
    let eigenvalues = jacobian.complex_eigenvalues();
    let eigs: Vec<Complex<f64>> = eigenvalues.iter().cloned().collect();

    let (dominant_idx, max_re) = eigs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.re.partial_cmp(&b.re).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, v.re))
        .unwrap_or((0, f64::NEG_INFINITY));

    // Symmetric part analysis: Jsym = (J + Jt)/2
    let jsym = (jacobian + jacobian.transpose()) * 0.5;
    let jsym_eigs = jsym.symmetric_eigenvalues();
    let jsym_max = jsym_eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Contractivity analysis
    let (is_contractive, contractivity_margin) =
        match (norm_scales, lyapunov_p) {
            (Some(scales), Some(p)) => check_contractivity(jacobian, scales, p, dim),
            _ => (false, 0.0),
        };

    StabilityReport {
        scenario: scenario.to_string(),
        eigenvalues: eigs,
        max_real_part: max_re,
        spectral_abscissa: max_re,
        is_stable: max_re < 0.0,
        dominant_mode_idx: dominant_idx,
        jacobian: jacobian.clone(),
        jsym_negative_definite: jsym_max < 0.0,
        jsym_max_eigenvalue: jsym_max,
        is_contractive,
        contractivity_margin,
    }
}

/// Check contractivity under a Lyapunov matrix P.
///
/// Normalizes the Jacobian: J_n[i,j] = J[i,j] * scales[j] / scales[i],
/// then computes Q = P*J_n + J_nt*P and checks all eigenvalues < 0.
fn check_contractivity(
    jacobian: &DMatrix<f64>,
    scales: &[f64],
    p: &DMatrix<f64>,
    dim: usize,
) -> (bool, f64) {
    let mut jn = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            jn[(i, j)] = jacobian[(i, j)] * scales[j] / scales[i];
        }
    }

    let q = p * &jn + jn.transpose() * p;
    let q_sym = (&q + q.transpose()) * 0.5;

    let q_eigs = q_sym.symmetric_eigenvalues();
    let q_max = q_eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    (q_max < 0.0, -q_max)
}

/// Format a stability report as a human-readable string.
pub fn format_report(reports: &[StabilityReport], channel_names: &[&str]) -> String {
    let dim = channel_names.len();
    let mut out = String::new();
    out.push_str("EIGENVALUE STABILITY REPORT\n");
    out.push_str("==========================\n\n");

    for report in reports {
        out.push_str(&format!("--- {} ---\n", report.scenario));

        // Print Jacobian matrix
        out.push_str("\n  Jacobian (coupling matrix):\n");
        out.push_str("         ");
        for name in channel_names {
            out.push_str(&format!("{:>10}", name));
        }
        out.push('\n');
        for i in 0..dim {
            out.push_str(&format!("    {}  [", channel_names[i]));
            for k in 0..dim {
                let v = report.jacobian[(i, k)];
                if v.abs() < 1e-12 {
                    out.push_str("         .");
                } else {
                    out.push_str(&format!("{:>10.4}", v));
                }
            }
            out.push_str(" ]\n");
        }

        // Print eigenvalues
        out.push_str("\n  Eigenvalues:\n");
        for (i, eig) in report.eigenvalues.iter().enumerate() {
            let label = if i < channel_names.len() {
                channel_names[i]
            } else {
                "?"
            };
            let marker = if i == report.dominant_mode_idx {
                " << dominant"
            } else {
                ""
            };
            if eig.im.abs() < 1e-10 {
                out.push_str(&format!(
                    "    lambda_{} = {:>10.6} (real){}\n",
                    label, eig.re, marker
                ));
            } else {
                out.push_str(&format!(
                    "    lambda_{} = {:>10.6} {:+.6}i{}\n",
                    label, eig.re, eig.im, marker
                ));
            }
        }

        out.push_str(&format!(
            "\n  Spectral abscissa:  {:.6}\n",
            report.spectral_abscissa
        ));
        out.push_str(&format!(
            "  Jsym max eigenvalue: {:.6}  {}\n",
            report.jsym_max_eigenvalue,
            if report.jsym_negative_definite {
                "(negative definite -> energy-dissipative)"
            } else {
                "(NOT negative definite -- transient growth possible)"
            }
        ));
        out.push_str(&format!(
            "  Contractivity:      {} (margin: {:.6})\n",
            if report.is_contractive {
                "CONTRACTIVE"
            } else {
                "not contractive"
            },
            report.contractivity_margin,
        ));
        out.push_str(&format!(
            "  Stability verdict:  {}\n\n",
            if report.is_stable {
                "STABLE (all Re(lambda) < 0)"
            } else {
                "UNSTABLE -- positive eigenvalue detected!"
            }
        ));
    }

    // Summary
    let all_stable = reports.iter().all(|r| r.is_stable);
    let all_contractive = reports.iter().all(|r| r.is_contractive);
    let min_margin = reports
        .iter()
        .map(|r| r.contractivity_margin)
        .fold(f64::INFINITY, f64::min);
    out.push_str("===========================================================\n");
    out.push_str(&format!(
        "  Overall: {}\n",
        if all_stable {
            "ALL SCENARIOS STABLE"
        } else {
            "INSTABILITY DETECTED"
        }
    ));
    if all_contractive {
        out.push_str(&format!(
            "  Contractivity: ALL SCENARIOS CONTRACTIVE (min margin: {:.6})\n",
            min_margin
        ));
    }
    out.push_str("===========================================================\n");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_stable_system() {
        let j = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![-1.0, -2.0, -0.5]));
        let report = analyze_stability("diagonal", &j, None, None);
        assert!(report.is_stable);
        assert!(report.jsym_negative_definite);
    }

    #[test]
    fn diagonal_unstable_system() {
        let j = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![-1.0, 0.5, -0.5]));
        let report = analyze_stability("unstable", &j, None, None);
        assert!(!report.is_stable);
    }

    #[test]
    fn identity_negative_is_stable() {
        let j = DMatrix::identity(4, 4) * -1.0;
        let report = analyze_stability("neg-identity", &j, None, None);
        assert!(report.is_stable);
        assert!((report.max_real_part - (-1.0)).abs() < 1e-10);
    }
}
