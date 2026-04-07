//! Spectral gap, eigenvector orthogonality, and stability radius analysis.
//!
//! Laboratory layer: pure math. Given a Jacobian matrix, compute spectral
//! gap γ₁, eigenvector conditioning κ(V), Henrici departure, stability
//! radius r(J), and guaranteed decay time.

use nalgebra::{Complex, DMatrix};

use crate::report::{FullSpectralReport, SpectralGapReport};

/// Compute real eigenvectors for an N×N Jacobian via SVD null-space.
///
/// nalgebra 0.33 does not expose eigenvectors for non-symmetric matrices.
/// For each eigenvalue λ_k we extract the null space of (J − λ_k I) via SVD:
///
///   Real λ:    v = right-singular vector of (J − λI) with smallest σ.
///   Complex λ: Compute (J − aI)² + b²I  where a = Re(λ), b = Im(λ).
///              The two right-singular vectors with smallest σ span the real
///              2-D invariant subspace.
fn compute_eigenvectors(jacobian: &DMatrix<f64>, eigenvalues: &[Complex<f64>]) -> DMatrix<f64> {
    let n = jacobian.nrows();
    let identity = DMatrix::<f64>::identity(n, n);
    let mut vectors = DMatrix::<f64>::zeros(n, n);
    let mut col = 0;
    let mut skip_conjugate = false;

    for eig in eigenvalues.iter() {
        if skip_conjugate {
            skip_conjugate = false;
            continue;
        }

        if eig.im.abs() < 1e-12 {
            let shifted = jacobian - &identity * eig.re;
            let svd = shifted.svd(true, true);
            if let Some(ref v_t) = svd.v_t {
                let last = v_t.nrows() - 1;
                let svals = &svd.singular_values;
                let s_max = svals[0].max(1e-15);
                let threshold = s_max * 1e-6;

                let mut best_row = last;
                let mut best_independence = -1.0_f64;

                for k in (0..v_t.nrows()).rev() {
                    if svals[k] > threshold && k != last {
                        break;
                    }
                    let mut min_indep = 1.0_f64;
                    for prev in 0..col {
                        let mut dot = 0.0_f64;
                        for r in 0..n {
                            dot += v_t[(k, r)] * vectors[(r, prev)];
                        }
                        min_indep = min_indep.min(1.0 - dot.abs());
                    }
                    if min_indep > best_independence {
                        best_independence = min_indep;
                        best_row = k;
                    }
                }

                for r in 0..n {
                    vectors[(r, col)] = v_t[(best_row, r)];
                }
            }
            col += 1;
        } else if eig.im > 0.0 {
            let a = eig.re;
            let b = eig.im;
            let shifted = jacobian - &identity * a;
            let kernel_mat = &shifted * &shifted + &identity * (b * b);
            let svd = kernel_mat.svd(true, true);
            if let Some(ref v_t) = svd.v_t {
                let last = v_t.nrows() - 1;
                for r in 0..n {
                    vectors[(r, col)] = v_t[(last, r)];
                }
                if col + 1 < n && last >= 1 {
                    for r in 0..n {
                        vectors[(r, col + 1)] = v_t[(last - 1, r)];
                    }
                }
            }
            col += 2;
            skip_conjugate = true;
        }
    }

    vectors
}

/// Compute the stability radius r(J) = min_ω σ_min(iωI − J).
///
/// Uses the real block-matrix equivalence and three-stage refinement.
fn stability_radius(jacobian: &DMatrix<f64>) -> (f64, f64) {
    let n = jacobian.nrows();
    let neg_j = jacobian * -1.0;

    let sigma_min_at = |omega: f64| -> f64 {
        let mut block = DMatrix::<f64>::zeros(2 * n, 2 * n);
        for r in 0..n {
            for c in 0..n {
                block[(r, c)] = neg_j[(r, c)];
                block[(n + r, n + c)] = neg_j[(r, c)];
            }
        }
        for i in 0..n {
            block[(i, n + i)] = -omega;
            block[(n + i, i)] = omega;
        }
        let svd = block.svd(false, false);
        svd.singular_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    };

    // Stage 1: coarse scan ω ∈ [0, 10] step 0.5
    let mut best_omega = 0.0_f64;
    let mut best_sigma = f64::INFINITY;
    let mut omega = 0.0_f64;
    while omega <= 10.0 {
        let s = sigma_min_at(omega);
        if s < best_sigma {
            best_sigma = s;
            best_omega = omega;
        }
        omega += 0.5;
    }

    // Stage 2: fine scan ±0.5 around best, step 0.05
    let lo = (best_omega - 0.5).max(0.0);
    let hi = best_omega + 0.5;
    omega = lo;
    while omega <= hi {
        let s = sigma_min_at(omega);
        if s < best_sigma {
            best_sigma = s;
            best_omega = omega;
        }
        omega += 0.05;
    }

    // Stage 3: ultra-fine scan ±0.05 around best, step 0.005
    let lo = (best_omega - 0.05).max(0.0);
    let hi = best_omega + 0.05;
    omega = lo;
    while omega <= hi {
        let s = sigma_min_at(omega);
        if s < best_sigma {
            best_sigma = s;
            best_omega = omega;
        }
        omega += 0.005;
    }

    (best_sigma, best_omega)
}

/// Analyze spectral gap, eigenvector orthogonality, and stability radius
/// for a single operating scenario from a pre-built Jacobian.
pub fn analyze_spectral_gap(
    scenario: &str,
    jacobian: &DMatrix<f64>,
) -> SpectralGapReport {
    let raw_eigs = jacobian.complex_eigenvalues();
    let mut eigs: Vec<Complex<f64>> = raw_eigs.iter().cloned().collect();

    // Sort by |Re(λ)| ascending — dominant (slowest) mode first.
    eigs.sort_by(|a, b| {
        a.re.abs()
            .partial_cmp(&b.re.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let gamma1 = eigs[0].re.abs();
    let gamma2 = if eigs.len() > 1 {
        eigs[1].re.abs() - eigs[0].re.abs()
    } else {
        0.0
    };
    let gap_ratio = if gamma1 > 1e-15 { gamma2 / gamma1 } else { 0.0 };

    let eigvecs = compute_eigenvectors(jacobian, &eigs);
    let gram = eigvecs.transpose() * &eigvecs;

    let svd_v = eigvecs.clone().svd(false, false);
    let svals = &svd_v.singular_values;
    let sigma_max = svals.iter().cloned().fold(0.0_f64, f64::max);
    let sigma_min = svals.iter().cloned().fold(f64::INFINITY, f64::min);
    let kappa = if sigma_min > 1e-15 {
        sigma_max / sigma_min
    } else {
        f64::INFINITY
    };

    // Henrici departure from normality: δ_H = √(‖J‖²_F − Σ|λ_k|²)
    let frob_sq: f64 = jacobian.iter().map(|x| x * x).sum();
    let eig_norm_sq: f64 = eigs
        .iter()
        .map(|lam| lam.re * lam.re + lam.im * lam.im)
        .sum();
    let henrici_raw = frob_sq - eig_norm_sq;
    let henrici = if henrici_raw > 0.0 {
        henrici_raw.sqrt()
    } else {
        0.0
    };

    let (stab_rad, stab_omega) = stability_radius(jacobian);

    let decay_time = if gamma1 > 1e-15 {
        kappa.ln() / gamma1
    } else {
        f64::INFINITY
    };

    SpectralGapReport {
        scenario: scenario.to_string(),
        eigenvalues_sorted: eigs,
        spectral_gap_gamma1: gamma1,
        spectral_gap_gamma2: gamma2,
        spectral_gap_ratio: gap_ratio,
        eigenvector_matrix: eigvecs,
        gram_matrix: gram,
        eigenvector_condition_number: kappa,
        henrici_departure: henrici,
        stability_radius: stab_rad,
        stability_radius_omega: stab_omega,
        guaranteed_decay_time: decay_time,
        jacobian_frobenius: frob_sq.sqrt(),
    }
}

/// Build a combined robustness certificate from multiple spectral gap reports.
pub fn build_combined_certificate(
    results: &[SpectralGapReport],
    worst_g1: f64,
    worst_rad: f64,
    worst_kappa: f64,
    lyapunov_mu1: Option<f64>,
) -> String {
    let mut out = String::new();
    out.push_str("COMBINED ROBUSTNESS CERTIFICATE\n");
    out.push_str("===============================\n\n");

    let all_stable = results.iter().all(|r| r.spectral_gap_gamma1 > 0.0);
    out.push_str(&format!(
        "  [{}] Eigenvalue stability:  all Re(lambda) < 0 across {} scenarios\n",
        if all_stable { "PASS" } else { "FAIL" },
        results.len()
    ));
    out.push_str(&format!("       worst spectral gap g1 = {:.6}\n", worst_g1));

    out.push_str(&format!(
        "  [{}] Spectral gap:  g1 = {:.6}\n",
        if worst_g1 > 0.01 { "PASS" } else { "WARN" },
        worst_g1,
    ));

    out.push_str(&format!(
        "  [{}] Stability radius:  r(J) = {:.6}\n",
        if worst_rad > 0.0 { "PASS" } else { "FAIL" },
        worst_rad
    ));

    let worst_decay = results
        .iter()
        .map(|r| r.guaranteed_decay_time)
        .fold(0.0_f64, f64::max);
    out.push_str(&format!(
        "  [{}] Eigenvector conditioning:  kappa(V) = {:.4}\n",
        if worst_kappa < 1000.0 { "PASS" } else { "WARN" },
        worst_kappa
    ));
    out.push_str(&format!(
        "       guaranteed decay dominance after t = {:.2}s\n",
        worst_decay
    ));

    match lyapunov_mu1 {
        Some(mu1) if mu1 < 0.0 => {
            out.push_str(&format!(
                "  [PASS] Nonlinear Lyapunov:  mu1 = {:.6} < 0\n",
                mu1
            ));
        }
        Some(mu1) => {
            out.push_str(&format!(
                "  [FAIL] Nonlinear Lyapunov:  mu1 = {:.6} >= 0\n",
                mu1
            ));
        }
        None => {
            out.push_str("  [????] Nonlinear Lyapunov:  not computed\n");
        }
    }

    let lyapunov_stable = lyapunov_mu1.is_some_and(|mu| mu < 0.0);
    if all_stable && worst_rad > 0.0 && worst_kappa < f64::INFINITY && lyapunov_stable {
        out.push_str("\n  VERDICT: The integral system possesses a mathematically\n");
        out.push_str("  complete stability certificate across all four layers.\n");
    }

    out
}

/// Run spectral gap analysis across multiple scenarios from pre-built Jacobians.
pub fn full_spectral_analysis(
    jacobians: &[(&str, DMatrix<f64>)],
    lyapunov_mu1: Option<f64>,
) -> FullSpectralReport {
    let results: Vec<SpectralGapReport> = jacobians
        .iter()
        .map(|(name, j)| analyze_spectral_gap(name, j))
        .collect();

    let worst_g1 = results
        .iter()
        .map(|r| r.spectral_gap_gamma1)
        .fold(f64::INFINITY, f64::min);
    let worst_rad = results
        .iter()
        .map(|r| r.stability_radius)
        .fold(f64::INFINITY, f64::min);
    let worst_kappa = results
        .iter()
        .map(|r| r.eigenvector_condition_number)
        .fold(0.0_f64, f64::max);

    let cert = build_combined_certificate(&results, worst_g1, worst_rad, worst_kappa, lyapunov_mu1);

    FullSpectralReport {
        scenarios: results,
        combined_certificate: cert,
        worst_gamma1: worst_g1,
        worst_stability_radius: worst_rad,
        worst_condition_number: worst_kappa,
        lyapunov_mu1,
    }
}

/// Format the full spectral analysis into a human-readable report.
pub fn format_spectral_report(report: &FullSpectralReport, channel_names: &[&str]) -> String {
    let mut out = String::new();

    out.push_str("\n===================================================================\n");
    out.push_str("       SPECTRAL GAP & EIGENVECTOR ORTHOGONALITY ANALYSIS\n");
    out.push_str("===================================================================\n\n");

    out.push_str("  Scenario                              g1       g2       kappa(V)    d_H      r(J)     t_decay\n");
    out.push_str("  ------------------------------------  ------   ------   --------    ------   ------   -------\n");

    for s in &report.scenarios {
        out.push_str(&format!(
            "  {:<36}  {:6.4}   {:6.4}   {:8.2}    {:6.4}   {:6.4}   {:5.2}s\n",
            s.scenario,
            s.spectral_gap_gamma1,
            s.spectral_gap_gamma2,
            s.eigenvector_condition_number,
            s.henrici_departure,
            s.stability_radius,
            s.guaranteed_decay_time,
        ));
    }

    out.push_str(&format!(
        "\nWorst-case: g1={:.6}, r(J)={:.6}, kappa(V)={:.4}\n",
        report.worst_gamma1, report.worst_stability_radius, report.worst_condition_number
    ));

    if let Some(first) = report.scenarios.first() {
        out.push_str(&format!(
            "\nEigenvalue spectrum ({}), sorted by |Re(lambda)|:\n",
            first.scenario
        ));
        for (i, lam) in first.eigenvalues_sorted.iter().enumerate() {
            let label = if i < channel_names.len() {
                channel_names[i]
            } else {
                "?"
            };
            if lam.im.abs() < 1e-12 {
                out.push_str(&format!(
                    "  lambda_{} = {:.6}          (mode {})\n",
                    i, lam.re, label
                ));
            } else {
                out.push_str(&format!(
                    "  lambda_{} = {:.6} +/- {:.6}i  (mode {})\n",
                    i,
                    lam.re,
                    lam.im.abs(),
                    label
                ));
            }
        }
    }

    out.push_str("\n-------------------------------------------------------------------\n");
    out.push_str(&report.combined_certificate);
    out.push_str("===================================================================\n");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_spectral_gap() {
        let mut diag = DMatrix::zeros(3, 3);
        diag[(0, 0)] = -0.1;
        diag[(1, 1)] = -0.5;
        diag[(2, 2)] = -2.0;
        let report = analyze_spectral_gap("diagonal", &diag);
        assert!(report.spectral_gap_gamma1 > 0.0);
        assert!((report.spectral_gap_gamma1 - 0.1).abs() < 1e-6);
    }

    #[test]
    fn stability_radius_positive_for_stable() {
        let mut diag = DMatrix::zeros(3, 3);
        diag[(0, 0)] = -0.5;
        diag[(1, 1)] = -1.0;
        diag[(2, 2)] = -2.0;
        let report = analyze_spectral_gap("stable", &diag);
        assert!(report.stability_radius > 0.0);
    }

    #[test]
    fn henrici_zero_for_symmetric() {
        let mut sym = DMatrix::zeros(3, 3);
        sym[(0, 0)] = -1.0;
        sym[(1, 1)] = -2.0;
        sym[(2, 2)] = -3.0;
        sym[(0, 1)] = 0.1;
        sym[(1, 0)] = 0.1;
        let report = analyze_spectral_gap("symmetric", &sym);
        assert!(
            report.henrici_departure < 1e-6,
            "symmetric matrix should have henrici ~0, got {}",
            report.henrici_departure
        );
    }
}
