//! Dyson series transient threat propagation analysis.
//!
//! Laboratory layer: pure math. Gauss-Legendre quadrature for the Dyson
//! series integrals, exponential integrator for time evolution.

use nalgebra::{DMatrix, DVector};

use crate::pade::mat_exp;
use crate::report::{
    CascadeReport, DysonAnalysisReport, DysonTerms, ImpulseResponseReport, ThreatProfile,
    TimeSeries,
};

// 16-point Gauss-Legendre nodes on [-1, 1].
const GL16_NODES: [f64; 16] = [
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.7554044083550030,
    -0.6178762444026438,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.0950125098376374,
    0.0950125098376374,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.7554044083550030,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];

// 16-point Gauss-Legendre weights on [-1, 1].
const GL16_WEIGHTS: [f64; 16] = [
    0.0271524594117541,
    0.0622535239386479,
    0.0951585116824928,
    0.1246289712555339,
    0.1495959888165767,
    0.1691565193950025,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.1691565193950025,
    0.1495959888165767,
    0.1246289712555339,
    0.0951585116824928,
    0.0622535239386479,
    0.0271524594117541,
];

/// Rescale GL nodes/weights from [-1,1] to [a,b].
fn gl16_rescale(a: f64, b: f64) -> ([f64; 16], [f64; 16]) {
    let half_len = (b - a) * 0.5;
    let mid = (a + b) * 0.5;
    let mut nodes = [0.0; 16];
    let mut weights = [0.0; 16];
    for i in 0..16 {
        nodes[i] = half_len * GL16_NODES[i] + mid;
        weights[i] = half_len * GL16_WEIGHTS[i];
    }
    (nodes, weights)
}

/// Evolve the linearized integral system under one or more threat profiles.
///
/// Uses the exponential integrator with LU solves:
///   x(t+dt) = G·x(t) + φ
///   where J·φ = (G − I)·u(t), solved via pre-computed LU factorization.
pub fn evolve(
    j: &DMatrix<f64>,
    threats: &[ThreatProfile],
    x0: &[f64],
    t_final: f64,
    dt: f64,
) -> TimeSeries {
    let dim = j.nrows();
    let n = (t_final / dt).ceil() as usize;
    let id = DMatrix::identity(dim, dim);

    let g0 = mat_exp(&(j * dt));
    let lu_j = j.clone().lu();
    let g0_minus_i = &g0 - &id;

    // Pre-compute propagators for coupling deltas
    let mut coupling_cache: Vec<(
        DMatrix<f64>,
        nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn>,
    )> = Vec::new();
    for threat in threats {
        if let Some(ref v) = threat.coupling_delta {
            let j_plus_v = j + v;
            let g_v = mat_exp(&(&j_plus_v * dt));
            let lu_jv = j_plus_v.lu();
            coupling_cache.push((g_v, lu_jv));
        }
    }

    let mut times = Vec::with_capacity(n + 1);
    let mut states: Vec<Vec<f64>> = Vec::with_capacity(n + 1);

    let mut x = DVector::from_column_slice(x0);
    let mut t = 0.0;

    times.push(t);
    states.push(x.as_slice().to_vec());

    for _ in 0..n {
        let mut u_total = DVector::zeros(dim);
        let mut active_coupling: Option<usize> = None;

        for (ti, threat) in threats.iter().enumerate() {
            if t >= threat.onset && t < threat.onset + threat.duration {
                let forcing = DVector::from_column_slice(&threat.forcing);
                u_total += &forcing;
                if threat.coupling_delta.is_some() {
                    let cache_idx = threats[..ti]
                        .iter()
                        .filter(|t| t.coupling_delta.is_some())
                        .count();
                    active_coupling = Some(cache_idx);
                }
            }
        }

        let (g, lu, g_ref) = if let Some(idx) = active_coupling {
            (
                &coupling_cache[idx].0,
                &coupling_cache[idx].1,
                &coupling_cache[idx].0 - &id,
            )
        } else {
            (&g0, &lu_j, g0_minus_i.clone())
        };

        x = g * &x;
        if u_total.norm() > 1e-15 {
            let rhs = &g_ref * &u_total;
            if let Some(phi) = lu.solve(&rhs) {
                x += phi;
            }
        }

        t += dt;
        times.push(t);
        states.push(x.as_slice().to_vec());
    }

    TimeSeries { times, states }
}

/// Compute the first two Dyson series correction terms using 16-point
/// Gauss-Legendre quadrature.
pub fn compute_dyson_terms(
    j: &DMatrix<f64>,
    v: &DMatrix<f64>,
    t_onset: f64,
    t_end: f64,
) -> DysonTerms {
    let dim = j.nrows();
    let t_total = t_end;

    let zeroth = mat_exp(&(j * t_total));

    let (outer_nodes, outer_weights) = gl16_rescale(t_onset, t_end);

    // First-order: U¹ = Σᵢ wᵢ · G₀(T−tᵢ) · V · G₀(tᵢ)
    let mut first = DMatrix::zeros(dim, dim);
    for i in 0..16 {
        let ti = outer_nodes[i];
        let g_left = mat_exp(&(j * (t_total - ti)));
        let g_right = mat_exp(&(j * ti));
        first += (g_left * v * g_right) * outer_weights[i];
    }

    // Second-order: nested GL quadrature
    let mut second = DMatrix::zeros(dim, dim);
    for i in 0..16 {
        let t2 = outer_nodes[i];
        let g_left = mat_exp(&(j * (t_total - t2)));

        if t2 > t_onset + 1e-12 {
            let (inner_nodes, inner_weights) = gl16_rescale(t_onset, t2);
            for k in 0..16 {
                let t1 = inner_nodes[k];
                let g_mid = mat_exp(&(j * (t2 - t1)));
                let g_right = mat_exp(&(j * t1));
                second +=
                    (&g_left * v * &g_mid * v * g_right) * (outer_weights[i] * inner_weights[k]);
            }
        }
    }

    let first_norm = first.norm();
    let second_norm = second.norm();
    let convergence_ratio = if first_norm > 1e-15 {
        second_norm / first_norm
    } else {
        0.0
    };

    DysonTerms {
        zeroth,
        first,
        second,
        convergence_ratio,
    }
}

/// Analyze the impulse response to a set of threats.
pub fn impulse_response(
    j: &DMatrix<f64>,
    threats: &[ThreatProfile],
    scenario: &str,
) -> ImpulseResponseReport {
    let dim = j.nrows();
    let x0 = vec![0.0; dim];
    let dt = 0.05;
    let t_final = 60.0;
    let ts = evolve(j, threats, &x0, t_final, dt);

    let mut peak_values = vec![0.0; dim];
    let mut peak_times = vec![0.0; dim];
    let mut recovery_times = vec![f64::INFINITY; dim];
    let mut past_peak = vec![false; dim];

    for (step, state) in ts.states.iter().enumerate() {
        let t = ts.times[step];
        for i in 0..dim {
            let v = state[i].abs();
            if v > peak_values[i] {
                peak_values[i] = v;
                peak_times[i] = t;
                past_peak[i] = false;
            } else if v < peak_values[i] {
                past_peak[i] = true;
            }

            if past_peak[i] && v < peak_values[i] * 0.1 && recovery_times[i] == f64::INFINITY {
                recovery_times[i] = t - peak_times[i];
            }
        }
    }

    ImpulseResponseReport {
        scenario: scenario.to_string(),
        peak_values,
        peak_times,
        recovery_times,
        time_series: ts,
    }
}

/// Run cascade analysis: threat A alone, threat B alone, then A→B sequentially.
pub fn cascade_analysis(
    j: &DMatrix<f64>,
    threat_a: &ThreatProfile,
    threat_b: &ThreatProfile,
) -> CascadeReport {
    let dim = j.nrows();
    let a_report = impulse_response(j, &[threat_a.clone()], &threat_a.name);
    let b_report = impulse_response(j, &[threat_b.clone()], &threat_b.name);
    let cascade_report = impulse_response(
        j,
        &[threat_a.clone(), threat_b.clone()],
        &format!("{} -> {}", threat_a.name, threat_b.name),
    );

    let mut compounding = vec![0.0; dim];
    for i in 0..dim {
        let individual_max = a_report.peak_values[i].max(b_report.peak_values[i]);
        compounding[i] = if individual_max > 1e-12 {
            cascade_report.peak_values[i] / individual_max
        } else {
            1.0
        };
    }

    CascadeReport {
        threat_a_alone: a_report,
        threat_b_alone: b_report,
        cascade: cascade_report,
        compounding_factors: compounding,
    }
}

/// Find the convergence radius — the perturbation magnitude α where
/// ‖U²‖/‖U¹‖ ≈ 1 (Dyson series diverges beyond this point).
pub fn convergence_radius(
    j: &DMatrix<f64>,
    v_direction: &DMatrix<f64>,
    t_onset: f64,
    t_end: f64,
) -> f64 {
    let v_norm = v_direction.norm();
    if v_norm < 1e-15 {
        return f64::INFINITY;
    }
    let v_hat = v_direction / v_norm;

    let mut lo = 0.01_f64;
    let mut hi = 1000.0_f64;

    let terms_lo = compute_dyson_terms(j, &(&v_hat * lo), t_onset, t_end);
    if terms_lo.convergence_ratio >= 1.0 {
        return lo * 0.5;
    }

    let terms_hi = compute_dyson_terms(j, &(&v_hat * hi), t_onset, t_end);
    if terms_hi.convergence_ratio < 1.0 {
        return hi;
    }

    for _ in 0..50 {
        let mid = (lo + hi) * 0.5;
        let terms = compute_dyson_terms(j, &(&v_hat * mid), t_onset, t_end);
        if terms.convergence_ratio < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) / lo < 0.01 {
            break;
        }
    }

    (lo + hi) * 0.5
}

/// Complete Dyson analysis from pre-built components.
///
/// Unlike the Phalanx version which hardcodes threat profiles, this takes
/// threats as parameters — the domain-specific threat construction stays
/// in the caller.
pub fn full_dyson_analysis(
    j: &DMatrix<f64>,
    threats: &[ThreatProfile],
    cascade_pair: Option<(&ThreatProfile, &ThreatProfile)>,
    dyson_perturbation: Option<(&DMatrix<f64>, f64, f64)>,
) -> DysonAnalysisReport {
    let dim = j.nrows();

    let threat_responses: Vec<ImpulseResponseReport> = threats
        .iter()
        .map(|t| impulse_response(j, &[t.clone()], &t.name))
        .collect();

    let cascade = match cascade_pair {
        Some((a, b)) => cascade_analysis(j, a, b),
        None if threats.len() >= 2 => cascade_analysis(j, &threats[0], &threats[1]),
        _ => {
            let dummy = ThreatProfile {
                name: "none".into(),
                onset: 0.0,
                duration: 0.0,
                forcing: vec![0.0; dim],
                coupling_delta: None,
            };
            cascade_analysis(j, &dummy, &dummy)
        }
    };

    let (dyson_terms, conv_radius) = match dyson_perturbation {
        Some((v, t_onset, t_end)) => {
            let terms = compute_dyson_terms(j, v, t_onset, t_end);
            let radius = convergence_radius(j, v, t_onset, t_end);
            (terms, radius)
        }
        None => {
            let terms = DysonTerms {
                zeroth: DMatrix::identity(dim, dim),
                first: DMatrix::zeros(dim, dim),
                second: DMatrix::zeros(dim, dim),
                convergence_ratio: 0.0,
            };
            (terms, f64::INFINITY)
        }
    };

    DysonAnalysisReport {
        threat_responses,
        cascade,
        dyson_terms,
        convergence_radius: conv_radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evolve_zero_forcing_decays() {
        // With no threats, the system should return to zero.
        let mut j = DMatrix::zeros(2, 2);
        j[(0, 0)] = -1.0;
        j[(1, 1)] = -2.0;
        let x0 = vec![1.0, 1.0];
        let ts = evolve(&j, &[], &x0, 10.0, 0.1);
        let last = ts.states.last().unwrap();
        assert!(last[0].abs() < 1e-3);
        assert!(last[1].abs() < 1e-3);
    }

    #[test]
    fn dyson_first_order_nonzero_for_nontrivial_v() {
        let mut j = DMatrix::zeros(2, 2);
        j[(0, 0)] = -1.0;
        j[(1, 1)] = -0.5;
        let mut v = DMatrix::zeros(2, 2);
        v[(0, 1)] = 1.0;
        let terms = compute_dyson_terms(&j, &v, 0.0, 5.0);
        assert!(terms.first.norm() > 1e-6);
    }
}
