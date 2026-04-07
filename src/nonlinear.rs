//! Nonlinear dynamics: Lyapunov exponent computation via Benettin's method.
//!
//! Laboratory layer: pure math. The domain-specific `NonlinearDynamics` trait
//! is defined in `integrators.rs`. This module provides the Lyapunov exponent
//! algorithm that works with any implementation.

use nalgebra::DVector;

use crate::integrators::{rk4_step, rk4_variational_step, NonlinearDynamics};

/// Result of the finite-time Lyapunov exponent computation.
#[derive(Debug, Clone)]
pub struct LyapunovResult {
    /// Maximal Lyapunov exponent μ₁. Negative → stable through the transient.
    pub mu1: f64,
    /// Running estimate of μ₁ at each renormalization event: (time, μ₁_running).
    pub running_estimate: Vec<(f64, f64)>,
    /// Number of renormalization events.
    pub renorm_count: usize,
}

/// Compute the maximal finite-time Lyapunov exponent through a simulation.
///
/// Co-evolves a perturbation vector δx alongside the state trajectory using
/// the variational equation dδ/dt = Df(x(t))·δ. Renormalization every
/// `renorm_interval` steps prevents overflow (Benettin et al., 1980).
///
/// The caller provides:
/// - `sys`: the nonlinear dynamics (must support `rhs` and `jacobian`)
/// - `x0`: initial state
/// - `dt`: time step
/// - `n_warmup`: steps to reach steady state before measurement
/// - `phases`: sequence of (n_steps, perturbation_active) pairs
/// - `renorm_interval`: steps between renormalization events
///
/// μ₁ < 0 is the definitive nonlinear stability certificate.
pub fn compute_lyapunov_exponent(
    sys: &mut dyn NonlinearDynamics,
    x0: &DVector<f64>,
    dt: f64,
    n_warmup: usize,
    phases: &[(usize, bool)],
    renorm_interval: usize,
) -> LyapunovResult {
    let dim = sys.dim();

    // Warmup to steady state
    let mut x = x0.clone();
    for _ in 0..n_warmup {
        x = rk4_step(sys, 0.0, &x, dt);
    }

    // Initialize perturbation as unit vector in first direction.
    // The asymptotic Lyapunov exponent is independent of initial direction.
    let mut delta = DVector::zeros(dim);
    delta[0] = 1.0;

    let mut lyap_sum = 0.0;
    let mut renorm_count = 0;
    let mut t = 0.0;
    let mut running_estimate = Vec::new();
    let mut step_count: usize = 0;

    for &(n_steps, perturbation_active) in phases {
        sys.set_perturbation(perturbation_active);

        for _ in 0..n_steps {
            let jac = sys.jacobian(t, &x);
            delta = rk4_variational_step(&jac, &delta, dt);
            x = rk4_step(sys, t, &x, dt);
            t += dt;
            step_count += 1;
            sys.tick(dt);

            if step_count % renorm_interval == 0 {
                let norm = delta.norm();
                if norm > 0.0 {
                    lyap_sum += norm.ln();
                    renorm_count += 1;
                    delta /= norm;
                    running_estimate.push((t, lyap_sum / t));
                }
            }
        }
    }

    let mu1 = if t > 0.0 { lyap_sum / t } else { 0.0 };

    LyapunovResult {
        mu1,
        running_estimate,
        renorm_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrators::NonlinearDynamics;
    use nalgebra::{DMatrix, DVector};

    struct LinearDecay;

    impl NonlinearDynamics for LinearDecay {
        fn dim(&self) -> usize {
            2
        }
        fn rhs(&self, _t: f64, x: &DVector<f64>) -> DVector<f64> {
            DVector::from_vec(vec![-0.5 * x[0], -1.0 * x[1]])
        }
        fn jacobian(&self, _t: f64, _x: &DVector<f64>) -> DMatrix<f64> {
            let mut j = DMatrix::zeros(2, 2);
            j[(0, 0)] = -0.5;
            j[(1, 1)] = -1.0;
            j
        }
    }

    #[test]
    fn lyapunov_negative_for_stable_linear() {
        let mut sys = LinearDecay;
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let result = compute_lyapunov_exponent(
            &mut sys,
            &x0,
            0.01,
            0,
            &[(10000, false)],
            50,
        );
        assert!(
            result.mu1 < 0.0,
            "Lyapunov exponent should be negative for stable system, got {}",
            result.mu1
        );
    }
}
