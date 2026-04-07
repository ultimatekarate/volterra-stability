//! Numerical ODE integration: fixed-step RK4 and adaptive Dormand-Prince
//! RK4(5) with PI step size control.
//!
//! Laboratory layer: pure math. Domain-independent — operates on any system
//! that implements the `NonlinearDynamics` trait.

use nalgebra::{DMatrix, DVector};

use crate::report::{AdaptiveStepConfig, StepStatistics};

/// ODE right-hand side for nonlinear analysis.
///
/// Implementors provide the dynamics dx/dt = f(t, x) and optionally
/// an instantaneous Jacobian for variational equation integration.
pub trait NonlinearDynamics {
    /// System dimension.
    fn dim(&self) -> usize;

    /// Evaluate the right-hand side dx/dt = f(t, x).
    fn rhs(&self, t: f64, x: &DVector<f64>) -> DVector<f64>;

    /// Instantaneous Jacobian Df(x) at the current state.
    /// Default: central finite differences (2N evaluations).
    fn jacobian(&self, t: f64, x: &DVector<f64>) -> DMatrix<f64> {
        let n = self.dim();
        let h = 1e-7;
        let mut jac = DMatrix::zeros(n, n);
        for col in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[col] += h;
            x_minus[col] -= h;
            let f_plus = self.rhs(t, &x_plus);
            let f_minus = self.rhs(t, &x_minus);
            for row in 0..n {
                jac[(row, col)] = (f_plus[row] - f_minus[row]) / (2.0 * h);
            }
        }
        jac
    }

    /// Called after each accepted step to update internal state (e.g., burst timers).
    /// Default: no-op.
    fn tick(&mut self, _dt: f64) {}

    /// Enable/disable a perturbation mode (e.g., network partition).
    /// Default: no-op.
    fn set_perturbation(&mut self, _active: bool) {}
}

// =====================================================================
// DORMAND-PRINCE RK4(5) BUTCHER TABLEAU
// =====================================================================

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// 5th-order weights
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

// Error weights (5th - 4th order)
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

/// Fourth-order Runge-Kutta step.
pub fn rk4_step(
    sys: &dyn NonlinearDynamics,
    t: f64,
    x: &DVector<f64>,
    dt: f64,
) -> DVector<f64> {
    let k1 = sys.rhs(t, x);
    let k2 = sys.rhs(t + 0.5 * dt, &(x + &k1 * (0.5 * dt)));
    let k3 = sys.rhs(t + 0.5 * dt, &(x + &k2 * (0.5 * dt)));
    let k4 = sys.rhs(t + dt, &(x + &k3 * dt));

    let mut x_new = x + &(&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) * (dt / 6.0);
    // Clamp non-negative (integral values cannot go below zero)
    for i in 0..x_new.nrows() {
        x_new[i] = x_new[i].max(0.0);
    }
    x_new
}

/// RK4 step for the variational equation dδ/dt = J(t)·δ.
pub fn rk4_variational_step(
    jac: &DMatrix<f64>,
    delta: &DVector<f64>,
    dt: f64,
) -> DVector<f64> {
    let k1 = jac * delta;
    let k2 = jac * &(delta + &k1 * (0.5 * dt));
    let k3 = jac * &(delta + &k2 * (0.5 * dt));
    let k4 = jac * &(delta + &k3 * dt);

    delta + &(&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) * (dt / 6.0)
}

/// One Dormand-Prince step: returns (x_new_5th, error_estimate).
pub fn dopri5_step(
    sys: &dyn NonlinearDynamics,
    t: f64,
    x: &DVector<f64>,
    dt: f64,
) -> (DVector<f64>, DVector<f64>) {
    let n = x.nrows();
    let k1 = sys.rhs(t, x);

    let k2 = sys.rhs(t + dt * A21, &(x + &k1 * (dt * A21)));
    let k3 = sys.rhs(
        t + dt * (A31 + A32),
        &(x + &k1 * (dt * A31) + &k2 * (dt * A32)),
    );
    let k4 = sys.rhs(
        t + dt * (A41 + A42 + A43),
        &(x + &k1 * (dt * A41) + &k2 * (dt * A42) + &k3 * (dt * A43)),
    );
    let k5 = sys.rhs(
        t + dt * (A51 + A52 + A53 + A54),
        &(x + &k1 * (dt * A51) + &k2 * (dt * A52) + &k3 * (dt * A53) + &k4 * (dt * A54)),
    );
    let k6 = sys.rhs(
        t + dt,
        &(x + &k1 * (dt * A61)
            + &k2 * (dt * A62)
            + &k3 * (dt * A63)
            + &k4 * (dt * A64)
            + &k5 * (dt * A65)),
    );

    // 5th-order solution
    let mut x_new = x + &(&k1 * B1 + &k3 * B3 + &k4 * B4 + &k5 * B5 + &k6 * B6) * dt;
    for i in 0..n {
        x_new[i] = x_new[i].max(0.0);
    }

    // Stage 7 (FSAL, for error estimate only)
    let k7 = sys.rhs(t + dt, &x_new);

    // Error estimate
    let err = (&k1 * E1 + &k3 * E3 + &k4 * E4 + &k5 * E5 + &k6 * E6 + &k7 * E7) * dt;

    (x_new, err)
}

/// Mixed-tolerance error norm (Hairer & Wanner convention).
pub fn error_norm(
    err: &DVector<f64>,
    x: &DVector<f64>,
    x_new: &DVector<f64>,
    atol: f64,
    rtol: f64,
) -> f64 {
    let dim = err.nrows();
    let mut sum_sq = 0.0;
    for i in 0..dim {
        let scale = atol + rtol * x[i].abs().max(x_new[i].abs());
        let ratio = err[i] / scale;
        sum_sq += ratio * ratio;
    }
    (sum_sq / dim as f64).sqrt()
}

/// Try one adaptive step. Returns (x_new, dt_used, dt_next, accepted).
pub fn adaptive_step(
    sys: &dyn NonlinearDynamics,
    t: f64,
    x: &DVector<f64>,
    dt: f64,
    acfg: &AdaptiveStepConfig,
) -> (DVector<f64>, f64, f64, bool) {
    let (x_new, err) = dopri5_step(sys, t, x, dt);
    let en = error_norm(&err, x, &x_new, acfg.atol, acfg.rtol);

    if en <= 1.0 {
        let growth = if en < 1e-10 {
            acfg.max_growth
        } else {
            (acfg.safety * en.powf(-0.2)).min(acfg.max_growth)
        };
        let dt_next = (dt * growth).min(acfg.dt_max);
        (x_new, dt, dt_next, true)
    } else {
        let shrink = (acfg.safety * en.powf(-0.2)).max(acfg.min_shrink);
        let dt_next = (dt * shrink).max(acfg.dt_min);
        (x.clone(), dt, dt_next, false)
    }
}

/// Advance the system from current time to `t_end`, recording states.
pub fn advance_to(
    sys: &mut dyn NonlinearDynamics,
    x: &mut DVector<f64>,
    t: &mut f64,
    t_end: f64,
    dt: &mut f64,
    times: &mut Vec<f64>,
    states: &mut Vec<Vec<f64>>,
    adaptive: Option<&AdaptiveStepConfig>,
    stats: &mut StepStatistics,
) {
    match adaptive {
        Some(acfg) => {
            while *t < t_end - 1e-12 {
                times.push(*t);
                states.push(x.as_slice().to_vec());

                let dt_try = (*dt).min(t_end - *t).max(acfg.dt_min);
                let (x_new, dt_used, dt_next, accepted) =
                    adaptive_step(sys, *t, x, dt_try, acfg);
                if accepted {
                    *x = x_new;
                    *t += dt_used;
                    stats.record_accepted(dt_used);
                    sys.tick(dt_used);
                    *dt = dt_next;
                } else {
                    stats.n_rejected += 1;
                    *dt = dt_next;
                }
            }
        }
        None => {
            let fixed_dt = *dt;
            let n_steps = ((t_end - *t) / fixed_dt).ceil() as usize;
            for _ in 0..n_steps {
                times.push(*t);
                states.push(x.as_slice().to_vec());
                sys.tick(fixed_dt);
                *x = rk4_step(sys, *t, x, fixed_dt);
                *t += fixed_dt;
                stats.record_accepted(fixed_dt);
            }
        }
    }
}

/// Linearly interpolate the state trajectory at a query time.
pub fn interpolate_state(times: &[f64], states: &[Vec<f64>], t_query: f64) -> Vec<f64> {
    debug_assert!(!times.is_empty());
    if t_query <= times[0] {
        return states[0].clone();
    }
    if t_query >= *times.last().unwrap() {
        return states.last().unwrap().clone();
    }

    let idx = match times.binary_search_by(|t| t.partial_cmp(&t_query).unwrap()) {
        Ok(i) => return states[i].clone(),
        Err(i) => i,
    };

    let t0 = times[idx - 1];
    let t1 = times[idx];
    let alpha = (t_query - t0) / (t1 - t0);
    let dim = states[0].len();

    let mut result = vec![0.0; dim];
    for j in 0..dim {
        result[j] = states[idx - 1][j] * (1.0 - alpha) + states[idx][j] * alpha;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ExponentialDecay {
        lambda: f64,
    }

    impl NonlinearDynamics for ExponentialDecay {
        fn dim(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, x: &DVector<f64>) -> DVector<f64> {
            DVector::from_element(1, -self.lambda * x[0])
        }
    }

    #[test]
    fn rk4_exponential_decay() {
        let sys = ExponentialDecay { lambda: 1.0 };
        let mut x = DVector::from_element(1, 1.0);
        let dt = 0.01;
        for _ in 0..100 {
            x = rk4_step(&sys, 0.0, &x, dt);
        }
        let expected = (-1.0_f64).exp();
        assert!(
            (x[0] - expected).abs() < 1e-6,
            "RK4 decay: got {}, expected {}",
            x[0],
            expected
        );
    }

    #[test]
    fn interpolation_midpoint() {
        let times = vec![0.0, 1.0, 2.0];
        let states = vec![vec![0.0], vec![2.0], vec![4.0]];
        let mid = interpolate_state(&times, &states, 0.5);
        assert!((mid[0] - 1.0).abs() < 1e-12);
    }
}
