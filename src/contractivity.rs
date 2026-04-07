//! Compile-time contractivity verification framework.
//!
//! Laboratory layer: pure math. Provides the generic machinery for proving
//! contractivity under a Lyapunov matrix P at compile time via Cholesky
//! factorization of -Q where Q = PJ_n + J_nᵀP.
//!
//! This module contains the reusable const-fn building blocks. The actual
//! proof instantiation — with domain-specific constants, operating region
//! vertices, and traffic regimes — stays in the domain crate.

/// Newton-iteration square root. Converges to f64 precision in ≤30 iterations.
/// Suitable for use in `const fn` contexts.
pub const fn const_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = x;
    let mut i = 0;
    while i < 30 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

/// Compute Q = P·J_n + (P·J_n)ᵀ for DIM×DIM matrices stored as nested arrays.
///
/// Caller provides P and J_n as `[[f64; DIM]; DIM]`.
/// Returns Q, which is symmetric by construction.
pub const fn compute_q<const DIM: usize>(
    p: [[f64; DIM]; DIM],
    jn: [[f64; DIM]; DIM],
) -> [[f64; DIM]; DIM] {
    // R = P × J_n
    let mut r = [[0.0f64; DIM]; DIM];
    let mut i = 0;
    while i < DIM {
        let mut j = 0;
        while j < DIM {
            let mut sum = 0.0;
            let mut k = 0;
            while k < DIM {
                sum += p[i][k] * jn[k][j];
                k += 1;
            }
            r[i][j] = sum;
            j += 1;
        }
        i += 1;
    }
    // Q = R + Rᵀ
    let mut q = [[0.0f64; DIM]; DIM];
    let mut i = 0;
    while i < DIM {
        let mut j = 0;
        while j < DIM {
            q[i][j] = r[i][j] + r[j][i];
            j += 1;
        }
        i += 1;
    }
    q
}

/// Cholesky factorization of −Q. Returns `true` iff Q is negative definite.
///
/// This is the core of the compile-time contractivity proof. If Cholesky
/// succeeds on −Q (i.e., −Q is positive definite), then Q ≺ 0.
pub const fn is_neg_def<const DIM: usize>(q: [[f64; DIM]; DIM]) -> bool {
    let mut l = [[0.0f64; DIM]; DIM];
    let mut i = 0;
    while i < DIM {
        let mut j = 0;
        while j <= i {
            let mut sum = 0.0;
            let mut k = 0;
            while k < j {
                sum += l[i][k] * l[j][k];
                k += 1;
            }
            if i == j {
                let val = -q[i][i] - sum;
                if val <= 0.0 {
                    return false;
                }
                l[i][j] = const_sqrt(val);
            } else {
                l[i][j] = (-q[i][j] - sum) / l[j][j];
            }
            j += 1;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_definite_2x2() {
        // Q = [[-2, 0], [0, -3]] is negative definite
        let q = [[-2.0, 0.0], [0.0, -3.0]];
        assert!(is_neg_def(q));
    }

    #[test]
    fn not_negative_definite_2x2() {
        // Q = [[1, 0], [0, -1]] is NOT negative definite
        let q = [[1.0, 0.0], [0.0, -1.0]];
        assert!(!is_neg_def(q));
    }

    #[test]
    fn compute_q_symmetric() {
        let p = [[1.0, 0.0], [0.0, 1.0]];
        let jn = [[-1.0, 0.5], [-0.5, -2.0]];
        let q = compute_q(p, jn);
        // Q = P·Jn + (P·Jn)ᵀ = Jn + Jnᵀ (since P = I)
        assert!((q[0][1] - q[1][0]).abs() < 1e-12, "Q should be symmetric");
    }

    #[test]
    fn const_sqrt_accuracy() {
        let vals = [0.0, 1.0, 2.0, 4.0, 100.0, 0.01];
        for v in vals {
            let result = const_sqrt(v);
            let expected = (v as f64).sqrt();
            assert!(
                (result - expected).abs() < 1e-12,
                "const_sqrt({}) = {}, expected {}",
                v,
                result,
                expected
            );
        }
    }
}
