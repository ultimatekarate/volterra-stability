//! Matrix exponential via Padé(13) scaling-and-squaring (Higham 2005).
//!
//! Laboratory layer: pure math. No domain knowledge, no IO.

use nalgebra::DMatrix;

/// Padé(13) coefficients for the matrix exponential.
const PADE13_B: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];

/// Padé(13) scaling-and-squaring matrix exponential.
///
/// Computes exp(A) for a square matrix A using the Higham (2005) algorithm.
/// All linear solves use LU factorization — no explicit matrix inversions.
pub fn mat_exp(a: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "mat_exp requires a square matrix");

    // 1-norm for scaling decision
    let norm1: f64 = (0..n)
        .map(|col| (0..n).map(|row| a[(row, col)].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    // Scaling: s = max(0, ceil(log2(norm1 / theta_13)))
    const THETA_13: f64 = 5.371920351148152;
    let s = if norm1 > THETA_13 {
        (norm1 / THETA_13).log2().ceil() as u32
    } else {
        0
    };

    let a_scaled = if s > 0 {
        a / (2.0_f64.powi(s as i32))
    } else {
        a.clone()
    };

    // Compute matrix powers: A², A⁴, A⁶
    let id = DMatrix::identity(n, n);
    let a2 = &a_scaled * &a_scaled;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;

    // Build U₁₃ and V₁₃ from Padé coefficients
    let v_inner = &a6 * PADE13_B[12] + &a4 * PADE13_B[10] + &a2 * PADE13_B[8];
    let v13 = &v_inner * &a6
        + &a6 * PADE13_B[6]
        + &a4 * PADE13_B[4]
        + &a2 * PADE13_B[2]
        + &id * PADE13_B[0];

    let u_inner = &a6 * PADE13_B[13] + &a4 * PADE13_B[11] + &a2 * PADE13_B[9];
    let u13 = &a_scaled
        * &(&u_inner * &a6
            + &a6 * PADE13_B[7]
            + &a4 * PADE13_B[5]
            + &a2 * PADE13_B[3]
            + &id * PADE13_B[1]);

    // Solve (V₁₃ − U₁₃)·R = V₁₃ + U₁₃ via LU factorization
    let lhs = &v13 - &u13;
    let rhs = &v13 + &u13;
    let lu = lhs.lu();
    let mut result = lu.solve(&rhs).expect("Padé denominator is singular");

    // Repeated squaring: R = R² done s times
    for _ in 0..s {
        result = &result * &result;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_exp_of_zero_is_identity() {
        let zero = DMatrix::zeros(4, 4);
        let result = mat_exp(&zero);
        let id = DMatrix::identity(4, 4);
        assert!(
            (&result - &id).norm() < 1e-12,
            "exp(0) should be identity"
        );
    }

    #[test]
    fn mat_exp_diagonal() {
        let lambdas = [-4.0, -0.5, -0.1, -1.0];
        let mut diag = DMatrix::zeros(4, 4);
        for i in 0..4 {
            diag[(i, i)] = lambdas[i];
        }
        let result = mat_exp(&diag);
        for i in 0..4 {
            let expected = lambdas[i].exp();
            assert!(
                (result[(i, i)] - expected).abs() < 1e-12,
                "exp(diag)[{i},{i}] = {}, expected {expected}",
                result[(i, i)]
            );
            for k in 0..4 {
                if k != i {
                    assert!(result[(i, k)].abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn mat_exp_nilpotent_2x2() {
        // exp([[0, 1], [0, 0]]) = [[1, 1], [0, 1]]
        let mut a = DMatrix::zeros(2, 2);
        a[(0, 1)] = 1.0;
        let result = mat_exp(&a);
        assert!((result[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((result[(0, 1)] - 1.0).abs() < 1e-12);
        assert!((result[(1, 0)] - 0.0).abs() < 1e-12);
        assert!((result[(1, 1)] - 1.0).abs() < 1e-12);
    }
}
