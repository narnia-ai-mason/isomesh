/// 3D vector operations and 3x3 symmetric eigendecomposition.

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    let l = length(a);
    if l < 1e-15 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / l)
    }
}

/// 3x3 symmetric matrix stored as [a00, a01, a02, a11, a12, a22].
pub type Sym3x3 = [f64; 6];

/// Multiply symmetric 3x3 matrix by vector.
pub fn sym_mul_vec(m: &Sym3x3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[1] * v[0] + m[3] * v[1] + m[4] * v[2],
        m[2] * v[0] + m[4] * v[1] + m[5] * v[2],
    ]
}

/// Eigendecomposition of a 3x3 symmetric matrix via Jacobi rotations.
///
/// Returns (eigenvalues, eigenvectors_as_columns).
/// eigenvalues are sorted descending.
/// Each eigenvector is a column of the returned 3x3 matrix.
pub fn jacobi_eigen_3x3(m: &Sym3x3) -> ([f64; 3], [[f64; 3]; 3]) {
    // Expand symmetric to full 3x3
    let mut s = [
        [m[0], m[1], m[2]],
        [m[1], m[3], m[4]],
        [m[2], m[4], m[5]],
    ];
    let mut v = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    for _ in 0..5 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if s[p][q].abs() < 1e-15 {
                continue;
            }
            let tau = (s[q][q] - s[p][p]) / (2.0 * s[p][q]);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let sn = t * c;

            // Rotate S
            let spp = s[p][p];
            let sqq = s[q][q];
            let spq = s[p][q];

            s[p][p] = c * c * spp - 2.0 * c * sn * spq + sn * sn * sqq;
            s[q][q] = sn * sn * spp + 2.0 * c * sn * spq + c * c * sqq;
            s[p][q] = 0.0;
            s[q][p] = 0.0;

            for r in 0..3 {
                if r != p && r != q {
                    let srp = s[r][p];
                    let srq = s[r][q];
                    s[r][p] = c * srp - sn * srq;
                    s[p][r] = s[r][p];
                    s[r][q] = sn * srp + c * srq;
                    s[q][r] = s[r][q];
                }
            }

            // Rotate V
            for r in 0..3 {
                let vrp = v[r][p];
                let vrq = v[r][q];
                v[r][p] = c * vrp - sn * vrq;
                v[r][q] = sn * vrp + c * vrq;
            }
        }
    }

    let mut eigenvalues = [s[0][0], s[1][1], s[2][2]];
    // Columns of v are eigenvectors
    let mut eigenvectors = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];

    // Sort descending by eigenvalue
    for i in 0..3 {
        for j in (i + 1)..3 {
            if eigenvalues[j] > eigenvalues[i] {
                eigenvalues.swap(i, j);
                eigenvectors.swap(i, j);
            }
        }
    }

    (eigenvalues, eigenvectors)
}

/// Solve Ax = b for a 3x3 symmetric positive-definite matrix via LDL^T factorization.
///
/// A is given as Sym3x3 = [a00, a01, a02, a11, a12, a22].
/// Precondition: A must be positive definite (guaranteed by probabilistic quadrics
/// regularization term σ_n² · I).
pub fn ldlt_solve_3x3(a: &Sym3x3, b: [f64; 3]) -> [f64; 3] {
    // LDL^T factorization (unrolled for 3x3)
    let d0 = a[0]; // a00
    let inv_d0 = 1.0 / d0;
    let l10 = a[1] * inv_d0; // a01 / a00
    let l20 = a[2] * inv_d0; // a02 / a00

    let d1 = a[3] - a[1] * l10; // a11 - a01²/a00
    let inv_d1 = 1.0 / d1;
    let l21 = (a[4] - a[2] * l10) * inv_d1; // (a12 - a02·l10) / d1

    let d2 = a[5] - a[2] * l20 - (a[4] - a[2] * l10) * l21;

    // Forward substitution: L y = b
    let y0 = b[0];
    let y1 = b[1] - l10 * y0;
    let y2 = b[2] - l20 * y0 - l21 * y1;

    // Diagonal solve: D z = y
    let z0 = y0 * inv_d0;
    let z1 = y1 * inv_d1;
    let z2 = y2 / d2;

    // Back substitution: L^T x = z
    let x2 = z2;
    let x1 = z1 - l21 * x2;
    let x0 = z0 - l10 * x1 - l20 * x2;

    [x0, x1, x2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_eigenvalues() {
        let m: Sym3x3 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let (vals, _) = jacobi_eigen_3x3(&m);
        for v in vals {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_diagonal_eigenvalues() {
        let m: Sym3x3 = [3.0, 0.0, 0.0, 2.0, 0.0, 1.0];
        let (vals, _) = jacobi_eigen_3x3(&m);
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10);
        assert!((vals[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_known_symmetric() {
        // [2, 1, 0; 1, 3, 1; 0, 1, 2] has known eigenvalues
        let m: Sym3x3 = [2.0, 1.0, 0.0, 3.0, 1.0, 2.0];
        let (vals, vecs) = jacobi_eigen_3x3(&m);
        // Verify Av = λv for each eigenpair
        for i in 0..3 {
            let av = sym_mul_vec(&m, vecs[i]);
            let lv = scale(vecs[i], vals[i]);
            for j in 0..3 {
                assert!((av[j] - lv[j]).abs() < 1e-8,
                    "Eigenvector {} component {} mismatch: {} vs {}", i, j, av[j], lv[j]);
            }
        }
    }

    #[test]
    fn test_ldlt_identity() {
        let a: Sym3x3 = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let b = [3.0, 5.0, 7.0];
        let x = ldlt_solve_3x3(&a, b);
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-12, "component {}: {}", i, x[i]);
        }
    }

    #[test]
    fn test_ldlt_known_spd() {
        // A = [[4,2,1],[2,5,3],[1,3,6]] (SPD)
        let a: Sym3x3 = [4.0, 2.0, 1.0, 5.0, 3.0, 6.0];
        let b = [1.0, 2.0, 3.0];
        let x = ldlt_solve_3x3(&a, b);
        let ax = sym_mul_vec(&a, x);
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-10,
                "component {}: Ax={} vs b={}", i, ax[i], b[i]);
        }
    }

    #[test]
    fn test_ldlt_diagonal() {
        let a: Sym3x3 = [2.0, 0.0, 0.0, 3.0, 0.0, 5.0];
        let b = [4.0, 9.0, 15.0];
        let x = ldlt_solve_3x3(&a, b);
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_ldlt_regularized_rank1() {
        // Rank-1 (n=[1,0,0] outer product) + ε·I simulates probabilistic quadrics
        let eps = 0.01;
        let a: Sym3x3 = [1.0 + eps, 0.0, 0.0, eps, 0.0, eps];
        let b = [0.5 + eps * 0.3, eps * 0.4, eps * 0.5];
        let x = ldlt_solve_3x3(&a, b);
        let ax = sym_mul_vec(&a, x);
        for i in 0..3 {
            assert!((ax[i] - b[i]).abs() < 1e-10,
                "component {}: Ax={} vs b={}", i, ax[i], b[i]);
        }
    }
}
