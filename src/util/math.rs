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
}
