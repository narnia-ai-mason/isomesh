use crate::util::math::{self, Sym3x3};

/// Compact QEF representation with probabilistic quadrics regularization.
///
/// Represents the expected quadric error under Gaussian noise on positions
/// and normals (Trettner & Kobbelt, CGF 2020). The σ_n² regularization term
/// makes A^T A always positive definite, enabling direct LDL^T solve
/// without SVD or rank-deficiency handling.
#[derive(Clone, Copy, Debug)]
pub struct QefData {
    /// Upper triangle of A^T A (symmetric 3x3), includes σ_n² · I regularization.
    pub ata: Sym3x3,
    /// A^T b vector, includes σ_n² · p bias term.
    pub atb: [f64; 3],
    /// b^T b scalar (for error computation).
    pub btb: f64,
    /// Mass point accumulator (for SVD fallback).
    pub mass_point_sum: [f64; 3],
    /// Number of data points accumulated.
    pub count: u32,
}

impl QefData {
    pub fn new() -> Self {
        Self {
            ata: [0.0; 6],
            atb: [0.0; 3],
            btb: 0.0,
            mass_point_sum: [0.0; 3],
            count: 0,
        }
    }

    /// Add a Hermite data point with probabilistic quadrics regularization.
    ///
    /// `sigma_n`: normal noise std dev (typically 0.01)
    /// `sigma_p`: position noise std dev (typically 0.01 × cell_size)
    pub fn add(&mut self, pos: [f64; 3], normal: [f64; 3], sigma_n: f64, sigma_p: f64) {
        let n = normal;
        let d = math::dot(n, pos);
        let sn2 = sigma_n * sigma_n;
        let sp2 = sigma_p * sigma_p;

        // A^T A += n * n^T + σ_n² · I₃
        self.ata[0] += n[0] * n[0] + sn2;
        self.ata[1] += n[0] * n[1];
        self.ata[2] += n[0] * n[2];
        self.ata[3] += n[1] * n[1] + sn2;
        self.ata[4] += n[1] * n[2];
        self.ata[5] += n[2] * n[2] + sn2;

        // A^T b += n * d + σ_n² · p
        self.atb[0] += n[0] * d + sn2 * pos[0];
        self.atb[1] += n[1] * d + sn2 * pos[1];
        self.atb[2] += n[2] * d + sn2 * pos[2];

        // b^T b += d² + σ_n²(p·p) + σ_p²(n·n) + 3·σ_p²·σ_n²
        self.btb += d * d
            + sn2 * math::dot(pos, pos)
            + sp2 * math::dot(n, n)
            + 3.0 * sp2 * sn2;

        // Mass point (retained for SVD fallback)
        self.mass_point_sum[0] += pos[0];
        self.mass_point_sum[1] += pos[1];
        self.mass_point_sum[2] += pos[2];
        self.count += 1;
    }

    /// Solve QEF via LDL^T Cholesky (probabilistic quadrics path).
    ///
    /// Requires that `add()` was called with σ_n > 0.
    /// Returns (position, error).
    pub fn solve(&self, cell_min: [f64; 3], cell_max: [f64; 3]) -> ([f64; 3], f64) {
        if self.count == 0 {
            let center = [
                (cell_min[0] + cell_max[0]) * 0.5,
                (cell_min[1] + cell_max[1]) * 0.5,
                (cell_min[2] + cell_max[2]) * 0.5,
            ];
            return (center, 0.0);
        }

        let mut x = math::ldlt_solve_3x3(&self.ata, self.atb);

        // Clamp to cell bounds
        for i in 0..3 {
            x[i] = x[i].clamp(cell_min[i], cell_max[i]);
        }

        // Compute error: btb - 2 * x^T * atb + x^T * ata * x
        let ata_x = math::sym_mul_vec(&self.ata, x);
        let error = self.btb - 2.0 * math::dot(x, self.atb) + math::dot(x, ata_x);

        (x, error.max(0.0))
    }

    /// Solve QEF via Jacobi eigendecomposition with mass-point fallback (original solver).
    ///
    /// Retained for comparison. Works best when `add()` is called without regularization,
    /// but also works with regularized data (the eigenvalues will simply all be above threshold).
    #[allow(dead_code)]
    pub fn solve_svd(&self, cell_min: [f64; 3], cell_max: [f64; 3]) -> ([f64; 3], f64) {
        if self.count == 0 {
            let center = [
                (cell_min[0] + cell_max[0]) * 0.5,
                (cell_min[1] + cell_max[1]) * 0.5,
                (cell_min[2] + cell_max[2]) * 0.5,
            ];
            return (center, 0.0);
        }

        let inv_count = 1.0 / self.count as f64;
        let mass_point = [
            self.mass_point_sum[0] * inv_count,
            self.mass_point_sum[1] * inv_count,
            self.mass_point_sum[2] * inv_count,
        ];

        let ata_mp = math::sym_mul_vec(&self.ata, mass_point);
        let atb_shifted = math::sub(self.atb, ata_mp);

        let (eigenvalues, eigenvectors) = math::jacobi_eigen_3x3(&self.ata);

        let threshold = 0.1 * eigenvalues[0].max(1e-12);

        let mut x_shifted = [0.0f64; 3];
        for i in 0..3 {
            if eigenvalues[i] > threshold {
                let proj = math::dot(eigenvectors[i], atb_shifted);
                let coeff = proj / eigenvalues[i];
                x_shifted[0] += eigenvectors[i][0] * coeff;
                x_shifted[1] += eigenvectors[i][1] * coeff;
                x_shifted[2] += eigenvectors[i][2] * coeff;
            }
        }

        let mut x = math::add(x_shifted, mass_point);

        for i in 0..3 {
            x[i] = x[i].clamp(cell_min[i], cell_max[i]);
        }

        let ata_x = math::sym_mul_vec(&self.ata, x);
        let error = self.btb - 2.0 * math::dot(x, self.atb) + math::dot(x, ata_x);

        (x, error.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SN: f64 = 0.01;
    const SP: f64 = 0.01;

    #[test]
    fn test_single_plane() {
        let mut qef = QefData::new();
        qef.add([0.0, 0.0, 0.5], [0.0, 0.0, 1.0], SN, SP);
        qef.add([1.0, 0.0, 0.5], [0.0, 0.0, 1.0], SN, SP);
        qef.add([0.0, 1.0, 0.5], [0.0, 0.0, 1.0], SN, SP);

        let (pos, _err) = qef.solve([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((pos[2] - 0.5).abs() < 0.01, "z={}", pos[2]);
    }

    #[test]
    fn test_sharp_edge() {
        let mut qef = QefData::new();
        qef.add([0.5, 0.0, 0.0], [1.0, 0.0, 0.0], SN, SP);
        qef.add([0.5, 1.0, 0.0], [1.0, 0.0, 0.0], SN, SP);
        qef.add([0.0, 0.5, 0.0], [0.0, 1.0, 0.0], SN, SP);
        qef.add([1.0, 0.5, 0.0], [0.0, 1.0, 0.0], SN, SP);

        let (pos, _err) = qef.solve([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((pos[0] - 0.5).abs() < 0.01, "x={}", pos[0]);
        assert!((pos[1] - 0.5).abs() < 0.01, "y={}", pos[1]);
    }

    #[test]
    fn test_sharp_corner() {
        let mut qef = QefData::new();
        qef.add([0.5, 0.0, 0.0], [1.0, 0.0, 0.0], SN, SP);
        qef.add([0.0, 0.5, 0.0], [0.0, 1.0, 0.0], SN, SP);
        qef.add([0.0, 0.0, 0.5], [0.0, 0.0, 1.0], SN, SP);

        let (pos, _err) = qef.solve([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        for i in 0..3 {
            assert!((pos[i] - 0.5).abs() < 0.01, "pos[{}]={}", i, pos[i]);
        }
    }

    #[test]
    fn test_single_point_no_panic() {
        // With probabilistic quadrics, even 1 sample produces a valid solve
        let mut qef = QefData::new();
        qef.add([0.3, 0.4, 0.5], [0.0, 0.0, 1.0], SN, SP);
        let (pos, err) = qef.solve([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite());
        assert!(err.is_finite());
        // z should be near 0.5 (the plane position)
        assert!((pos[2] - 0.5).abs() < 0.05, "z={}", pos[2]);
    }

    #[test]
    fn test_svd_vs_cholesky_agreement() {
        let mut qef = QefData::new();
        qef.add([0.5, 0.0, 0.0], [1.0, 0.0, 0.0], SN, SP);
        qef.add([0.5, 1.0, 0.0], [1.0, 0.0, 0.0], SN, SP);
        qef.add([0.0, 0.5, 0.0], [0.0, 1.0, 0.0], SN, SP);
        qef.add([1.0, 0.5, 0.0], [0.0, 1.0, 0.0], SN, SP);

        let bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let (pos_chol, _) = qef.solve(bounds.0, bounds.1);
        let (pos_svd, _) = qef.solve_svd(bounds.0, bounds.1);

        for i in 0..3 {
            assert!((pos_chol[i] - pos_svd[i]).abs() < 0.05,
                "axis {}: cholesky={} svd={}", i, pos_chol[i], pos_svd[i]);
        }
    }
}
