use crate::util::math::{self, Sym3x3};

/// Compact QEF representation: 14 floats, additively composable.
///
/// Represents the quadric error E(x) = Σ (n_i · (x - p_i))^2
/// accumulated from Hermite data (intersection point p_i, normal n_i).
#[derive(Clone, Copy, Debug)]
pub struct QefData {
    /// Upper triangle of A^T A (symmetric 3x3).
    pub ata: Sym3x3,
    /// A^T b vector.
    pub atb: [f64; 3],
    /// b^T b scalar (for error computation).
    pub btb: f64,
    /// Mass point accumulator (sum of intersection points).
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

    /// Add a Hermite data point (intersection position + normal).
    pub fn add(&mut self, pos: [f64; 3], normal: [f64; 3]) {
        let n = normal;
        let d = math::dot(n, pos);

        // A^T A += n * n^T (outer product, symmetric)
        self.ata[0] += n[0] * n[0];
        self.ata[1] += n[0] * n[1];
        self.ata[2] += n[0] * n[2];
        self.ata[3] += n[1] * n[1];
        self.ata[4] += n[1] * n[2];
        self.ata[5] += n[2] * n[2];

        // A^T b += n * d
        self.atb[0] += n[0] * d;
        self.atb[1] += n[1] * d;
        self.atb[2] += n[2] * d;

        // b^T b += d^2
        self.btb += d * d;

        // Mass point
        self.mass_point_sum[0] += pos[0];
        self.mass_point_sum[1] += pos[1];
        self.mass_point_sum[2] += pos[2];
        self.count += 1;
    }

    /// Solve the QEF with mass-point bias for rank-deficient cases.
    ///
    /// Returns (position, error).
    /// If the solution would be outside `cell_min`..`cell_max`, it is clamped.
    pub fn solve(&self, cell_min: [f64; 3], cell_max: [f64; 3]) -> ([f64; 3], f64) {
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

        // Shift system relative to mass_point: atb_shifted = atb - ata * mass_point
        let ata_mp = math::sym_mul_vec(&self.ata, mass_point);
        let atb_shifted = math::sub(self.atb, ata_mp);

        // Eigendecompose A^T A
        let (eigenvalues, eigenvectors) = math::jacobi_eigen_3x3(&self.ata);

        // Threshold for rank determination
        let threshold = 0.1 * eigenvalues[0].max(1e-12);

        // Build solution: x_shifted = Σ (v_i^T * atb_shifted / λ_i) * v_i
        // For rank-deficient directions (λ_i < threshold), use 0 (falls back to mass_point)
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

        // Clamp to cell bounds
        for i in 0..3 {
            x[i] = x[i].clamp(cell_min[i], cell_max[i]);
        }

        // Compute error: btb - 2 * x^T * atb + x^T * ata * x
        let ata_x = math::sym_mul_vec(&self.ata, x);
        let error = self.btb - 2.0 * math::dot(x, self.atb) + math::dot(x, ata_x);

        (x, error.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_plane() {
        // Plane at z=0.5 with normal [0,0,1]
        let mut qef = QefData::new();
        qef.add([0.0, 0.0, 0.5], [0.0, 0.0, 1.0]);
        qef.add([1.0, 0.0, 0.5], [0.0, 0.0, 1.0]);
        qef.add([0.0, 1.0, 0.5], [0.0, 0.0, 1.0]);

        let (pos, _err) = qef.solve([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // z should be ~0.5, x and y fall back to mass point
        assert!((pos[2] - 0.5).abs() < 0.01, "z={}", pos[2]);
    }

    #[test]
    fn test_sharp_edge() {
        // Two planes meeting at a 90-degree edge along Z-axis at x=0.5
        let mut qef = QefData::new();
        // Plane 1: x=0.5, normal [1,0,0]
        qef.add([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        qef.add([0.5, 1.0, 0.0], [1.0, 0.0, 0.0]);
        // Plane 2: y=0.5, normal [0,1,0]
        qef.add([0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
        qef.add([1.0, 0.5, 0.0], [0.0, 1.0, 0.0]);

        let (pos, _err) = qef.solve([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        // Edge should be at x=0.5, y=0.5
        assert!((pos[0] - 0.5).abs() < 0.01, "x={}", pos[0]);
        assert!((pos[1] - 0.5).abs() < 0.01, "y={}", pos[1]);
    }

    #[test]
    fn test_sharp_corner() {
        // Three orthogonal planes meeting at (0.5, 0.5, 0.5)
        let mut qef = QefData::new();
        qef.add([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]);
        qef.add([0.0, 0.5, 0.0], [0.0, 1.0, 0.0]);
        qef.add([0.0, 0.0, 0.5], [0.0, 0.0, 1.0]);

        let (pos, _err) = qef.solve([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        for i in 0..3 {
            assert!((pos[i] - 0.5).abs() < 0.01, "pos[{}]={}", i, pos[i]);
        }
    }
}
