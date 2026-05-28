use mol_utils::math::vec3::{Vec3d, VEC3_ZERO};

const TAU: f64 = 6.283185307179586476925286766559;

/// Complex number for cos/sin optimization via complex multiplication
#[derive(Copy, Clone, Debug)]
struct Complex { re: f64, im: f64 }

impl Complex {
    #[inline(always)] fn new(re: f64, im: f64) -> Self { Self { re, im } }
    #[inline(always)] fn mul(self, other: Self) -> Self {
        Self { re: self.re * other.re - self.im * other.im, im: self.re * other.im + self.im * other.re }
    }
}

/// Precompute cos(n·φ), sin(n·φ) for n=0..nmax using complex recurrence.
/// n=0: cos=1, sin=0; n>=1: z^n where z=exp(i·φ), computed by recurrence.
/// Only 1 cos/sin call + nmax complex multiplies for all harmonics.
/// Output arrays must have length >= nmax+1.
#[inline] fn precompute_harmonics(phi: f64, nmax: usize, cos_vals: &mut [f64], sin_vals: &mut [f64]) {
    cos_vals[0] = 1.0;
    sin_vals[0] = 0.0;
    if nmax == 0 { return; }
    // z = exp(i·φ) = cos(φ) + i·sin(φ)  -- THE ONLY sin/cos CALL
    let z = Complex::new(phi.cos(), phi.sin());
    // Recurrence: z^(n+1) = z^n · z, read cos(nφ)=Re(z^n), sin(nφ)=Im(z^n)
    let mut zn = z;
    for i in 1..=nmax {
        cos_vals[i] = zn.re;
        sin_vals[i] = zn.im;
        zn = zn.mul(z);
    }
}

/// Check if all k values are integers (0,1,2,3...) -- complex recurrence applies.
#[inline] fn all_integer_harmonics(k: &[f64]) -> bool {
    for &ki in k.iter() {
        if ki < 0.0 { return false; }
        if (ki - ki.round()).abs() > 1e-12 { return false; }
    }
    true
}

/// Precompute separable 1D basis functions and derivatives for one atom position.
/// Core optimization: for integer harmonics, uses complex recurrence (1 cos/sin + max_k complex muls).
/// For non-integer k, falls back to direct cos/sin per basis.
#[inline] fn precompute_1d_bases(
    u: f64, v: f64, z: f64,
    kx: &[f64], ky: &[f64], kz: &[f64], z0: &[f64],
    bx: &mut [f64], by: &mut [f64], bz: &mut [f64],
    dbx: &mut [f64], dby: &mut [f64], dbz: &mut [f64],
) {
    let nx = kx.len();
    let ny = ky.len();
    let nz = kz.len();

    // --- X harmonics ---
    if all_integer_harmonics(kx) {
        let kx_int: Vec<usize> = kx.iter().map(|&ki| ki.round() as usize).collect();
        let kx_max = kx_int.iter().copied().max().unwrap_or(0);
        let mut cux = vec![0.0f64; kx_max + 1];
        let mut sux = vec![0.0f64; kx_max + 1];
        precompute_harmonics(TAU * u, kx_max, &mut cux, &mut sux);
        for i in 0..nx {
            let n = kx_int[i];
            bx[i] = cux[n];
            dbx[i] = -TAU * kx[i] * sux[n];
        }
    } else {
        for i in 0..nx {
            let phi = TAU * kx[i] * u;
            bx[i] = phi.cos();
            dbx[i] = -TAU * kx[i] * phi.sin();
        }
    }

    // --- Y harmonics ---
    if all_integer_harmonics(ky) {
        let ky_int: Vec<usize> = ky.iter().map(|&ki| ki.round() as usize).collect();
        let ky_max = ky_int.iter().copied().max().unwrap_or(0);
        let mut cvy = vec![0.0f64; ky_max + 1];
        let mut svy = vec![0.0f64; ky_max + 1];
        precompute_harmonics(TAU * v, ky_max, &mut cvy, &mut svy);
        for i in 0..ny {
            let n = ky_int[i];
            by[i] = cvy[n];
            dby[i] = -TAU * ky[i] * svy[n];
        }
    } else {
        for i in 0..ny {
            let phi = TAU * ky[i] * v;
            by[i] = phi.cos();
            dby[i] = -TAU * ky[i] * phi.sin();
        }
    }

    // --- Z decay (exponential, each independent) ---
    for i in 0..nz {
        let dz = (z - z0[i]).max(0.0);
        let bz_i = (-kz[i] * dz).exp();
        bz[i] = bz_i;
        dbz[i] = if z > z0[i] { -kz[i] * bz_i } else { 0.0 };
    }
}

/// Folded surface potential with tensor-product separable basis.
/// Uses complex recurrence for harmonics: O(1) sin/cos per dimension instead of O(N_harmonics).
pub struct SurfaceFolded {
    // 2D lattice vectors (ax,bx,ay,by) where a=(ax,ay), b=(bx,by)
    pub ax: f64, pub bx: f64, pub ay: f64, pub by: f64,
    // Inverse lattice
    pub inv_ax: f64, pub inv_bx: f64,
    pub inv_ay: f64, pub inv_by: f64,
    // 1D harmonic frequencies [N_x], [N_y] and z-decay params [N_z]
    pub kx: Vec<f64>,
    pub ky: Vec<f64>,
    pub kz: Vec<f64>,
    pub z0: Vec<f64>,
    pub nx: usize, pub ny: usize, pub nz: usize,
    // Coefficients [natom_types * (nx*ny*nz)] for each interaction channel
    pub coeffs_q: Vec<f64>,   // electrostatics (charge)
    pub coeffs_p: Vec<f64>,  // Pauli repulsion
    pub coeffs_l: Vec<f64>,  // London dispersion
    pub ntypes: usize,
}

impl SurfaceFolded {
    pub fn new(ax: f64, bx: f64, ay: f64, by: f64,
               kx: Vec<f64>, ky: Vec<f64>, kz: Vec<f64>, z0: Vec<f64>, ntypes: usize) -> Self {
        let det = ax * by - bx * ay;
        assert!(det.abs() > 1e-12, "SurfaceFolded: degenerate 2D lattice");
        let idet = 1.0 / det;
        let nx = kx.len();
        let ny = ky.len();
        let nz = kz.len();
        let nbasis = nx * ny * nz;
        let ncoef = ntypes * nbasis;
        Self {
            ax, bx, ay, by,
            inv_ax: by * idet,  inv_bx: -bx * idet,
            inv_ay: -ay * idet, inv_by: ax * idet,
            kx, ky, kz, z0, nx, ny, nz,
            coeffs_q: vec![0.0; ncoef],
            coeffs_p: vec![0.0; ncoef],
            coeffs_l: vec![0.0; ncoef],
            ntypes,
        }
    }

    #[inline(always)] pub fn nbasis(&self) -> usize { self.nx * self.ny * self.nz }

    /// Set coefficients for atom type `ityp` and basis index `ib` (flattened ix,iy,iz)
    #[inline(always)] pub fn set_coeffs(&mut self, ityp: usize, ib: usize, q: f64, p: f64, l: f64) {
        let ioff = ityp * self.nbasis() + ib;
        self.coeffs_q[ioff] = q;
        self.coeffs_p[ioff] = p;
        self.coeffs_l[ioff] = l;
    }

    /// Convert REQ [RvdW, EvdW, Q, H] → PLQ [Pauli, London, Q, H]
    #[inline(always)] pub fn req2plq(req: [f64; 4], alpha: f64) -> [f64; 4] {
        let k = -alpha;
        let e = (k * req[0]).exp();
        let c_l = e * req[1].sqrt();
        let c_p = e * c_l;
        [c_p, c_l, req[2], req[3]]
    }

    /// Evaluate energy and force for one atom.
    /// Steps: 1) precompute 1D bases, 2) tensor-product accumulate with coefficients.
    #[inline(always)] pub fn eval_atom(&self, pos: Vec3d, ityp: usize, req: [f64; 4]) -> (f64, Vec3d) {
        if ityp >= self.ntypes { return (0.0, VEC3_ZERO); }

        // Fractional coordinates
        let u = self.inv_ax * pos.x + self.inv_ay * pos.y;
        let v = self.inv_bx * pos.x + self.inv_by * pos.y;
        let u = u - u.floor();
        let v = v - v.floor();

        // Precompute all 1D bases (THE OPTIMIZATION)
        let mut bx  = vec![0.0f64; self.nx];  let mut dbx = vec![0.0f64; self.nx];
        let mut by  = vec![0.0f64; self.ny];  let mut dby = vec![0.0f64; self.ny];
        let mut bz  = vec![0.0f64; self.nz];  let mut dbz = vec![0.0f64; self.nz];
        precompute_1d_bases(u, v, pos.z, &self.kx, &self.ky, &self.kz, &self.z0,
                            &mut bx, &mut by, &mut bz, &mut dbx, &mut dby, &mut dbz);

        let plq = Self::req2plq(req, 2.0);
        let ioff = ityp * self.nbasis();
        let mut ic = ioff;

        let mut e_tot = 0.0;
        let mut dEdu = 0.0;
        let mut dEdv = 0.0;
        let mut dEdz = 0.0;

        // Triple loop: tensor product combination
        for iz in 0..self.nz {
            let bz_iz = bz[iz];
            let dbz_iz = dbz[iz];
            for iy in 0..self.ny {
                let by_iy = by[iy];
                let dby_iy = dby[iy];
                // Precompute z-y combos
                let bz_by = bz_iz * by_iy;
                let dbz_by = dbz_iz * by_iy;
                let bz_dby = bz_iz * dby_iy;
                for ix in 0..self.nx {
                    let c = self.coeffs_q[ic] * req[2]
                        + self.coeffs_p[ic] * plq[0]
                        + self.coeffs_l[ic] * plq[1];
                    ic += 1;
                    let bx_ix = bx[ix];
                    let dbx_ix = dbx[ix];
                    e_tot  += c * (bx_ix * bz_by);
                    dEdu   += c * (dbx_ix * bz_by);
                    dEdv   += c * (bx_ix * bz_dby);
                    dEdz   += c * (bx_ix * dbz_by);
                }
            }
        }

        let fx = -(dEdu * self.inv_ax + dEdv * self.inv_bx);
        let fy = -(dEdu * self.inv_ay + dEdv * self.inv_by);
        let fz = -dEdz;

        (e_tot, Vec3d::new(fx, fy, fz))
    }

    /// Evaluate for all atoms, accumulate forces into `fapos`
    pub fn eval_all(&self, apos: &[Vec3d], atom_types: &[usize], reqs: &[[f64; 4]], fapos: &mut [Vec3d]) -> f64 {
        let mut etot = 0.0;
        for ia in 0..apos.len() {
            let (e, f) = self.eval_atom(apos[ia], atom_types[ia], reqs[ia]);
            etot += e;
            fapos[ia].add(f);
        }
        etot
    }
}

/// Create a NaCl-like substrate surface with:
/// - Electrostatics: checkerboard of alternating charges using cos(2π*x/a)*cos(2π*y/a)
/// - VdW: exponentially decaying attraction above surface (z > z0)
/// - Lattice: square a×a (ax=a, bx=0, ay=0, by=a)
pub fn setup_nacl_surface(a: f64, z0: f64, beta_vdw: f64, q_amp: f64, plq_amp: f64) -> SurfaceFolded {
    // Basis: [kx=1.0 (cos 2πx/a), ky=1.0 (cos 2πy/a)], z-decay [beta_vdw], z0
    // k=1.0 because TAU*1.0*x/a = 2π*x/a  -> period a/2, matching Na-Cl spacing
    let kx = vec![0.0, 1.0];  // 0: constant, 1.0: cos(2π*x/a)
    let ky = vec![0.0, 1.0];  // 0: constant, 1.0: cos(2π*y/a)
    let kz = vec![beta_vdw];
    let z0s = vec![z0];
    let nx = kx.len();
    let ny = ky.len();
    let nz = kz.len();

    let mut surf = SurfaceFolded::new(a, 0.0, 0.0, a, kx, ky, kz, z0s, 1);

    // Basis index mapping: ib = ix + nx*(iy + ny*iz)
    // ix=0,iy=0,iz=0: constant (all ones)
    // ix=1,iy=0,iz=0: cos(2πx/a)
    // ix=0,iy=1,iz=0: cos(2πy/a)
    // ix=1,iy=1,iz=0: cos(2πx/a)*cos(2πy/a)  <- checkerboard pattern

    let i_const = 0 + nx * (0 + ny * 0);    // (0,0,0)
    let i_cos_x = 1 + nx * (0 + ny * 0);    // (1,0,0)
    let i_cos_y = 0 + nx * (1 + ny * 0);    // (0,1,0)
    let i_check = 1 + nx * (1 + ny * 0);    // (1,1,0) checkerboard

    // Type 0: generic atom interacting with NaCl surface
    // Electrostatics: checkerboard pattern (alternating + and -)
    surf.set_coeffs(0, i_const, 0.0, 0.0, 0.0);      // no constant electrostatic
    surf.set_coeffs(0, i_cos_x, 0.0, 0.0, 0.0);      // no pure x-cos
    surf.set_coeffs(0, i_cos_y, 0.0, 0.0, 0.0);      // no pure y-cos
    surf.set_coeffs(0, i_check, q_amp, 0.0, 0.0);    // checkerboard charges

    // Pauli/London: smooth vdW wall (attractive above, repulsive when penetrating)
    // NOTE: set_coeffs overwrites (q,p,l) at once, so we must preserve q_amp on i_check.
    // Pauli (repulsive): dominates when close to surface / penetrating
    surf.set_coeffs(0, i_const, 0.0, plq_amp, 0.0);       // constant Pauli repulsion
    surf.set_coeffs(0, i_cos_x, 0.0, 0.0, 0.0);
    surf.set_coeffs(0, i_cos_y, 0.0, 0.0, 0.0);
    surf.set_coeffs(0, i_check, q_amp, 0.0, 0.0);         // keep electrostatics

    // London (attractive): decays above surface
    surf.set_coeffs(0, i_const, 0.0, 0.0, -plq_amp);      // attractive constant
    surf.set_coeffs(0, i_cos_x, 0.0, 0.0, 0.0);
    surf.set_coeffs(0, i_cos_y, 0.0, 0.0, 0.0);
    surf.set_coeffs(0, i_check, q_amp, 0.0, 0.0);         // keep electrostatics

    surf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonics_recurrence() {
        let phi = 0.7;
        let nmax = 5;
        let mut cos_vals = vec![0.0; nmax + 1];  // n=0..nmax
        let mut sin_vals = vec![0.0; nmax + 1];
        precompute_harmonics(phi, nmax, &mut cos_vals, &mut sin_vals);

        // n=0: cos=1, sin=0
        assert!((cos_vals[0] - 1.0).abs() < 1e-12, "cos(0) should be 1");
        assert!(sin_vals[0].abs() < 1e-12, "sin(0) should be 0");

        // n=1..nmax
        for n in 1..=nmax {
            let expected_cos = (n as f64 * phi).cos();
            let expected_sin = (n as f64 * phi).sin();
            assert!((cos_vals[n] - expected_cos).abs() < 1e-12, "cos({}*{}) mismatch: {} vs {}", n, phi, cos_vals[n], expected_cos);
            assert!((sin_vals[n] - expected_sin).abs() < 1e-12, "sin({}*{}) mismatch: {} vs {}", n, phi, sin_vals[n], expected_sin);
        }
    }

    #[test]
    fn test_surface_eval_constant() {
        // 10x10 square lattice, one constant basis (nx=0 harmonic=1, ny=0 harmonic=1, z-constant)
        let mut surf = SurfaceFolded::new(
            10.0, 0.0, 0.0, 10.0,
            vec![0.0], vec![0.0], vec![0.0], vec![0.0], // kx=[0], ky=[0], kz=[0], z0=[0]
            1
        );
        // Constant basis: bx=cos(0)=1, by=cos(0)=1, bz=exp(0)=1 → basis=1 everywhere
        surf.set_coeffs(0, 0, 1.0, 0.0, 0.0); // Q-coeff=1

        let req = [3.0, 0.1, 0.5, 0.0]; // Q=0.5
        let pos = Vec3d::new(3.0, 4.0, 1.0);
        let (e, f) = surf.eval_atom(pos, 0, req);

        // E = c_q * Q * basis = 1.0 * 0.5 * 1.0 = 0.5
        assert!((e - 0.5).abs() < 1e-10, "constant basis energy: {} vs 0.5", e);
        // Force should be zero for constant potential
        assert!(f.norm2() < 1e-10, "constant basis force should be zero: {:?}", f);
    }

    #[test]
    fn test_surface_eval_cos_x() {
        // cos(2π*x/10) basis in x, constant in y and z
        let mut surf = SurfaceFolded::new(
            10.0, 0.0, 0.0, 10.0,
            vec![1.0], vec![0.0], vec![0.0], vec![0.0],
            1
        );
        surf.set_coeffs(0, 0, 1.0, 0.0, 0.0);

        let req = [3.0, 0.1, 1.0, 0.0]; // Q=1.0

        // At x=0: cos(0) = 1
        let (e0, _) = surf.eval_atom(Vec3d::new(0.0, 0.0, 0.0), 0, req);
        assert!((e0 - 1.0).abs() < 1e-10, "cos(0) should be 1, got {}", e0);

        // At x=5: cos(π) = -1
        let (e5, _) = surf.eval_atom(Vec3d::new(5.0, 0.0, 0.0), 0, req);
        assert!((e5 - (-1.0)).abs() < 1e-10, "cos(π) should be -1, got {}", e5);

        // At x=2.5: cos(π/2) = 0
        let (e25, _) = surf.eval_atom(Vec3d::new(2.5, 0.0, 0.0), 0, req);
        assert!(e25.abs() < 1e-10, "cos(π/2) should be 0, got {}", e25);
    }

    #[test]
    fn test_surface_eval_z_decay() {
        // Constant in x,y, exponential decay in z
        let mut surf = SurfaceFolded::new(
            10.0, 0.0, 0.0, 10.0,
            vec![0.0], vec![0.0], vec![1.0], vec![0.0], // kz=1.0, z0=0.0
            1
        );
        surf.set_coeffs(0, 0, 1.0, 0.0, 0.0);

        let req = [3.0, 0.1, 1.0, 0.0];

        // At z=1: exp(-1.0 * 1.0) = exp(-1)
        let (e, _) = surf.eval_atom(Vec3d::new(0.0, 0.0, 1.0), 0, req);
        let expected = (-1.0_f64).exp();
        assert!((e - expected).abs() < 1e-10, "z decay: {} vs {}", e, expected);
    }

    #[test]
    fn test_req2plq() {
        let req = [3.0, 0.1, 0.5, 0.0]; // R=3, E=0.1, Q=0.5
        let plq = SurfaceFolded::req2plq(req, 2.0);
        let e = (-6.0_f64).exp();
        assert!((plq[0] - e * e * 0.1_f64.sqrt()).abs() < 1e-10, "Pauli mismatch");
        assert!((plq[1] - e * 0.1_f64.sqrt()).abs() < 1e-10, "London mismatch");
        assert!((plq[2] - 0.5).abs() < 1e-10, "Q should pass through");
    }
}
