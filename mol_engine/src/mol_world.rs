use mol_utils::math::vec3::{Vec3d, VEC3_ZERO};
use mol_topology::topology::Topology;
use crate::uff::Uff;
use crate::nonbonded::NonBondedFF;
use crate::surface::SurfaceFolded;

/// MolWorld orchestrates multiple forcefield engines for molecular dynamics.
/// Shared state (apos, fapos, vapos, reqs) lives in the Uff engine for efficiency.
/// Optional engines (nonbonded, surface) are evaluated after bonded forces.
pub struct MolWorld {
    pub uff: Uff,
    pub nonbonded: Option<NonBondedFF>,
    pub surface: Option<SurfaceFolded>,
    pub surface_atom_types: Vec<usize>,
}

impl MolWorld {
    pub fn from_topology(top: &Topology) -> Self {
        let uff = Uff::from_topology(top);
        Self::from_uff(uff)
    }

    pub fn from_uff(uff: Uff) -> Self {
        let natoms = uff.natoms as usize;
        Self {
            uff,
            nonbonded: None,
            surface: None,
            surface_atom_types: vec![0; natoms],
        }
    }

    pub fn natoms(&self) -> usize { self.uff.natoms as usize }

    /// Evaluate all forces, accumulating into uff.fapos.
    /// Returns: (eb, ea, ed, ei, enb, es) = (bond, angle, dihedral, inversion, nonbonded, surface)
    pub fn eval_forces(&mut self) -> (f64, f64, f64, f64, f64, f64) {
        // Bonded forces (zeros fapos, computes bonds/angles/dihedrals/inversions)
        let (eb, ea, ed, ei) = self.uff.eval_forces();
        let mut enb = 0.0;
        let mut es = 0.0;

        // Extract shared array references before optional engine borrows
        let natoms = self.natoms();
        let uff_fapos = self.uff.fapos.as_mut_slice();
        let uff_apos = self.uff.apos.as_slice();
        let uff_reqs = self.uff.reqs.as_slice();
        let sat = &self.surface_atom_types;

        // Non-bonded LJ + Coulomb
        if let Some(ref mut nb) = self.nonbonded {
            {
                let nb_reqs = nb.reqs.as_mut_slice();
                nb_reqs[0..natoms].copy_from_slice(&uff_reqs[0..natoms]);
            }
            enb = nb.eval(&mut uff_fapos[0..natoms], &uff_apos[0..natoms]);
        }

        // Surface interaction
        if let Some(ref surf) = self.surface {
            for ia in 0..natoms {
                let (e, f) = surf.eval_atom(uff_apos[ia], sat[ia], uff_reqs[ia]);
                es += e;
                uff_fapos[ia].add(f);
            }
        }

        (eb, ea, ed, ei, enb, es)
    }

    /// Single atom MD step. Returns (v·f, v·v, f·f) for convergence/instability checks.
    #[inline(always)]
    pub fn move_atom_md(&mut self, i: usize, dt: f64, flim: f64, cdamp: f64) -> (f64, f64, f64) {
        let f = self.uff.fapos.as_slice()[i];
        let v = self.uff.vapos.as_slice()[i];
        let p = self.uff.apos.as_slice()[i];
        let ff = v.dot(f);
        let vv = v.norm2();
        let f2 = f.norm2();
        let mut f_clamped = f;
        if f2 > flim * flim {
            f_clamped.mul(flim / f2.sqrt());
        }
        let mut v_new = v;
        v_new.mul(cdamp);
        v_new.add_mul(f_clamped, dt);
        let mut p_new = p;
        p_new.add_mul(v_new, dt);
        self.uff.apos.as_mut_slice()[i] = p_new;
        self.uff.vapos.as_mut_slice()[i] = v_new;
        (ff, vv, f2)
    }

    pub fn clean_velocity(&mut self) {
        for v in self.uff.vapos.as_mut_slice() { *v = VEC3_ZERO; }
    }

    /// Run MD for niter steps or until force convergence.
    pub fn run_md(&mut self, niter: i32, dt: f64, fconv: f64, flim: f64, damping: f64) -> i32 {
        let f2conv = fconv * fconv;
        let cdamp = { let c = 1.0 - damping; if c < 0.0 { 0.0 } else { c } };
        for itr in 0..niter {
            let (eb, ea, ed, ei, enb, es) = self.eval_forces();
            let _etot = eb + ea + ed + ei + enb + es;
            let mut ff = 0.0;
            let mut vv = 0.0;
            let mut vf = 0.0;
            for ia in 0..self.natoms() {
                let (ff_, vv_, vf_) = self.move_atom_md(ia, dt, flim, cdamp);
                ff += ff_;
                vv += vv_;
                vf += vf_;
            }
            if ff < 0.0 {
                self.clean_velocity();
            }
            if vf < f2conv {
                return itr + 1;
            }
        }
        niter
    }

    // === Topology setup wrappers (delegate to Uff) ===
    pub fn set_dummy_params(&mut self) { self.uff.set_dummy_params(); }
    pub fn make_neigh_bs(&mut self) { self.uff.make_neigh_bs(); }
    pub fn bake_angle_neighs(&mut self) { self.uff.bake_angle_neighs(); }
    pub fn bake_dihedral_neighs(&mut self) { self.uff.bake_dihedral_neighs(); }
    pub fn bake_inversion_neighs(&mut self) { self.uff.bake_inversion_neighs(); }
    pub fn map_atom_interactions(&mut self) { self.uff.map_atom_interactions(); }

    // === Convenience: attach surface ===
    pub fn setup_nacl_surface(&mut self, a: f64, z0: f64, beta_vdw: f64, q_amp: f64, plq_amp: f64) {
        self.surface = Some(crate::surface::setup_nacl_surface(a, z0, beta_vdw, q_amp, plq_amp));
        self.surface_atom_types.resize(self.natoms(), 0);
    }
}
