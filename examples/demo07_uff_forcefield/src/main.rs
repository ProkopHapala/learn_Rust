use std::path::PathBuf;

mod math;
mod util;
mod topology;
mod builder;
mod uff;
mod xyz;

fn main() {
    let mut args = std::env::args().skip(1);
    let xyz_path: PathBuf = args.next().expect("expected path to .xyz").into();

    let sys = xyz::read_xyz(&xyz_path).expect("read_xyz failed");

    let mut b = builder::Builder::from_positions_cutoff(&sys.apos, 1.8);
    let top = b.bake();
    let mut ff = uff::Uff::from_topology(&top);

    ff.make_neigh_bs();
    ff.bake_angle_neighs();
    ff.bake_dihedral_neighs();
    ff.bake_inversion_neighs();
    ff.map_atom_interactions();

    ff.set_dummy_params();

    let trj_path = std::path::PathBuf::from("trj.xyz");
    let niter = 1000;
    let dt = 0.05;
    let fconv = 1e-6;
    let flim = 1000.0;
    let damping = 0.1;
    let save_every = 10;

    xyz::write_xyz_frame(&trj_path, &sys.elems, ff.apos.as_slice(), "step=0 E=0", false).expect("write trj failed");

    let mut converged = false;
    let f2conv = fconv * fconv;
    let cdamp = { let c = 1.0 - damping; if c < 0.0 { 0.0 } else { c } };
    for itr in 0..niter {
        let (eb, ea, ed, ei) = ff.eval_forces();
        let etot = eb + ea + ed + ei;
        let mut ff_ = 0.0;
        let mut vf = 0.0;
        for ia in 0..ff.natoms as usize {
            let (ff_i, _vv, vf_i) = ff.move_atom_md(ia, dt, flim, cdamp);
            ff_ += ff_i;
            vf += vf_i;
        }
        if ff_ < 0.0 { ff.clean_velocity(); }
        if vf < f2conv { converged = true; println!("MD converged at step {}", itr + 1); break; }
        if itr % save_every == 0 {
            xyz::write_xyz_frame(&trj_path, &sys.elems, ff.apos.as_slice(), &format!("step={} E={:.4e}", itr + 1, etot), true).expect("write trj failed");
        }
    }
    if !converged { println!("MD NOT converged in {} steps", niter); }

    let fnorm: f64 = ff.fapos.as_slice().iter().map(|f| f.norm2()).sum::<f64>().sqrt();
    println!("natoms={} nbonds={} nangles={} ndihedrals={} ninversions={} |F|={}", ff.natoms, ff.nbonds, ff.nangles, ff.ndihedrals, ff.ninversions, fnorm);
}
