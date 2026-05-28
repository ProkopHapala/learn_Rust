use std::path::PathBuf;

// Smoke test using new mol_* crates
use mol_utils::xyz;
use mol_topology::builder;
use mol_topology::topology;
use mol_engine::mol_world::MolWorld;

fn main() {
    let mut args = std::env::args().skip(1);
    let xyz_path: PathBuf = args.next().expect("expected path to .xyz").into();
    let cutoff: f64 = args.next().map(|s| s.parse().expect("cutoff parse")).unwrap_or(1.8);

    let sys = xyz::read_xyz(&xyz_path).expect("read_xyz failed");

    let mut b = builder::Builder::from_positions_cutoff(&sys.apos, cutoff);
    let top = b.bake();
    let mut world = MolWorld::from_topology(&top);

    world.make_neigh_bs();
    world.bake_angle_neighs();
    world.bake_dihedral_neighs();
    world.bake_inversion_neighs();
    world.map_atom_interactions();

    // Set dummy parameters for smoke test
    world.set_dummy_params();

    // Distort geometry before optimization to test recovery
    let dist_amp = 0.05;
    for ia in 0..world.natoms() {
        world.uff.apos.as_mut_slice()[ia].x += dist_amp * (2.0 * (ia as f64 * 0.6180339887).fract() - 1.0);
        world.uff.apos.as_mut_slice()[ia].y += dist_amp * (2.0 * ((ia+1) as f64 * 0.6180339887).fract() - 1.0);
        world.uff.apos.as_mut_slice()[ia].z += dist_amp * (2.0 * ((ia+2) as f64 * 0.6180339887).fract() - 1.0);
    }
    world.uff.update_hneigh();

    let trj_path = std::path::PathBuf::from("trj.xyz");
    let niter = 20000;
    let dt = 0.01;
    let fconv = 1e-3;
    let flim = 1000.0;
    let damping = 0.05;
    let save_every = 10;

    xyz::write_xyz_frame(&trj_path, &sys.elems, world.uff.apos.as_slice(), "step=0 E=0", false).expect("write trj failed");

    let mut converged = false;
    let f2conv = fconv * fconv;
    let cdamp = { let c = 1.0 - damping; if c < 0.0 { 0.0 } else { c } };
    for itr in 0..niter {
        let (eb, ea, ed, ei, _enb, _es) = world.eval_forces();
        let etot = eb + ea + ed + ei + _enb + _es;
        let mut ff_ = 0.0;
        let mut vf = 0.0;
        for ia in 0..world.natoms() {
            let (ff_i, _vv, vf_i) = world.move_atom_md(ia, dt, flim, cdamp);
            ff_ += ff_i;
            vf += vf_i;
        }
        if ff_ < 0.0 { world.clean_velocity(); }
        if vf < f2conv { converged = true; println!("MD converged at step {}", itr + 1); break; }
        if itr % save_every == 0 {
            xyz::write_xyz_frame(&trj_path, &sys.elems, world.uff.apos.as_slice(), &format!("step={} E={:.4e}", itr + 1, etot), true).expect("write trj failed");
        }
    }
    if !converged { println!("MD NOT converged in {} steps", niter); }

    let fnorm: f64 = world.uff.fapos.as_slice().iter().map(|f| f.norm2()).sum::<f64>().sqrt();
    let uff = &world.uff;
    println!("natoms={} nbonds={} nangles={} ndihedrals={} ninversions={} |F|={}", uff.natoms, uff.nbonds, uff.nangles, uff.ndihedrals, uff.ninversions, fnorm);
}
