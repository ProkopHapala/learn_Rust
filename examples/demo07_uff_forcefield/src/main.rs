use std::path::PathBuf;

mod math;
mod util;
mod topology;
mod builder;
mod uff;
mod xyz;
mod nonbonded;
mod params;

fn main() {
    let mut args = std::env::args().skip(1);
    let xyz_path: PathBuf = args.next().expect("expected path to .xyz").into();
    let cutoff: f64 = args.next().map(|s| s.parse().expect("cutoff parse")).unwrap_or(1.8);

    let sys = xyz::read_xyz(&xyz_path).expect("read_xyz failed");

    let mut b = builder::Builder::from_positions_cutoff(&sys.apos, cutoff);
    let top = b.bake();
    let mut ff = uff::Uff::from_topology(&top);

    ff.make_neigh_bs();
    ff.bake_angle_neighs();
    ff.bake_dihedral_neighs();
    ff.bake_inversion_neighs();
    ff.map_atom_interactions();

    // Load forcefield parameters and setup forcefield via generic engine
    let mut p = params::Params::new();
    let datadir = "/home/prokop/git/FireCore/cpp/common_resources";
    p.load_element_types(format!("{}/ElementTypes.dat", datadir));
    p.load_atom_types(format!("{}/AtomTypes.dat", datadir));

    let mut nb = nonbonded::NonBondedFF::new(ff.natoms as usize);
    let _atypes = params::setup_forcefield(&mut ff, &mut nb, &p, &sys.elems);

    // Distort geometry before optimization to test recovery
    let dist_amp = 0.05;
    for ia in 0..ff.natoms as usize {
        ff.apos.as_mut_slice()[ia].x += dist_amp * (2.0 * (ia as f64 * 0.6180339887).fract() - 1.0);
        ff.apos.as_mut_slice()[ia].y += dist_amp * (2.0 * ((ia+1) as f64 * 0.6180339887).fract() - 1.0);
        ff.apos.as_mut_slice()[ia].z += dist_amp * (2.0 * ((ia+2) as f64 * 0.6180339887).fract() - 1.0);
    }
    ff.update_hneigh();

    // Optional PBC test: 10x10x10 A cubic box
    // nb.make_pbc_shifts([1,1,1], Vec3d::new(10.0,0.0,0.0), Vec3d::new(0.0,10.0,0.0), Vec3d::new(0.0,0.0,10.0));

    let trj_path = std::path::PathBuf::from("trj.xyz");
    let niter = 20000;
    let dt = 0.01;
    let fconv = 1e-3;
    let flim = 1000.0;
    let damping = 0.05;
    let save_every = 10;

    xyz::write_xyz_frame(&trj_path, &sys.elems, ff.apos.as_slice(), "step=0 E=0", false).expect("write trj failed");

    let mut converged = false;
    let f2conv = fconv * fconv;
    let cdamp = { let c = 1.0 - damping; if c < 0.0 { 0.0 } else { c } };
    for itr in 0..niter {
        let (eb, ea, ed, ei) = ff.eval_forces();
        let enb = nb.eval(ff.fapos.as_mut_slice(), ff.apos.as_slice());
        let etot = eb + ea + ed + ei + enb;
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
