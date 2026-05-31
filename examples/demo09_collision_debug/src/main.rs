use macroquad::prelude::*;
use mol_utils::xyz;
use mol_utils::math::vec3::Vec3d;
use mol_topology::builder;
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug)]
struct Aabb { min: Vec3, max: Vec3 }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PairKind { SkippedBonded, Collide, Considered }

#[derive(Copy, Clone, Debug)]
struct PairVis { i: usize, j: usize, kind: PairKind }

fn aabb_overlap(a: Aabb, b: Aabb, margin: f32) -> bool {
    a.max.x + margin >= b.min.x && a.min.x <= b.max.x + margin && a.max.y + margin >= b.min.y && a.min.y <= b.max.y + margin && a.max.z + margin >= b.min.z && a.min.z <= b.max.z + margin
}

fn point_aabb_dist2(p: Vec3, b: Aabb) -> f32 {
    let dx = (b.min.x - p.x).max(0.0).max(p.x - b.max.x);
    let dy = (b.min.y - p.y).max(0.0).max(p.y - b.max.y);
    let dz = (b.min.z - p.z).max(0.0).max(p.z - b.max.z);
    dx*dx + dy*dy + dz*dz
}

fn parse_arg_f32(args: &[String], key: &str, default: f32) -> f32 {
    for w in args.windows(2) {
        if w[0] == key { return w[1].parse::<f32>().unwrap_or(default); }
    }
    default
}

fn parse_arg_usize(args: &[String], key: &str, default: usize) -> usize {
    for w in args.windows(2) {
        if w[0] == key { return w[1].parse::<usize>().unwrap_or(default); }
    }
    default
}

fn parse_arg_path(args: &[String]) -> Option<PathBuf> {
    for a in args.iter().skip(1) {
        if !a.starts_with("--") { return Some(PathBuf::from(a)); }
    }
    None
}

fn color_group(g: usize) -> Color {
    let t = (g as f32 * 0.271828).fract();
    Color::new(0.2 + 0.7*t, 0.2 + 0.7*(1.0-t), 0.5 + 0.4*(t-0.5).abs(), 1.0)
}

fn build_bond_set(bonds: &[[i32; 2]]) -> HashSet<(usize, usize)> {
    let mut set = HashSet::new();
    for b in bonds {
        let i = b[0] as usize;
        let j = b[1] as usize;
        let (a, c) = if i < j { (i, j) } else { (j, i) };
        set.insert((a, c));
    }
    set
}

fn compute_bboxes(pos: &[Vec3], rad: &[f32], group_size: usize) -> Vec<Aabb> {
    let ng = (pos.len() + group_size - 1) / group_size;
    let mut out = vec![Aabb { min: vec3(1e10, 1e10, 1e10), max: vec3(-1e10, -1e10, -1e10) }; ng];
    for g in 0..ng {
        let i0 = g * group_size;
        let i1 = ((g + 1) * group_size).min(pos.len());
        let mut mn = vec3(1e10, 1e10, 1e10);
        let mut mx = vec3(-1e10, -1e10, -1e10);
        for i in i0..i1 {
            let p = pos[i];
            let r = rad[i];
            mn.x = mn.x.min(p.x - r); mn.y = mn.y.min(p.y - r); mn.z = mn.z.min(p.z - r);
            mx.x = mx.x.max(p.x + r); mx.y = mx.y.max(p.y + r); mx.z = mx.z.max(p.z + r);
        }
        out[g] = Aabb { min: mn, max: mx };
    }
    out
}

fn build_halo_lists(pos: &[Vec3], bboxes: &[Aabb], group_size: usize, max_ghosts: usize, bbox_margin: f32, margin_sq: f32) -> (Vec<usize>, Vec<usize>) {
    let ng = bboxes.len();
    let mut ghost_counts = vec![0usize; ng];
    let mut ghost_flat = vec![usize::MAX; ng * max_ghosts];

    for g in 0..ng {
        let myb = bboxes[g];
        let mut cnt = 0usize;
        for og in 0..ng {
            if og == g { continue; }
            if !aabb_overlap(myb, bboxes[og], bbox_margin) { continue; }
            let j0 = og * group_size;
            let j1 = ((og + 1) * group_size).min(pos.len());
            for j in j0..j1 {
                if cnt >= max_ghosts { break; }
                let d2 = point_aabb_dist2(pos[j], myb);
                if d2 < margin_sq {
                    ghost_flat[g * max_ghosts + cnt] = j;
                    cnt += 1;
                }
            }
            if cnt >= max_ghosts { break; }
        }
        ghost_counts[g] = cnt;
    }

    (ghost_flat, ghost_counts)
}

fn compute_pairs(pos: &[Vec3], rad: &[f32], bonds: &HashSet<(usize, usize)>, group_size: usize, max_ghosts: usize, ghost_flat: &[usize], ghost_counts: &[usize]) -> Vec<PairVis> {
    let ng = ghost_counts.len();
    let mut pairs = Vec::<PairVis>::new();

    for g in 0..ng {
        let i0 = g * group_size;
        let i1 = ((g + 1) * group_size).min(pos.len());
        let gcount = ghost_counts[g];
        for i in i0..i1 {
            for k in 0..gcount {
                let j = ghost_flat[g * max_ghosts + k];
                if j == usize::MAX || j == i { continue; }
                let (a, c) = if i < j { (i, j) } else { (j, i) };
                if bonds.contains(&(a, c)) {
                    pairs.push(PairVis { i, j, kind: PairKind::SkippedBonded });
                    continue;
                }
                let d = pos[j] - pos[i];
                let rsum = rad[i] + rad[j];
                let d2 = d.length_squared();
                if d2 < rsum*rsum && d2 > 1e-12 {
                    pairs.push(PairVis { i, j, kind: PairKind::Collide });
                } else {
                    pairs.push(PairVis { i, j, kind: PairKind::Considered });
                }
            }
        }
    }

    pairs
}

fn draw_aabb(b: Aabb, col: Color) {
    let c = 0.5 * (b.min + b.max);
    let s = b.max - b.min;
    draw_cube_wires(c, s, col);
}

fn draw_pair(pi: Vec3, pj: Vec3, k: PairKind) {
    let col = match k {
        PairKind::SkippedBonded => (GRAY, 1.0),
        PairKind::Considered => (Color::new(0.2, 0.8, 0.9, 0.35), 1.0),
        PairKind::Collide => (Color::new(1.0, 0.2, 0.2, 1.0), 3.0),
    }.0;
    draw_line_3d(pi, pj, col);
}

fn window_conf() -> Conf {
    Conf { window_title: "demo09_collision_debug".to_owned(), window_width: 1280, window_height: 800, ..Default::default() }
}

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    let group_size = parse_arg_usize(&args, "--group-size", 32);
    let max_ghosts = parse_arg_usize(&args, "--max-ghosts", 64);
    let bbox_margin = parse_arg_f32(&args, "--bbox-margin", 0.5);
    let margin = parse_arg_f32(&args, "--margin", 1.5);
    let atom_r = parse_arg_f32(&args, "--atom-r", 0.6);

    let workspace_root = std::env::current_dir().unwrap()
        .ancestors().nth(2)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let xyz_path = parse_arg_path(&args).unwrap_or_else(|| workspace_root.join("examples/demo07_uff_forcefield/water.xyz"));

    println!("XYZ: {:?}", xyz_path);
    println!("group_size={} max_ghosts={} bbox_margin={} margin={} atom_r={}", group_size, max_ghosts, bbox_margin, margin, atom_r);

    let sys = xyz::read_xyz(&xyz_path).expect("read_xyz failed");

    let apos_v3d: Vec<Vec3d> = sys.apos.clone();
    let radii: Vec<f64> = sys.elems.iter().map(|el| match el.as_str() { "H" => 0.31, "C" => 0.76, "N" => 0.71, "O" => 0.66, _ => 1.0 }).collect();
    let mut b = builder::Builder::from_positions_and_radii(&apos_v3d, &radii, 0.4);
    let top = b.bake();
    let bonds_set = build_bond_set(&top.bonds);

    let pos: Vec<Vec3> = top.apos.iter().map(|p| vec3(p.x as f32, p.y as f32, p.z as f32)).collect();
    let rad: Vec<f32> = vec![atom_r; pos.len()];

    let cam = Camera3D { position: vec3(0.0, 5.0, 10.0), up: vec3(0.0, 1.0, 0.0), target: vec3(0.0, 0.0, 0.0), ..Default::default() };

    let mut show_pairs = true;
    let mut show_ghosts = true;
    let mut show_aabbs = true;

    loop {
        if is_key_pressed(KeyCode::Key1) { show_aabbs = !show_aabbs; }
        if is_key_pressed(KeyCode::Key2) { show_ghosts = !show_ghosts; }
        if is_key_pressed(KeyCode::Key3) { show_pairs = !show_pairs; }

        let bboxes = compute_bboxes(&pos, &rad, group_size);
        let margin_sq = margin * margin;
        let (ghost_flat, ghost_counts) = build_halo_lists(&pos, &bboxes, group_size, max_ghosts, bbox_margin, margin_sq);
        let pairs = compute_pairs(&pos, &rad, &bonds_set, group_size, max_ghosts, &ghost_flat, &ghost_counts);

        set_camera(&cam);
        clear_background(BLACK);

        for (i, p) in pos.iter().enumerate() {
            let col = if i == 0 { YELLOW } else { WHITE };
            draw_sphere(*p, rad[i], None, col);
        }

        if show_aabbs {
            for (g, b) in bboxes.iter().enumerate() {
                draw_aabb(*b, color_group(g));
            }
        }

        if show_ghosts {
            for g in 0..ghost_counts.len() {
                let cnt = ghost_counts[g];
                for k in 0..cnt {
                    let j = ghost_flat[g * max_ghosts + k];
                    if j == usize::MAX { continue; }
                    draw_sphere(pos[j], rad[j] * 0.35, None, Color::new(1.0, 0.0, 1.0, 0.8));
                }
            }
        }

        if show_pairs {
            for pv in &pairs {
                draw_pair(pos[pv.i], pos[pv.j], pv.kind);
            }
        }

        set_default_camera();
        draw_text("1:AABB  2:halo  3:pairs  (CLI: --group-size N --max-ghosts N --bbox-margin f --margin f --atom-r f)", 20.0, 20.0, 20.0, WHITE);

        next_frame().await;
    }
}
