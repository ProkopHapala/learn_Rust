//! demo10_collision_balls — OpenCL collision sim with group-based broad phase.
//!
//! GPU does physics + AABB + overlap detection every frame.
//! CPU does rare rebalancing (Morton sort, greedy swaps, 2W retile) + mouse interaction.
//!
//! See README.md for full architecture, algorithm details, and open issues.
//!
//! Data layout: pos = (x, y, z, radius), vel = (vx, vy, vz, inv_mass).
//! Groups are contiguous: group g = particles [g*W, (g+1)*W).

use eframe::egui;
use ocl::{ProQue, Buffer, flags, prm::Float4};
use rand::Rng;
use std::time::Instant;

/// 2D Morton code: interleave bits of x and z (16-bit each -> 32-bit key).
/// Produces a Z-curve ordering where spatially close particles get nearby indices.
/// Used for initial sorting and full rebuilds — groups formed from consecutive
/// indices in Morton order are compact spatial patches.
fn morton2d(x: u16, z: u16) -> u32 {
    let mut r = 0u32;
    for i in 0..16 {
        r |= (((x >> i) & 1) as u32) << (2 * i);
        r |= (((z >> i) & 1) as u32) << (2 * i + 1);
    }
    r
}

/// Sort particles by 2D Morton Z-curve order. Rearranges both pos and vel arrays.
/// O(N log N). Called at init and on "Morton rebuild" button press.
/// This is the nuclear option — it reorders all groups simultaneously.
fn sort_by_morton(pos: &mut Vec<f32>, vel: &mut Vec<f32>, box_min: &[f32; 3], box_max: &[f32; 3]) {
    let n = pos.len() / 4;
    let scale_x = 65535.0 / (box_max[0] - box_min[0]);
    let scale_z = 65535.0 / (box_max[2] - box_min[2]);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by_key(|&i| {
        let x = ((pos[i*4] - box_min[0]) * scale_x).clamp(0.0, 65535.0) as u16;
        let z = ((pos[i*4+2] - box_min[2]) * scale_z).clamp(0.0, 65535.0) as u16;
        morton2d(x, z)
    });
    let old_pos = pos.clone();
    let old_vel = vel.clone();
    for (new_i, &old_i) in indices.iter().enumerate() {
        pos[new_i*4..new_i*4+4].copy_from_slice(&old_pos[old_i*4..old_i*4+4]);
        vel[new_i*4..new_i*4+4].copy_from_slice(&old_vel[old_i*4..old_i*4+4]);
    }
}

const KERNEL_SRC: &str = include_str!("collision_kernel.cl");
const GROUP_SIZE: usize = 32;
const PATHOLOGICAL_DEGREE: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum RebalanceStrategy { Retile, GreedySwaps, SwapsThenRetile, Morton }

impl RebalanceStrategy {
    fn label(self) -> &'static str {
        match self {
            Self::Retile => "2W retile",
            Self::GreedySwaps => "Greedy swaps",
            Self::SwapsThenRetile => "Swaps + retile",
            Self::Morton => "Morton rebuild",
        }
    }
}

struct GroupMap {
    amins: Vec<[f32; 4]>,
    amaxs: Vec<[f32; 4]>,
    bits: Vec<u32>,
    degree: Vec<u32>,
    n_words: usize,
}

impl GroupMap {
    fn has_edge(&self, g: usize, h: usize) -> bool {
        self.bits[g * self.n_words + h / 32] & (1u32 << (h % 32)) != 0
    }

    fn validate(&self) {
        let n_groups = self.degree.len();
        assert_eq!(self.amins.len(), n_groups);
        assert_eq!(self.amaxs.len(), n_groups);
        assert_eq!(self.bits.len(), n_groups * self.n_words);
        for g in 0..n_groups {
            assert!(self.amins[g][0].is_finite() && self.amins[g][2].is_finite() && self.amaxs[g][0].is_finite() && self.amaxs[g][2].is_finite(), "non-finite AABB for group {g}");
            assert!(self.amins[g][0] <= self.amaxs[g][0] && self.amins[g][2] <= self.amaxs[g][2], "invalid AABB for group {g}");
            assert!(!self.has_edge(g, g), "self-overlap bit set for group {g}");
            let mut d = 0u32;
            for word in 0..self.n_words {
                let mut mask = self.bits[g * self.n_words + word];
                d += mask.count_ones();
                while mask != 0 {
                    let bit = mask.trailing_zeros() as usize;
                    let h = word * 32 + bit;
                    assert!(h < n_groups, "tail overlap bit {g}->{h} exceeds group count {n_groups}");
                    assert!(self.has_edge(h, g), "asymmetric overlap edge {g}->{h}");
                    mask &= mask - 1;
                }
            }
            assert_eq!(d, self.degree[g], "degree mismatch for group {g}");
        }
    }
}

/// Manages OpenCL buffers and kernels for the collision simulation.
///
/// Buffers:
///   pos_in/out, vel_in/out — per-particle float4 arrays (N entries)
///   aabb_min_buf, aabb_max_buf — per-group float4 AABBs (n_groups entries)
///   overlap_bits_buf — exact n_groups * ceil(n_groups / 32) bit matrix
///   degree_buf — exact overlap degree used for fast/slow-path selection
///
/// Kernels are pre-built with fixed args; only dynamic args (dt, gravity, etc.)
/// are updated via set_arg in step(). Arg indices must match kernel declaration order.
///
/// CAVEAT: step() does NOT compute AABBs/overlaps — caller must call
/// compute_aabbs() and compute_overlaps() first. This is intentional,
/// to allow timing broad and narrow phases separately.
struct CollisionOcl {
    pro_que: ProQue,
    pos_in: Buffer<f32>,
    vel_in: Buffer<f32>,
    pos_out: Buffer<f32>,
    vel_out: Buffer<f32>,
    aabb_min_buf: Buffer<f32>,
    aabb_max_buf: Buffer<f32>,
    overlap_bits_buf: Buffer<u32>,
    degree_buf: Buffer<u32>,
    n: usize,
    n_groups: usize,
    n_words: usize,
    collision_normal_kernel: ocl::Kernel,
    collision_pathological_kernel: ocl::Kernel,
    aabb_kernel: ocl::Kernel,
    overlap_bits_kernel: ocl::Kernel,
    overlap_degrees_kernel: ocl::Kernel,
}

impl CollisionOcl {
    fn new(n: usize, w: usize, box_min: [f32; 3], box_max: [f32; 3], radius: f32) -> ocl::Result<Self> {
        assert_eq!(w, GROUP_SIZE, "demo10 requires W=32");
        assert!(n > 0 && n % GROUP_SIZE == 0, "particle count must be a positive multiple of 32");
        let mut rng = rand::thread_rng();
        let n_groups = n / GROUP_SIZE;
        let n_words = (n_groups + 31) / 32;

        // Initialize particles randomly in the box, 2D (y=0)
        let pos_host: Vec<f32> = (0..n).flat_map(|_| {
            let x = rng.gen_range(box_min[0] + radius..box_max[0] - radius);
            let y = 0.0f32;
            let z = rng.gen_range(box_min[2] + radius..box_max[2] - radius);
            [x, y, z, radius]
        }).collect();
        let vel_host: Vec<f32> = (0..n).flat_map(|_| {
            let vx = rng.gen_range(-0.5..0.5);
            let vy = 0.0f32;
            let vz = rng.gen_range(-0.5..0.5);
            let inv_mass = 1.0f32; // unit mass
            [vx, vy, vz, inv_mass]
        }).collect();

        // Sort by Morton Z-curve so consecutive groups form compact spatial patches
        let mut pos_host = pos_host;
        let mut vel_host = vel_host;
        sort_by_morton(&mut pos_host, &mut vel_host, &box_min, &box_max);

        let pro_que = ProQue::builder()
            .src(KERNEL_SRC)
            .dims(n)
            .build()?;

        let pos_in = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(n * 4)
            .copy_host_slice(&pos_host)
            .build()?;
        let vel_in = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(n * 4)
            .copy_host_slice(&vel_host)
            .build()?;
        let pos_out = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n * 4)
            .build()?;
        let vel_out = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n * 4)
            .build()?;
        let aabb_min_buf = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n_groups * 4)
            .build()?;
        let aabb_max_buf = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n_groups * 4)
            .build()?;
        let overlap_bits_buf = Buffer::<u32>::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n_groups * n_words)
            .build()?;
        let degree_buf = Buffer::<u32>::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n_groups)
            .build()?;
        let n_padded = n_groups * GROUP_SIZE;
        let collision_normal_kernel = pro_que.kernel_builder("collision_step_normal")
            .global_work_size(n_padded)
            .local_work_size(GROUP_SIZE)
            .arg(&pos_in)
            .arg(&vel_in)
            .arg(&pos_out)
            .arg(&vel_out)
            .arg(&overlap_bits_buf)
            .arg(&degree_buf)
            .arg(n_groups as i32)
            .arg(PATHOLOGICAL_DEGREE as u32)
            .arg(0.005f32)                              // dt
            .arg(Float4::new(0.0, 0.0, 9.81, 0.0))      // gravity (down = +z in xz view)
            .arg(0.5f32)                                 // restitution
            .arg(Float4::new(box_min[0], box_min[1], box_min[2], 0.0)) // box_min
            .arg(Float4::new(box_max[0], box_max[1], box_max[2], 0.0)) // box_max
            .arg(1000.0f32)                              // k_spring
            .arg(10.0f32)                                // k_damp
            .arg(0.999f32)                               // vel_damping
            .arg(1i32)                                   // constrain_2d
            .build()?;
        let collision_pathological_kernel = pro_que.kernel_builder("collision_step_pathological")
            .global_work_size(n_padded)
            .local_work_size(GROUP_SIZE)
            .arg(&pos_in)
            .arg(&vel_in)
            .arg(&pos_out)
            .arg(&vel_out)
            .arg(&overlap_bits_buf)
            .arg(&degree_buf)
            .arg(n_groups as i32)
            .arg(PATHOLOGICAL_DEGREE as u32)
            .arg(0.005f32)
            .arg(Float4::new(0.0, 0.0, 9.81, 0.0))
            .arg(0.5f32)
            .arg(Float4::new(box_min[0], box_min[1], box_min[2], 0.0))
            .arg(Float4::new(box_max[0], box_max[1], box_max[2], 0.0))
            .arg(1000.0f32)
            .arg(10.0f32)
            .arg(0.999f32)
            .arg(1i32)
            .build()?;

        // AABB kernel: launch n_groups workgroups of size w
        let aabb_kernel = pro_que.kernel_builder("compute_aabbs")
            .global_work_size(n_groups * w)
            .local_work_size(GROUP_SIZE)
            .arg(&pos_in)
            .arg(&aabb_min_buf)
            .arg(&aabb_max_buf)
            .arg(n as i32)
            .build()?;

        let overlap_bits_kernel = pro_que.kernel_builder("compute_overlap_bits")
            .global_work_size(n_groups * n_words)
            .arg(&aabb_min_buf)
            .arg(&aabb_max_buf)
            .arg(&overlap_bits_buf)
            .arg(n_groups as i32)
            .arg(n_words as i32)
            .build()?;
        let overlap_degrees_kernel = pro_que.kernel_builder("compute_overlap_degrees")
            .global_work_size(n_groups)
            .arg(&overlap_bits_buf)
            .arg(&degree_buf)
            .arg(n_groups as i32)
            .arg(n_words as i32)
            .build()?;

        Ok(Self { pro_que, pos_in, vel_in, pos_out, vel_out, aabb_min_buf, aabb_max_buf, overlap_bits_buf, degree_buf, n, n_groups, n_words, collision_normal_kernel, collision_pathological_kernel, aabb_kernel, overlap_bits_kernel, overlap_degrees_kernel })
    }

    fn step(&mut self, dt: f32, gravity: [f32; 3], restitution: f32, k_spring: f32, k_damp: f32, vel_damping: f32, constrain_2d: bool) -> ocl::Result<()> {
        // Caller must compute_aabbs() and compute_overlaps() before calling this.
        let gravity = Float4::new(gravity[0], gravity[1], gravity[2], 0.0);
        for kernel in [&mut self.collision_normal_kernel, &mut self.collision_pathological_kernel] {
            kernel.set_arg(0, &self.pos_in)?;
            kernel.set_arg(1, &self.vel_in)?;
            kernel.set_arg(2, &self.pos_out)?;
            kernel.set_arg(3, &self.vel_out)?;
            kernel.set_arg(8, dt)?;
            kernel.set_arg(9, gravity)?;
            kernel.set_arg(10, restitution)?;
            kernel.set_arg(13, k_spring)?;
            kernel.set_arg(14, k_damp)?;
            kernel.set_arg(15, vel_damping)?;
            kernel.set_arg(16, if constrain_2d { 1i32 } else { 0i32 })?;
        }
        unsafe {
            self.collision_normal_kernel.enq()?;
            self.collision_pathological_kernel.enq()?;
        }
        std::mem::swap(&mut self.pos_in, &mut self.pos_out);
        std::mem::swap(&mut self.vel_in, &mut self.vel_out);
        Ok(())
    }

    fn compute_aabbs(&mut self) -> ocl::Result<()> {
        self.aabb_kernel.set_arg(0, &self.pos_in)?;
        unsafe { self.aabb_kernel.enq()?; }
        Ok(())
    }

    fn compute_overlaps(&self) -> ocl::Result<()> {
        unsafe {
            self.overlap_bits_kernel.enq()?;
            self.overlap_degrees_kernel.enq()?;
        }
        Ok(())
    }

    fn read_positions(&self) -> Vec<[f32; 4]> {
        let mut buf = vec![0.0f32; self.n * 4];
        self.pos_in.read(&mut buf).enq().expect("read pos_in failed");
        (0..self.n).map(|i| [buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3]]).collect()
    }

    fn read_velocities(&self) -> Vec<[f32; 4]> {
        let mut buf = vec![0.0f32; self.n * 4];
        self.vel_in.read(&mut buf).enq().expect("read vel_in failed");
        (0..self.n).map(|i| [buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3]]).collect()
    }

    fn write_pos_vel(&self, pos: &[f32], vel: &[f32]) {
        assert_eq!(pos.len(), self.n * 4);
        assert_eq!(vel.len(), self.n * 4);
        self.pos_in.write(pos).enq().expect("write pos_in failed");
        self.vel_in.write(vel).enq().expect("write vel_in failed");
    }

    fn write_vel(&self, vel: &[f32]) {
        assert_eq!(vel.len(), self.n * 4);
        self.vel_in.write(vel).enq().expect("write vel_in failed");
    }

    fn read_aabbs(&self) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
        let mut amin = vec![0.0f32; self.n_groups * 4];
        let mut amax = vec![0.0f32; self.n_groups * 4];
        self.aabb_min_buf.read(&mut amin).enq().expect("read aabb_min failed");
        self.aabb_max_buf.read(&mut amax).enq().expect("read aabb_max failed");
        let mins: Vec<[f32; 4]> = (0..self.n_groups).map(|i| [amin[i*4], amin[i*4+1], amin[i*4+2], amin[i*4+3]]).collect();
        let maxs: Vec<[f32; 4]> = (0..self.n_groups).map(|i| [amax[i*4], amax[i*4+1], amax[i*4+2], amax[i*4+3]]).collect();
        (mins, maxs)
    }

    fn read_degrees(&self) -> Vec<u32> {
        let mut degree = vec![0u32; self.n_groups];
        self.degree_buf.read(&mut degree).enq().expect("read degree failed");
        degree
    }

    fn read_overlap_bits(&self) -> Vec<u32> {
        let mut bits = vec![0u32; self.n_groups * self.n_words];
        self.overlap_bits_buf.read(&mut bits).enq().expect("read overlap bits failed");
        bits
    }

    fn read_group_map(&self) -> GroupMap {
        let (amins, amaxs) = self.read_aabbs();
        let map = GroupMap { amins, amaxs, bits: self.read_overlap_bits(), degree: self.read_degrees(), n_words: self.n_words };
        map.validate();
        map
    }
}

// --- Quality metrics (CPU-side, used for rebalancing decisions) ---

/// 2D AABB overlap test (x,z axes only). Used by rebalancing functions
/// to find overlapping group pairs on the CPU.
fn aabb_overlap(a_min: &[f32;4], a_max: &[f32;4], b_min: &[f32;4], b_max: &[f32;4]) -> bool {
    a_min[0] <= b_max[0] && a_max[0] >= b_min[0] &&
    a_min[2] <= b_max[2] && a_max[2] >= b_min[2] // 2D: only x,z
}

fn aabb_surface_2d(mn: &[f32;4], mx: &[f32;4]) -> f32 {
    let dx = mx[0] - mn[0];
    let dz = mx[2] - mn[2];
    2.0 * (dx + dz) // perimeter as proxy
}

/// O(G^2) CPU overlap count + total surface. Superseded by GPU overlap kernel
/// for overlap counting, but still used for surface computation.
/// CAVEAT: O(G^2) is fine for 512 groups (131k checks) but doesn't scale.
/// TODO: Remove this function, use GPU degrees + CPU surface sum instead.
#[cfg(test)]
fn compute_quality(amins: &[[f32; 4]], amaxs: &[[f32; 4]]) -> (usize, f32) {
    let n_groups = amins.len();
    let mut n_overlap = 0;
    let mut total_surf = 0.0f32;
    for g in 0..n_groups {
        total_surf += aabb_surface_2d(&amins[g], &amaxs[g]);
        for h in (g+1)..n_groups {
            if aabb_overlap(&amins[g], &amaxs[g], &amins[h], &amaxs[h]) {
                n_overlap += 1;
            }
        }
    }
    (n_overlap, total_surf)
}

/// Compute group AABBs from particle positions on the CPU.
/// Redundant with the GPU compute_aabbs kernel — used inside rebalancing
/// functions to evaluate candidate swaps/retiles without GPU round-trips.
/// This is intentional: rebalancing is rare and needs many quick evaluations.
/// 2D: only x,z axes are computed.
fn compute_group_aabbs_host(pos: &[f32], n_groups: usize, w: usize, n: usize) -> (Vec<[f32;4]>, Vec<[f32;4]>) {
    let mut amins = vec![[1e30f32; 4]; n_groups];
    let mut amaxs = vec![[-1e30f32; 4]; n_groups];
    for i in 0..n {
        let g = i / w;
        let x = pos[i*4]; let z = pos[i*4+2]; let r = pos[i*4+3];
        amins[g][0] = amins[g][0].min(x - r);
        amins[g][2] = amins[g][2].min(z - r);
        amaxs[g][0] = amaxs[g][0].max(x + r);
        amaxs[g][2] = amaxs[g][2].max(z + r);
    }
    (amins, amaxs)
}

fn compute_group_aabb_host(pos: &[f32], g: usize, w: usize) -> ([f32; 4], [f32; 4]) {
    let mut mn = [1e30f32; 4];
    let mut mx = [-1e30f32; 4];
    for i in g*w..(g+1)*w {
        let x = pos[i*4]; let z = pos[i*4+2]; let r = pos[i*4+3];
        assert!(x.is_finite() && z.is_finite() && r.is_finite() && r >= 0.0, "invalid particle {i} while computing group {g} AABB");
        mn[0] = mn[0].min(x - r); mn[2] = mn[2].min(z - r);
        mx[0] = mx[0].max(x + r); mx[2] = mx[2].max(z + r);
    }
    (mn, mx)
}

fn validate_particle_state(pos: &[f32], vel: &[f32], n: usize) {
    assert_eq!(pos.len(), n * 4);
    assert_eq!(vel.len(), n * 4);
    for i in 0..n {
        assert!(pos[i*4].is_finite() && pos[i*4+1].is_finite() && pos[i*4+2].is_finite() && pos[i*4+3].is_finite(), "non-finite position record {i}");
        assert!(vel[i*4].is_finite() && vel[i*4+1].is_finite() && vel[i*4+2].is_finite() && vel[i*4+3].is_finite(), "non-finite velocity record {i}");
        assert!(pos[i*4+3] >= 0.0, "negative radius for particle {i}");
        assert!(vel[i*4+3] >= 0.0, "negative inverse mass for particle {i}");
    }
}

fn ranked_overlap_pairs(map: &GroupMap) -> Vec<(usize, usize)> {
    let n_groups = map.degree.len();
    let mut pairs: Vec<(u32, u32, f32, usize, usize)> = Vec::new();
    for g in 0..n_groups {
        for word in 0..map.n_words {
            let mut mask = map.bits[g * map.n_words + word];
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let h = word * 32 + bit;
                if h > g {
                    let ox = (map.amaxs[g][0].min(map.amaxs[h][0]) - map.amins[g][0].max(map.amins[h][0])).max(0.0);
                    let oz = (map.amaxs[g][2].min(map.amaxs[h][2]) - map.amins[g][2].max(map.amins[h][2])).max(0.0);
                    pairs.push((map.degree[g].max(map.degree[h]), map.degree[g] + map.degree[h], ox * oz, g, h));
                }
                mask &= mask - 1;
            }
        }
    }
    pairs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| b.2.total_cmp(&a.2)).then_with(|| a.3.cmp(&b.3)).then_with(|| a.4.cmp(&b.4)));
    pairs.into_iter().map(|(_, _, _, g, h)| (g, h)).collect()
}

/// Greedy pairwise swap rebalancing.
///
/// For each group, find the particle that most extends its AABB (farthest from
/// center + box extension score). If that particle falls inside another group's
/// AABB, find a swap partner in that group closest to the first group's center.
/// Accept the swap only if total AABB surface of both groups decreases.
///
/// CAVEAT: Greedy one-at-a-time swaps can miss cooperative improvements where
/// two particles need to exchange simultaneously for any gain. The balanced
/// merge-split approach handles this better. Candidate destination groups come
/// from the GPU overlap row; accepted swaps recompute only two group AABBs.
/// TODO: Replace with unified merge-split that inspects k particles crossing.
fn rebalance_swaps(pos: &mut Vec<f32>, vel: &mut Vec<f32>, map: &GroupMap, w: usize, n: usize, max_swaps: usize) -> usize {
    let n_groups = map.degree.len();
    assert_eq!(n_groups * w, n);
    validate_particle_state(pos, vel, n);
    let (mut amins, mut amaxs) = compute_group_aabbs_host(pos, n_groups, w, n);
    let mut centers = vec![[0.0f32; 2]; n_groups];
    for g in 0..n_groups {
        centers[g] = [(amins[g][0] + amaxs[g][0]) * 0.5, (amins[g][2] + amaxs[g][2]) * 0.5];
    }

    // For each group, find worst particle (farthest from center, weighted by contribution to box)
    let mut worst: Vec<(f32, usize, usize)> = Vec::new(); // (dist_sq, particle_idx, group_idx)
    for g in 0..n_groups {
        let lo = g * w;
        let hi = ((g + 1) * w).min(n);
        for i in lo..hi {
            let x = pos[i*4]; let z = pos[i*4+2];
            // Distance from center, plus how much this particle extends the box
            let dx = x - centers[g][0];
            let dz = z - centers[g][1];
            let score = dx*dx + dz*dz;
            worst.push((score, i, g));
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut n_swapped = 0;
    let mut used_groups = vec![false; n_groups];
    for &(_, i, g) in &worst {
        if n_swapped >= max_swaps { break; }
        if used_groups[g] { continue; }
        let x = pos[i*4]; let z = pos[i*4+2];

        // Find which other group's AABB contains this particle
        let mut best_target = None;
        let mut best_dist = f32::MAX;
        for word in 0..map.n_words {
            let mut mask = map.bits[g * map.n_words + word];
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let h = word * 32 + bit;
                if !used_groups[h] && x >= amins[h][0] && x <= amaxs[h][0] && z >= amins[h][2] && z <= amaxs[h][2] {
                    let dx = x - centers[h][0];
                    let dz = z - centers[h][1];
                    let d = dx*dx + dz*dz;
                    if d < best_dist { best_dist = d; best_target = Some(h); }
                }
                mask &= mask - 1;
            }
        }
        let Some(h) = best_target else { continue; };

        // Find swap partner in group h: particle closest to group g's center
        let lo_h = h * w;
        let hi_h = ((h + 1) * w).min(n);
        let mut best_j = None;
        let mut best_j_dist = f32::MAX;
        for j in lo_h..hi_h {
            let jx = pos[j*4]; let jz = pos[j*4+2];
            let dx = jx - centers[g][0];
            let dz = jz - centers[g][1];
            let d = dx*dx + dz*dz;
            if d < best_j_dist { best_j_dist = d; best_j = Some(j); }
        }
        let Some(j) = best_j else { continue; };

        // Evaluate incident overlap count first, then total pair perimeter.
        let cost_before = pair_cost(g, h, &amins[g], &amaxs[g], &amins[h], &amaxs[h], &amins, &amaxs);

        // Temporarily swap using ordinary temporaries; arithmetic float swaps
        // lose precision and can turn finite state into NaN/Inf.
        let old_pos_i = [pos[i*4], pos[i*4+1], pos[i*4+2], pos[i*4+3]];
        let old_vel_i = [vel[i*4], vel[i*4+1], vel[i*4+2], vel[i*4+3]];
        let old_pos_j = [pos[j*4], pos[j*4+1], pos[j*4+2], pos[j*4+3]];
        let old_vel_j = [vel[j*4], vel[j*4+1], vel[j*4+2], vel[j*4+3]];
        pos[i*4..i*4+4].copy_from_slice(&old_pos_j);
        vel[i*4..i*4+4].copy_from_slice(&old_vel_j);
        pos[j*4..j*4+4].copy_from_slice(&old_pos_i);
        vel[j*4..j*4+4].copy_from_slice(&old_vel_i);

        let (new_g_min, new_g_max) = compute_group_aabb_host(pos, g, w);
        let (new_h_min, new_h_max) = compute_group_aabb_host(pos, h, w);
        let cost_after = pair_cost(g, h, &new_g_min, &new_g_max, &new_h_min, &new_h_max, &amins, &amaxs);

        if cost_better(cost_after, cost_before) {
            // Accept swap
            amins[g] = new_g_min; amaxs[g] = new_g_max;
            amins[h] = new_h_min; amaxs[h] = new_h_max;
            centers[g] = [(amins[g][0] + amaxs[g][0]) * 0.5, (amins[g][2] + amaxs[g][2]) * 0.5];
            centers[h] = [(amins[h][0] + amaxs[h][0]) * 0.5, (amins[h][2] + amaxs[h][2]) * 0.5];
            used_groups[g] = true;
            used_groups[h] = true;
            n_swapped += 1;
        } else {
            // Revert swap
            pos[i*4..i*4+4].copy_from_slice(&old_pos_i);
            vel[i*4..i*4+4].copy_from_slice(&old_vel_i);
            pos[j*4..j*4+4].copy_from_slice(&old_pos_j);
            vel[j*4..j*4+4].copy_from_slice(&old_vel_j);
        }
    }
    n_swapped
}

/// 2W->W+W retiling: merge two overlapping groups, split along their center line.
///
/// For each overlapping pair (sorted by overlap area), merge all 2W particles,
/// project them onto the line between group centers, and split at the median.
/// Accept if total AABB surface decreases.
///
/// The candidate set tests x, z, and the center-to-center direction. Acceptance
/// first minimizes incident overlaps and then total AABB perimeter.
fn ordered_aabb(pos: &[f32], order: &[usize]) -> ([f32; 4], [f32; 4]) {
    let mut mn = [1e30f32; 4];
    let mut mx = [-1e30f32; 4];
    for &i in order {
        let x = pos[i*4]; let z = pos[i*4+2]; let r = pos[i*4+3];
        mn[0] = mn[0].min(x-r); mn[2] = mn[2].min(z-r);
        mx[0] = mx[0].max(x+r); mx[2] = mx[2].max(z+r);
    }
    (mn, mx)
}

fn pair_cost(g: usize, h: usize, gmin: &[f32; 4], gmax: &[f32; 4], hmin: &[f32; 4], hmax: &[f32; 4], amins: &[[f32; 4]], amaxs: &[[f32; 4]]) -> (usize, f32) {
    let mut overlaps = usize::from(aabb_overlap(gmin, gmax, hmin, hmax));
    for k in 0..amins.len() {
        if k == g || k == h { continue; }
        overlaps += usize::from(aabb_overlap(gmin, gmax, &amins[k], &amaxs[k]));
        overlaps += usize::from(aabb_overlap(hmin, hmax, &amins[k], &amaxs[k]));
    }
    (overlaps, aabb_surface_2d(gmin, gmax) + aabb_surface_2d(hmin, hmax))
}

fn cost_better(a: (usize, f32), b: (usize, f32)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 < b.1 - 1e-6)
}

fn rebalance_retile(pos: &mut Vec<f32>, vel: &mut Vec<f32>, map: &GroupMap, w: usize, n: usize, max_pairs: usize) -> usize {
    let n_groups = map.degree.len();
    assert_eq!(n_groups * w, n);
    validate_particle_state(pos, vel, n);
    let pairs = ranked_overlap_pairs(map);
    let (mut amins, mut amaxs) = compute_group_aabbs_host(pos, n_groups, w, n);
    let mut used_groups = vec![false; n_groups];
    let mut n_retiled = 0;

    for &(g, h) in &pairs {
        if n_retiled >= max_pairs { break; }
        if used_groups[g] || used_groups[h] { continue; }
        let lo_g = g*w; let lo_h = h*w;
        let mut merged_pos = Vec::with_capacity(2*w*4);
        let mut merged_vel = Vec::with_capacity(2*w*4);
        merged_pos.extend_from_slice(&pos[lo_g*4..(lo_g+w)*4]);
        merged_pos.extend_from_slice(&pos[lo_h*4..(lo_h+w)*4]);
        merged_vel.extend_from_slice(&vel[lo_g*4..(lo_g+w)*4]);
        merged_vel.extend_from_slice(&vel[lo_h*4..(lo_h+w)*4]);

        let old_cost = pair_cost(g, h, &amins[g], &amaxs[g], &amins[h], &amaxs[h], &amins, &amaxs);
        let dir_x = (amaxs[h][0] + amins[h][0] - amaxs[g][0] - amins[g][0]) * 0.5;
        let dir_z = (amaxs[h][2] + amins[h][2] - amaxs[g][2] - amins[g][2]) * 0.5;
        let mut best_cost = old_cost;
        let mut best: Option<(Vec<usize>, [f32; 4], [f32; 4], [f32; 4], [f32; 4])> = None;
        for &(dx, dz) in &[(1.0f32, 0.0f32), (0.0, 1.0), (dir_x, dir_z)] {
            if dx*dx + dz*dz <= 1e-20 { continue; }
            let mut order: Vec<usize> = (0..2*w).collect();
            order.sort_by(|&a, &b| {
                let sa = merged_pos[a*4]*dx + merged_pos[a*4+2]*dz;
                let sb = merged_pos[b*4]*dx + merged_pos[b*4+2]*dz;
                sa.total_cmp(&sb).then_with(|| a.cmp(&b))
            });
            let (gmin, gmax) = ordered_aabb(&merged_pos, &order[..w]);
            let (hmin, hmax) = ordered_aabb(&merged_pos, &order[w..]);
            let candidate_cost = pair_cost(g, h, &gmin, &gmax, &hmin, &hmax, &amins, &amaxs);
            if cost_better(candidate_cost, best_cost) {
                best_cost = candidate_cost;
                best = Some((order, gmin, gmax, hmin, hmax));
            }
        }
        let Some((order, gmin, gmax, hmin, hmax)) = best else { continue; };

        for (k, &src) in order.iter().enumerate() {
            let target = if k < w { lo_g + k } else { lo_h + k - w };
            pos[target*4..target*4+4].copy_from_slice(&merged_pos[src*4..src*4+4]);
            vel[target*4..target*4+4].copy_from_slice(&merged_vel[src*4..src*4+4]);
        }
        amins[g] = gmin; amaxs[g] = gmax; amins[h] = hmin; amaxs[h] = hmax;
        used_groups[g] = true; used_groups[h] = true;
        n_retiled += 1;
    }
    n_retiled
}

/// Generate a visually distinct color per group using golden ratio HSV hashing.
/// Produces well-spread hues even for 512 groups. Saturation/value kept moderate
/// to avoid eye strain when many groups are visible simultaneously.
fn color_group(g: usize) -> egui::Color32 {
    let golden = 0.61803398875f32;
    let h = (g as f32 * golden).fract();
    let s = 0.65f32;
    let v = 0.9f32;
    let i = (h * 6.0) as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, gr, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    egui::Color32::from_rgb((r * 255.0) as u8, (gr * 255.0) as u8, (b * 255.0) as u8)
}

/// egui application state for the collision simulation.
///
/// Holds simulation parameters, GPU readback buffers for rendering, timing
/// metrics, and mouse interaction state. The update() loop runs physics,
/// reads back data, and renders to the egui central panel.
///
/// pos_read is the previous rendered snapshot used for mouse picking. It is
/// refreshed once after physics; while dragging, only the selected entry is
/// updated so the dragged particle remains visually attached to the cursor.
struct CollisionApp {
    sim: CollisionOcl,
    dt: f32,
    gravity: [f32; 3],
    restitution: f32,
    k_spring: f32,
    k_damp: f32,
    vel_damping: f32,
    constrain_2d: bool,
    box_min: [f32; 3],
    box_max: [f32; 3],
    n: usize,
    w: usize,
    ms_per_step: f32,
    ms_broad: f32,
    ms_narrow: f32,
    pos_read: Vec<[f32; 4]>,
    aabb_min_read: Vec<[f32; 4]>,
    aabb_max_read: Vec<[f32; 4]>,
    show_aabbs: bool,
    show_box: bool,
    dots_only: bool,
    paused: bool,
    auto_rebuild: bool,
    rebalance_strategy: RebalanceStrategy,
    rebuild_interval: usize,
    rebuild_degree_trigger: usize,
    max_rebalance_ops: usize,
    ms_rebalance: f32,
    frame_count: usize,
    n_overlap: usize,           // total unique overlap pairs (from GPU)
    n_overflow: usize,           // groups whose exact degree exceeds the fast-path threshold
    max_overlap: usize,          // max overlaps any single group has
    total_surf: f32,
    last_swaps: usize,
    last_retiles: usize,
    show_particles: bool,
    picked: Option<usize>,       // LMB picked particle index
    mouse_world: [f32; 2],       // current mouse pos in world (x,z)
    mouse_world_prev: [f32; 2],  // previous frame mouse pos in world
    initialized: bool,
}

impl CollisionApp {
    fn new(n: usize, w: usize) -> Self {
        let box_min = [-20.0, -20.0, -20.0];
        let box_max = [20.0, 20.0, 20.0];
        let radius = 0.1f32;
        let sim = CollisionOcl::new(n, w, box_min, box_max, radius)
            .expect("Failed to init OpenCL collision sim — is an OpenCL runtime installed?");
        let pos_read = sim.read_positions();
        Self {
            sim,
            dt: 0.005,
            gravity: [0.0, 0.0, 9.81],
            restitution: 0.5,
            k_spring: 1000.0,
            k_damp: 10.0,
            vel_damping: 0.999,
            constrain_2d: true,
            box_min,
            box_max,
            n,
            w,
            ms_per_step: 0.0,
            ms_broad: 0.0,
            ms_narrow: 0.0,
            pos_read,
            aabb_min_read: vec![[0.0; 4]; (n + w - 1) / w],
            aabb_max_read: vec![[0.0; 4]; (n + w - 1) / w],
            show_aabbs: true,
            show_box: true,
            dots_only: true,
            paused: false,
            auto_rebuild: false,
            rebalance_strategy: RebalanceStrategy::Retile,
            rebuild_interval: 50,
            rebuild_degree_trigger: 24,
            max_rebalance_ops: 8,
            ms_rebalance: 0.0,
            frame_count: 0,
            n_overlap: 0,
            n_overflow: 0,
            max_overlap: 0,
            total_surf: 0.0,
            last_swaps: 0,
            last_retiles: 0,
            show_particles: true,
            picked: None,
            mouse_world: [0.0, 0.0],
            mouse_world_prev: [0.0, 0.0],
            initialized: true,
        }
    }
}

impl CollisionApp {
    fn refresh_group_map(&mut self) -> GroupMap {
        self.sim.compute_aabbs().expect("AABB kernel failed before rebalancing");
        self.sim.compute_overlaps().expect("overlap kernels failed before rebalancing");
        self.sim.pro_que.queue().finish().expect("queue failed before rebalancing");
        self.sim.read_group_map()
    }

    fn apply_rebalance(&mut self, strategy: RebalanceStrategy, map: &GroupMap) -> bool {
        let t0 = Instant::now();
        let mut pos: Vec<f32> = self.sim.read_positions().into_iter().flat_map(|p| p.into_iter()).collect();
        let mut vel: Vec<f32> = self.sim.read_velocities().into_iter().flat_map(|v| v.into_iter()).collect();
        validate_particle_state(&pos, &vel, self.n);
        self.last_swaps = 0;
        self.last_retiles = 0;
        let changed = match strategy {
            RebalanceStrategy::GreedySwaps => {
                self.last_swaps = rebalance_swaps(&mut pos, &mut vel, map, self.w, self.n, self.max_rebalance_ops);
                self.last_swaps > 0
            }
            RebalanceStrategy::Retile => {
                self.last_retiles = rebalance_retile(&mut pos, &mut vel, map, self.w, self.n, self.max_rebalance_ops);
                self.last_retiles > 0
            }
            RebalanceStrategy::SwapsThenRetile => {
                self.last_swaps = rebalance_swaps(&mut pos, &mut vel, map, self.w, self.n, self.max_rebalance_ops);
                self.last_retiles = rebalance_retile(&mut pos, &mut vel, map, self.w, self.n, self.max_rebalance_ops);
                self.last_swaps > 0 || self.last_retiles > 0
            }
            RebalanceStrategy::Morton => {
                sort_by_morton(&mut pos, &mut vel, &self.box_min, &self.box_max);
                true
            }
        };
        validate_particle_state(&pos, &vel, self.n);
        if changed {
            self.sim.write_pos_vel(&pos, &vel);
            self.sim.pro_que.queue().finish().expect("rebalanced state upload failed");
        }
        self.ms_rebalance = t0.elapsed().as_secs_f32() * 1000.0;
        changed
    }

    fn manual_rebalance(&mut self, strategy: RebalanceStrategy) {
        let map = self.refresh_group_map();
        if self.apply_rebalance(strategy, &map) {
            self.sim.compute_aabbs().expect("AABB rebuild failed after manual rebalancing");
            self.sim.compute_overlaps().expect("overlap rebuild failed after manual rebalancing");
            self.sim.pro_que.queue().finish().expect("queue failed after manual rebalancing");
        }
    }
}

impl eframe::App for CollisionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("OpenCL Collision Balls");
            if self.initialized {
                ui.label(format!("Device: {:?}", self.sim.pro_que.device()));
            }
            ui.label(format!("Particles: {} | W: {} | Groups: {}", self.n, self.w, self.sim.n_groups));
            ui.separator();
            ui.add(egui::Slider::new(&mut self.dt, 0.0005..=0.02).text("dt"));
            ui.add(egui::Slider::new(&mut self.gravity[2], 0.0..=20.0).text("gravity z"));
            ui.add(egui::Slider::new(&mut self.restitution, 0.0..=1.0).text("restitution"));
            ui.add(egui::Slider::new(&mut self.k_spring, 100.0..=5000.0).text("k_spring"));
            ui.add(egui::Slider::new(&mut self.k_damp, 0.0..=50.0).text("k_damp"));
            ui.add(egui::Slider::new(&mut self.vel_damping, 0.95..=1.0).text("vel_damping"));
            ui.checkbox(&mut self.constrain_2d, "Constrain 2D (y=0)");
            ui.checkbox(&mut self.show_aabbs, "Show AABBs");
            ui.checkbox(&mut self.show_box, "Show box");
            ui.checkbox(&mut self.show_particles, "Show particles");
            ui.checkbox(&mut self.dots_only, "Dots only (no radius)");
            ui.checkbox(&mut self.paused, "Paused");
            ui.separator();
            ui.heading("Rebalancing");
            ui.checkbox(&mut self.auto_rebuild, "Auto rebuild");
            egui::ComboBox::from_label("Strategy")
                .selected_text(self.rebalance_strategy.label())
                .show_ui(ui, |ui| {
                    for strategy in [RebalanceStrategy::Retile, RebalanceStrategy::GreedySwaps, RebalanceStrategy::SwapsThenRetile, RebalanceStrategy::Morton] {
                        ui.selectable_value(&mut self.rebalance_strategy, strategy, strategy.label());
                    }
                });
            ui.add(egui::Slider::new(&mut self.rebuild_interval, 10..=200).text("interval (frames)"));
            ui.add(egui::Slider::new(&mut self.rebuild_degree_trigger, 1..=64).text("degree trigger"));
            ui.add(egui::Slider::new(&mut self.max_rebalance_ops, 1..=32).text("max local repairs"));
            if ui.button("Morton rebuild").clicked() {
                self.manual_rebalance(RebalanceStrategy::Morton);
            }
            if ui.button("Greedy swaps").clicked() {
                self.manual_rebalance(RebalanceStrategy::GreedySwaps);
            }
            if ui.button("2W retiling").clicked() {
                self.manual_rebalance(RebalanceStrategy::Retile);
            }
            ui.separator();
            ui.label(format!("Overlaps: {} | Max/grp: {}", self.n_overlap, self.max_overlap));
            ui.label(format!("Pathological groups: {} | Surf: {:.1}", self.n_overflow, self.total_surf));
            ui.label(format!("Last swaps: {} | retiles: {}", self.last_swaps, self.last_retiles));
            ui.label(format!("Last rebalance: {:.2} ms", self.ms_rebalance));
            ui.separator();
            ui.label(format!("GPU step: {:.2} ms (broad: {:.2} narrow: {:.2})", self.ms_per_step, self.ms_broad, self.ms_narrow));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );
            let rect = response.rect;

            // 2D projection: x -> screen_x, z -> screen_y (top-down xz view)
            let world_w = self.box_max[0] - self.box_min[0];
            let world_h = self.box_max[2] - self.box_min[2];
            let scale = rect.width().min(rect.height()) / world_w.max(world_h) * 0.9;
            let cx = rect.center().x;
            let cy = rect.center().y;

            let to_screen = |wx: f32, wz: f32| -> egui::Pos2 {
                egui::pos2(cx + wx * scale, cy + wz * scale)
            };
            let to_world = |sp: egui::Pos2| -> [f32; 2] {
                [(sp.x - cx) / scale, (sp.y - cy) / scale]
            };

            // Track mouse state
            let mouse_pos = ui.input(|i| i.pointer.hover_pos());
            let lmb_down = ui.input(|i| i.pointer.primary_down());
            let rmb_down = ui.input(|i| i.pointer.secondary_down());

            self.mouse_world_prev = self.mouse_world;
            if let Some(mp) = mouse_pos {
                if rect.contains(mp) {
                    self.mouse_world = to_world(mp);
                }
            }

            // LMB: pick and drag nearest particle
            if lmb_down && rect.contains(mouse_pos.unwrap_or(egui::Pos2::ZERO)) {
                if self.picked.is_none() {
                    // Find nearest particle to mouse
                    let mut best_d = f32::MAX;
                    let mut best_i = None;
                    for (i, p) in self.pos_read.iter().enumerate() {
                        let dx = p[0] - self.mouse_world[0];
                        let dz = p[2] - self.mouse_world[1];
                        let d = dx*dx + dz*dz;
                        if d < best_d { best_d = d; best_i = Some(i); }
                    }
                    self.picked = best_i;
                }
                if let Some(idx) = self.picked {
                    // Directly set particle position to mouse, set velocity from mouse delta
                    let mut pos_flat: Vec<f32> = self.pos_read.iter().flat_map(|p| p.iter().copied()).collect();
                    let mut vel_flat: Vec<f32> = self.sim.read_velocities().iter().flat_map(|v| v.iter().copied()).collect();
                    let dvx = (self.mouse_world[0] - self.mouse_world_prev[0]) / self.dt;
                    let dvz = (self.mouse_world[1] - self.mouse_world_prev[1]) / self.dt;
                    pos_flat[idx*4]   = self.mouse_world[0];
                    pos_flat[idx*4+2] = self.mouse_world[1];
                    vel_flat[idx*4]   = dvx;
                    vel_flat[idx*4+2] = dvz;
                    self.sim.write_pos_vel(&pos_flat, &vel_flat);
                    // Update pos_read so rendering shows the dragged position
                    self.pos_read[idx] = [pos_flat[idx*4], pos_flat[idx*4+1], pos_flat[idx*4+2], pos_flat[idx*4+3]];
                }
            } else {
                self.picked = None;
            }

            // RMB: apply radial force to nearby particles
            let force_radius = 1.5f32;
            if rmb_down && rect.contains(mouse_pos.unwrap_or(egui::Pos2::ZERO)) {
                let force_strength = 50.0f32;
                let mut vel_flat: Vec<f32> = self.sim.read_velocities().iter().flat_map(|v| v.iter().copied()).collect();
                for (i, p) in self.pos_read.iter().enumerate() {
                    let dx = p[0] - self.mouse_world[0];
                    let dz = p[2] - self.mouse_world[1];
                    let d = (dx*dx + dz*dz).sqrt();
                    if d < force_radius && d > 1e-6 {
                        let falloff = 1.0 - d / force_radius;
                        let fx = dx / d * force_strength * falloff;
                        let fz = dz / d * force_strength * falloff;
                        vel_flat[i*4]   += fx * self.dt;
                        vel_flat[i*4+2] += fz * self.dt;
                    }
                }
                self.sim.write_vel(&vel_flat);
            }

            if !self.paused {
                let t_broad = Instant::now();
                self.sim.compute_aabbs().expect("aabb kernel failed");
                self.sim.compute_overlaps().expect("overlap kernel failed");
                self.sim.pro_que.queue().finish().expect("queue finish failed");
                self.ms_broad = t_broad.elapsed().as_secs_f32() * 1000.0;

                let auto_due = self.auto_rebuild && (self.frame_count + 1) % self.rebuild_interval == 0;
                if auto_due {
                    let map = self.sim.read_group_map();
                    let max_degree = map.degree.iter().copied().max().unwrap_or(0) as usize;
                    if max_degree >= self.rebuild_degree_trigger {
                        let strategy = self.rebalance_strategy;
                        if self.apply_rebalance(strategy, &map) {
                            self.sim.compute_aabbs().expect("AABB rebuild failed after rebalancing");
                            self.sim.compute_overlaps().expect("overlap rebuild failed after rebalancing");
                            self.sim.pro_que.queue().finish().expect("queue failed after rebalancing");
                        }
                    } else {
                        self.ms_rebalance = 0.0;
                        self.last_swaps = 0;
                        self.last_retiles = 0;
                    }
                }

                let t_narrow = Instant::now();
                self.sim.step(self.dt, self.gravity, self.restitution, self.k_spring, self.k_damp, self.vel_damping, self.constrain_2d)
                    .expect("collision kernel enqueue failed");
                self.sim.pro_que.queue().finish().expect("queue finish failed");
                self.ms_narrow = t_narrow.elapsed().as_secs_f32() * 1000.0;
                self.ms_per_step = self.ms_broad + self.ms_narrow;
                self.frame_count += 1;
            }

            // Read back for rendering (positions already read above for picking)
            if !self.picked.is_some() {
                self.pos_read = self.sim.read_positions();
            }
            let (amins, amaxs) = self.sim.read_aabbs();
            self.aabb_min_read = amins;
            self.aabb_max_read = amaxs;

            // GPU-computed overlap stats. The values describe the pre-step
            // snapshot used by the most recent collision dispatch.
            let overlap_degrees = self.sim.read_degrees();
            let total_overlaps: u32 = overlap_degrees.iter().sum();
            self.n_overlap = (total_overlaps / 2) as usize;
            self.max_overlap = overlap_degrees.iter().copied().max().unwrap_or(0) as usize;
            self.n_overflow = overlap_degrees.iter().filter(|&&d| d as usize > PATHOLOGICAL_DEGREE).count();

            // Surface (still CPU, cheap: O(G) sum)
            let surf: f32 = (0..self.aabb_min_read.len())
                .map(|g| aabb_surface_2d(&self.aabb_min_read[g], &self.aabb_max_read[g]))
                .sum();
            self.total_surf = surf;

            // 2D projection: x -> screen_x, z -> screen_y (top-down xz view)
            // (already computed above for mouse interaction)
            let _ = (world_w, world_h, scale, cx, cy); // suppress unused warnings

            // Draw enclosing box
            if self.show_box {
                let p00 = to_screen(self.box_min[0], self.box_min[2]);
                let p10 = to_screen(self.box_max[0], self.box_min[2]);
                let p11 = to_screen(self.box_max[0], self.box_max[2]);
                let p01 = to_screen(self.box_min[0], self.box_max[2]);
                let box_col = egui::Color32::from_rgb(100, 100, 100);
                painter.line_segment([p00, p10], egui::Stroke::new(2.0, box_col));
                painter.line_segment([p10, p11], egui::Stroke::new(2.0, box_col));
                painter.line_segment([p11, p01], egui::Stroke::new(2.0, box_col));
                painter.line_segment([p01, p00], egui::Stroke::new(2.0, box_col));
            }

            // Draw AABBs
            if self.show_aabbs {
                for (g, (mn, mx)) in self.aabb_min_read.iter().zip(self.aabb_max_read.iter()).enumerate() {
                    let p00 = to_screen(mn[0], mn[2]);
                    let p10 = to_screen(mx[0], mn[2]);
                    let p11 = to_screen(mx[0], mx[2]);
                    let p01 = to_screen(mn[0], mx[2]);
                    let col = color_group(g);
                    let stroke = egui::Stroke::new(1.5, col);
                    painter.line_segment([p00, p10], stroke);
                    painter.line_segment([p10, p11], stroke);
                    painter.line_segment([p11, p01], stroke);
                    painter.line_segment([p01, p00], stroke);
                }
            }

            // Draw particles, colored by their collision group
            if self.show_particles {
                let w = self.w;
                for (i, p) in self.pos_read.iter().enumerate() {
                let sp = to_screen(p[0], p[2]);
                let g = i / w;
                let color = if self.picked == Some(i) {
                    egui::Color32::from_rgb(255, 255, 0) // highlight picked particle
                } else {
                    color_group(g)
                };
                if self.dots_only {
                    painter.circle_filled(sp, if self.picked == Some(i) { 4.0 } else { 2.0 }, color);
                } else {
                    let r_screen = p[3] * scale;
                    painter.circle_filled(sp, r_screen, color);
                    painter.circle_stroke(sp, r_screen, egui::Stroke::new(1.0, egui::Color32::from_black_alpha(120)));
                }
            }
            }

            // Draw RMB force radius indicator
            if rmb_down && rect.contains(mouse_pos.unwrap_or(egui::Pos2::ZERO)) {
                let mp = to_screen(self.mouse_world[0], self.mouse_world[1]);
                painter.circle_stroke(mp, force_radius * scale, egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 100, 100)));
            }
        });

        ctx.request_repaint();
    }
}

/// Entry point. 16k particles, 512 groups of W=32.
/// Scale chosen to make group overlap detection a relevant problem (m=512)
/// while keeping per-frame GPU time manageable for interactive framerates.
fn main() -> eframe::Result {
    let n = 16384usize; // 2^14 = 512 groups of 32
    let w = 32usize;
    println!("Initializing OpenCL collision sim: {} particles, W={}", n, w);
    let app = CollisionApp::new(n, w);
    println!("OpenCL ready! Device: {:?}", app.sim.pro_que.device());
    eframe::run_native(
        "OpenCL Collision Balls",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )
}

#[cfg(test)]
mod tests {
    #[derive(Clone, Copy)]
    struct Box2 { min_x: f32, max_x: f32, min_z: f32, max_z: f32 }

    fn overlaps(a: Box2, b: Box2) -> bool {
        a.min_x <= b.max_x && a.max_x >= b.min_x &&
        a.min_z <= b.max_z && a.max_z >= b.min_z
    }

    fn build_bits(boxes: &[Box2]) -> (Vec<u32>, Vec<u32>) {
        let n_groups = boxes.len();
        let n_words = (n_groups + 31) / 32;
        let mut bits = vec![0u32; n_groups * n_words];
        let mut degree = vec![0u32; n_groups];
        for g in 0..n_groups {
            for h in 0..n_groups {
                if g == h || !overlaps(boxes[g], boxes[h]) { continue; }
                bits[g * n_words + h / 32] |= 1u32 << (h % 32);
                degree[g] += 1;
            }
        }
        (bits, degree)
    }

    fn make_group_map(pos: &[f32], n_groups: usize) -> super::GroupMap {
        let (amins, amaxs) = super::compute_group_aabbs_host(pos, n_groups, super::GROUP_SIZE, n_groups * super::GROUP_SIZE);
        let boxes: Vec<Box2> = (0..n_groups).map(|g| Box2 { min_x: amins[g][0], max_x: amaxs[g][0], min_z: amins[g][2], max_z: amaxs[g][2] }).collect();
        let (bits, degree) = build_bits(&boxes);
        let map = super::GroupMap { amins, amaxs, bits, degree, n_words: (n_groups + 31) / 32 };
        map.validate();
        map
    }

    #[test]
    fn exact_bitset_handles_more_than_32_neighbors() {
        let boxes = vec![Box2 { min_x: 0.0, max_x: 1.0, min_z: 0.0, max_z: 1.0 }; 40];
        let (bits, degree) = build_bits(&boxes);
        assert_eq!(degree[0], 39);
        assert_eq!(degree[39], 39);
        assert_eq!(bits.len(), 40 * 2);
        assert_eq!(bits[0] & 1, 0); // no self bit
        assert_ne!(bits[0] & (1u32 << 31), 0); // group 31 is retained
        assert_ne!(bits[0 * 2 + 1] & (1u32 << 7), 0); // group 39 is retained
    }

    #[test]
    fn exact_bitset_is_symmetric_and_excludes_non_overlaps() {
        let boxes = vec![
            Box2 { min_x: 0.0, max_x: 1.0, min_z: 0.0, max_z: 1.0 },
            Box2 { min_x: 0.5, max_x: 1.5, min_z: 0.5, max_z: 1.5 },
            Box2 { min_x: 5.0, max_x: 6.0, min_z: 5.0, max_z: 6.0 },
        ];
        let (bits, degree) = build_bits(&boxes);
        assert_eq!(degree, vec![1, 1, 0]);
        assert_ne!(bits[0] & (1u32 << 1), 0);
        assert_ne!(bits[1] & (1u32 << 0), 0);
        assert_eq!(bits[0] & (1u32 << 2), 0);
        assert_eq!(bits[2], 0);
    }

    #[test]
    #[should_panic(expected = "asymmetric overlap edge")]
    fn group_map_rejects_asymmetric_edges() {
        let map = super::GroupMap {
            amins: vec![[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            amaxs: vec![[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
            bits: vec![2, 0],
            degree: vec![1, 0],
            n_words: 1,
        };
        map.validate();
    }

    #[test]
    fn retile_uses_gpu_edges_and_preserves_particle_records() {
        let n = 2 * super::GROUP_SIZE;
        let mut pos = vec![0.0f32; n*4];
        let mut vel = vec![0.0f32; n*4];
        for i in 0..n {
            let local = i % super::GROUP_SIZE;
            let x = if i < super::GROUP_SIZE {
                if local < 16 { -10.0 } else { 10.0 }
            } else if local < 16 { -9.0 } else { 9.0 };
            pos[i*4] = x; pos[i*4+2] = 0.0; pos[i*4+3] = 0.1;
            vel[i*4] = i as f32; vel[i*4+1] = x; vel[i*4+3] = 1.0;
        }
        let map = make_group_map(&pos, 2);
        assert_eq!(map.degree, vec![1, 1]);
        let quality_before = super::compute_quality(&map.amins, &map.amaxs);
        let n_retiled = super::rebalance_retile(&mut pos, &mut vel, &map, super::GROUP_SIZE, n, 1);
        assert_eq!(n_retiled, 1);
        let (amins, amaxs) = super::compute_group_aabbs_host(&pos, 2, super::GROUP_SIZE, n);
        let quality_after = super::compute_quality(&amins, &amaxs);
        assert!(quality_after.0 < quality_before.0 || (quality_after.0 == quality_before.0 && quality_after.1 < quality_before.1));
        assert!(!super::aabb_overlap(&amins[0], &amaxs[0], &amins[1], &amaxs[1]));
        let mut ids: Vec<usize> = (0..n).map(|i| vel[i*4] as usize).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..n).collect::<Vec<_>>());
        for i in 0..n { assert_eq!(pos[i*4], vel[i*4+1], "position/velocity record split at slot {i}"); }
    }

    #[test]
    fn opencl_exact_bitset_and_pathological_kernel_smoke() {
        let n = 40 * super::GROUP_SIZE;
        let mut sim = super::CollisionOcl::new(
            n,
            super::GROUP_SIZE,
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            0.1,
        ).expect("OpenCL initialization failed");
        let pos = vec![[0.0f32, 0.0, 0.0, 0.1]; n].into_iter().flatten().collect::<Vec<_>>();
        let vel = vec![[0.0f32, 0.0, 0.0, 1.0]; n].into_iter().flatten().collect::<Vec<_>>();
        sim.write_pos_vel(&pos, &vel);
        sim.compute_aabbs().expect("AABB kernel failed");
        sim.compute_overlaps().expect("overlap kernels failed");
        sim.pro_que.queue().finish().expect("overlap queue failed");

        let degrees = sim.read_degrees();
        assert!(degrees.iter().all(|&degree| degree == 39));
        let bits = sim.read_overlap_bits();
        assert_ne!(bits[39 / 32] & (1u32 << (39 % 32)), 0);

        sim.step(0.001, [0.0, 0.0, 0.0], 0.5, 1000.0, 10.0, 1.0, true)
            .expect("collision kernels failed");
        sim.pro_que.queue().finish().expect("collision queue failed");
        let positions = sim.read_positions();
        assert!(positions.iter().all(|p| p.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn opencl_pathological_row_processes_group_39_and_preserves_input() {
        let n_groups = 40;
        let n = n_groups * super::GROUP_SIZE;
        let mut sim = super::CollisionOcl::new(n, super::GROUP_SIZE, [-20.0, -20.0, -20.0], [20.0, 20.0, 20.0], 0.1).expect("OpenCL initialization failed");
        let mut pos = vec![0.0f32; n*4];
        let mut vel = vec![0.0f32; n*4];
        for i in 0..n {
            let g = i / super::GROUP_SIZE;
            pos[i*4] = if g == 0 { 0.0 } else if g == 39 { 0.15 } else { 10.0 };
            pos[i*4+3] = 0.1;
            vel[i*4+3] = 1.0;
        }
        sim.write_pos_vel(&pos, &vel);
        let n_words = 2;
        let mut bits = vec![0u32; n_groups*n_words];
        let mut degree = vec![0u32; n_groups];
        for h in 1..n_groups {
            bits[h/32] |= 1u32 << (h%32);
            bits[h*n_words] |= 1;
            degree[0] += 1;
            degree[h] = 1;
        }
        sim.overlap_bits_buf.write(&bits).enq().expect("bit matrix upload failed");
        sim.degree_buf.write(&degree).enq().expect("degree upload failed");
        sim.step(0.001, [0.0, 0.0, 0.0], 0.5, 10.0, 0.0, 1.0, true).expect("collision kernels failed");
        sim.pro_que.queue().finish().expect("collision queue failed");

        let mut old_input = vec![0.0f32; n*4];
        sim.pos_out.read(&mut old_input).enq().expect("old input read failed");
        assert_eq!(old_input, pos, "collision kernel modified its input snapshot");
        let out = sim.read_positions();
        assert!(out[0][0] < -1e-6, "pathological group did not process partner 39");
        assert!(out[39*super::GROUP_SIZE][0] > 0.150001, "normal reverse row did not process group 0");
    }

    #[test]
    #[ignore = "manual target-scale performance diagnostic"]
    fn headless_target_scale_benchmark() {
        let n = 16384;
        let mut sim = super::CollisionOcl::new(n, super::GROUP_SIZE, [-20.0, -20.0, -20.0], [20.0, 20.0, 20.0], 0.1).expect("OpenCL initialization failed");
        for _ in 0..3 {
            sim.compute_aabbs().unwrap(); sim.compute_overlaps().unwrap();
            sim.step(0.005, [0.0, 0.0, 9.81], 0.5, 1000.0, 10.0, 0.999, true).unwrap();
        }
        sim.pro_que.queue().finish().unwrap();
        let mut broad_ms = 0.0f64;
        let mut narrow_ms = 0.0f64;
        for _ in 0..10 {
            let t = std::time::Instant::now();
            sim.compute_aabbs().unwrap(); sim.compute_overlaps().unwrap(); sim.pro_que.queue().finish().unwrap();
            broad_ms += t.elapsed().as_secs_f64() * 1000.0;
            let t = std::time::Instant::now();
            sim.step(0.005, [0.0, 0.0, 9.81], 0.5, 1000.0, 10.0, 0.999, true).unwrap(); sim.pro_que.queue().finish().unwrap();
            narrow_ms += t.elapsed().as_secs_f64() * 1000.0;
        }
        sim.compute_aabbs().unwrap(); sim.compute_overlaps().unwrap(); sim.pro_que.queue().finish().unwrap();
        let t = std::time::Instant::now();
        let map = sim.read_group_map();
        let mut pos: Vec<f32> = sim.read_positions().into_iter().flatten().collect();
        let mut vel: Vec<f32> = sim.read_velocities().into_iter().flatten().collect();
        let repaired = super::rebalance_retile(&mut pos, &mut vel, &map, super::GROUP_SIZE, n, 8);
        if repaired > 0 { sim.write_pos_vel(&pos, &vel); sim.pro_que.queue().finish().unwrap(); }
        let repair_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("target-scale: broad={:.3} ms, narrow={:.3} ms, retile={} in {:.3} ms", broad_ms/10.0, narrow_ms/10.0, repaired, repair_ms);
    }
}
