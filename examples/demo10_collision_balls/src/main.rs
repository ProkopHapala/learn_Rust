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
const MAX_OVERLAP: usize = 32;

/// Manages OpenCL buffers and kernels for the collision simulation.
///
/// Buffers:
///   pos_buf, vel_buf, force_buf — per-particle float4 arrays (N entries)
///   aabb_min_buf, aabb_max_buf — per-group float4 AABBs (n_groups entries)
///   overlap_list_buf — n_groups * MAX_OVERLAP int partner IDs
///   overlap_count_buf — n_groups int true overlap counts
///
/// Kernels are pre-built with fixed args; only dynamic args (dt, gravity, etc.)
/// are updated via set_arg in step(). Arg indices must match kernel declaration order.
///
/// CAVEAT: step() does NOT compute AABBs/overlaps — caller must call
/// compute_aabbs() and compute_overlaps() first. This is intentional,
/// to allow timing broad and narrow phases separately.
struct CollisionOcl {
    pro_que: ProQue,
    pos_buf: Buffer<f32>,
    vel_buf: Buffer<f32>,
    force_buf: Buffer<f32>,
    aabb_min_buf: Buffer<f32>,
    aabb_max_buf: Buffer<f32>,
    overlap_list_buf: Buffer<i32>,
    overlap_count_buf: Buffer<i32>,
    n: usize,
    n_groups: usize,
    w: usize,
    collision_kernel: ocl::Kernel,
    aabb_kernel: ocl::Kernel,
    overlap_kernel: ocl::Kernel,
}

impl CollisionOcl {
    fn new(n: usize, w: usize, box_min: [f32; 3], box_max: [f32; 3], radius: f32) -> ocl::Result<Self> {
        let mut rng = rand::thread_rng();
        let n_groups = (n + w - 1) / w;

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

        let pos_buf = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(n * 4)
            .copy_host_slice(&pos_host)
            .build()?;
        let vel_buf = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(n * 4)
            .copy_host_slice(&vel_host)
            .build()?;
        let force_buf = Buffer::builder()
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
        let overlap_list_buf = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n_groups * MAX_OVERLAP)
            .build()?;
        let overlap_count_buf = Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_WRITE)
            .len(n_groups)
            .build()?;

        let n_padded = n_groups * w; // global size must be multiple of local size
        let collision_kernel = pro_que.kernel_builder("collision_step")
            .global_work_size(n_padded)
            .local_work_size(w)
            .arg(&pos_buf)
            .arg(&vel_buf)
            .arg(&force_buf)
            .arg(&overlap_list_buf)
            .arg(&overlap_count_buf)
            .arg(n as i32)
            .arg(n_groups as i32)
            .arg(MAX_OVERLAP as i32)
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

        // AABB kernel: launch n_groups workgroups of size w
        let aabb_kernel = pro_que.kernel_builder("compute_aabbs")
            .global_work_size(n_groups * w)
            .local_work_size(w)
            .arg(&pos_buf)
            .arg(&aabb_min_buf)
            .arg(&aabb_max_buf)
            .arg(n as i32)
            .build()?;

        // Overlap kernel: launch n_groups workgroups of size w
        let overlap_kernel = pro_que.kernel_builder("compute_overlaps")
            .global_work_size(n_groups * w)
            .local_work_size(w)
            .arg(&aabb_min_buf)
            .arg(&aabb_max_buf)
            .arg(&overlap_list_buf)
            .arg(&overlap_count_buf)
            .arg(n_groups as i32)
            .build()?;

        Ok(Self { pro_que, pos_buf, vel_buf, force_buf, aabb_min_buf, aabb_max_buf, overlap_list_buf, overlap_count_buf, n, n_groups, w, collision_kernel, aabb_kernel, overlap_kernel })
    }

    fn step(&mut self, dt: f32, gravity: [f32; 3], restitution: f32, k_spring: f32, k_damp: f32, vel_damping: f32, constrain_2d: bool) -> ocl::Result<()> {
        // Caller must compute_aabbs() and compute_overlaps() before calling this.
        self.collision_kernel.set_arg(8, dt)?;
        self.collision_kernel.set_arg(9, Float4::new(gravity[0], gravity[1], gravity[2], 0.0))?;
        self.collision_kernel.set_arg(10, restitution)?;
        self.collision_kernel.set_arg(13, k_spring)?;
        self.collision_kernel.set_arg(14, k_damp)?;
        self.collision_kernel.set_arg(15, vel_damping)?;
        self.collision_kernel.set_arg(16, if constrain_2d { 1i32 } else { 0i32 })?;
        unsafe { self.collision_kernel.enq()?; }
        Ok(())
    }

    fn compute_aabbs(&self) -> ocl::Result<()> {
        unsafe { self.aabb_kernel.enq()?; }
        Ok(())
    }

    fn compute_overlaps(&self) -> ocl::Result<()> {
        unsafe { self.overlap_kernel.enq()?; }
        Ok(())
    }

    fn read_positions(&self) -> Vec<[f32; 4]> {
        let mut buf = vec![0.0f32; self.n * 4];
        self.pos_buf.read(&mut buf).enq().expect("read pos_buf failed");
        (0..self.n).map(|i| [buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3]]).collect()
    }

    fn read_velocities(&self) -> Vec<[f32; 4]> {
        let mut buf = vec![0.0f32; self.n * 4];
        self.vel_buf.read(&mut buf).enq().expect("read vel_buf failed");
        (0..self.n).map(|i| [buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3]]).collect()
    }

    fn write_pos_vel(&self, pos: &[f32], vel: &[f32]) {
        self.pos_buf.write(pos).enq().expect("write pos_buf failed");
        self.vel_buf.write(vel).enq().expect("write vel_buf failed");
    }

    fn write_vel(&self, vel: &[f32]) {
        self.vel_buf.write(vel).enq().expect("write vel_buf failed");
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

    fn read_overlaps(&self) -> (Vec<i32>, Vec<i32>) {
        let mut list = vec![0i32; self.n_groups * MAX_OVERLAP];
        let mut count = vec![0i32; self.n_groups];
        self.overlap_list_buf.read(&mut list).enq().expect("read overlap_list failed");
        self.overlap_count_buf.read(&mut count).enq().expect("read overlap_count failed");
        (list, count)
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
/// TODO: Remove this function, use GPU overlap_count + CPU surface sum instead.
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

/// Greedy pairwise swap rebalancing.
///
/// For each group, find the particle that most extends its AABB (farthest from
/// center + box extension score). If that particle falls inside another group's
/// AABB, find a swap partner in that group closest to the first group's center.
/// Accept the swap only if total AABB surface of both groups decreases.
///
/// CAVEAT: Greedy one-at-a-time swaps can miss cooperative improvements where
/// two particles need to exchange simultaneously for any gain. The balanced
/// merge-split approach (planned) handles this better.
/// CAVEAT: Recomputes ALL group AABBs after each swap attempt (O(N) per swap).
/// Could be optimized to recompute only the two affected groups.
/// TODO: Replace with unified merge-split that inspects k particles crossing.
fn rebalance_swaps(pos: &mut Vec<f32>, vel: &mut Vec<f32>, n_groups: usize, w: usize, n: usize, max_swaps: usize) -> usize {
    let (amins, amaxs) = compute_group_aabbs_host(pos, n_groups, w, n);
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
            let x = pos[i*4]; let z = pos[i*4+2]; let r = pos[i*4+3];
            // Distance from center, plus how much this particle extends the box
            let dx = (x - centers[g][0]).max(0.0);
            let dz = (z - centers[g][1]).max(0.0);
            let extend = (x + r - amaxs[g][0]).max(0.0).max((amins[g][0] - (x - r)).max(0.0))
                       + (z + r - amaxs[g][2]).max(0.0).max((amins[g][2] - (z - r)).max(0.0));
            let score = dx*dx + dz*dz + extend * extend * 4.0; // weighted
            worst.push((score, i, g));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut n_swapped = 0;
    let mut used = std::collections::HashSet::new();
    for &(_, i, g) in &worst {
        if n_swapped >= max_swaps { break; }
        if used.contains(&i) { continue; }
        let x = pos[i*4]; let z = pos[i*4+2];

        // Find which other group's AABB contains this particle
        let mut best_target = None;
        let mut best_dist = f32::MAX;
        for h in 0..n_groups {
            if h == g { continue; }
            if x >= amins[h][0] && x <= amaxs[h][0] && z >= amins[h][2] && z <= amaxs[h][2] {
                let dx = x - centers[h][0];
                let dz = z - centers[h][1];
                let d = dx*dx + dz*dz;
                if d < best_dist { best_dist = d; best_target = Some(h); }
            }
        }
        let Some(h) = best_target else { continue; };

        // Find swap partner in group h: particle closest to group g's center
        let lo_h = h * w;
        let hi_h = ((h + 1) * w).min(n);
        let mut best_j = None;
        let mut best_j_dist = f32::MAX;
        for j in lo_h..hi_h {
            if used.contains(&j) { continue; }
            let jx = pos[j*4]; let jz = pos[j*4+2];
            let dx = jx - centers[g][0];
            let dz = jz - centers[g][1];
            let d = dx*dx + dz*dz;
            if d < best_j_dist { best_j_dist = d; best_j = Some(j); }
        }
        let Some(j) = best_j else { continue; };

        // Evaluate: swap i and j, recompute both group AABBs, check if total surface decreases
        let surf_before = aabb_surface_2d(&amins[g], &amaxs[g]) + aabb_surface_2d(&amins[h], &amaxs[h]);

        // Temporarily swap
        for k in 0..4 {
            pos[i*4+k] = pos[i*4+k] + pos[j*4+k];
            pos[j*4+k] = pos[i*4+k] - pos[j*4+k];
            pos[i*4+k] = pos[i*4+k] - pos[j*4+k];
        }
        for k in 0..4 {
            vel[i*4+k] = vel[i*4+k] + vel[j*4+k];
            vel[j*4+k] = vel[i*4+k] - vel[j*4+k];
            vel[i*4+k] = vel[i*4+k] - vel[j*4+k];
        }

        let (new_amins, new_amaxs) = compute_group_aabbs_host(pos, n_groups, w, n);
        let surf_after = aabb_surface_2d(&new_amins[g], &new_amaxs[g]) + aabb_surface_2d(&new_amins[h], &new_amaxs[h]);

        if surf_after < surf_before {
            // Accept swap
            used.insert(i);
            used.insert(j);
            n_swapped += 1;
        } else {
            // Revert swap
            for k in 0..4 {
                pos[i*4+k] = pos[i*4+k] + pos[j*4+k];
                pos[j*4+k] = pos[i*4+k] - pos[j*4+k];
                pos[i*4+k] = pos[i*4+k] - pos[j*4+k];
            }
            for k in 0..4 {
                vel[i*4+k] = vel[i*4+k] + vel[j*4+k];
                vel[j*4+k] = vel[i*4+k] - vel[j*4+k];
                vel[i*4+k] = vel[i*4+k] - vel[j*4+k];
            }
        }
    }
    n_swapped
}

/// 2W->W+W retiling: merge two overlapping groups, sort by longest axis, split at median.
///
/// For each overlapping pair (sorted by overlap area), merge all 2W particles,
/// find the longest axis of the combined bounding box, sort by that axis,
/// and split at the median. Accept if total AABB surface decreases.
///
/// CAVEAT: Sorting by the longest single axis (x or z) is suboptimal when
/// groups are offset diagonally. The optimal balanced partition for fixed
/// centers sorts by projection onto the line connecting group centers:
///   Delta_i = |x_i - c_A|^2 - |x_i - c_B|^2
/// This is equivalent to sorting by dot(x_i, c_B - c_A), which picks the
/// cut plane perpendicular to the inter-center direction.
/// TODO: Replace axis-based sort with Delta_i projection + center iteration.
fn rebalance_retile(pos: &mut Vec<f32>, vel: &mut Vec<f32>, n_groups: usize, w: usize, n: usize, max_pairs: usize) -> usize {
    let (amins, amaxs) = compute_group_aabbs_host(pos, n_groups, w, n);

    // Find overlapping pairs, sorted by overlap volume (descending)
    let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
    for g in 0..n_groups {
        for h in (g+1)..n_groups {
            if aabb_overlap(&amins[g], &amaxs[g], &amins[h], &amaxs[h]) {
                let ox = (amaxs[g][0].min(amaxs[h][0]) - amins[g][0].max(amins[h][0])).max(0.0);
                let oz = (amaxs[g][2].min(amaxs[h][2]) - amins[g][2].max(amins[h][2])).max(0.0);
                pairs.push((ox * oz, g, h)); // overlap area
            }
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut n_retiled = 0;
    let mut used_groups = std::collections::HashSet::new();
    for &(_, g, h) in &pairs {
        if n_retiled >= max_pairs { break; }
        if used_groups.contains(&g) || used_groups.contains(&h) { continue; }

        let lo_g = g * w;
        let hi_g = ((g + 1) * w).min(n);
        let lo_h = h * w;
        let hi_h = ((h + 1) * w).min(n);
        let n_g = hi_g - lo_g;
        let n_h = hi_h - lo_h;
        if n_g + n_h == 0 { continue; }

        // Merge particles from g and h
        let mut merged_pos: Vec<f32> = Vec::with_capacity((n_g + n_h) * 4);
        let mut merged_vel: Vec<f32> = Vec::with_capacity((n_g + n_h) * 4);
        let mut orig_indices: Vec<usize> = Vec::with_capacity(n_g + n_h);
        for &lo in &[lo_g, lo_h] {
            let hi = if lo == lo_g { hi_g } else { hi_h };
            for i in lo..hi {
                merged_pos.extend_from_slice(&pos[i*4..i*4+4]);
                merged_vel.extend_from_slice(&vel[i*4..i*4+4]);
                orig_indices.push(i);
            }
        }
        let n_merged = orig_indices.len();

        // Find longest axis of combined bounding box
        let combined_min_x = merged_pos.iter().step_by(4).cloned().fold(f32::MAX, f32::min);
        let combined_max_x = merged_pos.iter().step_by(4).cloned().fold(f32::MIN, f32::max);
        let combined_min_z = merged_pos.iter().skip(2).step_by(4).cloned().fold(f32::MAX, f32::min);
        let combined_max_z = merged_pos.iter().skip(2).step_by(4).cloned().fold(f32::MIN, f32::max);
        let dx = combined_max_x - combined_min_x;
        let dz = combined_max_z - combined_min_z;
        let axis = if dx > dz { 0 } else { 2 }; // 0=x, 2=z

        // Sort merged by axis coordinate
        let mut sort_idx: Vec<usize> = (0..n_merged).collect();
        sort_idx.sort_by(|&a, &b| {
            merged_pos[a*4+axis].partial_cmp(&merged_pos[b*4+axis]).unwrap()
        });

        // Compute surface before
        let surf_before = aabb_surface_2d(&amins[g], &amaxs[g]) + aabb_surface_2d(&amins[h], &amaxs[h]);

        // Split at median: first half -> g, second half -> h
        let mid = n_merged / 2;
        let new_pos = merged_pos.clone();
        let new_vel = merged_vel.clone();
        for (k, &si) in sort_idx.iter().enumerate() {
            let target = if k < mid { lo_g + k } else { lo_h + (k - mid) };
            pos[target*4..target*4+4].copy_from_slice(&new_pos[si*4..si*4+4]);
            vel[target*4..target*4+4].copy_from_slice(&new_vel[si*4..si*4+4]);
        }

        // Compute surface after
        let (new_amins, new_amaxs) = compute_group_aabbs_host(pos, n_groups, w, n);
        let surf_after = aabb_surface_2d(&new_amins[g], &new_amaxs[g]) + aabb_surface_2d(&new_amins[h], &new_amaxs[h]);

        if surf_after < surf_before {
            used_groups.insert(g);
            used_groups.insert(h);
            n_retiled += 1;
        } else {
            // Revert: put original data back
            for (k, &oi) in orig_indices.iter().enumerate() {
                pos[oi*4..oi*4+4].copy_from_slice(&merged_pos[k*4..k*4+4]);
                vel[oi*4..oi*4+4].copy_from_slice(&merged_vel[k*4..k*4+4]);
            }
        }
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
/// Non-obvious: pos_read is read at the START of the frame (for mouse picking)
/// and again after physics step (for rendering). When dragging (picked.is_some()),
/// the second read is skipped to avoid overwriting the dragged position.
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
    rebuild_interval: usize,
    frame_count: usize,
    n_overlap: usize,           // total unique overlap pairs (from GPU)
    n_overflow: usize,           // groups whose overlap count > MAX_OVERLAP
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
            pos_read: vec![[0.0; 4]; n],
            aabb_min_read: vec![[0.0; 4]; (n + w - 1) / w],
            aabb_max_read: vec![[0.0; 4]; (n + w - 1) / w],
            show_aabbs: true,
            show_box: true,
            dots_only: true,
            paused: false,
            auto_rebuild: false,
            rebuild_interval: 50,
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
    fn do_morton_rebuild(&mut self) {
        let mut pos: Vec<f32> = self.sim.read_positions().into_iter().flat_map(|p| p.into_iter()).collect();
        let mut vel: Vec<f32> = self.sim.read_velocities().into_iter().flat_map(|v| v.into_iter()).collect();
        sort_by_morton(&mut pos, &mut vel, &self.box_min, &self.box_max);
        self.sim.write_pos_vel(&pos, &vel);
        println!("Morton rebuild done");
    }

    fn do_greedy_swaps(&mut self) {
        let mut pos: Vec<f32> = self.sim.read_positions().into_iter().flat_map(|p| p.into_iter()).collect();
        let mut vel: Vec<f32> = self.sim.read_velocities().into_iter().flat_map(|v| v.into_iter()).collect();
        let n_swapped = rebalance_swaps(&mut pos, &mut vel, self.sim.n_groups, self.w, self.n, 16);
        if n_swapped > 0 {
            self.sim.write_pos_vel(&pos, &vel);
        }
        self.last_swaps = n_swapped;
    }

    fn do_retile(&mut self) {
        let mut pos: Vec<f32> = self.sim.read_positions().into_iter().flat_map(|p| p.into_iter()).collect();
        let mut vel: Vec<f32> = self.sim.read_velocities().into_iter().flat_map(|v| v.into_iter()).collect();
        let n_retiled = rebalance_retile(&mut pos, &mut vel, self.sim.n_groups, self.w, self.n, 8);
        if n_retiled > 0 {
            self.sim.write_pos_vel(&pos, &vel);
        }
        self.last_retiles = n_retiled;
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
            ui.add(egui::Slider::new(&mut self.rebuild_interval, 10..=200).text("interval (frames)"));
            if ui.button("Morton rebuild").clicked() {
                self.do_morton_rebuild();
            }
            if ui.button("Greedy swaps").clicked() {
                self.do_greedy_swaps();
            }
            if ui.button("2W retiling").clicked() {
                self.do_retile();
            }
            ui.separator();
            ui.label(format!("Overlaps: {} | Max/grp: {}", self.n_overlap, self.max_overlap));
            ui.label(format!("Overflow groups: {} | Surf: {:.1}", self.n_overflow, self.total_surf));
            ui.label(format!("Last swaps: {} | retiles: {}", self.last_swaps, self.last_retiles));
            ui.separator();
            ui.label(format!("GPU step: {:.2} ms (broad: {:.2} narrow: {:.2})", self.ms_per_step, self.ms_broad, self.ms_narrow));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );
            let rect = response.rect;

            // Mouse interaction: read positions first (needed for picking)
            self.pos_read = self.sim.read_positions();

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

                let t_narrow = Instant::now();
                self.sim.step(self.dt, self.gravity, self.restitution, self.k_spring, self.k_damp, self.vel_damping, self.constrain_2d)
                    .expect("collision kernel enqueue failed");
                self.sim.pro_que.queue().finish().expect("queue finish failed");
                self.ms_narrow = t_narrow.elapsed().as_secs_f32() * 1000.0;
                self.ms_per_step = self.ms_broad + self.ms_narrow;
                self.frame_count += 1;

                if self.auto_rebuild && self.frame_count % self.rebuild_interval == 0 {
                    self.do_greedy_swaps();
                    self.do_retile();
                }
            }

            // Read back for rendering (positions already read above for picking)
            if !self.picked.is_some() {
                self.pos_read = self.sim.read_positions();
            }
            let _ = self.sim.compute_aabbs();
            let _ = self.sim.compute_overlaps();
            let (amins, amaxs) = self.sim.read_aabbs();
            self.aabb_min_read = amins;
            self.aabb_max_read = amaxs;

            // GPU-computed overlap stats
            let (_, overlap_counts) = self.sim.read_overlaps();
            let total_overlaps: i32 = overlap_counts.iter().sum();
            self.n_overlap = (total_overlaps / 2) as usize; // each pair counted from both sides
            self.max_overlap = overlap_counts.iter().map(|&c| c as usize).max().unwrap_or(0);
            self.n_overflow = overlap_counts.iter().filter(|&&c| c as usize > MAX_OVERLAP).count();

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
