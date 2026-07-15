//! demo11_collision_grid — OpenCL particle collision simulation with a uniform grid.
//!
//! This demo is the scalable comparison baseline for demo10's fixed group
//! partition. It rebuilds a bounded spatial index on the GPU, compacts all
//! associated particle records together, and lets each destination particle
//! gather from the surrounding 3x3 cells.
//!
//! The extra rebuild/scatter work is intentional: ownership is explicit,
//! crowded cells have no silent fixed-capacity overflow, and the resulting
//! timings can be compared against demo10's group-repair strategies. The
//! collision step still reads one immutable snapshot and writes another.

use eframe::egui;
use ocl::{flags, Buffer, Kernel, ProQue};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("collision_kernel.cl");
const SCAN_W: usize = 256;
const SCAN_ELEMS: usize = SCAN_W * 2;

struct ScanLevel {
    partial: Buffer<u32>,
    sums: Buffer<u32>,
    n: usize,
}

#[allow(dead_code)]
struct GridOcl {
    pro_que: ProQue,
    pos_a: Buffer<f32>,
    pos_b: Buffer<f32>,
    pos_sorted: Buffer<f32>,
    vel_a: Buffer<f32>,
    vel_b: Buffer<f32>,
    vel_sorted: Buffer<f32>,
    id_a: Buffer<u32>,
    id_b: Buffer<u32>,
    id_sorted: Buffer<u32>,
    key_current: Buffer<u32>,
    key_sorted: Buffer<u32>,
    cell_count: Buffer<u32>,
    cell_cursor: Buffer<u32>,
    candidate_count: Buffer<u32>,
    contact_count: Buffer<u32>,
    degenerate_count: Buffer<u32>,
    scan_levels: Vec<ScanLevel>,
    clear_cells_kernel: Kernel,
    clear_degenerate_kernel: Kernel,
    key_count_kernel: Kernel,
    scan_kernels: Vec<Kernel>,
    add_kernels: Vec<Kernel>,
    scatter_kernel: Kernel,
    collision_kernel: Kernel,
    current_is_a: bool,
    grid_ready: bool,
    n: usize,
    nx: usize,
    nz: usize,
    n_cells: usize,
    cell_h: f32,
    box_min: [f32; 3],
    box_max: [f32; 3],
}

impl GridOcl {
    fn new(n: usize, radius: f32, box_min: [f32; 3], box_max: [f32; 3]) -> ocl::Result<Self> {
        assert!(n > 0);
        assert!(
            n <= i32::MAX as usize,
            "particle count exceeds OpenCL int indexing"
        );
        assert!(
            radius.is_finite() && radius > 0.0,
            "particle radius must be finite and positive"
        );
        assert!(
            box_min.iter().chain(box_max.iter()).all(|x| x.is_finite()),
            "box bounds must be finite"
        );
        assert!(
            (0..3).all(|axis| box_max[axis] > box_min[axis]),
            "box maximum must exceed minimum on every axis"
        );
        let cell_h = 2.0 * radius;
        let nx = ((box_max[0] - box_min[0]) / cell_h).ceil() as usize;
        let nz = ((box_max[2] - box_min[2]) / cell_h).ceil() as usize;
        let n_cells = nx * nz;
        assert!(
            n_cells <= i32::MAX as usize,
            "dense grid exceeds OpenCL int indexing"
        );

        let mut rng = StdRng::seed_from_u64(0xD3A0_0011);
        let pos_host: Vec<f32> = (0..n)
            .flat_map(|_| {
                let x = rng.gen_range(box_min[0] + radius..box_max[0] - radius);
                let z = rng.gen_range(box_min[2] + radius..box_max[2] - radius);
                [x, 0.0, z, radius]
            })
            .collect();
        let vel_host: Vec<f32> = (0..n)
            .flat_map(|_| [rng.gen_range(-0.5..0.5), 0.0, rng.gen_range(-0.5..0.5), 1.0])
            .collect();
        let id_host: Vec<u32> = (0..n as u32).collect();

        let pro_que = ProQue::builder().src(KERNEL_SRC).dims(n).build()?;
        assert!(
            pro_que.max_wg_size()? >= SCAN_W,
            "OpenCL device maximum workgroup size is smaller than SCAN_W={SCAN_W}"
        );
        let queue = pro_que.queue().clone();
        let rw_f32 = |len: usize| -> ocl::Result<Buffer<f32>> {
            Buffer::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(len)
                .build()
        };
        let rw_u32 = |len: usize| -> ocl::Result<Buffer<u32>> {
            Buffer::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(len)
                .build()
        };
        let pos_a = Buffer::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(n * 4)
            .copy_host_slice(&pos_host)
            .build()?;
        let pos_b = rw_f32(n * 4)?;
        let pos_sorted = rw_f32(n * 4)?;
        let vel_a = Buffer::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(n * 4)
            .copy_host_slice(&vel_host)
            .build()?;
        let vel_b = rw_f32(n * 4)?;
        let vel_sorted = rw_f32(n * 4)?;
        let id_a = Buffer::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
            .len(n)
            .copy_host_slice(&id_host)
            .build()?;
        let id_b = rw_u32(n)?;
        let id_sorted = rw_u32(n)?;
        let key_current = rw_u32(n)?;
        let key_sorted = rw_u32(n)?;
        let cell_count = rw_u32(n_cells)?;
        let cell_cursor = rw_u32(n_cells)?;
        let candidate_count = rw_u32(n)?;
        let contact_count = rw_u32(n)?;
        let degenerate_count = rw_u32(1)?;

        let mut scan_levels = Vec::new();
        let mut scan_n = n_cells;
        while scan_n > 0 {
            let blocks = (scan_n + SCAN_ELEMS - 1) / SCAN_ELEMS;
            scan_levels.push(ScanLevel {
                partial: rw_u32(scan_n)?,
                sums: rw_u32(blocks)?,
                n: scan_n,
            });
            if blocks == 1 {
                break;
            }
            scan_n = blocks;
        }

        let clear_cells_kernel = pro_que
            .kernel_builder("clear_cell_buffers")
            .global_work_size(n_cells)
            .arg(&cell_count)
            .arg(&cell_cursor)
            .arg(n_cells as i32)
            .build()?;
        let clear_degenerate_kernel = pro_que
            .kernel_builder("clear_uint")
            .global_work_size(1usize)
            .arg(&degenerate_count)
            .arg(0u32)
            .arg(1i32)
            .build()?;
        let key_count_kernel = pro_que
            .kernel_builder("compute_cell_keys_and_count")
            .global_work_size(n)
            .arg(&pos_a)
            .arg(&key_current)
            .arg(&cell_count)
            .arg(n as i32)
            .arg(box_min[0])
            .arg(box_min[2])
            .arg(cell_h)
            .arg(nx as i32)
            .arg(nz as i32)
            .build()?;

        let mut scan_kernels = Vec::with_capacity(scan_levels.len());
        let mut input_buffer = cell_count.clone();
        for level in &scan_levels {
            let blocks = (level.n + SCAN_ELEMS - 1) / SCAN_ELEMS;
            scan_kernels.push(
                pro_que
                    .kernel_builder("scan_block")
                    .global_work_size(blocks * SCAN_ELEMS)
                    .local_work_size(SCAN_W)
                    .arg(&input_buffer)
                    .arg(&level.partial)
                    .arg(&level.sums)
                    .arg(level.n as i32)
                    .build()?,
            );
            input_buffer = level.sums.clone();
        }
        let mut add_kernels = Vec::new();
        for level in 0..scan_levels.len().saturating_sub(1) {
            let n_level = scan_levels[level].n;
            let global = ((n_level + SCAN_W - 1) / SCAN_W) * SCAN_W;
            add_kernels.push(
                pro_que
                    .kernel_builder("add_block_offsets")
                    .global_work_size(global)
                    .arg(&scan_levels[level].partial)
                    .arg(&scan_levels[level + 1].partial)
                    .arg(n_level as i32)
                    .build()?,
            );
        }

        let scatter_kernel = pro_que
            .kernel_builder("scatter_particles")
            .global_work_size(n)
            .arg(&pos_a)
            .arg(&vel_a)
            .arg(&id_a)
            .arg(&key_current)
            .arg(&scan_levels[0].partial)
            .arg(&cell_cursor)
            .arg(&pos_sorted)
            .arg(&vel_sorted)
            .arg(&id_sorted)
            .arg(&key_sorted)
            .arg(n as i32)
            .build()?;
        let collision_kernel = pro_que
            .kernel_builder("collision_step")
            .global_work_size(n)
            .arg(&pos_a)
            .arg(&vel_a)
            .arg(&id_a)
            .arg(&pos_b)
            .arg(&vel_b)
            .arg(&id_b)
            .arg(&key_sorted)
            .arg(&cell_count)
            .arg(&scan_levels[0].partial)
            .arg(&candidate_count)
            .arg(&contact_count)
            .arg(&degenerate_count)
            .arg(n as i32)
            .arg(nx as i32)
            .arg(nz as i32)
            .arg(0.005f32)
            .arg(0.0f32)
            .arg(0.0f32)
            .arg(9.81f32)
            .arg(0.5f32)
            .arg(box_min[0])
            .arg(box_min[1])
            .arg(box_min[2])
            .arg(box_max[0])
            .arg(box_max[1])
            .arg(box_max[2])
            .arg(1000.0f32)
            .arg(10.0f32)
            .arg(0.999f32)
            .arg(1i32)
            .build()?;

        Ok(Self {
            pro_que,
            pos_a,
            pos_b,
            pos_sorted,
            vel_a,
            vel_b,
            vel_sorted,
            id_a,
            id_b,
            id_sorted,
            key_current,
            key_sorted,
            cell_count,
            cell_cursor,
            candidate_count,
            contact_count,
            degenerate_count,
            scan_levels,
            clear_cells_kernel,
            clear_degenerate_kernel,
            key_count_kernel,
            scan_kernels,
            add_kernels,
            scatter_kernel,
            collision_kernel,
            current_is_a: true,
            grid_ready: false,
            n,
            nx,
            nz,
            n_cells,
            cell_h,
            box_min,
            box_max,
        })
    }

    fn build_grid(&mut self) -> ocl::Result<()> {
        let input_pos = if self.current_is_a {
            &self.pos_a
        } else {
            &self.pos_b
        };
        self.key_count_kernel.set_arg(0, input_pos)?;
        unsafe {
            self.clear_cells_kernel.enq()?;
        }
        unsafe {
            self.key_count_kernel.enq()?;
        }

        for kernel in &self.scan_kernels {
            unsafe {
                kernel.enq()?;
            }
        }
        for i in (0..self.add_kernels.len()).rev() {
            unsafe {
                self.add_kernels[i].enq()?;
            }
        }

        let input_vel = if self.current_is_a {
            &self.vel_a
        } else {
            &self.vel_b
        };
        let input_id = if self.current_is_a {
            &self.id_a
        } else {
            &self.id_b
        };
        self.scatter_kernel.set_arg(0, input_pos)?;
        self.scatter_kernel.set_arg(1, input_vel)?;
        self.scatter_kernel.set_arg(2, input_id)?;
        unsafe {
            self.scatter_kernel.enq()?;
        }
        self.grid_ready = true;
        Ok(())
    }

    fn step(
        &mut self,
        dt: f32,
        gravity: [f32; 3],
        restitution: f32,
        k_spring: f32,
        k_damp: f32,
        vel_damping: f32,
        constrain_2d: bool,
    ) -> ocl::Result<()> {
        assert!(
            self.grid_ready,
            "collision step requires build_grid() for the current particle state"
        );
        let (output_pos, output_vel, output_id) = if self.current_is_a {
            (&self.pos_b, &self.vel_b, &self.id_b)
        } else {
            (&self.pos_a, &self.vel_a, &self.id_a)
        };
        self.collision_kernel.set_arg(0, &self.pos_sorted)?;
        self.collision_kernel.set_arg(1, &self.vel_sorted)?;
        self.collision_kernel.set_arg(2, &self.id_sorted)?;
        self.collision_kernel.set_arg(3, output_pos)?;
        self.collision_kernel.set_arg(4, output_vel)?;
        self.collision_kernel.set_arg(5, output_id)?;
        self.collision_kernel.set_arg(15, dt)?;
        self.collision_kernel.set_arg(16, gravity[0])?;
        self.collision_kernel.set_arg(17, gravity[1])?;
        self.collision_kernel.set_arg(18, gravity[2])?;
        self.collision_kernel.set_arg(19, restitution)?;
        self.collision_kernel.set_arg(26, k_spring)?;
        self.collision_kernel.set_arg(27, k_damp)?;
        self.collision_kernel.set_arg(28, vel_damping)?;
        self.collision_kernel
            .set_arg(29, if constrain_2d { 1i32 } else { 0i32 })?;
        unsafe {
            self.clear_degenerate_kernel.enq()?;
        }
        unsafe {
            self.collision_kernel.enq()?;
        }
        self.current_is_a = !self.current_is_a;
        self.grid_ready = false;
        Ok(())
    }

    fn current_pos(&self) -> &Buffer<f32> {
        if self.current_is_a {
            &self.pos_a
        } else {
            &self.pos_b
        }
    }
    fn current_vel(&self) -> &Buffer<f32> {
        if self.current_is_a {
            &self.vel_a
        } else {
            &self.vel_b
        }
    }

    fn read_positions(&self) -> Vec<[f32; 4]> {
        let mut raw = vec![0.0f32; self.n * 4];
        self.current_pos()
            .read(&mut raw)
            .enq()
            .expect("read positions failed");
        (0..self.n)
            .map(|i| [raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
            .collect()
    }

    fn read_ids(&self) -> Vec<u32> {
        let mut raw = vec![0u32; self.n];
        let buf = if self.current_is_a {
            &self.id_a
        } else {
            &self.id_b
        };
        buf.read(&mut raw).enq().expect("read particle IDs failed");
        raw
    }

    fn read_velocities_flat(&self) -> Vec<f32> {
        let mut raw = vec![0.0f32; self.n * 4];
        self.current_vel()
            .read(&mut raw)
            .enq()
            .expect("read velocities failed");
        raw
    }

    fn write_pos_vel(&mut self, pos: &[f32], vel: &[f32]) {
        assert_eq!(pos.len(), self.n * 4);
        assert_eq!(vel.len(), self.n * 4);
        self.current_pos()
            .write(pos)
            .enq()
            .expect("write positions failed");
        self.current_vel()
            .write(vel)
            .enq()
            .expect("write velocities failed");
        self.grid_ready = false;
    }

    fn read_cell_counts(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.n_cells];
        self.cell_count
            .read(&mut counts)
            .enq()
            .expect("read cell counts failed");
        counts
    }

    fn read_cell_offsets(&self) -> Vec<u32> {
        let mut offsets = vec![0u32; self.n_cells];
        self.scan_levels[0]
            .partial
            .read(&mut offsets)
            .enq()
            .expect("read cell offsets failed");
        offsets
    }

    fn read_sorted_keys(&self) -> Vec<u32> {
        let mut keys = vec![0u32; self.n];
        self.key_sorted
            .read(&mut keys)
            .enq()
            .expect("read sorted cell keys failed");
        keys
    }

    fn read_pair_stats(&self) -> (u64, u64, u32) {
        let mut candidates = vec![0u32; self.n];
        let mut contacts = vec![0u32; self.n];
        let mut degenerate = vec![0u32; 1];
        self.candidate_count
            .read(&mut candidates)
            .enq()
            .expect("read candidate counts failed");
        self.contact_count
            .read(&mut contacts)
            .enq()
            .expect("read contact counts failed");
        self.degenerate_count
            .read(&mut degenerate)
            .enq()
            .expect("read degenerate count failed");
        (
            candidates.into_iter().map(u64::from).sum(),
            contacts.into_iter().map(u64::from).sum(),
            degenerate[0],
        )
    }
}

fn count_contacts_cpu(pos: &[[f32; 4]]) -> (u64, u64) {
    let mut contacts = 0u64;
    let mut degenerate = 0u64;
    for i in 0..pos.len() {
        for j in (i + 1)..pos.len() {
            let dx = pos[j][0] - pos[i][0];
            let dy = pos[j][1] - pos[i][1];
            let dz = pos[j][2] - pos[i][2];
            let d2 = dx * dx + dy * dy + dz * dz;
            let rsum = pos[i][3] + pos[j][3];
            if d2 < rsum * rsum {
                contacts += 1;
                if d2 <= 1.0e-12 {
                    degenerate += 1;
                }
            }
        }
    }
    (contacts, degenerate)
}

fn validate_grid_layout(counts: &[u32], offsets: &[u32], keys: &[u32], n: usize) {
    assert_eq!(counts.len(), offsets.len());
    let mut expected_offset = 0usize;
    for cell in 0..counts.len() {
        assert_eq!(
            offsets[cell] as usize, expected_offset,
            "exclusive scan mismatch at cell {cell}"
        );
        let end = expected_offset + counts[cell] as usize;
        assert!(end <= n, "cell {cell} range exceeds particle array");
        for &key in &keys[expected_offset..end] {
            assert_eq!(
                key as usize, cell,
                "particle scattered outside its cell range"
            );
        }
        expected_offset = end;
    }
    assert_eq!(
        expected_offset, n,
        "cell counts do not cover every particle exactly once"
    );
}

fn validate_particle_state(
    pos: &[[f32; 4]],
    vel: &[f32],
    ids: &[u32],
    box_min: [f32; 3],
    box_max: [f32; 3],
) {
    assert_eq!(vel.len(), pos.len() * 4);
    assert_eq!(ids.len(), pos.len());
    let mut seen = vec![false; pos.len()];
    for (i, p) in pos.iter().enumerate() {
        assert!(
            p.iter().all(|x| x.is_finite()),
            "non-finite position at slot {i}: {p:?}"
        );
        assert!(p[3] > 0.0, "non-positive radius at slot {i}: {}", p[3]);
        for axis in 0..3 {
            assert!(
                p[axis] >= box_min[axis] + p[3] - 1.0e-5
                    && p[axis] <= box_max[axis] - p[3] + 1.0e-5,
                "particle {i} outside wall bounds on axis {axis}: {}",
                p[axis]
            );
        }
        let v = &vel[4 * i..4 * i + 4];
        assert!(
            v.iter().all(|x| x.is_finite()),
            "non-finite velocity at slot {i}: {v:?}"
        );
        assert!(v[3] >= 0.0, "negative inverse mass at slot {i}: {}", v[3]);
        let id = ids[i] as usize;
        assert!(id < pos.len(), "particle ID out of range: {id}");
        assert!(!seen[id], "duplicate particle ID: {id}");
        seen[id] = true;
    }
    assert!(
        seen.into_iter().all(|x| x),
        "particle ID permutation has a hole"
    );
}

fn color_cell(cell: usize) -> egui::Color32 {
    let x = cell as u32;
    let hash = x.wrapping_mul(0x9E37_79B9).rotate_left(13) ^ 0xA341_316C;
    egui::Color32::from_rgb(
        (80 + (hash & 127)) as u8,
        (80 + ((hash >> 8) & 127)) as u8,
        (80 + ((hash >> 16) & 127)) as u8,
    )
}

struct GridApp {
    sim: GridOcl,
    dt: f32,
    gravity: [f32; 3],
    restitution: f32,
    k_spring: f32,
    k_damp: f32,
    vel_damping: f32,
    constrain_2d: bool,
    n: usize,
    pos_read: Vec<[f32; 4]>,
    ids_read: Vec<u32>,
    cell_counts_read: Vec<u32>,
    paused: bool,
    show_grid: bool,
    show_particles: bool,
    dots_only: bool,
    picked_id: Option<u32>,
    mouse_world: [f32; 2],
    mouse_world_prev: [f32; 2],
    have_grid: bool,
    frame_count: usize,
    ms_grid: f32,
    ms_collision: f32,
    occupied_cells: usize,
    max_occupancy: u32,
    candidate_pairs: u64,
    contact_pairs: u64,
    degenerate_contacts: u32,
    physics_has_run: bool,
}

impl GridApp {
    fn new(n: usize) -> Self {
        let box_min = [-20.0, -20.0, -20.0];
        let box_max = [20.0, 20.0, 20.0];
        let radius = 0.1;
        let sim =
            GridOcl::new(n, radius, box_min, box_max).expect("OpenCL grid initialization failed");
        let pos_read = sim.read_positions();
        let ids_read = sim.read_ids();
        let n_cells = sim.n_cells;
        Self {
            sim,
            dt: 0.005,
            gravity: [0.0, 0.0, 9.81],
            restitution: 0.5,
            k_spring: 1000.0,
            k_damp: 10.0,
            vel_damping: 0.999,
            constrain_2d: true,
            n,
            pos_read,
            ids_read,
            cell_counts_read: vec![0; n_cells],
            paused: false,
            show_grid: true,
            show_particles: true,
            dots_only: true,
            picked_id: None,
            mouse_world: [0.0; 2],
            mouse_world_prev: [0.0; 2],
            have_grid: false,
            frame_count: 0,
            ms_grid: 0.0,
            ms_collision: 0.0,
            occupied_cells: 0,
            max_occupancy: 0,
            candidate_pairs: 0,
            contact_pairs: 0,
            degenerate_contacts: 0,
            physics_has_run: false,
        }
    }

    fn refresh_readback(&mut self) {
        self.pos_read = self.sim.read_positions();
        self.ids_read = self.sim.read_ids();
        self.cell_counts_read = self.sim.read_cell_counts();
        self.occupied_cells = self.cell_counts_read.iter().filter(|&&x| x != 0).count();
        self.max_occupancy = self.cell_counts_read.iter().copied().max().unwrap_or(0);
        if self.physics_has_run {
            let (candidates, contacts, degenerate) = self.sim.read_pair_stats();
            self.candidate_pairs = candidates / 2;
            self.contact_pairs = contacts / 2;
            self.degenerate_contacts = degenerate;
        }
    }
}

impl eframe::App for GridApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Uniform Grid Collision");
            ui.label(format!("Device: {:?}", self.sim.pro_que.device()));
            ui.label(format!(
                "Particles: {} | Cells: {}x{}",
                self.n, self.sim.nx, self.sim.nz
            ));
            ui.label(format!("Cell width: {:.3} | stencil: 3x3", self.sim.cell_h));
            ui.separator();
            ui.add(egui::Slider::new(&mut self.dt, 0.0005..=0.02).text("dt"));
            ui.add(egui::Slider::new(&mut self.gravity[2], 0.0..=20.0).text("gravity z"));
            ui.add(egui::Slider::new(&mut self.restitution, 0.0..=1.0).text("restitution"));
            ui.add(egui::Slider::new(&mut self.k_spring, 100.0..=5000.0).text("k_spring"));
            ui.add(egui::Slider::new(&mut self.k_damp, 0.0..=50.0).text("k_damp"));
            ui.add(egui::Slider::new(&mut self.vel_damping, 0.95..=1.0).text("vel_damping"));
            ui.checkbox(&mut self.constrain_2d, "Constrain 2D (y=0)");
            ui.checkbox(&mut self.show_grid, "Show occupied cells");
            ui.checkbox(&mut self.show_particles, "Show particles");
            ui.checkbox(&mut self.dots_only, "Dots only (no radius)");
            ui.checkbox(&mut self.paused, "Paused");
            ui.separator();
            ui.label(format!(
                "Occupied cells: {} | Max occupancy: {}",
                self.occupied_cells, self.max_occupancy
            ));
            ui.label(format!(
                "Candidate pairs: {} | Contacts: {}",
                self.candidate_pairs, self.contact_pairs
            ));
            if self.degenerate_contacts == 0 {
                ui.label("Degenerate directed contacts: 0");
            } else {
                ui.colored_label(
                    egui::Color32::RED,
                    format!(
                        "Degenerate directed contacts: {} (deterministic fallback)",
                        self.degenerate_contacts
                    ),
                );
            }
            ui.label(format!(
                "GPU grid: {:.2} ms | collision: {:.2} ms",
                self.ms_grid, self.ms_collision
            ));
            ui.label(format!("Frame: {}", self.frame_count));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
            let rect = response.rect;
            let world_w = self.sim.box_max[0] - self.sim.box_min[0];
            let world_h = self.sim.box_max[2] - self.sim.box_min[2];
            let scale = rect.width().min(rect.height()) / world_w.max(world_h) * 0.9;
            let center = rect.center();
            let to_screen = |x: f32, z: f32| egui::pos2(center.x + x * scale, center.y + z * scale);
            let to_world = |p: egui::Pos2| [(p.x - center.x) / scale, (p.y - center.y) / scale];
            let mouse_pos = ui.input(|i| i.pointer.hover_pos());
            let lmb = ui.input(|i| i.pointer.primary_down());
            let rmb = ui.input(|i| i.pointer.secondary_down());
            self.mouse_world_prev = self.mouse_world;
            if let Some(mp) = mouse_pos.filter(|p| rect.contains(*p)) {
                self.mouse_world = to_world(mp);
            }

            if lmb && rect.contains(mouse_pos.unwrap_or(egui::Pos2::ZERO)) {
                if self.picked_id.is_none() {
                    let mut best = f32::MAX;
                    for (i, p) in self.pos_read.iter().enumerate() {
                        let dx = p[0] - self.mouse_world[0];
                        let dz = p[2] - self.mouse_world[1];
                        let d = dx * dx + dz * dz;
                        if d < best {
                            best = d;
                            self.picked_id = Some(self.ids_read[i]);
                        }
                    }
                }
                if let Some(id) = self.picked_id {
                    if let Some(i) = self.ids_read.iter().position(|&x| x == id) {
                        let mut pos = self
                            .pos_read
                            .iter()
                            .flat_map(|p| p.iter().copied())
                            .collect::<Vec<_>>();
                        let mut vel = self.sim.read_velocities_flat();
                        let particle_radius = pos[4 * i + 3];
                        let target_x = self.mouse_world[0].clamp(
                            self.sim.box_min[0] + particle_radius,
                            self.sim.box_max[0] - particle_radius,
                        );
                        let target_z = self.mouse_world[1].clamp(
                            self.sim.box_min[2] + particle_radius,
                            self.sim.box_max[2] - particle_radius,
                        );
                        pos[4 * i] = target_x;
                        pos[4 * i + 2] = target_z;
                        vel[4 * i] = (target_x - self.mouse_world_prev[0]) / self.dt;
                        vel[4 * i + 2] = (target_z - self.mouse_world_prev[1]) / self.dt;
                        self.sim.write_pos_vel(&pos, &vel);
                        self.have_grid = false;
                    }
                }
            } else if !lmb {
                self.picked_id = None;
            }

            if rmb && rect.contains(mouse_pos.unwrap_or(egui::Pos2::ZERO)) {
                let radius = 1.5f32;
                let strength = 50.0f32;
                let mut vel = self.sim.read_velocities_flat();
                for (i, p) in self.pos_read.iter().enumerate() {
                    let dx = p[0] - self.mouse_world[0];
                    let dz = p[2] - self.mouse_world[1];
                    let d = (dx * dx + dz * dz).sqrt();
                    if d < radius && d > 1.0e-6 {
                        let f = (1.0 - d / radius) * strength;
                        vel[4 * i] += dx / d * f * self.dt;
                        vel[4 * i + 2] += dz / d * f * self.dt;
                    }
                }
                self.sim.write_pos_vel(
                    &self
                        .pos_read
                        .iter()
                        .flat_map(|p| p.iter().copied())
                        .collect::<Vec<_>>(),
                    &vel,
                );
                self.have_grid = false;
            }

            if !self.paused || !self.have_grid {
                let t = Instant::now();
                self.sim.build_grid().expect("grid build failed");
                self.sim
                    .pro_que
                    .queue()
                    .finish()
                    .expect("grid queue finish failed");
                self.ms_grid = t.elapsed().as_secs_f32() * 1000.0;
                self.have_grid = true;
            }
            if !self.paused {
                let t = Instant::now();
                self.sim
                    .step(
                        self.dt,
                        self.gravity,
                        self.restitution,
                        self.k_spring,
                        self.k_damp,
                        self.vel_damping,
                        self.constrain_2d,
                    )
                    .expect("collision kernel failed");
                self.sim
                    .pro_que
                    .queue()
                    .finish()
                    .expect("collision queue finish failed");
                self.ms_collision = t.elapsed().as_secs_f32() * 1000.0;
                self.frame_count += 1;
                self.physics_has_run = true;
                self.have_grid = false;
            }
            self.refresh_readback();

            if self.show_grid {
                for cz in 0..self.sim.nz {
                    for cx in 0..self.sim.nx {
                        let cell = cx + self.sim.nx * cz;
                        if self.cell_counts_read[cell] == 0 {
                            continue;
                        }
                        let x0 = self.sim.box_min[0] + cx as f32 * self.sim.cell_h;
                        let z0 = self.sim.box_min[2] + cz as f32 * self.sim.cell_h;
                        let x1 = (x0 + self.sim.cell_h).min(self.sim.box_max[0]);
                        let z1 = (z0 + self.sim.cell_h).min(self.sim.box_max[2]);
                        let stroke = egui::Stroke::new(0.35, color_cell(cell));
                        painter.rect_stroke(
                            egui::Rect::from_two_pos(to_screen(x0, z0), to_screen(x1, z1)),
                            egui::Rounding::ZERO,
                            stroke,
                        );
                    }
                }
            }
            let p00 = to_screen(self.sim.box_min[0], self.sim.box_min[2]);
            let p11 = to_screen(self.sim.box_max[0], self.sim.box_max[2]);
            painter.rect_stroke(
                egui::Rect::from_two_pos(p00, p11),
                egui::Rounding::ZERO,
                egui::Stroke::new(2.0, egui::Color32::GRAY),
            );
            if self.show_particles {
                for (i, p) in self.pos_read.iter().enumerate() {
                    let cell_x = ((p[0] - self.sim.box_min[0]) / self.sim.cell_h)
                        .floor()
                        .clamp(0.0, (self.sim.nx - 1) as f32)
                        as usize;
                    let cell_z = ((p[2] - self.sim.box_min[2]) / self.sim.cell_h)
                        .floor()
                        .clamp(0.0, (self.sim.nz - 1) as f32)
                        as usize;
                    let selected = self.picked_id == Some(self.ids_read[i]);
                    let col = if selected {
                        egui::Color32::YELLOW
                    } else {
                        color_cell(cell_x + self.sim.nx * cell_z)
                    };
                    let sp = to_screen(p[0], p[2]);
                    if self.dots_only {
                        painter.circle_filled(sp, if selected { 4.0 } else { 2.0 }, col);
                    } else {
                        let rs = p[3] * scale;
                        painter.circle_filled(sp, rs, col);
                        painter.circle_stroke(
                            sp,
                            rs,
                            egui::Stroke::new(1.0, egui::Color32::from_black_alpha(120)),
                        );
                    }
                }
            }
            if rmb && rect.contains(mouse_pos.unwrap_or(egui::Pos2::ZERO)) {
                painter.circle_stroke(
                    to_screen(self.mouse_world[0], self.mouse_world[1]),
                    1.5 * scale,
                    egui::Stroke::new(1.0, egui::Color32::LIGHT_RED),
                );
            }
        });
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_contact_counter_handles_contact_and_separation() {
        let pos = [
            [0.0, 0.0, 0.0, 0.1],
            [0.15, 0.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, 0.1],
        ];
        assert_eq!(count_contacts_cpu(&pos), (1, 0));
    }

    #[test]
    fn compact_grid_ranges_cover_each_particle_once() {
        let counts = [2, 0, 3];
        let offsets = [0, 2, 2];
        let keys = [0, 0, 2, 2, 2];
        validate_grid_layout(&counts, &offsets, &keys, keys.len());
    }

    #[test]
    fn particle_state_requires_finite_bounded_unique_data() {
        let pos = [[-0.5, 0.0, 0.0, 0.1], [0.5, 0.0, 0.0, 0.1]];
        let vel = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let ids = [1, 0];
        validate_particle_state(&pos, &vel, &ids, [-1.0; 3], [1.0; 3]);
    }
}

fn main() -> eframe::Result {
    let n = 16_384usize;
    if std::env::args().any(|arg| arg == "--smoke") {
        let box_min = [-20.0, -20.0, -20.0];
        let box_max = [20.0, 20.0, 20.0];
        let n_smoke = 1_024usize;
        let mut sim = GridOcl::new(n_smoke, 0.1, box_min, box_max)
            .expect("OpenCL smoke initialization failed");
        for iteration in 0..3 {
            let input_pos = sim.read_positions();
            let (cpu_contacts, cpu_degenerate) = count_contacts_cpu(&input_pos);
            sim.build_grid().expect("smoke grid build failed");
            sim.pro_que
                .queue()
                .finish()
                .expect("smoke grid finish failed");
            let counts = sim.read_cell_counts();
            let offsets = sim.read_cell_offsets();
            let keys = sim.read_sorted_keys();
            validate_grid_layout(&counts, &offsets, &keys, n_smoke);
            sim.step(0.005, [0.0, 0.0, 9.81], 0.5, 1000.0, 10.0, 0.999, true)
                .expect("smoke collision enqueue failed");
            sim.pro_que
                .queue()
                .finish()
                .expect("smoke collision finish failed");
            let (directed_candidates, directed_contacts, directed_degenerate) =
                sim.read_pair_stats();
            assert_eq!(
                directed_candidates % 2,
                0,
                "candidate gather is not pair-symmetric"
            );
            assert_eq!(
                directed_contacts % 2,
                0,
                "contact gather is not pair-symmetric"
            );
            assert_eq!(
                directed_contacts / 2,
                cpu_contacts,
                "GPU/CPU contact mismatch at smoke iteration {iteration}"
            );
            assert_eq!(
                directed_degenerate as u64,
                2 * cpu_degenerate,
                "GPU/CPU degenerate-contact mismatch"
            );
            let output_pos = sim.read_positions();
            let output_vel = sim.read_velocities_flat();
            let output_ids = sim.read_ids();
            validate_particle_state(&output_pos, &output_vel, &output_ids, box_min, box_max);
            println!(
                "smoke iteration {iteration}: contacts={cpu_contacts}, candidates={}",
                directed_candidates / 2
            );
        }
        println!("Uniform-grid OpenCL smoke test passed");
        return Ok(());
    }
    println!("Initializing uniform-grid collision sim: {} particles", n);
    let app = GridApp::new(n);
    println!("OpenCL ready! Device: {:?}", app.sim.pro_que.device());
    eframe::run_native(
        "Uniform Grid Collision",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )
}
