use eframe::egui;
use ocl::{ProQue, Buffer, flags};
use rand::Rng;
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("nbody_kernel.cl");

// float4 = [x, y, z, w] where w = mass for pos, w = unused for vel/force
struct NBodyOcl {
    pro_que: ProQue,
    pos_buf: Buffer<f32>,   // n * 4 floats (float4 per particle)
    vel_buf: Buffer<f32>,
    force_buf: Buffer<f32>,
    n: usize,
    kernel: ocl::Kernel,
}

impl NBodyOcl {
    fn new(n: usize, dt: f32, softening: f32) -> ocl::Result<Self> {
        let mut rng = rand::thread_rng();
        let pos_host: Vec<f32> = (0..n).flat_map(|_| {
            let x = rng.gen_range(-0.8..0.8);
            let y = rng.gen_range(-0.8..0.8);
            let z = rng.gen_range(-0.3..0.3);
            let m = rng.gen_range(0.5..2.0);
            [x, y, z, m]
        }).collect();
        let vel_host: Vec<f32> = (0..n).flat_map(|_| {
            let vx = rng.gen_range(-0.05..0.05);
            let vy = rng.gen_range(-0.05..0.05);
            let vz = 0.0f32;
            [vx, vy, vz, 0.0]
        }).collect();

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

        let kernel = pro_que.kernel_builder("nbody_step")
            .arg(&pos_buf)
            .arg(&vel_buf)
            .arg(&force_buf)
            .arg(n as i32)
            .arg(dt)
            .arg(softening * softening)
            .build()?;

        Ok(Self { pro_que, pos_buf, vel_buf, force_buf, n, kernel })
    }

    fn step(&mut self, dt: f32, softening: f32) -> ocl::Result<()> {
        // Update kernel args that can change at runtime
        self.kernel.set_arg(4, dt)?;
        self.kernel.set_arg(5, softening * softening)?;
        unsafe { self.kernel.enq()?; }
        Ok(())
    }

    fn read_positions(&self) -> Vec<[f32; 3]> {
        let mut buf = vec![0.0f32; self.n * 4];
        self.pos_buf.read(&mut buf).enq().expect("read pos_buf failed");
        (0..self.n).map(|i| [buf[i*4], buf[i*4+1], buf[i*4+2]]).collect()
    }
}

struct OpenCLApp {
    sim: NBodyOcl,
    dt: f32,
    softening: f32,
    n: usize,
    ms_per_step: f32,
    pos_read: Vec<[f32; 3]>,
    initialized: bool,
}

impl OpenCLApp {
    fn new(n: usize) -> Self {
        let dt = 0.005;
        let softening = 0.05;
        let sim = NBodyOcl::new(n, dt, softening)
            .expect("Failed to init OpenCL N-body — is an OpenCL runtime installed?");
        Self {
            sim,
            dt,
            softening,
            n,
            ms_per_step: 0.0,
            pos_read: vec![[0.0; 3]; n],
            initialized: true,
        }
    }
}

impl eframe::App for OpenCLApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("OpenCL N-Body (real GPU)");
            if self.initialized {
                ui.label(format!("Device: {:?}", self.sim.pro_que.device()));
            }
            ui.label(format!("Particles: {}", self.n));
            ui.add(egui::Slider::new(&mut self.dt, 0.0005..=0.02).text("dt"));
            ui.add(egui::Slider::new(&mut self.softening, 0.01..=0.3).text("softening"));
            ui.label(format!("GPU step: {:.2} ms", self.ms_per_step));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::hover(),
            );
            let rect = response.rect;

            // GPU step
            let t0 = Instant::now();
            self.sim.step(self.dt, self.softening).expect("kernel enqueue failed");
            self.sim.pro_que.queue().finish().expect("queue finish failed");
            self.ms_per_step = t0.elapsed().as_secs_f32() * 1000.0;

            // Read back positions for rendering
            self.pos_read = self.sim.read_positions();

            // Project 3D -> 2D (simple orthographic with mild rotation)
            let angle = 0.3f32;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            for p in &self.pos_read {
                let x2 = p[0] * cos_a - p[2] * sin_a;
                let y2 = p[1];
                let screen_x = rect.min.x + (x2 + 1.0) * 0.5 * rect.width();
                let screen_y = rect.min.y + (y2 + 1.0) * 0.5 * rect.height();
                let screen_pos = egui::Pos2::new(screen_x, screen_y);
                let color = egui::Color32::from_rgb(100, 200, 255);
                painter.circle_filled(screen_pos, 2.5, color);
            }
        });

        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    let n = 1024;
    println!("Initializing OpenCL n-body with {} particles...", n);
    let app = OpenCLApp::new(n);
    println!("OpenCL ready! Device: {:?}", app.sim.pro_que.device());
    eframe::run_native(
        "OpenCL N-Body",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(app))),
    )
}
