use eframe::egui;
use nalgebra::{Vector3, Point2};
use rand::Rng;

struct Particle {
    pos: Vector3<f32>,
    vel: Vector3<f32>,
    mass: f32,
}

struct NBodyApp {
    particles: Vec<Particle>,
    g: f32,
    dt: f32,
    n: usize,
    softening: f32,
    // Camera
    rot_x: f32,
    rot_y: f32,
    zoom: f32,
}

impl Default for NBodyApp {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let n = 200;
        let particles: Vec<Particle> = (0..n).map(|_| Particle {
            pos: Vector3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
            vel: Vector3::new(rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1)),
            mass: rng.gen_range(0.1..1.0),
        }).collect();
        
        Self {
            particles,
            g: 0.5,
            dt: 0.01,
            n,
            softening: 0.05,
            rot_x: 0.0,
            rot_y: 0.0,
            zoom: 1.0,
        }
    }
}

impl NBodyApp {
    fn reset(&mut self) {
        *self = Self::default();
        self.n = self.particles.len();
    }

    fn integrate(&mut self) {
        let n = self.particles.len();
        let mut forces = vec![Vector3::<f32>::zeros(); n];
        
        for i in 0..n {
            for j in (i+1)..n {
                let r = self.particles[j].pos - self.particles[i].pos;
                let dist_sq = r.norm_squared() + self.softening * self.softening;
                let dist = dist_sq.sqrt();
                let f = self.g * self.particles[i].mass * self.particles[j].mass / (dist_sq * dist);
                let force_vec = f * r;
                forces[i] += force_vec;
                forces[j] -= force_vec;
            }
        }
        
        for i in 0..n {
            let acc = forces[i] / self.particles[i].mass;
            self.particles[i].vel += acc * self.dt;
            let vel = self.particles[i].vel;
            self.particles[i].pos += vel * self.dt;
        }
    }

    fn project(&self, p: &Vector3<f32>) -> Option<Point2<f32>> {
        // Simple rotation + perspective
        let cx = self.rot_x.cos();
        let sx = self.rot_x.sin();
        let cy = self.rot_y.cos();
        let sy = self.rot_y.sin();
        
        // Rotate around Y then X
        let x1 = cy * p.x + sy * p.z;
        let z1 = -sy * p.x + cy * p.z;
        let y1 = p.y;
        
        let x2 = x1;
        let y2 = cx * y1 - sx * z1;
        let z2 = sx * y1 + cx * z1;
        
        let distance = 3.0;
        let scale = self.zoom * distance / (distance + z2);
        if z2 > -distance + 0.1 {
            Some(Point2::new(x2 * scale + 0.5, y2 * scale + 0.5))
        } else {
            None
        }
    }
}

impl eframe::App for NBodyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("N-Body Controls");
            ui.add(egui::Slider::new(&mut self.g, 0.0..=2.0).text("G"));
            ui.add(egui::Slider::new(&mut self.dt, 0.001..=0.05).text("dt").logarithmic(true));
            ui.add(egui::Slider::new(&mut self.softening, 0.0..=0.2).text("Softening"));
            ui.add(egui::Slider::new(&mut self.rot_x, -3.14..=3.14).text("Rot X"));
            ui.add(egui::Slider::new(&mut self.rot_y, -3.14..=3.14).text("Rot Y"));
            ui.add(egui::Slider::new(&mut self.zoom, 0.1..=3.0).text("Zoom"));
            
            if ui.button("Reset").clicked() { self.reset(); }
            ui.label(format!("Particles: {}", self.particles.len()));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::hover(),
            );
            
            let rect = response.rect;
            
            // Physics step
            self.integrate();
            
            // Render particles
            for p in &self.particles {
                if let Some(proj) = self.project(&p.pos) {
                    // Manual coordinate transformation from [0,1] to screen rect
                    let screen_x = rect.min.x + proj.x * rect.width();
                    let screen_y = rect.min.y + proj.y * rect.height();
                    let screen_pos = egui::Pos2::new(screen_x, screen_y);
                    
                    let color = egui::Color32::from_rgb(
                        (p.mass * 255.0) as u8,
                        200,
                        (255.0 - p.mass * 200.0) as u8,
                    );
                    painter.circle_filled(screen_pos, 2.0 + p.mass * 3.0, color);
                }
            }
        });
        
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    eframe::run_native(
        "N-Body",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(NBodyApp::default()))),
    )
}
