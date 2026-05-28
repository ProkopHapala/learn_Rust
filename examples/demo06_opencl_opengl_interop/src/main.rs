use eframe::egui;
use rand::Rng;

struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

struct OpenCLInteropApp {
    particles: Vec<Particle>,
    dt: f32,
}

impl Default for OpenCLInteropApp {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let particles: Vec<Particle> = (0..500).map(|_| Particle {
            x: rng.gen_range(-1.0..1.0),
            y: rng.gen_range(-1.0..1.0),
            vx: rng.gen_range(-0.1..0.1),
            vy: rng.gen_range(-0.1..0.1),
        }).collect();
        
        Self {
            particles,
            dt: 0.01,
        }
    }
}

impl OpenCLInteropApp {
    fn integrate(&mut self) {
        // CPU integration (simulating what OpenCL would do)
        for p in &mut self.particles {
            p.x += p.vx * self.dt;
            p.y += p.vy * self.dt;
            
            if p.x < -1.0 || p.x > 1.0 {
                p.vx *= -0.8;
                p.x = p.x.clamp(-1.0, 1.0);
            }
            if p.y < -1.0 || p.y > 1.0 {
                p.vy *= -0.8;
                p.y = p.y.clamp(-1.0, 1.0);
            }
        }
    }
}

impl eframe::App for OpenCLInteropApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("OpenCL+OpenGL Interop Demo");
            ui.label("NOTE: OpenCL+OpenGL interop in Rust is");
            ui.label("poorly supported across platforms.");
            ui.label("The ocl-interop crate is unmaintained.");
            ui.label("");
            ui.label("For production code, use wgpu instead");
            ui.label("(see Demo 4).");
            ui.label("");
            ui.label("This demo shows the concept with CPU");
            ui.label("integration simulating the pattern.");
            ui.label("");
            ui.add(egui::Slider::new(&mut self.dt, 0.001..=0.05).text("dt"));
            ui.label(format!("Particles: {}", self.particles.len()));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::hover(),
            );
            
            let rect = response.rect;
            
            // Physics step (would be OpenCL in real interop)
            self.integrate();
            
            // Render particles (would use OpenGL VBO shared with OpenCL)
            for p in &self.particles {
                let screen_x = rect.min.x + (p.x + 1.0) * 0.5 * rect.width();
                let screen_y = rect.min.y + (p.y + 1.0) * 0.5 * rect.height();
                let screen_pos = egui::Pos2::new(screen_x, screen_y);
                
                let color = egui::Color32::from_rgb(255, 150, 100);
                painter.circle_filled(screen_pos, 3.0, color);
            }
        });
        
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    eframe::run_native(
        "OpenCL+OpenGL Interop Demo (Concept)",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(OpenCLInteropApp::default()))),
    )
}
