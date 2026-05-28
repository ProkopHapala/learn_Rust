use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use ndarray::Array2;

struct PoissonApp {
    grid: Array2<f32>,      // potential phi
    rho: Array2<f32>,       // charge density
    n: usize,
    charge_x: f32,
    charge_y: f32,
    charge_q: f32,
    iterations_per_frame: usize,
}

impl Default for PoissonApp {
    fn default() -> Self {
        let n = 64;
        let mut rho = Array2::<f32>::zeros((n, n));
        // Place a point charge in the center
        rho[[n/2, n/2]] = 1.0;
        Self {
            grid: Array2::zeros((n, n)),
            rho,
            n,
            charge_x: 0.5,
            charge_y: 0.5,
            charge_q: 1.0,
            iterations_per_frame: 10,
        }
    }
}

impl PoissonApp {
    fn jacobi_step(&mut self) {
        let n = self.n;
        let dx = 1.0 / n as f32;
        let dx2 = dx * dx;
        let mut new = self.grid.clone();
        
        // Interior points only
        for i in 1..n-1 {
            for j in 1..n-1 {
                new[[i, j]] = 0.25 * (
                    self.grid[[i+1, j]] + self.grid[[i-1, j]] +
                    self.grid[[i, j+1]] + self.grid[[i, j-1]] -
                    dx2 * self.rho[[i, j]]
                );
            }
        }
        // Boundary conditions (Dirichlet: phi=0 at edges)
        for i in 0..n {
            new[[i, 0]] = 0.0;
            new[[i, n-1]] = 0.0;
            new[[0, i]] = 0.0;
            new[[n-1, i]] = 0.0;
        }
        self.grid = new;
    }

    fn update_charge(&mut self) {
        self.rho.fill(0.0);
        let cx = (self.charge_x * (self.n - 1) as f32) as usize;
        let cy = (self.charge_y * (self.n - 1) as f32) as usize;
        let cx = cx.clamp(1, self.n - 2);
        let cy = cy.clamp(1, self.n - 2);
        self.rho[[cy, cx]] = self.charge_q;
    }
}

impl eframe::App for PoissonApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Poisson Solver: ∇²φ = ρ");
            
            // --- GUI Controls (Demo 3) ---
            ui.horizontal(|ui| {
                ui.label("Charge X:");
                ui.add(egui::Slider::new(&mut self.charge_x, 0.0..=1.0));
                ui.label("Charge Y:");
                ui.add(egui::Slider::new(&mut self.charge_y, 0.0..=1.0));
                ui.label("Charge Q:");
                ui.add(egui::Slider::new(&mut self.charge_q, -2.0..=2.0));
            });
            ui.horizontal(|ui| {
                ui.label("Iterations/frame:");
                ui.add(egui::Slider::new(&mut self.iterations_per_frame, 1..=100));
            });
            
            if ui.button("Reset Grid").clicked() {
                self.grid.fill(0.0);
            }

            // Update charge position from GUI
            self.update_charge();

            // --- Run Solver ---
            for _ in 0..self.iterations_per_frame {
                self.jacobi_step();
            }

            // --- Plot 1D slice through center (Demo 1) ---
            let n = self.n;
            let mid = n / 2;
            let points: PlotPoints = (0..n)
                .map(|i| [i as f64, self.grid[[mid, i]] as f64])
                .collect();
            
            let line = Line::new(points).name("φ(x, y=0.5)");
            Plot::new("potential_slice")
                .height(300.0)
                .show(ui, |plot_ui| plot_ui.line(line));
            
            // Show min/max for diagnostics
            ui.label(format!("φ_min: {:.3e}, φ_max: {:.3e}", 
                self.grid.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                self.grid.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))));
        });
        
        // Keep animating
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Poisson Solver",
        options,
        Box::new(|_cc| Ok(Box::new(PoissonApp::default()))),
    )
}
