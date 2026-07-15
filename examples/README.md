# Rust Physics & GPU Compute Examples

This directory contains progressive examples demonstrating Rust for physics simulations, GPU compute, and graphics programming, based on the notes in the parent NOTES/ directory.

## Workspace Configuration

**Important:** All examples are part of a Cargo workspace to share dependencies and build artifacts. This prevents duplicate compilation and saves disk space.

- **Global target directory:** All Rust projects on this computer share build artifacts in `$HOME/.cargo/shared_target/`
- Configured via environment variable in `~/.bashrc`: `export CARGO_TARGET_DIR="$HOME/.cargo/shared_target"`
- Works for any user (uses `$HOME` instead of hardcoded path)
- All dependencies (eframe, egui, wgpu, etc.) are shared across all Rust projects
- No individual `target/` directories in each example folder
- This is like system libraries in C++ - one copy for the entire computer

To build all examples:
```bash
cd /home/prokop/git/learn_Rust
cargo build
```

## Prerequisites

All examples require Rust to be installed. Some examples require specific system dependencies:

- **Demo 1, 2, 3, 6**: GUI examples using `eframe`/`egui` - work on Linux, macOS, Windows
- **Demo 4**: WGPU compute - requires Vulkan, Metal, or DirectX12 drivers
- **Demo 5**: Pure Rust - no system dependencies

## Running the Examples

Each example is in its own directory. To run:

```bash
cd examples/demo01_poisson_solver
cargo run
```

Or from the root:
```bash
cargo run -p demo01_poisson_solver
```

## Demo Descriptions

### Demo 1: 2D Poisson Solver with Live GUI Plot

**Location:** `demo01_poisson_solver/`

**Purpose:** Demonstrates solving the Poisson equation (∇²φ = ρ) for electrostatic potential using Jacobi iteration.

**Key Concepts:**
- `ndarray` for 2D grid operations (NumPy-like arrays)
- `egui` for immediate-mode GUI controls
- `egui_plot` for live data visualization
- Finite difference method for PDE solving

**Features:**
- Interactive sliders for charge position (X, Y) and charge value (Q)
- Adjustable iterations per frame
- Real-time 1D slice plot through the potential field
- Min/max value display for diagnostics

**Dependencies:**
- `eframe`, `egui`, `egui_plot` - GUI and plotting
- `ndarray` - Multi-dimensional arrays

**Run:**
```bash
cd demo01_poisson_solver && cargo run
```

---

### Demo 2: N-Body Gravity with 3D Projection & GUI

**Location:** `demo02_nbody/`

**Purpose:** Demonstrates O(N²) gravitational N-body simulation with 3D perspective projection.

**Key Concepts:**
- `nalgebra` for 3D vector math (Vector3)
- 3D to 2D projection with rotation matrices
- Real-time particle rendering with egui painter
- Borrow checker-friendly force accumulation pattern

**Features:**
- 200 particles interacting via gravity
- Camera controls (rotation X/Y, zoom)
- Adjustable parameters: G (gravity constant), dt (time step), softening
- Reset button to restart simulation
- Color-coded particles by mass

**Dependencies:**
- `eframe`, `egui` - GUI
- `nalgebra` - Linear algebra
- `rand` - Random initialization

**Run:**
```bash
cd demo02_nbody && cargo run
```

---

### Demo 3: OpenCL-style Particle Simulation

**Location:** `demo03_opencl/`

**Purpose:** Demonstrates particle simulation pattern that would use OpenCL for GPU acceleration.

**Key Concepts:**
- CPU integration (simulating OpenCL pattern)
- Boundary collision handling
- Concept of GPU compute acceleration

**Note:** This uses CPU integration for demonstration. For actual OpenCL GPU acceleration, you would use the `ocl` crate with OpenCL kernels. However, OpenCL+OpenGL interop is poorly supported in Rust - see Demo 4 for the modern approach.

**Features:**
- 500 particles with velocity and position
- Boundary bounce with damping
- Adjustable time step (dt)
- Visual rendering with egui

**Dependencies:**
- `eframe`, `egui` - GUI
- `ocl` - OpenCL bindings (included for reference)
- `rand` - Random initialization

**Run:**
```bash
cd demo03_opencl && cargo run
```

---

### Demo 4: WGPU Compute + Render (Modern GPU Approach)

**Location:** `demo04_wgpu_compute/`

**Purpose:** Demonstrates modern GPU compute and rendering using WGPU (WebGPU in Rust). This is the recommended replacement for OpenCL+OpenGL interop.

**Key Concepts:**
- WGPU compute shaders (WGSL) for GPU-side physics
- Dual-usage buffers (VERTEX + STORAGE) for zero-copy GPU-to-GPU data
- Same buffer used for compute and rendering (no CPU readback)
- WGSL compute shader updates positions, render pipeline draws them

**Why WGPU instead of OpenCL+OpenGL:**
- One API handles both compute and graphics
- Cross-platform (Vulkan, Metal, DirectX12, WebGPU)
- Better maintained in Rust ecosystem
- No platform-specific context sharing issues

**Features:**
- 10,000 particles computed entirely on GPU
- Compute shader handles position updates and boundary collisions
- Render pipeline draws particles as points
- Real-time performance with no CPU bottleneck

**Dependencies:**
- `wgpu` - Modern GPU abstraction
- `winit` - Windowing
- `pollster` - Async runtime
- `bytemuck` - Safe memory casting
- `rand` - Random initialization

**Run:**
```bash
cd demo04_wgpu_compute && cargo run
```

---

### Demo 5: Pointer Type Reinterpretation

**Location:** `demo05_pointer_reinterpret/`

**Purpose:** Demonstrates safe memory reinterpretation patterns for physics engines, replacing C++ union and pointer cast patterns.

**Key Concepts:**
- `#[repr(C)]` for C-compatible memory layout
- `bytemuck` for safe zero-cost type casting
- Replacing C++ unions with Rust methods
- "Numpy view" pattern: `&[Vec3]` ↔ `&[f64]`
- Forcefield-style kernels with flat arrays

**Examples Included:**
1. Cast Vec3 slice to flat f64 slice
2. Mutable casting for in-place modification
3. Named accessors (C++ union replacement)
4. Matrix column/row views
5. Quaternion force/energy views
6. Generic functions using traits
7. Forcefield-style kernel with flat arrays
8. Cast single struct to array

**Dependencies:**
- `bytemuck` - Safe memory casting

**Run:**
```bash
cd demo05_pointer_reinterpret && cargo run
```

---

### Demo 6: OpenCL+OpenGL Interop Concept

**Location:** `demo06_opencl_opengl_interop/`

**Purpose:** Explains the concept of OpenCL+OpenGL interop and why it's not recommended in Rust.

**Key Information:**
- OpenCL+OpenGL interop in Rust is poorly supported across platforms
- The `ocl-interop` crate is unmaintained
- Platform-specific context sharing is fragile in Rust
- WGPU (Demo 4) is the modern, maintained alternative

**This demo shows:**
- The conceptual pattern (CPU simulation)
- Explanation of why OpenCL+OpenGL interop is problematic in Rust
- Recommendation to use WGPU instead

**Dependencies:**
- `eframe`, `egui` - GUI
- `ocl` - OpenCL bindings (for reference)
- `rand` - Random initialization

**Run:**
```bash
cd demo06_opencl_opengl_interop && cargo run
```

---

### Demo 10: OpenCL Collision Balls with AABB Groups

**Location:** `demo10_collision_balls/`

**Purpose:** Demonstrates fixed-width particle groups, group AABBs, exact group-overlap classification, GPU narrow-phase collision, and CPU-side spatial rebalancing.

**Run:**
```bash
cargo run -p demo10_collision_balls
```

See its README and `ImprovementSuggestions.md` for the group-based architecture and rebalancing experiments.

---

### Demo 11: OpenCL Collision Grid

**Location:** `demo11_collision_grid/`

**Purpose:** Provides the uniform-grid comparison to Demo 10. Its main lesson is
that a cheap, regularly rebuilt spatial index can make neighbor ownership
clearer and more scalable than fixed AABB groups, while leaving the collision
response as a separate concern.

**Run:**
```bash
cargo run -p demo11_collision_grid
```

Run the GPU/CPU structural and contact-parity validation with:
```bash
cargo run -p demo11_collision_grid -- --smoke
```

This is a particle-cell neighbor-search demo, not a classical PIC field
solver. Read its README for the design rationale, interaction model, caveats,
and the current TODO list. Its current scope is equal-radius 2D soft-contact
particles.

---

## Learning Path

Recommended order for learning:

1. **Demo 5** - Start with memory patterns (no graphics, pure Rust)
2. **Demo 1** - Learn GUI and plotting with egui
3. **Demo 2** - Add 3D simulation and projection
4. **Demo 3** - Understand compute patterns (CPU version)
5. **Demo 4** - Modern GPU compute with WGPU
6. **Demo 6** - Understand why OpenCL+OpenGL interop is not recommended
7. **Demo 10** - Explore AABB-group collision broad phases and GPU/CPU rebalancing
8. **Demo 11** - Compare a compact uniform-grid particle-cell broad phase

## Key Crates Used

| Crate | Purpose | Used In |
|-------|---------|---------|
| `eframe`/`egui` | Immediate-mode GUI | 1, 2, 3, 6 |
| `egui_plot` | Plotting in egui | 1 |
| `ndarray` | N-dimensional arrays | 1 |
| `nalgebra` | Linear algebra | 2 |
| `wgpu` | Modern GPU compute/graphics | 4 |
| `winit` | Windowing | 4 |
| `ocl` | OpenCL bindings | 3, 6, 10, 11 |
| `bytemuck` | Safe memory casting | 4, 5 |
| `rand` | Random numbers | 2, 3, 4, 6, 10, 11 |

## Migration from C++

For those coming from C++ physics engines (like SimpleSimulationEngine/FireCore):

| C++ Pattern | Rust Equivalent |
|-------------|----------------|
| `double*` arrays | `ndarray` or flat `Vec<f64>` |
| `Vec3`, `Mat3`, `Quat` | `nalgebra` or custom with `#[repr(C)]` |
| Unions for type punning | `bytemuck` + methods |
| OpenCL kernels | WGPU compute shaders (WGSL) |
| OpenGL rendering | WGPU render pipeline |
| OpenCL+OpenGL shared VBO | WGPU buffer with `VERTEX \| STORAGE` |
| SDL2 + OpenGL | `winit` + `wgpu` |
| Dear ImGui | `egui` |

## Notes

The examples are based on the notes in the parent directory:
- `NOTES/rust_share_targets.md` - Cargo target sharing
- `NOTES/lern_Rust.md` - Learning resources and repos
- `NOTES/rust_examples_howto.md` - How to implement specific demos
- `NOTES/rust_pointer_type_reinterpret.md` - Memory safety and casting patterns
