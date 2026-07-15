# demo11_collision_grid

This demo asks a focused question: how much simpler and more scalable is a
uniform particle grid than the fixed AABB groups used by
[`demo10_collision_balls`](../demo10_collision_balls/)?

The answer is demonstrated with the smallest useful GPU design. Particles are
assigned to spatial cells, particles in each cell are made contiguous, and a
particle checks only its own cell and the eight neighboring cells. The grid is
therefore a broad-phase accelerator: it reduces the number of pairs that need
to be considered, but it does not decide the final collision response.

This is sometimes called a PIC-style grid in informal discussions, but this
program is not a classical particle-in-cell field solver. It does not deposit
particle quantities onto a grid or interpolate a grid field back to particles.
The grid is used only as a neighbor-search index.

## Why this design

The demo is intended as a transparent baseline for studying spatial
acceleration, not as a finished general-purpose physics engine. Its design
principles are:

- Use a bounded, dense grid so cell lookup is direct and easy to inspect.
- Keep the grid construction on the GPU and use compact cell ranges rather
  than a fixed per-cell capacity. No particles are silently dropped when a
  cell becomes crowded.
- Treat each frame as a snapshot. The collision kernel reads one immutable
  particle layout and writes a separate layout; this avoids relying on a
  global barrier that OpenCL workgroups do not provide.
- Let each destination particle gather its own response. A pair is evaluated
  from both sides, which costs some extra arithmetic but avoids floating-point
  force atomics and cross-workgroup write races.
- Keep the sorted particle arrays and their cell metadata together. Reading
  unsorted positions with sorted cell ranges is a subtle ownership error that
  can produce plausible-looking but invalid trajectories.

The important trade-off is deliberate: this version spends work rebuilding a
correct neighbor index every frame in exchange for simple ownership and clear
validation. It is a reference point for later measurements such as Verlet
lists, radix sorting, or occupancy-specialized kernels.

## Frame interaction

At each simulation step the host launches this pipeline:

```text
clear cell counts and write cursors
compute cell keys and counts
hierarchical exclusive scan of counts
scatter particles into compact cell ranges
gather contacts in the 3x3 neighborhood
integrate into the next particle snapshot
swap current and next snapshots
```

The default scene is a 2D view of a bounded `[-20, 20]^2` domain with equal
radii. The third coordinate is retained in the buffers because the shared
particle layout is 3D-shaped, while the demo constrains motion to the `x-z`
plane.

Run the interactive demo with:

```bash
cargo run -p demo11_collision_grid
```

The controls expose timestep, gravity, restitution, spring stiffness, contact
damping, velocity damping, 2D constraint, grid/particle display, and pause.
While the pointer is over the scene, left-drag moves nearby particles and
right-drag applies a radial force. The occupied-cell display is useful for
seeing the broad-phase structure, not for judging physical accuracy.

For a short headless correctness check, run:

```bash
cargo run -p demo11_collision_grid -- --smoke
```

The smoke path compares the GPU neighbor result with an exhaustive CPU
reference and checks scan offsets, compact-range coverage, stable particle
IDs, pair symmetry, and finite bounded state. It is the preferred first check
after changing the kernel or buffer layout.

## Non-obvious caveats

The cell width is `2 * radius`, and the nine-cell stencil is correct for the
current equal-radius model. Changing particle sizes without changing the
stencil can miss contacts. Extending the domain or using sparse occupancy also
requires revisiting the dense-grid indexing assumptions.

The response is an explicit soft spring/damping penalty. It is useful for
visualizing broad-phase behavior, but it is not a hard-contact solver and has
no continuous collision detection. Large timesteps or excessive stiffness can
therefore cause penetration, jitter, or tunneling. A near-coincident pair uses
a deterministic ID-based antisymmetric normal so the simulation remains
defined; this removes a numerical singularity, not the underlying physical
ambiguity. The UI reports how often this fallback is used.

Cell construction uses integer atomics, so the order within a cell is not
guaranteed to be bitwise deterministic across devices. The algorithm is
bounded and checked, but trajectories should not be treated as reproducible
reference data. GUI readback and CPU drawing are diagnostic overhead and are
not included in the displayed GPU kernel timings.

## Open issues and TODO

The current demo is complete as a validated baseline, but the following work
is intentionally open:

- Add a Verlet/skin neighbor list so the grid is not rebuilt on every step.
- Measure the baseline against demo10 and separate grid-build cost from
  collision cost under different particle distributions.
- Add substepping or continuous collision detection for fast particles and a
  more stable hard-contact or iterative constraint response.
- Support variable radii with a conservative stencil, then evaluate whether a
  multilevel grid is worthwhile.
- Add occupancy-adaptive handling for unusually crowded cells and a sparse
  hashed representation for domains that should not allocate a dense grid.
- Provide a deterministic radix-sort construction when reproducible ordering
  is required.
- Reduce visualization readback, or add a GPU-native rendering path, before
  using the demo as a performance benchmark.

The broader design discussion and alternatives are in
[`PIC_gridHass_collision_acceleration.md`](../../NOTES/PIC_gridHass_collision_acceleration.md).
