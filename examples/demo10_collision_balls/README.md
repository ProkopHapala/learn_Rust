# demo10_collision_balls

This demo studies a deliberately simple GPU collision architecture:

```text
contiguous particle groups
        -> group AABBs
        -> exact group-overlap map
        -> group-owned collision gathers
```

It is not intended to be a finished physics engine. It is a testing and
benchmarking platform for making the cost, ownership, synchronization, and
failure modes of alternative group-repair backends visible before comparing
the architecture with the uniform-grid approach in
[`demo11_collision_grid`](../demo11_collision_grid/).

## Purpose and motivation

The collision kernel works best when nearby particles are also nearby in the
flat particle array. A workgroup can then load one compact tile and reuse it
while checking many particle pairs. The difficulty is that particles move, so
the original grouping gradually becomes a poor spatial partition.

This demo explores how much can be gained by maintaining those fixed-size
groups without making the physics kernel responsible for changing its own
memory layout.

The default scene contains 16,384 particles in 512 groups of 32. It is a
2D simulation in the x-z plane; the buffers retain a 3D-shaped layout so the
data model remains compatible with the other examples.

## Design principles

The implementation is guided by a few constraints:

- Workgroups have exactly 32 active particles. Occupancy is part of the data
  structure, not an incidental optimization.
- A collision step reads one immutable particle snapshot and writes another.
  OpenCL workgroups cannot provide a general global barrier inside a kernel, so
  in-place neighbor reads are not safe.
- Each destination group owns its accumulated forces and output particles.
  A pair is evaluated from both sides; this costs arithmetic but avoids force
  atomics and ambiguous write ownership.
- Broad-phase data is conservative. A group AABB includes particle radii, so
  a non-overlapping pair of boxes is safe to skip, while an overlapping pair is
  only a candidate and still requires particle-level testing.
- Unexpected states are exposed. Overlap-map asymmetry, invalid degrees,
  non-finite state, and pathological group degrees are diagnostics, not silent
  fallback cases.

## Frame interaction

Each frame follows this conceptual sequence:

```text
current positions
    -> group AABBs
    -> exact overlap bit matrix and degrees
    -> normal collision kernel
    -> pathological collision kernel, if needed
    -> next positions
    -> swap current/next buffers
```

The normal and pathological kernels use the same snapshot and write disjoint
group ranges. A group with at most 32 neighboring groups uses the bounded fast
path. A group with more than 32 neighbors is not truncated or ignored: it is
handled by a separate complete path.

The value 32 is therefore a performance boundary and a repair signal, never a
correctness capacity.

## Why the overlap map is a bit matrix

The old fixed partner list made overflow a correctness bug: once a group had
too many partners, some collisions disappeared. The current representation
stores one exact bit per possible group pair. For 512 groups this is only 32 KiB
and gives deterministic degrees without atomics or dynamic allocation.

The same map is useful outside the collision kernel. When a repair is needed,
the CPU reads the GPU-produced map and considers only actual overlapping group
pairs. It does not rediscover the complete group graph with a second CPU-wide
search.

## Where repair enters the frame

Repair is inserted between broad phase and collision, after the map describes
the current particle snapshot:

```text
positions
    -> AABBs and exact map
    -> optional interval/degree trigger
    -> selected repair backend
    -> rebuild AABBs/map only if particle ownership changed
    -> collision kernels
    -> ping-pong swap
```

With auto-rebuild disabled, the extra repair path is absent. With it enabled,
the interval and degree trigger prevent the UI experiment from turning every
frame into a host synchronization benchmark. Manual repair uses the same
selected backend and then refreshes the broad-phase metadata before physics.

## Rebalancing and its interaction with physics

Rebalancing is intentionally outside the normal collision kernel. Changing
group membership while another workgroup is reading the groups would create
duplicate, missing, or partially updated particles.

The UI offers seven policies. Four are closely related retile algorithms whose
data paths can be compared directly:

- `CPU sequential retile` is the original full-readback reference. Accepted
  pairs update the AABB state seen by later pairs in the same pass.
- `CPU snapshot full-transfer` uses the parallel round semantics while still
  reading and writing the complete particle state. It is the fair CPU baseline
  for the hybrid and GPU backends.
- `Hybrid staged retile` selects disjoint pairs from the GPU map, gathers only
  their particle records, evaluates them on the CPU, and commits only the
  compact result.
- `GPU snapshot retile` uses the same disjoint pair plan and objective, but a
  64-work-item OpenCL workgroup loads, sorts, evaluates, and commits each pair
  without moving particle state to the host. The small pair plan is still
  selected on the CPU from the GPU map; moving that tiny decision is not the
  performance question being isolated here.

The remaining policies are deliberately different repair heuristics:

- `Greedy swaps` exchanges individual records between GPU-reported neighbors.
  It is cheap but can miss improvements that require several particles to move
  together.
- `Swaps + retile` applies both local heuristics in one CPU snapshot.
- `Morton rebuild` globally reorders all particles and is the broad repair
  fallback. It is effective but disruptive and expensive.

Repairs are interval-gated. The GPU fallback continues to preserve correctness
between repair intervals, so persistent pathology does not force a CPU repair
on every frame. The default trigger is deliberately below the hard degree
boundary, leaving room for the local repair to work before the slow path is
needed.

Accepted repairs preserve exact group occupancy and keep position/velocity
records together. Repair decisions first prefer fewer incident AABB overlaps,
then smaller perimeter. This is closer to the actual broad-phase cost than a
centroid-distance-only heuristic.

The hybrid and GPU implementations use Jacobi-style snapshot semantics: every
pair in a round is evaluated against the same pre-repair AABBs. This makes
parallel decisions reproducible and race-free. It is intentionally not
identical to the sequential CPU policy, where an earlier accepted pair can
change a later pair's decision. The selectable CPU snapshot implementation is
the canonical reference for hybrid/GPU parity.

Snapshot independence is a computational property, not a proof of global
optimality. Two individually improving pairs can still interact after both are
committed. The benchmark therefore reports the global overlap/perimeter change
instead of hiding regressions behind the accepted-pair count.

## Non-obvious context and caveats

The AABB is a broad collision box, not a diagnostic measure of grouping quality.
Neighboring compact groups can legitimately overlap because their particles
have finite radii. A future tight-box view of particle centers would help
distinguish expected contact neighborhoods from badly mixed groups.

The collision response is a penalty spring with damping and semi-implicit
integration. It is useful for studying neighbor-search behavior, but it is not
a hard-contact or continuous-collision solver. Large timesteps, high stiffness,
or near-coincident particles can still produce penetration, jitter, or unstable
motion.

The simulation is deterministic with respect to its logical ownership rules,
but floating-point execution and work-group scheduling on different OpenCL
devices may still produce different trajectories. The tests validate invariants
and parity conditions; they are not claiming bitwise cross-device trajectory
identity.

The visualizer performs readback and CPU drawing for inspection. Those costs
are not representative of a headless production simulation. The UI reports
estimated backend state-transfer bytes so the full CPU, compact hybrid, and
device-resident paths can be compared separately from rendering overhead.

The full GPU backend is not assumed to win. A normal repair round contains at
most 32 pair workgroups, often only a few, so launch and synchronization costs
can exceed the saved transfer time. The hybrid path also adds gather/commit
launches. Device type, particle count, pair count, and acceptance rate all
matter; this is why parity-gated measurement is part of the design.

The repair kernels never participate in the normal collision dispatch. This
keeps the base physics path easy to inspect and makes it possible to compare a
repair strategy's cost and grouping quality without quietly changing collision
ownership or adding repair-specific branches to every particle step.

## Open issues and unfinished work

The current design is a validated study implementation, not a finished
scalable partition manager. Open work includes:

- Add common-pool reassignment for cyclic misplacements involving more than two
  groups.
- Add cross-pair conflict prediction or a rollback round when snapshot-local
  improvements degrade the rebuilt global map.
- Unify swaps and retile into one bounded local repair algorithm.
- Add tight center boxes and better visualization of why a group is considered
  pathological.
- Explore whether a persistent/Verlet-style neighbor structure can reduce the
  per-frame broad-phase rebuild cost.
- Extend beyond equal-radius 2D particles. Variable radii require a conservative
  stencil or a different spatial hierarchy.
- Add stronger contact integration, substepping, or continuous collision
  detection for fast particles.
- Benchmark this architecture against the uniform grid across density,
  clustering, and pathological distributions.

## Validation and diagnostics

The tests cover exact overlap bits beyond 32 neighbors, symmetry and degree
validation, CPU record preservation during retile, immutable ping-pong input,
an OpenCL collision whose only meaningful contact is supplied through the
second bitset word, and exact CPU/hybrid/GPU snapshot-retile parity. The parity
fixtures cover both accepted multi-pair repair and rejected no-op behavior,
including exact particle-ID permutation and position/velocity record identity.

The ignored release benchmark measures broad phase and narrow phase, then runs
the same disjoint pair plan repeatedly through full-transfer CPU, compact
hybrid, and device-resident GPU backends. It checks exact final-state parity
before reporting average/minimum timings:

```bash
cargo test --release -p demo10_collision_balls headless_target_scale_benchmark -- --ignored --nocapture
```

For architectural comparison, see
[`demo11_collision_grid`](../demo11_collision_grid/) and the broader discussion
in [`PIC_gridHass_collision_acceleration.md`](../../NOTES/PIC_gridHass_collision_acceleration.md).
