# demo10_collision_balls — OpenCL Collision Simulation with Group-Based Broad Phase

## Overview

GPU-accelerated 2D particle collision simulation using OpenCL, with host-side
spatial rebalancing to maintain collision detection performance over time.

**Scale**: 16,384 particles (2^14), 512 groups of W=32. Box [-20,20]^3, 2D mode (y=0).

## Architecture: GPU/CPU Split

### GPU (OpenCL) — every frame

| Kernel | Purpose | Complexity |
|--------|---------|-----------|
| `compute_aabbs` | Per-group AABB (min/max via tree reduction in local memory) | O(W) per group |
| `compute_overlaps` | Tiled group-vs-all-groups AABB overlap detection | O(m) per group, O(m^2/W) total |
| `collision_step` | Narrow-phase collision forces + integration, using overlap list for pruning | O(W^2 * mColAv) per group |

**Execution order per frame**:
1. `compute_aabbs` — from current particle positions
2. `compute_overlaps` — from AABBs
3. `collision_step` — uses overlap list to skip non-overlapping groups

### CPU (Rust) — rare, on-demand

| Operation | When | Purpose |
|-----------|------|---------|
| `sort_by_morton` | Init + button press | Full spatial reorder via Morton Z-curve |
| `rebalance_swaps` | Button / auto-rebuild | Greedy pairwise particle swaps to reduce AABB surface |
| `rebalance_retile` | Button / auto-rebuild | 2W->W+W merge-sort-split for overlapping pairs |
| Mouse interaction | Every frame (if active) | LMB drag particle, RMB radial force |
| Quality metrics display | Every frame | Read overlap_count buffer, compute total surface |

## Broad-Phase Algorithm

Particles are partitioned into fixed-size groups of W=32. Each group has an AABB
that includes particle radii (broad box). Two groups whose broad AABBs do not
overlap cannot contain colliding particle pairs, so their W x W interaction is
skipped entirely.

**AABB margin**: The AABB is computed as `[p.xyz - r, p.xyz + r]` per particle,
so the group box already includes the collision radius. Two groups overlap iff
any particle pair could be within `2R` — exactly the collision-relevant
threshold. No additional margin is needed.

**Complexity**: With m groups, W group size, and mColAv average overlaps:
- Broad phase: O(m^2 / W) (tiled, each workgroup scans all m groups)
- Narrow phase: O(W^2 * m * mColAv) vs brute-force O(W^2 * m^2)
- With m=512, mColAv~8: ~64x fewer particle-particle checks

## Overlap List Representation

Each group stores up to `MAX_OVERLAP=32` partner group IDs in a fixed-size array.
The true overlap count is stored separately in `overlap_count[g]`.

**Overflow handling**: If a group has more than 32 overlapping partners, the list
is truncated (non-deterministic which partners are kept, due to atomic increment
ordering). The count is always correct. Collisions with unlisted groups are
**missed** — this is a correctness gap, not just a performance issue.

**Overflow as pathology signal**: A group exceeding MAX_OVERLAP is pathologically
spread out. The `Overflow groups` counter in the UI should trigger aggressive
rebalancing (Morton rebuild or common-pool reassignment). In normal operation
with compact groups, mColAv should be ~4-12.

## Rebalancing Heuristics (CPU)

### Morton Z-curve Rebuild
Full O(N log N) sort of all particles by 2D Morton code. Produces optimal
spatial locality but is expensive and disrupts all groups simultaneously.

### Greedy Swaps
For each group, find the particle that most extends its AABB. If that particle
falls inside another group's AABB, find a swap partner and accept if total
AABB surface decreases. Limited to `max_swaps` per call.

**Caveat**: Greedy one-at-a-time swaps can miss cooperative improvements where
two particles need to move together for any gain. The balanced merge-split
approach (planned) handles this better.

### 2W->W+W Retiling
For overlapping group pairs (sorted by overlap area), merge all 2W particles,
sort by the longest axis of the combined bounding box, split at median.
Accept if total surface decreases.

**Caveat**: Sorting by the longest single axis (x or z) is suboptimal when
groups are offset diagonally. The planned improvement is to sort by projection
onto the line connecting group centers (Delta_i = |x_i - c_A|^2 - |x_i - c_B|^2),
which is the optimal balanced partition for fixed centers.

## Non-Obvious Design Decisions

1. **Groups are contiguous index ranges** (group g = particles [g*W, (g+1)*W)).
   No separate group membership array. Rebalancing = permuting particles in
   the flat pos/vel arrays. This keeps GPU kernels simple (group_id = global_id / W).

2. **AABB computed redundantly on CPU** inside rebalancing functions
   (`compute_group_aabbs_host`) to evaluate swap/retile quality without
   round-tripping to GPU. This is intentional — rebalancing is rare and
   needs to evaluate many candidate swaps quickly.

3. **2D mode only** (y=0 constraint). The overlap kernel checks only x,z axes.
   Extending to 3D requires adding y-axis checks in `compute_overlaps` and
   `aabb_overlap`.

4. **Intra-group collisions checked unconditionally** (g vs g, W^2 pairs).
   This is always needed and cheap (W=32, so 1024 checks per group).

5. **Wall collision unrolled per-axis** because OpenCL C doesn't support
   dynamic vector component indexing (`.s[axis]` with variable axis).

## Open Issues

- **Overlap list overflow = missed collisions**: When a group has >32 partners,
  collisions with truncated partners are silently skipped. This is a correctness
  issue under severe pathology. Mitigation: trigger rebuild before overflow.

- **No tight-box vs broad-box distinction**: Currently AABBs include particle
  radius (broad box). For diagnosing grouping quality, tight boxes (centers only)
  would be better — neighboring compact groups are *expected* to have overlapping
  broad boxes, but tight-box overlap indicates actual misassignment.

- **Rebalancing reads/writes entire pos+vel arrays**: For 16k particles this is
  256KB per array, transferred every rebalance call. Could be optimized to
  transfer only affected groups.

- **No priority queue for repairs**: Overlapping pairs are processed in order of
  raw overlap area. A pathology score (N_overlap * (D/D_typical)^2) would better
  prioritize world-spanning groups.

## TODO

- [ ] Replace axis-based retile split with balanced Delta_i projection onto inter-center line
- [ ] Unify swaps + retile into single merge-split function (inspect k particles crossing)
- [ ] Add pathology score for repair prioritization
- [ ] Add common-pool multi-group reassignment for cyclic misassignments
- [ ] Add tight-box computation (particle centers only) for diagnosis
- [ ] GPU kernel to apply permutation (gather) instead of CPU writing full arrays
- [ ] Two-pass overlap kernel: count pass + fill pass for variable-size lists
- [ ] Benchmark broad vs narrow phase timing at different mColAv levels

## References

- `NOTES/BoundingBoxBalancing.md` — theoretical analysis of rebalancing strategies
- `examples/demo09_collision_debug/` — earlier CPU-only collision debug visualization
