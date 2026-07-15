https://chatgpt.com/share/6a57f282-78c4-83eb-9fd0-9998c55578bc

# USER

please review following code, scrutinmize ti, both from algorthmc point of view and general perfromance OpenCL and and host both GPU and CPU

suggest how to improve it and what to implement from what we were just dicussion, but shortly conceptually, psudocode, math, algorithms, data layout, memory layout, parallel partitioning, synchornization points,  not actuall code

---

I reviewed both [`collision_kernel.cl`](sandbox:/mnt/data/collision_kernel.cl) and [`main.rs`](sandbox:/mnt/data/main.rs). 

## Overall verdict

The basic architecture is appropriate:

[
\text{contiguous groups of }W=32
\rightarrow
\text{group AABBs}
\rightarrow
\text{group-overlap graph}
\rightarrow
\text{per-particle gather}
]

But the current implementation has two correctness-critical problems:

1. **The collision kernel reads and writes `pos/vel` in place across workgroups.**
2. **Overlap lists truncate at 32 and silently miss collisions.**

There are also several concrete bugs in the CPU swapping code and substantial measurement overhead from the GUI/readback path.

---

# 1. Critical GPU race in `collision_step`

Each workgroup loads its own particle state at kernel entry, but later loads neighboring groups directly from global `pos` and `vel`. Meanwhile, other workgroups may already be updating those arrays near the end of the same kernel.

Therefore group (A) may evaluate (A)-(B) using old (B), while group (B) evaluates (B)-(A) using new (A). There is no global synchronization between workgroups inside an ordinary OpenCL kernel.

This breaks:

* determinism;
* approximate Newton-pair symmetry;
* momentum conservation;
* reproducibility across devices and launches.

## Best fix: ping-pong state buffers

Use:

```text
pos_in,  vel_in
pos_out, vel_out
```

The collision kernel reads only from `*_in` and writes only to `*_out`. After the kernel, swap buffer handles.

```text
AABB(pos_in)
overlap(pos_in AABBs)
collision_integrate(pos_in, vel_in, overlapBits -> pos_out, vel_out)
swap(in, out)
```

This preserves the current three-kernel structure and avoids an additional integration kernel.

The alternative is:

```text
collision_force(pos, vel -> force)
integrate(pos, vel, force)
```

but that adds a kernel launch and another global-memory pass. Ping-pong is preferable here.

---

# 2. Replace `MAX_OVERLAP=32` by an exact bit matrix

At 512 groups,

[
512\times512\text{ bits}=32768\text{ bytes}.
]

This is smaller than many ordinary GPU workgroup-local allocations and completely eliminates overflow.

Use:

```text
uint overlapBits[nGroups][ceil(nGroups/32)]
```

For 512 groups:

```text
uint overlapBits[512][16]
```

Then the collision workgroup for group (g) enumerates its exact row:

```text
for word in 0..15:
    mask = overlapBits[g][word]

    while mask != 0:
        bit = ctz(mask)
        h   = 32*word + bit
        process group h
        mask &= mask - 1
```

Properties:

* no overlap limit;
* no missed collisions;
* no dynamic allocation;
* no atomics;
* only 16 zero-word checks for a compact group;
* pathological groups degrade in performance but remain correct.

`nMaxOverlap=32` should become only a **repair threshold**:

```text
if degree[g] > 32:
    mark group pathological
```

It must not be a storage capacity.

## Simplest overlap-bit kernel

At this size I would first test an unusually simple mapping:

> One GPU work-item produces one 32-bit word of one overlap row.

There are

[
512\times16=8192
]

work-items. Each performs 32 AABB tests and writes one `uint`.

```text
g    = global_id_y
word = global_id_x

gBox = boxes[g]
mask = 0

for lane in 0..31:
    h = 32*word + lane
    if h < nGroups and h != g:
        if overlap(gBox, boxes[h]):
            mask |= 1 << lane

overlapBits[g][word] = mask
```

Advantages:

* no local memory;
* no barriers;
* no atomics;
* no subgroup extensions;
* perfectly deterministic;
* 8192 independent work-items are sufficient parallelism.

The complete 512-box array is only roughly 16 KB in the present two-`float4` layout, so it should cache well. Only introduce a more complicated (R\times32) tiled/subgroup-ballot version if profiling demonstrates that this kernel is important.

---

# 3. Keep the group-centric collision kernel

A unique pair list with one workgroup per pair would expose more parallelism, but it creates the problem of several workgroups contributing forces to the same particles. That requires:

* atomics;
* per-pair partial-force buffers plus a reduction;
* or graph coloring.

Your current architecture instead assigns exactly one workgroup to every destination group and gathers all forces for its 32 particles. That is a good no-atomic design.

With the bit matrix:

```text
one workgroup per source group g
one lane per particle i in g

load particle i from pos_in

process own group
enumerate all set bits in overlapBits[g]
    cooperatively load neighbor group h
    each lane accumulates force on its own i

integrate i
write pos_out[i], vel_out[i]
```

Each physical group pair is processed in both directions. This doubles arithmetic, but enables atomic-free gather and gives symmetric forces when both directions read the same input snapshot.

For (G=512), only 512 workgroups are launched. That is normally enough because each workgroup performs substantial work. If this later becomes under-parallelized, split high-degree rows into neighbor chunks and reduce partial forces, but do not add this complexity yet.

---

# 4. Specialize kernels for (W=32)

The code allocates local arrays of `MAX_W=256` although the actual workgroup size is 32.

For example, the collision kernel reserves approximately:

[
2\times256\times16=8192\text{ bytes}
]

rather than the needed:

[
2\times32\times16=1024\text{ bytes}.
]

The AABB `float3` arrays also normally have 16-byte stride in OpenCL local memory.

Use a compile-time group size:

```text
GROUP_SIZE = 32
```

and local arrays of exactly 32 elements. Also communicate the fixed workgroup size to the compiler.

Benefits:

* substantially less local memory;
* better occupancy;
* more aggressive loop unrolling;
* no runtime `W`;
* simpler bounds and indexing.

At minimum, host initialization must assert:

```text
W is power of two
W <= MAX_W
N % W == 0
```

The AABB reduction assumes a power-of-two size. The partial final group is not fully correct in the collision kernel because dummy particles participate in the intra-group loop.

Since exact warp filling is a core design requirement, padding the particle arrays to a multiple of 32 is cleaner than supporting variable final-group cardinality.

---

# 5. AABB data and kernel

The current AABB reduction is fundamentally reasonable. For 32 lanes, five reduction stages are cheap.

For the 2D simulation, one box can be represented as one `float4`:

[
(c_x,c_z,h_x,h_z)
]

rather than two `float4`s containing min and max.

Overlap becomes:

[
|c_{gx}-c_{hx}|\le h_{gx}+h_{hx},
]

[
|c_{gz}-c_{hz}|\le h_{gz}+h_{hz}.
]

This halves AABB bandwidth and storage. It is not essential at 512 groups, but gives a clean representation.

## Produce two kinds of boxes

For diagnosis and rebalancing, distinguish:

```text
tightBox = AABB of particle centers
broadBox = AABB enlarged by particle radii/cutoffs
```

Use:

* `broadBox` for collision correctness;
* `tightBox` for aspect ratio, compactness, outliers, and CPU repair scoring.

Otherwise a physically necessary interaction radius is mixed with actual group pathology.

## Worst-particle metadata

The AABB workgroup can also produce the top (K=4) outliers:

[
q_i=|\mathbf x_i-\mathbf c_{\rm tight}|^2
]

or a box-shaped measure:

[
q_i=
\max\left(
\frac{|x_i-c_x|}{h_x+\epsilon},
\frac{|z_i-c_z|}{h_z+\epsilon}
\right).
]

Store:

```text
worstIds[group][4]
worstScores[group][4]
```

This information is useful for rare CPU repair and costs little compared with reading all particles merely to discover candidates.

---

# 6. Concrete bugs in `rebalance_swaps`

The current outlier score is incorrect.

## Negative-side outliers are ignored

The code uses effectively:

```text
dx = max(x-centerX, 0)
dz = max(z-centerZ, 0)
```

Particles far toward negative (x) or negative (z) receive no distance contribution.

It must conceptually be:

[
d_x=x-c_x,\qquad d_z=z-c_z,
]

[
q=d_x^2+d_z^2.
]

## The extension score is always zero

The code calculates terms equivalent to:

[
\max(x_i+r_i-x_{\max},0),
]

but by definition:

[
x_i+r_i\le x_{\max}.
]

Therefore this term cannot be positive. The corresponding minimum-side terms are also always non-positive before clamping.

So the supposedly weighted wall-extension score contributes exactly zero.

## AABBs and centers become stale

`amins`, `amaxs`, and `centers` are computed once before all swaps. After one accepted swap:

* the group AABBs change;
* their centers change;
* subsequent target searches use obsolete geometry;
* `surf_before` refers to the original box, not the current one.

After every accepted swap, recompute at least the two affected group boxes and centers.

## Arithmetic float swapping is unsafe

The three-addition/subtraction swap:

```text
a = a+b
b = a-b
a = a-b
```

should not be used for floating-point state.

It can:

* lose low-order bits;
* overflow unnecessarily;
* propagate infinities or NaNs;
* slowly corrupt positions and velocities through repeated accept/revert cycles.

Use an ordinary temporary particle record or swap operation.

## Candidate search is too narrow

It chooses:

1. one globally ranked particle;
2. one target group containing its center;
3. one target particle closest to the original group center.

Instead, for a selected pair of groups, take (K) candidates from each and evaluate all:

[
K^2
]

swaps. With (K=4), this is only 16 candidates.

Recompute only the two candidate AABBs:

[
O(W)
]

rather than every group AABB:

[
O(N).
]

---

# 7. Improve `2W\rightarrow W+W` retiling

The current longest-Cartesian-axis median split is a reasonable baseline, but use the center-to-center projection for the main candidate.

For current group centers (\mathbf c_A,\mathbf c_B), calculate:

[
\Delta_i
========

## |\mathbf x_i-\mathbf c_A|^2

|\mathbf x_i-\mathbf c_B|^2.
]

Sort the 64 merged particles by (\Delta_i):

```text
first 32  -> A
second 32 -> B
```

Because

[
\Delta_i
========

2\mathbf x_i\cdot(\mathbf c_B-\mathbf c_A)
+
|\mathbf c_A|^2-|\mathbf c_B|^2,
]

this is a balanced planar cut perpendicular to the center-to-center direction.

Perform two iterations:

```text
compute centers
sort/split by Delta
recompute centers
sort/split again
```

For robustness, test only a few candidate partitions:

* old partition;
* split along (x);
* split along (z);
* split along (\mathbf c_B-\mathbf c_A).

Choose the cheapest. With 64 particles, this is negligible CPU work.

## Better acceptance cost

Do not evaluate only:

[
P(A)+P(B),
]

where (P) is perimeter.

Use the CPU’s complete AABB set and evaluate incident overlap changes:

[
C(A,B)=
\lambda[P(A)+P(B)]
+
\mu
\sum_{k\ne A,B}
\left[
I(A\cap k)+I(B\cap k)
\right].
]

This checks whether making the pair itself compact accidentally creates many new overlaps with surrounding groups.

---

# 8. Common-pool repair

For a pathological group with many neighbors:

1. Select it and perhaps its 4–8 strongest-overlap neighbors.
2. Every participating group releases its (K) worst particles.
3. Retained particles anchor each group.
4. Reassign the pool while preserving exactly (K) available slots per group.

A simple assignment cost is:

[
C_{ig}=|\mathbf x_i-\mathbf c_g|^2.
]

A more AABB-oriented cost is:

[
C_{ig}
======

\Phi(B_g\cup b_i)-\Phi(B_g),
]

where (\Phi) is perimeter, diagonal, or overlap-related cost.

For a small pool, use regret-greedy assignment:

```text
for every unassigned particle:
    find best and second-best non-full destination
    regret = secondCost - bestCost

assign particle with largest regret to its best destination
repeat
```

This handles three-group and longer migration cycles that pairwise swaps cannot resolve.

---

# 9. CPU repair pipeline

The CPU should use GPU-produced metadata rather than repeat the complete overlap search.

Read rarely:

```text
tight and broad AABBs
overlapBits                 // 32 KB at G=512
degree[group]
worstIds[group][K]
possibly particle state
```

Build unique edges using only bits with (h>g), score them, and process worst first.

```text
read GPU snapshot once

construct priority edges from overlapBits

for edge in descending severity:
    if pair is strongly interpenetrating:
        merge-sort-split
    else:
        evaluate KxK light swaps

for severe high-degree components:
    common-pool repair

write particle arrays once
rebuild AABBs and overlapBits
```

Do not currently run `do_greedy_swaps()` and then `do_retile()` as two independent GPU readback/upload cycles. Read positions and velocities once, perform all CPU operations on that snapshot, and upload once.

For much larger systems, first read only metadata. Once pathological groups are selected, gather their particle records into a compact GPU staging buffer and read only that buffer.

---

# 10. Host-side performance currently dominates the demonstration

The GUI path does considerably more work than the reported kernel timings reveal.

Every frame currently includes approximately:

* a complete position read before simulation;
* another complete position read after simulation;
* AABB min and max reads;
* overlap-list and overlap-count reads;
* a second AABB and overlap computation after the physics step;
* full velocity reads and writes during interaction;
* allocation and conversion of many temporary vectors;
* drawing 16,384 independent egui primitives.

This is not a reliable simulation-performance benchmark.

## Immediate host changes

### Read positions once, or less frequently

Use the previous rendered snapshot for mouse picking. Read a new position snapshot after simulation, perhaps only every few frames.

A mouse click need not trigger a fresh full-array read.

### Do not recompute broad phase twice

Currently broad phase is executed before collision and again after integration for visualization.

Either:

* display the pre-step AABBs with a one-step lag; or
* compute post-step AABBs once and reuse them as the next step’s broad phase, unless mouse interaction invalidates them.

### Do not read the fixed overlap list

Once using `overlapBits`, read only:

* counts for frequent statistics;
* the bit matrix only when CPU repair is considered.

### Use persistent host arrays

Avoid creating `Vec<f32>` and then converting it into `Vec<[f32;4]>` every read. Maintain persistent aligned `Float4` arrays.

### Mouse interaction

Writing all positions and velocities to move one particle is disproportionately expensive. Either:

* write only the selected buffer element using an offset;
* or enqueue a tiny interaction kernel.

The radial mouse force is also better implemented as a GPU kernel than by full velocity read-modify-write.

### Timing

`Instant + queue.finish()` measures submission plus synchronization and prevents normal queue pipelining. Use OpenCL profiling events for kernel execution times.

Maintain a separate headless benchmark mode with:

* no rendering;
* no per-frame readback;
* warm-up iterations;
* hundreds or thousands of timed simulation steps.

---

# 11. Smaller GPU issues

* The broad-phase complexity comment should say **(O(G^2)) total work**, not (O(G^2/W)). (O(G/W)) is closer to per-workgroup depth.
* The per-tile barrier in the current overlap kernel is unnecessary for the local atomic counter; one final barrier before reading `count` is sufficient. The bit-matrix design eliminates the atomic and this issue entirely.
* `force_buf` is only useful for diagnostics. Remove its global write in performance mode.
* Gravity is clearer and safer as direct acceleration:
  [
  \mathbf v\mathrel{+}=\mathbf g,dt.
  ]
  The current `gravity / inv_mass`, followed by multiplication by `inv_mass`, fails for fixed particles with `inv_mass=0`.
* The buffers are described as SoA, but `float4` per particle is an array of packed records. This is not a problem: aligned `float4` records are appropriate for the current kernels.
* Use `total_cmp` or explicit finite-value checks in CPU sorting. `partial_cmp(...).unwrap()` panics when the simulation produces a NaN.
* Kernel argument indices such as `set_arg(13, ...)` are fragile. At least centralize them as named constants.

---

# Recommended implementation order

1. **Fix the collision race with ping-pong `pos/vel`.**
2. **Replace truncated overlap lists by the exact 32 KB bit matrix.**
3. **Specialize all kernels for (W=32).**
4. **Remove duplicate per-frame broad phase and most readbacks.**
5. **Fix CPU swap bugs: signed distance, ordinary swap, current AABBs.**
6. **Use GPU overlap bits to prioritize CPU pair repairs.**
7. **Replace longest-axis-only retile with projection-based balanced split.**
8. **Add the local common-pool repair for high-degree pathological groups.**
9. **Only then optimize subgroup operations, local-memory loading, and collision arithmetic.**

The first two changes are correctness requirements. The next two will probably produce the largest measured performance improvement.

---

# USER

notice - not sure what attidude or desing decitions to have about overlaps beyond 32 neighboring collision groups. I think we should strive that this situation nover happens, or run sparate kernel to solver the rest. But we should not compromise performance of the base kernel. We should take this as safty measure whouch should happn rarely and we should try to tune balancing algorithm to avoid that.

---

# GPT 5.6 sol hi

I agree. **More than 32 overlapping groups should be treated as a pathological fallback condition, not as part of the normal kernel design.**

Do not burden every collision step with bitset iteration, CSR offsets, dynamic loops, or oversized neighbor storage merely to handle a state that the rebalancer should prevent.

The architecture should be:

[
\boxed{
\text{fast fixed-capacity path}
+
\text{rare exact overflow path}
+
\text{urgent rebalancing}
}
]

## 1. Normal representation

Per group:

```text
neighborCount[g]          // true count, may exceed 32
neighbors[g][32]          // valid only up to min(count,32)
overflowFlag[g]
```

The overlap-detection kernel must continue counting after 32, even though it stops writing partner IDs:

```text
count = 0

for all candidate groups h:
    if AABB_overlap(g,h):
        if count < 32:
            neighbors[g][count] = h
        count++

neighborCount[g] = count
overflowFlag[g] = count > 32
```

Thus overflow is detected exactly, but the common structure remains small.

The normal collision kernel should assume:

[
d_g\le32.
]

No overflow branch inside its hot interaction loop.

---

# 2. Partition groups before collision evaluation

After overlap detection, compact group IDs into two queues:

```text
normalGroups[]
overflowGroups[]
```

where:

[
g\in\text{normalGroups}\iff d_g\le32,
]

[
g\in\text{overflowGroups}\iff d_g>32.
]

This can be done by:

* a small GPU prefix scan;
* atomic append, because there are only 512 groups;
* or even CPU readback when overflow handling is already rare.

Then launch:

```text
collision_normal(normalGroups)
```

and only if necessary:

```text
collision_overflow(overflowGroups)
```

The normal kernel sees only valid fixed-size lists and remains fully optimized.

## Normal kernel

```text
one workgroup per normal group g

for k = 0 .. neighborCount[g)-1:
    h = neighbors[g][k]
    process 32×32 tile
```

The compiler knows the loop is bounded by 32. Local arrays and scheduling remain fixed.

## Overflow kernel

The overflow kernel can be slower and more general because it should almost never run.

For every overflow group (g), it can simply rescan all group AABBs:

```text
one workgroup per overflow group g

for h = 0 .. nGroups-1:
    if AABB_overlap(g,h):
        process group h
```

For (G=512), even a catastrophic group performs only 512 box tests before its particle interactions. That is entirely acceptable as a safety path.

This is probably simpler than constructing CSR or a complete bit matrix.

---

# 3. You may not need to store overflow neighbors at all

Because the number of groups is only 512, the fallback kernel can recompute the row of the overlap matrix on demand.

For one pathological group:

[
512
]

AABB tests are negligible compared with potentially:

[
d_g\times 32^2
]

particle-pair tests.

Therefore the fallback can use:

```text
for targetBlock = 0 .. 511 step 32:
    cooperatively test 32 target AABBs
    for each overlapping target:
        load target particles
        evaluate interactions
```

This avoids:

* overflow buffers;
* dynamic memory;
* scans;
* CSR construction;
* extra persistent storage.

Only if there are many overflow groups does this rescan become noticeable—and at that point rebalancing is already urgently required.

---

# 4. Correct interaction coverage

Your gather formulation processes forces separately for each destination group.

Therefore:

* a normal group processes forces acting on its own particles using its complete list of at most 32 neighbors;
* an overflow group processes forces acting on its own particles using the fallback full scan.

If normal group (A) overlaps overflow group (B):

* the normal kernel evaluates forces on particles in (A) due to (B);
* the overflow kernel evaluates forces on particles in (B) due to (A).

This remains correct, provided both kernels read from the same immutable input-state buffers and write to separate output-state slots.

There is no need to move all pairs touching an overflow group into the fallback kernel. Only the **source group’s degree** determines which kernel updates its particles.

---

# 5. Avoid divergence and extra registers in the base kernel

Do not write:

```text
if neighborCount[g] <= 32:
    normal path
else:
    overflow path
```

inside one large kernel.

Even when the branch is uniform within each workgroup, the combined kernel may still suffer from:

* larger register use;
* more instructions;
* reduced compiler specialization;
* more complex loop structure;
* larger code footprint.

Use two separately compiled kernels:

```text
collisionGroups32
collisionGroupsOverflow
```

The common kernel should contain no fallback logic other than consuming its supplied list of valid group IDs.

You can even compile it with a constant:

```text
MAX_OVERLAP = 32
```

and possibly unroll or partially unroll the neighbor iteration.

---

# 6. Overflow should trigger repair before it reaches 32

You should not wait until the hard capacity has already been exceeded.

Use two thresholds:

[
D_{\rm soft}<D_{\rm hard}=32.
]

For example:

[
D_{\rm soft}=24,\qquad D_{\rm hard}=32.
]

Interpretation:

* (d_g\le24): healthy;
* (24<d_g\le32): schedule rebalancing soon;
* (d_g>32): fallback kernel now and urgent repair.

This leaves headroom for motion between CPU rebalancing passes.

A stronger trigger can include the rate of growth:

[
d_g(t)-d_g(t-\Delta t)>D_{\rm growth}.
]

A group rapidly increasing from 15 to 27 overlaps is more alarming than a stable group with 26 neighbors.

---

# 7. Rebalancing priority

Overflow groups should enter the CPU repair queue with effectively infinite priority.

A useful ordering is:

[
P_g =
A\max(0,d_g-32)
+
B\max(0,d_g-D_{\rm soft})
+
C,\mathrm{aspect}(g)
+
D,\mathrm{size}(g).
]

With:

[
A\gg B,C,D.
]

Thus:

1. repair actual overflow first;
2. then repair groups approaching overflow;
3. then improve merely elongated or inefficient groups.

For an overflow group, pairwise swap may be too weak. Start with:

* merge–sort–split against its strongest-overlap partner;
* if its degree remains high, use a local multi-group pool;
* if still pathological, rebuild the connected local component.

---

# 8. What the overlap list should contain

For groups with (d_g\le32), all overlaps fit and ordering is irrelevant for correctness.

But sorting can still improve reuse or early diagnostics. Possible orderings are:

1. descending overlap area;
2. increasing center distance;
3. increasing group ID.

I would initially avoid GPU sorting. Append in group-index order because it is deterministic and cheap.

For CPU rebalancing, compute overlap severity separately from AABBs. The runtime collision list need not be sorted.

For overflow groups, do not trust the first stored 32 as the “important” subset for physics. The overflow kernel must rescan all groups. The truncated list can still be used as a diagnostic sample, but not for collision completeness.

---

# 9. Sparse overflow metadata

The only additional always-present data can be tiny:

```text
uint neighborCount[G]
uint overflowCount
uint overflowGroupIds[G]   // maximum capacity, usually almost empty
```

For (G=512), this is a few kilobytes.

Overlap detection can append overflow groups after it has completed each row:

```text
if count > 32:
    slot = atomic_inc(overflowCount)
    overflowGroupIds[slot] = g
```

An atomic operation once per pathological group is irrelevant. In the healthy case, there are no atomics executed except the failed branch test.

If avoiding atomics absolutely is preferred, write `overflowFlag[g]` and compact later. But here the atomic append is simpler and occurs only in an abnormal condition.

---

# 10. Safety behavior when overflow becomes widespread

Define a second global threshold:

[
N_{\rm overflow}^{\rm catastrophic}
]

or

[
E_{\rm estimated}^{\rm catastrophic}.
]

For example, when more than 5–10% of groups overflow:

```text
skip local repairs
perform broad Morton/local-component rebuild
```

Because if 100 of 512 groups exceed 32 neighbors, the system is no longer experiencing isolated defects. The grouping has globally degraded.

A possible policy:

```text
overflowCount == 0:
    normal kernel only

0 < overflowCount <= 8:
    normal kernel + overflow fallback
    schedule local CPU repair

8 < overflowCount <= 32:
    fallback remains correct
    perform local-component/common-pool repair immediately

overflowCount > 32:
    perform global or large-component rebuild
```

The exact numbers should be empirical, but the tiered behavior is useful.

---

# 11. Recommended design decision

For the present implementation I would choose:

```text
Fast persistent structure:
    neighbors[G][32]
    neighborCount[G]

Normal collision:
    fixed-capacity optimized kernel

Overflow detection:
    neighborCount[g] > 32

Overflow collision:
    separate kernel
    rescan all G AABBs for each overflow source group

Repair:
    soft threshold around 24–28
    hard threshold 32
    overflow groups repaired first
```

This preserves exactly the priority you want:

[
\boxed{
\text{optimize aggressively for the healthy state;
make pathology correct but deliberately slower.}
}
]

A dense bit matrix is a good general representation, but it is not necessary in the hottest path at this scale. It becomes useful only if overflow is common—which itself means the grouping algorithm is failing and should be repaired rather than normalized into the collision-kernel design.
