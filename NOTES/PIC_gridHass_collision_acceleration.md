https://chatgpt.com/share/6a57eb77-6890-83eb-9f02-33537306d1da

# USER

What is the current state-of-te-art method how to accelrate collision (or other short range contact interaction) between particles accelrated by uniforma grid (kinda like pairtice-in-cell (PIC)

there are some articles I found starting from classica GPU gems spaticall-hasing which I tak as a base

https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda

https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus

https://www.reddit.com/r/GraphicsProgramming/comments/a901xv/hash_table_particle_collision_w_compute_shaders/

https://arxiv.org/html/2406.16091v1

https://www.mdpi.com/2673-3951/7/1/27

https://github.com/iweinbau/UnityGPUDynamicHashGrid

https://scispace.com/pdf/fast-uniform-grid-construction-on-gpgpus-using-atomic-1vd4i0m6zf.pdf

read thos but sinf some other, perhaps you know better this is quite profound problem

an you review it and write pseudo codes and how they differ, where are advantages/distandvatrages, tradeoffs and use cases?

---

# GPT 5.6 sol hi

# Bottom line

For **similar-size particles with a fixed short interaction cutoff**, the current practical GPU baseline is still a **uniform cell-linked grid**, but not usually the literal linked-list “spatial hash” used in older implementations.

A modern implementation normally combines:

1. **Compact cell construction**

   * either `count → prefix sum → scatter`,
   * or `compute cell key → radix sort particles by key`.

2. **Contiguous particles within each cell**

   * not pointer-linked lists,
   * not fixed-capacity buckets unless the occupancy bound is guaranteed.

3. **Particle-centric gather**

   * one thread per target particle,
   * scan neighboring cells,
   * accumulate only the target force,
   * usually evaluate each pair twice to avoid atomics.

4. **Verlet buffering / dual neighbor lists**

   * rebuild the spatial structure less often,
   * prune a larger outer list into a smaller active list more frequently.

5. **Spatial reordering and structure-of-arrays**

   * particles near each other in space should also be near each other in memory.

For high cell occupancy, the interaction kernel may switch to **one block per cell or cluster pair**, cooperatively loading source particles into shared/local memory. For strong size disparity, nonuniform density, or widely varying cutoffs, a **BVH, multilevel grid, or SFC/octree cluster list** can outperform a uniform grid.

So the state of the art is not a new replacement for the uniform grid. It is a hierarchy of layouts and execution strategies around the same underlying neighborhood problem.

---

# 1. Three separate problems are often confused

It is useful to distinguish:

### A. Spatial-index construction

Given particle positions, produce something equivalent to

```text
cell -> contiguous range of particle indices
```

### B. Candidate enumeration

For particle (i), visit all cells whose spatial extent can overlap its cutoff sphere, then test individual particles.

### C. Interaction accumulation or collision resolution

After finding the pair ((i,j)), determine where the result is written:

```text
force[i] += Fij
force[j] -= Fij
```

Many papers optimize only one of these. For example, a paper may show a fast interaction kernel while excluding sorting or grid rebuilding. Another may optimize particle-to-grid deposition rather than particle-particle interaction.

The original GPU Gems algorithms established many of the core ideas—cell decomposition, sorting by cell, compact cell ranges and conflict-free processing—but some implementation choices were dictated by early CUDA and graphics hardware. ([NVIDIA Developer][1])

---

# 2. Recommended default: sorted compact grid

For your typical case—many similar particles, cutoff (r_c), bounded simulation box, OpenCL, preference to avoid atomics—I would start here.

## Grid geometry

For grid spacing (h),

[
\mathbf c_i =
\left\lfloor
\frac{\mathbf x_i-\mathbf x_{\min}}{h}
\right\rfloor .
]

For a dense bounded grid,

[
k_i=c_x+n_x(c_y+n_yc_z).
]

For a sparse or very large domain, use a packed integer coordinate or Morton key. A hash bucket alone is not an identity: when multiple spatial cells map to the same bucket, the **full cell coordinate/key must still be compared**.

The simplest choice is

[
h \approx r_c,
]

giving a (3^3=27)-cell stencil in 3D.

Smaller cells reduce the number of particle candidates but increase the number of cell headers visited:

[
N_{\rm stencil}
===============

\left(2\left\lceil\frac{r_c}{h}\right\rceil+1\right)^3.
]

For example:

* (h=r_c): 27 cells,
* (h=r_c/2): 125 cells,
* (h=r_c/3): 343 cells.

The best (h) depends on occupancy and interaction cost; modern packages often tune bin dimensions rather than assuming exactly (h=r_c). ([LAMMPS Documentation][2])

---

## 2.1 Atomic-free sort construction

```cpp
// One work item per particle
kernel makeCellKeys(
    int n,
    float4* pos,
    ulong* cellKey,
    uint* particleId
){
    int i = get_global_id(0);
    if(i >= n) return;

    int3 c = floor_int3((pos[i].xyz - gridOrigin) * invCellSize);

    // Dense linear index, Morton key, or packed signed coordinates
    cellKey[i]   = encodeCell(c);
    particleId[i] = i;
}

// GPU radix sort pairs:
//     (cellKey, particleId) sorted by cellKey
radixSortPairs(cellKey, particleId, n);
```

After sorting, construct cell ranges.

```cpp
kernel markCellBoundaries(
    int n,
    ulong* sortedKey,
    uint* cellStart,
    uint* cellEnd
){
    int k = get_global_id(0);
    if(k >= n) return;

    ulong key = sortedKey[k];

    if(k == 0 || sortedKey[k-1] != key){
        cellStart[key] = k;
    }

    if(k == n-1 || sortedKey[k+1] != key){
        cellEnd[key] = k + 1;
    }
}
```

There is no race because each cell has exactly one first and one last element.

For a sparse domain, you normally do not allocate `cellStart[key]` over the entire coordinate space. Instead:

```cpp
boundary[k] = (k == 0) || (key[k] != key[k-1]);

activeCellId[k] = exclusiveScan(boundary);

if(boundary[k]){
    uniqueCellKey[activeCellId[k]] = key[k];
    activeStart [activeCellId[k]] = k;
}
```

Then neighbor-cell lookup uses:

* a compact open-addressed hash from `fullCellKey → activeCellId`,
* binary search in `uniqueCellKey`,
* or a dense lookup table covering only the active bounding box.

A global bitonic sort, as used by some educational GPU hash-grid projects, is simple but scales as (O(N\log^2N)). A radix sort is normally the appropriate global sort for integer cell keys. The Unity project you linked is useful pedagogically, but its bitonic sorting and finite hashed bucket space are not the layout I would choose for a large production simulation. ([GitHub][3])

---

## 2.2 Reorder particle data

After sorting, either retain an indirection:

```cpp
j = particleId[k];
xj = pos[j];
```

or physically reorder frequently accessed arrays:

```cpp
kernel reorderParticles(
    uint* sortedParticleId,
    float4* posOld,
    float4* velOld,
    float4* posSorted,
    float4* velSorted
){
    int k = get_global_id(0);
    int i = sortedParticleId[k];

    posSorted[k] = posOld[i];
    velSorted[k] = velOld[i];
}
```

Physical reordering is usually worthwhile when the list is reused for several force evaluations. Adjacent GPU lanes then process spatially adjacent particles and query many of the same cell ranges.

A useful compromise is:

* keep stable physical particle IDs,
* store simulation arrays in current spatial order,
* maintain `spatialIndex → particleID` and `particleID → spatialIndex`.

---

# 3. Particle-centric gather: the robust default kernel

```cpp
kernel shortRangeForces(
    int n,
    float4* pos,            // spatially sorted
    uint* cellStart,
    uint* cellEnd,
    float4* force
){
    int i = get_global_id(0);
    if(i >= n) return;

    float3 xi = pos[i].xyz;
    int3 ci   = positionToCell(xi);

    float3 fi = (float3)(0.0f);

    for(int dz=-1; dz<=1; dz++){
        for(int dy=-1; dy<=1; dy++){
            for(int dx=-1; dx<=1; dx++){

                int3 cj = ci + (int3)(dx,dy,dz);
                uint key = linearCellIndex(cj);

                uint begin = cellStart[key];
                uint end   = cellEnd[key];

                for(uint k=begin; k<end; k++){
                    if(k == i) continue;

                    float3 d  = pos[k].xyz - xi;
                    float r2  = dot(d,d);

                    if(r2 < cutoff2){
                        fi += evaluatePairForce(d, r2, i, k);
                    }
                }
            }
        }
    }

    force[i] = (float4)(fi, 0.0f);
}
```

This has several important properties:

* no force atomics,
* no races,
* regular ownership: each work item writes one output,
* works naturally with Jacobi-like collision or constraint iterations,
* simple enough for the compiler to optimize,
* often faster than more elaborate cell-centric kernels when each cell contains only a few particles.

The 2024 GPU interaction study you linked found exactly this kind of result: particle-centric kernels were competitive or superior on NVIDIA hardware for low and moderate cell occupancy, while cell/block kernels suffered from idle lanes when occupancy did not match warp size. Shared-memory cell tiling became clearly useful mainly at much higher occupancy in their tested cases. These thresholds are hardware- and potential-dependent rather than universal. ([arXiv][4])

---

## Why computing every pair twice can be faster

The kernel above computes both

[
F_{ij}
]

when thread (i) visits (j), and

[
F_{ji}
]

when thread (j) visits (i).

This doubles pair arithmetic, but avoids:

* atomics to `force[j]`,
* temporary pair-force storage,
* segmented reductions,
* cell coloring,
* synchronization between workgroups.

For cheap pair potentials, duplicated arithmetic is frequently less expensive than irregular writes. LAMMPS GPU modes similarly support configurations in which pair contributions are duplicated to avoid conflicting updates—commonly described as “Newton off.” ([LAMMPS Documentation][2])

The disadvantages are:

* approximately twice the pair evaluations,
* total momentum is not guaranteed to cancel bit-for-bit because the two evaluations are independent,
* expensive contact models or complex anisotropic interactions may make duplicate evaluation unattractive.

For a symmetric force law in single precision, the momentum drift is usually a roundoff-level issue. For exact pairwise momentum conservation, compute each pair once and reduce its two signed contributions separately.

---

# 4. Count–scan–scatter: often the fastest grid construction

If a few integer atomics are acceptable, the standard compact-cell builder is:

```cpp
// Step 1: count
fill(cellCount, 0);

kernel countParticlesPerCell(){
    int i = get_global_id(0);
    int c = cellOf(pos[i]);

    atomic_inc(&cellCount[c]);
}

// Step 2: offsets
exclusiveScan(cellCount, cellOffset);

// Step 3: scatter
fill(cellCursor, 0);

kernel scatterParticles(){
    int i = get_global_id(0);
    int c = cellOf(pos[i]);

    int local = atomic_inc(&cellCursor[c]);
    sortedParticleId[cellOffset[c] + local] = i;
}
```

Its advantages:

* linear expected work,
* no global sort,
* compact contiguous cell ranges,
* straightforward dense-grid implementation.

Its disadvantages:

* atomics become contended in highly occupied cells,
* order inside a cell is nondeterministic,
* it does not automatically spatially reorder particles between neighboring cells beyond the cell grouping,
* clearing large dense arrays can itself be expensive when most cells are empty.

The 2024 paper describes this count/prefix/scatter construction as the conventional GPU cell-list build. ([arXiv][4])

For your preference to avoid atomics, radix sorting is reasonable. But these are integer atomics used only during infrequent index construction, not floating-point force atomics inside the hot pair loop. I would benchmark them rather than rejecting them categorically.

A useful improvement is workgroup-local aggregation:

```cpp
// Each workgroup processes a spatial or input tile.
// Accumulate counts for repeated cell IDs locally,
// then issue one global atomic add per distinct cell.
```

This only helps when a workgroup’s particles exhibit enough spatial coherence. If input order is random, local aggregation provides little reduction.

---

# 5. Atomic linked-cell list

The simplest dynamic hash-grid construction is:

```cpp
fill(cellHead, -1);

kernel insertParticles(){
    int i = get_global_id(0);
    int c = cellOf(pos[i]);

    next[i] = atomic_exchange(&cellHead[c], i);
}
```

Traversal is then:

```cpp
for(int j = cellHead[c]; j != -1; j = next[j]){
    ...
}
```

## Advantages

* one pass,
* one atomic exchange per particle,
* no prefix scan,
* no sorting,
* useful when the structure must be rebuilt every step,
* supports dynamic insertion naturally.

## Disadvantages

* pointer chasing,
* incoherent memory access,
* cell contents appear in arbitrary order,
* poor cache-line utilization,
* no easy vectorized range iteration,
* nondeterministic ordering,
* more difficult to tile into local/shared memory.

The “fast uniform grid construction using atomic operations” work was motivated by the increasing speed of global atomics on then-new GPUs. That observation remains valid for construction, but in my view its linked traversal is usually less attractive than compact contiguous ranges for a repeatedly evaluated particle simulation. That last judgment is an architectural inference rather than a universal result. ([SciSpace][5])

I would use linked lists when:

* construction dominates,
* only one or very few queries follow each build,
* the particle count per cell is tiny,
* or particles are inserted/deleted dynamically.

---

# 6. Fixed-capacity cell arrays

A common real-time implementation is:

```cpp
cellCount[c] = atomic_inc(...);

if(cellCount[c] < MAX_PER_CELL){
    cellParticles[c * MAX_PER_CELL + cellCount[c]] = i;
}else{
    overflow = true;
}
```

This is extremely fast when a rigorous bound exists. It is also dangerous when the bound is merely “normally sufficient.”

Options for handling overflow include:

1. slow fallback list,
2. secondary overflow buffer,
3. rebuilding with a larger capacity,
4. recursively subdividing the overloaded cell,
5. putting excess particles into a BVH,
6. maintaining two-level pages per cell.

The old GPU Gems rigid-particle example stored only a small fixed number of particles per voxel using texture channels and multiple rendering passes. The 27-neighbor-cell idea survives, but the fixed RGBA storage and graphics-pipeline construction are now historical implementation details. ([NVIDIA Developer][6])

For molecular dynamics or deformable particle systems, I would not use a hard `MAX_PER_CELL` without an exact physical occupancy bound.

---

# 7. Cell-centric and tiled kernels

When cells contain many particles, one block can own one target cell.

```cpp
kernel cellBlockForces(){
    int targetCell = get_group_id(0);
    int lane       = get_local_id(0);

    int targetBegin = cellStart[targetCell];
    int targetEnd   = cellEnd[targetCell];

    // One or several target particles per lane
    for(int ti = targetBegin + lane;
            ti < targetEnd;
            ti += get_local_size(0))
    {
        float3 xi = pos[ti].xyz;
        float3 fi = 0;

        for(each neighborCell){

            int sourceBegin = cellStart[neighborCell];
            int sourceEnd   = cellEnd[neighborCell];

            for(int tile=sourceBegin;
                    tile<sourceEnd;
                    tile+=LOCAL_SIZE)
            {
                int sj = tile + lane;

                if(sj < sourceEnd){
                    localPos[lane] = pos[sj];
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                int tileCount = min(LOCAL_SIZE, sourceEnd-tile);

                for(int q=0; q<tileCount; q++){
                    float3 d = localPos[q].xyz - xi;

                    if(dot(d,d) < cutoff2){
                        fi += pairForce(d);
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        force[ti] = fi;
    }
}
```

## Advantages

* each source tile is loaded once and reused by many target particles,
* regular shared/local-memory access,
* good arithmetic intensity for dense cells,
* can naturally process (M\times N) particle tiles.

## Disadvantages

* many inactive lanes for cells containing fewer particles than a warp/wave,
* work imbalance between cells,
* barriers and local memory have fixed overhead,
* very dense cells can require multiple block iterations,
* one huge cell can become a serial load-balancing bottleneck,
* separate workgroups cannot synchronize while jointly updating one target cell.

The 2024 study suggests particle-centric kernels are the safer baseline at low occupancy, while block-per-cell and shared-memory methods become attractive at high occupancy. ([arXiv][4])

A practical hybrid is:

```text
if occupancy(cell) < threshold:
    process particles with ordinary particle-centric kernel
else:
    append cell to denseCellList
    process denseCellList with tiled cell kernel
```

This avoids imposing the expensive dense-cell strategy on the majority of sparse cells.

---

# 8. Verlet lists: often more important than faster grid construction

For molecular dynamics and any system with bounded particle displacement, use a list radius

[
r_{\rm list}=r_c+r_{\rm skin}.
]

Build neighbors using (r_{\rm list}), but evaluate forces only for

[
r<r_c.
]

The list remains valid while

[
2\max_i
\left|\mathbf x_i-\mathbf x_i^{\rm build}\right|
< r_{\rm skin}.
]

The factor of two accounts for two particles moving toward each other.

## Two-pass CSR construction

```cpp
kernel countNeighbors(){
    int i = get_global_id(0);

    int count = 0;

    for(each relevant cell){
        for(j in cell){
            if(j != i){
                float3 d = pos[j] - pos[i];

                if(dot(d,d) < listRadius2){
                    count++;
                }
            }
        }
    }

    neighborCount[i] = count;
}

exclusiveScan(neighborCount, neighborOffset);

kernel fillNeighbors(){
    int i = get_global_id(0);

    int out = neighborOffset[i];

    for(each relevant cell){
        for(j in cell){
            if(j != i &&
               distance2(pos[i],pos[j]) < listRadius2)
            {
                neighbors[out++] = j;
            }
        }
    }
}
```

Force evaluation then becomes:

```cpp
kernel forceFromNeighborList(){
    int i = get_global_id(0);

    float3 fi = 0;

    for(int k=neighborOffset[i];
            k<neighborOffset[i+1];
            k++)
    {
        int j = neighbors[k];
        float3 d = pos[j] - pos[i];

        if(dot(d,d) < cutoff2){
            fi += pairForce(d);
        }
    }

    force[i] = fi;
}
```

This is a **full neighbor list**: both `i→j` and `j→i` are stored. It uses more memory but is ideal for race-free GPU gather.

Current HOOMD-blue implementations use a buffer distance and postpone rebuilding until particle displacement consumes the buffer. GROMACS goes further with clustered pair lists and dynamic pruning. ([hoomd-blue.readthedocs.io][7])

---

## Dual-list approach

A more advanced arrangement has:

* outer radius (r_{\rm outer}),
* inner list radius (r_{\rm inner}),
* physical cutoff (r_c).

```text
Rarely:
    rebuild outer list using spatial grid

More often:
    prune outer list into inner list

Every force step:
    evaluate inner list with exact cutoff test
```

GROMACS reports outer-list lifetimes of order hundreds of MD steps and much shorter inner-list lifetimes in its dual-list scheme, although exact values depend on timestep, temperature, buffer settings and system. ([mpinat.mpg.de][8])

This can be more valuable than reducing a grid rebuild from, say, 0.5 ms to 0.3 ms, because the expensive global construction is executed much less frequently.

---

# 9. Cluster-pair lists: the production MD approach

Modern molecular-dynamics kernels often operate on small clusters rather than individual pairs.

Suppose particles are sorted spatially and grouped into clusters of (C) particles:

```cpp
struct Cluster {
    float3 aabbMin;
    float3 aabbMax;
    int firstParticle;
    int count;
};
```

Construct a list of potentially interacting cluster pairs:

```text
cluster A -> {cluster B0, cluster B1, ...}
```

Then process each candidate as a regular (C\times C) tile:

```cpp
for each cluster pair (A,B):
    load particles of A
    load particles of B

    for ia in A:
        for ib in B:
            if distance2 < cutoff2:
                accumulate interaction
```

A bitmask can mark potentially active lanes:

```text
mask[A,B] = which particle combinations may overlap
```

## Why clustering helps

* fewer neighbor-list entries,
* fewer cell/BVH lookups,
* regular SIMD/SIMT tiles,
* better coordinate reuse,
* fewer branch decisions,
* easier shared-memory staging.

## Cost

Cluster bounding boxes create false positives: if one particle in cluster (A) can interact with one particle in (B), the kernel may inspect all (C^2) combinations.

GROMACS uses clustered pair lists to map interactions efficiently onto SIMD and GPU hardware. ([GROMACS Documentation][9])

A new 2026 SFC/octree method pushes this further:

* particles ordered along a Hilbert-like space-filling curve,
* clusters of eight spatially consecutive particles,
* 64-particle superclusters,
* octree traversal,
* compressed sorted neighbor-cluster indices,
* per-cluster bitmasks.

The preprint reports only a few bytes of neighbor-list storage per particle for examples with roughly 200 neighbors, but performance is hardware-dependent: on one tested architecture the full list remained faster, while on another the compressed cluster representation won. It is promising but not yet a universal production standard. ([arXiv][10])

---

# 10. Uniform grid versus BVH

A uniform grid is geometrically inefficient because the spherical cutoff volume is embedded in a collection of cubic cells.

For (h=r_c), a particle inspects 27 cells with total volume

[
V_{\rm grid}=27r_c^3,
]

while the desired sphere has

[
V_{\rm sphere}=\frac{4\pi}{3}r_c^3.
]

The raw volume ratio is

[
\frac{V_{\rm grid}}{V_{\rm sphere}}
===================================

\frac{81}{4\pi}
\approx 6.45.
]

Boundary-cell distance tests and smaller cells improve this, but the fundamental mismatch remains.

A quantized BVH study for molecular dynamics reported roughly (2)–(4\times) speedups over its compared cell-list implementation across several benchmarks and argued that BVHs deserve serious consideration for GPU molecular dynamics. 

However, current HOOMD guidance is more nuanced:

* cell lists are usually preferable for monodisperse systems,
* tree/BVH lists can help with moderate particle-size asymmetry,
* actual performance should be benchmarked. ([hoomd-blue.readthedocs.io][11])

## Uniform grid tends to win when

* radii and cutoffs are similar,
* density is roughly uniform,
* cells have moderate occupancy,
* the simulation box is compact,
* construction must be extremely simple,
* the grid is already needed for another purpose.

## BVH tends to win when

* particle radii vary strongly,
* pair cutoffs differ strongly,
* density has large empty regions,
* particles form filaments, surfaces, sheets or isolated clusters,
* the bounding domain is huge relative to occupied volume,
* one wants swept AABBs for moving rigid objects.

## BVH disadvantages

* construction/refitting is more complicated,
* traversal is branchier,
* stack or rope traversal adds state,
* memory layout is less trivial,
* for homogeneous dense systems, grid traversal is often more regular.

---

# 11. Multilevel and stenciled grids

A single grid sized according to the largest particle radius is disastrous for polydisperse systems.

Suppose

[
r_{\max}=10r_{\min}.
]

Using (h\sim r_{\max}) puts potentially (10^3) times more small-particle volume into one cell than a grid designed for the small particles.

Alternatives are:

### Per-pair stencils

For each interaction type (a,b), precompute cell offsets that can overlap cutoff (r_{ab}).

```cpp
for(offset in stencil[type_i][type_j]){
    ...
}
```

This works when the number of types is small and the ratio of cutoffs is moderate.

### Hierarchical grids

Assign particle (i) to a level satisfying

[
h_\ell \sim r_i.
]

Then search:

* the particle’s own level,
* relevant finer/coarser levels.

This is more complicated but preserves bounded occupancy.

### Hybrid grid + BVH

* ordinary particles in uniform grid,
* very large particles in a separate BVH or coarse grid,
* query both structures.

Often a small number of exceptional large particles should not determine the resolution for millions of small ones.

The original GPU Gems broad-phase chapter already identified size disparity as a weakness of a single uniform subdivision and discussed hierarchical alternatives. ([NVIDIA Developer][1])

---

# 12. Sparse hashed grids

For a very large or effectively infinite domain, a dense cell-header array is wasteful.

The robust representation is:

```text
full 64-bit cell key
        ↓
hash table maps key to compact active-cell index
        ↓
activeCellStart[index], activeCellEnd[index]
        ↓
contiguous sorted particle range
```

Do not make the hash bucket itself the cell identity.

```cpp
struct HashEntry {
    ulong fullCellKey;
    uint  activeCellId;
};
```

Lookup:

```cpp
uint findActiveCell(ulong key){
    uint slot = hash(key) & tableMask;

    for(int probe=0; probe<MAX_PROBES; probe++){
        HashEntry e = table[slot];

        if(e.fullCellKey == key) return e.activeCellId;
        if(e.fullCellKey == EMPTY_KEY) return INVALID;

        slot = (slot + 1) & tableMask;
    }

    return INVALID;
}
```

This permits unrelated cells to collide in hash space without being interpreted as spatial neighbors.

For static or slowly changing cell keys, sorted `uniqueCellKey` plus binary search may be competitive and deterministic. With only 27 queries per particle, however, a well-sized open-addressed table is normally preferable to approximately (\log_2 N_{\rm cells}) comparisons per neighbor cell.

---

# 13. Particle-to-grid deposition is a related but different problem

The MDPI FLIP article you linked uses:

1. collision-free linear cell keys,
2. GPU sorting,
3. cell start/end ranges,
4. block cooperation,
5. grid-centric gather and reduction.

Instead of particles scattering atomically to grid nodes, one block owns a grid cell and gathers contributions from relevant particles, finishing with one owner write. The paper reports substantial particle-to-grid timing reductions at multi-million-particle scale. ([MDPI][12])

That is a good pattern for PIC/FLIP deposition:

```cpp
kernel gridGather(){
    int gridNode = get_group_id(0);

    float contribution = 0;

    for(particle in cells overlapping gridNode){
        contribution += shapeFunction(particle, gridNode)
                      * particleQuantity;
    }

    contribution = workgroupReduce(contribution);

    if(local_id == 0){
        grid[gridNode] = contribution;
    }
}
```

But it does not directly establish that one-block-per-cell is best for particle-particle forces. In P2G, many particles write a relatively small grid and atomics are a central bottleneck. In particle-particle gather, every thread naturally owns one particle output.

---

# 14. Hard collisions are not the same as conservative pair forces

For a soft pair potential,

[
\mathbf F_i = \sum_j \mathbf F_{ij},
]

a gather kernel is sufficient.

For hard contacts, each pair defines an impulse or constraint involving both particles:

[
\Delta\mathbf v_i = +M_i^{-1}\mathbf J_{ij},
\qquad
\Delta\mathbf v_j = -M_j^{-1}\mathbf J_{ij}.
]

There are four main GPU strategies.

## 14.1 Jacobi gather

First generate contacts:

```cpp
contacts[k] = {i, j, normal, penetration, ...};
```

Evaluate each contact independently using the old velocities:

```cpp
impulse[k] = solveContact(contacts[k], stateOld);
```

Then gather by particle:

```cpp
deltaV[i] = sum of signed impulses incident on i;
stateNew[i] = stateOld[i] + deltaV[i];
```

This is parallel and race-free, but converges more slowly than Gauss-Seidel.

The incidence list can be built as CSR:

```text
particle -> incident contacts
```

or by sorting signed records:

```text
(i, +Jij)
(j, -Jij)
```

by particle ID and segmented-reducing them.

## 14.2 Atomic scatter

```cpp
atomic_add(velocity[i], +deltaVi);
atomic_add(velocity[j], -deltaVj);
```

Simple, but nondeterministic and potentially highly contended.

## 14.3 Contact or cell coloring

Partition contacts so no particle appears twice in one color.

```text
for color:
    solve all contacts of that color in parallel
    synchronize globally
```

The old GPU Gems broad-phase method used spatial cell phases to prevent simultaneous conflicting updates. This approach preserves more Gauss-Seidel character but requires multiple kernel launches or device-wide synchronization. ([NVIDIA Developer][1])

## 14.4 Pair buffer plus reduction

Compute each pair once:

```cpp
pairContribution[2*k  ] = {i, +J};
pairContribution[2*k+1] = {j, -J};
```

Then sort or bin contributions by particle and reduce.

This has the cleanest conservation properties but creates substantial temporary memory traffic.

For your gather-oriented, no-atomic design preference, I would use Jacobi contact impulses with per-particle incident-contact lists, possibly combined with local sequential iterations inside independent cell clusters.

---

# 15. Continuous collision detection

A grid broad phase does not prevent tunneling.

If particles can move farther than their diameter or cutoff margin during one timestep, index either:

### Swept AABB

[
\mathrm{AABB}_i =
\mathrm{bounds}\left(
\mathbf x_i(t),\mathbf x_i(t+\Delta t)
\right)
\oplus r_i.
]

Insert the swept box into every overlapped cell.

### Conservative search radius

[
r_{\rm broad}
=============

r_i+r_j+
\left|\mathbf v_i-\mathbf v_j\right|\Delta t
+\text{acceleration margin}.
]

### Substepping

Choose (\Delta t) so that displacement is a small fraction of the interaction scale.

For hard, fast particles, this issue is often more important than the difference between sorting and atomic cell insertion.

---

# 16. Emerging RTX/RT-core methods

Recent work has mapped fixed-radius neighbor search onto hardware ray-tracing BVHs:

* represent particles by small geometric primitives or AABBs,
* cast a ray/query encoding the neighborhood,
* let RT hardware traverse the BVH,
* optionally calculate interactions inside the hit shader/payload.

A 2026 preprint presents several RT-core fixed-radius-neighbor variants, including versions that avoid explicit neighbor-list storage and adaptive BVH refit/rebuild strategies. Its results are not universally superior: RT methods performed well for certain small-radius or irregular distributions, while conventional GPU cell lists remained better for larger common radii and dense clustered cases. It is also NVIDIA/OptiX-specific. ([arXiv][13])

This is an interesting frontier, especially when:

* a BVH already exists for rendering or collision,
* the distribution is highly irregular,
* memory for explicit neighbor lists is problematic,
* hardware portability is not required.

I would not yet use it as the default foundation of a general OpenCL particle solver.

---

# 17. Comparison matrix

| Method                         |               Construction | Query locality            |              Memory | Best use                                    | Main weakness                |
| ------------------------------ | -------------------------: | ------------------------- | ------------------: | ------------------------------------------- | ---------------------------- |
| Fixed-capacity dense bins      |            (O(N)), atomics | Excellent                 |  `cells × capacity` | Strict occupancy bound, real-time graphics  | Overflow                     |
| Atomic linked cells            |                     (O(N)) | Poor                      |            (O(N+C)) | Very frequent rebuild, few queries          | Pointer chasing              |
| Count–scan–scatter             |                   (O(N+C)) | Excellent                 |            (O(N+C)) | Dense bounded grid, general GPU baseline    | Integer atomics, clearing    |
| Radix-sort cell list           |        (O(N)) radix passes | Excellent                 | (O(N)) sort buffers | Atomic-free, deterministic, spatial reorder | Sorting cost                 |
| Particle-centric grid gather   |             No extra build | Good                      |             Minimal | Low/moderate occupancy                      | Pair evaluated twice         |
| Block-per-cell tiled           |             No extra build | Excellent for dense cells |  Local-memory tiles | High occupancy, costly pair function        | Idle lanes, imbalance        |
| Verlet CSR                     | Expensive occasional build | Excellent                 |   (O(NN_{\rm nbr})) | Slowly moving MD particles                  | List memory                  |
| Cluster-pair list              |               More complex | Very regular              |             Compact | Production MD/SIMT kernels                  | False-positive (C^2) pairs   |
| Multilevel grid                |           Moderate/complex | Good                      |            Moderate | Strong size disparity                       | Cross-level logic            |
| LBVH / quantized BVH           |        (O(N))–(O(N\log N)) | Irregular                 |          Tree nodes | Sparse, clustered, polydisperse             | Branchy traversal            |
| SFC/octree compressed clusters |                    Complex | Good cluster locality     |            Very low | Large systems, memory-bound lists           | Emerging, hardware-dependent |
| RT-core BVH                    |      Driver/hardware build | Hardware accelerated      |                 BVH | Irregular NVIDIA workloads                  | Vendor-specific, emerging    |

---

# 18. Review of the sources you supplied

## GPU Gems Chapter 32

Still conceptually important:

* map objects to cells,
* sort by cell ID,
* identify contiguous cell runs,
* process spatial subdivisions in conflict-free phases.

Less appropriate today for homogeneous point particles:

* duplicating objects into “home” and “phantom” cells,
* explicit home/phantom bit masks,
* multiple cell-color passes,
* one thread per collision cell,
* custom assumptions tied to early CUDA hardware.

For particles with one center and cutoff sphere, store each particle once in its center cell and have the query inspect neighboring cells. Object duplication is mainly needed for broad-phase objects whose AABBs span multiple cells. ([NVIDIA Developer][1])

## GPU Gems Chapter 29

Historically interesting, but the fixed small number of particles per voxel, RGBA texture storage and multi-pass rendering construction should not be copied into a modern compute implementation. ([NVIDIA Developer][6])

## Reddit discussion

Reasonable introductory intuition—hash position to cell, keep data GPU-resident—but not a sufficiently precise algorithmic source. It does not seriously address contiguous storage, overflow, sorting, force ownership, load balancing or neighbor-list reuse. ([Reddit][14])

## 2024 GPU interaction paper

The most directly relevant of your supplied research sources. Its strongest message is that sophisticated block/cell/shared-memory algorithms are not automatically faster: one-thread-per-particle remains very strong when cells contain relatively few particles. ([arXiv][4])

## FLIP/PIC block-cooperation paper

Good demonstration of replacing particle-to-grid scatter atomics by grid-centric gathering and block reduction. Very relevant for PIC deposition, but only indirectly relevant to particle-particle collision. ([MDPI][12])

## Unity dynamic hash grid

Useful as readable compute-shader code. Not a large-scale state-of-the-art reference because of global bitonic sorting and the need to manage hash collisions carefully. ([GitHub][3])

## Atomic grid-construction paper

Important evidence that atomics can make construction very cheap. I would retain the atomic build idea but replace linked traversal by compact ranges when the structure is queried repeatedly. ([SciSpace][5])

---

# 19. What I would implement for your OpenCL/GTX 3090 case

## Version 1: simple, reliable baseline

```text
1. Set h = rc.
2. Compute a collision-free dense cell ID.
3. GPU radix-sort (cellID, particleID).
4. Reorder position/type arrays into cell order.
5. Construct cellStart/cellEnd.
6. One work item per particle.
7. Scan 27 cells.
8. Accumulate only force[i].
9. Evaluate every pair twice.
10. Use SoA and single precision.
```

Do **not** initially use:

* local memory,
* one workgroup per cell,
* half neighbor lists,
* force atomics,
* fixed cell capacities,
* linked lists.

This gives a strong, understandable reference against which every optimization can be measured.

## Version 2: amortize construction

Add:

```text
rlist = rc + skin
```

and a full CSR Verlet list. Rebuild only when maximum displacement exceeds `skin/2`.

## Version 3: spatially clustered kernel

Group sorted particles into clusters of 8 or perhaps 16. Build cluster AABBs and candidate cluster pairs. Use an (8\times8) or (16\times16) interaction tile.

For OpenCL 1.2 and NVIDIA, I would start with:

```text
cluster size = 8
workgroup size = 64 or 128
```

rather than hard-coding a warp-centric algorithm everywhere.

## Version 4: occupancy-adaptive dispatch

Measure cell occupancy histogram:

```text
mean occupancy
95th percentile
maximum occupancy
fraction of particles in cells with occupancy > 32
fraction > 64
```

Dispatch:

```text
ordinary cells -> particle-centric gather
dense cells    -> block-tiled kernel
```

This is more robust than selecting one kernel globally.

## Version 5: leave the uniform grid when necessary

Switch to BVH or a multilevel grid when:

```text
max cutoff / min cutoff > approximately 2–4
```

or when active-cell occupancy becomes extremely heterogeneous. That numerical range is a heuristic, not a universal threshold; the proper crossover must be benchmarked for the actual distribution and force cost.

---

# Final assessment

For a homogeneous particle system, I would describe the current state of the art as:

[
\boxed{
\text{spatial sort}
+
\text{compact cell ranges}
+
\text{particle gather}
+
\text{buffered/clustered neighbor lists}
}
]

not simply “spatial hashing.”

The most important optimization decisions, in approximate order, are:

1. avoid global scattered force writes;
2. reuse neighbor information across timesteps;
3. keep spatial neighbors contiguous in memory;
4. choose particle-centric versus cell-tiled execution from occupancy;
5. use clusters to regularize SIMT work;
6. abandon a one-level uniform grid when sizes or densities become strongly heterogeneous.

For your style of GPU programming, the best first production design is likely an **atomic-free radix-sorted compact grid with particle-centric full-list gather**, followed by a **Verlet/cluster list**. Shared-memory cell kernels should be a measured high-occupancy specialization, not the starting assumption.

[1]: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda "Chapter 32. Broad-Phase Collision Detection with CUDA | NVIDIA Developer"
[2]: https://docs.lammps.org/package.html?utm_source=chatgpt.com "package command"
[3]: https://github.com/iweinbau/UnityGPUDynamicHashGrid "GitHub - iweinbau/UnityGPUDynamicHashGrid: Implementation of a dynamic hash grid on the GPU · GitHub"
[4]: https://arxiv.org/html/2406.16091v1 "Efficient GPU Implementation of Particle Interactions with Cutoff Radius and Few Particles per Cell."
[5]: https://scispace.com/pdf/fast-uniform-grid-construction-on-gpgpus-using-atomic-1vd4i0m6zf.pdf?utm_source=chatgpt.com "fast-uniform-grid-construction-on-gpgpus-using-atomic- ..."
[6]: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus "Chapter 29. Real-Time Rigid Body Simulation on GPUs | NVIDIA Developer"
[7]: https://hoomd-blue.readthedocs.io/en/v4.0.1/module-md-nlist.html?utm_source=chatgpt.com "md.nlist - HOOMD-blue 4.0.1 documentation"
[8]: https://www.mpinat.mpg.de/634401/Kutzner_2019_JCC.pdf "More Bang for Your Buck: Improved use of GPU Nodes for GROMACS 2018"
[9]: https://manual.gromacs.org/current/reference-manual/algorithms/molecular-dynamics.html?utm_source=chatgpt.com "Molecular Dynamics - GROMACS 2026.3 documentation"
[10]: https://arxiv.org/html/2602.19873v1 "GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data Layout Swiss Platform for Advanced Scientific Computing (PASC)."
[11]: https://hoomd-blue.readthedocs.io/en/v5.1.1/hoomd/md/nlist/tree.html?utm_source=chatgpt.com "Tree - HOOMD-blue 5.1.1 documentation"
[12]: https://www.mdpi.com/2673-3951/7/1/27 "GPU-Accelerated FLIP Fluid Simulation Based on Spatial Hashing Index and Thread Block-Level Cooperation | MDPI"
[13]: https://arxiv.org/html/2601.15633v1 "Advancing RT Core-Accelerated Fixed-Radius Nearest Neighbor Search"
[14]: https://www.reddit.com/r/GraphicsProgramming/comments/a901xv/hash_table_particle_collision_w_compute_shaders/ "Hash Table Particle collision w/ Compute Shaders : r/GraphicsProgramming"



