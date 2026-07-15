https://chatgpt.com/share/6a56b248-a0c0-83eb-9070-e6aa8841eaf3

# USER

I'm thiniking about rebalancking strategies for AABB and spherical spatial partitioning. Note I do not want BVH like try, only one level. My AABB groups are associated with GPU workgroups, typically 16-32 threads (tiles). I always find which AABB can overlap and then solve colisions in the interasections. AABB alrready contains marin fo Rc which is the interaction radius of each particle.

Now I can think about several strategies.

1) Simplest and most efficient seems to me that during the broad collision phase each particle check all boxes with whom it overlaps, and insert itself in the nearest if this decrease cost and if there is free place in target WG (remove itself from current WG and insert into new). Problem of this algorithm is that the groups must have free slots and their size would varry (i.e. it will not perfecly utilize all threads). Advantage is that it can be part of collision kernel without extra cost and kernel execution overheads.

2) Another algorithm could be swapping, if two particles interact with each other (e.g. collision) there an be a check if swapping them would decrease cost (e.g. decrease sum of distances from AABB COG). But this has problems:
i) first problem is synchronization - we must make sure that the same particle is not swaped to different workgroup. Simple strategy can be terporal gating that in each pass of kernel only one workgroup does swapping, or there can be more sophisticated stragegies by graph coloring.
ii) second problem branch divergence - most of workgroups will be not swapping. But perhaps thios is not problem as branch divergence only within one workgroup is problem, if whole workgroup does swaping nor not it is OK?

3) we can run re-insertion every few frames. The disadvantag is that this may be complicated process, it is only every few frames, or it require run another kernel and pay the overhead. Advantage is that it can have some global information which is not available to local swapping stragegies. But it is perhps more complicated.

Any opther ideas?
Can you analyze which is best?
What strategies are used for similar problem in practice.

Note that we really try to adjust our datastructuire to hardware architecture, and our workroupsize is choosen by that. If we do not fill one wrap or if we are left with some surious partilces (i.e. not integer multiples of wrap size) we have idle threads, or more complicated scheduling (distribution of particles among threads), and we want to keep it simple (typicaly symple algorothm use less regiuster and is much faster on gpu). 


---

# GPT 5.6 sol

## Main conclusion

For your constraints, I would use:

> **Fixed-size groups of exactly (W=16) or (32) particles, periodically rebuilt by spatial ordering, with an optional cheap local (2W\rightarrow W+W) repartition between rebuilds.**

I would **not modify group membership directly inside the collision kernel**. You can calculate migration or swap proposals there almost for free, but committing them safely requires either another kernel, double buffering, or a global synchronization mechanism.

The established method closest to your idea is the fixed-cardinality clustering used by GROMACS: particles are spatially ordered and packed into hardware-sized clusters, with dummy padding only where necessary. The cluster size is selected to match SIMD/SIMT execution, and compact cluster AABBs are used to build cluster-pair lists. 

---

# 1. First define the quantity you actually want to minimize

Distance to the group center of gravity is only a weak proxy. For an AABB broad phase, the actual cost is determined by **box overlap**, not by variance around the center.

For group (g), including the per-particle interaction margin (R_i),

[
\mathbf l_g=\min_{i\in g}(\mathbf x_i-R_i),\qquad
\mathbf u_g=\max_{i\in g}(\mathbf x_i+R_i).
]

A good approximate objective is

[
C_{\rm pair}
============

\sum_{g<h}
I(A_g\cap A_h\neq\varnothing), n_g n_h
+
\sum_g \frac{n_g(n_g-1)}{2},
]

where (n_g) is the number of particles in group (g).

For fixed (n_g=W), this simplifies essentially to minimizing the **number of overlapping group pairs**:

[
C_{\rm pair}\approx W^2 N_{\rm overlap}.
]

A slightly smoother proxy, useful for deciding local moves, is

[
C_{\rm geom}
============

\sum_g
\left[
\alpha V_g+
\beta S_g+
\gamma D_g
\right],
]

where

[
V_g=d_xd_yd_z,
]

[
S_g=2(d_xd_y+d_yd_z+d_zd_x),
]

[
D_g=d_x^2+d_y^2+d_z^2.
]

I would emphasize surface area or overlap count over volume. A very thin but long box may have small volume while intersecting many neighboring boxes.

The complete dynamic objective should include migration:

[
C =
C_{\rm pair}
+
\lambda C_{\rm geom}
+
\mu N_{\rm moved}.
]

The last term provides hysteresis and prevents particles from oscillating between groups.

## Why center-of-gravity distance is insufficient

Consider a group with one particle defining the maximum (x)-face. Moving that particle may reduce the box length by 50%, even though it is not the most distant particle from the centroid in Euclidean distance.

Conversely, exchanging two particles can reduce centroid variance while leaving all six AABB faces unchanged.

For AABBs, the important particles are the ones defining

[
x_{\min},x_{\max},y_{\min},y_{\max},z_{\min},z_{\max}.
]

Maintaining the first and second extrema in each direction makes the effect of removing one particle cheap to estimate.

---

# 2. Strategy 1: opportunistic insertion into overlapping groups

The appealing part is correct: the collision pass already identifies geometrically relevant destination groups, so you obtain candidate destinations almost for free.

But direct insertion has several deeper problems.

## Occupancy

Suppose the physical group capacity is 32, but many groups contain 25–30 particles. The force kernel still launches 32 lanes. Unless you combine groups dynamically, the unused lanes directly reduce utilization.

The average utilization is

[
\eta_{\rm lane}
===============

\frac{\sum_g n_g}{W N_g}.
]

Even 28 particles per 32-lane group gives only (87.5%) lane utilization, before considering irregular work between groups.

## Shrinking the source box

Expanding a destination AABB is cheap:

[
l'=\min(l,x_i-R_i),\qquad
u'=\max(u,x_i+R_i).
]

Shrinking the source AABB is not cheap if the removed particle owned one of its faces. You must either rescan the group or maintain second extrema.

A rescan of 16–32 particles is not terrible, but it means insertion is not truly free.

## Concurrent modification

The serious issue is correctness.

If collision workgroups are reading group membership and AABBs while other workgroups are changing them:

* a particle can be processed twice;
* a particle can temporarily disappear;
* a box can shrink before another workgroup finishes testing it;
* a destination box can expand after its overlap tests were already completed;
* two workgroups can insert into the same slot.

Therefore, the collision kernel can safely produce something like

```text
proposal[particle] = destination_group
proposal_gain[particle] = estimated_cost_reduction
```

but it should not normally commit the operation to the currently active grouping.

A global barrier between collision testing and mutation does not exist in an ordinary CUDA or OpenCL kernel. A persistent cooperative kernel could provide one on restricted hardware, but that complicates the design considerably.

## Verdict

This is useful only in one of these forms:

1. **Deferred proposals**, committed after the collision kernel.
2. A small **spill/overflow structure** for temporarily misplaced particles.
3. A deliberately overallocated structure with perhaps (W+2) storage slots but only (W) active computational lanes.

I would not let variable occupancy become the normal steady-state representation.

---

# 3. Strategy 2: pairwise swapping

Swapping is substantially better than insertion because it preserves exact occupancy.

## Branch divergence

Your understanding is essentially correct, with one refinement:

> Divergence matters within a hardware subgroup—on NVIDIA, within a 32-thread warp—not between independent warps or workgroups.

If every lane of a warp takes the same “perform swap” or “do not perform swap” branch, there is no branch divergence. Different warps may follow completely different paths independently. ([NVIDIA Docs][1])

So branch divergence is not the main problem. The larger issues are:

* conflicting swaps;
* global synchronization;
* weak improvement from exchanging only one pair;
* inability to escape local minima.

## Avoid serial activation of one workgroup

Allowing only one group to swap per pass removes almost all parallelism. Graph coloring the dynamic box-overlap graph is possible, but likely more expensive and complicated than the repartition itself.

A much cleaner non-atomic schedule is:

1. Each group selects its best neighboring partner.
2. A pair is accepted only if the proposals are mutual:
   [
   p(g)=h,\qquad p(h)=g.
   ]
3. Break ties deterministically using group IDs.
4. Each accepted group belongs to at most one pair.
5. Repartition all accepted pairs in parallel.

This requires a proposal stage and a commit stage, but does not need atomics for ownership.

## Single-particle swaps are too restrictive

Suppose group (A) contains four particles spatially belonging to (B), and (B) contains four belonging to (A). A sequence of single swaps might eventually fix it, but each intermediate swap may appear unprofitable.

A better operation is to repartition both groups simultaneously.

---

# 4. Better local method: (2W\rightarrow W+W) retiling

This is probably the most useful additional idea for your architecture.

Take two neighboring or overlapping groups (A) and (B), each with (W) particles:

[
P=A\cup B,\qquad |P|=2W.
]

Load all (2W) particle descriptors into local memory and repartition them into two new groups of exactly (W) particles.

## Simplest median split

Compute the bounding box of all (2W) particles and choose its longest axis:

[
a=\arg\max_{k\in{x,y,z}}
(u_k-l_k).
]

Sort the (2W) particles by (x_a), and split at the median:

[
A'={p_0,\ldots,p_{W-1}},
\qquad
B'={p_W,\ldots,p_{2W-1}}.
]

For (2W=32) or (64), an in-workgroup bitonic sort is small and regular.

You then accept the repartition only when

[
C(A',B')+\mu N_{\rm moved}
<
C(A,B).
]

This operation:

* preserves perfect occupancy;
* can move many particles at once;
* has no inter-pair synchronization after mutual pairing;
* uses regular local-memory operations;
* is much less likely to get trapped than a single swap;
* naturally fits a 32- or 64-thread workgroup.

## Better split directions

The longest AABB axis is probably the best simplicity/performance compromise. Alternatives are:

* direction between group centers;
* approximate principal axis;
* line through the farthest pair;
* local Morton key;
* testing (x), (y), and (z) splits and selecting the cheapest.

Testing all three Cartesian median splits is still inexpensive for (W\le32). You do not necessarily need three complete sorts: sorting networks for the three coordinates may be acceptable if rebalancing is infrequent, or you can use approximate histograms and selection.

## Evaluate external interactions, not only the two boxes

A repartition may improve (A)-(B) compactness but worsen overlap with other groups.

For each candidate pair, ideally compare

[
C_{\rm local}
=============

\sum_{k\in \mathcal N(A)\cup\mathcal N(B)}
\left[
I(A\cap k)Wn_k+
I(B\cap k)Wn_k
\right].
]

Only existing neighboring boxes and perhaps boxes intersecting the candidate new bounds need to be checked.

This information is already close to what your broad phase generates.

---

# 5. Strategy 3: periodic global reinsertion or rebuilding

I think this should be your primary mechanism.

The additional kernel launches are a real cost, but the relevant comparison is not

[
\text{rebuild cost versus zero};
]

it is

[
\text{amortized rebuild cost}
\quad\text{versus}\quad
\text{all extra candidate pairs generated by degraded groups}.
]

If the partition is reused for (M) frames, the effective rebuild cost per frame is

[
T_{\rm rebuild}/M.
]

A regular, branch-light collision kernel using tightly packed groups can easily justify an occasional reordering pass.

Established GPU particle codes use this general pattern: construct or rebuild spatial neighbor structures periodically, rather than continuously mutating them during force evaluation. HOOMD-blue uses fixed-width cell lists and exposes configurable neighbor-list rebuild checks; it also periodically sorts particle memory along a space-filling curve to improve spatial and cache locality. ([hoomd-blue.readthedocs.io][2])

GROMACS similarly uses fixed-lifetime outer pair lists and cheaper pruning between full rebuilds. Its GPU-oriented setup deliberately trades some extra candidate interactions for highly regular fixed-cluster computation. 

## Best rebuilding methods for your case

### A. Morton/Hilbert ordering followed by fixed chunks

For every particle, generate a spatial key:

[
k_i=\operatorname{Morton}(x_i,y_i,z_i)
]

or a Hilbert key, sort particles by (k_i), and define

[
G_j={p_{jW},\ldots,p_{jW+W-1}}.
]

This gives exactly full groups except the final group.

It is not a BVH. It is simply a one-dimensional spatial ordering followed by fixed-cardinality chunking.

HOOMD-blue uses space-filling-curve sorting specifically to place spatially close particles close in memory. ([hoomd-blue.readthedocs.io][3])

Morton order is much easier to generate than Hilbert order. Hilbert normally has somewhat better locality, but Morton keys are cheap bit interleavings and are convenient for radix sorting.

### B. GROMACS-style column packing

The GROMACS construction is particularly relevant:

1. bin particles on a regular (xy) grid;
2. sort particles in each (xy) column by (z);
3. take consecutive fixed-size groups;
4. pad the end of individual columns when required.

This produces fixed-cardinality spatial groups and avoids arbitrary cell occupancy. 

You could generalize it by choosing the column direction according to the system anisotropy.

For surfaces or slabs, (xy)-binning followed by (z)-sorting is probably excellent. For fully 3D systems, Morton chunking is more symmetric.

### C. Coarse counting sort rather than full radix sort

Generate a coarse cell ID for each particle, build cell counts, prefix-sum them, and scatter into cell-contiguous storage. Then subdivide each cell stream into (W)-particle groups.

The classical GPU spatial-subdivision implementation similarly generates particle–cell records, sorts or groups them by cell ID, and processes contiguous runs. ([NVIDIA Developer][4])

This may be cheaper than a high-resolution global Morton sort, but creates padding or awkward groups at cell boundaries. You can reduce that by allowing adjacent cells to share a packing stream.

---

# 6. Rebuild on degradation, not merely every fixed number of frames

A fixed rebuild interval is simple, but a quality-triggered policy may be better for your simulations.

Track inexpensive statistics:

[
Q_1=N_{\rm overlapping\ group\ pairs},
]

[
Q_2=N_{\rm candidate\ particle\ pairs},
]

[
Q_3=\sum_g S_g,
]

[
Q_4=\max_g \frac{V_g}{V_{g,0}},
]

or, best of all,

[
Q_5=
\frac{\text{tested particle pairs}}
{\text{accepted physical interactions}}.
]

Rebuild when, for example,

[
Q(t) > (1+\epsilon)Q(t_{\rm last\ rebuild}),
]

with perhaps (\epsilon=0.2) or (0.5), determined empirically.

Also impose a minimum interval to avoid rebuilding repeatedly during transient motion.

---

# 7. Handling different particle radii or cutoffs

If (R_i) varies strongly, spatial proximity alone is insufficient.

One large-(R_i) particle can expand the AABB of an entire 32-particle group. Therefore, consider sorting by something like

[
(\text{spatial coarse key},\ \text{radius class},\ \text{fine spatial key}).
]

Or maintain several group classes:

[
R_i\in [R_0,2R_0),\quad
[2R_0,4R_0),\quad \ldots
]

and build groups separately within each class.

HOOMD-blue explicitly notes that ordinary fixed-width cell lists become inefficient with large cutoff asymmetry; its documentation recommends the cell-list approach mainly when the largest-to-smallest cutoff ratio is below roughly (2:1). ([hoomd-blue.readthedocs.io][2])

For your group AABBs the precise threshold will differ, but the same mechanism applies: mixing one long-range particle with many short-range particles inflates the whole group broad-phase volume.

---

# 8. A practical hybrid design

I would implement the following.

## Normal frames

Each group contains exactly (W) valid particles.

The collision kernel:

1. reads immutable group membership;
2. performs box and particle collision tests;
3. accumulates a quality metric for each group;
4. optionally records its most promising neighboring group for retiling.

No membership changes occur.

## Cheap local maintenance

Every few frames, or when local quality deteriorates:

1. each group proposes one partner;
2. retain only mutual pairs;
3. one workgroup loads the (2W) particles;
4. perform a longest-axis median split;
5. compare old and new AABB/overlap cost;
6. write accepted partitions into the next membership buffer.

This preserves exact workgroup occupancy.

## Global rebuild

When the global candidate-pair count rises sufficiently:

1. calculate Morton or coarse-grid keys;
2. radix/counting sort;
3. pack consecutive blocks of (W);
4. pad only the final block;
5. reorder the particle SoA physically, if memory locality is important;
6. recompute group AABBs.

GROMACS and HOOMD demonstrate the practical value of precisely this broad philosophy: periodic spatial ordering, hardware-oriented fixed clustering, and reuse/pruning rather than continuously maintaining an optimal geometric partition. 

---

# Ranking your proposed methods

| Method                      |                Occupancy |        Synchronization |        Partition quality | Recommendation                        |
| --------------------------- | -----------------------: | ---------------------: | -----------------------: | ------------------------------------- |
| Insert into free slot       |                 Variable |              Difficult |                 Moderate | Only as deferred overflow mechanism   |
| Single-particle swap        |                    Exact |     Moderate/difficult |   Weak local improvement | Useful, but inferior to (2W) retiling |
| Periodic global reinsertion |                    Exact | Simple between kernels |                     High | Best primary method                   |
| Pairwise (2W) retiling      |                    Exact |  Simple after matching | Strong local improvement | Best maintenance method               |
| Morton/Hilbert rebuild      |                    Exact |           Sorting pass |          High and robust | Best general rebuild                  |
| Fixed grid cells            | Variable unless repacked |                 Simple | Good for uniform density | Useful as intermediate binning        |

## Final recommendation

The cleanest architecture is:

[
\boxed{
\text{periodic Morton/grid rebuild}
+
\text{fixed }W\text{-particle groups}
+
\text{occasional }2W\rightarrow W+W\text{ local retiling}
}
]

Your idea of exploiting overlap information during the collision pass is still valuable—but use it to **measure quality and propose rebalancing**, not to modify the active partition immediately.

This keeps the hot collision kernel simple, keeps every warp full, avoids atomics, and confines irregular reorganization to infrequent kernels where a modest amount of additional work is acceptable.

[1]: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html?utm_source=chatgpt.com "1.2. Programming Model — CUDA Programming Guide"
[2]: https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/nlist/cell.html "Cell - HOOMD-blue 7.1.0 documentation"
[3]: https://hoomd-blue.readthedocs.io/en/v6.1.1/hoomd/tune/particlesorter.html?utm_source=chatgpt.com "ParticleSorter - HOOMD-blue 6.1.1 documentation"
[4]: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda "Chapter 32. Broad-Phase Collision Detection with CUDA | NVIDIA Developer"


---

# USER


Some of the operation can be done easily in the broad phase where we update size of AABB box by iterating over all the particles. In that phase we can also identify few (e.g. 4) worst particles (which are most far from the center of the box, therefore are potential candidate for replacement), then we can cross-check with the nearby overlaping boxes if it does not have partciles worth swaping. We can basically always go in the list of the worst particle by priority and swap in greedy manner, what you think? 

---

# GPT 5.6 sol hi

Yes—this is a sensible **cheap local maintenance algorithm**, especially because most of the bookkeeping can be fused into the AABB-refit pass. I would use it, but with three modifications:

1. Select candidates by **how much their removal can shrink the box**, not primarily by distance from its center.
2. Evaluate the **combined swap gain** before committing.
3. Let each group interact with only one selected neighboring group per rebalancing pass, avoiding conflicts.

## 1. “Farthest from center” is not quite the right criterion for AABBs

For a spherical group, the farthest particles are naturally the particles controlling its radius. For an AABB, however, a distant particle matters only when it controls one of the six faces:

[
x_{\min},x_{\max},y_{\min},y_{\max},z_{\min},z_{\max}.
]

A particle can be far from the center yet not define any face, because another particle is slightly farther in each coordinate. Removing it then does not shrink the AABB at all.

Conversely, a particle may be only moderately far in Euclidean distance but be the unique owner of (x_{\max}); removing it may shrink the box substantially.

During the existing (O(W)) AABB-refit pass, maintain for every coordinate:

[
x_{\min}^{(1)},\quad x_{\min}^{(2)},\quad
x_{\max}^{(1)},\quad x_{\max}^{(2)},
]

including the IDs of the particles owning the first extrema. Do the same for (y,z).

Then, for any particle (i), you can calculate the box after removing it in (O(1)):

[
x_{\min}^{(-i)}
===============

\begin{cases}
x_{\min}^{(2)}, & i=\operatorname{owner}(x_{\min}^{(1)}),\
x_{\min}^{(1)}, & \text{otherwise}.
\end{cases}
]

Likewise for the other five faces.

This gives an exact removal score such as

[
G_i^{\rm remove}
================

C(A)-C(A\setminus i),
]

where (C) can be volume, surface area, diagonal length, or preferably an estimate of broad-phase work.

For example,

[
C(A)=
\alpha V(A)+\beta S(A)+\gamma D^2(A).
]

I would use a small candidate pool containing:

* the six face owners;
* perhaps the top few particles by normalized distance from the box center;
* with duplicates removed.

This will usually produce only about 6–10 candidates and is more reliable than taking only the four largest Euclidean distances.

## 2. Candidate selection should be directional

For a particular neighboring group (B), the most plausible particle to move from (A) to (B) is often not the globally worst particle in (A). It is the particle lying on the side of (A) facing (B).

Let

[
\mathbf d_{AB}
==============

\frac{\mathbf c_B-\mathbf c_A}
{|\mathbf c_B-\mathbf c_A|}.
]

Then particles in (A) with large

[
s_i=\mathbf x_i\cdot \mathbf d_{AB}
]

are the ones extending toward (B), while particles in (B) with small projection along the same direction extend toward (A).

So for the group pair (A,B), a particularly effective candidate pool is:

[
C_A =
\text{top }K\text{ particles by }
\mathbf x_i\cdot \mathbf d_{AB},
]

[
C_B =
\text{bottom }K\text{ particles by }
\mathbf x_j\cdot \mathbf d_{AB}.
]

You can combine this with the AABB removal score:

[
q_i
===

\lambda G_i^{\rm remove}
+
(1-\lambda),
\mathbf x_i\cdot\mathbf d_{AB}.
]

This targets particles that are simultaneously:

* geometrically on the wrong side;
* responsible for making their current box large.

For AABBs, you can avoid computing arbitrary projections by keeping candidates from the six Cartesian directions. Given (\mathbf d_{AB}), select the appropriate (+x/-x,+y/-y,+z/-z) candidate lists.

## 3. Evaluate the complete swap, not two independent moves

For candidate particles

[
i\in A,\qquad j\in B,
]

construct

[
A'=(A\setminus i)\cup j,
\qquad
B'=(B\setminus j)\cup i.
]

Then evaluate

[
G_{ij}
======

C(A)+C(B)-C(A')-C(B').
]

Using first and second extrema, the candidate AABBs (A') and (B') can be calculated cheaply:

1. remove (i) from (A);
2. expand the resulting AABB by particle (j);
3. do the symmetric operation for (B).

For (K=4), there are only

[
K^2=16
]

candidate swaps. That is extremely cheap.

A better cost than box volume alone is the expected broad-phase work:

[
C(A)
\approx
\sum_{N\in\mathcal N(A)}
I(A\cap N),W^2.
]

Thus, after constructing candidate (A'), check it against the nearby boxes already present in the broad-phase neighbor list. Then the gain directly measures the number of cluster-pair tests removed or added.

A two-level evaluation is probably best:

1. evaluate all (K^2) swaps using a cheap AABB size metric;
2. evaluate the best one or two using actual neighboring-box overlap counts.

## 4. Greedy repeated swaps within one box pair

Once groups (A) and (B) have been assigned to each other, your greedy procedure is good:

```text
repeat up to NSWAP times:
    construct candidate lists CA and CB
    evaluate all candidate pairs
    take the pair with largest positive gain

    if gain < minimum_gain:
        stop

    swap particles
    recompute local AABBs and candidate lists
```

With (W=16) or (32), recomputing both AABBs after each accepted swap is inexpensive and simplifies correctness.

I would allow perhaps two to four swaps per paired group. More than that begins to approach a full (2W)-particle repartition anyway.

### Why recompute after every accepted swap

Suppose particles (i_1,i_2) both contribute to (x_{\max}). Removing (i_1) may produce no shrinkage because (i_2) becomes the new extreme. After removing or replacing (i_2), however, the value of removing (i_1) changes dramatically.

Therefore, a static initial ranking followed blindly is unreliable. The greedy ranking should be refreshed after each swap.

## 5. Greedy is good, but it is not globally optimal

Greedy swapping can miss cooperative moves:

* swap 1 alone gives a small loss;
* swap 2 alone gives a small loss;
* both swaps together substantially shrink both boxes.

For (K=4), you could actually test every possible matching among the candidate pools. The number of partial one-to-one matchings is only

[
\sum_{m=0}^{4}
\binom{4}{m}^2m!
================

# 1+16+72+96+24

209.

]

That is still tiny. However, each matching requires calculating the final AABBs after several simultaneous exchanges.

Therefore:

* **greedy (K^2)** is appropriate as the frequent, very cheap operation;
* **full candidate matching** can be an occasional higher-quality local optimization;
* a periodic complete spatial rebuild remains useful when the local algorithm stagnates.

## 6. The main unresolved problem remains group-pair conflicts

Suppose group (A) overlaps groups (B,C,D). All three may independently decide to swap with (A). This cannot safely be committed simultaneously without atomics or locks.

A clean atomic-free schedule is:

### Proposal pass

Each group chooses one best neighboring partner:

[
p(A)=\arg\max_B G(A,B).
]

### Mutual-pair acceptance

Accept only pairs satisfying

[
p(A)=B,\qquad p(B)=A.
]

Each accepted group then belongs to exactly one pair. All accepted pairs can be processed independently in parallel.

A deterministic tie-breaking rule using group IDs handles equal scores.

This is much better than activating only one workgroup globally. You may have many non-conflicting group pairs rebalancing simultaneously.

The practical pipeline becomes:

```text
Kernel 1: refit AABBs and collect candidate/extremum data
Kernel 2: broad phase and choose best partner proposal
Kernel 3: process mutual pairs and perform greedy swaps
```

If additional kernel launches are undesirable, proposals can be collected during frame (t) and applied at the beginning of frame (t+1). The one-frame lag is generally harmless because rebalancing is only a performance optimization, not part of the physical dynamics.

## 7. Preventing oscillation

Pure greedy optimization can cause particles to move back and forth when two arrangements have almost equal cost.

Accept a swap only when

[
G_{ij}>G_{\min},
]

where (G_{\min}) is a meaningful fractional improvement, for example:

[
G_{ij} >
\epsilon,[C(A)+C(B)].
]

You can also add a migration penalty:

[
G_{ij}^{\rm effective}
======================

## G_{ij}

\mu,
]

or temporarily prohibit a particle from moving again for several rebalancing passes.

A deterministic strict-improvement condition is important. Otherwise numerical ties or small geometric oscillations can continually reshuffle memory.

## My recommended version

During AABB refitting, collect:

* first and second extrema in all six directions;
* face-owner IDs;
* optionally two or four additional radial outliers.

During the broad phase, for every overlapping or nearby group pair:

* select neighbor-directional candidate particles;
* evaluate (K^2) swap possibilities;
* calculate the best predicted local gain.

Then:

1. each group proposes one partner;
2. retain mutual proposals;
3. greedily perform up to two or four swaps per accepted pair;
4. recompute the two local boxes after every swap;
5. stop when no swap gives a sufficiently large positive gain.

This is likely a very good compromise:

[
\boxed{
\text{small fixed candidate pools}
+
\text{mutual group pairing}
+
\text{greedy exact-occupancy swaps}
}
]

It preserves full workgroup occupancy, adds little work to the AABB pass, requires no atomics, and is much cheaper than global re-sorting. I would still retain an infrequent global rebuild, because local swaps cannot repair every badly entangled partition.

---

# USER

OK, up to now we were thinking about parallel algorithm sitting on GPU, but these swaps/rebalancing are rare and perhaps we can do it serially on CPU (not every time step, only sparsely), at the same time we can use information which GPU anyway compute every step.

1) GPU update AABB box regularly (if not every frame than very often)
2) GPU compute which groups have non-zero overlap. In fact every group has list of other groups with whom it overlpas, which can be perhaps even sorted.

CPU can take pairs with highest overlaps and rebalance them first. I like the idea of "merge-sort-split" for seriously overlaping groups, and swap for less overlaping groups. swap is basically like light-sort (sorting typically consist of multiple swaps)

Another meaningfull algorithm is that all groups release N worst particles to common pool, and then the particles in pool are assinged to nearest (most suitable) group. 

Elaborate on this therem/ideas

I do not worry too much about global optimality - the goal is eliminate fast obviously patological solutuion (lile group spread ovew whole simulation world)

Also I think it is not so important in the end if we use distance, or other measure (change of volume or change of surface). I do not agree that particle be close to COG and still determine position of AABB wall. COG is by definition in the center of the AABB box. So particle which is close to center cannot be at the same time near the wall. Particle which is far from the center is typically problematic. Although it is posible that there is another particle which is even more problematic both of these particles should be eliminated (typically this make the box prolonged in some diraction like narrow rectangle, which we do not want - we wamnt to have nicely compact cube-like bounding boxes)

---

# GPT 5.6 sol hi

Yes. Once rebalancing is **rare and CPU-side**, the problem becomes much easier. You can use more global information, process conflicts serially, and optimize for simplicity rather than for a perfectly parallel update.

The most useful conceptual view is:

> Your groups are equal-capacity spatial clusters. Rebalancing is a capacity-constrained reassignment of particles between clusters.

Pair swaps, merge–sort–split, and the common-pool method are all versions of the same operation at different scales.

## One terminology correction about the center

There are two different centers:

[
\mathbf c_{\rm AABB}=\frac{\mathbf l+\mathbf u}{2},
]

and

[
\mathbf c_{\rm COG}=\frac{1}{W}\sum_{i=1}^{W}\mathbf x_i.
]

The particle center of gravity is not generally the center of the AABB. For example, if 31 particles are at (x=0) and one is at (x=10), then

[
c_{\rm COG}=0.3125,\qquad c_{\rm AABB}=5.
]

But if by “COG” you mean the **AABB center**, then your argument is correct: a particle defining a distant wall must be far from that center along at least one coordinate.

More importantly, for your actual objective—rapidly eliminating elongated or world-spanning groups—the exact outlier metric is not critical. Ranking particles by

[
s_i=\left|\mathbf x_i-\mathbf c_{\rm AABB}\right|^2
]

is a perfectly reasonable pathology detector. It preferentially removes particles extending a box along its long direction. It may not tell you exactly how much the box shrinks after removing one particle, but after several worst particles are removed, it usually will shrink.

So I agree with your practical argument: the sophisticated “which particle uniquely owns which AABB wall” analysis is optional.

---

# 1. Serial CPU processing removes most synchronization problems

The GPU can periodically export a snapshot containing:

* group AABBs;
* group centers and dimensions;
* overlap-pair list;
* some overlap severity measure;
* optionally the IDs of the (K) farthest particles in each group;
* particle positions, either globally or only for selected groups.

The CPU then processes pathological groups serially:

1. Sort group pairs by severity.
2. Repair the worst pair.
3. Recompute the two affected boxes and their incident overlap scores.
4. Continue until the worst remaining pathology is below a threshold or a repair budget is exhausted.
5. Send the resulting particle permutation back to the GPU.

There are no swap conflicts, atomics, graph coloring, or mutual-pair protocols. The CPU can simply decide that group (A) is being repaired and update everything consistently.

Because the pass is rare, the CPU calculation itself is likely negligible. Data transfer and GPU array reordering will probably dominate.

---

# 2. Swap and merge–sort–split are fundamentally the same operation

Suppose groups (A) and (B) both contain (W) particles. Any new equal-size partition of their union is

[
A'\cup B'=A\cup B,
\qquad
|A'|=|B'|=W.
]

Let (k) particles leave (A). Because (A') must still have (W) particles, exactly (k) particles must enter (A) from (B). Therefore,

[
|A\setminus A'|=|B\setminus B'|=k.
]

Thus any merge–split repartition can be implemented as exactly (k) pair swaps.

So:

* one swap is a (k=1) repartition;
* “light sorting” means restricting (k) to perhaps (1\ldots4);
* full merge–sort–split allows any (0\le k\le W).

The difference is not mathematical expressiveness. The difference is how broadly you search before choosing the set of swaps.

A sequential greedy swap algorithm chooses swaps one at a time. Merge–sort–split selects all (k) exchanges coherently.

---

# 3. A particularly simple pairwise merge–sort–split

For two groups (A,B), merge their (2W) particles. Let their current centers be (\mathbf c_A,\mathbf c_B).

For each particle, calculate

[
\Delta_i
========

## |\mathbf x_i-\mathbf c_A|^2

|\mathbf x_i-\mathbf c_B|^2.
]

Then sort all (2W) particles by (\Delta_i).

Assign:

[
A'=\text{the }W\text{ smallest }\Delta_i,
]

[
B'=\text{the }W\text{ largest }\Delta_i.
]

For fixed centers, this is the exact minimum of

[
\sum_{i\in A'}|\mathbf x_i-\mathbf c_A|^2
+
\sum_{i\in B'}|\mathbf x_i-\mathbf c_B|^2
]

subject to both groups containing exactly (W) particles.

There is also a useful geometrical identity:

[
\Delta_i
========

2\mathbf x_i\cdot(\mathbf c_B-\mathbf c_A)
+
|\mathbf c_A|^2-|\mathbf c_B|^2.
]

Therefore, sorting by (\Delta_i) is equivalent to sorting particles by their projection onto the line connecting the two group centers.

So this pair repair is simply:

> Cut the combined particles by a plane perpendicular to the line between the two group centers, while enforcing exactly (W) particles on each side.

This naturally produces compact, roughly spherical groups.

## Iterative version

Because the centers change after repartitioning, you can perform two or three iterations:

1. compute (\Delta_i);
2. balanced split into (W+W);
3. recompute centers;
4. repeat.

This is essentially a very small **balanced two-cluster (k)-means** problem.

For (2W=32) or (64), the cost on the CPU is trivial.

## Preserving group identity

There are two possible labels for the resulting clusters. Choose the orientation that moves fewer particles:

[
N_{\rm move}
============

|A\setminus A'|+|B\setminus B'|.
]

This prevents the two groups from unnecessarily swapping their complete identities.

You can also add a small movement penalty:

[
\Delta_i
========

## |\mathbf x_i-\mathbf c_A|^2

|\mathbf x_i-\mathbf c_B|^2
+
\mu,q_i,
]

where (q_i) biases particles toward their current group. This provides hysteresis.

---

# 4. When to use light swaps versus full merge–split

Because the full pair split is so cheap on the CPU, I would calculate it even for moderately overlapping pairs.

Then inspect the number (k) of particles that would cross:

* (k=0): no repair is needed;
* (k=1) or (2): apply as a light swap;
* (k=3\ldots 6): moderate repair;
* large (k): the groups were seriously interpenetrating, so apply the complete repartition.

Thus you do not need separate decision algorithms. Run the balanced split, and its result tells you whether the repair is a small swap or a substantial reorganization.

You can still limit movement. For example, for mild overlaps accept only the best (k_{\max}=2) exchanges. For severe pathologies, allow the full split.

## Why full pair repair is better than greedy swapping

Suppose two particles from (A) belong geometrically to (B), and two particles from (B) belong to (A). The first individual swap might not improve the boxes much because the second outlier still keeps the old wall in place.

A simultaneous two-particle exchange can improve both groups substantially.

The merge–split sees this cooperative improvement automatically. Greedy single swaps may not.

---

# 5. The common-pool algorithm is the multi-group generalization

Suppose you select (M) groups and each releases (N) particles. Then the common pool contains

[
P=MN
]

particles, and each group has exactly (N) empty slots.

The task is to assign every pool particle to a group such that:

[
|\mathcal P_g|=N
]

for every participating group (g).

A natural objective is

[
\min_{{\mathcal P_g}}
\sum_g
\sum_{i\in \mathcal P_g}
|\mathbf x_i-\mathbf c_g|^2.
]

This is a **capacity-constrained assignment problem**.

The retained (W-N) particles anchor each group, while the released particles are redistributed among the available slots.

## Why this is more powerful than pair swaps

Any capacity-preserving reassignment can be decomposed into particle-migration cycles.

For example:

[
A\rightarrow B,\qquad
B\rightarrow C,\qquad
C\rightarrow A.
]

This three-group cycle may be beneficial even when no pairwise swap is beneficial:

* the bad particle in (A) belongs to (B);
* the bad particle in (B) belongs to (C);
* the bad particle in (C) belongs to (A).

Pairwise swaps only directly represent two-cycles. The pool method naturally permits cycles of arbitrary length.

This is probably its greatest advantage.

---

# 6. Do not make one global pool unless the system is small

Having every group in the simulation release (N) particles is possible, but usually unnecessary.

Instead, construct a pool from a **local pathological component** of the overlap graph.

For example:

1. Choose the currently worst group (A).
2. Add its (L) strongest-overlap neighbors.
3. Optionally add neighbors that are also elongated or high-degree.
4. Let these (M) groups release (N) particles each.
5. Reassign the resulting (MN)-particle pool.

Typical sizes might be:

[
M=4\ldots16,
\qquad
N=2\ldots8.
]

Then the pool contains tens rather than thousands of particles.

This is already sufficiently global to repair:

* one group containing particles from several distant regions;
* cyclic misassignments;
* groups that have become strongly interwoven;
* situations where repairing one pair merely transfers the problem to another pair.

---

# 7. Choosing which particles enter the pool

Your simple rule is good:

[
s_i=|\mathbf x_i-\mathbf c_g|^2,
]

and release the (N) largest (s_i).

For AABB-oriented grouping, use the AABB center. This directly targets long protrusions.

A slightly more tailored score is

[
s_i=
\sum_{\alpha=x,y,z}
w_\alpha(x_{i\alpha}-c_{g\alpha})^2,
]

where larger weights can be assigned to the currently long dimensions. For example,

[
w_\alpha = L_\alpha^2
]

strongly favors removing particles extending along the longest axis.

But ordinary Euclidean distance is likely sufficient for the pathology-repair objective.

An additional useful candidate is a particle that is much closer to another group center than to its present center:

[
m_i
===

## |\mathbf x_i-\mathbf c_{\rm current}|^2

\min_{h\ne g}
|\mathbf x_i-\mathbf c_h|^2.
]

Large positive (m_i) means that the particle is clearly assigned to the wrong group.

A practical combined score is

[
s_i
===

|\mathbf x_i-\mathbf c_g|^2
+
\lambda m_i.
]

But again, using only radial outlier distance is a defensible simple choice.

---

# 8. Assigning the pool back to groups

For each released particle (i) and candidate group (g), define an assignment cost.

The simplest is

[
C_{ig}=|\mathbf x_i-\mathbf c_g|^2.
]

You could instead use the AABB-center (L^\infty) distance,

[
C_{ig}
======

\max_{\alpha}
|x_{i\alpha}-c_{g\alpha}|^2,
]

which more directly favors small cubic boxes.

Or estimate how much the particle expands the AABB of the retained particles:

[
C_{ig}
======

\Phi(B_g\cup B_i)-\Phi(B_g),
]

where (\Phi) can be diagonal squared, surface area, or volume.

Since the CPU pass is infrequent, one good compromise is:

1. assign using center distance;
2. build the resulting exact AABBs;
3. accept only if the real broad-phase cost improves.

## Exact assignment

Duplicate every group into (N) abstract slots. Then solve the (MN\times MN) particle-to-slot assignment problem.

For small local pools, the Hungarian algorithm or min-cost flow is perfectly feasible.

Because all (N) slots of one group initially have the same cost, this is a balanced transportation problem rather than an ordinary unconstrained nearest-neighbor assignment.

## Simpler regret-greedy assignment

A simpler algorithm is:

1. For every particle, find its best and second-best group with remaining capacity.
2. Define its regret:
   [
   r_i=C_{i,2}-C_{i,1}.
   ]
3. Assign the particle with largest regret to its best group.
4. Reduce that group’s remaining capacity.
5. Recalculate affected choices.
6. Continue until all slots are filled.

This works better than assigning particles in arbitrary order. A particle with only one good destination is assigned early, while flexible particles are left for later.

## Iteration

After assignment:

1. recompute each group center;
2. solve the balanced assignment again;
3. repeat two or three times.

This is balanced multi-group (k)-means, with the retained particles anchoring each cluster.

---

# 9. A hierarchy of repair operators

All your proposed operations fit naturally into one hierarchy:

| Participating groups | Released particles | Result                                        |
| -------------------- | -----------------: | --------------------------------------------- |
| 2                    |              (N=1) | one pair swap                                 |
| 2                    |          small (N) | light multi-swap                              |
| 2                    |              (N=W) | complete merge–sort–split                     |
| (M>2)                |          small (N) | local common-pool repair                      |
| (M>2)                |              (N=W) | complete rebuild of a local overlap component |
| all groups           |              (N=W) | global repartition                            |

So there is really one general operation:

[
\boxed{
\text{release selected particles}
\rightarrow
\text{balanced reassignment to fixed-capacity groups}
}
]

The only decisions are:

* how many groups participate;
* how many particles each group releases;
* how accurately the assignment is solved.

---

# 10. Prioritizing repairs on the CPU

Your idea of processing the highest-overlap pairs first is good, particularly for rapid elimination of pathological arrangements.

I would construct a priority queue of overlap edges:

[
(g,h,S_{gh}).
]

When the highest-priority pair is repaired, recompute all edges incident to (g) or (h).

Because previous repairs make stored scores stale, attach a version counter to each group. When an edge is popped:

* if its group versions still match, process it;
* otherwise recompute its score and put it back into the queue.

This is a standard and simple way to maintain a serial greedy priority process without continually rebuilding the complete sorted list.

## Raw overlap is not always pathology

Since your collision AABBs already include (R_c), neighboring compact groups are expected to overlap. Large overlap does not necessarily mean poor grouping.

It would be useful to maintain two boxes:

[
B_g^{\rm tight}
===============

\operatorname{AABB}{\mathbf x_i},
]

and

[
B_g^{\rm broad}
===============

\operatorname{AABB}{\mathbf x_i\pm R_i}.
]

Use (B^{\rm broad}) for collision detection, but use tight-box shape and overlap to diagnose grouping.

For example, a pair is suspicious when:

* the tight boxes strongly interpenetrate;
* one or both groups have a large aspect ratio;
* a group overlaps an unusually large number of other groups;
* many (W^2) particle tests produce very few actual interactions.

A very simple group pathology score could be

[
P_g
===

N_{\rm overlap}(g)
\left(\frac{D_g}{D_{\rm typical}}\right)^2,
]

where (D_g) is the tight AABB diagonal. A group spread across the simulation world will immediately receive an enormous score.

You do not need a mathematically ideal score. You only need a score that makes obvious pathologies appear near the top of the queue.

---

# 11. Recommended serial repair policy

I would use the following tiered procedure.

### Mild pair contamination

For a pair with moderate overlap:

1. combine the two groups;
2. calculate a balanced split;
3. inspect how many particles would move;
4. accept only the best one or two cross-exchanges.

### Serious pair interpenetration

For strongly overlapping or elongated pairs:

1. combine all (2W) particles;
2. run two or three balanced split iterations;
3. accept the complete equal-size partition.

### High-degree pathological group

When one group overlaps many groups or has a huge diagonal:

1. take that group and perhaps its 4–8 strongest neighbors;
2. release (N=2\ldots8) worst particles from every participating group;
3. perform balanced pool assignment;
4. iterate centers and assignments a few times.

### Catastrophic configuration

If a connected component remains badly mixed:

1. release all particles from that small component;
2. perform a full equal-capacity local repartition.

A global rebuild is only the final fallback.

---

# 12. GPU–CPU implementation pattern

The cleanest sparse update is probably:

1. GPU writes the current metadata and overlap edges.
2. CPU synchronously reads the snapshot during a rare maintenance step.
3. CPU computes a new particle-to-group assignment.
4. CPU sends a permutation or list of reassigned particle slots to the GPU.
5. GPU applies the permutation.
6. AABBs and overlap lists are invalidated and rebuilt.

If groups correspond to contiguous ranges in a structure-of-arrays layout, a complete permutation applied through a gather kernel is usually cleaner than many in-place swaps. However, for only a few exchanges, a list of disjoint swaps or cycles is sufficient.

Because any fixed-capacity reassignment decomposes into cycles, the CPU can explicitly output:

```text
slot 17 -> slot 83
slot 83 -> slot 201
slot 201 -> slot 17
```

The GPU then applies each disjoint cycle safely, or uses a temporary buffer.

## Overall assessment

Moving the rare rebalancing work to the CPU changes my recommendation substantially. You no longer need to constrain yourself to locally synchronized pair swaps.

The best approach is:

[
\boxed{
\begin{aligned}
&\text{GPU continuously detects overlap and compactness problems,}\
&\text{CPU occasionally repairs them using balanced fixed-capacity clustering.}
\end{aligned}
}
]

Pairwise merge–sort–split is ideal for strong two-group mixing. Light swaps are simply small merge–split results. A common outlier pool is the natural extension to multiple groups and is particularly valuable because it permits three-group and longer migration cycles that pairwise swaps cannot discover directly.

For your goal—not global optimality, but rapidly destroying obviously pathological groups—the combination of a serial priority queue, pairwise balanced splitting, and occasional local common-pool redistribution should be both simple and highly effective.
