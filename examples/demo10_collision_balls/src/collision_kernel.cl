/// @file collision_kernel.cl
/// @brief Group-owned collision, broad-phase, and experimental repair kernels.
///
/// This file exists to study a fixed-size group broad phase: spatially nearby
/// particles are kept contiguous so one workgroup can reuse a local tile. The
/// price is that motion gradually makes group ownership stale, so repair is a
/// separate operation rather than a hidden side effect of collision response.
///
/// The normal frame contract is:
///   1. compute_aabbs(pos_in)
///   2. compute_overlap_bits(aabbs)
///   3. compute_overlap_degrees(overlap_bits)
///   4. collision_step_normal(pos_in -> pos_out)
///   5. collision_step_pathological(pos_in -> pos_out)
///   6. swap the ping-pong buffers on the host
///
/// Collision workgroups read one immutable snapshot and own their destination
/// group. The exact bit matrix deliberately has no 32-neighbor correctness
/// limit; degree > 32 selects a rare complete fallback path. The repair kernels
/// at the end use a separate, host-validated disjoint-pair plan and never run
/// concurrently with collision.
///
/// === AUTO-DOC BEGIN ===
/// Key invariants: W=32, position/velocity records move together, overlap rows
/// are exact and symmetric, and every repair destination has one owner.
/// === AUTO-DOC END ===

#define GROUP_SIZE 32
#define OVERLAP_WORD_BITS 32
#define RETILE_SIZE (2 * GROUP_SIZE)

/// @brief Evaluate one soft-sphere spring/damping interaction.
inline float3 pair_force(float4 pi, float4 vi, float4 pj, float4 vj,
                         float k_spring, float k_damp) {
    float3 r = pj.xyz - pi.xyz;
    float dist_sq = dot(r, r);
    float rsum = pi.w + pj.w;
    if (dist_sq >= rsum * rsum || dist_sq <= 1e-12f) return (float3)(0.0f);

    float dist = sqrt(dist_sq);
    float3 n_hat = r / dist;
    float overlap = rsum - dist;
    float3 v_rel = vj.xyz - vi.xyz;
    float v_n = dot(v_rel, n_hat);
    float f_total = -k_spring * overlap;
    if (v_n < 0.0f) f_total += k_damp * v_n;
    return n_hat * f_total;
}

/// @brief Accumulate one complete neighboring group through a reusable local tile.
/// Every work-item participates in both barriers; callers must not diverge
/// inside this helper or the workgroup can deadlock.
inline void accumulate_neighbor(
    int h,
    int lid,
    __global const float4* pos_in,
    __global const float4* vel_in,
    float4 pi,
    float4 vi,
    float3* fi,
    float k_spring,
    float k_damp,
    __local float4* tile_pos,
    __local float4* tile_vel
) {
    int j = h * GROUP_SIZE + lid;
    tile_pos[lid] = pos_in[j];
    tile_vel[lid] = vel_in[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < GROUP_SIZE; k++) {
        *fi += pair_force(pi, vi, tile_pos[k], tile_vel[k], k_spring, k_damp);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

/// @brief Shared collision implementation for normal and pathological dispatches.
/// The only deliberate difference between them is which degree class is
/// admitted; the pathological dispatch scans every set bit in its exact row.
inline void collision_step_impl(
    __global const float4* pos_in,
    __global const float4* vel_in,
    __global float4* pos_out,
    __global float4* vel_out,
    __global const uint* overlap_bits,
    __global const uint* degree,
    const int n_groups,
    const uint pathological_degree,
    const int pathological_mode,
    const float dt,
    const float4 gravity,
    const float restitution,
    const float4 box_min,
    const float4 box_max,
    const float k_spring,
    const float k_damp,
    const float vel_damping,
    const int constrain_2d,
    __local float4* tile_pos,
    __local float4* tile_vel
) {
    int g = get_group_id(0);
    int lid = get_local_id(0);
    if (g >= n_groups) return;

    uint is_pathological = degree[g] > pathological_degree;
    if ((pathological_mode != 0) != (is_pathological != 0)) return;

    int i = g * GROUP_SIZE + lid;
    float4 pi = pos_in[i];
    float4 vi = vel_in[i];
    float3 fi = (float3)(0.0f);

    tile_pos[lid] = pi;
    tile_vel[lid] = vi;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < GROUP_SIZE; k++) {
        if (k != lid) {
            fi += pair_force(pi, vi, tile_pos[k], tile_vel[k], k_spring, k_damp);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_words = (n_groups + OVERLAP_WORD_BITS - 1) / OVERLAP_WORD_BITS;
    for (int word = 0; word < n_words; word++) {
        uint mask = overlap_bits[g * n_words + word];
        while (mask != 0u) {
            uint bit = (uint)ctz(mask);
            int h = word * OVERLAP_WORD_BITS + (int)bit;
            accumulate_neighbor(h, lid, pos_in, vel_in,
                                pi, vi, &fi, k_spring, k_damp,
                                tile_pos, tile_vel);
            mask &= mask - 1u;
        }
    }

    float inv_mass = vi.w;
    float4 v = vi;
    if (inv_mass > 0.0f) v.xyz += (fi * inv_mass + gravity.xyz) * dt;
    v.xyz *= vel_damping;
    if (constrain_2d) v.y = 0.0f;

    float4 p = pi;
    p.xyz += v.xyz * dt;

    float r = pi.w;
    if (p.x < box_min.x + r) { p.x = box_min.x + r; if (v.x < 0.0f) v.x = -v.x * restitution; }
    if (p.x > box_max.x - r) { p.x = box_max.x - r; if (v.x > 0.0f) v.x = -v.x * restitution; }
    if (p.y < box_min.y + r) { p.y = box_min.y + r; if (v.y < 0.0f) v.y = -v.y * restitution; }
    if (p.y > box_max.y - r) { p.y = box_max.y - r; if (v.y > 0.0f) v.y = -v.y * restitution; }
    if (p.z < box_min.z + r) { p.z = box_min.z + r; if (v.z < 0.0f) v.z = -v.z * restitution; }
    if (p.z > box_max.z - r) { p.z = box_max.z - r; if (v.z > 0.0f) v.z = -v.z * restitution; }
    if (constrain_2d) p.y = 0.0f;

    pos_out[i] = p;
    vel_out[i] = v;
}

/// @brief Process groups below the pathological degree threshold.
/// It still uses the exact
/// bit matrix, while the degree gate keeps pathological rows out of this pass.
__kernel void collision_step_normal(
    __global const float4* pos_in,
    __global const float4* vel_in,
    __global float4* pos_out,
    __global float4* vel_out,
    __global const uint* overlap_bits,
    __global const uint* degree,
    const int n_groups,
    const uint pathological_degree,
    const float dt,
    const float4 gravity,
    const float restitution,
    const float4 box_min,
    const float4 box_max,
    const float k_spring,
    const float k_damp,
    const float vel_damping,
    const int constrain_2d
) {
    __local float4 tile_pos[GROUP_SIZE];
    __local float4 tile_vel[GROUP_SIZE];
    collision_step_impl(pos_in, vel_in, pos_out, vel_out,
                        overlap_bits, degree, n_groups, pathological_degree, 0, dt, gravity,
                        restitution, box_min, box_max, k_spring, k_damp,
                        vel_damping, constrain_2d, tile_pos, tile_vel);
}

/// @brief Process high-degree groups through the complete exact-row fallback.
/// Its lower expected
/// frequency permits the complete row scan without burdening a fixed neighbor
/// array or silently dropping contacts beyond 32.
__kernel void collision_step_pathological(
    __global const float4* pos_in,
    __global const float4* vel_in,
    __global float4* pos_out,
    __global float4* vel_out,
    __global const uint* overlap_bits,
    __global const uint* degree,
    const int n_groups,
    const uint pathological_degree,
    const float dt,
    const float4 gravity,
    const float restitution,
    const float4 box_min,
    const float4 box_max,
    const float k_spring,
    const float k_damp,
    const float vel_damping,
    const int constrain_2d
) {
    __local float4 tile_pos[GROUP_SIZE];
    __local float4 tile_vel[GROUP_SIZE];
    collision_step_impl(pos_in, vel_in, pos_out, vel_out,
                        overlap_bits, degree, n_groups, pathological_degree, 1, dt, gravity,
                        restitution, box_min, box_max, k_spring, k_damp,
                        vel_damping, constrain_2d, tile_pos, tile_vel);
}

/// @brief Reduce one conservative radius-expanded AABB per contiguous group.
/// The
/// result is broad-phase metadata: overlap means "candidate", not contact.
__kernel void compute_aabbs(
    __global const float4* pos,
    __global float4* aabb_min,
    __global float4* aabb_max,
    const int n
) {
    int g = get_group_id(0);
    int lid = get_local_id(0);
    int i = g * GROUP_SIZE + lid;

    __local float3 lmin[GROUP_SIZE];
    __local float3 lmax[GROUP_SIZE];
    float4 p = pos[i];
    lmin[lid] = p.xyz - p.w;
    lmax[lid] = p.xyz + p.w;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = GROUP_SIZE / 2; step > 0; step >>= 1) {
        if (lid < step) {
            lmin[lid] = min(lmin[lid], lmin[lid + step]);
            lmax[lid] = max(lmax[lid], lmax[lid + step]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        aabb_min[g] = (float4)(lmin[0], 0.0f);
        aabb_max[g] = (float4)(lmax[0], 0.0f);
    }
}

/// @brief Produce one word of one exact overlap row.
/// A bit matrix is used because the
/// group count is small enough that exact storage is cheaper than overflow
/// handling in every collision workgroup.
__kernel void compute_overlap_bits(
    __global const float4* aabb_min,
    __global const float4* aabb_max,
    __global uint* overlap_bits,
    const int n_groups,
    const int n_words
) {
    int id = get_global_id(0);
    int total = n_groups * n_words;
    if (id >= total) return;

    int g = id / n_words;
    int word = id - g * n_words;
    float4 gmin = aabb_min[g];
    float4 gmax = aabb_max[g];
    uint mask = 0u;

    for (int bit = 0; bit < OVERLAP_WORD_BITS; bit++) {
        int h = word * OVERLAP_WORD_BITS + bit;
        if (h >= n_groups || h == g) continue;
        float4 hmin = aabb_min[h];
        float4 hmax = aabb_max[h];
        bool ov = gmin.x <= hmax.x && gmax.x >= hmin.x &&
                  gmin.z <= hmax.z && gmax.z >= hmin.z;
        if (ov) mask |= 1u << bit;
    }
    overlap_bits[id] = mask;
}

/// @brief Count exact row degree and produce the pathological dispatch signal.
/// Degree is both a diagnostic and the dispatch signal
/// separating the common collision path from the rare complete fallback.
__kernel void compute_overlap_degrees(
    __global const uint* overlap_bits,
    __global uint* degree,
    const int n_groups,
    const int n_words
) {
    int g = get_global_id(0);
    if (g >= n_groups) return;

    uint d = 0u;
    for (int word = 0; word < n_words; word++) {
        d += (uint)popcount(overlap_bits[g * n_words + word]);
    }
    degree[g] = d;
}

/// @brief Gather a compact pair-major snapshot for the hybrid CPU retile backend.
/// Pair p occupies RETILE_SIZE records, group g followed by h. The compact
/// transfer is the experiment: CPU decision quality is held constant while
/// host traffic changes from O(N) to O(number of selected pairs).
__kernel void gather_retile_pairs(
    __global const float4* pos,
    __global const float4* vel,
    __global const uint* pair_groups,
    __global float4* stage_pos,
    __global float4* stage_vel,
    const int n_pairs
) {
    int i = get_global_id(0);
    if (i >= n_pairs * RETILE_SIZE) return;
    int pair = i / RETILE_SIZE;
    int slot = i - pair * RETILE_SIZE;
    int side = slot / GROUP_SIZE;
    int lid = slot - side * GROUP_SIZE;
    int src = (int)pair_groups[pair * 2 + side] * GROUP_SIZE + lid;
    stage_pos[i] = pos[src];
    stage_vel[i] = vel[src];
}

/// @brief Commit the compact hybrid result into uniquely owned destinations.
/// Pair groups are host-validated as
/// disjoint, so every destination is owned by exactly one work-item and no
/// atomic exchange or in-place cycle resolution is needed.
__kernel void commit_retile_pairs(
    __global const float4* stage_pos,
    __global const float4* stage_vel,
    __global const uint* pair_groups,
    __global float4* pos,
    __global float4* vel,
    const int n_pairs
) {
    int i = get_global_id(0);
    if (i >= n_pairs * RETILE_SIZE) return;
    int pair = i / RETILE_SIZE;
    int slot = i - pair * RETILE_SIZE;
    int side = slot / GROUP_SIZE;
    int lid = slot - side * GROUP_SIZE;
    int dst = (int)pair_groups[pair * 2 + side] * GROUP_SIZE + lid;
    pos[dst] = stage_pos[i];
    vel[dst] = stage_vel[i];
}

// Keep projection ordering reproducible without changing contraction in the
// normal collision kernels above.
#pragma OPENCL FP_CONTRACT OFF

/// @brief Test conservative overlap in the x-z plane.
inline int aabb_overlap_2d(float2 amin, float2 amax, float2 bmin, float2 bmax) {
    return amin.x <= bmax.x && amax.x >= bmin.x &&
           amin.y <= bmax.y && amax.y >= bmin.y;
}

/// @brief Compute the perimeter proxy used by repair acceptance.
inline float aabb_perimeter_2d(float2 mn, float2 mx) {
    return 2.0f * ((mx.x - mn.x) + (mx.y - mn.y));
}

/// @brief Evaluate a candidate pair against the immutable group map snapshot.
inline uint retile_overlap_count(
    int g,
    int h,
    float2 gmin,
    float2 gmax,
    float2 hmin,
    float2 hmax,
    __global const float4* aabb_min,
    __global const float4* aabb_max,
    int n_groups
) {
    uint overlaps = (uint)aabb_overlap_2d(gmin, gmax, hmin, hmax);
    for (int k = 0; k < n_groups; k++) {
        if (k == g || k == h) continue;
        float4 kmn4 = aabb_min[k];
        float4 kmx4 = aabb_max[k];
        float2 kmn = (float2)(kmn4.x, kmn4.z);
        float2 kmx = (float2)(kmx4.x, kmx4.z);
        overlaps += (uint)aabb_overlap_2d(gmin, gmax, kmn, kmx);
        overlaps += (uint)aabb_overlap_2d(hmin, hmax, kmn, kmx);
    }
    return overlaps;
}

/// @brief Apply the shared lexicographic overlap-then-perimeter acceptance rule.
inline int retile_cost_better(uint overlaps, float perimeter, uint best_overlaps, float best_perimeter) {
    return overlaps < best_overlaps || (overlaps == best_overlaps && perimeter < best_perimeter - 1e-6f);
}

/// @brief Deterministically order 64 records for the GPU parity path.
/// Particle-local index breaks equal projection keys so the CPU and GPU have a
/// stable structural result instead of merely comparable quality values.
inline void sort_retile_projection(
    __local const float4* pair_pos,
    __local float* keys,
    __local uint* order,
    float2 dir,
    int lid
) {
    float sx = pair_pos[lid].x * dir.x;
    float sz = pair_pos[lid].z * dir.y;
    keys[lid] = sx + sz;
    order[lid] = (uint)lid;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int width = 2; width <= RETILE_SIZE; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            int other = lid ^ stride;
            if (other > lid) {
                float ka = keys[lid];
                float kb = keys[other];
                uint ia = order[lid];
                uint ib = order[other];
                int greater = ka > kb || (ka == kb && ia > ib);
                int less = ka < kb || (ka == kb && ia < ib);
                int do_swap = ((lid & width) == 0) ? greater : less;
                if (do_swap) {
                    keys[lid] = kb;
                    keys[other] = ka;
                    order[lid] = ib;
                    order[other] = ia;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

/// @brief Compute the two candidate AABBs after projection sorting.
/// The two halves
/// reduce independently because each half is the future owner of one group.
inline void reduce_retile_aabbs(
    __local const float4* pair_pos,
    __local const uint* order,
    __local float2* mins,
    __local float2* maxs,
    int lid
) {
    float4 p = pair_pos[order[lid]];
    mins[lid] = (float2)(p.x - p.w, p.z - p.w);
    maxs[lid] = (float2)(p.x + p.w, p.z + p.w);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = GROUP_SIZE >> 1; step > 0; step >>= 1) {
        if ((lid & (GROUP_SIZE - 1)) < step) {
            mins[lid] = min(mins[lid], mins[lid + step]);
            maxs[lid] = max(maxs[lid], maxs[lid + step]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/// @brief Perform device-resident snapshot-batch 2W -> W+W retile.
/// Pair selection is kept on
/// the host because the exact map is tiny; particle records and candidate
/// evaluation remain on the device. One workgroup owns one disjoint pair.
///
/// The kernel reads both groups completely before writing them back to the same
/// state buffers. This in-place update is safe only because the host plan makes
/// pair ownership disjoint and no collision kernel is in flight.
__kernel void retile_pairs_gpu(
    __global float4* pos,
    __global float4* vel,
    __global const float4* aabb_min,
    __global const float4* aabb_max,
    __global const uint* pair_groups,
    __global uint* accepted_flags,
    const int n_groups,
    const int n_pairs
) {
    int pair = get_group_id(0);
    int lid = get_local_id(0);
    if (pair >= n_pairs) return;

    int g = (int)pair_groups[pair * 2];
    int h = (int)pair_groups[pair * 2 + 1];
    int src = lid < GROUP_SIZE ? g * GROUP_SIZE + lid : h * GROUP_SIZE + lid - GROUP_SIZE;

    __local float4 pair_pos[RETILE_SIZE];
    __local float4 pair_vel[RETILE_SIZE];
    __local float keys[RETILE_SIZE];
    __local uint order[RETILE_SIZE];
    __local uint best_order[RETILE_SIZE];
    __local float2 mins[RETILE_SIZE];
    __local float2 maxs[RETILE_SIZE];
    __local uint best_overlaps;
    __local float best_perimeter;
    __local int take_candidate;
    __local int accepted;

    pair_pos[lid] = pos[src];
    pair_vel[lid] = vel[src];
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 gmn4 = aabb_min[g];
    float4 gmx4 = aabb_max[g];
    float4 hmn4 = aabb_min[h];
    float4 hmx4 = aabb_max[h];
    float2 gmn = (float2)(gmn4.x, gmn4.z);
    float2 gmx = (float2)(gmx4.x, gmx4.z);
    float2 hmn = (float2)(hmn4.x, hmn4.z);
    float2 hmx = (float2)(hmx4.x, hmx4.z);
    if (lid == 0) {
        best_overlaps = retile_overlap_count(g, h, gmn, gmx, hmn, hmx, aabb_min, aabb_max, n_groups);
        best_perimeter = aabb_perimeter_2d(gmn, gmx) + aabb_perimeter_2d(hmn, hmx);
        accepted = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float2 center_dir = (float2)(hmx.x + hmn.x - gmx.x - gmn.x, hmx.y + hmn.y - gmx.y - gmn.y) * 0.5f;
    for (int candidate = 0; candidate < 3; candidate++) {
        float2 dir = candidate == 0 ? (float2)(1.0f, 0.0f) :
                     candidate == 1 ? (float2)(0.0f, 1.0f) : center_dir;
        if (dot(dir, dir) > 1e-20f) {
            sort_retile_projection(pair_pos, keys, order, dir, lid);
            reduce_retile_aabbs(pair_pos, order, mins, maxs, lid);
            if (lid == 0) {
                uint overlaps = retile_overlap_count(g, h, mins[0], maxs[0], mins[GROUP_SIZE], maxs[GROUP_SIZE], aabb_min, aabb_max, n_groups);
                float perimeter = aabb_perimeter_2d(mins[0], maxs[0]) + aabb_perimeter_2d(mins[GROUP_SIZE], maxs[GROUP_SIZE]);
                take_candidate = retile_cost_better(overlaps, perimeter, best_overlaps, best_perimeter);
                if (take_candidate) {
                    best_overlaps = overlaps;
                    best_perimeter = perimeter;
                    accepted = 1;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (take_candidate) best_order[lid] = order[lid];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (accepted) {
        int dst = lid < GROUP_SIZE ? g * GROUP_SIZE + lid : h * GROUP_SIZE + lid - GROUP_SIZE;
        uint selected = best_order[lid];
        pos[dst] = pair_pos[selected];
        vel[dst] = pair_vel[selected];
    }
    if (lid == 0) accepted_flags[pair] = (uint)accepted;
}
