/// collision_kernel.cl — GPU kernels for group-based collision simulation.
///
/// Three kernels work in sequence each frame:
///   1. compute_aabbs   — per-group AABB via tree reduction
///   2. compute_overlaps — tiled group-vs-all-groups overlap detection
///   3. collision_step   — narrow-phase collision using overlap list for pruning
///
/// Data layout (Structure of Arrays, float4 per particle):
///   pos[i] = (x, y, z, radius)     — position + collision radius
///   vel[i] = (vx, vy, vz, inv_mass) — velocity + inverse mass
///
/// Groups are contiguous index ranges: group g = particles [g*W, (g+1)*W).
/// Group membership is implicit from index — no separate membership array.
/// This keeps kernels simple (g = get_group_id(0), i = g*W + lid) but means
/// rebalancing requires permuting particles in the flat arrays on the CPU.
///
/// Complexity (N = n_particles, W = group_size, m = n_groups, mColAv = avg overlaps):
///   Broad phase:  O(m^2 / W)     — tiled, each workgroup scans all m groups
///   Narrow phase: O(W^2 * m * mColAv)  vs brute-force O(W^2 * m^2) = O(N^2)
///
/// CAVEAT: If a group has more than MAX_OVERLAP partners, the overlap list is
/// truncated. Collisions with unlisted groups are silently missed. The
/// overlap_count still holds the true count — overflow should trigger CPU
/// rebalancing before it happens.
///
/// TODO: Two-pass overlap kernel (count pass + fill pass) for variable-size lists.
/// TODO: Tight-box (centers only) vs broad-box (with radius) for diagnosis.

#define MAX_W 256

/// Narrow-phase collision step with group-based broad-phase pruning.
///
/// Each workgroup processes one group g. Particles in g check collisions:
///   - Intra-group: against own W particles (always, W^2 checks)
///   - Inter-group: against each overlapping group h in overlap_list[g] (W^2 per h)
///
/// Force model: linear penalty spring + velocity damping (normal component only).
/// Integration: semi-implicit Euler. Wall collisions: positional clamp + velocity reflection.
///
/// Non-obvious: Inactive threads (i >= n) still participate in shared memory loads
/// and barriers (needed for cooperative tile loading) but skip force accumulation
/// and integration. Dummy particles get inv_mass=1e10 so they never generate
/// meaningful forces if accidentally queried.
///
/// CAVEAT: Wall collision checks are unrolled per-axis because OpenCL C does not
/// support dynamic vector component indexing (.s[axis] with variable axis).
/// CAVEAT: If overlap_count[g] > max_overlap, only the first max_overlap partners
/// are checked — collisions with truncated groups are missed.
__kernel void collision_step(
    __global float4* pos,        // xyz = position, w = radius
    __global float4* vel,        // xyz = velocity, w = inv_mass
    __global float4* force,      // xyz = accumulated force, w = unused
    __global int* overlap_list,  // n_groups * MAX_OVERLAP partner group IDs
    __global int* overlap_count, // n_groups true overlap counts
    const int n,                 // particle count
    const int n_groups,
    const int max_overlap,       // MAX_OVERLAP (capacity of overlap_list per group)
    const float dt,
    const float4 gravity,        // acceleration vector (xyz), w unused
    const float restitution,     // wall bounciness [0,1]
    const float4 box_min,        // enclosing box min (xyz), w unused
    const float4 box_max,        // enclosing box max (xyz), w unused
    const float k_spring,        // collision penalty stiffness
    const float k_damp,          // collision velocity damping
    const float vel_damping,     // global velocity damping per step [0,1]
    const int constrain_2d       // 1 = force y=0
) {
    int g = get_group_id(0);     // group index
    int lid = get_local_id(0);
    int W = get_local_size(0);
    int i = g * W + lid;         // particle index

    float4 pi, vi;
    bool active = (i < n);
    if (active) {
        pi = pos[i];
        vi = vel[i];
    } else {
        pi = (float4)(0.0f);
        vi = (float4)(0.0f, 0.0f, 0.0f, 1e10f); // infinite mass dummy
    }

    float3 fi = (float3)(0.0f);

    __local float4 tile_pos[MAX_W];
    __local float4 tile_vel[MAX_W];

    // --- Intra-group collisions (g vs g) ---
    tile_pos[lid] = pi;
    tile_vel[lid] = vi;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (active) {
        for (int k = 0; k < W; k++) {
            if (k == lid) continue;
            float4 pj = tile_pos[k];
            float4 vj = tile_vel[k];
            float3 r = pj.xyz - pi.xyz;
            float dist_sq = dot(r, r);
            float rsum = pi.w + pj.w;
            if (dist_sq < rsum * rsum && dist_sq > 1e-12f) {
                float dist = sqrt(dist_sq);
                float3 n_hat = r / dist;
                float overlap = rsum - dist;
                float3 v_rel = vj.xyz - vi.xyz;
                float v_n = dot(v_rel, n_hat);
                float f_total = -k_spring * overlap;
                if (v_n < 0.0f) f_total += k_damp * v_n;
                fi += n_hat * f_total;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Inter-group collisions: iterate over overlap list ---
    int n_ov = min(overlap_count[g], max_overlap);
    for (int oi = 0; oi < n_ov; oi++) {
        int h = overlap_list[g * max_overlap + oi];
        if (h < 0) break;

        // Load group h's particles cooperatively into shared memory
        int j = h * W + lid;
        if (j < n) {
            tile_pos[lid] = pos[j];
            tile_vel[lid] = vel[j];
        } else {
            tile_pos[lid] = (float4)(0.0f);
            tile_vel[lid] = (float4)(0.0f, 0.0f, 0.0f, 1e10f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (active) {
            for (int k = 0; k < W; k++) {
                if (h * W + k >= n) break;
                float4 pj = tile_pos[k];
                float4 vj = tile_vel[k];
                float3 r = pj.xyz - pi.xyz;
                float dist_sq = dot(r, r);
                float rsum = pi.w + pj.w;
                if (dist_sq < rsum * rsum && dist_sq > 1e-12f) {
                    float dist = sqrt(dist_sq);
                    float3 n_hat = r / dist;
                    float overlap = rsum - dist;
                    float3 v_rel = vj.xyz - vi.xyz;
                    float v_n = dot(v_rel, n_hat);
                    float f_total = -k_spring * overlap;
                    if (v_n < 0.0f) f_total += k_damp * v_n;
                    fi += n_hat * f_total;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!active) return;

    // Gravity (F = m * g = g / inv_mass)
    float inv_mass = vi.w;
    fi += gravity.xyz / inv_mass;

    force[i] = (float4)(fi, 0.0f);

    // Semi-implicit Euler
    float4 v = vi;
    v.xyz += fi * dt * inv_mass;
    v.xyz *= vel_damping;

    if (constrain_2d) v.y = 0.0f;

    // Update position
    float4 p = pi;
    p.xyz += v.xyz * dt;

    // Box wall collisions (unrolled — OpenCL C doesn't support dynamic .s[axis])
    float r = pi.w;
    if (p.x < box_min.x + r) { p.x = box_min.x + r; if (v.x < 0.0f) v.x = -v.x * restitution; }
    if (p.x > box_max.x - r) { p.x = box_max.x - r; if (v.x > 0.0f) v.x = -v.x * restitution; }
    if (p.y < box_min.y + r) { p.y = box_min.y + r; if (v.y < 0.0f) v.y = -v.y * restitution; }
    if (p.y > box_max.y - r) { p.y = box_max.y - r; if (v.y > 0.0f) v.y = -v.y * restitution; }
    if (p.z < box_min.z + r) { p.z = box_min.z + r; if (v.z < 0.0f) v.z = -v.z * restitution; }
    if (p.z > box_max.z - r) { p.z = box_max.z - r; if (v.z > 0.0f) v.z = -v.z * restitution; }

    if (constrain_2d) p.y = 0.0f;

    pos[i] = p;
    vel[i] = v;
}

/// Compute broad AABB for each group of W particles.
///
/// One workgroup per group. Tree reduction in local memory finds min/max
/// of (position - radius) and (position + radius) across all W particles.
///
/// The AABB includes particle radius — this is a "broad box" suitable for
/// collision pruning. Two groups whose broad AABBs overlap may contain
/// colliding particle pairs (within 2R). Non-overlapping groups cannot.
///
/// Non-obvious: For 2D mode, y-components are still computed but unused
/// by the overlap kernel (which checks only x,z). This wastes a small
/// amount of compute but keeps the kernel 3D-ready.
///
/// TODO: Add tight-box variant (centers only, no radius) for diagnosis.
__kernel void compute_aabbs(
    __global float4* pos,        // xyz = position, w = radius
    __global float4* aabb_min,   // xyz = min corner, w unused
    __global float4* aabb_max,   // xyz = max corner, w unused
    const int n                  // particle count
) {
    int g = get_group_id(0);
    int lid = get_local_id(0);
    int W = get_local_size(0);
    int i = g * W + lid;

    __local float3 lmin[MAX_W];
    __local float3 lmax[MAX_W];

    if (i < n) {
        float4 p = pos[i];
        lmin[lid] = p.xyz - p.w;
        lmax[lid] = p.xyz + p.w;
    } else {
        lmin[lid] = (float3)(1e30f);
        lmax[lid] = (float3)(-1e30f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = W / 2; step > 0; step >>= 1) {
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

/// Compute group-to-group AABB overlaps in tiled manner.
///
/// Each workgroup processes one group g, tiles through all n_groups AABBs
/// in chunks of W. Stores up to MAX_OVERLAP partner IDs per group.
/// overlap_count[g] holds the true count (may exceed MAX_OVERLAP).
///
/// If count > MAX_OVERLAP, the list is truncated — which partners are kept
/// is non-deterministic (atomic_inc ordering). The overflow itself signals
/// pathology: a group with >32 overlapping neighbors is pathologically
/// spread out and needs CPU rebalancing.
///
/// Non-obvious: The list is initialized to -1 sentinel values, so the
/// collision_step kernel can break early on -1 entries. The atomic_inc
/// on count is necessary because multiple threads in the same workgroup
/// may find overlaps in the same tile.
///
/// CAVEAT: Truncated overlap lists cause missed collisions in collision_step.
/// This is a correctness issue, not just performance. The overflow count
/// in the UI is the early-warning signal.
/// CAVEAT: 2D only — checks x,z axes. For 3D, add y-axis overlap check.
///
/// TODO: Two-pass version: pass 1 counts (no list), pass 2 fills variable-size list.
/// TODO: Sort overlaps by overlap area so the most important partners are kept on truncation.
#define MAX_OVERLAP 32

__kernel void compute_overlaps(
    __global float4* aabb_min,
    __global float4* aabb_max,
    __global int* overlap_list,     // n_groups * MAX_OVERLAP
    __global int* overlap_count,    // n_groups (true count, may exceed MAX_OVERLAP)
    const int n_groups
) {
    int g = get_group_id(0);
    int lid = get_local_id(0);
    int W = get_local_size(0);

    __local int count;
    __local int list[MAX_OVERLAP];

    if (lid == 0) count = 0;
    for (int i = lid; i < MAX_OVERLAP; i += W) list[i] = -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 gmin = aabb_min[g];
    float4 gmax = aabb_max[g];

    int n_tiles = (n_groups + W - 1) / W;
    for (int t = 0; t < n_tiles; t++) {
        int j = t * W + lid;
        if (j < n_groups && j != g) {
            float4 jmin = aabb_min[j];
            float4 jmax = aabb_max[j];
            // 2D overlap check (x, z only)
            bool ov = gmin.x <= jmax.x && gmax.x >= jmin.x &&
                      gmin.z <= jmax.z && gmax.z >= jmin.z;
            if (ov) {
                int slot = atomic_inc(&count);
                if (slot < MAX_OVERLAP) list[slot] = j;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        overlap_count[g] = count;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_write = min(count, MAX_OVERLAP);
    for (int i = lid; i < n_write; i += W) {
        overlap_list[g * MAX_OVERLAP + i] = list[i];
    }
}
