/// OpenCL kernels for the fixed-W=32 group collision simulation.
///
/// Frame order:
///   1. compute_aabbs(pos_in)
///   2. compute_overlap_bits(aabbs)
///   3. compute_overlap_degrees(overlap_bits)
///   4. collision_step_normal(pos_in, vel_in -> pos_out, vel_out)
///   5. collision_step_pathological(pos_in, vel_in -> pos_out, vel_out)
///
/// The collision kernels use ping-pong state buffers. Every workgroup reads a
/// single immutable input snapshot and writes only its own group range.

#define GROUP_SIZE 32
#define OVERLAP_WORD_BITS 32

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

/// Fast path. Every processed group has at most PATHOLOGICAL_DEGREE partners.
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

/// Slow correctness path. It is dispatched separately and handles every bit
/// in a pathological row instead of truncating the collision neighborhood.
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

/// Compute broad AABBs. The demo requires exactly GROUP_SIZE particles/group.
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

/// One work-item creates one 32-bit word of one exact overlap row.
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

/// Compute exact row degrees and pathological flags from the bit matrix.
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
