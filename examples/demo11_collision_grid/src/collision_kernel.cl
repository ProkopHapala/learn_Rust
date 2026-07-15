/// @file collision_kernel.cl
/// @brief GPU construction and particle-gather kernels for the uniform-grid baseline.
///
/// This demo is the comparison point for demo10's fixed group partition. A
/// dense bounded grid is rebuilt from the current snapshot, particles are made
/// contiguous by cell, and each destination particle gathers contacts from its
/// own cell plus the eight neighbors. The grid is an index, not a PIC field:
/// no particle quantity is deposited or interpolated.
///
/// The design intentionally accepts a rebuild and integer scatter each frame
/// in exchange for simple ownership and a compact, capacity-free cell layout.
/// The collision kernel then reads one immutable sorted snapshot and writes a
/// separate snapshot, because OpenCL workgroups have no general global barrier.
///
/// === AUTO-DOC BEGIN ===
/// Key invariant: cell offsets/counts describe the same sorted position,
/// velocity, ID, and key arrays. The ID buffer is the parity/debug witness.
/// === AUTO-DOC END ===

#define SCAN_W 256
#define SCAN_ELEMS (2 * SCAN_W)
#define CONTACT_EPS2 1.0e-12f

/// @brief Clear a compact metadata array before the next construction phase.
__kernel void clear_uint(__global uint* values, const uint value, const int n) {
    int i = get_global_id(0);
    if (i < n) values[i] = value;
}

/// @brief Reset cell counts and scatter cursors for a new grid build.
/// Both are separate because counts define
/// final ranges while cursors are consumed by the unordered particle scatter.
__kernel void clear_cell_buffers(
    __global uint* cell_count,
    __global uint* cell_cursor,
    const int n_cells
) {
    int i = get_global_id(0);
    if (i < n_cells) {
        cell_count[i] = 0;
        cell_cursor[i] = 0;
    }
}

/// @brief Assign each particle to a bounded cell and count occupancy.
/// The atomic is
/// acceptable here because it occurs during index construction; collision
/// response remains particle-owned and atomic-free.
__kernel void compute_cell_keys_and_count(
    __global const float4* pos,
    __global uint* cell_key,
    __global uint* cell_count,
    const int n,
    const float origin_x,
    const float origin_z,
    const float cell_h,
    const int nx,
    const int nz
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float4 p = pos[i];
    int cx = (int)floor((p.x - origin_x) / cell_h);
    int cz = (int)floor((p.z - origin_z) / cell_h);
    cx = clamp(cx, 0, nx - 1);
    cz = clamp(cz, 0, nz - 1);
    uint cell = (uint)(cx + nx * cz);
    cell_key[i] = cell;
    atomic_inc((volatile __global uint*)&cell_count[cell]);
}

/// @brief Scan one block and emit its total.
/// The host launches a hierarchy because a
/// workgroup-local barrier cannot synchronize the whole cell-count array.
__kernel void scan_block(
    __global const uint* input,
    __global uint* output,
    __global uint* block_sums,
    const int n
) {
    __local uint temp[SCAN_ELEMS];
    int lid = get_local_id(0);
    int group = get_group_id(0);
    int base = group * SCAN_ELEMS;
    int i0 = base + 2 * lid;
    int i1 = i0 + 1;

    temp[2 * lid] = i0 < n ? input[i0] : 0;
    temp[2 * lid + 1] = i1 < n ? input[i1] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < SCAN_ELEMS; offset <<= 1) {
        int index = (lid + 1) * offset * 2 - 1;
        if (index < SCAN_ELEMS) temp[index] += temp[index - offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        block_sums[group] = temp[SCAN_ELEMS - 1];
        temp[SCAN_ELEMS - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = SCAN_ELEMS >> 1; offset > 0; offset >>= 1) {
        int index = (lid + 1) * offset * 2 - 1;
        if (index < SCAN_ELEMS) {
            uint t = temp[index - offset];
            temp[index - offset] = temp[index];
            temp[index] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i0 < n) output[i0] = temp[2 * lid];
    if (i1 < n) output[i1] = temp[2 * lid + 1];
}

/// @brief Add scanned block totals to form global cell offsets.
__kernel void add_block_offsets(
    __global uint* values,
    __global const uint* block_offsets,
    const int n
) {
    int i = get_global_id(0);
    if (i < n) values[i] += block_offsets[i / SCAN_ELEMS];
}

/// @brief Compact all particle records into cell-contiguous ranges.
/// The atomic cursor
/// chooses a unique slot, while writing every associated array together keeps
/// sorted ranges and IDs aligned.
__kernel void scatter_particles(
    __global const float4* pos_in,
    __global const float4* vel_in,
    __global const uint* id_in,
    __global const uint* cell_key,
    __global const uint* cell_offset,
    __global uint* cell_cursor,
    __global float4* pos_out,
    __global float4* vel_out,
    __global uint* id_out,
    __global uint* sorted_key,
    const int n
) {
    int i = get_global_id(0);
    if (i >= n) return;

    uint cell = cell_key[i];
    uint local_slot = atomic_inc((volatile __global uint*)&cell_cursor[cell]);
    uint dst = cell_offset[cell] + local_slot;
    pos_out[dst] = pos_in[i];
    vel_out[dst] = vel_in[i];
    id_out[dst] = id_in[i];
    sorted_key[dst] = cell;
}

/// @brief Gather collision response over the particle's 3x3 cell stencil.
/// Each work-item
/// owns one destination record, so no force atomics or cross-workgroup output
/// ownership are needed; the ID output preserves the permutation for tests.
__kernel void collision_step(
    __global const float4* pos_in,
    __global const float4* vel_in,
    __global const uint* id_in,
    __global float4* pos_out,
    __global float4* vel_out,
    __global uint* id_out,
    __global const uint* sorted_key,
    __global const uint* cell_count,
    __global const uint* cell_offset,
    __global uint* candidate_count,
    __global uint* contact_count,
    __global uint* degenerate_count,
    const int n,
    const int nx,
    const int nz,
    const float dt,
    const float gravity_x,
    const float gravity_y,
    const float gravity_z,
    const float restitution,
    const float box_min_x,
    const float box_min_y,
    const float box_min_z,
    const float box_max_x,
    const float box_max_y,
    const float box_max_z,
    const float k_spring,
    const float k_damp,
    const float vel_damping,
    const int constrain_2d
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float4 pi = pos_in[i];
    float4 vi = vel_in[i];
    int key = (int)sorted_key[i];
    int cx = key % nx;
    int cz = key / nx;
    float3 fi = (float3)(0.0f);
    uint candidates = 0;
    uint contacts = 0;

    for (int dz = -1; dz <= 1; dz++) {
        int qz = cz + dz;
        if (qz < 0 || qz >= nz) continue;
        for (int dx = -1; dx <= 1; dx++) {
            int qx = cx + dx;
            if (qx < 0 || qx >= nx) continue;
            uint cell = (uint)(qx + nx * qz);
            uint begin = cell_offset[cell];
            uint end = begin + cell_count[cell];
            for (uint j = begin; j < end; j++) {
                if (j == (uint)i) continue;
                candidates++;
                float4 pj = pos_in[j];
                float4 vj = vel_in[j];
                float3 r = pj.xyz - pi.xyz;
                float dist_sq = dot(r, r);
                float rsum = pi.w + pj.w;
                if (dist_sq < rsum * rsum) {
                    contacts++;
                    float3 n_hat;
                    float dist;
                    if (dist_sq <= CONTACT_EPS2) {
                        atomic_inc((volatile __global uint*)degenerate_count);
                        uint id_i = id_in[i];
                        uint id_j = id_in[j];
                        uint lo = min(id_i, id_j);
                        uint hi = max(id_i, id_j);
                        uint hash = (lo * 0x9e3779b9U) ^ (hi * 0x85ebca6bU);
                        switch (hash & 3U) {
                            case 0U: n_hat = (float3)(1.0f, 0.0f, 0.0f); break;
                            case 1U: n_hat = (float3)(0.0f, 0.0f, 1.0f); break;
                            case 2U: n_hat = (float3)(0.70710678f, 0.0f, 0.70710678f); break;
                            default: n_hat = (float3)(0.70710678f, 0.0f, -0.70710678f); break;
                        }
                        if (id_i > id_j) n_hat = -n_hat;
                        dist = 0.0f;
                    } else {
                        dist = sqrt(dist_sq);
                        n_hat = r / dist;
                    }
                    float overlap = rsum - dist;
                    float3 v_rel = vj.xyz - vi.xyz;
                    float v_n = dot(v_rel, n_hat);
                    float f_total = -k_spring * overlap;
                    if (v_n < 0.0f) f_total += k_damp * v_n;
                    fi += n_hat * f_total;
                }
            }
        }
    }

    candidate_count[i] = candidates;
    contact_count[i] = contacts;

    float inv_mass = vi.w;
    float3 acceleration = fi * inv_mass + (float3)(gravity_x, gravity_y, gravity_z);
    float4 v = vi;
    v.xyz += acceleration * dt;
    v.xyz *= vel_damping;
    if (constrain_2d) v.y = 0.0f;

    float4 p = pi;
    p.xyz += v.xyz * dt;
    float r = pi.w;
    if (p.x < box_min_x + r) { p.x = box_min_x + r; if (v.x < 0.0f) v.x = -v.x * restitution; }
    if (p.x > box_max_x - r) { p.x = box_max_x - r; if (v.x > 0.0f) v.x = -v.x * restitution; }
    if (p.y < box_min_y + r) { p.y = box_min_y + r; if (v.y < 0.0f) v.y = -v.y * restitution; }
    if (p.y > box_max_y - r) { p.y = box_max_y - r; if (v.y > 0.0f) v.y = -v.y * restitution; }
    if (p.z < box_min_z + r) { p.z = box_min_z + r; if (v.z < 0.0f) v.z = -v.z * restitution; }
    if (p.z > box_max_z - r) { p.z = box_max_z - r; if (v.z > 0.0f) v.z = -v.z * restitution; }
    if (constrain_2d) p.y = 0.0f;

    pos_out[i] = p;
    vel_out[i] = v;
    id_out[i] = id_in[i];
}
