// N-body gravity kernel: each work-item computes total force on one particle (O(N²))
// Then integrates velocity + position (semi-implicit Euler)

__kernel void nbody_step(
    __global float4* pos,      // xyz = position, w = mass
    __global float4* vel,      // xyz = velocity, w = unused
    __global float4* force,    // xyz = accumulated force, w = unused
    const int n,
    const float dt,
    const float softening_sq
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float4 pi = pos[i];
    float3 fi = (float3)(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        float4 pj = pos[j];
        float3 r = pj.xyz - pi.xyz;
        float dist_sq = dot(r, r) + softening_sq;
        float inv_dist = rsqrt(dist_sq);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float f = pi.w * pj.w * inv_dist3;
        fi += r * f;
    }

    force[i] = (float4)(fi, 0.0f);

    // Semi-implicit Euler: update velocity first, then position
    float4 v = vel[i];
    v.xyz += fi.xyz * dt / pi.w;
    v.xyz *= 0.999f; // mild damping
    pos[i].xyz += v.xyz * dt;
    vel[i] = v;
}
