#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

struct CanonicalSplat {
    packed_float3 position;
    packed_half4 color;
    float rotationX;
    float rotationY;
    float rotationZ;
    float rotationW;
    packed_float3 scale;
};

// Copy paste the covariance computation function from https://github.com/yunjinli/TRASE/blob/030b59d133afc8a6b6403e6584bf8cecf1a46688/scene/gaussian_model.py#L39
void compute_cov(float4 rot, float3 scale, thread packed_half3 &covA, thread packed_half3 &covB) {
    float x = rot.x, y = rot.y, z = rot.z, w = rot.w;
    float3x3 R = float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
        2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)
    );
    float3x3 S = float3x3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    float3x3 M = R * S;
    float3x3 Sigma = M * transpose(M);
    covA = packed_half3(half(Sigma[0][0]), half(Sigma[0][1]), half(Sigma[0][2]));
    covB = packed_half3(half(Sigma[1][1]), half(Sigma[1][2]), half(Sigma[2][2]));
}

// Extract xyz and t from the canonical Gaussians.
kernel void extract_graph_inputs(
    device const CanonicalSplat* inSplats [[ buffer(0) ]],
    device float* outXYZ                [[ buffer(1) ]],
    device float* outT                  [[ buffer(2) ]],
    constant float& time                [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]]
) {
    CanonicalSplat s = inSplats[id];
    outXYZ[id * 3 + 0] = s.position.x;
    outXYZ[id * 3 + 1] = s.position.y;
    outXYZ[id * 3 + 2] = s.position.z;
    outT[id] = time;
}

// Apply d_xyz, d_rotation, d_scaling to the canonical Gaussians.
kernel void apply_graph_outputs(
    device const CanonicalSplat* inSplats [[ buffer(0) ]],
    device const float* dXYZ              [[ buffer(1) ]],
    device const float* dRot              [[ buffer(2) ]],
    device const float* dScale            [[ buffer(3) ]],
    device Splat* outSplats               [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    CanonicalSplat input = inSplats[id];
    
    float3 d_xyz = float3(dXYZ[id*3+0], dXYZ[id*3+1], dXYZ[id*3+2]);
    float4 d_rotation = float4(dRot[id*4+0], dRot[id*4+1], dRot[id*4+2], dRot[id*4+3]);
    float3 d_scaling = float3(dScale[id*3+0], dScale[id*3+1], dScale[id*3+2]);
    
    // Apply
    float3 new_pos = input.position + d_xyz;
    float4 rot = float4(input.rotationX, input.rotationY, input.rotationZ, input.rotationW);
    float4 new_rot = normalize(rot) + d_rotation;
    float3 new_scale = exp(input.scale) + d_scaling;
    new_scale = log(new_scale);
    
    Splat out;
    out.position = packed_float3(new_pos);
    out.color = input.color;
    compute_cov(new_rot, new_scale, out.covA, out.covB);
    
    outSplats[id] = out;
}

