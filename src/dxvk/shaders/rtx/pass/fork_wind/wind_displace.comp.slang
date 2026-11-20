// src/dxvk/shaders/rtx/pass/fork_wind/wind_displace.comp.slang
//
// Fork-owned. Foliage wind: writes the displaced positions of one mesh (tight float3) from its rest positions
// (interleaved). The sway is computed in the root space of the tree (z up, origin at the base): nothing moves below
// heightStart, the bend grows with height. Bark and leaves share the low frequency terms so leaves stay on their
// branches; only alpha tested surfaces (flutter > 0) get the high frequency leaf flutter.
#include "rtx/pass/fork_wind/wind_args.h"

layout(binding = WIND_BINDING_REST_INPUT)
StructuredBuffer<float> restData;

layout(binding = WIND_BINDING_OUTPUT)
RWStructuredBuffer<float> outData;

layout(push_constant)
ConstantBuffer<WindArgs> cb;

float3 affine(float4 r0, float4 r1, float4 r2, float3 p)
{
  const float4 h = float4(p, 1.0);
  return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
  const uint idx = tid.x;
  if (idx >= cb.numVertices) {
    return;
  }

  const uint s = cb.srcOffset + idx * cb.srcStride;
  const float3 rest = float3(restData[s], restData[s + 1], restData[s + 2]);
  const float3 p = affine(cb.meshToRoot0, cb.meshToRoot1, cb.meshToRoot2, rest);

  float w = saturate((p.z - cb.heightStart) * cb.invHeightRange);
  w *= w; // the trunk base stays put

  const float t = cb.time;
  const float ph = cb.meshPhase;
  const float gust = 0.35 + 0.65 * (0.5 + 0.5 * sin(t * cb.gustFrequency + ph))
                                 * (0.5 + 0.5 * sin(t * cb.gustFrequency * 0.37 + ph * 2.3 + 1.7));

  // whole tree bend, no per vertex phase: bark and leaves move together
  const float sway = 0.4 + sin(t * cb.speed + ph) + 0.4 * sin(t * cb.speed * 2.3 + ph * 1.7 + p.z * 0.3);
  float3 d = float3(cb.direction, 0.0) * (sway * w * cb.strength * gust);

  // branches: low spatial frequency, the same for bark and leaf vertices
  const float bp = dot(p, float3(0.7, 0.9, 0.5));
  d += float3(sin(t * cb.speed * 2.7 + bp), cos(t * cb.speed * 3.1 + bp * 1.3), 0.5 * sin(t * cb.speed * 2.1 + bp * 0.7))
       * (0.25 * cb.strength * w * gust);

  // leaf flutter: phase hashed from the rest position, so vertices duplicated along seams stay welded
  if (cb.flutter > 0.0) {
    const float vp = frac(sin(dot(p, float3(12.9898, 78.233, 37.719))) * 43758.5453) * 6.2831853;
    d += float3(sin(t * cb.speed * 9.0 + vp), cos(t * cb.speed * 8.0 + vp), sin(t * cb.speed * 11.0 + vp))
         * (cb.flutter * gust * saturate(w * 4.0));
  }

  const float3 o = affine(cb.rootToMesh0, cb.rootToMesh1, cb.rootToMesh2, p + d);
  outData[idx * 3 + 0] = o.x;
  outData[idx * 3 + 1] = o.y;
  outData[idx * 3 + 2] = o.z;
}
