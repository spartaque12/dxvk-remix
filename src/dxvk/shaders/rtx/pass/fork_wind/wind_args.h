// src/dxvk/shaders/rtx/pass/fork_wind/wind_args.h
//
// Fork-owned. Shared C++ / Slang layout of the foliage wind pass (rtx_fork_wind.cpp, wind_displace.comp.slang).
#pragma once

#include "rtx/utility/shader_types.h"

#define WIND_BINDING_REST_INPUT    0
#define WIND_BINDING_OUTPUT        1

// 160 bytes, push constants
struct WindArgs {
  // rows of the affine transforms between mesh space and the root space of the tree (game object space: z up, origin
  // at the base of the tree)
  vec4 meshToRoot0;
  vec4 meshToRoot1;
  vec4 meshToRoot2;
  vec4 rootToMesh0;
  vec4 rootToMesh1;
  vec4 rootToMesh2;

  vec2 direction;     // root space xy
  float time;
  float strength;

  float speed;
  float gustFrequency;
  float flutter;
  float meshPhase;

  float heightStart;
  float invHeightRange;
  uint srcOffset;     // in floats
  uint srcStride;     // in floats

  uint numVertices;
  uint pad0;
  uint pad1;
  uint pad2;
};
