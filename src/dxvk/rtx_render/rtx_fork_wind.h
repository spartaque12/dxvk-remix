#pragma once

// rtx_fork_wind.h - fork-owned. Foliage wind (rtx.wind.*): geometry in the Wind category (rtx.windTextures,
// remix_category:wind, REMIXAPI_INSTANCE_CATEGORY_BIT_WIND) is displaced on the GPU every frame. Once per unique mesh
// (BlasEntry): the displaced positions go to two buffers owned by the geometry (this frame / last frame), so the
// existing previous-position path gives motion vectors and the dynamic BLAS refit moves shadows and reflections.
// Shader in shaders/rtx/pass/fork_wind/.

#include "rtx_option.h"
#include "../util/util_vector.h"

namespace dxvk {

  class RtxContext;
  class SceneManager;

  class WindOptions {
  public:
    RTX_OPTION("rtx.wind", bool, enable, false,
               "Sways geometry in the Wind category (rtx.windTextures, remix_category:wind).");
    RTX_OPTION("rtx.wind", float, strength, 0.15f, "Sway amplitude in world units at full height.");
    RTX_OPTION("rtx.wind", float, speed, 1.2f, "Sway frequency (radians per second).");
    RTX_OPTION("rtx.wind", float, gustFrequency, 0.25f, "Frequency of the slow gust envelope (radians per second).");
    RTX_OPTION("rtx.wind", Vector2, direction, Vector2(1.f, 0.f),
               "Wind direction, xy in the space of the tree. shortcut: object space, trees rotated differently lean differently; per instance world direction needs a BLAS per instance.");
    RTX_OPTION("rtx.wind", float, flutter, 0.015f, "Leaf flutter amplitude in world units (alpha tested surfaces only).");
    RTX_OPTION("rtx.wind", float, heightStart, 1.0f, "Height above the base of a tree below which nothing moves.");
    RTX_OPTION("rtx.wind", float, heightRange, 8.0f, "Height span over which the sway grows to full strength.");
    RTX_OPTION("rtx.wind", float, maxDistance, 80.f, "Meshes whose nearest instance is farther from the camera stay static.");

    static void showImguiSettings();
  };

  namespace fork_hooks {
    // Displaces the Wind category geometry of this frame. SceneManager::prepareSceneData, after the instances of the
    // frame are known and before the buffer table is built. Implementation in rtx_fork_wind.cpp.
    void windUpdate(SceneManager& scene, RtxContext& ctx);

    // Body of the "Foliage Wind" header under Game Setup -> Parameter Tuning.
    void showWindSettings();
  }

} // namespace dxvk
