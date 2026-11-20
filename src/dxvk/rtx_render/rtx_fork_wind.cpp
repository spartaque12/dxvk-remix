// src/dxvk/rtx_render/rtx_fork_wind.cpp
//
// Fork-owned file. Foliage wind, see rtx_fork_wind.h. One compute dispatch per unique mesh in the Wind category
// (instances share their BLAS, so all instances of a tree model sway together; meshPhase de-syncs different models).

#include "rtx_fork_wind.h"

#include "rtx_fork_hooks.h"
#include "rtx_asset_replacer.h"
#include "rtx_context.h"
#include "rtx_draw_call_cache.h"
#include "rtx_imgui.h"
#include "rtx_scene_manager.h"
#include "rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "../util/util_global_time.h"

#include "rtx/pass/fork_wind/wind_args.h"

#include <rtx_shaders/wind_displace.h>

#include "../imgui/imgui.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace dxvk {

  namespace fork_hooks {
    // SceneManager friend: the draw call cache, the buffer table slots and the scene generation are private
    struct WindSceneAccess {
      static DrawCallCache& cache(SceneManager& s) { return s.m_drawCallCache; }
      static void updateBufferCache(SceneManager& s, BlasEntry* b) { s.updateBufferCache(b); }
      static void sceneChanged(SceneManager& s) { s.m_instanceManager.notifySceneChanged(); }
    };
  }

  namespace {

    class WindDisplaceShader : public ManagedShader {
      SHADER_SOURCE(WindDisplaceShader, VK_SHADER_STAGE_COMPUTE_BIT, wind_displace)

      PUSH_CONSTANTS(WindArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(WIND_BINDING_REST_INPUT)
        RW_STRUCTURED_BUFFER(WIND_BINDING_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(WindDisplaceShader);

    // rows of the affine part of a column major Matrix4
    void rows(const Matrix4& m, vec4& r0, vec4& r1, vec4& r2) {
      r0 = vec4(m[0][0], m[1][0], m[2][0], m[3][0]);
      r1 = vec4(m[0][1], m[1][1], m[2][1], m[3][1]);
      r2 = vec4(m[0][2], m[1][2], m[2][2], m[3][2]);
    }

    Rc<DxvkBuffer> createWindBuffer(DxvkDevice& device, uint32_t vertexCount) {
      DxvkBufferCreateInfo info;
      info.size = align(VkDeviceSize(vertexCount) * 12u, VkDeviceSize(256));
      info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      return device.createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXAccelerationStructure, "Fork Wind Positions");
    }

    // back to the rest positions, wind state dropped
    void release(SceneManager& scene, BlasEntry& blas, uint32_t frame) {
      RaytraceGeometry& geo = blas.modifiedGeometryData;
      if (geo.positionBuffer.defined() && geo.positionBuffer.buffer() == geo.windBuffer[0]) {
        geo.positionBuffer = geo.windRestPositionBuffer;
      }
      geo.previousPositionBuffer = RaytraceBuffer();
      geo.windBuffer[0] = nullptr;
      geo.windBuffer[1] = nullptr;
      geo.windRestPositionBuffer = RaytraceBuffer();
      geo.windFrame = kInvalidFrameIndex;
      blas.frameLastUpdated = frame; // one refit back to rest
      fork_hooks::WindSceneAccess::updateBufferCache(scene, &blas);
    }

  } // anonymous namespace

  void WindOptions::showImguiSettings() {
    RemixGui::Checkbox("Enable Foliage Wind", &enableObject());
    if (enable()) {
      ImGui::Indent();
      RemixGui::DragFloat("Strength", &strengthObject(), 0.001f, 0.f, 0.3f, "%.3f");
      RemixGui::DragFloat("Speed", &speedObject(), 0.01f, 0.f, 10.f, "%.2f");
      RemixGui::DragFloat("Gust Frequency", &gustFrequencyObject(), 0.01f, 0.f, 5.f, "%.2f");
      RemixGui::SliderFloat2("Direction", &directionObject(), -1.f, 1.f, "%.2f");
      RemixGui::DragFloat("Leaf Flutter", &flutterObject(), 0.001f, 0.f, 0.2f, "%.3f");
      RemixGui::SliderFloat("Height Start", &heightStartObject(), 0.f, 1.f, "%.2f");
      RemixGui::SliderFloat("Height Range", &heightRangeObject(), 0.01f, 1.f, "%.2f");
      RemixGui::DragFloat("Max Distance", &maxDistanceObject(), 1.f, 1.f, 1000.f, "%.0f");
      ImGui::Unindent();
    }
  }

namespace fork_hooks {

  void showWindSettings() {
    WindOptions::showImguiSettings();
  }

  void windUpdate(SceneManager& scene, RtxContext& ctx) {
    const bool enabled = WindOptions::enable();
    static bool s_owned = false; // some geometry still carries wind state
    if (!enabled && !s_owned) {
      return;
    }

    DxvkDevice& device = *ctx.getDevice();
    const uint32_t frame = device.getCurrentFrameId();
    const float time = float(uint32_t(GlobalTime::get().absoluteTimeMs()) & 0xFFFFFFu) / 1000.f;
    const Vector3 camera = scene.getCamera().getPosition();
    const float maxDistance = std::max(WindOptions::maxDistance(), 1.f);

    bool changed = false;
    bool owned = false;
    bool dispatched = false;

    for (auto& [hash, blas] : WindSceneAccess::cache(scene).getEntries()) {
      RaytraceGeometry& geo = blas.modifiedGeometryData;
      const bool hadWind = geo.windBuffer[0].ptr() != nullptr;
      const bool touched = blas.frameLastTouched == frame;

      const bool eligible = enabled &&
                            blas.input.testCategoryFlags(InstanceCategories::Wind) &&
                            blas.input.getSkinningState().numBones == 0 &&
                            !blas.input.usesVertexShader &&
                            geo.vertexCount > 0;

      if (!eligible) {
        if (hadWind) {
          release(scene, blas, frame);
          changed = true;
        }
        continue;
      }

      if (!touched) {
        owned |= hadWind; // out of view: keep the state, the history guard below handles its return
        continue;
      }

      RtInstance* nearest = nullptr;
      float nearestDistance = FLT_MAX;
      for (RtInstance* inst : blas.getLinkedInstances()) {
        const float d = length(inst->getWorldPosition() - camera);
        if (d < nearestDistance) {
          nearestDistance = d;
          nearest = inst;
        }
      }

      if (nearest == nullptr || nearestDistance > maxDistance) {
        if (hadWind) {
          release(scene, blas, frame);
          changed = true;
        }
        continue;
      }

      const bool ours = hadWind && geo.positionBuffer.defined() && geo.positionBuffer.buffer() == geo.windBuffer[0];
      if (!ours) {
        // first time, or the dynamic path re-pointed the positions at freshly interleaved data: that is the new rest pose
        if (!geo.positionBuffer.defined() ||
            geo.positionBuffer.vertexFormat() != VK_FORMAT_R32G32B32_SFLOAT ||
            (geo.positionBuffer.stride() & 3u) != 0 || (geo.positionBuffer.offsetFromSlice() & 3u) != 0 ||
            (geo.positionBuffer.offset() & 3u) != 0) {
          continue;
        }
        geo.windRestPositionBuffer = geo.positionBuffer;
        const VkDeviceSize needed = VkDeviceSize(geo.vertexCount) * 12u;
        for (auto& buffer : geo.windBuffer) {
          if (buffer.ptr() == nullptr || buffer->info().size < needed) {
            buffer = createWindBuffer(device, geo.vertexCount);
            geo.windFrame = kInvalidFrameIndex; // no usable history in a new buffer
          }
        }
      }

      std::swap(geo.windBuffer[0], geo.windBuffer[1]);

      // the tree's root space: the replacement's transform to the object it replaces (game mesh space, z up, origin at
      // the base of the tree), identity for non-replacement meshes
      Matrix4 meshToRoot;
      // height span of the whole model in root space: the union of all its meshes, so bark and leaves bend alike
      float zMin = FLT_MAX, zMax = -FLT_MAX;
      auto addBox = [&](const AxisAlignedBoundingBox& box, const Matrix4& toRoot) {
        if (!box.isValid()) {
          return;
        }
        for (uint32_t c = 0; c < 8; c++) {
          const Vector3 corner((c & 1) ? box.maxPos.x : box.minPos.x, (c & 2) ? box.maxPos.y : box.minPos.y, (c & 4) ? box.maxPos.z : box.minPos.z);
          const float z = (toRoot * Vector4(corner, 1.f)).z;
          zMin = std::min(zMin, z);
          zMax = std::max(zMax, z);
        }
      };
      {
        auto& owner = nearest->getPrimInstanceOwner();
        const ReplacementInstance* ri = owner.getReplacementInstance();
        const size_t index = owner.getReplacementIndex();
        if (ri != nullptr && ri->activeReplacements != nullptr && index < ri->activeReplacements->replacements.size()) {
          meshToRoot = ri->activeReplacements->replacements[index].replacementToObject;
          for (const AssetReplacement& r : ri->activeReplacements->replacements) {
            if (r.type == AssetReplacement::eMesh && r.geometry) {
              addBox(r.geometry->data.boundingBox, r.replacementToObject);
            }
          }
        }
      }
      if (zMin > zMax) {
        addBox(blas.input.getGeometryData().boundingBox, meshToRoot); // game mesh, or bounds not known yet
      }
      if (zMin > zMax) {
        continue;
      }
      const float height = std::max(zMax - zMin, 0.05f);
      const Matrix4 rootToMesh = inverse(meshToRoot);

      const RaytraceBuffer& rest = geo.windRestPositionBuffer;
      const float fade = std::clamp((maxDistance - nearestDistance) / (0.25f * maxDistance), 0.f, 1.f); // no pop at the cutoff

      WindArgs args {};
      rows(meshToRoot, args.meshToRoot0, args.meshToRoot1, args.meshToRoot2);
      rows(rootToMesh, args.rootToMesh0, args.rootToMesh1, args.rootToMesh2);
      const Vector2 dir = WindOptions::direction();
      const float dirLength = std::sqrt(dir.x * dir.x + dir.y * dir.y);
      args.direction = dirLength > 1e-4f ? vec2(dir.x / dirLength, dir.y / dirLength) : vec2(1.f, 0.f);
      args.time = time;
      args.strength = WindOptions::strength() * height * fade; // a bush bends as much as a tree, relative to its size
      args.speed = WindOptions::speed();
      args.gustFrequency = WindOptions::gustFrequency();
      args.flutter = nearest->isOpaque() ? 0.f : WindOptions::flutter() * fade;
      args.meshPhase = float(hash & 0xFFFFu) / 65535.f * 6.2831853f;
      args.heightStart = zMin + WindOptions::heightStart() * height;
      args.invHeightRange = 1.f / (std::max(WindOptions::heightRange(), 0.01f) * height);
      args.srcOffset = uint32_t((rest.offset() + rest.offsetFromSlice()) / 4u);
      args.srcStride = rest.stride() / 4u;
      args.numVertices = geo.vertexCount;

      if (!dispatched) {
        ScopedGpuProfileZone(&ctx, "Fork Foliage Wind");
        ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, WindDisplaceShader::getShader());
        ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
        dispatched = true;
      }
      ctx.bindResourceBuffer(WIND_BINDING_REST_INPUT, DxvkBufferSlice(rest.buffer()));
      ctx.bindResourceBuffer(WIND_BINDING_OUTPUT, DxvkBufferSlice(geo.windBuffer[0]));
      ctx.pushConstants(0, sizeof(WindArgs), &args);
      ctx.dispatch((geo.vertexCount + 127u) / 128u, 1, 1);

      // this frame's displaced positions; last frame's as the previous positions for motion vectors
      const bool history = geo.windFrame != kInvalidFrameIndex && geo.windFrame + 1u == frame;
      geo.positionBuffer = RaytraceBuffer(DxvkBufferSlice(geo.windBuffer[0]), 0, 12, VK_FORMAT_R32G32B32_SFLOAT);
      geo.previousPositionBuffer = history ? RaytraceBuffer(DxvkBufferSlice(geo.windBuffer[1]), 0, 12, VK_FORMAT_R32G32B32_SFLOAT) : RaytraceBuffer();
      geo.windFrame = frame;
      blas.frameLastUpdated = frame; // dynamic BLAS refit, geometry info refilled
      WindSceneAccess::updateBufferCache(scene, &blas);

      if (history) {
        for (RtInstance* inst : blas.getLinkedInstances()) {
          inst->surface.isStatic = false;
        }
      }

      owned = true;
      changed = true;
    }

    if (dispatched) {
      // the BLAS builds and the ray tracing passes read the displaced positions
      ctx.emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }

    if (changed) {
      WindSceneAccess::sceneChanged(scene);
    }
    s_owned = owned;
  }

} // namespace fork_hooks

} // namespace dxvk
