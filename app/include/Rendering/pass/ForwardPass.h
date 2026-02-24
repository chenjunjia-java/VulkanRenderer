#pragma once

#include "Rendering/core/FrameManager.h"
#include "Rendering/core/RenderPass.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/GraphicsPipeline.h"
#include "Rendering/mesh/GpuMesh.h"
#include "Rendering/mesh/GlobalMeshBuffer.h"
#include "Resource/model/Model.h"

#include <cstdint>
#include <vector>
#include <glm/mat4x4.hpp>

/// Precomputed draw slot: (nodeLinearIndex, meshIndex). Material properties derived from mesh at init.
struct DrawSlot {
    uint32_t nodeLinearIndex = 0;
    uint32_t meshIndex = 0;
    uint32_t matIndex = 0;
    bool doubleSided = false;
};

/// Per-draw item for render queue. Opaque/Mask go to opaque queue; Blend goes to transparent queue.
struct ForwardDrawItem {
    uint32_t meshIndex = 0;
    uint32_t matIndex = 0;
    glm::mat4 worldFromNode{1.0f};
    bool enableBlend = false;
    bool doubleSided = false;
    float sortDepth = 0.0f;  // Used for transparent: -viewZ (larger = farther, draw first)
};

class ForwardPass : public RenderPass {
public:
    ForwardPass(GraphicsPipeline& pipeline, FrameManager& frameManager, Model& model, std::vector<GpuMesh>& meshes,
                GlobalMeshBuffer& globalMeshBuffer, uint32_t maxDraws,
                Rendergraph& rendergraph, bool clearDepth = false, bool clearColor = true);
    std::optional<vk::ImageLayout> getRequiredOutputLayout(const std::string& resource) const override;

protected:
    void beginPass(const PassExecuteContext& ctx) override;
    void render(const PassExecuteContext& ctx) override;
    void endPass(const PassExecuteContext& ctx) override;

private:
    GraphicsPipeline* pipeline = nullptr;
    FrameManager* frameManager = nullptr;
    Model* model = nullptr;
    std::vector<GpuMesh>* meshes = nullptr;
    GlobalMeshBuffer* globalMeshBuffer = nullptr;
    uint32_t maxDraws = 0;
    Rendergraph* rendergraph = nullptr;
    bool clearDepth = false;
    bool clearColor = true;

    // Reused per-frame render queues to avoid allocations.
    std::vector<ForwardDrawItem> transparentItems;

    // Cached flattened transparent slots (built once at init). Opaque is shared via FrameManager.
    std::vector<DrawSlot> transparentSlots;

    void rebuildDrawSlots();
};

