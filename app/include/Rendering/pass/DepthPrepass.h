#pragma once

#include "Rendering/core/RenderPass.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/mesh/GlobalMeshBuffer.h"
#include "Rendering/mesh/GpuMesh.h"
#include "Rendering/pipeline/DepthPrepassPipeline.h"
#include "Resource/model/Model.h"

#include <vector>

class FrameManager;

class DepthPrepass : public RenderPass {
public:
    DepthPrepass(DepthPrepassPipeline& pipeline, FrameManager& frameManager, Model& model, std::vector<GpuMesh>& meshes,
                 GlobalMeshBuffer& globalMeshBuffer, uint32_t maxDraws, Rendergraph& rendergraph, bool enableDepthResolve);

protected:
    void beginPass(const PassExecuteContext& ctx) override;
    void render(const PassExecuteContext& ctx) override;
    void endPass(const PassExecuteContext& ctx) override;

private:
    DepthPrepassPipeline* pipeline = nullptr;
    FrameManager* frameManager = nullptr;
    Model* model = nullptr;
    std::vector<GpuMesh>* meshes = nullptr;
    GlobalMeshBuffer* globalMeshBuffer = nullptr;
    uint32_t maxDraws = 1;
    Rendergraph* rendergraph = nullptr;
    bool enableDepthResolve = false;
};

