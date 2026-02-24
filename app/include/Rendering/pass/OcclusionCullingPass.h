#pragma once

#include "Rendering/core/FrameManager.h"
#include "Rendering/core/RenderPass.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/OcclusionPipeline.h"
#include "Resource/model/Model.h"

class OcclusionCullingPass : public RenderPass {
public:
    OcclusionCullingPass(OcclusionPipeline& pipeline, FrameManager& frameManager, Model& model, Rendergraph& rendergraph);

protected:
    void beginPass(const PassExecuteContext& ctx) override;
    void render(const PassExecuteContext& ctx) override;
    void endPass(const PassExecuteContext& ctx) override;

private:
    OcclusionPipeline* pipeline = nullptr;
    FrameManager* frameManager = nullptr;
    Model* model = nullptr;
    Rendergraph* rendergraph = nullptr;
};

