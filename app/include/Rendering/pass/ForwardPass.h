#pragma once

#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/core/FrameManager.h"
#include "Rendering/core/RenderPass.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/GraphicsPipeline.h"
#include "Rendering/mesh/GpuMesh.h"

class ForwardPass : public RenderPass {
public:
    ForwardPass(GraphicsPipeline& pipeline, FrameManager& frameManager, GpuMesh& mesh,
                Rendergraph& rendergraph, SwapChain& swapChain);

protected:
    void beginPass(const PassExecuteContext& ctx) override;
    void render(const PassExecuteContext& ctx) override;
    void endPass(const PassExecuteContext& ctx) override;

private:
    GraphicsPipeline* pipeline = nullptr;
    FrameManager* frameManager = nullptr;
    GpuMesh* mesh = nullptr;
    Rendergraph* rendergraph = nullptr;
    SwapChain* swapChain = nullptr;
};

