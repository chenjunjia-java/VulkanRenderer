#pragma once

#include "Rendering/core/RenderPass.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"

#include <vulkan/vulkan_raii.hpp>

class SkyboxPipeline;
class FrameManager;
class Rendergraph;

class SkyboxPass : public RenderPass {
public:
    SkyboxPass(SkyboxPipeline& pipeline, FrameManager& frameManager, Rendergraph& rendergraph, SwapChain& swapChain);

protected:
    void beginPass(const PassExecuteContext& ctx) override;
    void render(const PassExecuteContext& ctx) override;
    void endPass(const PassExecuteContext& ctx) override;

private:
    SkyboxPipeline* pipeline = nullptr;
    FrameManager* frameManager = nullptr;
    Rendergraph* rendergraph = nullptr;
    SwapChain* swapChain = nullptr;
};
