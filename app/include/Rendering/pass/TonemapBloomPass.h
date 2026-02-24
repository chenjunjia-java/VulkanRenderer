#pragma once

#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/core/FrameManager.h"
#include "Rendering/core/RenderPass.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/PostProcessPipeline.h"

class TonemapBloomPass : public RenderPass {
public:
    TonemapBloomPass(PostProcessPipeline& pipeline, FrameManager& frameManager, Rendergraph& rendergraph, SwapChain& swapChain);

    std::optional<vk::ImageLayout> getRequiredInputLayout(const std::string& resource) const override;
    std::optional<vk::ImageLayout> getRequiredOutputLayout(const std::string& resource) const override;

protected:
    void beginPass(const PassExecuteContext& ctx) override;
    void render(const PassExecuteContext& ctx) override;
    void endPass(const PassExecuteContext& ctx) override;

private:
    PostProcessPipeline* pipeline = nullptr;
    FrameManager* frameManager = nullptr;
    Rendergraph* rendergraph = nullptr;
    SwapChain* swapChain = nullptr;
};

