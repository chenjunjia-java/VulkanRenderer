#pragma once

#include "Rendering/core/FrameManager.h"
#include "Rendering/core/RenderPass.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/PostProcessPipeline.h"

class BloomExtractPass : public RenderPass {
public:
    BloomExtractPass(PostProcessPipeline& pipeline, FrameManager& frameManager, Rendergraph& rendergraph);

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
};

