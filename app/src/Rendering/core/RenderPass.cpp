#include "Rendering/core/RenderPass.h"

RenderPass::RenderPass(std::string passName, std::vector<std::string> inputResources,
                      std::vector<std::string> outputResources)
    : name(std::move(passName))
    , inputs(std::move(inputResources))
    , outputs(std::move(outputResources))
{
}

void RenderPass::execute(const PassExecuteContext& ctx)
{
    beginPass(ctx);
    render(ctx);
    endPass(ctx);
}

