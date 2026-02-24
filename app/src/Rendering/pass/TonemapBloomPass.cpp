#include "Rendering/pass/TonemapBloomPass.h"

#include "Configs/RuntimeConfig.h"

#include <array>

TonemapBloomPass::TonemapBloomPass(PostProcessPipeline& inPipeline, FrameManager& inFrameManager, Rendergraph& inRendergraph, SwapChain& inSwapChain)
    : RenderPass("TonemapBloomPass", {"scene_color", "bloom_a"}, {"swapchain"})
    , pipeline(&inPipeline)
    , frameManager(&inFrameManager)
    , rendergraph(&inRendergraph)
    , swapChain(&inSwapChain)
{
}

std::optional<vk::ImageLayout> TonemapBloomPass::getRequiredInputLayout(const std::string& resource) const
{
    if (resource == "scene_color" || resource == "bloom_a") {
        return vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    return std::nullopt;
}

std::optional<vk::ImageLayout> TonemapBloomPass::getRequiredOutputLayout(const std::string& resource) const
{
    if (resource == "swapchain") {
        return vk::ImageLayout::eColorAttachmentOptimal;
    }
    return std::nullopt;
}

void TonemapBloomPass::beginPass(const PassExecuteContext& ctx)
{
    vk::ImageView swapChainImageView = swapChain->getImageView(ctx.imageIndex);
    vk::RenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.setImageView(swapChainImageView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue{std::array<float, 4>{0.02f, 0.02f, 0.02f, 1.0f}});

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, frameManager->getSwapChainExtent()})
        .setLayerCount(1)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&colorAttachment);
    ctx.commandBuffer.beginRendering(renderingInfo);
}

void TonemapBloomPass::render(const PassExecuteContext& ctx)
{
    vk::Pipeline passPipeline = pipeline->getPipeline(PostProcessPipeline::Mode::Tonemap);
    vk::PipelineLayout layout = pipeline->getPipelineLayout();
    if (!passPipeline || !layout) return;

    vk::ImageView sceneColorView = rendergraph->GetImageView("scene_color");
    vk::ImageView bloomView = rendergraph->GetImageView("bloom_a");
    const uint32_t frameIdx = frameManager->getCurrentFrame();
    frameManager->updatePostProcessDescriptorSet(frameIdx, FrameManager::PostProcessSetSlot::Tonemap, sceneColorView, bloomView);
    vk::DescriptorSet dset = frameManager->getPostProcessDescriptorSet(frameIdx, FrameManager::PostProcessSetSlot::Tonemap);

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(frameManager->getSwapChainExtent().width);
    viewport.height = static_cast<float>(frameManager->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    ctx.commandBuffer.setViewport(0, viewport);
    ctx.commandBuffer.setScissor(0, vk::Rect2D{{0, 0}, frameManager->getSwapChainExtent()});

    PostProcessPipeline::PushConstants pc{};
    pc.params0 = glm::vec4(RuntimeConfig::bloomThreshold, RuntimeConfig::bloomSoftKnee, RuntimeConfig::bloomIntensity, RuntimeConfig::tonemapExposure);
    pc.params1 = glm::vec4(static_cast<float>(RuntimeConfig::postprocessDebugView), 0.0f, 0.0f, 0.0f);

    ctx.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, passPipeline);
    ctx.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {dset}, nullptr);
    ctx.commandBuffer.pushConstants<PostProcessPipeline::PushConstants>(layout, vk::ShaderStageFlagBits::eFragment, 0, {pc});
    ctx.commandBuffer.draw(3, 1, 0, 0);
}

void TonemapBloomPass::endPass(const PassExecuteContext& ctx)
{
    ctx.commandBuffer.endRendering();
}

