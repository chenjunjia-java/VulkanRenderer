#include "Rendering/pass/BloomExtractPass.h"

#include "Configs/RuntimeConfig.h"

#include <array>

BloomExtractPass::BloomExtractPass(PostProcessPipeline& inPipeline, FrameManager& inFrameManager, Rendergraph& inRendergraph)
    : RenderPass("BloomExtractPass", {"scene_color"}, {"bloom_a"})
    , pipeline(&inPipeline)
    , frameManager(&inFrameManager)
    , rendergraph(&inRendergraph)
{
}

std::optional<vk::ImageLayout> BloomExtractPass::getRequiredInputLayout(const std::string& resource) const
{
    if (resource == "scene_color") {
        return vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    return std::nullopt;
}

std::optional<vk::ImageLayout> BloomExtractPass::getRequiredOutputLayout(const std::string& resource) const
{
    if (resource == "bloom_a") {
        return vk::ImageLayout::eColorAttachmentOptimal;
    }
    return std::nullopt;
}

void BloomExtractPass::beginPass(const PassExecuteContext& ctx)
{
    vk::ImageView outputView = rendergraph->GetImageView("bloom_a");
    const vk::Extent2D bloomExtent = rendergraph->GetResourceExtent("bloom_a");

    vk::RenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.setImageView(outputView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, bloomExtent})
        .setLayerCount(1)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&colorAttachment);
    ctx.commandBuffer.beginRendering(renderingInfo);
}

void BloomExtractPass::render(const PassExecuteContext& ctx)
{
    vk::Pipeline passPipeline = pipeline->getPipeline(PostProcessPipeline::Mode::Extract);
    vk::PipelineLayout layout = pipeline->getPipelineLayout();
    if (!passPipeline || !layout) return;

    vk::ImageView sceneColorView = rendergraph->GetImageView("scene_color");
    const uint32_t frameIdx = frameManager->getCurrentFrame();
    frameManager->updatePostProcessDescriptorSet(
        frameIdx, FrameManager::PostProcessSetSlot::Extract, sceneColorView, sceneColorView);

    vk::DescriptorSet dset = frameManager->getPostProcessDescriptorSet(frameIdx, FrameManager::PostProcessSetSlot::Extract);

    const vk::Extent2D bloomExtent = rendergraph->GetResourceExtent("bloom_a");

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(bloomExtent.width);
    viewport.height = static_cast<float>(bloomExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    ctx.commandBuffer.setViewport(0, viewport);
    ctx.commandBuffer.setScissor(0, vk::Rect2D{{0, 0}, bloomExtent});

    PostProcessPipeline::PushConstants pc{};
    pc.params0 = glm::vec4(RuntimeConfig::bloomThreshold, RuntimeConfig::bloomSoftKnee, RuntimeConfig::bloomIntensity, 0.0f);

    ctx.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, passPipeline);
    ctx.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {dset}, nullptr);
    ctx.commandBuffer.pushConstants<PostProcessPipeline::PushConstants>(layout, vk::ShaderStageFlagBits::eFragment, 0, {pc});
    ctx.commandBuffer.draw(3, 1, 0, 0);
}

void BloomExtractPass::endPass(const PassExecuteContext& ctx)
{
    ctx.commandBuffer.endRendering();
}

