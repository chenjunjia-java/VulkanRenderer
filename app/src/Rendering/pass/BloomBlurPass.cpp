#include "Rendering/pass/BloomBlurPass.h"

#include "Configs/RuntimeConfig.h"

#include <algorithm>
#include <array>

BloomBlurPass::BloomBlurPass(const std::string& passName, const std::string& inputName, const std::string& outputName, bool horizontal,
                             PostProcessPipeline& inPipeline, FrameManager& inFrameManager, Rendergraph& inRendergraph)
    : RenderPass(passName, {inputName}, {outputName})
    , inputResource(inputName)
    , outputResource(outputName)
    , blurHorizontal(horizontal)
    , pipeline(&inPipeline)
    , frameManager(&inFrameManager)
    , rendergraph(&inRendergraph)
{
}

std::optional<vk::ImageLayout> BloomBlurPass::getRequiredInputLayout(const std::string& resource) const
{
    if (resource == inputResource) {
        return vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    return std::nullopt;
}

std::optional<vk::ImageLayout> BloomBlurPass::getRequiredOutputLayout(const std::string& resource) const
{
    if (resource == outputResource) {
        return vk::ImageLayout::eColorAttachmentOptimal;
    }
    return std::nullopt;
}

void BloomBlurPass::beginPass(const PassExecuteContext& ctx)
{
    vk::ImageView outputView = rendergraph->GetImageView(outputResource);
    const vk::Extent2D outputExtent = rendergraph->GetResourceExtent(outputResource);

    vk::RenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.setImageView(outputView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, outputExtent})
        .setLayerCount(1)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&colorAttachment);
    ctx.commandBuffer.beginRendering(renderingInfo);
}

void BloomBlurPass::render(const PassExecuteContext& ctx)
{
    vk::Pipeline passPipeline = pipeline->getPipeline(PostProcessPipeline::Mode::Blur);
    vk::PipelineLayout layout = pipeline->getPipelineLayout();
    if (!passPipeline || !layout) return;

    vk::ImageView inputView = rendergraph->GetImageView(inputResource);
    const uint32_t frameIdx = frameManager->getCurrentFrame();
    const FrameManager::PostProcessSetSlot slot =
        blurHorizontal ? FrameManager::PostProcessSetSlot::BlurH : FrameManager::PostProcessSetSlot::BlurV;
    frameManager->updatePostProcessDescriptorSet(frameIdx, slot, inputView, inputView);
    vk::DescriptorSet dset = frameManager->getPostProcessDescriptorSet(frameIdx, slot);

    const vk::Extent2D inputExtent = rendergraph->GetResourceExtent(inputResource);
    const vk::Extent2D outputExtent = rendergraph->GetResourceExtent(outputResource);

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(outputExtent.width);
    viewport.height = static_cast<float>(outputExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    ctx.commandBuffer.setViewport(0, viewport);
    ctx.commandBuffer.setScissor(0, vk::Rect2D{{0, 0}, outputExtent});

    const float invWidth = 1.0f / static_cast<float>(std::max(1u, inputExtent.width));
    const float invHeight = 1.0f / static_cast<float>(std::max(1u, inputExtent.height));

    PostProcessPipeline::PushConstants pc{};
    pc.params0 = glm::vec4(0.0f, 0.0f, 0.0f, RuntimeConfig::bloomBlurRadius);
    pc.params1 = glm::vec4(invWidth, invHeight, blurHorizontal ? 1.0f : 0.0f, blurHorizontal ? 0.0f : 1.0f);

    ctx.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, passPipeline);
    ctx.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {dset}, nullptr);
    ctx.commandBuffer.pushConstants<PostProcessPipeline::PushConstants>(layout, vk::ShaderStageFlagBits::eFragment, 0, {pc});
    ctx.commandBuffer.draw(3, 1, 0, 0);
}

void BloomBlurPass::endPass(const PassExecuteContext& ctx)
{
    ctx.commandBuffer.endRendering();
}

