#include "Rendering/pass/SkyboxPass.h"
#include "Rendering/core/FrameManager.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/SkyboxPipeline.h"

namespace {

const std::array<float, 108> CUBE_VERTICES = {
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
};

}  // namespace

SkyboxPass::SkyboxPass(SkyboxPipeline& pipe, FrameManager& fm, Rendergraph& rg, SwapChain& sc)
    : RenderPass("SkyboxPass", {}, {"color_msaa", "depth"})
    , pipeline(&pipe)
    , frameManager(&fm)
    , rendergraph(&rg)
    , swapChain(&sc)
{
}

void SkyboxPass::beginPass(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;

    vk::ImageView colorImageView = rendergraph->GetImageView("color_msaa");
    vk::ImageView depthImageView = rendergraph->GetImageView("depth");

    vk::RenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.setImageView(colorImageView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

    vk::RenderingAttachmentInfoKHR depthAttachment{};
    depthAttachment.setImageView(depthImageView)
        .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue{1.0f, 0});

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, frameManager->getSwapChainExtent()})
        .setLayerCount(1)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&colorAttachment)
        .setPDepthAttachment(&depthAttachment);

    cb.beginRendering(renderingInfo);
}

void SkyboxPass::render(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;

    vk::Pipeline pipe = pipeline->getPipeline();
    vk::PipelineLayout layout = pipeline->getPipelineLayout();
    if (!pipe || !layout) return;

    vk::DescriptorSet dset = frameManager->getSkyboxDescriptorSet(frameManager->getCurrentFrame());
    if (!dset) return;

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(frameManager->getSwapChainExtent().width);
    viewport.height = static_cast<float>(frameManager->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    cb.setViewport(0, viewport);

    vk::Rect2D scissor{{0, 0}, frameManager->getSwapChainExtent()};
    cb.setScissor(0, scissor);

    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipe);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, {dset}, nullptr);
    cb.bindVertexBuffers(0, {frameManager->getSkyboxVertexBuffer()}, {0ull});
    cb.draw(36, 1, 0, 0);
}

void SkyboxPass::endPass(const PassExecuteContext& ctx)
{
    ctx.commandBuffer.endRendering();
}
