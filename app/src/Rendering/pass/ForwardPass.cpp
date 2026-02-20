#include "Rendering/pass/ForwardPass.h"

#include <array>

ForwardPass::ForwardPass(GraphicsPipeline& pipeline, FrameManager& frameManager, GpuMesh& mesh,
                         Rendergraph& rendergraph, SwapChain& swapChain)
    : RenderPass("ScenePass", {}, {"color_msaa", "depth", "swapchain"})
    , pipeline(&pipeline)
    , frameManager(&frameManager)
    , mesh(&mesh)
    , rendergraph(&rendergraph)
    , swapChain(&swapChain)
{
}

void ForwardPass::beginPass(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;
    uint32_t imageIndex = ctx.imageIndex;

    vk::ImageView colorImageView = rendergraph->GetImageView("color_msaa");
    vk::ImageView depthImageView = rendergraph->GetImageView("depth");
    vk::ImageView swapChainImageView = swapChain->getImageView(imageIndex);

    vk::RenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.setImageView(colorImageView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}})
        .setResolveMode(vk::ResolveModeFlagBits::eAverage)
        .setResolveImageView(swapChainImageView)
        .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfoKHR depthAttachment{};
    depthAttachment.setImageView(depthImageView)
        .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setClearValue(vk::ClearDepthStencilValue{1.0f, 0});

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, frameManager->getSwapChainExtent()})
        .setLayerCount(1)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&colorAttachment)
        .setPDepthAttachment(&depthAttachment);

    cb.beginRendering(renderingInfo);
}

void ForwardPass::render(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;

    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->getPipeline());

    vk::Buffer vertexBuffers[] = {mesh->getVertexBuffer()};
    vk::DeviceSize offsets[] = {0};
    cb.bindVertexBuffers(0, vertexBuffers, offsets);
    cb.bindIndexBuffer(mesh->getIndexBuffer(), 0, vk::IndexType::eUint32);

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(frameManager->getSwapChainExtent().width);
    viewport.height = static_cast<float>(frameManager->getSwapChainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    cb.setViewport(0, viewport);

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = frameManager->getSwapChainExtent();
    cb.setScissor(0, scissor);

    vk::DescriptorSet descriptorSet = static_cast<vk::DescriptorSet>(frameManager->getDescriptorSets()[frameManager->getCurrentFrame()]);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, frameManager->getPipelineLayout(), 0, descriptorSet, nullptr);

    PBRMaterialPushConstants materialPC{};
    materialPC.baseColorFactor = glm::vec4(1.0f);
    materialPC.metallicFactor = 0.0f;
    materialPC.roughnessFactor = 0.7f;
    materialPC.alphaCutoff = 0.5f;
    // vulkan-hpp RAII uses ArrayProxy-based overload for push constants.
    const std::array<PBRMaterialPushConstants, 1> materialPCs = {materialPC};
    cb.pushConstants<PBRMaterialPushConstants>(
        frameManager->getPipelineLayout(), vk::ShaderStageFlagBits::eFragment, 0, materialPCs);

    cb.drawIndexed(mesh->getIndexCount(), 1, 0, 0, 0);
}

void ForwardPass::endPass(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;
    cb.endRendering();
}

