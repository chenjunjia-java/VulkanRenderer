#include "Rendering/pass/DepthPrepass.h"
#include "Rendering/core/FrameManager.h"

#include <array>

DepthPrepass::DepthPrepass(DepthPrepassPipeline& inPipeline, FrameManager& inFrameManager, Model& inModel, std::vector<GpuMesh>& inMeshes,
                           GlobalMeshBuffer& inGlobalMeshBuffer, uint32_t inMaxDraws,
                           Rendergraph& inRendergraph, bool inEnableDepthResolve)
    : RenderPass("DepthPrepass", {}, {"depth"})
    , pipeline(&inPipeline)
    , frameManager(&inFrameManager)
    , model(&inModel)
    , meshes(&inMeshes)
    , globalMeshBuffer(&inGlobalMeshBuffer)
    , maxDraws(std::max(1u, inMaxDraws))
    , rendergraph(&inRendergraph)
    , enableDepthResolve(inEnableDepthResolve)
{
}

void DepthPrepass::beginPass(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;
    vk::ImageView depthImageView = rendergraph->GetImageView("depth");
    vk::ImageView depthResolveView = frameManager->getDepthResolveImageView();
    vk::ImageView normalMsaaView = frameManager->getNormalPrepassImageView();
    vk::ImageView normalResolveView = frameManager->getNormalResolveImageView();
    vk::ImageView linearDepthMsaaView = frameManager->getLinearDepthPrepassImageView();
    vk::ImageView linearDepthResolveView = frameManager->getLinearDepthResolveImageView();

    vk::RenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.setImageView(normalMsaaView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue(std::array<float, 4>{0.5f, 0.5f, 1.0f, 1.0f}));
    if (enableDepthResolve && normalResolveView) {
        colorAttachment.setResolveMode(vk::ResolveModeFlagBits::eAverage)
            .setResolveImageView(normalResolveView)
            .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
    }

    vk::RenderingAttachmentInfoKHR linearDepthAttachment{};
    linearDepthAttachment.setImageView(linearDepthMsaaView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}));
    if (enableDepthResolve && linearDepthResolveView) {
        linearDepthAttachment.setResolveMode(vk::ResolveModeFlagBits::eAverage)
            .setResolveImageView(linearDepthResolveView)
            .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
    }

    vk::RenderingAttachmentInfoKHR depthAttachment{};
    depthAttachment.setImageView(depthImageView)
        .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearDepthStencilValue{1.0f, 0});
    if (enableDepthResolve && depthResolveView) {
        depthAttachment.setResolveMode(vk::ResolveModeFlagBits::eSampleZero)
            .setResolveImageView(depthResolveView)
            .setResolveImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }

    std::array<vk::RenderingAttachmentInfoKHR, 2> colorAttachments = {colorAttachment, linearDepthAttachment};

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, frameManager->getSwapChainExtent()})
        .setLayerCount(1)
        .setColorAttachmentCount(static_cast<uint32_t>(colorAttachments.size()))
        .setPColorAttachments(colorAttachments.data())
        .setPDepthAttachment(&depthAttachment);

    cb.beginRendering(renderingInfo);
}

void DepthPrepass::render(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;

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

    if (!model || !meshes || !pipeline || !globalMeshBuffer || globalMeshBuffer->getMeshCount() == 0) {
        return;
    }

    const auto& materials = model->getMaterials();
    const uint32_t frameIdx = frameManager->getCurrentFrame();
    const auto& bucketSpans = frameManager->getSharedOpaqueBucketSpans();

    const vk::Buffer indirectBuffer = frameManager->getIndirectCommandsBuffer(frameIdx);
    const vk::Buffer globalVB = globalMeshBuffer->getVertexBuffer();
    const vk::Buffer globalIB = globalMeshBuffer->getIndexBuffer();
    if (!indirectBuffer || !globalVB || !globalIB) {
        return;
    }

    vk::Buffer vertexBuffers[] = {globalVB};
    vk::DeviceSize offsets[] = {0};
    cb.bindVertexBuffers(0, vertexBuffers, offsets);
    cb.bindIndexBuffer(globalIB, 0, vk::IndexType::eUint32);

    for (const auto& span : bucketSpans) {
        if (span.firstCommand >= maxDraws) {
            break;
        }
        const uint32_t drawCount = std::min(span.drawCount, maxDraws - span.firstCommand);
        if (drawCount == 0) {
            continue;
        }
        const bool doubleSided = span.doubleSided;
        const uint32_t matIndex = span.matIndex;
        const Material* mat = (!materials.empty() && matIndex < materials.size()) ? &materials[matIndex] : nullptr;

        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->getPipeline(doubleSided));
        vk::DescriptorSet descriptorSet = frameManager->getDescriptorSet(frameIdx, matIndex);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, frameManager->getPipelineLayout(), 0, descriptorSet, nullptr);

        PBRPushConstants pc{};
        pc.model = glm::mat4(1.0f);
        pc.baseColorFactor = mat ? mat->baseColorFactor : glm::vec4(1.0f);
        const float alphaCutoff = mat ? mat->alphaCutoff : 0.5f;
        const float alphaMode = (mat && mat->alphaMode == AlphaMode::Mask) ? 1.0f : 0.0f;
        pc.materialParams0 = glm::vec4(0.0f, 0.0f, alphaCutoff, 0.0f);
        pc.materialParams1 = glm::vec4(0.0f, alphaMode, 0.0f, 0.0f);
        cb.pushConstants<PBRPushConstants>(
            frameManager->getPipelineLayout(),
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0,
            {pc});

        cb.drawIndexedIndirect(
            indirectBuffer,
            static_cast<vk::DeviceSize>(span.firstCommand) * sizeof(vk::DrawIndexedIndirectCommand),
            drawCount,
            sizeof(vk::DrawIndexedIndirectCommand));
        if (ctx.stats) {
            ctx.stats->depthDrawCalls += drawCount;
        }
    }
}

void DepthPrepass::endPass(const PassExecuteContext& ctx)
{
    ctx.commandBuffer.endRendering();
}

