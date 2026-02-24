#include "Rendering/pass/OcclusionCullingPass.h"

#include "Engine/Math/Frustum.h"
#include "Rendering/RHI/Vulkan/VulkanTypes.h"

#include <array>
#include <glm/gtc/matrix_transform.hpp>

OcclusionCullingPass::OcclusionCullingPass(OcclusionPipeline& inPipeline, FrameManager& inFrameManager, Model& inModel, Rendergraph& inRendergraph)
    : RenderPass("OcclusionPass", {"depth"}, {"depth"})
    , pipeline(&inPipeline)
    , frameManager(&inFrameManager)
    , model(&inModel)
    , rendergraph(&inRendergraph)
{
}

void OcclusionCullingPass::beginPass(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;

    if (!ctx.enableOcclusionQueries) {
        return;
    }

    // Reset queries for this frame before recording any beginQuery.
    const vk::QueryPool qp = frameManager->getOcclusionQueryPool(frameManager->getCurrentFrame());
    const uint32_t queryCount = frameManager->getOcclusionQueryCount();
    if (qp && queryCount > 0) {
        cb.resetQueryPool(qp, 0, queryCount);
    }

    vk::ImageView depthImageView = rendergraph->GetImageView("depth");

    vk::RenderingAttachmentInfoKHR depthAttachment{};
    depthAttachment.setImageView(depthImageView)
        .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore);

    vk::RenderingInfoKHR renderingInfo{};
    renderingInfo.setRenderArea(vk::Rect2D{{0, 0}, frameManager->getSwapChainExtent()})
        .setLayerCount(1)
        .setColorAttachmentCount(0)
        .setPColorAttachments(nullptr)
        .setPDepthAttachment(&depthAttachment);

    cb.beginRendering(renderingInfo);
}

void OcclusionCullingPass::render(const PassExecuteContext& ctx)
{
    if (!ctx.enableOcclusionQueries) {
        return;
    }
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

    if (!pipeline || !frameManager || !model) {
        return;
    }

    const vk::QueryPool qp = frameManager->getOcclusionQueryPool(frameManager->getCurrentFrame());
    const uint32_t queryCount = frameManager->getOcclusionQueryCount();
    if (!qp || queryCount == 0) {
        return;
    }

    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->getPipeline());

    // Bind any valid descriptor set: UBO is required for view/proj; textures unused.
    vk::DescriptorSet descriptorSet = frameManager->getDescriptorSet(frameManager->getCurrentFrame(), 0);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, frameManager->getPipelineLayout(), 0, descriptorSet, nullptr);

    const Frustum* frustum = ctx.frustum;

    auto renderNodes = [&](auto&& self, const std::vector<Node*>& nodes, const glm::mat4& parentWorld) -> void {
        for (Node* node : nodes) {
            if (!node) continue;
            if (!node->hasSubtreeBounds) {
                if (!node->children.empty()) {
                    self(self, node->children, parentWorld * node->getLocalMatrix());
                }
                continue;
            }

            const glm::mat4 worldFromNode = parentWorld * node->getLocalMatrix();
            BoundingBox worldBounds = node->subtreeBounds;
            worldBounds.Transform(worldFromNode);

            if (frustum && !frustum->Intersects(worldBounds)) {
                continue;
            }

            const uint32_t qIndex = frameManager->getOcclusionQueryIndex(node->linearIndex);
            const bool shouldQuery = (qIndex != UINT32_MAX) && (qIndex < queryCount);
            if (shouldQuery) {
                const glm::vec3 center = 0.5f * (worldBounds.min + worldBounds.max);
                const glm::vec3 halfExt = 0.5f * (worldBounds.max - worldBounds.min);
                const glm::mat4 boxModel = glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), halfExt);

                cb.beginQuery(qp, qIndex, vk::QueryControlFlags{});

                PBRPushConstants pc{};
                pc.model = boxModel;
                const std::array<PBRPushConstants, 1> pcs = {pc};
                cb.pushConstants<PBRPushConstants>(
                    frameManager->getPipelineLayout(),
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                    0,
                    pcs);

                cb.draw(36, 1, 0, 0);
                if (ctx.stats) {
                    ctx.stats->occlusionDrawCalls++;
                    ctx.stats->occlusionQueriedNodes++;
                }
                cb.endQuery(qp, qIndex);
            }

            if (!node->children.empty()) {
                self(self, node->children, worldFromNode);
            }
        }
    };

    renderNodes(renderNodes, model->getRootNodes(), ctx.modelMatrix);
}

void OcclusionCullingPass::endPass(const PassExecuteContext& ctx)
{
    if (!ctx.enableOcclusionQueries) {
        return;
    }
    ctx.commandBuffer.endRendering();
}

