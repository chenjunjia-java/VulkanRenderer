#include "Rendering/pass/ForwardPass.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

#include "Configs/AppConfig.h"
#include "Engine/Math/BoundingBox.h"
#include "Rendering/mesh/GlobalMeshBuffer.h"
#include "Resource/model/Material.h"
#include "Resource/model/Mesh.h"
#include "Resource/model/Node.h"

ForwardPass::ForwardPass(GraphicsPipeline& pipeline, FrameManager& frameManager, Model& model, std::vector<GpuMesh>& meshes,
                         GlobalMeshBuffer& inGlobalMeshBuffer, uint32_t inMaxDraws,
                         Rendergraph& rendergraph, bool inClearDepth, bool inClearColor)
    : RenderPass("ScenePass", {"depth", "rtao_full"}, {"color_msaa", "depth", "scene_color"})
    , pipeline(&pipeline)
    , frameManager(&frameManager)
    , model(&model)
    , meshes(&meshes)
    , globalMeshBuffer(&inGlobalMeshBuffer)
    , maxDraws(inMaxDraws)
    , rendergraph(&rendergraph)
    , clearDepth(inClearDepth)
    , clearColor(inClearColor)
{
    rebuildDrawSlots();
}

std::optional<vk::ImageLayout> ForwardPass::getRequiredOutputLayout(const std::string& resource) const
{
    if (resource == "color_msaa" || resource == "scene_color") {
        return vk::ImageLayout::eColorAttachmentOptimal;
    }
    if (resource == "depth") {
        return vk::ImageLayout::eDepthStencilAttachmentOptimal;
    }
    return std::nullopt;
}

void ForwardPass::rebuildDrawSlots()
{
    transparentSlots.clear();
    if (!model || !meshes) return;

    const auto& cpuMeshes = model->getMeshes();
    const auto& materials = model->getMaterials();

    auto collect = [&](auto&& self, const std::vector<Node*>& nodes, const glm::mat4& /*parentWorld*/) -> void {
        for (Node* node : nodes) {
            if (!node || node->linearIndex == UINT32_MAX) continue;

            for (uint32_t meshIndex : node->meshIndices) {
                if (meshIndex >= meshes->size() || meshIndex >= cpuMeshes.size()) continue;

                const Mesh& cpuMesh = cpuMeshes[meshIndex];
                const int matIdxRaw = cpuMesh.materialIndex;
                const uint32_t matIndex = matIdxRaw >= 0 ? static_cast<uint32_t>(matIdxRaw) : 0u;
                const Material* mat = (!materials.empty() && matIndex < materials.size()) ? &materials[matIndex] : nullptr;
                const bool enableBlend = (mat && mat->alphaMode == AlphaMode::Blend);
                const bool doubleSided = (mat && mat->doubleSided);

                DrawSlot slot{};
                slot.nodeLinearIndex = node->linearIndex;
                slot.meshIndex = meshIndex;
                slot.matIndex = matIndex;
                slot.doubleSided = doubleSided;

                if (enableBlend) {
                    transparentSlots.push_back(slot);
                }
            }

            if (!node->children.empty()) {
                self(self, node->children, glm::mat4(1.0f));
            }
        }
    };

    collect(collect, model->getRootNodes(), glm::mat4(1.0f));

}

void ForwardPass::beginPass(const PassExecuteContext& ctx)
{
    vk::raii::CommandBuffer& cb = ctx.commandBuffer;

    vk::ImageView colorImageView = rendergraph->GetImageView("color_msaa");
    vk::ImageView depthImageView = rendergraph->GetImageView("depth");
    vk::ImageView sceneColorImageView = rendergraph->GetImageView("scene_color");

    vk::RenderingAttachmentInfoKHR colorAttachment{};
    colorAttachment.setImageView(colorImageView)
        .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setLoadOp(clearColor ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}})
        .setResolveMode(vk::ResolveModeFlagBits::eAverage)
        .setResolveImageView(sceneColorImageView)
        .setResolveImageLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::RenderingAttachmentInfoKHR depthAttachment{};
    depthAttachment.setImageView(depthImageView)
        .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
        .setLoadOp(clearDepth ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad)
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
    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto toMs = [](auto dt) -> double { return std::chrono::duration<double, std::milli>(dt).count(); };

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

    if (!model || !meshes) {
        return;
    }

    const auto& cpuMeshes = model->getMeshes();
    const auto& materials = model->getMaterials();
    const glm::mat4 viewMat = ctx.camera ? ctx.camera->getViewMatrix() : glm::mat4(1.0f);
    const auto& linearNodes = model->getLinearNodes();
    const auto& sharedNodeWorldMatrices = frameManager->getSharedNodeWorldMatrices();
    const auto& sharedBucketSpans = frameManager->getSharedOpaqueBucketSpans();
    const uint32_t sharedOpaqueDrawCount = frameManager->getSharedOpaqueDrawCount();

    transparentItems.clear();
    if (transparentItems.capacity() < transparentSlots.size()) {
        transparentItems.reserve(std::max(size_t(64), transparentSlots.size()));
    }

    // Phase 1: shared opaque collect is prepared once in Renderer; here we only gather transparent queue.
    const auto tCollect0 = now();
    for (const DrawSlot& slot : transparentSlots) {
        if (slot.nodeLinearIndex >= sharedNodeWorldMatrices.size()) continue;
        const glm::mat4 worldFromNode = sharedNodeWorldMatrices[slot.nodeLinearIndex];
        Node* node = linearNodes[slot.nodeLinearIndex];
        const Mesh& cpuMesh = cpuMeshes[slot.meshIndex];
        glm::vec3 worldCenter = glm::vec3(worldFromNode[3]);
        if (cpuMesh.hasBounds) {
            const glm::vec3 localCenter = 0.5f * (cpuMesh.bounds.min + cpuMesh.bounds.max);
            worldCenter = glm::vec3(worldFromNode * glm::vec4(localCenter, 1.0f));
        } else if (node && node->hasSubtreeBounds) {
            BoundingBox wb = node->subtreeBounds;
            wb.Transform(worldFromNode);
            worldCenter = 0.5f * (wb.min + wb.max);
        }
        glm::vec4 viewPos = viewMat * glm::vec4(worldCenter, 1.0f);
        ForwardDrawItem item{};
        item.meshIndex = slot.meshIndex;
        item.matIndex = slot.matIndex;
        item.worldFromNode = worldFromNode;
        item.enableBlend = true;
        item.doubleSided = slot.doubleSided;
        item.sortDepth = -viewPos.z;
        transparentItems.push_back(item);
    }
    const double collectMs = toMs(now() - tCollect0);

    if (ctx.stats) {
        ctx.stats->opaqueItems = sharedOpaqueDrawCount;
        ctx.stats->transparentItems = static_cast<uint64_t>(transparentItems.size());
        ctx.stats->forwardCollectMs = collectMs;
    }

    // Phase 2: sort queues. Opaque already from pre-sorted slots, no sort needed.
    const auto tSort0 = now();
    std::stable_sort(transparentItems.begin(), transparentItems.end(),
        [](const ForwardDrawItem& a, const ForwardDrawItem& b) { return a.sortDepth > b.sortDepth; });
    const double sortMs = toMs(now() - tSort0);
    if (ctx.stats) {
        ctx.stats->forwardSortMs = sortMs;
    }

    const auto tIssue0 = now();
    uint64_t pipelineBindCount = 0;
    uint64_t descriptorBindCount = 0;
    uint64_t vertexBindCount = 0;
    uint64_t indexBindCount = 0;
    uint64_t forwardDrawCallsCount = 0;

    if (!globalMeshBuffer || globalMeshBuffer->getMeshCount() == 0) {
        // Fallback: no global buffer, skip (should not happen after init).
    } else {
        const auto& meshInfos = globalMeshBuffer->getMeshInfos();
        const vk::Buffer globalVB = globalMeshBuffer->getVertexBuffer();
        const vk::Buffer globalIB = globalMeshBuffer->getIndexBuffer();
        const uint32_t frameIdx = frameManager->getCurrentFrame();
        glm::mat4* drawDataMapped = static_cast<glm::mat4*>(frameManager->getDrawDataMapped(frameIdx));

        vk::Buffer indirectBuffer = frameManager->getIndirectCommandsBuffer(frameIdx);
        if (indirectBuffer && globalVB && globalIB) {
            vk::Buffer vertexBuffers[] = {globalVB};
            vk::DeviceSize offsets[] = {0};
            cb.bindVertexBuffers(0, vertexBuffers, offsets);
            cb.bindIndexBuffer(globalIB, 0, vk::IndexType::eUint32);
            vertexBindCount = 1;
            indexBindCount = 1;

            for (const auto& span : sharedBucketSpans) {
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

                const vk::Pipeline pipelineHandle = pipeline->getPipeline(false, doubleSided);
                cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelineHandle);
                pipelineBindCount++;

                vk::DescriptorSet descriptorSet = frameManager->getDescriptorSet(frameIdx, matIndex);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, frameManager->getPipelineLayout(), 0, descriptorSet, nullptr);
                descriptorBindCount++;

                PBRPushConstants pc{};
                pc.model = glm::mat4(1.0f);
                pc.baseColorFactor = mat ? mat->baseColorFactor : glm::vec4(1.0f);
                pc.emissiveFactor = mat ? glm::vec4(mat->emissiveFactor, 0.0f) : glm::vec4(0.0f);
                const float metallic = mat ? mat->metallicFactor : 1.0f;
                const float roughness = mat ? mat->roughnessFactor : 1.0f;
                const float alphaCutoff = mat ? mat->alphaCutoff : 1.0f;
                const float normalScale = mat ? mat->normalScale : 1.0f;
                const float occlusionStrength = mat ? mat->occlusionStrength : 1.0f;
                float alphaMode = mat && mat->alphaMode == AlphaMode::Mask ? 1.0f : 0.0f;
                pc.materialParams0 = glm::vec4(metallic, roughness, alphaCutoff, normalScale);
                float reflective = (AppConfig::ENABLE_RAY_TRACED_REFLECTION && mat && mat->reflective) ? 1.0f : 0.0f;
                pc.materialParams1 = glm::vec4(occlusionStrength, alphaMode, reflective, 0.0f);

                cb.pushConstants<PBRPushConstants>(frameManager->getPipelineLayout(),
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, {pc});

                cb.drawIndexedIndirect(indirectBuffer,
                                       static_cast<vk::DeviceSize>(span.firstCommand) * sizeof(vk::DrawIndexedIndirectCommand),
                                       drawCount,
                                       sizeof(vk::DrawIndexedIndirectCommand));
                forwardDrawCallsCount += drawCount;
            }
        }
    }

    const vk::Buffer globalVB = globalMeshBuffer ? globalMeshBuffer->getVertexBuffer() : vk::Buffer{};
    const vk::Buffer globalIB = globalMeshBuffer ? globalMeshBuffer->getIndexBuffer() : vk::Buffer{};
    const std::vector<MeshDrawInfo>* meshInfosForTrans = globalMeshBuffer ? &globalMeshBuffer->getMeshInfos() : nullptr;
    const bool useGlobalForTrans = globalVB && globalIB && meshInfosForTrans && !meshInfosForTrans->empty();

    if (!transparentItems.empty() && useGlobalForTrans && vertexBindCount == 0) {
        vk::Buffer vertexBuffers[] = {globalVB};
        vk::DeviceSize offsets[] = {0};
        cb.bindVertexBuffers(0, vertexBuffers, offsets);
        cb.bindIndexBuffer(globalIB, 0, vk::IndexType::eUint32);
        vertexBindCount = 1;
        indexBindCount = 1;
    }

    for (size_t ti = 0; ti < transparentItems.size(); ++ti) {
        const ForwardDrawItem& item = transparentItems[ti];
        const Material* mat = (!materials.empty() && item.matIndex < materials.size()) ? &materials[item.matIndex] : nullptr;
        const GpuMesh& gpuMesh = (*meshes)[item.meshIndex];

        const vk::Pipeline pipelineHandle = pipeline->getPipeline(true, item.doubleSided);
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelineHandle);
        pipelineBindCount++;

        if (!useGlobalForTrans || !meshInfosForTrans || item.meshIndex >= meshInfosForTrans->size()) {
            vk::Buffer vertexBuffers[] = {gpuMesh.getVertexBuffer()};
            vk::DeviceSize offsets[] = {0};
            cb.bindVertexBuffers(0, vertexBuffers, offsets);
            cb.bindIndexBuffer(gpuMesh.getIndexBuffer(), 0, vk::IndexType::eUint32);
            vertexBindCount++;
            indexBindCount++;
        }

        const uint32_t frameIdx = frameManager->getCurrentFrame();
        uint32_t transDrawId = sharedOpaqueDrawCount + static_cast<uint32_t>(ti);
        if (transDrawId >= maxDraws) {
            break;  // Avoid out-of-bounds drawData/baseInstance in shader
        }
        glm::mat4* drawDataMapped = static_cast<glm::mat4*>(frameManager->getDrawDataMapped(frameIdx));
        if (drawDataMapped) {
            drawDataMapped[transDrawId] = item.worldFromNode;
        }

        vk::DescriptorSet descriptorSet = frameManager->getDescriptorSet(frameIdx, item.matIndex);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, frameManager->getPipelineLayout(), 0, descriptorSet, nullptr);
        descriptorBindCount++;

        PBRPushConstants pc{};
        pc.model = item.worldFromNode;
        pc.baseColorFactor = mat ? mat->baseColorFactor : glm::vec4(1.0f);
        pc.emissiveFactor = mat ? glm::vec4(mat->emissiveFactor, 0.0f) : glm::vec4(0.0f);
        const float metallic = mat ? mat->metallicFactor : 1.0f;
        const float roughness = mat ? mat->roughnessFactor : 1.0f;
        const float alphaCutoff = mat ? mat->alphaCutoff : 0.5f;
        const float normalScale = mat ? mat->normalScale : 1.0f;
        const float occlusionStrength = mat ? mat->occlusionStrength : 1.0f;
        float alphaMode = 2.0f;
        pc.materialParams0 = glm::vec4(metallic, roughness, alphaCutoff, normalScale);
        float reflective = (AppConfig::ENABLE_RAY_TRACED_REFLECTION && mat && mat->reflective) ? 1.0f : 0.0f;
        pc.materialParams1 = glm::vec4(occlusionStrength, alphaMode, reflective, 0.0f);

        cb.pushConstants<PBRPushConstants>(frameManager->getPipelineLayout(),
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, {pc});

        if (useGlobalForTrans && meshInfosForTrans && item.meshIndex < meshInfosForTrans->size()) {
            const MeshDrawInfo& info = (*meshInfosForTrans)[item.meshIndex];
            cb.drawIndexed(info.indexCount, 1, info.firstIndex, static_cast<int32_t>(info.vertexOffset), transDrawId);
        } else {
            cb.drawIndexed(gpuMesh.getIndexCount(), 1, 0, 0, transDrawId);
        }
        forwardDrawCallsCount++;
    }

    const double issueMs = toMs(now() - tIssue0);
    if (ctx.stats) {
        ctx.stats->forwardDrawCalls = forwardDrawCallsCount;
        ctx.stats->forwardPipelineBinds = pipelineBindCount;
        ctx.stats->forwardDescriptorBinds = descriptorBindCount;
        ctx.stats->forwardVertexBufferBinds = vertexBindCount;
        ctx.stats->forwardIndexBufferBinds = indexBindCount;
        ctx.stats->forwardIssueMs = issueMs;
    }
}

void ForwardPass::endPass(const PassExecuteContext& ctx)
{
    ctx.commandBuffer.endRendering();
}

