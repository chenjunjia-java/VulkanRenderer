#include "Rendering/core/FrameManager.h"

#include <array>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <map>

#include <glm/gtc/matrix_transform.hpp>
#include "Resource/model/Mesh.h"
#include "Resource/model/Material.h"
#include "Resource/model/Node.h"
#include "Rendering/mesh/GlobalMeshBuffer.h"
#include "Configs/RuntimeConfig.h"

void FrameManager::init(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
                        Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator,
                        Model& model, RayTracingContext& rayTracingContext, uint32_t inMaxDraws)
{
    (void)rendergraph;
    devicePtr = &context.getDevice();
    maxDraws = std::max(1u, inMaxDraws);
    pipelineLayoutHandle = pipeline.getPipelineLayout();
    swapChainExtent = swapChain.getExtent();
    materialCount = std::max(1u, static_cast<uint32_t>(model.getMaterials().size()));
    lastViewProj = glm::mat4(1.0f);
    uniformFrameIndex = 0;

    createCommandBuffers(context.getDevice(), resourceCreator, swapChain);
    createSyncObjects(context.getDevice(), swapChain);
    createUniformBuffers(context.getDevice(), resourceCreator);
    createDrawDataBuffers(context.getDevice(), resourceCreator);
    createIndirectCommandBuffers(context.getDevice(), resourceCreator);
    createDefaultPbrTextures(resourceCreator);
    createDefaultIblTextures(resourceCreator);
    createDepthResolveTexture(resourceCreator);
    createNormalTextures(resourceCreator, context.getMsaaSamples());
    createLinearDepthTextures(resourceCreator, context.getMsaaSamples());
    createRtaoComputeTextures(resourceCreator);
    createReflectionBuffers(resourceCreator, model);
    createDescriptorPool(context.getDevice());
    createDescriptorSets(context.getDevice(), resourceCreator, pipeline, model, rayTracingContext);
    buildSharedDrawSlots(model);
}

void FrameManager::recreate(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
                            Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator, Model& model,
                            RayTracingContext& rayTracingContext, uint32_t inMaxDraws)
{
    (void)rendergraph;
    devicePtr = &context.getDevice();
    maxDraws = std::max(1u, inMaxDraws);
    pipelineLayoutHandle = pipeline.getPipelineLayout();
    swapChainExtent = swapChain.getExtent();
    lastViewProj = glm::mat4(1.0f);
    uniformFrameIndex = 0;
    cleanupSwapChainResources(context.getDevice());
    materialCount = std::max(1u, static_cast<uint32_t>(model.getMaterials().size()));
    createCommandBuffers(context.getDevice(), resourceCreator, swapChain);
    createSyncObjects(context.getDevice(), swapChain);
    createUniformBuffers(context.getDevice(), resourceCreator);
    createDrawDataBuffers(context.getDevice(), resourceCreator);
    createIndirectCommandBuffers(context.getDevice(), resourceCreator);
    createDefaultPbrTextures(resourceCreator);
    createDefaultIblTextures(resourceCreator);
    createDepthResolveTexture(resourceCreator);
    createNormalTextures(resourceCreator, context.getMsaaSamples());
    createLinearDepthTextures(resourceCreator, context.getMsaaSamples());
    createRtaoComputeTextures(resourceCreator);
    createReflectionBuffers(resourceCreator, model);
    createDescriptorPool(context.getDevice());
    createDescriptorSets(context.getDevice(), resourceCreator, pipeline, model, rayTracingContext);
    buildSharedDrawSlots(model);
    if (skyboxDescriptorSets) {
        updateSkyboxDescriptorBuffers(context.getDevice());
    }
}

void FrameManager::buildSharedDrawSlots(const Model& model)
{
    sharedOpaqueSlots.clear();
    sharedOpaqueBucketSpans.clear();
    sharedOpaqueDrawCount = 0;

    const auto& cpuMeshes = model.getMeshes();
    const auto& materials = model.getMaterials();
    auto collect = [&](auto&& self, const std::vector<Node*>& nodes) -> void {
        for (Node* node : nodes) {
            if (!node || node->linearIndex == UINT32_MAX) continue;
            for (uint32_t meshIndex : node->meshIndices) {
                if (meshIndex >= cpuMeshes.size()) continue;
                const Mesh& cpuMesh = cpuMeshes[meshIndex];
                const int matIdxRaw = cpuMesh.materialIndex;
                const uint32_t matIndex = matIdxRaw >= 0 ? static_cast<uint32_t>(matIdxRaw) : 0u;
                const Material* mat = (!materials.empty() && matIndex < materials.size()) ? &materials[matIndex] : nullptr;
                if (mat && mat->alphaMode == AlphaMode::Blend) continue;
                SharedOpaqueDrawSlot slot{};
                slot.nodeLinearIndex = node->linearIndex;
                slot.meshIndex = meshIndex;
                slot.matIndex = matIndex;
                slot.doubleSided = (mat && mat->doubleSided);
                sharedOpaqueSlots.push_back(slot);
            }
            if (!node->children.empty()) self(self, node->children);
        }
    };
    collect(collect, model.getRootNodes());
    std::stable_sort(sharedOpaqueSlots.begin(), sharedOpaqueSlots.end(),
        [](const SharedOpaqueDrawSlot& a, const SharedOpaqueDrawSlot& b) {
            if (a.doubleSided != b.doubleSided) return a.doubleSided < b.doubleSided;
            if (a.matIndex != b.matIndex) return a.matIndex < b.matIndex;
            return a.meshIndex < b.meshIndex;
        });
}

void FrameManager::prepareSharedOpaqueIndirect(const Model& model, const GlobalMeshBuffer& globalMeshBuffer, const glm::mat4& modelMatrix)
{
    const auto& linearNodes = model.getLinearNodes();
    sharedNodeWorldMatrices.resize(linearNodes.size());
    for (Node* n : linearNodes) {
        if (!n || n->linearIndex >= linearNodes.size()) continue;
        const glm::mat4 parentWorld = n->parent ? sharedNodeWorldMatrices[n->parent->linearIndex] : modelMatrix;
        sharedNodeWorldMatrices[n->linearIndex] = parentWorld * n->getLocalMatrix();
    }

    sharedOpaqueBucketSpans.clear();
    sharedOpaqueDrawCount = 0;
    const uint32_t frameIdx = getCurrentFrame();
    auto* drawDataMapped = static_cast<glm::mat4*>(getDrawDataMapped(frameIdx));
    auto* indirectMapped = static_cast<vk::DrawIndexedIndirectCommand*>(getIndirectCommandsMapped(frameIdx));
    const auto& meshInfos = globalMeshBuffer.getMeshInfos();

    using BucketKey = std::pair<bool, uint32_t>;
    std::map<BucketKey, std::vector<vk::DrawIndexedIndirectCommand>> buckets;
    uint32_t drawId = 0;
    for (const auto& slot : sharedOpaqueSlots) {
        if (drawId >= maxDraws) break;
        if (slot.nodeLinearIndex >= sharedNodeWorldMatrices.size() || slot.meshIndex >= meshInfos.size()) continue;
        if (drawDataMapped) drawDataMapped[drawId] = sharedNodeWorldMatrices[slot.nodeLinearIndex];
        const MeshDrawInfo& info = meshInfos[slot.meshIndex];
        vk::DrawIndexedIndirectCommand cmd{};
        cmd.indexCount = info.indexCount;
        cmd.instanceCount = 1;
        cmd.firstIndex = info.firstIndex;
        cmd.vertexOffset = static_cast<int32_t>(info.vertexOffset);
        cmd.firstInstance = drawId;
        buckets[{slot.doubleSided, slot.matIndex}].push_back(cmd);
        ++drawId;
    }

    size_t indirectOffset = 0;
    for (const auto& [key, commands] : buckets) {
        if (commands.empty()) continue;
        const size_t canCopy = (indirectOffset < maxDraws)
            ? std::min(commands.size(), static_cast<size_t>(maxDraws) - indirectOffset)
            : 0u;
        if (indirectMapped && canCopy > 0) {
            std::memcpy(indirectMapped + indirectOffset, commands.data(),
                        canCopy * sizeof(vk::DrawIndexedIndirectCommand));
        }
        SharedOpaqueBucketSpan span{};
        span.doubleSided = key.first;
        span.matIndex = key.second;
        span.firstCommand = static_cast<uint32_t>(indirectOffset);
        const uint32_t available = (span.firstCommand < maxDraws) ? (maxDraws - span.firstCommand) : 0u;
        span.drawCount = static_cast<uint32_t>(std::min(commands.size(), static_cast<size_t>(available)));
        if (span.drawCount > 0) {
            sharedOpaqueBucketSpans.push_back(span);
        }
        indirectOffset += commands.size();
    }
    sharedOpaqueDrawCount = drawId;
}

vk::DescriptorSet FrameManager::getDescriptorSet(uint32_t frameIndex, uint32_t materialIndex) const
{
    if (!descriptorSets) {
        return vk::DescriptorSet{};
    }
    if (materialCount == 0) {
        return vk::DescriptorSet{};
    }
    frameIndex %= AppConfig::MAX_FRAMES_IN_FLIGHT;
    materialIndex = std::min(materialIndex, materialCount - 1);
    const uint32_t flatIndex = frameIndex * materialCount + materialIndex;
    return static_cast<vk::DescriptorSet>((*descriptorSets)[flatIndex]);
}

vk::Buffer FrameManager::getUniformBuffer(uint32_t frameIndex) const
{
    if (uniformBuffers.empty()) {
        return vk::Buffer{};
    }
    frameIndex %= static_cast<uint32_t>(uniformBuffers.size());
    return static_cast<vk::Buffer>(*uniformBuffers[frameIndex]);
}

namespace {
const std::array<float, 108> SKYBOX_CUBE_VERTICES = {
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

void FrameManager::createSkyboxVertexBuffer(VulkanResourceCreator& resourceCreator)
{
    const vk::DeviceSize vbSize = sizeof(SKYBOX_CUBE_VERTICES);
    BufferAllocation staging = resourceCreator.createBuffer(
        vbSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* mapped = staging.memory.mapMemory(0, vbSize);
    std::memcpy(mapped, SKYBOX_CUBE_VERTICES.data(), vbSize);
    staging.memory.unmapMemory();

    BufferAllocation gpu = resourceCreator.createBuffer(
        vbSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*staging.buffer, *gpu.buffer, vbSize);
    skyboxVertexBuffer = std::move(gpu.buffer);
    skyboxVertexBufferMemory = std::move(gpu.memory);
}

void FrameManager::createSkyboxResources(VulkanResourceCreator& resourceCreator, vk::DescriptorSetLayout skyboxLayout,
                                        vk::ImageView envCubeView, vk::Sampler envCubeSampler)
{
    vk::raii::Device& device = resourceCreator.getDevice();

    createSkyboxVertexBuffer(resourceCreator);

    vk::DescriptorPoolSize poolSizes[2];
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = AppConfig::MAX_FRAMES_IN_FLIGHT;
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = AppConfig::MAX_FRAMES_IN_FLIGHT;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = AppConfig::MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    skyboxDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(AppConfig::MAX_FRAMES_IN_FLIGHT, skyboxLayout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *skyboxDescriptorPool;
    allocInfo.descriptorSetCount = AppConfig::MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    skyboxDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = envCubeView;
    imageInfo.sampler = envCubeSampler;

    for (uint32_t i = 0; i < AppConfig::MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = *uniformBuffers[i];
        bufferInfo.offset = 64;   // skip model, PBR UBO has view at 64, proj at 128
        bufferInfo.range = 128;

        std::array<vk::WriteDescriptorSet, 2> writes{};
        writes[0].dstSet = (*skyboxDescriptorSets)[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufferInfo;
        writes[1].dstSet = (*skyboxDescriptorSets)[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &imageInfo;
        device.updateDescriptorSets(writes, nullptr);
    }
}

void FrameManager::createPostProcessResources(vk::raii::Device& device, vk::DescriptorSetLayout postLayout)
{
    postDescriptorSets.reset();
    postDescriptorPool.reset();
    postSampler.reset();

    constexpr uint32_t slotCount = static_cast<uint32_t>(PostProcessSetSlot::Count);

    vk::DescriptorPoolSize poolSize{};
    poolSize.type = vk::DescriptorType::eCombinedImageSampler;
    poolSize.descriptorCount = AppConfig::MAX_FRAMES_IN_FLIGHT * slotCount * 2u;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = AppConfig::MAX_FRAMES_IN_FLIGHT * slotCount;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    postDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(AppConfig::MAX_FRAMES_IN_FLIGHT * slotCount, postLayout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *postDescriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();
    postDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    postSampler = vk::raii::Sampler(device, samplerInfo);
}

void FrameManager::setIblResources(vk::raii::Device& device, vk::ImageView irradianceView, vk::ImageView prefilterView,
                                   vk::ImageView brdfLutView, vk::Sampler iblSampler)
{
    if (!descriptorSets) return;

    vk::DescriptorImageInfo irradianceInfo{iblSampler, irradianceView, vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo prefilterInfo{iblSampler, prefilterView, vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo brdfLutInfo{iblSampler, brdfLutView, vk::ImageLayout::eShaderReadOnlyOptimal};

    const uint32_t setCount = AppConfig::MAX_FRAMES_IN_FLIGHT * materialCount;
    for (uint32_t i = 0; i < setCount; ++i) {
        std::array<vk::WriteDescriptorSet, 3> writes{{
            vk::WriteDescriptorSet{(*descriptorSets)[i], 12, 0, 1, vk::DescriptorType::eCombinedImageSampler, &irradianceInfo},
            vk::WriteDescriptorSet{(*descriptorSets)[i], 13, 0, 1, vk::DescriptorType::eCombinedImageSampler, &prefilterInfo},
            vk::WriteDescriptorSet{(*descriptorSets)[i], 14, 0, 1, vk::DescriptorType::eCombinedImageSampler, &brdfLutInfo},
        }};
        device.updateDescriptorSets(writes, nullptr);
    }
}

void FrameManager::updateSkyboxDescriptorBuffers(vk::raii::Device& device)
{
    if (!skyboxDescriptorSets || uniformBuffers.size() < AppConfig::MAX_FRAMES_IN_FLIGHT) return;
    for (uint32_t i = 0; i < AppConfig::MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = *uniformBuffers[i];
        bufferInfo.offset = 64;
        bufferInfo.range = 128;
        vk::WriteDescriptorSet write{};
        write.dstSet = (*skyboxDescriptorSets)[i];
        write.dstBinding = 0;
        write.descriptorType = vk::DescriptorType::eUniformBuffer;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;
        device.updateDescriptorSets({write}, nullptr);
    }
}

void FrameManager::updatePostProcessDescriptorSet(uint32_t frameIndex, PostProcessSetSlot slot, vk::ImageView sourceView, vk::ImageView bloomView)
{
    if (!devicePtr || !postDescriptorSets || !postSampler) return;
    frameIndex %= AppConfig::MAX_FRAMES_IN_FLIGHT;
    const uint32_t slotIdx = static_cast<uint32_t>(slot);
    constexpr uint32_t slotCount = static_cast<uint32_t>(PostProcessSetSlot::Count);
    const uint32_t flatIndex = frameIndex * slotCount + slotIdx;

    vk::DescriptorImageInfo sourceInfo{};
    sourceInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    sourceInfo.imageView = sourceView;
    sourceInfo.sampler = static_cast<vk::Sampler>(*postSampler);

    vk::DescriptorImageInfo bloomInfo{};
    bloomInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    bloomInfo.imageView = bloomView;
    bloomInfo.sampler = static_cast<vk::Sampler>(*postSampler);

    std::array<vk::WriteDescriptorSet, 2> writes{};
    writes[0].dstSet = (*postDescriptorSets)[flatIndex];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &sourceInfo;
    writes[1].dstSet = (*postDescriptorSets)[flatIndex];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &bloomInfo;

    devicePtr->updateDescriptorSets(writes, nullptr);
}

vk::DescriptorSet FrameManager::getSkyboxDescriptorSet(uint32_t imageIndex) const
{
    if (!skyboxDescriptorSets) return vk::DescriptorSet{};
    imageIndex %= AppConfig::MAX_FRAMES_IN_FLIGHT;
    return static_cast<vk::DescriptorSet>((*skyboxDescriptorSets)[imageIndex]);
}

vk::DescriptorSet FrameManager::getPostProcessDescriptorSet(uint32_t frameIndex, PostProcessSetSlot slot) const
{
    if (!postDescriptorSets) return vk::DescriptorSet{};
    frameIndex %= AppConfig::MAX_FRAMES_IN_FLIGHT;
    const uint32_t slotIdx = static_cast<uint32_t>(slot);
    constexpr uint32_t slotCount = static_cast<uint32_t>(PostProcessSetSlot::Count);
    const uint32_t flatIndex = frameIndex * slotCount + slotIdx;
    return static_cast<vk::DescriptorSet>((*postDescriptorSets)[flatIndex]);
}

vk::Buffer FrameManager::getSkyboxVertexBuffer() const
{
    return skyboxVertexBuffer ? static_cast<vk::Buffer>(*skyboxVertexBuffer) : vk::Buffer{};
}

void FrameManager::cleanup(vk::raii::Device& device)
{
    (void)device;
    cleanupSwapChainResources(device);
    skyboxDescriptorSets.reset();
    skyboxDescriptorPool.reset();
    skyboxVertexBuffer.reset();
    skyboxVertexBufferMemory.reset();
    postDescriptorSets.reset();
    postDescriptorPool.reset();
    postSampler.reset();
    imageAvailableFence.reset();
    renderFinishedSemaphores.clear();
    inFlightFences.clear();
    sharedOpaqueSlots.clear();
    sharedNodeWorldMatrices.clear();
    sharedOpaqueBucketSpans.clear();
    sharedOpaqueDrawCount = 0;
    devicePtr = nullptr;
}

void FrameManager::updateUniformBuffer(uint32_t currentImage, vk::Extent2D extent, const Camera& camera,
                                       const glm::mat4& modelMatrix)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    PBRUniformBufferObject ubo{};
    ubo.model = modelMatrix;
    ubo.view = camera.getViewMatrix();
    // Use a larger far plane so common glTF scenes (e.g. Sponza) are not clipped away.
    ubo.proj = camera.getProjMatrix(extent.width / static_cast<float>(extent.height), 0.1f, 1000.0f);
    ubo.prevViewProj = lastViewProj;

    setPbrLights(ubo);
    ubo.camPos = glm::vec4(camera.getPosition(), 1.0f);

    const glm::mat4 currViewProj = ubo.proj * ubo.view;
    lastViewProj = currViewProj;

    (void)time;
    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    ++uniformFrameIndex;
}

void FrameManager::setPbrLights(PBRUniformBufferObject& ubo)
{
    // 1 directional light + 3 point lights.
    // Directional: direction points toward light source (e.g. from above).
    ubo.directionalLightDir = glm::vec4(-0.3f, -0.8f, -0.5f, 1.0f);  // w=1: enable
    // 下午 3–4 点：略暖偏白，不像黄昏那般金黄
    glm::vec3 lightColor = glm::vec3(0.92f, 0.82f, 0.72f) * 3.0f;
    ubo.directionalLightColor = glm::vec4(lightColor, 1.0f);
    // x=太阳半角(rad)，0.00465≈真实太阳，0.01~0.02 更明显软阴影；y=软阴影采样数(1=硬阴影，8推荐)
    ubo.directionalLightParams = glm::vec4(0.01f, 8.0f, 0.0f, 0.0f);

    ubo.lightPositions[0] = glm::vec4(2.0f, 4.0f, 2.0f, 1.0f);
    ubo.lightColors[0] = glm::vec4(3.0f, 2.0f, 1.0f, 1.0f);   // 暖红/珊瑚色

    ubo.lightPositions[1] = glm::vec4(-2.0f, 2.0f, -2.0f, 1.0f);
    ubo.lightColors[1] = glm::vec4(2.0f, 6.0f, 0.0f, 1.0f);   // 青色

    ubo.lightPositions[2] = glm::vec4(0.0f, 3.0f, 2.0f, 1.0f);
    ubo.lightColors[2] = glm::vec4(5.0f, 10.0f, 4.0f, 1.0f);  // 黄绿色

    const float exposure = 0.6f;
    const float gamma = 2.2f;
    const float ambientStrength = 0.03f;
    const float pointLightCount = AppConfig::ENABLE_POINT_LIGHTS ? 3.0f : 0.0f;
    ubo.params = glm::vec4(exposure, gamma, ambientStrength, pointLightCount);

    // Debug view selector (set in AppConfig.h)
    // iblParams.w: 0=Off, 1=NdotL, 2=baseColor, 3=Ng, 4=AO, 5~8=同4, 9=normal.y, 10=bias, 11=footprint, 12=current
    float debugViewW = (RuntimeConfig::debugViewMode > 0) ? static_cast<float>(RuntimeConfig::debugViewMode) : 0.0f;
    ubo.iblParams = glm::vec4(
        RuntimeConfig::enableDiffuseIbl ? RuntimeConfig::diffuseIblStrength : 0.0f,
        RuntimeConfig::enableSpecularIbl ? RuntimeConfig::specularIblStrength : 0.0f,
        RuntimeConfig::enableAo ? 1.0f : 0.0f,
        debugViewW);

    ubo.rtaoParams0 = glm::vec4(
        AppConfig::ENABLE_RTAO ? 1.0f : 0.0f,
        static_cast<float>(AppConfig::RTAO_RAY_COUNT),
        AppConfig::RTAO_RADIUS,
        AppConfig::RTAO_BIAS);
    ubo.rtaoParams1 = glm::vec4(
        AppConfig::RTAO_STRENGTH,
        AppConfig::RTAO_TEMPORAL_ALPHA,
        AppConfig::RTAO_DISOCCLUSION_THRESHOLD,
        static_cast<float>(uniformFrameIndex));
}

void FrameManager::createCommandBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator, SwapChain& swapChain)
{
    commandPoolPtr = &resourceCreator.getCommandPool();
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = *commandPoolPtr;
    allocInfo.commandBufferCount = static_cast<uint32_t>(swapChain.getImages().size());

    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void FrameManager::createSyncObjects(vk::raii::Device& device, SwapChain& swapChain)
{
    imageAvailableFence.reset();
    renderFinishedSemaphores.clear();
    inFlightFences.clear();

    uint32_t imageCount = static_cast<uint32_t>(swapChain.getImages().size());
    renderFinishedSemaphores.reserve(imageCount);
    inFlightFences.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    imageAvailableFence = vk::raii::Fence(device, fenceInfo);

    for (uint32_t i = 0; i < imageCount; i++) {
        renderFinishedSemaphores.push_back(vk::raii::Semaphore(device, semaphoreInfo));
    }
    for (size_t i = 0; i < AppConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        inFlightFences.push_back(vk::raii::Fence(device, fenceInfo));
    }
}

void FrameManager::createDrawDataBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator)
{
    (void)device;
    const vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(maxDraws) * sizeof(glm::mat4);
    drawDataBuffers.clear();
    drawDataBuffersMemory.clear();
    drawDataBuffersMapped.clear();
    drawDataBuffers.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);
    drawDataBuffersMemory.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);
    drawDataBuffersMapped.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < AppConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        BufferAllocation alloc = resourceCreator.createBuffer(bufferSize,
                                                              vk::BufferUsageFlagBits::eStorageBuffer,
                                                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        drawDataBuffers.push_back(std::move(alloc.buffer));
        drawDataBuffersMemory.push_back(std::move(alloc.memory));
        auto mapResult = drawDataBuffersMemory.back().mapMemory(0, bufferSize);
        drawDataBuffersMapped.push_back(mapResult);
    }
}

void* FrameManager::getDrawDataMapped(uint32_t frameIndex) const
{
    if (frameIndex >= drawDataBuffersMapped.size()) return nullptr;
    return drawDataBuffersMapped[frameIndex];
}

vk::Buffer FrameManager::getDrawDataBuffer(uint32_t frameIndex) const
{
    if (frameIndex >= drawDataBuffers.size()) return vk::Buffer{};
    return static_cast<vk::Buffer>(*drawDataBuffers[frameIndex]);
}

void FrameManager::createIndirectCommandBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator)
{
    (void)device;
    const vk::DeviceSize bufferSize = static_cast<vk::DeviceSize>(maxDraws) * sizeof(vk::DrawIndexedIndirectCommand);
    indirectCommandBuffers.clear();
    indirectCommandBuffersMemory.clear();
    indirectCommandBuffersMapped.clear();
    indirectCommandBuffers.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);
    indirectCommandBuffersMemory.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);
    indirectCommandBuffersMapped.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < AppConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        BufferAllocation alloc = resourceCreator.createBuffer(bufferSize,
                                                             vk::BufferUsageFlagBits::eIndirectBuffer,
                                                             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        indirectCommandBuffers.push_back(std::move(alloc.buffer));
        indirectCommandBuffersMemory.push_back(std::move(alloc.memory));
        auto mapResult = indirectCommandBuffersMemory.back().mapMemory(0, bufferSize);
        indirectCommandBuffersMapped.push_back(mapResult);
    }
}

void* FrameManager::getIndirectCommandsMapped(uint32_t frameIndex) const
{
    if (frameIndex >= indirectCommandBuffersMapped.size()) return nullptr;
    return indirectCommandBuffersMapped[frameIndex];
}

vk::Buffer FrameManager::getIndirectCommandsBuffer(uint32_t frameIndex) const
{
    if (frameIndex >= indirectCommandBuffers.size()) return vk::Buffer{};
    return static_cast<vk::Buffer>(*indirectCommandBuffers[frameIndex]);
}

void FrameManager::createUniformBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator)
{
    vk::DeviceSize bufferSize = sizeof(PBRUniformBufferObject);
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();
    uniformBuffers.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.reserve(AppConfig::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < AppConfig::MAX_FRAMES_IN_FLIGHT; i++) {
        BufferAllocation alloc = resourceCreator.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                                             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        uniformBuffers.push_back(std::move(alloc.buffer));
        uniformBuffersMemory.push_back(std::move(alloc.memory));
        auto mapResult = uniformBuffersMemory.back().mapMemory(0, bufferSize);
        uniformBuffersMapped.push_back(mapResult);
    }
}

void FrameManager::createDefaultPbrTextures(VulkanResourceCreator& resourceCreator)
{
    if (defaultBaseColor.view && defaultBaseColor.sampler
        && defaultMetallicRoughness.view && defaultMetallicRoughness.sampler
        && defaultNormal.view && defaultNormal.sampler
        && defaultOcclusion.view && defaultOcclusion.sampler
        && defaultEmissive.view && defaultEmissive.sampler) {
        return;
    }

    auto createSolid1x1 = [&](vk::Format format, const std::array<uint8_t, 4>& rgba, GpuTexture& out) {
        vk::raii::Device& device = resourceCreator.getDevice();

        const vk::DeviceSize imageSize = 4;
        BufferAllocation staging = resourceCreator.createBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        void* mapped = staging.memory.mapMemory(0, imageSize);
        std::memcpy(mapped, rgba.data(), static_cast<size_t>(imageSize));
        staging.memory.unmapMemory();

        ImageAllocation imgAlloc = resourceCreator.createImage(
            1, 1, 1, vk::SampleCountFlagBits::e1,
            format, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        resourceCreator.transitionImageLayout(
            static_cast<vk::Image>(*imgAlloc.image), format,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
        resourceCreator.copyBufferToImage(static_cast<vk::Buffer>(*staging.buffer), static_cast<vk::Image>(*imgAlloc.image), 1, 1);
        resourceCreator.transitionImageLayout(
            static_cast<vk::Image>(*imgAlloc.image), format,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

        out.image = std::move(imgAlloc.image);
        out.memory = std::move(imgAlloc.memory);
        out.view = resourceCreator.createImageView(static_cast<vk::Image>(*out.image), format, vk::ImageAspectFlagBits::eColor, 1);

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        const vk::PhysicalDeviceFeatures features = resourceCreator.getPhysicalDevice().getFeatures();
        if (features.samplerAnisotropy) {
            samplerInfo.anisotropyEnable = VK_TRUE;
            samplerInfo.maxAnisotropy = 16.0f;
        } else {
            samplerInfo.anisotropyEnable = VK_FALSE;
            samplerInfo.maxAnisotropy = 1.0f;
        }

        out.sampler = vk::raii::Sampler(device, samplerInfo);
    };

    // Default PBR fallbacks.
    createSolid1x1(vk::Format::eR8G8B8A8Srgb, {255, 255, 255, 255}, defaultBaseColor);
    // glTF metallic-roughness: G=roughness, B=metallic.
    createSolid1x1(vk::Format::eR8G8B8A8Unorm, {0, 255, 0, 255}, defaultMetallicRoughness);
    createSolid1x1(vk::Format::eR8G8B8A8Unorm, {128, 128, 255, 255}, defaultNormal);
    createSolid1x1(vk::Format::eR8G8B8A8Unorm, {255, 255, 255, 255}, defaultOcclusion);
    createSolid1x1(vk::Format::eR8G8B8A8Unorm, {0, 0, 0, 255}, defaultEmissive);
}

void FrameManager::createDefaultIblTextures(VulkanResourceCreator& resourceCreator)
{
    if (defaultIblIrradiance.view && defaultIblIrradiance.sampler) return;

    vk::raii::Device& device = resourceCreator.getDevice();
    const uint32_t cubeSize = 1;
    const vk::DeviceSize cubeFaceSize = cubeSize * cubeSize * 8;  // RGBA16F = 8 bytes
    const vk::DeviceSize cubeTotalSize = cubeFaceSize * 6;
    // Half-float grey (0.03, 0.05, 0.08, 1.0) - packed as uint16
    std::array<uint16_t, 4> greyHalf = {0x2F14, 0x2F33, 0x2F85, 0x3C00};

    BufferAllocation cubeStaging = resourceCreator.createBuffer(
        cubeTotalSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* cubeMap = cubeStaging.memory.mapMemory(0, cubeTotalSize);
    for (uint32_t i = 0; i < 6; ++i) {
        std::memcpy(static_cast<char*>(cubeMap) + i * cubeFaceSize, greyHalf.data(), 8);
    }
    cubeStaging.memory.unmapMemory();

    ImageAllocation cubeAlloc = resourceCreator.createImage(
        cubeSize, cubeSize, 1, vk::SampleCountFlagBits::e1,
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, 6, vk::ImageCreateFlagBits::eCubeCompatible);

    std::vector<vk::BufferImageCopy> cubeRegions(6);
    for (uint32_t i = 0; i < 6; ++i) {
        cubeRegions[i].imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, i, 1};
        cubeRegions[i].imageExtent = vk::Extent3D{cubeSize, cubeSize, 1};
        cubeRegions[i].bufferOffset = i * cubeFaceSize;
    }
    resourceCreator.transitionImageLayout(static_cast<vk::Image>(*cubeAlloc.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1, 6);
    resourceCreator.copyBufferToImage(*cubeStaging.buffer, static_cast<vk::Image>(*cubeAlloc.image), cubeRegions);
    resourceCreator.transitionImageLayout(static_cast<vk::Image>(*cubeAlloc.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1, 6);

    defaultIblIrradiance.image = std::move(cubeAlloc.image);
    defaultIblIrradiance.memory = std::move(cubeAlloc.memory);
    defaultIblIrradiance.view = resourceCreator.createImageView(*defaultIblIrradiance.image,
        vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, 1, vk::ImageViewType::eCube, 0, 6);

    ImageAllocation cubeAlloc2 = resourceCreator.createImage(
        cubeSize, cubeSize, 1, vk::SampleCountFlagBits::e1,
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, 6, vk::ImageCreateFlagBits::eCubeCompatible);
    resourceCreator.transitionImageLayout(static_cast<vk::Image>(*cubeAlloc2.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1, 6);
    resourceCreator.copyBufferToImage(*cubeStaging.buffer, static_cast<vk::Image>(*cubeAlloc2.image), cubeRegions);
    resourceCreator.transitionImageLayout(static_cast<vk::Image>(*cubeAlloc2.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1, 6);
    defaultIblPrefilter.image = std::move(cubeAlloc2.image);
    defaultIblPrefilter.memory = std::move(cubeAlloc2.memory);
    defaultIblPrefilter.view = resourceCreator.createImageView(*defaultIblPrefilter.image,
        vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor, 1, vk::ImageViewType::eCube, 0, 6);

    std::array<uint16_t, 2> brdfLutPixel = {0x3C00, 0x3C00};  // 1.0f in half
    const vk::DeviceSize brdfSize = 4;
    BufferAllocation brdfStaging = resourceCreator.createBuffer(brdfSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* brdfMap = brdfStaging.memory.mapMemory(0, brdfSize);
    std::memcpy(brdfMap, brdfLutPixel.data(), 4);
    brdfStaging.memory.unmapMemory();

    ImageAllocation brdfAlloc = resourceCreator.createImage(1, 1, 1, vk::SampleCountFlagBits::e1,
        vk::Format::eR16G16Sfloat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.transitionImageLayout(static_cast<vk::Image>(*brdfAlloc.image), vk::Format::eR16G16Sfloat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
    resourceCreator.copyBufferToImage(*brdfStaging.buffer, static_cast<vk::Image>(*brdfAlloc.image), 1, 1);
    resourceCreator.transitionImageLayout(static_cast<vk::Image>(*brdfAlloc.image), vk::Format::eR16G16Sfloat,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

    defaultIblBrdf.image = std::move(brdfAlloc.image);
    defaultIblBrdf.memory = std::move(brdfAlloc.memory);
    defaultIblBrdf.view = resourceCreator.createImageView(*defaultIblBrdf.image,
        vk::Format::eR16G16Sfloat, vk::ImageAspectFlagBits::eColor, 1);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    defaultIblIrradiance.sampler = vk::raii::Sampler(device, samplerInfo);
    defaultIblPrefilter.sampler = vk::raii::Sampler(device, samplerInfo);
    defaultIblBrdf.sampler = vk::raii::Sampler(device, samplerInfo);
}

void FrameManager::createDepthResolveTexture(VulkanResourceCreator& resourceCreator)
{
    if (depthResolve.view && depthResolve.image && depthResolve.memory) {
        return;
    }

    depthResolveFormat = resourceCreator.findDepthFormat();
    ImageAllocation alloc = resourceCreator.createImage(
        swapChainExtent.width, swapChainExtent.height, 1, vk::SampleCountFlagBits::e1,
        depthResolveFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    depthResolve.image = std::move(alloc.image);
    depthResolve.memory = std::move(alloc.memory);
    depthResolve.view = resourceCreator.createImageView(
        static_cast<vk::Image>(*depthResolve.image), depthResolveFormat, vk::ImageAspectFlagBits::eDepth, 1);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eNearest;
    samplerInfo.minFilter = vk::Filter::eNearest;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.maxLod = 0.0f;
    depthResolve.sampler = vk::raii::Sampler(resourceCreator.getDevice(), samplerInfo);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*depthResolve.image),
        depthResolveFormat,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal,
        1);
}

void FrameManager::createNormalTextures(VulkanResourceCreator& resourceCreator, vk::SampleCountFlagBits msaaSamples)
{
    if (normalResolve.view && normalResolve.image && normalResolve.memory && normalResolve.sampler) {
        if (msaaSamples == vk::SampleCountFlagBits::e1 || (normalPrepass.view && normalPrepass.image && normalPrepass.memory)) {
            return;
        }
    }

    auto supportsNormalFormat = [&](vk::Format fmt) -> bool {
        const vk::FormatProperties props = resourceCreator.getPhysicalDevice().getFormatProperties(fmt);
        const vk::FormatFeatureFlags features = props.optimalTilingFeatures;
        const bool sampled = static_cast<bool>(features & vk::FormatFeatureFlagBits::eSampledImage);
        const bool colorAttachment = static_cast<bool>(features & vk::FormatFeatureFlagBits::eColorAttachment);
        return sampled && colorAttachment;
    };

    // Keep the normal attachment format consistent with the pipeline's dynamic-rendering formats.
    // (A mismatch here can lead to undefined results / corrupted normals.)
    normalFormat = vk::Format::eR16G16B16A16Sfloat;
    if (!supportsNormalFormat(normalFormat)) {
        throw std::runtime_error("normal prepass requires VK_FORMAT_R16G16B16A16_SFLOAT sampled+color-attachment support");
    }

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eNearest;
    samplerInfo.minFilter = vk::Filter::eNearest;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.maxLod = 0.0f;

    if (msaaSamples != vk::SampleCountFlagBits::e1) {
        ImageAllocation msaaAlloc = resourceCreator.createImage(
            swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
            normalFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        normalPrepass.image = std::move(msaaAlloc.image);
        normalPrepass.memory = std::move(msaaAlloc.memory);
        normalPrepass.view = resourceCreator.createImageView(
            static_cast<vk::Image>(*normalPrepass.image), normalFormat, vk::ImageAspectFlagBits::eColor, 1);
    }

    ImageAllocation resolveAlloc = resourceCreator.createImage(
        swapChainExtent.width, swapChainExtent.height, 1, vk::SampleCountFlagBits::e1,
        normalFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    normalResolve.image = std::move(resolveAlloc.image);
    normalResolve.memory = std::move(resolveAlloc.memory);
    normalResolve.view = resourceCreator.createImageView(
        static_cast<vk::Image>(*normalResolve.image), normalFormat, vk::ImageAspectFlagBits::eColor, 1);
    normalResolve.sampler = vk::raii::Sampler(resourceCreator.getDevice(), samplerInfo);

    if (normalPrepass.image) {
        resourceCreator.transitionImageLayout(
            static_cast<vk::Image>(*normalPrepass.image),
            normalFormat,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            1);
    }

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*normalResolve.image),
        normalFormat,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        1);
}

void FrameManager::createLinearDepthTextures(VulkanResourceCreator& resourceCreator, vk::SampleCountFlagBits msaaSamples)
{
    if (linearDepthResolve.view && linearDepthResolve.image && linearDepthResolve.memory && linearDepthResolve.sampler) {
        if (msaaSamples == vk::SampleCountFlagBits::e1 || (linearDepthPrepass.view && linearDepthPrepass.image && linearDepthPrepass.memory)) {
            return;
        }
    }

    auto supportsLinearDepthFormat = [&](vk::Format fmt) -> bool {
        const vk::FormatProperties props = resourceCreator.getPhysicalDevice().getFormatProperties(fmt);
        const vk::FormatFeatureFlags features = props.optimalTilingFeatures;
        const bool sampled = static_cast<bool>(features & vk::FormatFeatureFlagBits::eSampledImage);
        const bool colorAttachment = static_cast<bool>(features & vk::FormatFeatureFlagBits::eColorAttachment);
        return sampled && colorAttachment;
    };

    linearDepthFormat = vk::Format::eR16Sfloat;
    if (!supportsLinearDepthFormat(linearDepthFormat)) {
        throw std::runtime_error("linear depth prepass requires VK_FORMAT_R16_SFLOAT sampled+color-attachment support");
    }

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eNearest;
    samplerInfo.minFilter = vk::Filter::eNearest;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.maxLod = 0.0f;

    if (msaaSamples != vk::SampleCountFlagBits::e1) {
        ImageAllocation msaaAlloc = resourceCreator.createImage(
            swapChainExtent.width, swapChainExtent.height, 1, msaaSamples,
            linearDepthFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        linearDepthPrepass.image = std::move(msaaAlloc.image);
        linearDepthPrepass.memory = std::move(msaaAlloc.memory);
        linearDepthPrepass.view = resourceCreator.createImageView(
            static_cast<vk::Image>(*linearDepthPrepass.image), linearDepthFormat, vk::ImageAspectFlagBits::eColor, 1);
    }

    ImageAllocation resolveAlloc = resourceCreator.createImage(
        swapChainExtent.width, swapChainExtent.height, 1, vk::SampleCountFlagBits::e1,
        linearDepthFormat, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    linearDepthResolve.image = std::move(resolveAlloc.image);
    linearDepthResolve.memory = std::move(resolveAlloc.memory);
    linearDepthResolve.view = resourceCreator.createImageView(
        static_cast<vk::Image>(*linearDepthResolve.image), linearDepthFormat, vk::ImageAspectFlagBits::eColor, 1);
    linearDepthResolve.sampler = vk::raii::Sampler(resourceCreator.getDevice(), samplerInfo);

    if (linearDepthPrepass.image) {
        resourceCreator.transitionImageLayout(
            static_cast<vk::Image>(*linearDepthPrepass.image),
            linearDepthFormat,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            1);
    }

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*linearDepthResolve.image),
        linearDepthFormat,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        1);
}

void FrameManager::createRtaoComputeTextures(VulkanResourceCreator& resourceCreator)
{
    auto supportsStorageImage = [&](vk::Format fmt) -> bool {
        const vk::FormatProperties props = resourceCreator.getPhysicalDevice().getFormatProperties(fmt);
        return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);
    };

    rtaoFormat = vk::Format::eR16Sfloat;
    if (!supportsStorageImage(rtaoFormat)) {
        throw std::runtime_error("RTAO requires VK_FORMAT_R16_SFLOAT storage-image support");
    }

    const uint32_t halfWidth = swapChainExtent.width;
    const uint32_t halfHeight = swapChainExtent.height;

    vk::SamplerCreateInfo halfSamplerInfo{};
    halfSamplerInfo.magFilter = vk::Filter::eLinear;
    halfSamplerInfo.minFilter = vk::Filter::eLinear;
    halfSamplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    halfSamplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    halfSamplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    halfSamplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    halfSamplerInfo.maxLod = 0.0f;

    vk::SamplerCreateInfo fullSamplerInfo = halfSamplerInfo;
    fullSamplerInfo.magFilter = vk::Filter::eLinear;
    fullSamplerInfo.minFilter = vk::Filter::eLinear;

    auto createR16fTexture = [&](uint32_t width, uint32_t height, GpuTexture& outTex, const vk::SamplerCreateInfo& samplerInfo) {
        ImageAllocation alloc = resourceCreator.createImage(
            width, height, 1, vk::SampleCountFlagBits::e1,
            rtaoFormat, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        outTex.image = std::move(alloc.image);
        outTex.memory = std::move(alloc.memory);
        outTex.view = resourceCreator.createImageView(
            static_cast<vk::Image>(*outTex.image), rtaoFormat, vk::ImageAspectFlagBits::eColor, 1);
        outTex.sampler = vk::raii::Sampler(resourceCreator.getDevice(), samplerInfo);

        vk::Image imageHandle = static_cast<vk::Image>(*outTex.image);
        resourceCreator.executeSingleTimeCommands([&](vk::raii::CommandBuffer& cb) {
            vk::ImageSubresourceRange subresourceRange{};
            subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            subresourceRange.baseMipLevel = 0;
            subresourceRange.levelCount = 1;
            subresourceRange.baseArrayLayer = 0;
            subresourceRange.layerCount = 1;

            vk::ImageMemoryBarrier toTransfer{};
            toTransfer.oldLayout = vk::ImageLayout::eUndefined;
            toTransfer.newLayout = vk::ImageLayout::eTransferDstOptimal;
            toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.image = imageHandle;
            toTransfer.subresourceRange = subresourceRange;
            toTransfer.srcAccessMask = {};
            toTransfer.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer,
                {},
                {},
                {},
                toTransfer);

            vk::ClearColorValue clearColor(std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f});
            cb.clearColorImage(imageHandle, vk::ImageLayout::eTransferDstOptimal, clearColor, subresourceRange);

            vk::ImageMemoryBarrier toGeneral{};
            toGeneral.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            toGeneral.newLayout = vk::ImageLayout::eGeneral;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = imageHandle;
            toGeneral.subresourceRange = subresourceRange;
            toGeneral.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            toGeneral.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;

            cb.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eFragmentShader,
                {},
                {},
                {},
                toGeneral);
        });
    };

    for (uint32_t i = 0; i < 2; ++i) {
        if (!rtaoHalfHistory[i].view) {
            createR16fTexture(halfWidth, halfHeight, rtaoHalfHistory[i], halfSamplerInfo);
        }
        if (!rtaoAtrousPingPong[i].view) {
            createR16fTexture(halfWidth, halfHeight, rtaoAtrousPingPong[i], halfSamplerInfo);
        }
    }

    if (!rtaoFull.view) {
        createR16fTexture(swapChainExtent.width, swapChainExtent.height, rtaoFull, fullSamplerInfo);
    }
}

vk::ImageView FrameManager::getDepthResolveImageView() const
{
    return depthResolve.view ? static_cast<vk::ImageView>(*depthResolve.view) : vk::ImageView{};
}

vk::Image FrameManager::getDepthResolveImage() const
{
    return depthResolve.image ? static_cast<vk::Image>(*depthResolve.image) : vk::Image{};
}

vk::Sampler FrameManager::getDepthResolveSampler() const
{
    return depthResolve.sampler ? static_cast<vk::Sampler>(*depthResolve.sampler) : vk::Sampler{};
}

vk::ImageView FrameManager::getNormalPrepassImageView() const
{
    if (normalPrepass.view) {
        return static_cast<vk::ImageView>(*normalPrepass.view);
    }
    return normalResolve.view ? static_cast<vk::ImageView>(*normalResolve.view) : vk::ImageView{};
}

vk::ImageView FrameManager::getNormalResolveImageView() const
{
    return normalResolve.view ? static_cast<vk::ImageView>(*normalResolve.view) : vk::ImageView{};
}

vk::Image FrameManager::getNormalResolveImage() const
{
    return normalResolve.image ? static_cast<vk::Image>(*normalResolve.image) : vk::Image{};
}

vk::Sampler FrameManager::getNormalResolveSampler() const
{
    return normalResolve.sampler ? static_cast<vk::Sampler>(*normalResolve.sampler) : vk::Sampler{};
}

vk::ImageView FrameManager::getLinearDepthPrepassImageView() const
{
    if (linearDepthPrepass.view) {
        return static_cast<vk::ImageView>(*linearDepthPrepass.view);
    }
    return linearDepthResolve.view ? static_cast<vk::ImageView>(*linearDepthResolve.view) : vk::ImageView{};
}

vk::ImageView FrameManager::getLinearDepthResolveImageView() const
{
    return linearDepthResolve.view ? static_cast<vk::ImageView>(*linearDepthResolve.view) : vk::ImageView{};
}

vk::Image FrameManager::getLinearDepthResolveImage() const
{
    return linearDepthResolve.image ? static_cast<vk::Image>(*linearDepthResolve.image) : vk::Image{};
}

vk::Sampler FrameManager::getLinearDepthResolveSampler() const
{
    return linearDepthResolve.sampler ? static_cast<vk::Sampler>(*linearDepthResolve.sampler) : vk::Sampler{};
}

vk::ImageView FrameManager::getRtaoHalfHistoryImageViewForFrame(uint32_t frameIndex, bool previous) const
{
    const uint32_t index = previous ? (((frameIndex % 2) + 1) % 2) : (frameIndex % 2);
    return rtaoHalfHistory[index].view ? static_cast<vk::ImageView>(*rtaoHalfHistory[index].view) : vk::ImageView{};
}

vk::Image FrameManager::getRtaoHalfHistoryImageForFrame(uint32_t frameIndex, bool previous) const
{
    const uint32_t index = previous ? (((frameIndex % 2) + 1) % 2) : (frameIndex % 2);
    return rtaoHalfHistory[index].image ? static_cast<vk::Image>(*rtaoHalfHistory[index].image) : vk::Image{};
}

vk::Sampler FrameManager::getRtaoHalfHistorySampler() const
{
    return rtaoHalfHistory[0].sampler ? static_cast<vk::Sampler>(*rtaoHalfHistory[0].sampler) : vk::Sampler{};
}

vk::ImageView FrameManager::getRtaoAtrousImageView(uint32_t pingPongIndex) const
{
    const uint32_t idx = pingPongIndex % 2;
    return rtaoAtrousPingPong[idx].view ? static_cast<vk::ImageView>(*rtaoAtrousPingPong[idx].view) : vk::ImageView{};
}

vk::Image FrameManager::getRtaoAtrousImage(uint32_t pingPongIndex) const
{
    const uint32_t idx = pingPongIndex % 2;
    return rtaoAtrousPingPong[idx].image ? static_cast<vk::Image>(*rtaoAtrousPingPong[idx].image) : vk::Image{};
}

vk::Sampler FrameManager::getRtaoAtrousSampler() const
{
    return rtaoAtrousPingPong[0].sampler ? static_cast<vk::Sampler>(*rtaoAtrousPingPong[0].sampler) : vk::Sampler{};
}

vk::ImageView FrameManager::getRtaoFullImageView() const
{
    return rtaoFull.view ? static_cast<vk::ImageView>(*rtaoFull.view) : vk::ImageView{};
}

vk::Image FrameManager::getRtaoFullImage() const
{
    return rtaoFull.image ? static_cast<vk::Image>(*rtaoFull.image) : vk::Image{};
}

vk::Sampler FrameManager::getRtaoFullSampler() const
{
    return rtaoFull.sampler ? static_cast<vk::Sampler>(*rtaoFull.sampler) : vk::Sampler{};
}

vk::Buffer FrameManager::getReflectionInstanceLUTBuffer() const
{
    return instanceLUTBuffer ? static_cast<vk::Buffer>(*instanceLUTBuffer) : vk::Buffer{};
}

vk::Buffer FrameManager::getReflectionIndexBuffer() const
{
    return reflectionIndexBuffer ? static_cast<vk::Buffer>(*reflectionIndexBuffer) : vk::Buffer{};
}

vk::Buffer FrameManager::getReflectionUVBuffer() const
{
    return reflectionUVBuffer ? static_cast<vk::Buffer>(*reflectionUVBuffer) : vk::Buffer{};
}

vk::Buffer FrameManager::getReflectionMaterialParamsBuffer() const
{
    return reflectionMaterialParamsBuffer ? static_cast<vk::Buffer>(*reflectionMaterialParamsBuffer) : vk::Buffer{};
}

const std::array<vk::DescriptorImageInfo, AppConfig::MAX_REFLECTION_MATERIAL_COUNT>& FrameManager::getReflectionBaseColorArrayInfos() const
{
    return reflectionBaseColorArrayInfos;
}

void FrameManager::createReflectionBuffers(VulkanResourceCreator& resourceCreator, const Model& model)
{
    instanceLUTBuffer.reset();
    instanceLUTMemory.reset();
    reflectionIndexBuffer.reset();
    reflectionIndexMemory.reset();
    reflectionUVBuffer.reset();
    reflectionUVMemory.reset();
    reflectionMaterialParamsBuffer.reset();
    reflectionMaterialParamsMemory.reset();

    const auto& meshes = model.getMeshes();
    reflectionMeshCount = static_cast<uint32_t>(meshes.size());
    if (reflectionMeshCount == 0) {
        // 创建最小 buffer 以便 descriptor 有效
        vk::raii::Device& device = resourceCreator.getDevice();
        BufferAllocation dummy = resourceCreator.createBuffer(16, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        instanceLUTBuffer = std::move(dummy.buffer);
        instanceLUTMemory = std::move(dummy.memory);
        BufferAllocation idxDummy = resourceCreator.createBuffer(16, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        reflectionIndexBuffer = std::move(idxDummy.buffer);
        reflectionIndexMemory = std::move(idxDummy.memory);
        BufferAllocation uvDummy = resourceCreator.createBuffer(16, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        reflectionUVBuffer = std::move(uvDummy.buffer);
        reflectionUVMemory = std::move(uvDummy.memory);
        BufferAllocation matParamsDummy = resourceCreator.createBuffer(16, vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        reflectionMaterialParamsBuffer = std::move(matParamsDummy.buffer);
        reflectionMaterialParamsMemory = std::move(matParamsDummy.memory);
        return;
    }

    // 计算每 mesh 的 vertexOffset 与 indexBufferOffset
    std::vector<uint32_t> vertexOffsets(reflectionMeshCount);
    std::vector<uint32_t> indexOffsets(reflectionMeshCount);
    uint32_t totalVertices = 0;
    uint32_t totalIndices = 0;
    for (uint32_t i = 0; i < reflectionMeshCount; ++i) {
        vertexOffsets[i] = totalVertices;
        indexOffsets[i] = totalIndices;
        totalVertices += static_cast<uint32_t>(meshes[i].vertices.size());
        totalIndices += static_cast<uint32_t>(meshes[i].indices.size());
    }

    // Instance LUT: meshIndex -> { materialID, indexBufferOffset }
    std::vector<InstanceLUTEntry> lutEntries(reflectionMeshCount);
    const auto& materials = model.getMaterials();
    for (uint32_t i = 0; i < reflectionMeshCount; ++i) {
        int matIdx = meshes[i].materialIndex;
        lutEntries[i].materialID = (matIdx >= 0 && matIdx < static_cast<int>(materials.size()))
                                      ? static_cast<uint32_t>(matIdx)
                                      : 0u;
        lutEntries[i].indexBufferOffset = indexOffsets[i];
    }

    // 合并 UV buffer：按顶点顺序拼接 texCoord
    std::vector<glm::vec2> allUVs;
    allUVs.reserve(totalVertices);
    for (const auto& mesh : meshes) {
        for (const auto& v : mesh.vertices) {
            allUVs.push_back(v.texCoord);
        }
    }

    // 合并 index buffer：每个 mesh 的 index 需加上 vertexOffset 以引用全局顶点
    std::vector<uint32_t> allIndices;
    allIndices.reserve(totalIndices);
    for (uint32_t i = 0; i < reflectionMeshCount; ++i) {
        uint32_t voff = vertexOffsets[i];
        for (uint32_t idx : meshes[i].indices) {
            allIndices.push_back(voff + idx);
        }
    }

    vk::raii::Device& device = resourceCreator.getDevice();

    // 创建 Instance LUT buffer
    const vk::DeviceSize lutSize = static_cast<vk::DeviceSize>(lutEntries.size()) * sizeof(InstanceLUTEntry);
    BufferAllocation lutAlloc = resourceCreator.createBuffer(
        lutSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    instanceLUTBuffer = std::move(lutAlloc.buffer);
    instanceLUTMemory = std::move(lutAlloc.memory);
    void* lutMapped = instanceLUTMemory->mapMemory(0, lutSize);
    std::memcpy(lutMapped, lutEntries.data(), static_cast<size_t>(lutSize));
    instanceLUTMemory->unmapMemory();

    // 创建 combined index buffer (device local)
    const vk::DeviceSize indexSize = static_cast<vk::DeviceSize>(allIndices.size()) * sizeof(uint32_t);
    BufferAllocation indexStaging = resourceCreator.createBuffer(
        indexSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* idxMapped = indexStaging.memory.mapMemory(0, indexSize);
    std::memcpy(idxMapped, allIndices.data(), static_cast<size_t>(indexSize));
    indexStaging.memory.unmapMemory();

    BufferAllocation indexGpu = resourceCreator.createBuffer(
        indexSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    reflectionIndexBuffer = std::move(indexGpu.buffer);
    reflectionIndexMemory = std::move(indexGpu.memory);
    resourceCreator.copyBuffer(static_cast<vk::Buffer>(*indexStaging.buffer), static_cast<vk::Buffer>(*reflectionIndexBuffer), indexSize);

    // 创建 combined UV buffer (device local)
    const vk::DeviceSize uvSize = static_cast<vk::DeviceSize>(allUVs.size()) * sizeof(glm::vec2);
    BufferAllocation uvStaging = resourceCreator.createBuffer(
        uvSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* uvMapped = uvStaging.memory.mapMemory(0, uvSize);
    std::memcpy(uvMapped, allUVs.data(), static_cast<size_t>(uvSize));
    uvStaging.memory.unmapMemory();

    BufferAllocation uvGpu = resourceCreator.createBuffer(
        uvSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    reflectionUVBuffer = std::move(uvGpu.buffer);
    reflectionUVMemory = std::move(uvGpu.memory);
    resourceCreator.copyBuffer(static_cast<vk::Buffer>(*uvStaging.buffer), static_cast<vk::Buffer>(*reflectionUVBuffer), uvSize);

    // Material params: vec4 per material (x=alphaCutoff, y=alphaMode: 0=Opaque, 1=Mask, 2=Blend)
    const vk::DeviceSize materialParamsSize =
        static_cast<vk::DeviceSize>(AppConfig::MAX_REFLECTION_MATERIAL_COUNT) * sizeof(glm::vec4);
    std::vector<glm::vec4> materialParamsData(AppConfig::MAX_REFLECTION_MATERIAL_COUNT);
    for (uint32_t m = 0; m < AppConfig::MAX_REFLECTION_MATERIAL_COUNT; ++m) {
        float alphaMode = 0.0f;
        float alphaCutoff = 0.5f;
        if (m < static_cast<uint32_t>(materials.size())) {
            const Material& mat = materials[m];
            alphaCutoff = mat.alphaCutoff;
            if (mat.alphaMode == AlphaMode::Mask) {
                alphaMode = 1.0f;
            } else if (mat.alphaMode == AlphaMode::Blend) {
                alphaMode = 2.0f;
            }
        }
        materialParamsData[m] = glm::vec4(alphaCutoff, alphaMode, 0.0f, 0.0f);
    }
    BufferAllocation matParamsAlloc = resourceCreator.createBuffer(
        materialParamsSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    reflectionMaterialParamsBuffer = std::move(matParamsAlloc.buffer);
    reflectionMaterialParamsMemory = std::move(matParamsAlloc.memory);
    void* matParamsMapped = reflectionMaterialParamsMemory->mapMemory(0, materialParamsSize);
    std::memcpy(matParamsMapped, materialParamsData.data(), static_cast<size_t>(materialParamsSize));
    reflectionMaterialParamsMemory->unmapMemory();
}

void FrameManager::createDescriptorPool(vk::raii::Device& device)
{
    std::array<vk::DescriptorPoolSize, 4> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(AppConfig::MAX_FRAMES_IN_FLIGHT * materialCount);
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(AppConfig::MAX_FRAMES_IN_FLIGHT * materialCount * (5 + AppConfig::MAX_REFLECTION_MATERIAL_COUNT + 3 + 1));
    poolSizes[2].type = vk::DescriptorType::eAccelerationStructureKHR;
    poolSizes[2].descriptorCount = static_cast<uint32_t>(AppConfig::MAX_FRAMES_IN_FLIGHT * materialCount);
    poolSizes[3].type = vk::DescriptorType::eStorageBuffer;
    poolSizes[3].descriptorCount = static_cast<uint32_t>(AppConfig::MAX_FRAMES_IN_FLIGHT * materialCount * 4);  // 3 reflection + 1 drawData

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(AppConfig::MAX_FRAMES_IN_FLIGHT * materialCount);

    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void FrameManager::createDescriptorSets(vk::raii::Device& device, VulkanResourceCreator& resourceCreator,
                                        GraphicsPipeline& pipeline, const Model& model,
                                        RayTracingContext& rayTracingContext)
{
    (void)resourceCreator;
    const uint32_t setCount = AppConfig::MAX_FRAMES_IN_FLIGHT * materialCount;
    std::vector<vk::DescriptorSetLayout> layouts(setCount, pipeline.getDescriptorSetLayout());
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(setCount);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    const vk::AccelerationStructureKHR topLevelAS = rayTracingContext.getTopLevelAS();
    if (!topLevelAS) {
        throw std::runtime_error("ray tracing TLAS is not initialized");
    }

    const auto& materials = model.getMaterials();
    const auto& textures = model.getTextures();

    auto fillImageInfo = [&](int textureIndex, const GpuTexture& fallback, vk::DescriptorImageInfo& outInfo) {
        outInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        if (textureIndex >= 0 && textureIndex < static_cast<int>(textures.size())) {
            const GltfTexture& t = textures[static_cast<size_t>(textureIndex)];
            if (t.imageView && t.vkSampler) {
                outInfo.imageView = static_cast<vk::ImageView>(*t.imageView);
                outInfo.sampler = static_cast<vk::Sampler>(*t.vkSampler);
                return;
            }
        }
        outInfo.imageView = static_cast<vk::ImageView>(*fallback.view);
        outInfo.sampler = static_cast<vk::Sampler>(*fallback.sampler);
    };

    // 构建 RTAO/反射用 baseColor 纹理数组（供 getReflectionBaseColorArrayInfos）
    for (uint32_t m = 0; m < AppConfig::MAX_REFLECTION_MATERIAL_COUNT; ++m) {
        reflectionBaseColorArrayInfos[m].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        if (m < materialCount && !materials.empty() && m < static_cast<uint32_t>(materials.size())) {
            const Material* mat = &materials[m];
            fillImageInfo(mat->baseColorTextureIndex, defaultBaseColor, reflectionBaseColorArrayInfos[m]);
        } else {
            reflectionBaseColorArrayInfos[m].imageView = static_cast<vk::ImageView>(*defaultBaseColor.view);
            reflectionBaseColorArrayInfos[m].sampler = static_cast<vk::Sampler>(*defaultBaseColor.sampler);
        }
    }

    for (uint32_t frame = 0; frame < AppConfig::MAX_FRAMES_IN_FLIGHT; frame++) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = *uniformBuffers[frame];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(PBRUniformBufferObject);

        vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
        accelInfo.accelerationStructureCount = 1;
        accelInfo.pAccelerationStructures = &topLevelAS;

        for (uint32_t matIdx = 0; matIdx < materialCount; ++matIdx) {
            const uint32_t flatIndex = frame * materialCount + matIdx;

            const Material* mat = nullptr;
            if (!materials.empty() && matIdx < static_cast<uint32_t>(materials.size())) {
                mat = &materials[matIdx];
            }

            vk::DescriptorImageInfo baseColorInfo{};
            vk::DescriptorImageInfo metallicRoughnessInfo{};
            vk::DescriptorImageInfo normalInfo{};
            vk::DescriptorImageInfo occlusionInfo{};
            vk::DescriptorImageInfo emissiveInfo{};

            fillImageInfo(mat ? mat->baseColorTextureIndex : -1, defaultBaseColor, baseColorInfo);
            fillImageInfo(mat ? mat->metallicRoughnessTextureIndex : -1, defaultMetallicRoughness, metallicRoughnessInfo);
            fillImageInfo(mat ? mat->normalTextureIndex : -1, defaultNormal, normalInfo);
            fillImageInfo(mat ? mat->occlusionTextureIndex : -1, defaultOcclusion, occlusionInfo);
            fillImageInfo(mat ? mat->emissiveTextureIndex : -1, defaultEmissive, emissiveInfo);

            std::array<vk::WriteDescriptorSet, 7> descriptorWrites{};
            descriptorWrites[0].dstSet = (*descriptorSets)[flatIndex];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].dstSet = (*descriptorSets)[flatIndex];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &baseColorInfo;

            descriptorWrites[2].dstSet = (*descriptorSets)[flatIndex];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pImageInfo = &metallicRoughnessInfo;

            descriptorWrites[3].dstSet = (*descriptorSets)[flatIndex];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pImageInfo = &normalInfo;

            descriptorWrites[4].dstSet = (*descriptorSets)[flatIndex];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pImageInfo = &occlusionInfo;

            descriptorWrites[5].dstSet = (*descriptorSets)[flatIndex];
            descriptorWrites[5].dstBinding = 5;
            descriptorWrites[5].dstArrayElement = 0;
            descriptorWrites[5].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[5].descriptorCount = 1;
            descriptorWrites[5].pImageInfo = &emissiveInfo;

            descriptorWrites[6].dstSet = (*descriptorSets)[flatIndex];
            descriptorWrites[6].dstBinding = 6;
            descriptorWrites[6].dstArrayElement = 0;
            descriptorWrites[6].descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
            descriptorWrites[6].descriptorCount = 1;
            descriptorWrites[6].pNext = &accelInfo;

            vk::DescriptorBufferInfo lutInfo{};
            vk::DescriptorBufferInfo indexInfo{};
            vk::DescriptorBufferInfo uvInfo{};
            if (instanceLUTBuffer && reflectionIndexBuffer && reflectionUVBuffer) {
                lutInfo.buffer = *instanceLUTBuffer;
                lutInfo.offset = 0;
                lutInfo.range = VK_WHOLE_SIZE;
                indexInfo.buffer = *reflectionIndexBuffer;
                indexInfo.offset = 0;
                indexInfo.range = VK_WHOLE_SIZE;
                uvInfo.buffer = *reflectionUVBuffer;
                uvInfo.offset = 0;
                uvInfo.range = VK_WHOLE_SIZE;
            }

            std::array<vk::WriteDescriptorSet, 11> descriptorWritesFull{};
            for (size_t i = 0; i < 7; ++i) {
                descriptorWritesFull[i] = descriptorWrites[i];
            }
            descriptorWritesFull[7].dstSet = (*descriptorSets)[flatIndex];
            descriptorWritesFull[7].dstBinding = 7;
            descriptorWritesFull[7].dstArrayElement = 0;
            descriptorWritesFull[7].descriptorType = vk::DescriptorType::eStorageBuffer;
            descriptorWritesFull[7].descriptorCount = 1;
            descriptorWritesFull[7].pBufferInfo = &lutInfo;

            descriptorWritesFull[8].dstSet = (*descriptorSets)[flatIndex];
            descriptorWritesFull[8].dstBinding = 8;
            descriptorWritesFull[8].dstArrayElement = 0;
            descriptorWritesFull[8].descriptorType = vk::DescriptorType::eStorageBuffer;
            descriptorWritesFull[8].descriptorCount = 1;
            descriptorWritesFull[8].pBufferInfo = &indexInfo;

            descriptorWritesFull[9].dstSet = (*descriptorSets)[flatIndex];
            descriptorWritesFull[9].dstBinding = 9;
            descriptorWritesFull[9].dstArrayElement = 0;
            descriptorWritesFull[9].descriptorType = vk::DescriptorType::eStorageBuffer;
            descriptorWritesFull[9].descriptorCount = 1;
            descriptorWritesFull[9].pBufferInfo = &uvInfo;

            descriptorWritesFull[10].dstSet = (*descriptorSets)[flatIndex];
            descriptorWritesFull[10].dstBinding = 10;
            descriptorWritesFull[10].dstArrayElement = 0;
            descriptorWritesFull[10].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWritesFull[10].descriptorCount = AppConfig::MAX_REFLECTION_MATERIAL_COUNT;
            descriptorWritesFull[10].pImageInfo = reflectionBaseColorArrayInfos.data();

            vk::DescriptorBufferInfo drawDataInfo{};
            if (!drawDataBuffers.empty() && frame < drawDataBuffers.size()) {
                drawDataInfo.buffer = *drawDataBuffers[frame];
                drawDataInfo.offset = 0;
                drawDataInfo.range = static_cast<vk::DeviceSize>(maxDraws) * sizeof(glm::mat4);
            }

            vk::WriteDescriptorSet drawDataWrite{};
            drawDataWrite.dstSet = (*descriptorSets)[flatIndex];
            drawDataWrite.dstBinding = 11;
            drawDataWrite.dstArrayElement = 0;
            drawDataWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
            drawDataWrite.descriptorCount = 1;
            drawDataWrite.pBufferInfo = &drawDataInfo;

            vk::DescriptorImageInfo irradianceInfo{};
            irradianceInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            irradianceInfo.imageView = static_cast<vk::ImageView>(*defaultIblIrradiance.view);
            irradianceInfo.sampler = static_cast<vk::Sampler>(*defaultIblIrradiance.sampler);
            vk::DescriptorImageInfo prefilterInfo{};
            prefilterInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            prefilterInfo.imageView = static_cast<vk::ImageView>(*defaultIblPrefilter.view);
            prefilterInfo.sampler = static_cast<vk::Sampler>(*defaultIblPrefilter.sampler);
            vk::DescriptorImageInfo brdfLutInfo{};
            brdfLutInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            brdfLutInfo.imageView = static_cast<vk::ImageView>(*defaultIblBrdf.view);
            brdfLutInfo.sampler = static_cast<vk::Sampler>(*defaultIblBrdf.sampler);
            vk::DescriptorImageInfo rtaoFullInfo{};
            rtaoFullInfo.imageLayout = vk::ImageLayout::eGeneral;
            rtaoFullInfo.imageView = getRtaoFullImageView();
            rtaoFullInfo.sampler = getRtaoFullSampler();

            std::array<vk::WriteDescriptorSet, 16> descriptorWritesWithDrawData{};
            for (size_t i = 0; i < 11; ++i) {
                descriptorWritesWithDrawData[i] = descriptorWritesFull[i];
            }
            descriptorWritesWithDrawData[11] = drawDataWrite;
            descriptorWritesWithDrawData[12] = vk::WriteDescriptorSet{(*descriptorSets)[flatIndex], 12, 0, 1, vk::DescriptorType::eCombinedImageSampler, &irradianceInfo};
            descriptorWritesWithDrawData[13] = vk::WriteDescriptorSet{(*descriptorSets)[flatIndex], 13, 0, 1, vk::DescriptorType::eCombinedImageSampler, &prefilterInfo};
            descriptorWritesWithDrawData[14] = vk::WriteDescriptorSet{(*descriptorSets)[flatIndex], 14, 0, 1, vk::DescriptorType::eCombinedImageSampler, &brdfLutInfo};
            descriptorWritesWithDrawData[15] = vk::WriteDescriptorSet{(*descriptorSets)[flatIndex], 15, 0, 1, vk::DescriptorType::eCombinedImageSampler, &rtaoFullInfo};

            device.updateDescriptorSets(descriptorWritesWithDrawData, nullptr);
        }
    }
}

void FrameManager::cleanupSwapChainResources(vk::raii::Device& device)
{
    (void)device;
    if (!uniformBuffersMapped.empty()) {
        for (size_t i = 0; i < AppConfig::MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffersMemory[i].unmapMemory();
        }
        uniformBuffersMapped.clear();
    }
    uniformBuffers.clear();
    uniformBuffersMemory.clear();

    if (!drawDataBuffersMapped.empty()) {
        for (size_t i = 0; i < drawDataBuffersMemory.size(); i++) {
            drawDataBuffersMemory[i].unmapMemory();
        }
        drawDataBuffersMapped.clear();
    }
    drawDataBuffers.clear();
    drawDataBuffersMemory.clear();

    if (!indirectCommandBuffersMapped.empty()) {
        for (size_t i = 0; i < indirectCommandBuffersMemory.size(); i++) {
            indirectCommandBuffersMemory[i].unmapMemory();
        }
        indirectCommandBuffersMapped.clear();
    }
    indirectCommandBuffers.clear();
    indirectCommandBuffersMemory.clear();

    defaultBaseColor.sampler.reset();
    defaultBaseColor.view.reset();
    defaultBaseColor.image.reset();
    defaultBaseColor.memory.reset();
    defaultMetallicRoughness.sampler.reset();
    defaultMetallicRoughness.view.reset();
    defaultMetallicRoughness.image.reset();
    defaultMetallicRoughness.memory.reset();
    defaultNormal.sampler.reset();
    defaultNormal.view.reset();
    defaultNormal.image.reset();
    defaultNormal.memory.reset();
    defaultOcclusion.sampler.reset();
    defaultOcclusion.view.reset();
    defaultOcclusion.image.reset();
    defaultOcclusion.memory.reset();
    defaultEmissive.sampler.reset();
    defaultEmissive.view.reset();
    defaultEmissive.image.reset();
    defaultEmissive.memory.reset();
    defaultIblIrradiance.sampler.reset();
    defaultIblIrradiance.view.reset();
    defaultIblIrradiance.image.reset();
    defaultIblIrradiance.memory.reset();
    defaultIblPrefilter.sampler.reset();
    defaultIblPrefilter.view.reset();
    defaultIblPrefilter.image.reset();
    defaultIblPrefilter.memory.reset();
    defaultIblBrdf.sampler.reset();
    defaultIblBrdf.view.reset();
    defaultIblBrdf.image.reset();
    defaultIblBrdf.memory.reset();
    depthResolve.sampler.reset();
    depthResolve.view.reset();
    depthResolve.image.reset();
    depthResolve.memory.reset();
    depthResolveFormat = vk::Format::eUndefined;
    normalPrepass.sampler.reset();
    normalPrepass.view.reset();
    normalPrepass.image.reset();
    normalPrepass.memory.reset();
    normalResolve.sampler.reset();
    normalResolve.view.reset();
    normalResolve.image.reset();
    normalResolve.memory.reset();
    normalFormat = vk::Format::eUndefined;
    linearDepthPrepass.sampler.reset();
    linearDepthPrepass.view.reset();
    linearDepthPrepass.image.reset();
    linearDepthPrepass.memory.reset();
    linearDepthResolve.sampler.reset();
    linearDepthResolve.view.reset();
    linearDepthResolve.image.reset();
    linearDepthResolve.memory.reset();
    linearDepthFormat = vk::Format::eUndefined;
    for (auto& tex : rtaoHalfHistory) {
        tex.sampler.reset();
        tex.view.reset();
        tex.image.reset();
        tex.memory.reset();
    }
    for (auto& tex : rtaoAtrousPingPong) {
        tex.sampler.reset();
        tex.view.reset();
        tex.image.reset();
        tex.memory.reset();
    }
    rtaoFull.sampler.reset();
    rtaoFull.view.reset();
    rtaoFull.image.reset();
    rtaoFull.memory.reset();
    rtaoFormat = vk::Format::eUndefined;

    descriptorSets.reset();
    descriptorPool.reset();
    postDescriptorSets.reset();
    postDescriptorPool.reset();
    postSampler.reset();

    instanceLUTBuffer.reset();
    instanceLUTMemory.reset();
    reflectionIndexBuffer.reset();
    reflectionIndexMemory.reset();
    reflectionUVBuffer.reset();
    reflectionUVMemory.reset();
    reflectionMaterialParamsBuffer.reset();
    reflectionMaterialParamsMemory.reset();
    reflectionMeshCount = 0;

    commandBuffers.reset();

    materialCount = 1;
}
