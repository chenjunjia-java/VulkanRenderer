#include "Rendering/core/FrameManager.h"

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>

void FrameManager::init(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
                        Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator, GpuMesh& mesh,
                        Texture& texture, RayTracingContext& rayTracingContext)
{
    (void)rendergraph;
    pipelineLayoutHandle = pipeline.getPipelineLayout();
    swapChainExtent = swapChain.getExtent();
    sceneMesh = &mesh;

    createCommandBuffers(context.getDevice(), resourceCreator, swapChain);
    createSyncObjects(context.getDevice(), swapChain);
    createUniformBuffers(context.getDevice(), resourceCreator);
    createShadowTransparencyBuffers(resourceCreator, mesh);
    createDescriptorPool(context.getDevice());
    createDescriptorSets(context.getDevice(), pipeline, mesh, texture, rayTracingContext);
}

void FrameManager::recreate(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
                            Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator, Texture& texture,
                            RayTracingContext& rayTracingContext)
{
    (void)rendergraph;
    if (sceneMesh == nullptr) {
        throw std::runtime_error("scene mesh is not initialized for shadow transparency buffers");
    }
    swapChainExtent = swapChain.getExtent();
    cleanupSwapChainResources(context.getDevice());
    createCommandBuffers(context.getDevice(), resourceCreator, swapChain);
    createSyncObjects(context.getDevice(), swapChain);
    createUniformBuffers(context.getDevice(), resourceCreator);
    createShadowTransparencyBuffers(resourceCreator, *sceneMesh);
    createDescriptorPool(context.getDevice());
    createDescriptorSets(context.getDevice(), pipeline, *sceneMesh, texture, rayTracingContext);
}

void FrameManager::cleanup(vk::raii::Device& device)
{
    cleanupSwapChainResources(device);
    imageAvailableFence.reset();
    renderFinishedSemaphores.clear();
    inFlightFences.clear();
    sceneMesh = nullptr;
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
    ubo.proj = camera.getProjMatrix(extent.width / static_cast<float>(extent.height), 0.1f, 10.0f);

    // Simple test lighting (direct lights only, no IBL).
    // Use physically plausible inverse-square attenuation in the shader, so keep intensities higher than LDR phong.
    ubo.lightPositions[0] = glm::vec4(2.0f, 4.0f, 2.0f, 1.0f);
    ubo.lightColors[0] = glm::vec4(250.0f, 250.0f, 250.0f, 1.0f);

    ubo.lightPositions[1] = glm::vec4(-2.0f, 2.0f, -2.0f, 1.0f);
    ubo.lightColors[1] = glm::vec4(50.0f, 100.0f, 250.0f, 1.0f);

    ubo.lightPositions[2] = glm::vec4(0.0f);
    ubo.lightColors[2] = glm::vec4(0.0f);
    ubo.lightPositions[3] = glm::vec4(0.0f);
    ubo.lightColors[3] = glm::vec4(0.0f);

    ubo.camPos = glm::vec4(camera.getPosition(), 1.0f);

    const float exposure = 1.0f;
    const float gamma = 2.2f;
    const float ambientStrength = 0.03f;
    const float lightCount = 2.0f;
    ubo.params = glm::vec4(exposure, gamma, ambientStrength, lightCount);

    (void)time;
    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
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
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    imageAvailableFence = vk::raii::Fence(device, fenceInfo);

    for (uint32_t i = 0; i < imageCount; i++) {
        renderFinishedSemaphores.push_back(vk::raii::Semaphore(device, semaphoreInfo));
    }
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        inFlightFences.push_back(vk::raii::Fence(device, fenceInfo));
    }
}

void FrameManager::createUniformBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator)
{
    vk::DeviceSize bufferSize = sizeof(PBRUniformBufferObject);
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    uniformBuffersMapped.clear();
    uniformBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.reserve(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.reserve(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        BufferAllocation alloc = resourceCreator.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                                             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        uniformBuffers.push_back(std::move(alloc.buffer));
        uniformBuffersMemory.push_back(std::move(alloc.memory));
        auto mapResult = uniformBuffersMemory.back().mapMemory(0, bufferSize);
        uniformBuffersMapped.push_back(mapResult);
    }
}

void FrameManager::createShadowTransparencyBuffers(VulkanResourceCreator& resourceCreator, const GpuMesh& mesh)
{
    const std::vector<Vertex>& vertices = mesh.getVertices();
    const std::vector<uint32_t>& indices = mesh.getIndices();
    if (vertices.empty() || indices.empty()) {
        throw std::runtime_error("mesh data is empty, cannot create shadow transparency buffers");
    }

    std::vector<glm::vec2> texCoords(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        texCoords[i] = vertices[i].texCoord;
    }

    const vk::DeviceSize uvBufferSize = sizeof(glm::vec2) * texCoords.size();
    const vk::DeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();

    BufferAllocation uvStaging = resourceCreator.createBuffer(
        uvBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* uvStagingPtr = uvStaging.memory.mapMemory(0, uvBufferSize);
    std::memcpy(uvStagingPtr, texCoords.data(), static_cast<size_t>(uvBufferSize));
    uvStaging.memory.unmapMemory();

    BufferAllocation uvDevice = resourceCreator.createBuffer(
        uvBufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*uvStaging.buffer, *uvDevice.buffer, uvBufferSize);

    shadowVertexUvBuffer = std::move(uvDevice.buffer);
    shadowVertexUvMemory = std::move(uvDevice.memory);

    BufferAllocation indexStaging = resourceCreator.createBuffer(
        indexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* indexStagingPtr = indexStaging.memory.mapMemory(0, indexBufferSize);
    std::memcpy(indexStagingPtr, indices.data(), static_cast<size_t>(indexBufferSize));
    indexStaging.memory.unmapMemory();

    BufferAllocation indexDevice = resourceCreator.createBuffer(
        indexBufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*indexStaging.buffer, *indexDevice.buffer, indexBufferSize);

    shadowIndexBuffer = std::move(indexDevice.buffer);
    shadowIndexMemory = std::move(indexDevice.memory);
}

void FrameManager::createDescriptorPool(vk::raii::Device& device)
{
    std::array<vk::DescriptorPoolSize, 4> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[2].type = vk::DescriptorType::eAccelerationStructureKHR;
    poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[3].type = vk::DescriptorType::eStorageBuffer;
    poolSizes[3].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 2);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void FrameManager::createDescriptorSets(vk::raii::Device& device, GraphicsPipeline& pipeline, const GpuMesh& mesh, Texture& texture,
                                        RayTracingContext& rayTracingContext)
{
    (void)mesh;
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, pipeline.getDescriptorSetLayout());
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    const vk::AccelerationStructureKHR topLevelAS = rayTracingContext.getTopLevelAS();
    if (!topLevelAS) {
        throw std::runtime_error("ray tracing TLAS is not initialized");
    }
    if (!shadowVertexUvBuffer || !shadowIndexBuffer) {
        throw std::runtime_error("shadow transparency buffers are not initialized");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = *uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(PBRUniformBufferObject);

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo.imageView = texture.getImageView();
        imageInfo.sampler = texture.getSampler();

        vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
        accelInfo.accelerationStructureCount = 1;
        accelInfo.pAccelerationStructures = &topLevelAS;

        vk::DescriptorBufferInfo shadowUvInfo{};
        shadowUvInfo.buffer = *shadowVertexUvBuffer;
        shadowUvInfo.offset = 0;
        shadowUvInfo.range = VK_WHOLE_SIZE;

        vk::DescriptorBufferInfo shadowIndexInfo{};
        shadowIndexInfo.buffer = *shadowIndexBuffer;
        shadowIndexInfo.offset = 0;
        shadowIndexInfo.range = VK_WHOLE_SIZE;

        std::array<vk::WriteDescriptorSet, 5> descriptorWrites{};
        descriptorWrites[0].dstSet = (*descriptorSets)[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].dstSet = (*descriptorSets)[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        descriptorWrites[2].dstSet = (*descriptorSets)[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pNext = &accelInfo;

        descriptorWrites[3].dstSet = (*descriptorSets)[i];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].dstArrayElement = 0;
        descriptorWrites[3].descriptorType = vk::DescriptorType::eStorageBuffer;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].pBufferInfo = &shadowUvInfo;

        descriptorWrites[4].dstSet = (*descriptorSets)[i];
        descriptorWrites[4].dstBinding = 4;
        descriptorWrites[4].dstArrayElement = 0;
        descriptorWrites[4].descriptorType = vk::DescriptorType::eStorageBuffer;
        descriptorWrites[4].descriptorCount = 1;
        descriptorWrites[4].pBufferInfo = &shadowIndexInfo;

        device.updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void FrameManager::cleanupSwapChainResources(vk::raii::Device& device)
{
    (void)device;
    if (!uniformBuffersMapped.empty()) {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffersMemory[i].unmapMemory();
        }
        uniformBuffersMapped.clear();
    }
    uniformBuffers.clear();
    uniformBuffersMemory.clear();
    shadowVertexUvBuffer.reset();
    shadowVertexUvMemory.reset();
    shadowIndexBuffer.reset();
    shadowIndexMemory.reset();

    descriptorSets.reset();
    descriptorPool.reset();

    commandBuffers.reset();
}

