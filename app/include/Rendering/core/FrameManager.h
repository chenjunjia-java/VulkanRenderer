#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/RayTracingContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/GraphicsPipeline.h"
#include "Rendering/mesh/GpuMesh.h"
#include "Resource/texture/Texture.h"
#include "Engine/Camera/Camera.h"

#include <optional>
#include <vector>

class FrameManager {
public:
    FrameManager() = default;

    void init(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
              Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator, GpuMesh& mesh,
              Texture& texture, RayTracingContext& rayTracingContext);
    void recreate(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
                  Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator, Texture& texture,
                  RayTracingContext& rayTracingContext);
    void cleanup(vk::raii::Device& device);

    void updateUniformBuffer(uint32_t currentImage, vk::Extent2D swapChainExtent, const Camera& camera,
                             const glm::mat4& modelMatrix);

    void setFramebufferResized(bool resized) { framebufferResized = resized; }
    bool getFramebufferResized() const { return framebufferResized; }
    void clearFramebufferResized() { framebufferResized = false; }

    vk::raii::CommandBuffers& getCommandBuffers() { return *commandBuffers; }
    const vk::raii::CommandBuffers& getCommandBuffers() const { return *commandBuffers; }
    const vk::raii::DescriptorSets& getDescriptorSets() const { return *descriptorSets; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayoutHandle; }
    vk::Extent2D getSwapChainExtent() const { return swapChainExtent; }
    uint32_t getCurrentFrame() const { return currentFrame; }
    void advanceFrame() { currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT; }

    vk::Semaphore getRenderFinishedSemaphore(uint32_t imageIndex) const { return *renderFinishedSemaphores[imageIndex]; }
    vk::Fence getInFlightFence() const { return *inFlightFences[currentFrame]; }
    vk::Fence getImageAvailableFence() const { return *imageAvailableFence; }

private:
    void createCommandBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator, SwapChain& swapChain);
    void createSyncObjects(vk::raii::Device& device, SwapChain& swapChain);
    void createUniformBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator);
    void createShadowTransparencyBuffers(VulkanResourceCreator& resourceCreator, const GpuMesh& mesh);
    void createDescriptorPool(vk::raii::Device& device);
    void createDescriptorSets(vk::raii::Device& device, GraphicsPipeline& pipeline, const GpuMesh& mesh, Texture& texture,
                              RayTracingContext& rayTracingContext);

    void cleanupSwapChainResources(vk::raii::Device& device);

    vk::raii::CommandPool* commandPoolPtr = nullptr;
    std::optional<vk::raii::CommandBuffers> commandBuffers;

    std::optional<vk::raii::DescriptorPool> descriptorPool;
    std::optional<vk::raii::DescriptorSets> descriptorSets;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    std::optional<vk::raii::Buffer> shadowVertexUvBuffer;
    std::optional<vk::raii::DeviceMemory> shadowVertexUvMemory;
    std::optional<vk::raii::Buffer> shadowIndexBuffer;
    std::optional<vk::raii::DeviceMemory> shadowIndexMemory;

    std::optional<vk::raii::Fence> imageAvailableFence;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    uint32_t currentFrame = 0;
    bool framebufferResized = false;
    GpuMesh* sceneMesh = nullptr;

    vk::PipelineLayout pipelineLayoutHandle = nullptr;
    vk::Extent2D swapChainExtent{};
};

