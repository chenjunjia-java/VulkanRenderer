#pragma once

#include "Configs/AppConfig.h"
#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/RayTracingContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/GraphicsPipeline.h"
#include "Resource/model/Model.h"
#include "Engine/Camera/Camera.h"

#include <array>
#include <optional>
#include <vector>

class GlobalMeshBuffer;

class FrameManager {
public:
    FrameManager() = default;
    struct SharedOpaqueBucketSpan {
        bool doubleSided = false;
        uint32_t matIndex = 0;
        uint32_t firstCommand = 0;
        uint32_t drawCount = 0;
    };

    enum class PostProcessSetSlot : uint32_t {
        Extract = 0,
        BlurH = 1,
        BlurV = 2,
        Tonemap = 3,
        Count = 4,
    };

    void init(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
              Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator,
              Model& model, RayTracingContext& rayTracingContext, uint32_t maxDraws);
    void recreate(VulkanContext& context, SwapChain& swapChain, GraphicsPipeline& pipeline,
                  Rendergraph& rendergraph, VulkanResourceCreator& resourceCreator, Model& model,
                  RayTracingContext& rayTracingContext, uint32_t maxDraws);
    void cleanup(vk::raii::Device& device);

    void updateUniformBuffer(uint32_t currentImage, vk::Extent2D swapChainExtent, const Camera& camera,
                             const glm::mat4& modelMatrix);

    void setFramebufferResized(bool resized) { framebufferResized = resized; }
    bool getFramebufferResized() const { return framebufferResized; }
    void clearFramebufferResized() { framebufferResized = false; }

    vk::raii::CommandBuffers& getCommandBuffers() { return *commandBuffers; }
    const vk::raii::CommandBuffers& getCommandBuffers() const { return *commandBuffers; }
    const vk::raii::DescriptorSets& getDescriptorSets() const { return *descriptorSets; }
    vk::DescriptorSet getDescriptorSet(uint32_t frameIndex, uint32_t materialIndex) const;
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayoutHandle; }
    vk::Extent2D getSwapChainExtent() const { return swapChainExtent; }
    vk::Buffer getUniformBuffer(uint32_t frameIndex) const;
    uint32_t getCurrentFrame() const { return currentFrame; }
    void advanceFrame() { currentFrame = (currentFrame + 1) % AppConfig::MAX_FRAMES_IN_FLIGHT; }

    vk::Semaphore getRenderFinishedSemaphore(uint32_t imageIndex) const { return *renderFinishedSemaphores[imageIndex]; }
    vk::Fence getInFlightFence() const { return *inFlightFences[currentFrame]; }
    vk::Fence getImageAvailableFence() const { return *imageAvailableFence; }

    void* getDrawDataMapped(uint32_t frameIndex) const;
    vk::Buffer getDrawDataBuffer(uint32_t frameIndex) const;
    void* getIndirectCommandsMapped(uint32_t frameIndex) const;
    vk::Buffer getIndirectCommandsBuffer(uint32_t frameIndex) const;
    uint32_t getMaxDraws() const { return maxDraws; }
    void prepareSharedOpaqueIndirect(const Model& model, const GlobalMeshBuffer& globalMeshBuffer, const glm::mat4& modelMatrix);
    const std::vector<glm::mat4>& getSharedNodeWorldMatrices() const { return sharedNodeWorldMatrices; }
    const std::vector<SharedOpaqueBucketSpan>& getSharedOpaqueBucketSpans() const { return sharedOpaqueBucketSpans; }
    uint32_t getSharedOpaqueDrawCount() const { return sharedOpaqueDrawCount; }

    void createSkyboxResources(VulkanResourceCreator& resourceCreator, vk::DescriptorSetLayout skyboxLayout,
                               vk::ImageView envCubeView, vk::Sampler envCubeSampler);
    void createPostProcessResources(vk::raii::Device& device, vk::DescriptorSetLayout postLayout);
    void setIblResources(vk::raii::Device& device, vk::ImageView irradianceView, vk::ImageView prefilterView,
                        vk::ImageView brdfLutView, vk::Sampler iblSampler);
    void updateSkyboxDescriptorBuffers(vk::raii::Device& device);
    void updatePostProcessDescriptorSet(uint32_t frameIndex, PostProcessSetSlot slot, vk::ImageView sourceView, vk::ImageView bloomView);
    vk::DescriptorSet getSkyboxDescriptorSet(uint32_t imageIndex) const;
    vk::DescriptorSet getPostProcessDescriptorSet(uint32_t frameIndex, PostProcessSetSlot slot) const;
    vk::Buffer getSkyboxVertexBuffer() const;
    vk::Format getRtaoFormat() const { return rtaoFormat; }
    vk::ImageView getDepthResolveImageView() const;
    vk::Image getDepthResolveImage() const;
    vk::Sampler getDepthResolveSampler() const;
    vk::Format getDepthResolveFormat() const { return depthResolveFormat; }
    vk::ImageView getNormalPrepassImageView() const;
    vk::ImageView getNormalResolveImageView() const;
    vk::Image getNormalResolveImage() const;
    vk::Sampler getNormalResolveSampler() const;
    vk::Format getNormalFormat() const { return normalFormat; }
    vk::ImageView getLinearDepthPrepassImageView() const;
    vk::ImageView getLinearDepthResolveImageView() const;
    vk::Image getLinearDepthResolveImage() const;
    vk::Sampler getLinearDepthResolveSampler() const;
    vk::Format getLinearDepthFormat() const { return linearDepthFormat; }
    vk::ImageView getRtaoHalfHistoryImageViewForFrame(uint32_t frameIndex, bool previous) const;
    vk::Image getRtaoHalfHistoryImageForFrame(uint32_t frameIndex, bool previous) const;
    vk::Sampler getRtaoHalfHistorySampler() const;
    vk::ImageView getRtaoAtrousImageView(uint32_t pingPongIndex) const;
    vk::Image getRtaoAtrousImage(uint32_t pingPongIndex) const;
    vk::Sampler getRtaoAtrousSampler() const;
    vk::ImageView getRtaoFullImageView() const;
    vk::Image getRtaoFullImage() const;
    vk::Sampler getRtaoFullSampler() const;

    vk::Buffer getReflectionInstanceLUTBuffer() const;
    vk::Buffer getReflectionIndexBuffer() const;
    vk::Buffer getReflectionUVBuffer() const;
    vk::Buffer getReflectionMaterialParamsBuffer() const;
    const std::array<vk::DescriptorImageInfo, AppConfig::MAX_REFLECTION_MATERIAL_COUNT>& getReflectionBaseColorArrayInfos() const;

private:
    struct GpuTexture {
        std::optional<vk::raii::Image> image;
        std::optional<vk::raii::DeviceMemory> memory;
        std::optional<vk::raii::ImageView> view;
        std::optional<vk::raii::Sampler> sampler;
    };

    void createCommandBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator, SwapChain& swapChain);
    void createSyncObjects(vk::raii::Device& device, SwapChain& swapChain);
    void createUniformBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator);
    void createDrawDataBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator);
    void createIndirectCommandBuffers(vk::raii::Device& device, VulkanResourceCreator& resourceCreator);
    void createDefaultPbrTextures(VulkanResourceCreator& resourceCreator);
    void createDefaultIblTextures(VulkanResourceCreator& resourceCreator);
    void createDepthResolveTexture(VulkanResourceCreator& resourceCreator);
    void createNormalTextures(VulkanResourceCreator& resourceCreator, vk::SampleCountFlagBits msaaSamples);
    void createLinearDepthTextures(VulkanResourceCreator& resourceCreator, vk::SampleCountFlagBits msaaSamples);
    void createRtaoComputeTextures(VulkanResourceCreator& resourceCreator);
    void createDescriptorPool(vk::raii::Device& device);
    void createDescriptorSets(vk::raii::Device& device, VulkanResourceCreator& resourceCreator,
                              GraphicsPipeline& pipeline, const Model& model, RayTracingContext& rayTracingContext);
    void createReflectionBuffers(VulkanResourceCreator& resourceCreator, const Model& model);
    void createSkyboxVertexBuffer(VulkanResourceCreator& resourceCreator);
    void buildSharedDrawSlots(const Model& model);

    void cleanupSwapChainResources(vk::raii::Device& device);
    void setPbrLights(PBRUniformBufferObject& ubo);

    vk::raii::CommandPool* commandPoolPtr = nullptr;
    vk::raii::Device* devicePtr = nullptr;
    std::optional<vk::raii::CommandBuffers> commandBuffers;

    std::optional<vk::raii::DescriptorPool> descriptorPool;
    std::optional<vk::raii::DescriptorSets> descriptorSets;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    GpuTexture defaultBaseColor;
    GpuTexture defaultMetallicRoughness;
    GpuTexture defaultNormal;
    GpuTexture defaultOcclusion;
    GpuTexture defaultEmissive;
    GpuTexture defaultIblIrradiance;
    GpuTexture defaultIblPrefilter;
    GpuTexture defaultIblBrdf;
    GpuTexture depthResolve;
    vk::Format depthResolveFormat = vk::Format::eUndefined;
    GpuTexture normalPrepass;
    GpuTexture normalResolve;
    vk::Format normalFormat = vk::Format::eUndefined;
    GpuTexture linearDepthPrepass;
    GpuTexture linearDepthResolve;
    vk::Format linearDepthFormat = vk::Format::eUndefined;
    std::array<GpuTexture, 2> rtaoHalfHistory{};
    std::array<GpuTexture, 2> rtaoAtrousPingPong{};
    GpuTexture rtaoFull;
    vk::Format rtaoFormat = vk::Format::eUndefined;

    std::optional<vk::raii::Fence> imageAvailableFence;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    uint32_t currentFrame = 0;
    bool framebufferResized = false;
    uint32_t materialCount = 1;
    uint32_t maxDraws = 1;

    std::vector<vk::raii::Buffer> drawDataBuffers;
    std::vector<vk::raii::DeviceMemory> drawDataBuffersMemory;
    std::vector<void*> drawDataBuffersMapped;

    std::vector<vk::raii::Buffer> indirectCommandBuffers;
    std::vector<vk::raii::DeviceMemory> indirectCommandBuffersMemory;
    std::vector<void*> indirectCommandBuffersMapped;
    struct SharedOpaqueDrawSlot {
        uint32_t nodeLinearIndex = 0;
        uint32_t meshIndex = 0;
        uint32_t matIndex = 0;
        bool doubleSided = false;
    };
    std::vector<SharedOpaqueDrawSlot> sharedOpaqueSlots;
    std::vector<glm::mat4> sharedNodeWorldMatrices;
    std::vector<SharedOpaqueBucketSpan> sharedOpaqueBucketSpans;
    uint32_t sharedOpaqueDrawCount = 0;

    // 光追反射：Instance LUT + 合并 index/UV buffer（教程 Task 9/10/11）
    std::optional<vk::raii::Buffer> instanceLUTBuffer;
    std::optional<vk::raii::DeviceMemory> instanceLUTMemory;
    std::optional<vk::raii::Buffer> reflectionIndexBuffer;
    std::optional<vk::raii::DeviceMemory> reflectionIndexMemory;
    std::optional<vk::raii::Buffer> reflectionUVBuffer;
    std::optional<vk::raii::DeviceMemory> reflectionUVMemory;
    std::optional<vk::raii::Buffer> reflectionMaterialParamsBuffer;
    std::optional<vk::raii::DeviceMemory> reflectionMaterialParamsMemory;
    std::array<vk::DescriptorImageInfo, AppConfig::MAX_REFLECTION_MATERIAL_COUNT> reflectionBaseColorArrayInfos{};
    uint32_t reflectionMeshCount = 0;

    vk::PipelineLayout pipelineLayoutHandle = nullptr;
    vk::Extent2D swapChainExtent{};

    std::optional<vk::raii::DescriptorPool> skyboxDescriptorPool;
    std::optional<vk::raii::DescriptorSets> skyboxDescriptorSets;
    std::optional<vk::raii::DescriptorPool> postDescriptorPool;
    std::optional<vk::raii::DescriptorSets> postDescriptorSets;
    std::optional<vk::raii::Sampler> postSampler;
    std::optional<vk::raii::Buffer> skyboxVertexBuffer;
    std::optional<vk::raii::DeviceMemory> skyboxVertexBufferMemory;

    glm::mat4 lastViewProj{1.0f};
    uint32_t uniformFrameIndex = 0;
};

