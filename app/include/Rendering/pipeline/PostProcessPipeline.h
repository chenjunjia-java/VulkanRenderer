#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Resource/shader/Shader.h"

#include <array>
#include <glm/vec4.hpp>
#include <optional>

class PostProcessPipeline {
public:
    enum class Mode : uint32_t {
        Extract = 0,
        Blur = 1,
        Tonemap = 2,
        Count = 3,
    };

    struct PushConstants {
        alignas(16) glm::vec4 params0{0.0f};  // x=threshold, y=softKnee, z=intensity
        alignas(16) glm::vec4 params1{0.0f};  // x=invWidth, y=invHeight, z=dirX, w=dirY
    };

    PostProcessPipeline() = default;

    void init(VulkanContext& context, VulkanResourceCreator& resourceCreator, vk::Format hdrColorFormat, vk::Format swapchainColorFormat,
              Shader& fullscreenVertShader, Shader& bloomExtractFragShader, Shader& bloomBlurFragShader,
              Shader& tonemapBloomFragShader);
    void recreate(VulkanContext& context, VulkanResourceCreator& resourceCreator, vk::Format hdrColorFormat, vk::Format swapchainColorFormat,
                  Shader& fullscreenVertShader, Shader& bloomExtractFragShader, Shader& bloomBlurFragShader,
                  Shader& tonemapBloomFragShader);
    void cleanup();

    vk::Pipeline getPipeline(Mode mode) const;
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout ? static_cast<vk::PipelineLayout>(*pipelineLayout) : vk::PipelineLayout{}; }
    vk::DescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout ? static_cast<vk::DescriptorSetLayout>(*descriptorSetLayout) : vk::DescriptorSetLayout{}; }

private:
    void createDescriptorSetLayout(vk::raii::Device& device);
    void createPipelines(vk::raii::Device& device, vk::Format hdrColorFormat, vk::Format swapchainColorFormat, Shader& fullscreenVertShader,
                         Shader& bloomExtractFragShader, Shader& bloomBlurFragShader, Shader& tonemapBloomFragShader);

    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::PipelineLayout> pipelineLayout;
    std::array<std::optional<vk::raii::Pipeline>, static_cast<size_t>(Mode::Count)> pipelines{};
};

