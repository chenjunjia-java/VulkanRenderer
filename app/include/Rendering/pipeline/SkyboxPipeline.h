#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"

#include <optional>
#include <vulkan/vulkan.hpp>

class Shader;

class SkyboxPipeline {
public:
    SkyboxPipeline() = default;

    void init(vk::raii::Device& device, VulkanResourceCreator& resourceCreator, vk::Format colorFormat, vk::Format depthFormat,
              vk::SampleCountFlagBits msaaSamples, Shader& vertShader, Shader& fragShader);
    void cleanup();

    vk::Pipeline getPipeline() const { return pipeline ? static_cast<vk::Pipeline>(*pipeline) : vk::Pipeline{}; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout ? static_cast<vk::PipelineLayout>(*pipelineLayout) : vk::PipelineLayout{}; }
    vk::DescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout ? static_cast<vk::DescriptorSetLayout>(*descriptorSetLayout) : vk::DescriptorSetLayout{}; }

private:
    void createDescriptorSetLayout(vk::raii::Device& device);
    void createPipeline(vk::raii::Device& device, VulkanResourceCreator& resourceCreator, vk::Format colorFormat, vk::Format depthFormat,
                        vk::SampleCountFlagBits msaaSamples, Shader& vertShader, Shader& fragShader);

    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::PipelineLayout> pipelineLayout;
    std::optional<vk::raii::Pipeline> pipeline;
};
