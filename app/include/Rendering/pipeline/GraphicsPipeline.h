#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Resource/shader/Shader.h"

#include <optional>
#include <vector>

class GraphicsPipeline {
public:
    GraphicsPipeline() = default;

    void init(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
              Shader& vertShader, Shader& fragShader);
    void cleanup();
    void recreate(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                  Shader& vertShader, Shader& fragShader);

    vk::Format getColorFormat() const { return colorFormat; }
    vk::Format getDepthFormat() const { return depthFormat; }

    vk::PipelineLayout getPipelineLayout() const { return *pipelineLayout; }
    vk::Pipeline getPipeline() const { return *graphicsPipeline; }
    vk::DescriptorSetLayout getDescriptorSetLayout() const { return *descriptorSetLayout; }

private:
    void createDescriptorSetLayout(vk::raii::Device& device);
    void createGraphicsPipeline(vk::raii::Device& device, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                vk::SampleCountFlagBits msaaSamples, Shader& vertShader, Shader& fragShader);

    std::optional<vk::raii::PipelineLayout> pipelineLayout;
    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::Pipeline> graphicsPipeline;
    vk::Format colorFormat = vk::Format::eUndefined;
    vk::Format depthFormat = vk::Format::eUndefined;
};

