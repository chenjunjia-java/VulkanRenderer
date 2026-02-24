#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Resource/shader/Shader.h"

#include <optional>
#include <array>
#include <vulkan/vulkan.hpp>

class GraphicsPipeline {
public:
    GraphicsPipeline() = default;

    void init(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
              Shader& vertShader, Shader& fragShader, vk::Format targetColorFormat);
    void cleanup();
    void recreate(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                  Shader& vertShader, Shader& fragShader, vk::Format targetColorFormat);

    vk::Format getColorFormat() const { return colorFormat; }
    vk::Format getDepthFormat() const { return depthFormat; }

    vk::PipelineLayout getPipelineLayout() const { return *pipelineLayout; }
    vk::Pipeline getPipeline(bool enableBlend = false, bool doubleSided = false) const;
    vk::DescriptorSetLayout getDescriptorSetLayout() const { return *descriptorSetLayout; }

private:
    void createDescriptorSetLayout(vk::raii::Device& device);
    void createGraphicsPipeline(vk::raii::Device& device, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                vk::SampleCountFlagBits msaaSamples, Shader& vertShader, Shader& fragShader,
                                vk::Format targetColorFormat);

    std::optional<vk::raii::PipelineLayout> pipelineLayout;
    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    // Variants: 0=OpaqueCull, 1=OpaqueDoubleSided, 2=BlendCull, 3=BlendDoubleSided
    std::array<std::optional<vk::raii::Pipeline>, 4> pipelines{};
    vk::Format colorFormat = vk::Format::eUndefined;
    vk::Format depthFormat = vk::Format::eUndefined;
};

