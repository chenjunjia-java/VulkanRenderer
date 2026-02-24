#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/pipeline/GraphicsPipeline.h"
#include "Resource/shader/Shader.h"

#include <optional>
#include <array>

class DepthPrepassPipeline {
public:
    DepthPrepassPipeline() = default;

    void init(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
              const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader);
    void cleanup();
    void recreate(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                  const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader);

    // 0 = backface cull (single-sided), 1 = no cull (double-sided)
    vk::Pipeline getPipeline(bool doubleSided) const
    {
        const size_t idx = doubleSided ? 1u : 0u;
        return pipelines[idx] ? static_cast<vk::Pipeline>(*pipelines[idx]) : vk::Pipeline{};
    }

private:
    void createPipelines(vk::raii::Device& device, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                        vk::SampleCountFlagBits msaaSamples, vk::PipelineLayout pipelineLayout,
                        Shader& vertShader, Shader& fragShader);

    std::array<std::optional<vk::raii::Pipeline>, 2> pipelines{};
    vk::Format depthFormat = vk::Format::eUndefined;
};

