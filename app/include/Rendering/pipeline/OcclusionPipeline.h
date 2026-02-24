#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/pipeline/GraphicsPipeline.h"
#include "Resource/shader/Shader.h"

#include <optional>

class OcclusionPipeline {
public:
    OcclusionPipeline() = default;

    void init(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
              const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader);
    void cleanup();
    void recreate(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                  const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader);

    vk::Pipeline getPipeline() const { return pipeline ? static_cast<vk::Pipeline>(*pipeline) : vk::Pipeline{}; }

private:
    void createPipeline(vk::raii::Device& device, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                        vk::SampleCountFlagBits msaaSamples, vk::PipelineLayout pipelineLayout,
                        Shader& vertShader, Shader& fragShader);

    std::optional<vk::raii::Pipeline> pipeline;
    vk::Format depthFormat = vk::Format::eUndefined;
};

