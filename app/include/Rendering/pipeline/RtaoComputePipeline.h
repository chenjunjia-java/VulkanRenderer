#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Resource/shader/Shader.h"

#include <optional>

class RtaoComputePipeline {
public:
    RtaoComputePipeline() = default;

    void init(VulkanContext& context, Shader& traceShader, Shader& atrousShader, Shader& upsampleShader);
    void cleanup();
    void recreate(VulkanContext& context, Shader& traceShader, Shader& atrousShader, Shader& upsampleShader);

    vk::Pipeline getTracePipeline() const { return tracePipeline ? static_cast<vk::Pipeline>(*tracePipeline) : vk::Pipeline{}; }
    vk::Pipeline getAtrousPipeline() const { return atrousPipeline ? static_cast<vk::Pipeline>(*atrousPipeline) : vk::Pipeline{}; }
    vk::Pipeline getUpsamplePipeline() const { return upsamplePipeline ? static_cast<vk::Pipeline>(*upsamplePipeline) : vk::Pipeline{}; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout ? static_cast<vk::PipelineLayout>(*pipelineLayout) : vk::PipelineLayout{}; }
    vk::DescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout ? static_cast<vk::DescriptorSetLayout>(*descriptorSetLayout) : vk::DescriptorSetLayout{}; }

private:
    void createDescriptorSetLayout(vk::raii::Device& device);
    void createPipelineLayout(vk::raii::Device& device);
    void createPipelines(vk::raii::Device& device, Shader& traceShader, Shader& atrousShader, Shader& upsampleShader);

    std::optional<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::optional<vk::raii::PipelineLayout> pipelineLayout;
    std::optional<vk::raii::Pipeline> tracePipeline;
    std::optional<vk::raii::Pipeline> atrousPipeline;
    std::optional<vk::raii::Pipeline> upsamplePipeline;
};

