#include "Rendering/pipeline/RtaoComputePipeline.h"

#include "Configs/AppConfig.h"

#include <array>

void RtaoComputePipeline::init(VulkanContext& context, Shader& traceShader, Shader& atrousShader, Shader& upsampleShader)
{
    vk::raii::Device& device = context.getDevice();
    createDescriptorSetLayout(device);
    createPipelineLayout(device);
    createPipelines(device, traceShader, atrousShader, upsampleShader);
}

void RtaoComputePipeline::cleanup()
{
    tracePipeline.reset();
    atrousPipeline.reset();
    upsamplePipeline.reset();
    pipelineLayout.reset();
    descriptorSetLayout.reset();
}

void RtaoComputePipeline::recreate(VulkanContext& context, Shader& traceShader, Shader& atrousShader, Shader& upsampleShader)
{
    cleanup();
    init(context, traceShader, atrousShader, upsampleShader);
}

void RtaoComputePipeline::createDescriptorSetLayout(vk::raii::Device& device)
{
    std::array<vk::DescriptorSetLayoutBinding, 18> bindings{};

    bindings[0] = vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
    bindings[1] = vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr}; // depth resolve
    bindings[2] = vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr}; // normal resolve
    bindings[3] = vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr}; // half history prev (temporal)
    bindings[4] = vk::DescriptorSetLayoutBinding{4, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr};         // half history curr out
    bindings[5] = vk::DescriptorSetLayoutBinding{5, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr}; // half history curr sampled (atrous iter0)
    bindings[6] = vk::DescriptorSetLayoutBinding{6, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr}; // atrous ping0 sampled
    bindings[7] = vk::DescriptorSetLayoutBinding{7, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr}; // atrous ping1 sampled
    bindings[8] = vk::DescriptorSetLayoutBinding{8, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr};         // atrous ping0 out
    bindings[9] = vk::DescriptorSetLayoutBinding{9, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr};         // atrous ping1 out
    bindings[10] = vk::DescriptorSetLayoutBinding{10, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr};       // full-res out
    bindings[11] = vk::DescriptorSetLayoutBinding{11, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
    // Alpha-test support (reuse reflection data):
    bindings[12] = vk::DescriptorSetLayoutBinding{12, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};       // InstanceLUT (meshIndex -> materialID/indexOffset)
    bindings[13] = vk::DescriptorSetLayoutBinding{13, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};       // combined index buffer (uint indices[])
    bindings[14] = vk::DescriptorSetLayoutBinding{14, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};       // combined UV buffer (vec2 uvs[])
    bindings[15] = vk::DescriptorSetLayoutBinding{15, vk::DescriptorType::eCombinedImageSampler, AppConfig::MAX_REFLECTION_MATERIAL_COUNT,
                                                  vk::ShaderStageFlagBits::eCompute, nullptr};                                                  // baseColor texture array
    bindings[16] = vk::DescriptorSetLayoutBinding{16, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};       // material alpha params (vec4[mat])
    bindings[17] = vk::DescriptorSetLayoutBinding{17, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute, nullptr}; // linear depth resolve

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void RtaoComputePipeline::createPipelineLayout(vk::raii::Device& device)
{
    vk::PushConstantRange pushRange{};
    pushRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 4;

    vk::PipelineLayoutCreateInfo layoutInfo{};
    vk::DescriptorSetLayout setLayout = static_cast<vk::DescriptorSetLayout>(*descriptorSetLayout);
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    pipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);
}

void RtaoComputePipeline::createPipelines(vk::raii::Device& device, Shader& traceShader, Shader& atrousShader, Shader& upsampleShader)
{
    auto createOne = [&](Shader& shader) -> vk::raii::Pipeline {
        vk::PipelineShaderStageCreateInfo stageInfo{};
        stageInfo.stage = shader.getStage();
        stageInfo.module = shader.getShaderModule();
        stageInfo.pName = "main";

        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = static_cast<vk::PipelineLayout>(*pipelineLayout);
        return vk::raii::Pipeline(device, nullptr, pipelineInfo);
    };

    tracePipeline = createOne(traceShader);
    atrousPipeline = createOne(atrousShader);
    upsamplePipeline = createOne(upsampleShader);
}

