#include "Rendering/pipeline/PostProcessPipeline.h"

#include <array>

void PostProcessPipeline::init(VulkanContext& context, VulkanResourceCreator& resourceCreator, vk::Format hdrColorFormat, vk::Format swapchainColorFormat,
                               Shader& fullscreenVertShader, Shader& bloomExtractFragShader, Shader& bloomBlurFragShader,
                               Shader& tonemapBloomFragShader)
{
    (void)resourceCreator;
    createDescriptorSetLayout(context.getDevice());
    createPipelines(context.getDevice(), hdrColorFormat, swapchainColorFormat, fullscreenVertShader, bloomExtractFragShader, bloomBlurFragShader, tonemapBloomFragShader);
}

void PostProcessPipeline::recreate(VulkanContext& context, VulkanResourceCreator& resourceCreator, vk::Format hdrColorFormat, vk::Format swapchainColorFormat,
                                   Shader& fullscreenVertShader, Shader& bloomExtractFragShader, Shader& bloomBlurFragShader,
                                   Shader& tonemapBloomFragShader)
{
    (void)resourceCreator;
    cleanup();
    createDescriptorSetLayout(context.getDevice());
    createPipelines(context.getDevice(), hdrColorFormat, swapchainColorFormat, fullscreenVertShader, bloomExtractFragShader, bloomBlurFragShader, tonemapBloomFragShader);
}

void PostProcessPipeline::cleanup()
{
    for (auto& p : pipelines) {
        p.reset();
    }
    pipelineLayout.reset();
    descriptorSetLayout.reset();
}

vk::Pipeline PostProcessPipeline::getPipeline(Mode mode) const
{
    const size_t idx = static_cast<size_t>(mode);
    return pipelines[idx] ? static_cast<vk::Pipeline>(*pipelines[idx]) : vk::Pipeline{};
}

void PostProcessPipeline::createDescriptorSetLayout(vk::raii::Device& device)
{
    vk::DescriptorSetLayoutBinding srcBinding{};
    srcBinding.binding = 0;
    srcBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    srcBinding.descriptorCount = 1;
    srcBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding auxBinding{};
    auxBinding.binding = 1;
    auxBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    auxBinding.descriptorCount = 1;
    auxBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {srcBinding, auxBinding};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    vk::PushConstantRange pushRange{};
    pushRange.stageFlags = vk::ShaderStageFlagBits::eFragment;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    vk::DescriptorSetLayout layoutHandle = static_cast<vk::DescriptorSetLayout>(*descriptorSetLayout);
    pipelineLayoutInfo.pSetLayouts = &layoutHandle;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);
}

void PostProcessPipeline::createPipelines(vk::raii::Device& device, vk::Format hdrColorFormat, vk::Format swapchainColorFormat, Shader& fullscreenVertShader,
                                          Shader& bloomExtractFragShader, Shader& bloomBlurFragShader, Shader& tonemapBloomFragShader)
{
    auto createPipelineForFrag = [&](Shader& fragShader, vk::Format colorFormat) -> vk::raii::Pipeline {
        vk::PipelineShaderStageCreateInfo vertStage{};
        vertStage.stage = fullscreenVertShader.getStage();
        vertStage.module = fullscreenVertShader.getShaderModule();
        vertStage.pName = "main";

        vk::PipelineShaderStageCreateInfo fragStage{};
        fragStage.stage = fragShader.getStage();
        fragStage.module = fragShader.getShaderModule();
        fragStage.pName = "main";

        std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {vertStage, fragStage};

        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        // Fullscreen triangle: avoid winding issues under Vulkan viewport conventions.
        rasterizer.cullMode = vk::CullModeFlagBits::eNone;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.lineWidth = 1.0f;

        vk::PipelineMultisampleStateCreateInfo multisample{};
        multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = VK_FALSE;

        vk::PipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &colorBlendAttachment;

        std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        vk::PipelineRenderingCreateInfoKHR renderingInfo{};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisample;
        pipelineInfo.pColorBlendState = &colorBlend;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = static_cast<vk::PipelineLayout>(*pipelineLayout);

        return vk::raii::Pipeline(device, nullptr, pipelineInfo);
    };

    pipelines[static_cast<size_t>(Mode::Extract)] = createPipelineForFrag(bloomExtractFragShader, hdrColorFormat);
    pipelines[static_cast<size_t>(Mode::Blur)] = createPipelineForFrag(bloomBlurFragShader, hdrColorFormat);
    pipelines[static_cast<size_t>(Mode::Tonemap)] = createPipelineForFrag(tonemapBloomFragShader, swapchainColorFormat);
}

