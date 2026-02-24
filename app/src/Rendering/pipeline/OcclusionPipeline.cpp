#include "Rendering/pipeline/OcclusionPipeline.h"

#include <array>

void OcclusionPipeline::init(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                            const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader)
{
    createPipeline(context.getDevice(), swapChain, resourceCreator, context.getMsaaSamples(), basePipeline.getPipelineLayout(),
                   vertShader, fragShader);
}

void OcclusionPipeline::cleanup()
{
    pipeline.reset();
}

void OcclusionPipeline::recreate(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader)
{
    pipeline.reset();
    createPipeline(context.getDevice(), swapChain, resourceCreator, context.getMsaaSamples(), basePipeline.getPipelineLayout(),
                   vertShader, fragShader);
}

void OcclusionPipeline::createPipeline(vk::raii::Device& device, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                      vk::SampleCountFlagBits msaaSamples, vk::PipelineLayout pipelineLayout,
                                      Shader& vertShader, Shader& fragShader)
{
    depthFormat = resourceCreator.findDepthFormat();

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vertShader.getStage();
    vertShaderStageInfo.module = vertShader.getShaderModule();
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = fragShader.getStage();
    fragShaderStageInfo.module = fragShader.getShaderModule();
    fragShaderStageInfo.pName = "main";

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChain.getExtent().width);
    viewport.height = static_cast<float>(swapChain.getExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = swapChain.getExtent();

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;

    vk::PipelineDepthStencilStateCreateInfo depthStencilState{};
    depthStencilState.depthTestEnable = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_FALSE;
    depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.attachmentCount = 0;
    colorBlendState.pAttachments = nullptr;

    vk::PipelineRenderingCreateInfoKHR renderingCreateInfo{};
    renderingCreateInfo.colorAttachmentCount = 0;
    renderingCreateInfo.pColorAttachmentFormats = nullptr;
    renderingCreateInfo.depthAttachmentFormat = depthFormat;

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.pNext = &renderingCreateInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencilState;
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;

    pipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

