#include "Rendering/pipeline/DepthPrepassPipeline.h"

#include "Resource/model/Vertex.h"

#include <array>

void DepthPrepassPipeline::init(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader)
{
    createPipelines(context.getDevice(), swapChain, resourceCreator, context.getMsaaSamples(), basePipeline.getPipelineLayout(),
                    vertShader, fragShader);
}

void DepthPrepassPipeline::cleanup()
{
    for (auto& p : pipelines) {
        p.reset();
    }
}

void DepthPrepassPipeline::recreate(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                    const GraphicsPipeline& basePipeline, Shader& vertShader, Shader& fragShader)
{
    for (auto& p : pipelines) {
        p.reset();
    }
    createPipelines(context.getDevice(), swapChain, resourceCreator, context.getMsaaSamples(), basePipeline.getPipelineLayout(),
                    vertShader, fragShader);
}

void DepthPrepassPipeline::createPipelines(vk::raii::Device& device, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                          vk::SampleCountFlagBits msaaSamples, vk::PipelineLayout pipelineLayout,
                                          Shader& vertShader, Shader& fragShader)
{
    depthFormat = resourceCreator.findDepthFormat();
    const vk::Format normalFormat = vk::Format::eR16G16B16A16Sfloat;
    const vk::Format linearDepthFormat = vk::Format::eR16Sfloat;

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vertShader.getStage();
    vertShaderStageInfo.module = vertShader.getShaderModule();
    vertShaderStageInfo.pName = "main";

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = fragShader.getStage();
    fragShaderStageInfo.module = fragShader.getShaderModule();
    fragShaderStageInfo.pName = "main";

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

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
    depthStencilState.depthWriteEnable = VK_TRUE;
    depthStencilState.depthCompareOp = vk::CompareOp::eLess;
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR
                                        | vk::ColorComponentFlagBits::eG
                                        | vk::ColorComponentFlagBits::eB
                                        | vk::ColorComponentFlagBits::eA;
    std::array<vk::PipelineColorBlendAttachmentState, 2> colorBlendAttachments = {colorBlendAttachment, colorBlendAttachment};
    vk::PipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
    colorBlendState.pAttachments = colorBlendAttachments.data();

    vk::PipelineRenderingCreateInfoKHR renderingCreateInfo{};
    std::array<vk::Format, 2> colorFormats = {normalFormat, linearDepthFormat};
    renderingCreateInfo.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
    renderingCreateInfo.pColorAttachmentFormats = colorFormats.data();
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

    // Variant 0: single-sided (backface cull)
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    pipelines[0] = vk::raii::Pipeline(device, nullptr, pipelineInfo);

    // Variant 1: double-sided (no cull) to match forward pass for doubleSided materials.
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    pipelines[1] = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

