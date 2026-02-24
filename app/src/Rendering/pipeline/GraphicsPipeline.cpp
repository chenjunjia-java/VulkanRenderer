#include "Rendering/pipeline/GraphicsPipeline.h"

#include <array>

#include "Configs/AppConfig.h"
#include "Resource/model/Vertex.h"

namespace {
uint32_t pipelineVariantIndex(bool enableBlend, bool doubleSided)
{
    const uint32_t blendBit = enableBlend ? 1u : 0u;
    const uint32_t sideBit = doubleSided ? 1u : 0u;
    // 0..3 mapping described in header.
    return (blendBit << 1u) | sideBit;
}
} // namespace

void GraphicsPipeline::init(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                            Shader& vertShader, Shader& fragShader, vk::Format targetColorFormat)
{
    createDescriptorSetLayout(context.getDevice());
    createGraphicsPipeline(context.getDevice(), swapChain, resourceCreator, context.getMsaaSamples(), vertShader, fragShader, targetColorFormat);
}

void GraphicsPipeline::cleanup()
{
    for (auto& p : pipelines) {
        p.reset();
    }
    pipelineLayout.reset();
    descriptorSetLayout.reset();
}

void GraphicsPipeline::recreate(VulkanContext& context, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                Shader& vertShader, Shader& fragShader, vk::Format targetColorFormat)
{
    for (auto& p : pipelines) {
        p.reset();
    }
    pipelineLayout.reset();
    descriptorSetLayout.reset();

    createDescriptorSetLayout(context.getDevice());
    createGraphicsPipeline(context.getDevice(), swapChain, resourceCreator, context.getMsaaSamples(), vertShader, fragShader, targetColorFormat);
}

vk::Pipeline GraphicsPipeline::getPipeline(bool enableBlend, bool doubleSided) const
{
    const uint32_t idx = pipelineVariantIndex(enableBlend, doubleSided);
    if (!pipelines[idx]) {
        return vk::Pipeline{};
    }
    return static_cast<vk::Pipeline>(*pipelines[idx]);
}

void GraphicsPipeline::createDescriptorSetLayout(vk::raii::Device& device)
{
    vk::DescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;

    auto makeSamplerBinding = [](uint32_t binding) {
        vk::DescriptorSetLayoutBinding b{};
        b.binding = binding;
        b.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        b.descriptorCount = 1;
        b.stageFlags = vk::ShaderStageFlagBits::eFragment;
        return b;
    };

    vk::DescriptorSetLayoutBinding baseColorBinding = makeSamplerBinding(1);
    vk::DescriptorSetLayoutBinding metallicRoughnessBinding = makeSamplerBinding(2);
    vk::DescriptorSetLayoutBinding normalBinding = makeSamplerBinding(3);
    vk::DescriptorSetLayoutBinding occlusionBinding = makeSamplerBinding(4);
    vk::DescriptorSetLayoutBinding emissiveBinding = makeSamplerBinding(5);

    vk::DescriptorSetLayoutBinding accelerationStructureBinding{};
    accelerationStructureBinding.binding = 6;
    accelerationStructureBinding.descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
    accelerationStructureBinding.descriptorCount = 1;
    accelerationStructureBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding instanceLUTBinding{};
    instanceLUTBinding.binding = 7;
    instanceLUTBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    instanceLUTBinding.descriptorCount = 1;
    instanceLUTBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding indexBufferBinding{};
    indexBufferBinding.binding = 8;
    indexBufferBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    indexBufferBinding.descriptorCount = 1;
    indexBufferBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding uvBufferBinding{};
    uvBufferBinding.binding = 9;
    uvBufferBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    uvBufferBinding.descriptorCount = 1;
    uvBufferBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding baseColorArrayBinding{};
    baseColorArrayBinding.binding = 10;
    baseColorArrayBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    baseColorArrayBinding.descriptorCount = AppConfig::MAX_REFLECTION_MATERIAL_COUNT;
    baseColorArrayBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutBinding drawDataBinding{};
    drawDataBinding.binding = 11;
    drawDataBinding.descriptorType = vk::DescriptorType::eStorageBuffer;
    drawDataBinding.descriptorCount = 1;
    drawDataBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

    vk::DescriptorSetLayoutBinding irradianceBinding = makeSamplerBinding(12);
    vk::DescriptorSetLayoutBinding prefilterBinding = makeSamplerBinding(13);
    vk::DescriptorSetLayoutBinding brdfLutBinding = makeSamplerBinding(14);
    vk::DescriptorSetLayoutBinding rtaoFullBinding = makeSamplerBinding(15);

    std::array<vk::DescriptorSetLayoutBinding, 16> bindings = {
        uboLayoutBinding,
        baseColorBinding,
        metallicRoughnessBinding,
        normalBinding,
        occlusionBinding,
        emissiveBinding,
        accelerationStructureBinding,
        instanceLUTBinding,
        indexBufferBinding,
        uvBufferBinding,
        baseColorArrayBinding,
        drawDataBinding,
        irradianceBinding,
        prefilterBinding,
        brdfLutBinding,
        rtaoFullBinding};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void GraphicsPipeline::createGraphicsPipeline(vk::raii::Device& device, SwapChain& swapChain, VulkanResourceCreator& resourceCreator,
                                             vk::SampleCountFlagBits msaaSamples, Shader& vertShader, Shader& fragShader,
                                             vk::Format targetColorFormat)
{
    colorFormat = targetColorFormat;
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

    vk::Extent2D swapChainExtent = swapChain.getExtent();
    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D{0, 0};
    scissor.extent = swapChainExtent;

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
    // Important for alpha-masked geometry:
    // When sample shading is enabled, the fragment shader can run per-sample and interpolants vary per sample.
    // With alpha-test (discard), this produces per-sample coverage patterns that show up as stippled/white flickering edges after MSAA resolve.
    // Keep it off to match the depth prepass behavior and stabilize mask edges.
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 0.0f;
    // Keep alpha-to-coverage off by default: it can produce stippled/white dotted edges on foliage.
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
        | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &colorBlendAttachment;

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    vk::DescriptorSetLayout layoutHandle = *descriptorSetLayout;
    pipelineLayoutInfo.pSetLayouts = &layoutHandle;

    vk::PushConstantRange pushRange{};
    pushRange.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
    pushRange.offset = 0;
    pushRange.size = sizeof(PBRPushConstants);
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::PipelineDepthStencilStateCreateInfo depthStencilState{};
    depthStencilState.depthTestEnable = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_TRUE;
    // This project uses a depth prepass. Using Less would reject equal-depth fragments in the forward pass.
    depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = VK_FALSE;

    vk::PipelineRenderingCreateInfoKHR renderingCreateInfo{};
    renderingCreateInfo.colorAttachmentCount = 1;
    renderingCreateInfo.pColorAttachmentFormats = &colorFormat;
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
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pDepthStencilState = &depthStencilState;
    pipelineInfo.layout = *pipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;

    // Create 4 variants (opaque/blend) x (cull/double-sided).
    for (uint32_t idx = 0; idx < pipelines.size(); ++idx) {
        const bool enableBlend = (idx & 0b10u) != 0;
        const bool doubleSided = (idx & 0b01u) != 0;

        rasterizer.cullMode = doubleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack;

        if (enableBlend) {
            colorBlendAttachment.blendEnable = VK_TRUE;
            // Premultiplied alpha blending: output.rgb is already multiplied by alpha in shader.
            colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne;
            colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd;
            colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
            colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero;
            colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
            depthStencilState.depthWriteEnable = VK_FALSE;
        } else {
            colorBlendAttachment.blendEnable = VK_FALSE;
            depthStencilState.depthWriteEnable = VK_TRUE;
        }

        pipelines[idx] = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }
}

