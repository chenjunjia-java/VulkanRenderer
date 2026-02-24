#include "Rendering/ibl/IblPrecompute.h"

#include "Configs/AppConfig.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

namespace {

const std::array<float, 108> CUBE_VERTICES = {
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
};

const std::array<glm::mat4, 6> CAPTURE_VIEWS = {
    glm::lookAt(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    // Match Vulkan cubemap face order: +Y then -Y (layer2=+Y, layer3=-Y).
    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
};

struct CaptureUniforms {
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 view;
};

struct PrefilterUniforms {
    alignas(16) glm::mat4 projection;
    alignas(4) float roughness;
};

// Fullscreen quad: pos (x,y), uv (u,v). 4 verts, stride 16.
const std::array<float, 16> QUAD_VERTICES = {
    -1.0f, -1.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f, 1.0f,
     1.0f,  1.0f, 1.0f, 1.0f,
};

std::vector<char> loadSpv(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) return {};
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> code(size);
    file.read(code.data(), static_cast<std::streamsize>(size));
    return code;
}

uint32_t mipCount(uint32_t size)
{
    return static_cast<uint32_t>(std::floor(std::log2(size))) + 1;
}

} // namespace

IblResult IblPrecompute::compute(VulkanResourceCreator& resourceCreator,
                                  vk::ImageView envCubemapView,
                                  vk::Sampler envCubemapSampler,
                                  uint32_t irradianceSize,
                                  uint32_t prefilterSize,
                                  uint32_t brdfLutSize)
{
    IblResult result;
    vk::raii::Device& device = resourceCreator.getDevice();
    std::string basePath = AppConfig::ASSETS_PATH + "shaders/";
    glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    // Match Vulkan clip space convention (same as Camera::getProjMatrix).
    projection[1][1] *= -1.0f;

    // Depth buffer for cubemap rendering (prevents backfaces/other cube sides from overwriting a face).
    // Without depth test, multiple cube triangles can overlap in screen space, producing blocky artifacts
    // that often show up only on some faces (driver-dependent).
    const vk::Format cubeDepthFormat = resourceCreator.findDepthFormat();

    // Shared cube VB
    BufferAllocation vbStaging = resourceCreator.createBuffer(
        sizeof(CUBE_VERTICES), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* vbMap = vbStaging.memory.mapMemory(0, sizeof(CUBE_VERTICES));
    std::memcpy(vbMap, CUBE_VERTICES.data(), sizeof(CUBE_VERTICES));
    vbStaging.memory.unmapMemory();

    BufferAllocation cubeVbGpu = resourceCreator.createBuffer(
        sizeof(CUBE_VERTICES), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*vbStaging.buffer, *cubeVbGpu.buffer, sizeof(CUBE_VERTICES));

    // === 1. Irradiance convolution ===
    auto irrVertCode = loadSpv(basePath + "VertShaders/cubemap_capture.vert.spv");
    auto irrFragCode = loadSpv(basePath + "FragShaders/irradiance_convolution.frag.spv");
    if (irrVertCode.empty() || irrFragCode.empty()) return result;

    vk::raii::ShaderModule irrVertModule(device, {{}, irrVertCode.size(), reinterpret_cast<const uint32_t*>(irrVertCode.data())});
    vk::raii::ShaderModule irrFragModule(device, {{}, irrFragCode.size(), reinterpret_cast<const uint32_t*>(irrFragCode.data())});

    ImageAllocation irradianceAlloc = resourceCreator.createImage(
        irradianceSize, irradianceSize, 1, vk::SampleCountFlagBits::e1,
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, 6, vk::ImageCreateFlagBits::eCubeCompatible);

    ImageAllocation irrDepthAlloc = resourceCreator.createImage(
        irradianceSize, irradianceSize, 1, vk::SampleCountFlagBits::e1,
        cubeDepthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*irradianceAlloc.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 1, 6);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*irrDepthAlloc.image), cubeDepthFormat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

    vk::ImageAspectFlags depthAspect = vk::ImageAspectFlagBits::eDepth;
    if (cubeDepthFormat == vk::Format::eD32SfloatS8Uint || cubeDepthFormat == vk::Format::eD24UnormS8Uint) {
        depthAspect |= vk::ImageAspectFlagBits::eStencil;
    }
    vk::raii::ImageView irrDepthView = resourceCreator.createImageView(
        static_cast<vk::Image>(*irrDepthAlloc.image), cubeDepthFormat, depthAspect, 1);

    vk::DescriptorSetLayoutBinding irrSamplerBinding{0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
    std::array<vk::DescriptorSetLayoutBinding, 1> irrBindings = {irrSamplerBinding};
    vk::raii::DescriptorSetLayout irrDescriptorLayout(device, {{}, 1, irrBindings.data()});

    std::array<vk::DescriptorPoolSize, 1> irrPoolSizes{{
        {vk::DescriptorType::eCombinedImageSampler, 1},
    }};
    vk::raii::DescriptorPool irrDescriptorPool(device, {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, 1, irrPoolSizes.data()});

    vk::DescriptorSetLayout irrLayoutHandle = *irrDescriptorLayout;
    vk::DescriptorSetAllocateInfo irrAllocInfo{};
    irrAllocInfo.descriptorPool = *irrDescriptorPool;
    irrAllocInfo.descriptorSetCount = 1;
    irrAllocInfo.pSetLayouts = &irrLayoutHandle;
    vk::raii::DescriptorSets irrDescriptorSets(device, irrAllocInfo);

    vk::DescriptorImageInfo irrImageInfo{envCubemapSampler, envCubemapView, vk::ImageLayout::eShaderReadOnlyOptimal};
    std::array<vk::WriteDescriptorSet, 1> irrWrites{{
        vk::WriteDescriptorSet{*irrDescriptorSets[0], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &irrImageInfo},
    }};
    device.updateDescriptorSets(irrWrites, nullptr);

    vk::VertexInputBindingDescription irrBinding{0, 12, vk::VertexInputRate::eVertex};
    vk::VertexInputAttributeDescription irrAttr{0, 0, vk::Format::eR32G32B32Sfloat, 0};
    vk::PipelineVertexInputStateCreateInfo irrVertInput{{}, 1, &irrBinding, 1, &irrAttr};

    vk::PipelineShaderStageCreateInfo irrVertStage{{}, vk::ShaderStageFlagBits::eVertex, *irrVertModule, "main"};
    vk::PipelineShaderStageCreateInfo irrFragStage{{}, vk::ShaderStageFlagBits::eFragment, *irrFragModule, "main"};
    std::array<vk::PipelineShaderStageCreateInfo, 2> irrStages = {irrVertStage, irrFragStage};

    vk::PipelineInputAssemblyStateCreateInfo irrInputAssembly{{}, vk::PrimitiveTopology::eTriangleList};
    vk::PipelineViewportStateCreateInfo irrViewport{{}, 1, nullptr, 1, nullptr};
    std::array<vk::DynamicState, 2> irrDynStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo irrDynState{{}, 2, irrDynStates.data()};
    vk::PipelineRasterizationStateCreateInfo irrRaster{{}, false, false, vk::PolygonMode::eFill};
    // Avoid winding/cull mismatches during cubemap capture/precompute.
    // If a face gets fully culled and loadOp=DontCare, that face can become undefined (often appears as 0).
    irrRaster.cullMode = vk::CullModeFlagBits::eNone;
    irrRaster.frontFace = vk::FrontFace::eCounterClockwise;
    irrRaster.lineWidth = 1.0f;
    vk::PipelineMultisampleStateCreateInfo irrMsaa{};
    irrMsaa.rasterizationSamples = vk::SampleCountFlagBits::e1;
    vk::PipelineDepthStencilStateCreateInfo irrDepthStencil{};
    irrDepthStencil.depthTestEnable = VK_TRUE;
    irrDepthStencil.depthWriteEnable = VK_TRUE;
    irrDepthStencil.depthCompareOp = vk::CompareOp::eLessOrEqual;
    irrDepthStencil.depthBoundsTestEnable = VK_FALSE;
    irrDepthStencil.stencilTestEnable = VK_FALSE;
    vk::PipelineColorBlendAttachmentState irrBlend{};
    irrBlend.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                             vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendStateCreateInfo irrBlendState{{}, false, vk::LogicOp::eCopy, 1, &irrBlend};

    vk::PushConstantRange irrPcRange{vk::ShaderStageFlagBits::eVertex, 0, sizeof(CaptureUniforms)};
    vk::PipelineLayoutCreateInfo irrPipeLayoutInfo{};
    irrPipeLayoutInfo.setLayoutCount = 1;
    irrPipeLayoutInfo.pSetLayouts = &irrLayoutHandle;
    irrPipeLayoutInfo.pushConstantRangeCount = 1;
    irrPipeLayoutInfo.pPushConstantRanges = &irrPcRange;
    vk::raii::PipelineLayout irrPipeLayout(device, irrPipeLayoutInfo);
    vk::Format irrColorFormat = vk::Format::eR16G16B16A16Sfloat;
    vk::PipelineRenderingCreateInfoKHR irrRendering{};
    irrRendering.colorAttachmentCount = 1;
    irrRendering.pColorAttachmentFormats = &irrColorFormat;
    irrRendering.depthAttachmentFormat = cubeDepthFormat;

    vk::GraphicsPipelineCreateInfo irrPipeInfo{};
    irrPipeInfo.pNext = &irrRendering;
    irrPipeInfo.stageCount = 2;
    irrPipeInfo.pStages = irrStages.data();
    irrPipeInfo.pVertexInputState = &irrVertInput;
    irrPipeInfo.pInputAssemblyState = &irrInputAssembly;
    irrPipeInfo.pViewportState = &irrViewport;
    irrPipeInfo.pRasterizationState = &irrRaster;
    irrPipeInfo.pMultisampleState = &irrMsaa;
    irrPipeInfo.pDepthStencilState = &irrDepthStencil;
    irrPipeInfo.pColorBlendState = &irrBlendState;
    irrPipeInfo.pDynamicState = &irrDynState;
    irrPipeInfo.layout = *irrPipeLayout;
    vk::raii::Pipeline irrPipeline(device, nullptr, irrPipeInfo);

    std::vector<vk::raii::ImageView> irrFaceViews;
    irrFaceViews.reserve(6);
    for (uint32_t f = 0; f < 6; ++f) {
        irrFaceViews.push_back(resourceCreator.createImageView(*irradianceAlloc.image, vk::Format::eR16G16B16A16Sfloat,
            vk::ImageAspectFlagBits::eColor, 1, vk::ImageViewType::e2D, f, 1));
    }

    resourceCreator.executeSingleTimeCommands([&](vk::raii::CommandBuffer& cb) {
        const float irrSize = static_cast<float>(irradianceSize);
        cb.setViewport(0, vk::Viewport{0.0f, irrSize, irrSize, -irrSize, 0.0f, 1.0f});
        cb.setScissor(0, vk::Rect2D{{0, 0}, {irradianceSize, irradianceSize}});
        cb.bindVertexBuffers(0, {*cubeVbGpu.buffer}, {0ull});
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *irrPipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *irrPipeLayout, 0, {*irrDescriptorSets[0]}, nullptr);

        for (uint32_t face = 0; face < 6; ++face) {
            CaptureUniforms pcData{};
            pcData.projection = projection;
            pcData.view = CAPTURE_VIEWS[face];
            vk::CommandBuffer rawCb = static_cast<vk::CommandBuffer>(cb);
            rawCb.pushConstants(*irrPipeLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(CaptureUniforms),
                                reinterpret_cast<const void*>(&pcData));

            vk::RenderingAttachmentInfoKHR colorAttachment{};
            colorAttachment.setImageView(*irrFaceViews[face])
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore);
            colorAttachment.setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

            vk::RenderingAttachmentInfoKHR depthAttachment{};
            depthAttachment.setImageView(*irrDepthView)
                .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eDontCare);
            depthAttachment.setClearValue(vk::ClearDepthStencilValue{1.0f, 0});

            vk::RenderingInfoKHR renderingInfo{};
            renderingInfo.renderArea = vk::Rect2D{{0, 0}, {irradianceSize, irradianceSize}};
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &colorAttachment;
            renderingInfo.pDepthAttachment = &depthAttachment;

            cb.beginRendering(renderingInfo);
            cb.draw(36, 1, 0, 0);
            cb.endRendering();
        }
    });

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*irradianceAlloc.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1, 6);

    result.irradianceImage = std::move(irradianceAlloc.image);
    result.irradianceMemory = std::move(irradianceAlloc.memory);
    result.irradianceView = resourceCreator.createImageView(*result.irradianceImage, vk::Format::eR16G16B16A16Sfloat,
        vk::ImageAspectFlagBits::eColor, 1, vk::ImageViewType::eCube, 0, 6);

    // === 2. Prefilter ===
    const uint32_t prefilterMipLevels = mipCount(prefilterSize);
    auto preVertCode = loadSpv(basePath + "VertShaders/prefilter_capture.vert.spv");
    auto preFragCode = loadSpv(basePath + "FragShaders/prefilter.frag.spv");
    if (preVertCode.empty() || preFragCode.empty()) return result;

    vk::raii::ShaderModule preVertModule(device, {{}, preVertCode.size(), reinterpret_cast<const uint32_t*>(preVertCode.data())});
    vk::raii::ShaderModule preFragModule(device, {{}, preFragCode.size(), reinterpret_cast<const uint32_t*>(preFragCode.data())});

    ImageAllocation prefilterAlloc = resourceCreator.createImage(
        prefilterSize, prefilterSize, prefilterMipLevels, vk::SampleCountFlagBits::e1,
        vk::Format::eR16G16B16A16Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal, 6, vk::ImageCreateFlagBits::eCubeCompatible);

    ImageAllocation preDepthAlloc = resourceCreator.createImage(
        prefilterSize, prefilterSize, 1, vk::SampleCountFlagBits::e1,
        cubeDepthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::DescriptorSetLayoutBinding preUboBinding{0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment};
    vk::DescriptorSetLayoutBinding preSamplerBinding{1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
    std::array<vk::DescriptorSetLayoutBinding, 2> preBindings = {preUboBinding, preSamplerBinding};
    vk::raii::DescriptorSetLayout preDescriptorLayout(device, {{}, 2, preBindings.data()});

    BufferAllocation preUboAlloc = resourceCreator.createBuffer(
        sizeof(PrefilterUniforms), vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* preUboMapped = preUboAlloc.memory.mapMemory(0, sizeof(PrefilterUniforms));

    std::array<vk::DescriptorPoolSize, 2> prePoolSizes{{
        {vk::DescriptorType::eUniformBuffer, 1},
        {vk::DescriptorType::eCombinedImageSampler, 1},
    }};
    vk::raii::DescriptorPool preDescriptorPool(device, {{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet}, 1, 2, prePoolSizes.data()});

    vk::DescriptorSetLayout preLayoutHandle = *preDescriptorLayout;
    vk::DescriptorSetAllocateInfo preAllocInfo{};
    preAllocInfo.descriptorPool = *preDescriptorPool;
    preAllocInfo.descriptorSetCount = 1;
    preAllocInfo.pSetLayouts = &preLayoutHandle;
    vk::raii::DescriptorSets preDescriptorSets(device, preAllocInfo);

    vk::DescriptorBufferInfo preBufferInfo{*preUboAlloc.buffer, 0, sizeof(PrefilterUniforms)};
    vk::DescriptorImageInfo preImageInfo{envCubemapSampler, envCubemapView, vk::ImageLayout::eShaderReadOnlyOptimal};
    std::array<vk::WriteDescriptorSet, 2> preWrites{{
        vk::WriteDescriptorSet{*preDescriptorSets[0], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &preBufferInfo},
        vk::WriteDescriptorSet{*preDescriptorSets[0], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &preImageInfo},
    }};
    device.updateDescriptorSets(preWrites, nullptr);

    vk::PushConstantRange prePushRange{vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4)};
    vk::PipelineLayoutCreateInfo prePipeLayoutInfo{};
    prePipeLayoutInfo.setLayoutCount = 1;
    prePipeLayoutInfo.pSetLayouts = &preLayoutHandle;
    prePipeLayoutInfo.pushConstantRangeCount = 1;
    prePipeLayoutInfo.pPushConstantRanges = &prePushRange;
    vk::raii::PipelineLayout prePipeLayout(device, prePipeLayoutInfo);

    vk::PipelineShaderStageCreateInfo preVertStage{{}, vk::ShaderStageFlagBits::eVertex, *preVertModule, "main"};
    vk::PipelineShaderStageCreateInfo preFragStage{{}, vk::ShaderStageFlagBits::eFragment, *preFragModule, "main"};
    std::array<vk::PipelineShaderStageCreateInfo, 2> preStages = {preVertStage, preFragStage};

    vk::GraphicsPipelineCreateInfo prePipeInfo{};
    prePipeInfo.pNext = &irrRendering;
    prePipeInfo.stageCount = 2;
    prePipeInfo.pStages = preStages.data();
    prePipeInfo.pVertexInputState = &irrVertInput;
    prePipeInfo.pInputAssemblyState = &irrInputAssembly;
    prePipeInfo.pViewportState = &irrViewport;
    prePipeInfo.pRasterizationState = &irrRaster;
    prePipeInfo.pMultisampleState = &irrMsaa;
    prePipeInfo.pDepthStencilState = &irrDepthStencil;
    prePipeInfo.pColorBlendState = &irrBlendState;
    prePipeInfo.pDynamicState = &irrDynState;
    prePipeInfo.layout = *prePipeLayout;
    vk::raii::Pipeline prePipeline(device, nullptr, prePipeInfo);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*prefilterAlloc.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, prefilterMipLevels, 6);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*preDepthAlloc.image), cubeDepthFormat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

    vk::raii::ImageView preDepthView = resourceCreator.createImageView(
        static_cast<vk::Image>(*preDepthAlloc.image), cubeDepthFormat, depthAspect, 1);

    for (uint32_t mip = 0; mip < prefilterMipLevels; ++mip) {
        uint32_t mipSize = prefilterSize >> mip;
        if (mipSize < 1) mipSize = 1;

        // IMPORTANT: Keep this as perceptual roughness in [0,1]. The GGX importance sampling
        // in `prefilter.frag` will square it (a = roughness^2). Squaring here would effectively
        // use roughness^4 and make reflections too sharp.
        float roughness = static_cast<float>(mip) / static_cast<float>(std::max(1u, prefilterMipLevels - 1));

        PrefilterUniforms preUbo;
        preUbo.projection = projection;
        preUbo.roughness = roughness;
        std::memcpy(preUboMapped, &preUbo, sizeof(PrefilterUniforms));

        std::vector<vk::raii::ImageView> preFaceViews;
        preFaceViews.reserve(6);
        for (uint32_t f = 0; f < 6; ++f) {
            preFaceViews.push_back(resourceCreator.createImageView(*prefilterAlloc.image, vk::Format::eR16G16B16A16Sfloat,
                vk::ImageAspectFlagBits::eColor, prefilterMipLevels, vk::ImageViewType::e2D, f, 1, mip, 1));
        }

        resourceCreator.executeSingleTimeCommands([&](vk::raii::CommandBuffer& cb) {
                const float mipSz = static_cast<float>(mipSize);
                cb.setViewport(0, vk::Viewport{0.0f, mipSz, mipSz, -mipSz, 0.0f, 1.0f});
                cb.setScissor(0, vk::Rect2D{{0, 0}, {mipSize, mipSize}});
                cb.bindVertexBuffers(0, {*cubeVbGpu.buffer}, {0ull});
                cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *prePipeline);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *prePipeLayout, 0, {*preDescriptorSets[0]}, nullptr);

            for (uint32_t face = 0; face < 6; ++face) {
                vk::CommandBuffer rawCb = static_cast<vk::CommandBuffer>(cb);
                rawCb.pushConstants(*prePipeLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4),
                                    reinterpret_cast<const void*>(&CAPTURE_VIEWS[face]));

                vk::RenderingAttachmentInfoKHR colorAttachment{};
                colorAttachment.setImageView(*preFaceViews[face])
                    .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                    .setLoadOp(vk::AttachmentLoadOp::eClear)
                    .setStoreOp(vk::AttachmentStoreOp::eStore);
                colorAttachment.setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

                vk::RenderingAttachmentInfoKHR depthAttachment{};
                depthAttachment.setImageView(*preDepthView)
                    .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
                    .setLoadOp(vk::AttachmentLoadOp::eClear)
                    .setStoreOp(vk::AttachmentStoreOp::eDontCare);
                depthAttachment.setClearValue(vk::ClearDepthStencilValue{1.0f, 0});

                vk::RenderingInfoKHR renderingInfo{};
                renderingInfo.renderArea = vk::Rect2D{{0, 0}, {mipSize, mipSize}};
                renderingInfo.layerCount = 1;
                renderingInfo.colorAttachmentCount = 1;
                renderingInfo.pColorAttachments = &colorAttachment;
                renderingInfo.pDepthAttachment = &depthAttachment;

                cb.beginRendering(renderingInfo);
                cb.draw(36, 1, 0, 0);
                cb.endRendering();
            }
        });
    }

    preUboAlloc.memory.unmapMemory();

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*prefilterAlloc.image), vk::Format::eR16G16B16A16Sfloat,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, prefilterMipLevels, 6);

    result.prefilterImage = std::move(prefilterAlloc.image);
    result.prefilterMemory = std::move(prefilterAlloc.memory);
    result.prefilterView = resourceCreator.createImageView(*result.prefilterImage, vk::Format::eR16G16B16A16Sfloat,
        vk::ImageAspectFlagBits::eColor, prefilterMipLevels, vk::ImageViewType::eCube, 0, 6);

    // === 3. BRDF LUT ===
    auto brdfVertCode = loadSpv(basePath + "VertShaders/brdf_quad.vert.spv");
    auto brdfFragCode = loadSpv(basePath + "FragShaders/brdf_integrate.frag.spv");
    if (brdfVertCode.empty() || brdfFragCode.empty()) return result;

    vk::raii::ShaderModule brdfVertModule(device, {{}, brdfVertCode.size(), reinterpret_cast<const uint32_t*>(brdfVertCode.data())});
    vk::raii::ShaderModule brdfFragModule(device, {{}, brdfFragCode.size(), reinterpret_cast<const uint32_t*>(brdfFragCode.data())});

    ImageAllocation brdfAlloc = resourceCreator.createImage(
        brdfLutSize, brdfLutSize, 1, vk::SampleCountFlagBits::e1,
        vk::Format::eR16G16Sfloat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*brdfAlloc.image), vk::Format::eR16G16Sfloat,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, 1);

    BufferAllocation brdfVbStaging = resourceCreator.createBuffer(
        sizeof(QUAD_VERTICES), vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* brdfVbMap = brdfVbStaging.memory.mapMemory(0, sizeof(QUAD_VERTICES));
    std::memcpy(brdfVbMap, QUAD_VERTICES.data(), sizeof(QUAD_VERTICES));
    brdfVbStaging.memory.unmapMemory();

    BufferAllocation brdfVbGpu = resourceCreator.createBuffer(
        sizeof(QUAD_VERTICES), vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*brdfVbStaging.buffer, *brdfVbGpu.buffer, sizeof(QUAD_VERTICES));

    vk::PipelineLayoutCreateInfo brdfPipeLayoutInfo{};
    vk::raii::PipelineLayout brdfPipeLayout(device, brdfPipeLayoutInfo);

    vk::VertexInputBindingDescription brdfBinding{0, 16, vk::VertexInputRate::eVertex};
    std::array<vk::VertexInputAttributeDescription, 2> brdfAttrs = {
        vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32Sfloat, 0},
        vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32Sfloat, 8},
    };
    vk::PipelineVertexInputStateCreateInfo brdfVertInput{{}, 1, &brdfBinding, 2, brdfAttrs.data()};

    vk::PipelineShaderStageCreateInfo brdfVertStage{{}, vk::ShaderStageFlagBits::eVertex, *brdfVertModule, "main"};
    vk::PipelineShaderStageCreateInfo brdfFragStage{{}, vk::ShaderStageFlagBits::eFragment, *brdfFragModule, "main"};
    std::array<vk::PipelineShaderStageCreateInfo, 2> brdfStages = {brdfVertStage, brdfFragStage};

    vk::Format brdfColorFormat = vk::Format::eR16G16Sfloat;
    vk::PipelineRenderingCreateInfoKHR brdfRendering{{}, 1, &brdfColorFormat};

    vk::GraphicsPipelineCreateInfo brdfPipeInfo{};
    brdfPipeInfo.pNext = &brdfRendering;
    brdfPipeInfo.stageCount = 2;
    brdfPipeInfo.pStages = brdfStages.data();
    brdfPipeInfo.pVertexInputState = &brdfVertInput;
    vk::PipelineInputAssemblyStateCreateInfo brdfInputAssembly{{}, vk::PrimitiveTopology::eTriangleStrip};
    brdfPipeInfo.pInputAssemblyState = &brdfInputAssembly;
    brdfPipeInfo.pViewportState = &irrViewport;
    brdfPipeInfo.pRasterizationState = &irrRaster;
    brdfPipeInfo.pMultisampleState = &irrMsaa;
    brdfPipeInfo.pColorBlendState = &irrBlendState;
    brdfPipeInfo.pDynamicState = &irrDynState;
    brdfPipeInfo.layout = *brdfPipeLayout;

    vk::raii::Pipeline brdfPipeline(device, nullptr, brdfPipeInfo);

    vk::raii::ImageView brdfView = resourceCreator.createImageView(*brdfAlloc.image, vk::Format::eR16G16Sfloat,
        vk::ImageAspectFlagBits::eColor, 1);

    resourceCreator.executeSingleTimeCommands([&](vk::raii::CommandBuffer& cb) {
        cb.setViewport(0, vk::Viewport{0.0f, 0.0f, static_cast<float>(brdfLutSize), static_cast<float>(brdfLutSize), 0.0f, 1.0f});
        cb.setScissor(0, vk::Rect2D{{0, 0}, {brdfLutSize, brdfLutSize}});
        cb.bindVertexBuffers(0, {*brdfVbGpu.buffer}, {0ull});
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *brdfPipeline);

        vk::RenderingAttachmentInfoKHR colorAttachment{};
        colorAttachment.setImageView(*brdfView)
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);
        colorAttachment.setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

        vk::RenderingInfoKHR renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D{{0, 0}, {brdfLutSize, brdfLutSize}};
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

        cb.beginRendering(renderingInfo);
        cb.draw(4, 1, 0, 0);
        cb.endRendering();
    });

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*brdfAlloc.image), vk::Format::eR16G16Sfloat,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);

    result.brdfLutImage = std::move(brdfAlloc.image);
    result.brdfLutMemory = std::move(brdfAlloc.memory);
    result.brdfLutView = resourceCreator.createImageView(*result.brdfLutImage, vk::Format::eR16G16Sfloat,
        vk::ImageAspectFlagBits::eColor, 1);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(prefilterMipLevels - 1);
    result.sampler = vk::raii::Sampler(device, samplerInfo);

    return result;
}
