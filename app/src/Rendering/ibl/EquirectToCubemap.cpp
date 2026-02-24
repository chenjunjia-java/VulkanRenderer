#include "Rendering/ibl/EquirectToCubemap.h"

#include "Configs/AppConfig.h"

#include <array>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

namespace {

// Unit cube vertices (position only) - 8 corners, 12 triangles
const std::array<float, 108> CUBE_VERTICES = {
    // back face (-Z)
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    // front face (+Z)
    -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    // left face (-X)
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
    // right face (+X)
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    // bottom face (-Y)
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    // top face (+Y)
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
};

// Capture views in Vulkan cubemap face order: +X, -X, +Y, -Y, +Z, -Z
const std::array<glm::mat4, 6> CAPTURE_VIEWS = {
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    // +Y / -Y need the correct "up" to avoid rotation; do NOT swap faces (layer2 must be +Y, layer3 must be -Y).
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
    glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
};

struct CaptureUniforms {
    alignas(16) glm::mat4 projection;
    alignas(16) glm::mat4 view;
};

std::vector<char> loadSpv(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> code(size);
    file.read(code.data(), static_cast<std::streamsize>(size));
    return code;
}

} // namespace

CubemapResult EquirectToCubemap::convert(
    VulkanResourceCreator& resourceCreator,
    vk::ImageView equirectView,
    vk::Sampler equirectSampler,
    uint32_t cubeSize)
{
    CubemapResult result;
    vk::raii::Device& device = resourceCreator.getDevice();
    // Use full float for environment cubemap to avoid half-float overflow on very bright HDR skies
    // (e.g. sun pixels). Half overflow can become Inf and later show up as white blocks in IBL maps.
    const vk::Format cubeFormat = vk::Format::eR32G32B32A32Sfloat;

    // Load shaders
    std::string basePath = AppConfig::ASSETS_PATH + "shaders/";
    auto vertCode = loadSpv(basePath + "VertShaders/cubemap_capture.vert.spv");
    auto fragCode = loadSpv(basePath + "FragShaders/equirect_to_cubemap.frag.spv");
    if (vertCode.empty() || fragCode.empty()) {
        return result;
    }

    vk::ShaderModuleCreateInfo vertInfo{};
    vertInfo.codeSize = vertCode.size();
    vertInfo.pCode = reinterpret_cast<const uint32_t*>(vertCode.data());
    vk::raii::ShaderModule vertModule(device, vertInfo);

    vk::ShaderModuleCreateInfo fragInfo{};
    fragInfo.codeSize = fragCode.size();
    fragInfo.pCode = reinterpret_cast<const uint32_t*>(fragCode.data());
    vk::raii::ShaderModule fragModule(device, fragInfo);

    // Create cubemap image
    ImageAllocation cubemapAlloc = resourceCreator.createImage(
        cubeSize, cubeSize, 1,
        vk::SampleCountFlagBits::e1,
        cubeFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        6,
        vk::ImageCreateFlagBits::eCubeCompatible);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*cubemapAlloc.image),
        cubeFormat,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        1,
        6);

    // Depth buffer for correct face selection (otherwise multiple cube faces can overlap at edges).
    const vk::Format depthFormat = resourceCreator.findDepthFormat();
    ImageAllocation depthAlloc = resourceCreator.createImage(
        cubeSize, cubeSize, 1,
        vk::SampleCountFlagBits::e1,
        depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*depthAlloc.image),
        depthFormat,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal,
        1);

    vk::ImageAspectFlags depthAspect = vk::ImageAspectFlagBits::eDepth;
    if (depthFormat == vk::Format::eD32SfloatS8Uint || depthFormat == vk::Format::eD24UnormS8Uint) {
        depthAspect |= vk::ImageAspectFlagBits::eStencil;
    }
    vk::raii::ImageView depthView = resourceCreator.createImageView(
        static_cast<vk::Image>(*depthAlloc.image), depthFormat, depthAspect, 1);

    // Descriptor set layout: sampler2D (binding 0)
    vk::DescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding = 0;
    samplerBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    samplerBinding.descriptorCount = 1;
    samplerBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    std::array<vk::DescriptorSetLayoutBinding, 1> layoutBindings = {samplerBinding};
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();
    vk::raii::DescriptorSetLayout descriptorSetLayout(device, layoutInfo);

    // Pipeline layout
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    vk::DescriptorSetLayout layoutHandle = *descriptorSetLayout;
    pipelineLayoutInfo.pSetLayouts = &layoutHandle;
    vk::PushConstantRange pcRange{};
    pcRange.stageFlags = vk::ShaderStageFlagBits::eVertex;
    pcRange.offset = 0;
    pcRange.size = sizeof(CaptureUniforms);
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;
    vk::raii::PipelineLayout pipelineLayout(device, pipelineLayoutInfo);

    glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    // Match Vulkan clip space convention (same as Camera::getProjMatrix).
    projection[1][1] *= -1.0f;

    // Descriptor pool and set
    vk::DescriptorPoolSize poolSizes[1];
    poolSizes[0].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[0].descriptorCount = 1;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    vk::raii::DescriptorPool descriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layoutHandle;
    vk::raii::DescriptorSets descriptorSets(device, allocInfo);
    vk::DescriptorSet descriptorSet = *descriptorSets[0];

    vk::DescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    imageInfo.imageView = equirectView;
    imageInfo.sampler = equirectSampler;

    std::array<vk::WriteDescriptorSet, 1> writes{};
    writes[0].dstSet = descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfo;
    device.updateDescriptorSets(writes, nullptr);

    // Vertex buffer for cube
    const vk::DeviceSize vbSize = sizeof(CUBE_VERTICES);
    BufferAllocation vbStaging = resourceCreator.createBuffer(
        vbSize, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* vbMapped = vbStaging.memory.mapMemory(0, vbSize);
    std::memcpy(vbMapped, CUBE_VERTICES.data(), vbSize);
    vbStaging.memory.unmapMemory();

    BufferAllocation vbGpu = resourceCreator.createBuffer(
        vbSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*vbStaging.buffer, *vbGpu.buffer, vbSize);

    // Pipeline - minimal vertex input (only vec3 position)
    vk::VertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(float) * 3;
    bindingDesc.inputRate = vk::VertexInputRate::eVertex;

    vk::VertexInputAttributeDescription attrDesc{};
    attrDesc.binding = 0;
    attrDesc.location = 0;
    attrDesc.format = vk::Format::eR32G32B32Sfloat;
    attrDesc.offset = 0;

    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &attrDesc;

    vk::PipelineShaderStageCreateInfo vertStage{};
    vertStage.stage = vk::ShaderStageFlagBits::eVertex;
    vertStage.module = *vertModule;
    vertStage.pName = "main";

    vk::PipelineShaderStageCreateInfo fragStage{};
    fragStage.stage = vk::ShaderStageFlagBits::eFragment;
    fragStage.module = *fragModule;
    fragStage.pName = "main";

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {vertStage, fragStage};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;   // avoid winding/cull mismatches during capture
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisample{};
    multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                         vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    vk::PipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &colorBlendAttachment;

    std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eLessOrEqual;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    vk::Format colorFormat = cubeFormat;
    vk::PipelineRenderingCreateInfoKHR renderingInfo{};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat = depthFormat;

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
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = *pipelineLayout;

    vk::raii::Pipeline pipeline(device, nullptr, pipelineInfo);

    // Create 6 image views for each face (each is a 2D view of one layer)
    std::array<vk::raii::ImageView, 6> faceViews = {
        resourceCreator.createImageView(*cubemapAlloc.image, cubeFormat,
                                       vk::ImageAspectFlagBits::eColor, 1,
                                       vk::ImageViewType::e2D, 0, 1),
        resourceCreator.createImageView(*cubemapAlloc.image, cubeFormat,
                                       vk::ImageAspectFlagBits::eColor, 1,
                                       vk::ImageViewType::e2D, 1, 1),
        resourceCreator.createImageView(*cubemapAlloc.image, cubeFormat,
                                       vk::ImageAspectFlagBits::eColor, 1,
                                       vk::ImageViewType::e2D, 2, 1),
        resourceCreator.createImageView(*cubemapAlloc.image, cubeFormat,
                                       vk::ImageAspectFlagBits::eColor, 1,
                                       vk::ImageViewType::e2D, 3, 1),
        resourceCreator.createImageView(*cubemapAlloc.image, cubeFormat,
                                       vk::ImageAspectFlagBits::eColor, 1,
                                       vk::ImageViewType::e2D, 4, 1),
        resourceCreator.createImageView(*cubemapAlloc.image, cubeFormat,
                                       vk::ImageAspectFlagBits::eColor, 1,
                                       vk::ImageViewType::e2D, 5, 1),
    };

    // Render each face; flip viewport Y so cubemap faces match Vulkan framebuffer convention (Y-down).
    resourceCreator.executeSingleTimeCommands([&](vk::raii::CommandBuffer& cb) {
        const float size = static_cast<float>(cubeSize);
        cb.setViewport(0, vk::Viewport{0.0f, size, size, -size, 0.0f, 1.0f});
        cb.setScissor(0, vk::Rect2D{{0, 0}, {cubeSize, cubeSize}});
        cb.bindVertexBuffers(0, {*vbGpu.buffer}, {0ull});
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, {descriptorSet}, nullptr);

        for (uint32_t face = 0; face < 6; ++face) {
            CaptureUniforms pcData{};
            pcData.projection = projection;
            pcData.view = CAPTURE_VIEWS[face];
            vk::CommandBuffer rawCb = static_cast<vk::CommandBuffer>(cb);
            rawCb.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(CaptureUniforms),
                                reinterpret_cast<const void*>(&pcData));

            vk::RenderingAttachmentInfoKHR colorAttachment{};
            colorAttachment.setImageView(*faceViews[face])
                .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setClearValue(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});

            vk::RenderingAttachmentInfoKHR depthAttachment{};
            depthAttachment.setImageView(*depthView)
                .setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                .setClearValue(vk::ClearDepthStencilValue{1.0f, 0});

            vk::RenderingInfoKHR renderingInfo{};
            renderingInfo.renderArea = vk::Rect2D{{0, 0}, {cubeSize, cubeSize}};
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &colorAttachment;
            renderingInfo.pDepthAttachment = &depthAttachment;

            cb.beginRendering(renderingInfo);
            cb.draw(36, 1, 0, 0);
            cb.endRendering();
        }
    });

    // Transition cubemap to shader read
    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*cubemapAlloc.image),
        cubeFormat,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        1,
        6);

    // Create cube view and sampler
    result.image = std::move(cubemapAlloc.image);
    result.memory = std::move(cubemapAlloc.memory);
    result.cubeView = resourceCreator.createImageView(
        *result.image, cubeFormat,
        vk::ImageAspectFlagBits::eColor, 1,
        vk::ImageViewType::eCube, 0, 6);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    // This cubemap has only 1 mip level; avoid undefined/driver-dependent mip blending.
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    result.sampler = vk::raii::Sampler(device, samplerInfo);

    return result;
}
