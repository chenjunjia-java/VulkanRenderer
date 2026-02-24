#include "Rendering/pass/RtaoComputePass.h"

#include "Configs/AppConfig.h"

#include <algorithm>
#include <array>
#include <vector>

namespace {
uint32_t divUp(uint32_t x, uint32_t y)
{
    return (x + y - 1u) / y;
}
}

RtaoComputePass::RtaoComputePass(vk::raii::Device& inDevice, RtaoComputePipeline& inPipeline, FrameManager& inFrameManager, RayTracingContext& inRayTracingContext)
    : RenderPass("RtaoComputePass", {"depth"}, {"rtao_full"})
    , device(&inDevice)
    , pipeline(&inPipeline)
    , frameManager(&inFrameManager)
    , rayTracingContext(&inRayTracingContext)
{
    createDescriptorPool();
    createDescriptorSets();
}

void RtaoComputePass::beginPass(const PassExecuteContext& ctx)
{
    vk::Image depthImage = frameManager->getDepthResolveImage();
    if (!depthImage) return;

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    barrier.newLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = depthImage;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    ctx.commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eLateFragmentTests,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {},
        {},
        barrier);

    vk::Image normalImage = frameManager->getNormalResolveImage();
    if (!normalImage) return;

    vk::ImageMemoryBarrier normalBarrier{};
    normalBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    normalBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    normalBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    normalBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    normalBarrier.image = normalImage;
    normalBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    normalBarrier.subresourceRange.baseMipLevel = 0;
    normalBarrier.subresourceRange.levelCount = 1;
    normalBarrier.subresourceRange.baseArrayLayer = 0;
    normalBarrier.subresourceRange.layerCount = 1;
    normalBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    normalBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    ctx.commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {},
        {},
        normalBarrier);

    vk::Image linearDepthImage = frameManager->getLinearDepthResolveImage();
    if (!linearDepthImage) return;

    vk::ImageMemoryBarrier linearDepthBarrier{};
    linearDepthBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    linearDepthBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    linearDepthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    linearDepthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    linearDepthBarrier.image = linearDepthImage;
    linearDepthBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    linearDepthBarrier.subresourceRange.baseMipLevel = 0;
    linearDepthBarrier.subresourceRange.levelCount = 1;
    linearDepthBarrier.subresourceRange.baseArrayLayer = 0;
    linearDepthBarrier.subresourceRange.layerCount = 1;
    linearDepthBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    linearDepthBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    ctx.commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {},
        {},
        linearDepthBarrier);
}

void RtaoComputePass::render(const PassExecuteContext& ctx)
{
    if (!AppConfig::ENABLE_RTAO) {
        return;
    }

    const uint32_t frameIdx = frameManager->getCurrentFrame();
    updateDescriptorsForFrame(frameIdx);

    vk::raii::CommandBuffer& cb = ctx.commandBuffer;
    dispatchTrace(cb, frameIdx);

    // trace -> atrous read barrier
    {
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = vk::ImageLayout::eGeneral;
        barrier.newLayout = vk::ImageLayout::eGeneral;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = frameManager->getRtaoHalfHistoryImageForFrame(frameIdx, false);
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, barrier);
    }

    dispatchAtrous(cb, frameIdx);

    const uint32_t atrousIterations = std::max(1u, AppConfig::ENABLE_RTAO_SPATIAL_DENOISE ? AppConfig::RTAO_ATROUS_ITERATIONS : 1u);
    const uint32_t finalAtrousIndex = ((atrousIterations - 1u) % 2u == 0u) ? 0u : 1u;

    // atrous -> upsample read barrier
    {
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = vk::ImageLayout::eGeneral;
        barrier.newLayout = vk::ImageLayout::eGeneral;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = frameManager->getRtaoAtrousImage(finalAtrousIndex);
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, barrier);
    }

    dispatchUpsample(cb, frameIdx);
}

void RtaoComputePass::endPass(const PassExecuteContext& ctx)
{
    vk::Image fullImage = frameManager->getRtaoFullImage();
    if (fullImage) {
        vk::ImageMemoryBarrier fullBarrier{};
        fullBarrier.oldLayout = vk::ImageLayout::eGeneral;
        fullBarrier.newLayout = vk::ImageLayout::eGeneral;
        fullBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fullBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fullBarrier.image = fullImage;
        fullBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        fullBarrier.subresourceRange.baseMipLevel = 0;
        fullBarrier.subresourceRange.levelCount = 1;
        fullBarrier.subresourceRange.baseArrayLayer = 0;
        fullBarrier.subresourceRange.layerCount = 1;
        fullBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        fullBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        ctx.commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eFragmentShader,
            {},
            {},
            {},
            fullBarrier);
    }

    vk::Image depthImage = frameManager->getDepthResolveImage();
    if (depthImage) {
        vk::ImageMemoryBarrier depthBarrier{};
        depthBarrier.oldLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        depthBarrier.newLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        depthBarrier.image = depthImage;
        depthBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
        depthBarrier.subresourceRange.baseMipLevel = 0;
        depthBarrier.subresourceRange.levelCount = 1;
        depthBarrier.subresourceRange.baseArrayLayer = 0;
        depthBarrier.subresourceRange.layerCount = 1;
        depthBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        depthBarrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
        ctx.commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            {},
            {},
            {},
            depthBarrier);
    }

    vk::Image normalImage = frameManager->getNormalResolveImage();
    if (normalImage) {
        vk::ImageMemoryBarrier normalBarrier{};
        normalBarrier.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        normalBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        normalBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        normalBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        normalBarrier.image = normalImage;
        normalBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        normalBarrier.subresourceRange.baseMipLevel = 0;
        normalBarrier.subresourceRange.levelCount = 1;
        normalBarrier.subresourceRange.baseArrayLayer = 0;
        normalBarrier.subresourceRange.layerCount = 1;
        normalBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        normalBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        ctx.commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {},
            {},
            {},
            normalBarrier);
    }

    vk::Image linearDepthImage = frameManager->getLinearDepthResolveImage();
    if (linearDepthImage) {
        vk::ImageMemoryBarrier linearDepthBarrier{};
        linearDepthBarrier.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        linearDepthBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        linearDepthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        linearDepthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        linearDepthBarrier.image = linearDepthImage;
        linearDepthBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        linearDepthBarrier.subresourceRange.baseMipLevel = 0;
        linearDepthBarrier.subresourceRange.levelCount = 1;
        linearDepthBarrier.subresourceRange.baseArrayLayer = 0;
        linearDepthBarrier.subresourceRange.layerCount = 1;
        linearDepthBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
        linearDepthBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        ctx.commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {},
            {},
            {},
            linearDepthBarrier);
    }
}

void RtaoComputePass::createDescriptorPool()
{
    std::array<vk::DescriptorPoolSize, 5> poolSizes{};
    poolSizes[0] = vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, AppConfig::MAX_FRAMES_IN_FLIGHT};
    poolSizes[1] = vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler,
                                          AppConfig::MAX_FRAMES_IN_FLIGHT * (7u + AppConfig::MAX_REFLECTION_MATERIAL_COUNT)};
    poolSizes[2] = vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, AppConfig::MAX_FRAMES_IN_FLIGHT * 4u};
    poolSizes[3] = vk::DescriptorPoolSize{vk::DescriptorType::eAccelerationStructureKHR, AppConfig::MAX_FRAMES_IN_FLIGHT};
    poolSizes[4] = vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, AppConfig::MAX_FRAMES_IN_FLIGHT * 4u};

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.maxSets = AppConfig::MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    descriptorPool = vk::raii::DescriptorPool(*device, poolInfo);
}

void RtaoComputePass::createDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(AppConfig::MAX_FRAMES_IN_FLIGHT, pipeline->getDescriptorSetLayout());
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = AppConfig::MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    descriptorSets = vk::raii::DescriptorSets(*device, allocInfo);
}

void RtaoComputePass::updateDescriptorsForFrame(uint32_t frameIndex)
{
    vk::DescriptorSet set = (*descriptorSets)[frameIndex % AppConfig::MAX_FRAMES_IN_FLIGHT];

    vk::DescriptorBufferInfo uboInfo{};
    uboInfo.buffer = frameManager->getUniformBuffer(frameIndex);
    uboInfo.offset = 0;
    uboInfo.range = sizeof(PBRUniformBufferObject);

    vk::DescriptorImageInfo depthInfo{};
    depthInfo.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    depthInfo.imageView = frameManager->getDepthResolveImageView();
    depthInfo.sampler = frameManager->getDepthResolveSampler();

    vk::DescriptorImageInfo normalInfo{};
    normalInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    normalInfo.imageView = frameManager->getNormalResolveImageView();
    normalInfo.sampler = frameManager->getNormalResolveSampler();

    vk::DescriptorImageInfo linearDepthInfo{};
    linearDepthInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    linearDepthInfo.imageView = frameManager->getLinearDepthResolveImageView();
    linearDepthInfo.sampler = frameManager->getLinearDepthResolveSampler();

    vk::DescriptorBufferInfo instanceLutInfo{};
    instanceLutInfo.buffer = frameManager->getReflectionInstanceLUTBuffer();
    instanceLutInfo.offset = 0;
    instanceLutInfo.range = VK_WHOLE_SIZE;

    vk::DescriptorBufferInfo indexInfo{};
    indexInfo.buffer = frameManager->getReflectionIndexBuffer();
    indexInfo.offset = 0;
    indexInfo.range = VK_WHOLE_SIZE;

    vk::DescriptorBufferInfo uvInfo{};
    uvInfo.buffer = frameManager->getReflectionUVBuffer();
    uvInfo.offset = 0;
    uvInfo.range = VK_WHOLE_SIZE;

    vk::DescriptorBufferInfo materialParamsInfo{};
    materialParamsInfo.buffer = frameManager->getReflectionMaterialParamsBuffer();
    materialParamsInfo.offset = 0;
    materialParamsInfo.range = VK_WHOLE_SIZE;

    const auto& baseColorArrayInfos = frameManager->getReflectionBaseColorArrayInfos();

    vk::DescriptorImageInfo historyIn{};
    historyIn.imageLayout = vk::ImageLayout::eGeneral;
    historyIn.imageView = frameManager->getRtaoHalfHistoryImageViewForFrame(frameIndex, true);
    historyIn.sampler = frameManager->getRtaoHalfHistorySampler();

    vk::DescriptorImageInfo historyOut{};
    historyOut.imageLayout = vk::ImageLayout::eGeneral;
    historyOut.imageView = frameManager->getRtaoHalfHistoryImageViewForFrame(frameIndex, false);
    historyOut.sampler = VK_NULL_HANDLE;

    vk::DescriptorImageInfo historyCurrSampled{};
    historyCurrSampled.imageLayout = vk::ImageLayout::eGeneral;
    historyCurrSampled.imageView = frameManager->getRtaoHalfHistoryImageViewForFrame(frameIndex, false);
    historyCurrSampled.sampler = frameManager->getRtaoHalfHistorySampler();

    vk::DescriptorImageInfo ping0Sampled{};
    ping0Sampled.imageLayout = vk::ImageLayout::eGeneral;
    ping0Sampled.imageView = frameManager->getRtaoAtrousImageView(0);
    ping0Sampled.sampler = frameManager->getRtaoAtrousSampler();
    vk::DescriptorImageInfo ping1Sampled{};
    ping1Sampled.imageLayout = vk::ImageLayout::eGeneral;
    ping1Sampled.imageView = frameManager->getRtaoAtrousImageView(1);
    ping1Sampled.sampler = frameManager->getRtaoAtrousSampler();

    vk::DescriptorImageInfo ping0Out{};
    ping0Out.imageLayout = vk::ImageLayout::eGeneral;
    ping0Out.imageView = frameManager->getRtaoAtrousImageView(0);
    ping0Out.sampler = VK_NULL_HANDLE;
    vk::DescriptorImageInfo ping1Out{};
    ping1Out.imageLayout = vk::ImageLayout::eGeneral;
    ping1Out.imageView = frameManager->getRtaoAtrousImageView(1);
    ping1Out.sampler = VK_NULL_HANDLE;

    vk::DescriptorImageInfo fullOut{};
    fullOut.imageLayout = vk::ImageLayout::eGeneral;
    fullOut.imageView = frameManager->getRtaoFullImageView();
    fullOut.sampler = VK_NULL_HANDLE;

    const vk::AccelerationStructureKHR topLevelAS = rayTracingContext->getTopLevelAS();
    vk::WriteDescriptorSetAccelerationStructureKHR accelInfo{};
    accelInfo.accelerationStructureCount = 1;
    accelInfo.pAccelerationStructures = &topLevelAS;

    std::array<vk::WriteDescriptorSet, 18> writes{};
    writes[0] = vk::WriteDescriptorSet{};
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    writes[0].pBufferInfo = &uboInfo;

    writes[1] = vk::WriteDescriptorSet{};
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[1].pImageInfo = &depthInfo;

    writes[2] = vk::WriteDescriptorSet{};
    writes[2].dstSet = set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[2].pImageInfo = &normalInfo;

    writes[3] = vk::WriteDescriptorSet{};
    writes[3].dstSet = set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[3].pImageInfo = &historyIn;

    writes[4] = vk::WriteDescriptorSet{};
    writes[4].dstSet = set;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = vk::DescriptorType::eStorageImage;
    writes[4].pImageInfo = &historyOut;

    writes[5] = vk::WriteDescriptorSet{};
    writes[5].dstSet = set;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[5].pImageInfo = &historyCurrSampled;

    writes[6] = vk::WriteDescriptorSet{};
    writes[6].dstSet = set;
    writes[6].dstBinding = 6;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[6].pImageInfo = &ping0Sampled;

    writes[7] = vk::WriteDescriptorSet{};
    writes[7].dstSet = set;
    writes[7].dstBinding = 7;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[7].pImageInfo = &ping1Sampled;

    writes[8] = vk::WriteDescriptorSet{};
    writes[8].dstSet = set;
    writes[8].dstBinding = 8;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = vk::DescriptorType::eStorageImage;
    writes[8].pImageInfo = &ping0Out;

    writes[9] = vk::WriteDescriptorSet{};
    writes[9].dstSet = set;
    writes[9].dstBinding = 9;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = vk::DescriptorType::eStorageImage;
    writes[9].pImageInfo = &ping1Out;

    writes[10] = vk::WriteDescriptorSet{};
    writes[10].dstSet = set;
    writes[10].dstBinding = 10;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType = vk::DescriptorType::eStorageImage;
    writes[10].pImageInfo = &fullOut;

    writes[11] = vk::WriteDescriptorSet{};
    writes[11].dstSet = set;
    writes[11].dstBinding = 11;
    writes[11].descriptorCount = 1;
    writes[11].descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
    writes[11].pNext = &accelInfo;

    writes[12] = vk::WriteDescriptorSet{};
    writes[12].dstSet = set;
    writes[12].dstBinding = 12;
    writes[12].descriptorCount = 1;
    writes[12].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[12].pBufferInfo = &instanceLutInfo;

    writes[13] = vk::WriteDescriptorSet{};
    writes[13].dstSet = set;
    writes[13].dstBinding = 13;
    writes[13].descriptorCount = 1;
    writes[13].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[13].pBufferInfo = &indexInfo;

    writes[14] = vk::WriteDescriptorSet{};
    writes[14].dstSet = set;
    writes[14].dstBinding = 14;
    writes[14].descriptorCount = 1;
    writes[14].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[14].pBufferInfo = &uvInfo;

    writes[15] = vk::WriteDescriptorSet{};
    writes[15].dstSet = set;
    writes[15].dstBinding = 15;
    writes[15].descriptorCount = AppConfig::MAX_REFLECTION_MATERIAL_COUNT;
    writes[15].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[15].pImageInfo = baseColorArrayInfos.data();

    writes[16] = vk::WriteDescriptorSet{};
    writes[16].dstSet = set;
    writes[16].dstBinding = 16;
    writes[16].descriptorCount = 1;
    writes[16].descriptorType = vk::DescriptorType::eStorageBuffer;
    writes[16].pBufferInfo = &materialParamsInfo;

    writes[17] = vk::WriteDescriptorSet{};
    writes[17].dstSet = set;
    writes[17].dstBinding = 17;
    writes[17].descriptorCount = 1;
    writes[17].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writes[17].pImageInfo = &linearDepthInfo;

    device->updateDescriptorSets(writes, nullptr);
}

void RtaoComputePass::dispatchTrace(vk::raii::CommandBuffer& cb, uint32_t frameIndex)
{
    const vk::Extent2D extent = frameManager->getSwapChainExtent();
    const uint32_t halfWidth = extent.width;
    const uint32_t halfHeight = extent.height;

    vk::DescriptorSet set = (*descriptorSets)[frameIndex % AppConfig::MAX_FRAMES_IN_FLIGHT];
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getTracePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->getPipelineLayout(), 0, {set}, nullptr);
    PushParams push{};
    push.width = halfWidth;
    push.height = halfHeight;
    cb.pushConstants<PushParams>(pipeline->getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, push);
    cb.dispatch(divUp(halfWidth, 8u), divUp(halfHeight, 8u), 1);
}

void RtaoComputePass::dispatchAtrous(vk::raii::CommandBuffer& cb, uint32_t frameIndex)
{
    const vk::Extent2D extent = frameManager->getSwapChainExtent();
    const uint32_t halfWidth = extent.width;
    const uint32_t halfHeight = extent.height;

    vk::DescriptorSet set = (*descriptorSets)[frameIndex % AppConfig::MAX_FRAMES_IN_FLIGHT];
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getAtrousPipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->getPipelineLayout(), 0, {set}, nullptr);

    const uint32_t iterations = std::max(1u, AppConfig::ENABLE_RTAO_SPATIAL_DENOISE ? AppConfig::RTAO_ATROUS_ITERATIONS : 1u);
    for (uint32_t i = 0; i < iterations; ++i) {
        PushParams push{};
        push.width = halfWidth;
        push.height = halfHeight;
        push.step = 1u << i;
        push.iteration = i;
        cb.pushConstants<PushParams>(pipeline->getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, push);
        cb.dispatch(divUp(halfWidth, 8u), divUp(halfHeight, 8u), 1);

        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = vk::ImageLayout::eGeneral;
        barrier.newLayout = vk::ImageLayout::eGeneral;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = frameManager->getRtaoAtrousImage((i & 1u) ? 1u : 0u);
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, barrier);
    }
}

void RtaoComputePass::dispatchUpsample(vk::raii::CommandBuffer& cb, uint32_t frameIndex)
{
    const vk::Extent2D extent = frameManager->getSwapChainExtent();
    vk::DescriptorSet set = (*descriptorSets)[frameIndex % AppConfig::MAX_FRAMES_IN_FLIGHT];
    const uint32_t atrousIterations = std::max(1u, AppConfig::ENABLE_RTAO_SPATIAL_DENOISE ? AppConfig::RTAO_ATROUS_ITERATIONS : 1u);
    const uint32_t finalAtrousIndex = ((atrousIterations - 1u) % 2u == 0u) ? 0u : 1u;

    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getUpsamplePipeline());
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->getPipelineLayout(), 0, {set}, nullptr);
    PushParams push{};
    push.width = extent.width;
    push.height = extent.height;
    push.iteration = finalAtrousIndex;
    cb.pushConstants<PushParams>(pipeline->getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, push);
    cb.dispatch(divUp(extent.width, 8u), divUp(extent.height, 8u), 1);
}

