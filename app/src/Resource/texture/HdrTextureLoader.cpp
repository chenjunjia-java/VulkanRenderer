#include "Resource/texture/HdrTextureLoader.h"

// System
#include <cstring>

// stb_image (implementation from Texture.cpp, only declarations here)
#include <stb_image.h>

std::optional<HdrTextureResult> HdrTextureLoader::loadFromFile(
    const std::string& path,
    VulkanResourceCreator* resourceCreator,
    vk::SamplerAddressMode addressModeU,
    vk::SamplerAddressMode addressModeV)
{
    if (!resourceCreator) {
        return std::nullopt;
    }

    int width = 0, height = 0, channels = 0;
    float* data = stbi_loadf(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!data || width <= 0 || height <= 0) {
        return std::nullopt;
    }

    HdrTextureResult result;
    result.width = static_cast<uint32_t>(width);
    result.height = static_cast<uint32_t>(height);
    result.format = vk::Format::eR32G32B32A32Sfloat;

    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * 4 * sizeof(float);

    BufferAllocation staging = resourceCreator->createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* mapped = staging.memory.mapMemory(0, imageSize);
    std::memcpy(mapped, data, static_cast<size_t>(imageSize));
    staging.memory.unmapMemory();
    stbi_image_free(data);

    const uint32_t mipLevels = 1;
    ImageAllocation imgAlloc = resourceCreator->createImage(
        result.width, result.height, mipLevels,
        vk::SampleCountFlagBits::e1, result.format,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator->transitionImageLayout(
        static_cast<vk::Image>(*imgAlloc.image), result.format,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);

    resourceCreator->copyBufferToImage(
        static_cast<vk::Buffer>(*staging.buffer),
        static_cast<vk::Image>(*imgAlloc.image),
        result.width, result.height);

    resourceCreator->transitionImageLayout(
        static_cast<vk::Image>(*imgAlloc.image), result.format,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, mipLevels);

    result.image = std::move(imgAlloc.image);
    result.memory = std::move(imgAlloc.memory);
    result.imageView = resourceCreator->createImageView(
        static_cast<vk::Image>(*result.image), result.format,
        vk::ImageAspectFlagBits::eColor, mipLevels);

    vk::SamplerCreateInfo si{};
    si.magFilter = vk::Filter::eLinear;
    si.minFilter = vk::Filter::eLinear;
    si.mipmapMode = vk::SamplerMipmapMode::eLinear;
    si.addressModeU = addressModeU;
    si.addressModeV = addressModeV;
    si.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    si.anisotropyEnable = VK_FALSE;
    si.borderColor = vk::BorderColor::eFloatOpaqueBlack;
    si.unnormalizedCoordinates = VK_FALSE;
    si.compareEnable = VK_FALSE;
    si.compareOp = vk::CompareOp::eAlways;
    si.mipLodBias = 0.0f;
    si.minLod = 0.0f;
    si.maxLod = 0.0f;
    result.sampler = vk::raii::Sampler(resourceCreator->getDevice(), si);

    return result;
}
