#include "Resource/texture/Texture.h"

#include "Configs/AppConfig.h"
#include "Resource/core/ResourceManager.h"

#include <cmath>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

unsigned char* Texture::loadImageData(const std::string& filePath, int* width, int* height, int* channels)
{
    return stbi_load(filePath.c_str(), width, height, channels, STBI_rgb_alpha);
}

void Texture::freeImageData(unsigned char* data)
{
    if (data) {
        stbi_image_free(data);
    }
}

void Texture::createVulkanImage(unsigned char* data, int width, int height, int channels)
{
    VulkanResourceCreator* resourceCreator = GetResourceManager()->getResourceCreator();
    vk::raii::Device& device = resourceCreator->getDevice();
    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(width) * height * 4;

    (void)channels;
    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;

    BufferAllocation staging = resourceCreator->createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                                                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    auto mapResult = staging.memory.mapMemory(0, imageSize);
    void* mappedData = mapResult;
    memcpy(mappedData, data, static_cast<size_t>(imageSize));
    staging.memory.unmapMemory();

    ImageAllocation texAlloc = resourceCreator->createImage(width, height, mipLevels, vk::SampleCountFlagBits::e1,
                                                           vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                                                           vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator->transitionImageLayout(*texAlloc.image, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
    resourceCreator->copyBufferToImage(*staging.buffer, *texAlloc.image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    resourceCreator->generateMipmaps(*texAlloc.image, vk::Format::eR8G8B8A8Srgb, static_cast<uint32_t>(width), static_cast<uint32_t>(height), mipLevels);

    textureImage = std::move(texAlloc.image);
    textureImageMemory = std::move(texAlloc.memory);

    textureImageView = resourceCreator->createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    const auto feats = resourceCreator->getPhysicalDevice().getFeatures();
    samplerInfo.anisotropyEnable = feats.samplerAnisotropy ? VK_TRUE : VK_FALSE;
    samplerInfo.maxAnisotropy = feats.samplerAnisotropy ? 16.0f : 1.0f;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (mipLevels > 0) ? static_cast<float>(mipLevels - 1) : 0.0f;

    textureSampler = vk::raii::Sampler(device, samplerInfo);
}

bool Texture::doLoad()
{
    std::string filePath = AppConfig::ASSETS_PATH + "textures/" + GetId() + ".png";

    unsigned char* data = loadImageData(filePath, &textureWidth, &textureHeight, &textureChannels);
    if (!data) {
        return false;
    }

    createVulkanImage(data, textureWidth, textureHeight, textureChannels);
    freeImageData(data);

    return true;
}

void Texture::doUnload()
{
    textureSampler.reset();
    textureImageView.reset();
    textureImage.reset();
    textureImageMemory.reset();
    textureWidth = 0;
    textureHeight = 0;
    textureChannels = 0;
}

