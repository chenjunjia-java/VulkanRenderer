#include "Resource/texture/KtxTextureLoader.h"

// System
#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

// KTX-Software
#include <ktx.h>
#include <ktxvulkan.h>

namespace {

ktx_transcode_fmt_e chooseTranscodeFormat(const VulkanResourceCreator* resourceCreator)
{
    if (!resourceCreator) {
        return KTX_TTF_RGBA32;
    }
    const vk::PhysicalDeviceFeatures features = resourceCreator->getPhysicalDevice().getFeatures();
    if (features.textureCompressionBC) return KTX_TTF_BC7_RGBA;
    if (features.textureCompressionASTC_LDR) return KTX_TTF_ASTC_4x4_RGBA;
    if (features.textureCompressionETC2) return KTX_TTF_ETC2_RGBA;
    return KTX_TTF_RGBA32;
}

vk::Format toSrgbFormat(vk::Format f)
{
    switch (f) {
        case vk::Format::eR8G8B8A8Unorm: return vk::Format::eR8G8B8A8Srgb;
        case vk::Format::eB8G8R8A8Unorm: return vk::Format::eB8G8R8A8Srgb;

        case vk::Format::eBc1RgbUnormBlock: return vk::Format::eBc1RgbSrgbBlock;
        case vk::Format::eBc1RgbaUnormBlock: return vk::Format::eBc1RgbaSrgbBlock;
        case vk::Format::eBc2UnormBlock: return vk::Format::eBc2SrgbBlock;
        case vk::Format::eBc3UnormBlock: return vk::Format::eBc3SrgbBlock;
        case vk::Format::eBc7UnormBlock: return vk::Format::eBc7SrgbBlock;

        case vk::Format::eEtc2R8G8B8A8UnormBlock: return vk::Format::eEtc2R8G8B8A8SrgbBlock;
        case vk::Format::eAstc4x4UnormBlock: return vk::Format::eAstc4x4SrgbBlock;

        default: return f;
    }
}

vk::Format toLinearFormat(vk::Format f)
{
    switch (f) {
        case vk::Format::eR8G8B8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
        case vk::Format::eB8G8R8A8Srgb: return vk::Format::eB8G8R8A8Unorm;

        case vk::Format::eBc1RgbSrgbBlock: return vk::Format::eBc1RgbUnormBlock;
        case vk::Format::eBc1RgbaSrgbBlock: return vk::Format::eBc1RgbaUnormBlock;
        case vk::Format::eBc2SrgbBlock: return vk::Format::eBc2UnormBlock;
        case vk::Format::eBc3SrgbBlock: return vk::Format::eBc3UnormBlock;
        case vk::Format::eBc7SrgbBlock: return vk::Format::eBc7UnormBlock;

        case vk::Format::eEtc2R8G8B8A8SrgbBlock: return vk::Format::eEtc2R8G8B8A8UnormBlock;
        case vk::Format::eAstc4x4SrgbBlock: return vk::Format::eAstc4x4UnormBlock;

        default: return f;
    }
}

bool parseKtx2FromMemory(const uint8_t* data,
                         size_t size,
                         const VulkanResourceCreator* resourceCreator,
                         std::optional<bool> colorIsSrgb,
                         KtxTextureResult& out)
{
    ktxTexture2* ktx2 = nullptr;
    KTX_error_code result = ktxTexture2_CreateFromMemory(
        data, static_cast<ktx_size_t>(size),
        KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
        &ktx2);
    if (result != KTX_SUCCESS || !ktx2) {
        return false;
    }

    out.isCompressed = (ktx2->isCompressed != 0);
    out.wasTranscoded = false;

    if (out.isCompressed && ktxTexture2_NeedsTranscoding(ktx2)) {
        const ktx_transcode_fmt_e transcodeFmt = chooseTranscodeFormat(resourceCreator);
        result = ktxTexture2_TranscodeBasis(ktx2, transcodeFmt, 0);
        if (result != KTX_SUCCESS) {
            ktxTexture2_Destroy(ktx2);
            return false;
        }
        out.wasTranscoded = true;
    }

    out.width = static_cast<uint32_t>(ktx2->baseWidth);
    out.height = static_cast<uint32_t>(ktx2->baseHeight);
    out.mipLevels = static_cast<uint32_t>(ktx2->numLevels);
    out.format = static_cast<vk::Format>(ktxTexture2_GetVkFormat(ktx2));
    if (colorIsSrgb.has_value()) {
        out.format = colorIsSrgb.value() ? toSrgbFormat(out.format) : toLinearFormat(out.format);
    }

    ktxTexture* base = ktxTexture(ktx2);
    const ktx_size_t dataSize = ktxTexture_GetDataSize(base);
    const uint8_t* dataPtr = ktxTexture_GetData(base);
    out.data.assign(dataPtr, dataPtr + static_cast<size_t>(dataSize));

    out.levels.clear();
    out.levels.reserve(out.mipLevels);
    for (uint32_t level = 0; level < out.mipLevels; ++level) {
        ktx_size_t offset = 0;
        (void)ktxTexture_GetImageOffset(base, level, 0, 0, &offset);
        const ktx_size_t levelSize = ktxTexture_GetImageSize(base, level);
        const uint32_t w = std::max(1u, out.width >> level);
        const uint32_t h = std::max(1u, out.height >> level);
        out.levels.push_back(KtxTextureLevel{
            level, w, h, static_cast<size_t>(offset), static_cast<size_t>(levelSize)});
    }

    ktxTexture2_Destroy(ktx2);
    return true;
}

bool uploadToGpu(VulkanResourceCreator& resourceCreator,
                 const KtxTextureResult& parsed,
                 const KtxSamplerParams* samplerParams,
                 KtxTextureResult& out)
{
    if (parsed.data.empty() || parsed.width == 0 || parsed.height == 0 ||
        parsed.mipLevels == 0 || parsed.format == vk::Format::eUndefined) {
        return false;
    }

    const vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(parsed.data.size());
    BufferAllocation staging = resourceCreator.createBuffer(
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* mapped = staging.memory.mapMemory(0, imageSize);
    std::memcpy(mapped, parsed.data.data(), static_cast<size_t>(imageSize));
    staging.memory.unmapMemory();

    ImageAllocation imgAlloc = resourceCreator.createImage(
        parsed.width, parsed.height, parsed.mipLevels,
        vk::SampleCountFlagBits::e1, parsed.format,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*imgAlloc.image), parsed.format,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, parsed.mipLevels);

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(parsed.levels.empty() ? 1u : parsed.levels.size());
    if (!parsed.levels.empty()) {
        for (const KtxTextureLevel& lv : parsed.levels) {
            vk::BufferImageCopy r{};
            r.bufferOffset = static_cast<vk::DeviceSize>(lv.offset);
            r.imageSubresource = {vk::ImageAspectFlagBits::eColor, lv.level, 0, 1};
            r.imageOffset = vk::Offset3D{0, 0, 0};
            r.imageExtent = vk::Extent3D{lv.width, lv.height, 1};
            regions.push_back(r);
        }
    } else {
        vk::BufferImageCopy r{};
        r.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        r.imageExtent = vk::Extent3D{parsed.width, parsed.height, 1};
        regions.push_back(r);
    }

    resourceCreator.copyBufferToImage(
        static_cast<vk::Buffer>(*staging.buffer),
        static_cast<vk::Image>(*imgAlloc.image), regions);

    resourceCreator.transitionImageLayout(
        static_cast<vk::Image>(*imgAlloc.image), parsed.format,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        parsed.mipLevels);

    out.image = std::move(imgAlloc.image);
    out.memory = std::move(imgAlloc.memory);
    out.imageView = resourceCreator.createImageView(
        static_cast<vk::Image>(*out.image), parsed.format,
        vk::ImageAspectFlagBits::eColor, parsed.mipLevels);

    if (samplerParams) {
        vk::SamplerCreateInfo si{};
        si.magFilter = samplerParams->magFilter;
        si.minFilter = samplerParams->minFilter;
        si.mipmapMode = samplerParams->mipmapMode;
        si.addressModeU = samplerParams->addressModeU;
        si.addressModeV = samplerParams->addressModeV;
        si.addressModeW = samplerParams->addressModeW;
        si.mipLodBias = 0.0f;
        si.minLod = 0.0f;
        si.maxLod = (parsed.mipLevels > 0) ? static_cast<float>(parsed.mipLevels - 1) : 0.0f;
        si.borderColor = vk::BorderColor::eIntOpaqueBlack;
        si.unnormalizedCoordinates = VK_FALSE;
        si.anisotropyEnable = samplerParams->anisotropy ? VK_TRUE : VK_FALSE;
        si.maxAnisotropy = samplerParams->maxAnisotropy;

        const auto features = resourceCreator.getPhysicalDevice().getFeatures();
        if (!features.samplerAnisotropy) {
            si.anisotropyEnable = VK_FALSE;
            si.maxAnisotropy = 1.0f;
        }

        out.sampler = vk::raii::Sampler(resourceCreator.getDevice(), si);
    }

    return true;
}

} // namespace

std::optional<KtxTextureResult> KtxTextureLoader::loadFromFile(
    const std::string& path,
    VulkanResourceCreator* resourceCreator,
    const KtxSamplerParams* samplerParams,
    std::optional<bool> colorIsSrgb)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return std::nullopt;

    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) return std::nullopt;

    const std::string name = path.substr(path.find_last_of("/\\") + 1);
    return loadFromMemory(data.data(), data.size(), resourceCreator, samplerParams, name, colorIsSrgb);
}

std::optional<KtxTextureResult> KtxTextureLoader::loadFromMemory(
    const uint8_t* data,
    size_t size,
    VulkanResourceCreator* resourceCreator,
    const KtxSamplerParams* samplerParams,
    const std::string& name,
    std::optional<bool> colorIsSrgb)
{
    KtxTextureResult parsed;
    parsed.name = name.empty() ? "ktx_texture" : name;

    if (!parseKtx2FromMemory(data, size, resourceCreator, colorIsSrgb, parsed)) {
        return std::nullopt;
    }

    if (resourceCreator) {
        KtxTextureResult uploaded;
        uploaded.name = parsed.name;
        uploaded.format = parsed.format;
        uploaded.width = parsed.width;
        uploaded.height = parsed.height;
        uploaded.mipLevels = parsed.mipLevels;
        uploaded.isCompressed = parsed.isCompressed;
        uploaded.wasTranscoded = parsed.wasTranscoded;

        if (!uploadToGpu(*resourceCreator, parsed, samplerParams, uploaded)) {
            return std::nullopt;
        }
        return uploaded;
    }

    return parsed;
}
