#pragma once

// System
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Vulkan
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

// Project
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"

/** 单层 mip 信息 */
struct KtxTextureLevel {
    uint32_t level = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t offset = 0;
    size_t size = 0;
};

/** 采样器参数，适用于天空盒、模型纹理等不同场景 */
struct KtxSamplerParams {
    vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eRepeat;
    vk::SamplerAddressMode addressModeW = vk::SamplerAddressMode::eRepeat;
    vk::Filter magFilter = vk::Filter::eLinear;
    vk::Filter minFilter = vk::Filter::eLinear;
    vk::SamplerMipmapMode mipmapMode = vk::SamplerMipmapMode::eLinear;
    bool anisotropy = true;
    float maxAnisotropy = 16.0f;
};

/** 天空盒默认采样参数：ClampToEdge 避免接缝 */
inline KtxSamplerParams ktxSkyboxSamplerParams() {
    KtxSamplerParams p;
    p.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    p.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    p.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    return p;
}

/** 经纬度(equirect) HDR 采样参数：U 方向必须 Repeat（经度环绕），V ClampToEdge */
inline KtxSamplerParams ktxEquirectSamplerParams() {
    KtxSamplerParams p;
    p.addressModeU = vk::SamplerAddressMode::eRepeat;
    p.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    p.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    return p;
}

/** KTX2 纹理加载结果 */
struct KtxTextureResult {
    std::string name;
    vk::Format format = vk::Format::eUndefined;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 0;
    bool isCompressed = false;
    bool wasTranscoded = false;

    std::vector<uint8_t> data;
    std::vector<KtxTextureLevel> levels;

    std::optional<vk::raii::Image> image;
    std::optional<vk::raii::DeviceMemory> memory;
    std::optional<vk::raii::ImageView> imageView;
    std::optional<vk::raii::Sampler> sampler;
};

/**
 * 通用 KTX2 纹理加载器，支持：
 * - 从文件加载（天空盒、独立纹理等）
 * - 从内存加载（glTF 内嵌纹理）
 */
class KtxTextureLoader {
public:
    /**
     * 从文件加载 KTX2 纹理
     * @param path 文件路径（.ktx2）
     * @param resourceCreator 非空时上传到 GPU 并创建 Image/ImageView，可选创建 Sampler
     * @param samplerParams 非空时创建 Sampler（天空盒请使用 ktxSkyboxSamplerParams()）
     */
    static std::optional<KtxTextureResult> loadFromFile(
        const std::string& path,
        VulkanResourceCreator* resourceCreator = nullptr,
        const KtxSamplerParams* samplerParams = nullptr,
        std::optional<bool> colorIsSrgb = std::nullopt);

    /**
     * 从内存加载 KTX2 纹理（用于 glTF 内嵌）
     * @param data KTX2 原始数据
     * @param size 数据大小
     * @param resourceCreator 非空时上传到 GPU
     * @param samplerParams 非空时创建 Sampler，glTF 场景通常传 nullptr 由调用方按 GltfSampler 创建
     * @param name 纹理名称（可选）
     */
    static std::optional<KtxTextureResult> loadFromMemory(
        const uint8_t* data,
        size_t size,
        VulkanResourceCreator* resourceCreator = nullptr,
        const KtxSamplerParams* samplerParams = nullptr,
        const std::string& name = {},
        std::optional<bool> colorIsSrgb = std::nullopt);
};
