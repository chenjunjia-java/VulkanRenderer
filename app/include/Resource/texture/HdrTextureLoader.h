#pragma once

// System
#include <optional>
#include <string>

// Vulkan
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

// Project
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"

/** HDR 纹理加载结果 */
struct HdrTextureResult {
    uint32_t width = 0;
    uint32_t height = 0;
    vk::Format format = vk::Format::eUndefined;

    std::optional<vk::raii::Image> image;
    std::optional<vk::raii::DeviceMemory> memory;
    std::optional<vk::raii::ImageView> imageView;
    std::optional<vk::raii::Sampler> sampler;
};

/**
 * 传统 .hdr 等距柱状贴图加载器，用于天空盒等 IBL 流程。
 * 使用 stb_image 的 stbi_loadf 读取 Radiance RGBE 格式。
 */
class HdrTextureLoader {
public:
    /**
     * 从文件加载 HDR 纹理
     * @param path 文件路径（.hdr）
     * @param resourceCreator 非空时上传到 GPU 并创建 Image/ImageView/Sampler
     * @param addressModeU 经度方向（equirect 需 eRepeat）
     * @param addressModeV 纬度方向（equirect 需 eClampToEdge）
     */
    static std::optional<HdrTextureResult> loadFromFile(
        const std::string& path,
        VulkanResourceCreator* resourceCreator,
        vk::SamplerAddressMode addressModeU = vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode addressModeV = vk::SamplerAddressMode::eClampToEdge);
};
