#pragma once

#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"

#include <optional>
#include <vulkan/vulkan.hpp>

struct IblResult {
    std::optional<vk::raii::Image> irradianceImage;
    std::optional<vk::raii::DeviceMemory> irradianceMemory;
    std::optional<vk::raii::ImageView> irradianceView;

    std::optional<vk::raii::Image> prefilterImage;
    std::optional<vk::raii::DeviceMemory> prefilterMemory;
    std::optional<vk::raii::ImageView> prefilterView;

    std::optional<vk::raii::Image> brdfLutImage;
    std::optional<vk::raii::DeviceMemory> brdfLutMemory;
    std::optional<vk::raii::ImageView> brdfLutView;

    std::optional<vk::raii::Sampler> sampler;
};

/**
 * Precomputes IBL maps from environment cubemap at runtime:
 * - Irradiance map (32x32, diffuse)
 * - Prefilter map (128x128 with mips, specular)
 * - BRDF LUT (512x512, RG16F)
 */
class IblPrecompute {
public:
    static IblResult compute(VulkanResourceCreator& resourceCreator,
                             vk::ImageView envCubemapView,
                             vk::Sampler envCubemapSampler,
                             uint32_t irradianceSize = 32,
                             uint32_t prefilterSize = 128,
                             uint32_t brdfLutSize = 512);

private:
    IblPrecompute() = default;
};
