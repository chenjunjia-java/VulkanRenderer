#pragma once

#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"

#include <optional>
#include <vulkan/vulkan.hpp>

struct CubemapResult {
    std::optional<vk::raii::Image> image;
    std::optional<vk::raii::DeviceMemory> memory;
    std::optional<vk::raii::ImageView> cubeView;
    std::optional<vk::raii::Sampler> sampler;
};

/**
 * Converts equirectangular 2D HDR texture to cubemap at runtime.
 * Uses dynamic rendering to render to each of the 6 faces.
 */
class EquirectToCubemap {
public:
    static CubemapResult convert(
        VulkanResourceCreator& resourceCreator,
        vk::ImageView equirectView,
        vk::Sampler equirectSampler,
        uint32_t cubeSize = 512);

private:
    EquirectToCubemap() = default;
};
