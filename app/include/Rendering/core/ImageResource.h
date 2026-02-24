#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"

#include <optional>
#include <string>

struct ImageResource {
    std::string name;
    vk::Format format;
    vk::Extent2D extent;
    vk::ImageUsageFlags usage;
    vk::ImageLayout initialLayout;
    vk::ImageLayout finalLayout;
    vk::ImageAspectFlags aspectFlags;
    vk::SampleCountFlagBits samples;
    uint32_t extentDivisor = 1;
    bool isExternal = false;

    // Tracked by Rendergraph for barrier insertion.
    // For external resources, the rendergraph tracks per-swapchain-image layout separately.
    vk::ImageLayout currentLayout = vk::ImageLayout::eUndefined;

    std::optional<vk::raii::Image> image;
    std::optional<vk::raii::DeviceMemory> memory;
    std::optional<vk::raii::ImageView> view;
};

