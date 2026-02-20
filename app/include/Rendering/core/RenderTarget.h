#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/core/ImageResource.h"

#include <optional>
#include <string>

class RenderTarget {
public:
    RenderTarget() = default;

    void create(VulkanResourceCreator& creator, const std::string& name, vk::Format format,
                vk::Extent2D extent, vk::ImageUsageFlags usage, vk::ImageLayout initialLayout,
                vk::ImageLayout finalLayout, vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor,
                vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);

    void destroy();

    vk::ImageView getImageView() const;
    vk::Image getImage() const;
    vk::Extent2D getExtent() const { return extent; }
    bool isValid() const { return view.has_value(); }

private:
    std::string name;
    vk::Format format{};
    vk::Extent2D extent{};
    vk::ImageUsageFlags usage{};
    vk::ImageLayout initialLayout{};
    vk::ImageLayout finalLayout{};
    vk::ImageAspectFlags aspectFlags{};
    vk::SampleCountFlagBits samples{};

    std::optional<vk::raii::Image> image;
    std::optional<vk::raii::DeviceMemory> memory;
    std::optional<vk::raii::ImageView> view;
};

