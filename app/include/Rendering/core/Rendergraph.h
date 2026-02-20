#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/core/ImageResource.h"
#include "Rendering/core/RenderPass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ExternalResourceView {
    vk::Image image;
    vk::ImageView imageView;
};

class Rendergraph {
public:
    explicit Rendergraph(vk::raii::Device& device, VulkanResourceCreator& resourceCreator);

    void AddResource(const std::string& name, vk::Format format, vk::Extent2D extent,
                     vk::ImageUsageFlags usage, vk::ImageLayout initialLayout,
                     vk::ImageLayout finalLayout, vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor,
                     vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1);

    void AddExternalResource(const std::string& name, vk::Format format, vk::Extent2D extent,
                             vk::ImageLayout initialLayout, vk::ImageLayout finalLayout);

    void AddPass(std::unique_ptr<RenderPass> pass);

    void Compile();
    void Recompile(vk::Extent2D newExtent);
    void Cleanup();

    void Execute(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex = 0,
                 const std::unordered_map<std::string, ExternalResourceView>& externalViews = {});

    vk::ImageView GetImageView(const std::string& name) const;
    vk::Extent2D GetExtent() const { return extent; }
    bool IsCompiled() const { return compiled; }

private:
    void allocateInternalResources();

    vk::raii::Device& device;
    VulkanResourceCreator& resourceCreator;

    std::unordered_map<std::string, ImageResource> resources;
    std::vector<std::unique_ptr<RenderPass>> passes;
    std::vector<size_t> executionOrder;

    // Track layout per external VkImage handle (e.g. swapchain images).
    std::unordered_map<std::string, std::unordered_map<uint64_t, vk::ImageLayout>> externalImageLayouts;

    vk::Extent2D extent{};
    bool compiled = false;
};

