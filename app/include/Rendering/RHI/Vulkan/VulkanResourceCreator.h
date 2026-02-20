#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanContext.h"

#include <vector>
#include <optional>
#include <functional>

struct BufferAllocation {
    vk::raii::Buffer buffer;
    vk::raii::DeviceMemory memory;
};

struct ImageAllocation {
    vk::raii::Image image;
    vk::raii::DeviceMemory memory;
};

class VulkanResourceCreator {
public:
    VulkanResourceCreator() = default;

    void init(VulkanContext& context);
    void cleanup();

    BufferAllocation createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);
    ImageAllocation createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits samples,
                               vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties);
    vk::raii::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels);

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels);
    void generateMipmaps(vk::Image image, vk::Format format, uint32_t texWidth, uint32_t texHeight, uint32_t mipLevels);

    template<typename Func>
    void executeSingleTimeCommands(Func&& func);

    vk::Format findDepthFormat();
    vk::raii::CommandPool& getCommandPool() { return *commandPool; }
    const vk::raii::CommandPool& getCommandPool() const { return *commandPool; }
    vk::raii::Device& getDevice() { return *device; }
    const vk::raii::Device& getDevice() const { return *device; }

private:
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    static bool hasStencilComponent(vk::Format format);

    vk::raii::Device* device = nullptr;
    vk::raii::PhysicalDevice* physicalDevice = nullptr;
    vk::raii::Queue graphicsQueue{nullptr};
    std::optional<vk::raii::CommandPool> commandPool;
};

template<typename Func>
void VulkanResourceCreator::executeSingleTimeCommands(Func&& func)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = *commandPool;
    allocInfo.commandBufferCount = 1;

    vk::raii::CommandBuffers commandBuffers(*device, allocInfo);
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffers[0].begin(beginInfo);

    func(commandBuffers[0]);

    commandBuffers[0].end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    vk::CommandBuffer cb = commandBuffers[0];
    submitInfo.pCommandBuffers = &cb;
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();
}

