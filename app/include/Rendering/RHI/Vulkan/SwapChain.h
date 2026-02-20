#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanContext.h"

#include <vector>
#include <optional>

class SwapChain {
public:
    SwapChain() = default;

    void init(VulkanContext& context, GLFWwindow* window);
    void recreate(VulkanContext& context, GLFWwindow* window);
    void cleanup();

    vk::SwapchainKHR getSwapChain() const { return static_cast<vk::SwapchainKHR>(*swapChain); }
    vk::ResultValue<uint32_t> acquireNextImage(uint64_t timeout, vk::Semaphore semaphore, vk::Fence fence) const;
    const std::vector<vk::Image>& getImages() const { return swapChainImages; }
    vk::Format getImageFormat() const { return swapChainImageFormat; }
    vk::Extent2D getExtent() const { return swapChainExtent; }
    std::vector<vk::ImageView> getImageViews() const;
    vk::ImageView getImageView(uint32_t index) const;

private:
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const;
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const;
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) const;

    void createSwapChain(VulkanContext& context, GLFWwindow* window);
    void createImageViews(vk::raii::Device& device);
    vk::raii::ImageView createImageView(vk::raii::Device& device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const;

    std::optional<vk::raii::SwapchainKHR> swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat{};
    vk::Extent2D swapChainExtent{};
    std::vector<vk::raii::ImageView> swapChainImageViews;

    GLFWwindow* window = nullptr;
};

