#include "Rendering/RHI/Vulkan/SwapChain.h"

#include <limits>
#include <algorithm>
#include <stdexcept>

vk::raii::ImageView SwapChain::createImageView(vk::raii::Device& device, vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const
{
    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = image;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    return vk::raii::ImageView(device, viewInfo);
}

void SwapChain::init(VulkanContext& context, GLFWwindow* inWindow)
{
    window = inWindow;

    int width = 0, height = 0;
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(inWindow, &width, &height);
        glfwWaitEvents();
    }

    createSwapChain(context, window);
    createImageViews(context.getDevice());
}

void SwapChain::recreate(VulkanContext& context, GLFWwindow* inWindow)
{
    int width = 0, height = 0;
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(inWindow, &width, &height);
        glfwWaitEvents();
    }

    context.getDevice().waitIdle();
    cleanup();

    window = inWindow;
    createSwapChain(context, window);
    createImageViews(context.getDevice());
}

void SwapChain::cleanup()
{
    swapChainImageViews.clear();
    swapChain.reset();
}

vk::ResultValue<uint32_t> SwapChain::acquireNextImage(uint64_t timeout, vk::Semaphore semaphore, vk::Fence fence) const
{
    return swapChain->acquireNextImage(timeout, semaphore, fence);
}

std::vector<vk::ImageView> SwapChain::getImageViews() const
{
    std::vector<vk::ImageView> result;
    result.reserve(swapChainImageViews.size());
    for (const auto& iv : swapChainImageViews) {
        result.push_back(static_cast<vk::ImageView>(iv));
    }
    return result;
}

vk::ImageView SwapChain::getImageView(uint32_t index) const
{
    if (index < swapChainImageViews.size()) {
        return static_cast<vk::ImageView>(swapChainImageViews[index]);
    }
    return vk::ImageView{};
}

vk::SurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) const
{
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

vk::PresentModeKHR SwapChain::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) const
{
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eFifo) {
            return availablePresentMode;
        }
    }
    return availablePresentModes[0];
}

vk::Extent2D SwapChain::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, GLFWwindow* win) const
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(win, &width, &height);

    vk::Extent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };
    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return actualExtent;
}

void SwapChain::createSwapChain(VulkanContext& context, GLFWwindow* win)
{
    SwapChainSupportDetails swapChainSupport = context.querySwapChainSupport(context.getPhysicalDevice());

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities, win);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = context.getSurface();
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    QueueFamilyIndices indices = context.findQueueFamilies(context.getPhysicalDevice());
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = swapChain ? static_cast<vk::SwapchainKHR>(*swapChain) : nullptr;

    vk::raii::Device& device = context.getDevice();
    swapChain = device.createSwapchainKHR(createInfo);
    swapChainImages = swapChain->getImages();
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void SwapChain::createImageViews(vk::raii::Device& device)
{
    swapChainImageViews.clear();
    swapChainImageViews.reserve(swapChainImages.size());
    for (vk::Image image : swapChainImages) {
        swapChainImageViews.push_back(createImageView(device, image, swapChainImageFormat, vk::ImageAspectFlagBits::eColor, 1));
    }
}

