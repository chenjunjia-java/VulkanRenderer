#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Rendering/RHI/Vulkan/VulkanTypes.h"

#include <vector>
#include <optional>

class VulkanContext {
public:
    VulkanContext() = default;

    void init(GLFWwindow* window);
    void cleanup();

    vk::raii::Instance& getInstance() { return *instance; }
    const vk::raii::Instance& getInstance() const { return *instance; }
    vk::raii::PhysicalDevice& getPhysicalDevice() { return *physicalDevice; }
    const vk::raii::PhysicalDevice& getPhysicalDevice() const { return *physicalDevice; }
    vk::raii::Device& getDevice() { return *device; }
    const vk::raii::Device& getDevice() const { return *device; }
    vk::raii::Queue getGraphicsQueue() const { return device->getQueue(graphicsQueueFamilyIndex, 0); }
    bool hasDevice() const { return device.has_value(); }
    vk::raii::Queue getPresentQueue() const { return device->getQueue(presentQueueFamilyIndex, 0); }
    vk::raii::SurfaceKHR& getSurface() { return *surface; }
    const vk::raii::SurfaceKHR& getSurface() const { return *surface; }
    vk::SampleCountFlagBits getMsaaSamples() const { return msaaSamples; }

    QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& dev) const;
    SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& dev) const;

private:
    void createInstance();
    bool checkValidationLayerSupport() const;
    std::vector<const char*> getRequiredExtensions() const;
    void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo) const;
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    bool isDeviceSuitable(const vk::raii::PhysicalDevice& dev) const;
    bool hasRequiredRayTracingFeatures(const vk::raii::PhysicalDevice& dev) const;
    void createLogicalDevice();
    bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& dev) const;
    vk::SampleCountFlagBits getMaxUsableSampleCount() const;

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageType,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

    vk::raii::Context context;
    std::optional<vk::raii::Instance> instance;
    std::optional<vk::raii::DebugUtilsMessengerEXT> debugMessenger;
    std::optional<vk::raii::SurfaceKHR> surface;
    std::optional<vk::raii::PhysicalDevice> physicalDevice;
    std::optional<vk::raii::Device> device;
    uint32_t graphicsQueueFamilyIndex = 0;
    uint32_t presentQueueFamilyIndex = 0;
    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
};

