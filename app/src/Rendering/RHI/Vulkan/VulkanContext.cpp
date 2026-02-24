#include "Rendering/RHI/Vulkan/VulkanContext.h"

#include <set>
#include <cstring>
#include <stdexcept>
#include <iostream>

vk::Bool32 VulkanContext::debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    (void)messageSeverity;
    (void)messageType;
    (void)pUserData;
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

void VulkanContext::init(GLFWwindow* window)
{
    createInstance();
    setupDebugMessenger();
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
}

void VulkanContext::cleanup()
{
    device.reset();
    surface.reset();
    debugMessenger.reset();
    instance.reset();
}

void VulkanContext::createInstance()
{
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "Vulkan Application";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    auto glfwExtensions = getRequiredExtensions();
    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
    createInfo.ppEnabledExtensionNames = glfwExtensions.data();

    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
    }

    instance = vk::raii::Instance(context, createInfo);
}

bool VulkanContext::checkValidationLayerSupport() const
{
    auto availableLayers = context.enumerateInstanceLayerProperties();
    for (const char* layerName : validationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }
    return true;
}

std::vector<const char*> VulkanContext::getRequiredExtensions() const
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

void VulkanContext::populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo) const
{
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
        | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
        | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = debugCallback;
}

void VulkanContext::setupDebugMessenger()
{
    if (!enableValidationLayers) return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    debugMessenger = instance->createDebugUtilsMessengerEXT(createInfo);
}

void VulkanContext::createSurface(GLFWwindow* window)
{
    VkSurfaceKHR surf;
    if (glfwCreateWindowSurface(static_cast<vk::Instance>(*instance), window, nullptr, &surf) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(*instance, surf);
}

void VulkanContext::pickPhysicalDevice()
{
    vk::raii::PhysicalDevices physicalDevices(*instance);
    if (physicalDevices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    size_t pickIndex = 0;
    bool foundSuitable = false;
    for (size_t i = 0; i < physicalDevices.size(); i++) {
        if (isDeviceSuitable(physicalDevices[i])) {
            pickIndex = i;
            foundSuitable = true;
            break;
        }
    }

    if (!foundSuitable) {
        throw std::runtime_error(
            "failed to find a suitable GPU with required ray tracing shadow features "
            "(rayQuery + accelerationStructure + bufferDeviceAddress + fragmentStoresAndAtomics)!");
    }

    physicalDevice = std::move(physicalDevices[pickIndex]);
    msaaSamples = getMaxUsableSampleCount();
}

bool VulkanContext::isDeviceSuitable(const vk::raii::PhysicalDevice& dev) const
{
    QueueFamilyIndices indices = findQueueFamilies(dev);
    bool extensionsSupported = checkDeviceExtensionSupport(dev);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(dev);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    auto supportedFeatures = dev.getFeatures();
    // Indirect-driven forward path requires:
    // - multiDrawIndirect: vkCmdDrawIndexedIndirect with drawCount > 1
    // - shaderDrawParameters: gl_BaseInstance for firstInstance->drawId mapping (DrawParameters capability)
    auto features11Chain = dev.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features>();
    const auto& vulkan11Features = features11Chain.get<vk::PhysicalDeviceVulkan11Features>();

    return indices.isComplete() && extensionsSupported && swapChainAdequate
        && supportedFeatures.samplerAnisotropy
        && supportedFeatures.multiDrawIndirect
        && supportedFeatures.fragmentStoresAndAtomics
        && vulkan11Features.shaderDrawParameters
        && hasRequiredRayTracingFeatures(dev);
}

bool VulkanContext::hasRequiredRayTracingFeatures(const vk::raii::PhysicalDevice& dev) const
{
    auto featuresChain = dev.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceRayQueryFeaturesKHR,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();

    const auto& vulkan12Features = featuresChain.get<vk::PhysicalDeviceVulkan12Features>();
    const auto& rayQueryFeatures = featuresChain.get<vk::PhysicalDeviceRayQueryFeaturesKHR>();
    const auto& accelFeatures = featuresChain.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();

    return vulkan12Features.bufferDeviceAddress
        && accelFeatures.accelerationStructure
        && rayQueryFeatures.rayQuery
        && vulkan12Features.shaderSampledImageArrayNonUniformIndexing;
}

QueueFamilyIndices VulkanContext::findQueueFamilies(const vk::raii::PhysicalDevice& dev) const
{
    QueueFamilyIndices indices;
    auto queueFamilies = dev.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto& queueFamily = queueFamilies[i];
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }
        if (dev.getSurfaceSupportKHR(i, *surface)) {
            indices.presentFamily = i;
        }
        if (indices.isComplete()) break;
    }
    return indices;
}

bool VulkanContext::checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& dev) const
{
    auto availableExtensions = dev.enumerateDeviceExtensionProperties();
    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}

SwapChainSupportDetails VulkanContext::querySwapChainSupport(const vk::raii::PhysicalDevice& dev) const
{
    SwapChainSupportDetails details;
    details.capabilities = dev.getSurfaceCapabilitiesKHR(*surface);
    details.formats = dev.getSurfaceFormatsKHR(*surface);
    details.presentModes = dev.getSurfacePresentModesKHR(*surface);
    return details;
}

vk::SampleCountFlagBits VulkanContext::getMaxUsableSampleCount() const
{
    auto props = physicalDevice->getProperties();
    vk::SampleCountFlags counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2) return vk::SampleCountFlagBits::e2;
    return vk::SampleCountFlagBits::e1;
}

void VulkanContext::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(*physicalDevice);
    graphicsQueueFamilyIndex = indices.graphicsFamily.value();
    presentQueueFamilyIndex = indices.presentFamily.value();

    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    const vk::PhysicalDeviceFeatures supported = physicalDevice->getFeatures();
    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = supported.samplerAnisotropy ? VK_TRUE : VK_FALSE;
    deviceFeatures.sampleRateShading = supported.sampleRateShading ? VK_TRUE : VK_FALSE;
    deviceFeatures.multiDrawIndirect = supported.multiDrawIndirect ? VK_TRUE : VK_FALSE;
    // Required by RTAO: fragment shader writes to storage image (imageStore).
    deviceFeatures.fragmentStoresAndAtomics = supported.fragmentStoresAndAtomics ? VK_TRUE : VK_FALSE;

    vk::PhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.bufferDeviceAddress = VK_TRUE;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    vk::PhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.shaderDrawParameters = VK_TRUE;
    vulkan11Features.pNext = &vulkan12Features;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{};
    accelFeatures.accelerationStructure = VK_TRUE;
    accelFeatures.pNext = &vulkan11Features;

    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.rayQuery = VK_TRUE;
    rayQueryFeatures.pNext = &accelFeatures;

    vk::PhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeature{};
    dynamicRenderingFeature.dynamicRendering = VK_TRUE;
    dynamicRenderingFeature.pNext = &rayQueryFeatures;

    vk::DeviceCreateInfo createInfo{};
    createInfo.pNext = &dynamicRenderingFeature;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    try {
        device = vk::raii::Device(*physicalDevice, createInfo);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Ray tracing shadow requires VK_KHR_ray_query, ")
                                 + "VK_KHR_acceleration_structure and VK_KHR_buffer_device_address: "
                                 + e.what());
    }
}

