#pragma once

#define GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "Engine/Math/GlmConfig.h"

#include <optional>
#include <vector>
#include <string>

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct PBRUniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 directionalLightDir;   // xyz = direction (to light), w = enable (0/1)
    alignas(16) glm::vec4 directionalLightColor;
    alignas(16) glm::vec4 directionalLightParams; // x=sunAngularRadius(rad), y=softShadowSampleCount
    alignas(16) glm::vec4 lightPositions[3];     // 3 point lights
    alignas(16) glm::vec4 lightColors[3];
    alignas(16) glm::vec4 camPos;
    alignas(16) glm::vec4 params;               // x=exposure, y=gamma, z=ambientStrength, w=pointLightCount
    alignas(16) glm::vec4 iblParams;            // x=enableDiffuseIBL, y=enableSpecularIBL, z=enableAO, w=debugView
    alignas(16) glm::vec4 rtaoParams0;          // x=enableRTAO, y=rayCount, z=radius, w=bias
    alignas(16) glm::vec4 rtaoParams1;          // x=strength, y=temporalAlpha, z=disocclusionThreshold, w=frameIndex
    alignas(16) glm::mat4 prevViewProj;         // Previous frame clip transform for AO history reprojection
};

/// 光追反射 Instance LUT：instanceID(meshIndex) -> materialID, indexBufferOffset
struct InstanceLUTEntry {
    uint32_t materialID = 0;
    uint32_t indexBufferOffset = 0;
};

struct PBRPushConstants {
    // Per-draw model transform.
    alignas(16) glm::mat4 model{1.0f};

    alignas(16) glm::vec4 baseColorFactor;
    // xyz emissive, w unused (kept 16-byte aligned)
    alignas(16) glm::vec4 emissiveFactor{0.0f, 0.0f, 0.0f, 0.0f};

    // x metallicFactor, y roughnessFactor, z alphaCutoff, w normalScale
    alignas(16) glm::vec4 materialParams0{1.0f, 1.0f, 0.5f, 1.0f};
    // x occlusionStrength, y alphaMode(0=Opaque,1=Mask,2=Blend), z reflective(0/1), w unused
    alignas(16) glm::vec4 materialParams1{1.0f, 0.0f, 0.0f, 0.0f};
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = false;  // 临时关闭以便性能测试；调试时可改回 true
#endif

