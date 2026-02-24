#pragma once

// System
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Vulkan/Third-party
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

struct GltfSampler {
    // Keep raw glTF numeric enums (tinygltf uses GL constants).
    int magFilter = -1;
    int minFilter = -1;
    int wrapS = 10497; // REPEAT
    int wrapT = 10497; // REPEAT
};

struct GltfTextureLevel {
    uint32_t level = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    size_t offset = 0;
    size_t size = 0;
};

struct GltfTexture {
    std::string name;
    int imageIndex = -1;
    int samplerIndex = -1;
    GltfSampler sampler{};

    vk::Format vkFormat = vk::Format::eUndefined;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipLevels = 0;

    bool isCompressed = false;
    bool wasTranscoded = false;

    // Transcoded (or raw) image payload as a single blob; use `levels` for per-mip offsets.
    std::vector<uint8_t> data;
    std::vector<GltfTextureLevel> levels;

    // GPU-side resources (created during glTF load when Vulkan is available).
    std::optional<vk::raii::Image> image;
    std::optional<vk::raii::DeviceMemory> memory;
    std::optional<vk::raii::ImageView> imageView;
    std::optional<vk::raii::Sampler> vkSampler;
};

