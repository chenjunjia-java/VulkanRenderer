#pragma once

#ifndef VULKANLEARNING_RESOURCE_MODEL_VERTEX_H
#define VULKANLEARNING_RESOURCE_MODEL_VERTEX_H

// System
#include <array>

// Vulkan/Third-party
#include <vulkan/vulkan.hpp>

// Project
#include "Engine/Math/GlmConfig.h"

struct Vertex {
    glm::vec3 pos{};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    glm::vec2 texCoord{0.0f, 0.0f};

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        std::array<vk::VertexInputAttributeDescription, 4> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[2].offset = offsetof(Vertex, color);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }

    bool operator==(const Vertex& other) const
    {
        return pos == other.pos && normal == other.normal && color == other.color && texCoord == other.texCoord;
    }
};

namespace std {
template <>
struct hash<Vertex> {
    size_t operator()(Vertex const& vertex) const
    {
        const size_t posHash = hash<glm::vec3>()(vertex.pos);
        const size_t normalHash = hash<glm::vec3>()(vertex.normal);
        const size_t colorHash = hash<glm::vec3>()(vertex.color);
        const size_t uvHash = hash<glm::vec2>()(vertex.texCoord);
        return (((posHash ^ (normalHash << 1)) >> 1) ^ (colorHash << 1)) ^ (uvHash << 2);
    }
};
} // namespace std

#endif // VULKANLEARNING_RESOURCE_MODEL_VERTEX_H

