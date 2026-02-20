#pragma once

// System
#include <cstdint>
#include <optional>
#include <vector>

// Vulkan/Third-party
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"

class GpuMesh {
public:
    GpuMesh() = default;
    ~GpuMesh() { reset(); }

    GpuMesh(const GpuMesh&) = delete;
    GpuMesh& operator=(const GpuMesh&) = delete;
    GpuMesh(GpuMesh&&) = default;
    GpuMesh& operator=(GpuMesh&&) = default;

    bool isUploaded() const { return vertexBuffer.has_value() && indexBuffer.has_value(); }

    void upload(VulkanResourceCreator& resourceCreator,
                const std::vector<Vertex>& inVertices,
                const std::vector<uint32_t>& inIndices);

    void reset();

    vk::Buffer getVertexBuffer() const { return *vertexBuffer; }
    vk::Buffer getIndexBuffer() const { return *indexBuffer; }
    uint32_t getVertexCount() const { return static_cast<uint32_t>(vertices.size()); }
    uint32_t getIndexCount() const { return static_cast<uint32_t>(indices.size()); }
    const std::vector<Vertex>& getVertices() const { return vertices; }
    const std::vector<uint32_t>& getIndices() const { return indices; }

private:
    void createVertexBuffer(VulkanResourceCreator& resourceCreator, const std::vector<Vertex>& verts);
    void createIndexBuffer(VulkanResourceCreator& resourceCreator, const std::vector<uint32_t>& idx);

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::optional<vk::raii::Buffer> vertexBuffer;
    std::optional<vk::raii::DeviceMemory> vertexBufferMemory;
    std::optional<vk::raii::Buffer> indexBuffer;
    std::optional<vk::raii::DeviceMemory> indexBufferMemory;
};

