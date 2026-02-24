#pragma once

#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"

#include <cstdint>
#include <optional>
#include <vector>

class GpuMesh;

/// Per-mesh metadata for indirect draw (VkDrawIndexedIndirectCommand fields).
struct MeshDrawInfo {
    uint32_t vertexOffset = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
};

/// Merged vertex + index buffer for all meshes; supports vkCmdDrawIndexedIndirect.
class GlobalMeshBuffer {
public:
    GlobalMeshBuffer() = default;

    void init(VulkanResourceCreator& resourceCreator, const std::vector<GpuMesh>& meshes);
    void cleanup();

    vk::Buffer getVertexBuffer() const;
    vk::Buffer getIndexBuffer() const;
    const std::vector<MeshDrawInfo>& getMeshInfos() const { return meshInfos; }
    uint32_t getMeshCount() const { return static_cast<uint32_t>(meshInfos.size()); }

private:
    std::optional<vk::raii::Buffer> vertexBuffer;
    std::optional<vk::raii::DeviceMemory> vertexBufferMemory;
    std::optional<vk::raii::Buffer> indexBuffer;
    std::optional<vk::raii::DeviceMemory> indexBufferMemory;
    std::vector<MeshDrawInfo> meshInfos;
};
