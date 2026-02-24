#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/mesh/GpuMesh.h"

#include <optional>
#include <vector>

struct RayTracingInstanceDesc {
    uint32_t meshIndex = 0;
    glm::mat4 transform{1.0f};
};

class RayTracingContext {
public:
    RayTracingContext() = default;

    void init(VulkanContext& context,
              VulkanResourceCreator& resourceCreator,
              const std::vector<GpuMesh>& meshes,
              const std::vector<uint8_t>& meshOpaqueFlags,
              const std::vector<RayTracingInstanceDesc>& instances);
    void cleanup();
    void updateTopLevelAS(vk::raii::CommandBuffer& commandBuffer,
                          const std::vector<RayTracingInstanceDesc>& instances);

    vk::AccelerationStructureKHR getTopLevelAS() const {
        return topLevelAS ? static_cast<vk::AccelerationStructureKHR>(*topLevelAS) : VK_NULL_HANDLE;
    }

private:
    BufferAllocation createDeviceAddressBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) const;
    vk::DeviceAddress getBufferDeviceAddress(vk::Buffer buffer) const;
    vk::DeviceAddress getAccelerationStructureAddress(vk::AccelerationStructureKHR accelerationStructure) const;
    vk::TransformMatrixKHR toVkTransformMatrix(const glm::mat4& matrix) const;
    void writeInstances(const std::vector<RayTracingInstanceDesc>& instances) const;
    void recordTopLevelASBuild(vk::raii::CommandBuffer& commandBuffer, vk::BuildAccelerationStructureModeKHR mode);
    void buildOrUpdateTopLevelAS(vk::BuildAccelerationStructureModeKHR mode);

    void buildBottomLevelASes(const std::vector<GpuMesh>& meshes, const std::vector<uint8_t>& meshOpaqueFlags);
    void buildTopLevelAS(uint32_t instanceCount);

    VulkanResourceCreator* resourceCreator = nullptr;
    vk::raii::Device* device = nullptr;

    struct BlasEntry {
        std::optional<vk::raii::AccelerationStructureKHR> as;
        std::optional<vk::raii::Buffer> buffer;
        std::optional<vk::raii::DeviceMemory> memory;
    };
    std::vector<BlasEntry> bottomLevelASes;

    std::optional<vk::raii::AccelerationStructureKHR> topLevelAS;
    std::optional<vk::raii::Buffer> topLevelASBuffer;
    std::optional<vk::raii::DeviceMemory> topLevelASMemory;

    std::optional<vk::raii::Buffer> instanceBuffer;
    std::optional<vk::raii::DeviceMemory> instanceMemory;
    void* instanceMapped = nullptr;
    uint32_t instanceCapacity = 0;

    std::optional<vk::raii::Buffer> topLevelScratchBuffer;
    std::optional<vk::raii::DeviceMemory> topLevelScratchMemory;

    vk::BuildAccelerationStructureFlagsKHR topLevelBuildFlags{};
    vk::DeviceSize topLevelBuildScratchSize = 0;
    vk::DeviceSize topLevelUpdateScratchSize = 0;
};
