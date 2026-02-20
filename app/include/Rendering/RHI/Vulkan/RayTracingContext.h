#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Rendering/mesh/GpuMesh.h"

#include <optional>

class RayTracingContext {
public:
    RayTracingContext() = default;

    void init(VulkanContext& context, VulkanResourceCreator& resourceCreator, const GpuMesh& mesh,
              const glm::mat4& modelMatrix);
    void cleanup();
    void updateTopLevelAS(vk::raii::CommandBuffer& commandBuffer, const glm::mat4& modelMatrix);

    vk::AccelerationStructureKHR getTopLevelAS() const {
        return topLevelAS ? static_cast<vk::AccelerationStructureKHR>(*topLevelAS) : VK_NULL_HANDLE;
    }

private:
    BufferAllocation createDeviceAddressBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) const;
    vk::DeviceAddress getBufferDeviceAddress(vk::Buffer buffer) const;
    vk::DeviceAddress getAccelerationStructureAddress(vk::AccelerationStructureKHR accelerationStructure) const;
    vk::TransformMatrixKHR toVkTransformMatrix(const glm::mat4& matrix) const;
    void writeInstanceTransform(const glm::mat4& modelMatrix) const;
    void recordTopLevelASBuild(vk::raii::CommandBuffer& commandBuffer, vk::BuildAccelerationStructureModeKHR mode);
    void buildOrUpdateTopLevelAS(vk::BuildAccelerationStructureModeKHR mode);

    void buildBottomLevelAS(const GpuMesh& mesh);
    void buildTopLevelAS();

    VulkanResourceCreator* resourceCreator = nullptr;
    vk::raii::Device* device = nullptr;

    std::optional<vk::raii::AccelerationStructureKHR> bottomLevelAS;
    std::optional<vk::raii::Buffer> bottomLevelASBuffer;
    std::optional<vk::raii::DeviceMemory> bottomLevelASMemory;

    std::optional<vk::raii::AccelerationStructureKHR> topLevelAS;
    std::optional<vk::raii::Buffer> topLevelASBuffer;
    std::optional<vk::raii::DeviceMemory> topLevelASMemory;

    std::optional<vk::raii::Buffer> instanceBuffer;
    std::optional<vk::raii::DeviceMemory> instanceMemory;
    std::optional<vk::raii::Buffer> topLevelScratchBuffer;
    std::optional<vk::raii::DeviceMemory> topLevelScratchMemory;

    vk::BuildAccelerationStructureFlagsKHR topLevelBuildFlags{};
    vk::DeviceSize topLevelBuildScratchSize = 0;
    vk::DeviceSize topLevelUpdateScratchSize = 0;
};
