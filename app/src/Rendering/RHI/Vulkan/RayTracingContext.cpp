#include "Rendering/RHI/Vulkan/RayTracingContext.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

BufferAllocation RayTracingContext::createDeviceAddressBuffer(
    vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) const
{
    return resourceCreator->createBuffer(size, usage | vk::BufferUsageFlagBits::eShaderDeviceAddress, properties);
}

vk::DeviceAddress RayTracingContext::getBufferDeviceAddress(vk::Buffer buffer) const
{
    vk::BufferDeviceAddressInfo addrInfo{};
    addrInfo.buffer = buffer;
    return device->getBufferAddress(addrInfo);
}

vk::DeviceAddress RayTracingContext::getAccelerationStructureAddress(vk::AccelerationStructureKHR accelerationStructure) const
{
    vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.accelerationStructure = accelerationStructure;
    return device->getAccelerationStructureAddressKHR(addressInfo);
}

vk::TransformMatrixKHR RayTracingContext::toVkTransformMatrix(const glm::mat4& matrix) const
{
    vk::TransformMatrixKHR transform{};
    transform.matrix[0][0] = matrix[0][0];
    transform.matrix[0][1] = matrix[1][0];
    transform.matrix[0][2] = matrix[2][0];
    transform.matrix[0][3] = matrix[3][0];
    transform.matrix[1][0] = matrix[0][1];
    transform.matrix[1][1] = matrix[1][1];
    transform.matrix[1][2] = matrix[2][1];
    transform.matrix[1][3] = matrix[3][1];
    transform.matrix[2][0] = matrix[0][2];
    transform.matrix[2][1] = matrix[1][2];
    transform.matrix[2][2] = matrix[2][2];
    transform.matrix[2][3] = matrix[3][2];
    return transform;
}

void RayTracingContext::writeInstanceTransform(const glm::mat4& modelMatrix) const
{
    vk::AccelerationStructureInstanceKHR instance{};
    instance.transform = toVkTransformMatrix(modelMatrix);
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = static_cast<uint8_t>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
    instance.accelerationStructureReference = getAccelerationStructureAddress(*bottomLevelAS);

    void* mapped = instanceMemory->mapMemory(0, sizeof(vk::AccelerationStructureInstanceKHR));
    std::memcpy(mapped, &instance, sizeof(vk::AccelerationStructureInstanceKHR));
    instanceMemory->unmapMemory();
}

void RayTracingContext::recordTopLevelASBuild(vk::raii::CommandBuffer& commandBuffer,
                                              vk::BuildAccelerationStructureModeKHR mode)
{
    if (!topLevelAS || !instanceBuffer || !topLevelScratchBuffer) {
        throw std::runtime_error("TLAS resources are not initialized");
    }

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = getBufferDeviceAddress(*instanceBuffer);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.geometryType = vk::GeometryTypeKHR::eInstances;
    geometry.geometry.instances = instancesData;

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    buildInfo.flags = topLevelBuildFlags;
    buildInfo.mode = mode;
    buildInfo.srcAccelerationStructure = (mode == vk::BuildAccelerationStructureModeKHR::eUpdate)
        ? static_cast<vk::AccelerationStructureKHR>(*topLevelAS)
        : VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = *topLevelAS;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(*topLevelScratchBuffer);

    vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = 1;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;

    std::array<const vk::AccelerationStructureBuildRangeInfoKHR*, 1> rangeInfos = {&rangeInfo};
    commandBuffer.buildAccelerationStructuresKHR(buildInfo, rangeInfos);
}

void RayTracingContext::buildOrUpdateTopLevelAS(vk::BuildAccelerationStructureModeKHR mode)
{
    const vk::DeviceSize scratchSize = (mode == vk::BuildAccelerationStructureModeKHR::eUpdate)
        ? topLevelUpdateScratchSize
        : topLevelBuildScratchSize;
    if (scratchSize == 0) {
        throw std::runtime_error("invalid TLAS scratch size for build/update");
    }

    resourceCreator->executeSingleTimeCommands([&](vk::raii::CommandBuffer& cb) {
        recordTopLevelASBuild(cb, mode);
    });
}

void RayTracingContext::init(VulkanContext& context, VulkanResourceCreator& inResourceCreator, const GpuMesh& mesh,
                             const glm::mat4& modelMatrix)
{
    resourceCreator = &inResourceCreator;
    device = &context.getDevice();

    buildBottomLevelAS(mesh);
    buildTopLevelAS();
    writeInstanceTransform(modelMatrix);
    buildOrUpdateTopLevelAS(vk::BuildAccelerationStructureModeKHR::eUpdate);
}

void RayTracingContext::cleanup()
{
    instanceBuffer.reset();
    instanceMemory.reset();
    topLevelScratchBuffer.reset();
    topLevelScratchMemory.reset();

    topLevelAS.reset();
    topLevelASBuffer.reset();
    topLevelASMemory.reset();

    bottomLevelAS.reset();
    bottomLevelASBuffer.reset();
    bottomLevelASMemory.reset();

    resourceCreator = nullptr;
    device = nullptr;
    topLevelBuildFlags = {};
    topLevelBuildScratchSize = 0;
    topLevelUpdateScratchSize = 0;
}

void RayTracingContext::updateTopLevelAS(vk::raii::CommandBuffer& commandBuffer, const glm::mat4& modelMatrix)
{
    if (!topLevelAS || !instanceBuffer || !instanceMemory || !bottomLevelAS || !topLevelScratchBuffer) {
        throw std::runtime_error("cannot update TLAS before ray tracing structures are initialized");
    }
    writeInstanceTransform(modelMatrix);

    vk::MemoryBarrier preBarrier{};
    preBarrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR
        | vk::AccessFlagBits::eTransferWrite
        | vk::AccessFlagBits::eShaderRead;
    preBarrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR
        | vk::AccessFlagBits::eAccelerationStructureWriteKHR;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR
        | vk::PipelineStageFlagBits::eTransfer
        | vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        {},
        preBarrier,
        {},
        {});

    recordTopLevelASBuild(commandBuffer, vk::BuildAccelerationStructureModeKHR::eUpdate);

    vk::MemoryBarrier postBarrier{};
    postBarrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
    postBarrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eFragmentShader,
        {},
        postBarrier,
        {},
        {});
}

void RayTracingContext::buildBottomLevelAS(const GpuMesh& mesh)
{
    vk::DeviceAddress vertexAddress = getBufferDeviceAddress(mesh.getVertexBuffer());
    vk::DeviceAddress indexAddress = getBufferDeviceAddress(mesh.getIndexBuffer());

    vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(Vertex);
    triangles.maxVertex = mesh.getVertexCount();
    triangles.indexType = vk::IndexType::eUint32;
    triangles.indexData.deviceAddress = indexAddress;

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
    geometry.flags = {};
    geometry.geometry.triangles = triangles;

    const uint32_t primitiveCount = mesh.getIndexCount() / 3;

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
    buildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    vk::AccelerationStructureBuildSizesInfoKHR buildSizeInfo{};
    buildSizeInfo = device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCount);

    BufferAllocation asStorage = createDeviceAddressBuffer(
        buildSizeInfo.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    bottomLevelASBuffer = std::move(asStorage.buffer);
    bottomLevelASMemory = std::move(asStorage.memory);

    vk::AccelerationStructureCreateInfoKHR createInfo{};
    createInfo.buffer = *bottomLevelASBuffer;
    createInfo.size = buildSizeInfo.accelerationStructureSize;
    createInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
    bottomLevelAS = vk::raii::AccelerationStructureKHR(*device, createInfo);

    BufferAllocation scratch = createDeviceAddressBuffer(
        buildSizeInfo.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    buildInfo.dstAccelerationStructure = *bottomLevelAS;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(*scratch.buffer);

    vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;

    std::array<const vk::AccelerationStructureBuildRangeInfoKHR*, 1> rangeInfos = {&rangeInfo};
    resourceCreator->executeSingleTimeCommands([&](vk::raii::CommandBuffer& cb) {
        cb.buildAccelerationStructuresKHR(buildInfo, rangeInfos);
    });
}

void RayTracingContext::buildTopLevelAS()
{
    BufferAllocation instAlloc = createDeviceAddressBuffer(
        sizeof(vk::AccelerationStructureInstanceKHR),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    instanceBuffer = std::move(instAlloc.buffer);
    instanceMemory = std::move(instAlloc.memory);

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = getBufferDeviceAddress(*instanceBuffer);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.geometryType = vk::GeometryTypeKHR::eInstances;
    geometry.geometry.instances = instancesData;

    constexpr uint32_t primitiveCount = 1;

    topLevelBuildFlags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
        | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    buildInfo.flags = topLevelBuildFlags;
    buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    vk::AccelerationStructureBuildSizesInfoKHR buildSizeInfo{};
    buildSizeInfo = device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo, primitiveCount);
    topLevelBuildScratchSize = buildSizeInfo.buildScratchSize;
    topLevelUpdateScratchSize = buildSizeInfo.updateScratchSize;
    const vk::DeviceSize maxScratchSize = std::max(topLevelBuildScratchSize, topLevelUpdateScratchSize);

    BufferAllocation asStorage = createDeviceAddressBuffer(
        buildSizeInfo.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    topLevelASBuffer = std::move(asStorage.buffer);
    topLevelASMemory = std::move(asStorage.memory);

    vk::AccelerationStructureCreateInfoKHR createInfo{};
    createInfo.buffer = *topLevelASBuffer;
    createInfo.size = buildSizeInfo.accelerationStructureSize;
    createInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    topLevelAS = vk::raii::AccelerationStructureKHR(*device, createInfo);

    BufferAllocation scratchAlloc = createDeviceAddressBuffer(
        maxScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    topLevelScratchBuffer = std::move(scratchAlloc.buffer);
    topLevelScratchMemory = std::move(scratchAlloc.memory);

    writeInstanceTransform(glm::mat4(1.0f));
    buildOrUpdateTopLevelAS(vk::BuildAccelerationStructureModeKHR::eBuild);

    if (!topLevelAS) {
        throw std::runtime_error("failed to build top-level acceleration structure");
    }
}
