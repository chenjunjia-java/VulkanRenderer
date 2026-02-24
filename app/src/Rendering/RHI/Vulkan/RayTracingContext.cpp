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

void RayTracingContext::writeInstances(const std::vector<RayTracingInstanceDesc>& instances) const
{
    if (!instanceMapped || !instanceMemory || !instanceBuffer) {
        throw std::runtime_error("instance buffer is not initialized/mapped");
    }
    if (instances.size() != static_cast<size_t>(instanceCapacity)) {
        throw std::runtime_error("instance count mismatch: RayTracingContext must be re-initialized");
    }

    auto* out = reinterpret_cast<vk::AccelerationStructureInstanceKHR*>(instanceMapped);
    for (uint32_t i = 0; i < instanceCapacity; ++i) {
        const RayTracingInstanceDesc& src = instances[i];
        if (src.meshIndex >= bottomLevelASes.size() || !bottomLevelASes[src.meshIndex].as) {
            throw std::runtime_error("invalid meshIndex for ray tracing instance");
        }

        vk::AccelerationStructureInstanceKHR inst{};
        inst.transform = toVkTransformMatrix(src.transform);
        inst.instanceCustomIndex = src.meshIndex;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = static_cast<uint8_t>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
        inst.accelerationStructureReference = getAccelerationStructureAddress(*bottomLevelASes[src.meshIndex].as);
        out[i] = inst;
    }
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
    rangeInfo.primitiveCount = instanceCapacity;
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

void RayTracingContext::init(VulkanContext& context,
                             VulkanResourceCreator& inResourceCreator,
                             const std::vector<GpuMesh>& meshes,
                             const std::vector<uint8_t>& meshOpaqueFlags,
                             const std::vector<RayTracingInstanceDesc>& instances)
{
    resourceCreator = &inResourceCreator;
    device = &context.getDevice();

    buildBottomLevelASes(meshes, meshOpaqueFlags);
    buildTopLevelAS(static_cast<uint32_t>(instances.size()));
    writeInstances(instances);
    buildOrUpdateTopLevelAS(vk::BuildAccelerationStructureModeKHR::eBuild);
}

void RayTracingContext::cleanup()
{
    if (instanceMemory && instanceMapped) {
        instanceMemory->unmapMemory();
    }
    instanceMapped = nullptr;
    instanceCapacity = 0;

    instanceBuffer.reset();
    instanceMemory.reset();
    topLevelScratchBuffer.reset();
    topLevelScratchMemory.reset();

    topLevelAS.reset();
    topLevelASBuffer.reset();
    topLevelASMemory.reset();

    bottomLevelASes.clear();

    resourceCreator = nullptr;
    device = nullptr;
    topLevelBuildFlags = {};
    topLevelBuildScratchSize = 0;
    topLevelUpdateScratchSize = 0;
}

void RayTracingContext::updateTopLevelAS(vk::raii::CommandBuffer& commandBuffer,
                                         const std::vector<RayTracingInstanceDesc>& instances)
{
    if (!topLevelAS || !instanceBuffer || !instanceMemory || !topLevelScratchBuffer) {
        throw std::runtime_error("cannot update TLAS before ray tracing structures are initialized");
    }
    writeInstances(instances);

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eHost,
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        {},
        vk::MemoryBarrier{
            vk::AccessFlagBits::eHostWrite,
            vk::AccessFlagBits::eAccelerationStructureReadKHR
        },
        {},
        {});

    recordTopLevelASBuild(commandBuffer, vk::BuildAccelerationStructureModeKHR::eUpdate);

    vk::MemoryBarrier postBarrier{};
    postBarrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
    postBarrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        vk::PipelineStageFlagBits::eFragmentShader,
        {},
        postBarrier,
        {},
        {});
}

void RayTracingContext::buildBottomLevelASes(const std::vector<GpuMesh>& meshes, const std::vector<uint8_t>& meshOpaqueFlags)
{
    bottomLevelASes.clear();
    bottomLevelASes.resize(meshes.size());

    for (size_t i = 0; i < meshes.size(); ++i) {
        const GpuMesh& mesh = meshes[i];
        if (!mesh.isUploaded() || mesh.getVertexCount() == 0 || mesh.getIndexCount() == 0) {
            continue;
        }

        vk::DeviceAddress vertexAddress = getBufferDeviceAddress(mesh.getVertexBuffer());
        vk::DeviceAddress indexAddress = getBufferDeviceAddress(mesh.getIndexBuffer());

        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
        triangles.vertexData.deviceAddress = vertexAddress;
        triangles.vertexStride = sizeof(Vertex);
        triangles.maxVertex = mesh.getVertexCount() > 0 ? (mesh.getVertexCount() - 1) : 0;
        triangles.indexType = vk::IndexType::eUint32;
        triangles.indexData.deviceAddress = indexAddress;

        vk::AccelerationStructureGeometryKHR geometry{};
        geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
        // Per-mesh opaque policy:
        // - true  => mark geometry opaque for faster traversal
        // - false => allow candidate traversal + alpha test in ray query
        const bool isOpaque = (i < meshOpaqueFlags.size()) ? (meshOpaqueFlags[i] != 0u) : true;
        geometry.flags = isOpaque ? vk::GeometryFlagBitsKHR::eOpaque : vk::GeometryFlagsKHR{};
        geometry.geometry.triangles = triangles;

        const uint32_t primitiveCount = mesh.getIndexCount() / 3;
        if (primitiveCount == 0) {
            continue;
        }

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
        bottomLevelASes[i].buffer = std::move(asStorage.buffer);
        bottomLevelASes[i].memory = std::move(asStorage.memory);

        vk::AccelerationStructureCreateInfoKHR createInfo{};
        createInfo.buffer = *bottomLevelASes[i].buffer;
        createInfo.size = buildSizeInfo.accelerationStructureSize;
        createInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
        bottomLevelASes[i].as = vk::raii::AccelerationStructureKHR(*device, createInfo);

        BufferAllocation scratch = createDeviceAddressBuffer(
            buildSizeInfo.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        buildInfo.dstAccelerationStructure = *bottomLevelASes[i].as;
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
}

void RayTracingContext::buildTopLevelAS(uint32_t instanceCount)
{
    if (instanceCount == 0) {
        throw std::runtime_error("cannot build TLAS with zero instances");
    }

    instanceCapacity = instanceCount;
    BufferAllocation instAlloc = createDeviceAddressBuffer(
        sizeof(vk::AccelerationStructureInstanceKHR) * static_cast<vk::DeviceSize>(instanceCount),
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    instanceBuffer = std::move(instAlloc.buffer);
    instanceMemory = std::move(instAlloc.memory);
    instanceMapped = instanceMemory->mapMemory(0, VK_WHOLE_SIZE);

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = getBufferDeviceAddress(*instanceBuffer);

    vk::AccelerationStructureGeometryKHR geometry{};
    geometry.geometryType = vk::GeometryTypeKHR::eInstances;
    geometry.geometry.instances = instancesData;

    const uint32_t primitiveCount = instanceCount;

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

    if (!topLevelAS) {
        throw std::runtime_error("failed to build top-level acceleration structure");
    }
}
