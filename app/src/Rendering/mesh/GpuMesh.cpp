#include "Rendering/mesh/GpuMesh.h"

// System
#include <cstring>

void GpuMesh::upload(VulkanResourceCreator& resourceCreator,
                     const std::vector<Vertex>& inVertices,
                     const std::vector<uint32_t>& inIndices)
{
    vertices = inVertices;
    indices = inIndices;

    if (vertices.empty() || indices.empty()) {
        reset();
        return;
    }

    createVertexBuffer(resourceCreator, vertices);
    createIndexBuffer(resourceCreator, indices);
}

void GpuMesh::reset()
{
    indexBuffer.reset();
    indexBufferMemory.reset();
    vertexBuffer.reset();
    vertexBufferMemory.reset();

    vertices.clear();
    indices.clear();
}

void GpuMesh::createVertexBuffer(VulkanResourceCreator& resourceCreator, const std::vector<Vertex>& verts)
{
    const vk::DeviceSize bufferSize = sizeof(verts[0]) * verts.size();

    BufferAllocation staging = resourceCreator.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = staging.memory.mapMemory(0, bufferSize);
    std::memcpy(data, verts.data(), static_cast<size_t>(bufferSize));
    staging.memory.unmapMemory();

    const vk::BufferUsageFlags vertexUsage = vk::BufferUsageFlagBits::eTransferDst
        | vk::BufferUsageFlagBits::eVertexBuffer
        | vk::BufferUsageFlagBits::eStorageBuffer
        | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
        | vk::BufferUsageFlagBits::eShaderDeviceAddress;

    BufferAllocation vertAlloc = resourceCreator.createBuffer(
        bufferSize,
        vertexUsage,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator.copyBuffer(*staging.buffer, *vertAlloc.buffer, bufferSize);

    vertexBuffer = std::move(vertAlloc.buffer);
    vertexBufferMemory = std::move(vertAlloc.memory);
}

void GpuMesh::createIndexBuffer(VulkanResourceCreator& resourceCreator, const std::vector<uint32_t>& idx)
{
    const vk::DeviceSize bufferSize = sizeof(idx[0]) * idx.size();

    BufferAllocation staging = resourceCreator.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = staging.memory.mapMemory(0, bufferSize);
    std::memcpy(data, idx.data(), static_cast<size_t>(bufferSize));
    staging.memory.unmapMemory();

    const vk::BufferUsageFlags indexUsage = vk::BufferUsageFlagBits::eTransferDst
        | vk::BufferUsageFlagBits::eIndexBuffer
        | vk::BufferUsageFlagBits::eStorageBuffer
        | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
        | vk::BufferUsageFlagBits::eShaderDeviceAddress;

    BufferAllocation idxAlloc = resourceCreator.createBuffer(
        bufferSize,
        indexUsage,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    resourceCreator.copyBuffer(*staging.buffer, *idxAlloc.buffer, bufferSize);

    indexBuffer = std::move(idxAlloc.buffer);
    indexBufferMemory = std::move(idxAlloc.memory);
}

