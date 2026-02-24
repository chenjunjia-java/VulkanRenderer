#include "Rendering/mesh/GlobalMeshBuffer.h"
#include "Rendering/mesh/GpuMesh.h"
#include "Resource/model/Vertex.h"

#include <cstring>

void GlobalMeshBuffer::init(VulkanResourceCreator& resourceCreator, const std::vector<GpuMesh>& meshes)
{
    cleanup();
    if (meshes.empty()) return;

    uint32_t totalVertices = 0;
    uint32_t totalIndices = 0;
    meshInfos.resize(meshes.size());

    for (size_t i = 0; i < meshes.size(); ++i) {
        const GpuMesh& m = meshes[i];
        meshInfos[i].vertexOffset = totalVertices;
        meshInfos[i].firstIndex = totalIndices;
        meshInfos[i].indexCount = m.getIndexCount();
        totalVertices += m.getVertexCount();
        totalIndices += m.getIndexCount();
    }

    const vk::DeviceSize vertexBufferSize = static_cast<vk::DeviceSize>(totalVertices) * sizeof(Vertex);
    const vk::DeviceSize indexBufferSize = static_cast<vk::DeviceSize>(totalIndices) * sizeof(uint32_t);

    // Staging vertex buffer
    std::vector<Vertex> allVertices;
    allVertices.reserve(totalVertices);
    for (const auto& m : meshes) {
        for (const Vertex& v : m.getVertices()) {
            allVertices.push_back(v);
        }
    }

    BufferAllocation vertexStaging = resourceCreator.createBuffer(
        vertexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* vmap = vertexStaging.memory.mapMemory(0, vertexBufferSize);
    std::memcpy(vmap, allVertices.data(), static_cast<size_t>(vertexBufferSize));
    vertexStaging.memory.unmapMemory();

    BufferAllocation vertexGpu = resourceCreator.createBuffer(
        vertexBufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*vertexStaging.buffer, *vertexGpu.buffer, vertexBufferSize);
    vertexBuffer = std::move(vertexGpu.buffer);
    vertexBufferMemory = std::move(vertexGpu.memory);

    // Staging index buffer: keep each mesh's indices local; use vertexOffset (baseVertex) at draw time.
    std::vector<uint32_t> allIndices;
    allIndices.reserve(totalIndices);
    for (size_t i = 0; i < meshes.size(); ++i) {
        for (uint32_t idx : meshes[i].getIndices()) {
            allIndices.push_back(idx);
        }
    }

    BufferAllocation indexStaging = resourceCreator.createBuffer(
        indexBufferSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    void* imap = indexStaging.memory.mapMemory(0, indexBufferSize);
    std::memcpy(imap, allIndices.data(), static_cast<size_t>(indexBufferSize));
    indexStaging.memory.unmapMemory();

    BufferAllocation indexGpu = resourceCreator.createBuffer(
        indexBufferSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    resourceCreator.copyBuffer(*indexStaging.buffer, *indexGpu.buffer, indexBufferSize);
    indexBuffer = std::move(indexGpu.buffer);
    indexBufferMemory = std::move(indexGpu.memory);
}

void GlobalMeshBuffer::cleanup()
{
    indexBuffer.reset();
    indexBufferMemory.reset();
    vertexBuffer.reset();
    vertexBufferMemory.reset();
    meshInfos.clear();
}

vk::Buffer GlobalMeshBuffer::getVertexBuffer() const
{
    return vertexBuffer ? static_cast<vk::Buffer>(*vertexBuffer) : vk::Buffer{};
}

vk::Buffer GlobalMeshBuffer::getIndexBuffer() const
{
    return indexBuffer ? static_cast<vk::Buffer>(*indexBuffer) : vk::Buffer{};
}
