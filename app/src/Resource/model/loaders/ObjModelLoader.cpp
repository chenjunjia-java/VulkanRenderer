#include "Resource/model/loaders/ObjModelLoader.h"

// System
#include <limits>
#include <unordered_map>

// Project
#include "Resource/model/Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

bool ObjModelLoader::loadFromFile(const std::string& filePath, Model& outModel)
{
    outModel.clear();

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str())) {
        return false;
    }

    // Single default material
    outModel.materials.push_back(Material{});

    Mesh mesh{};
    mesh.materialIndex = 0;

    const float inf = std::numeric_limits<float>::infinity();
    glm::vec3 minPos(inf);
    glm::vec3 maxPos(-inf);

    std::unordered_map<Vertex, uint32_t> uniqueVertices;
    for (const tinyobj::shape_t& shape : shapes) {
        for (const tinyobj::index_t& idx : shape.mesh.indices) {
            Vertex v{};
            v.pos = glm::vec3(
                attrib.vertices[3 * idx.vertex_index + 0],
                attrib.vertices[3 * idx.vertex_index + 1],
                attrib.vertices[3 * idx.vertex_index + 2]);

            if (idx.normal_index >= 0 && !attrib.normals.empty()) {
                v.normal = glm::normalize(glm::vec3(
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2]));
            }

            if (idx.texcoord_index >= 0 && !attrib.texcoords.empty()) {
                v.texCoord = glm::vec2(
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]);
            }

            auto it = uniqueVertices.find(v);
            if (it == uniqueVertices.end()) {
                const uint32_t newIndex = static_cast<uint32_t>(mesh.vertices.size());
                uniqueVertices.emplace(v, newIndex);
                mesh.vertices.push_back(v);
                mesh.indices.push_back(newIndex);
                minPos = glm::min(minPos, v.pos);
                maxPos = glm::max(maxPos, v.pos);
            } else {
                mesh.indices.push_back(it->second);
            }
        }
    }

    if (mesh.vertices.empty() || mesh.indices.empty()) {
        return false;
    }

    mesh.bounds = BoundingBox(minPos, maxPos);
    mesh.hasBounds = true;

    const uint32_t meshIndex = static_cast<uint32_t>(outModel.meshes.size());
    outModel.meshes.push_back(std::move(mesh));

    // Single root node
    outModel.ownedNodes.push_back(std::make_unique<Node>());
    Node* root = outModel.ownedNodes.back().get();
    root->name = "Root";
    root->meshIndices.push_back(meshIndex);
    outModel.nodes.push_back(root);

    outModel.rebuildLinearNodes();
    return true;
}

