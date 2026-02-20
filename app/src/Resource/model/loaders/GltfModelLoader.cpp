#include "Resource/model/loaders/GltfModelLoader.h"

// System
#include <vector>

// Third-party
#include <glm/gtc/quaternion.hpp>

// Project
#include "Resource/model/Model.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

namespace {

bool getAccessorRawView(const tinygltf::Model& model, const tinygltf::Accessor& accessor,
                        const unsigned char*& data, size_t& stride, int& numComponents)
{
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(model.bufferViews.size())) {
        return false;
    }
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size())) {
        return false;
    }
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    const size_t componentSize = tinygltf::GetComponentSizeInBytes(accessor.componentType);
    if (componentSize == 0) {
        return false;
    }

    numComponents = tinygltf::GetNumComponentsInType(accessor.type);
    if (numComponents <= 0) {
        return false;
    }

    const size_t defaultStride = componentSize * static_cast<size_t>(numComponents);
    const int byteStride = accessor.ByteStride(view);
    stride = byteStride > 0 ? static_cast<size_t>(byteStride) : defaultStride;

    const size_t startOffset = static_cast<size_t>(view.byteOffset) + static_cast<size_t>(accessor.byteOffset);
    if (startOffset >= buffer.data.size()) {
        return false;
    }
    data = buffer.data.data() + startOffset;
    return true;
}

glm::mat4 readNodeMatrix(const tinygltf::Node& srcNode)
{
    glm::mat4 out(1.0f);
    if (srcNode.matrix.size() == 16) {
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                out[c][r] = static_cast<float>(srcNode.matrix[c * 4 + r]);
            }
        }
    }
    return out;
}

glm::vec3 readVec3(const std::vector<double>& v, const glm::vec3& fallback)
{
    if (v.size() == 3) {
        return glm::vec3(static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]));
    }
    return fallback;
}

glm::quat readQuat(const std::vector<double>& v, const glm::quat& fallback)
{
    if (v.size() == 4) {
        // glTF stores quaternion as (x,y,z,w)
        return glm::quat(static_cast<float>(v[3]), static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]));
    }
    return fallback;
}

} // namespace

bool GltfModelLoader::loadFromFile(const std::string& filePath, Model& outModel)
{
    outModel.clear();

    tinygltf::TinyGLTF loader;
    tinygltf::Model gltf;
    std::string warn;
    std::string err;

    bool loaded = false;
    if (filePath.size() >= 4 && filePath.substr(filePath.size() - 4) == ".glb") {
        loaded = loader.LoadBinaryFromFile(&gltf, &err, &warn, filePath);
    } else {
        loaded = loader.LoadASCIIFromFile(&gltf, &err, &warn, filePath);
    }
    if (!loaded) {
        return false;
    }

    // Materials
    outModel.materials.reserve(gltf.materials.size());
    for (const tinygltf::Material& srcMaterial : gltf.materials) {
        Material m{};
        const tinygltf::PbrMetallicRoughness& pbr = srcMaterial.pbrMetallicRoughness;
        if (pbr.baseColorFactor.size() == 4) {
            m.baseColorFactor = glm::vec4(
                static_cast<float>(pbr.baseColorFactor[0]),
                static_cast<float>(pbr.baseColorFactor[1]),
                static_cast<float>(pbr.baseColorFactor[2]),
                static_cast<float>(pbr.baseColorFactor[3]));
        }
        m.metallicFactor = static_cast<float>(pbr.metallicFactor);
        m.roughnessFactor = static_cast<float>(pbr.roughnessFactor);
        if (srcMaterial.emissiveFactor.size() == 3) {
            m.emissiveFactor = glm::vec3(
                static_cast<float>(srcMaterial.emissiveFactor[0]),
                static_cast<float>(srcMaterial.emissiveFactor[1]),
                static_cast<float>(srcMaterial.emissiveFactor[2]));
        }
        m.alphaCutoff = static_cast<float>(srcMaterial.alphaCutoff);
        m.doubleSided = srcMaterial.doubleSided;

        m.baseColorTextureIndex = pbr.baseColorTexture.index;
        m.metallicRoughnessTextureIndex = pbr.metallicRoughnessTexture.index;
        m.normalTextureIndex = srcMaterial.normalTexture.index;
        m.occlusionTextureIndex = srcMaterial.occlusionTexture.index;
        m.emissiveTextureIndex = srcMaterial.emissiveTexture.index;

        outModel.materials.push_back(m);
    }

    // Nodes (owned + pointers table)
    outModel.ownedNodes.resize(gltf.nodes.size());
    std::vector<Node*> nodePtrs(gltf.nodes.size(), nullptr);
    for (size_t i = 0; i < gltf.nodes.size(); ++i) {
        outModel.ownedNodes[i] = std::make_unique<Node>();
        nodePtrs[i] = outModel.ownedNodes[i].get();
    }

    for (size_t i = 0; i < gltf.nodes.size(); ++i) {
        const tinygltf::Node& srcNode = gltf.nodes[i];
        Node& dst = *nodePtrs[i];
        dst.name = srcNode.name;
        dst.translation = readVec3(srcNode.translation, glm::vec3(0.0f));
        dst.rotation = readQuat(srcNode.rotation, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
        dst.scale = readVec3(srcNode.scale, glm::vec3(1.0f));
        dst.matrix = readNodeMatrix(srcNode);

        // Mesh primitives -> multiple Mesh entries
        if (srcNode.mesh >= 0 && srcNode.mesh < static_cast<int>(gltf.meshes.size())) {
            const tinygltf::Mesh& srcMesh = gltf.meshes[static_cast<size_t>(srcNode.mesh)];
            for (const tinygltf::Primitive& prim : srcMesh.primitives) {
                const int mode = prim.mode == -1 ? TINYGLTF_MODE_TRIANGLES : prim.mode;
                if (mode != TINYGLTF_MODE_TRIANGLES) {
                    continue;
                }

                auto posIt = prim.attributes.find("POSITION");
                if (posIt == prim.attributes.end()) {
                    continue;
                }
                const int posAccessorIndex = posIt->second;
                if (posAccessorIndex < 0 || posAccessorIndex >= static_cast<int>(gltf.accessors.size())) {
                    continue;
                }

                const tinygltf::Accessor& posAccessor = gltf.accessors[static_cast<size_t>(posAccessorIndex)];
                if (posAccessor.type != TINYGLTF_TYPE_VEC3 || posAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    continue;
                }

                const unsigned char* posData = nullptr;
                size_t posStride = 0;
                int posComponents = 0;
                if (!getAccessorRawView(gltf, posAccessor, posData, posStride, posComponents) || posComponents < 3) {
                    continue;
                }

                const unsigned char* normalData = nullptr;
                size_t normalStride = 0;
                int normalComponents = 0;
                bool hasNormal = false;
                auto normalIt = prim.attributes.find("NORMAL");
                if (normalIt != prim.attributes.end()
                    && normalIt->second >= 0
                    && normalIt->second < static_cast<int>(gltf.accessors.size())) {
                    const tinygltf::Accessor& a = gltf.accessors[static_cast<size_t>(normalIt->second)];
                    if (a.type == TINYGLTF_TYPE_VEC3
                        && a.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                        && getAccessorRawView(gltf, a, normalData, normalStride, normalComponents)
                        && normalComponents >= 3) {
                        hasNormal = true;
                    }
                }

                const unsigned char* uvData = nullptr;
                size_t uvStride = 0;
                int uvComponents = 0;
                bool hasUv = false;
                auto uvIt = prim.attributes.find("TEXCOORD_0");
                if (uvIt != prim.attributes.end()
                    && uvIt->second >= 0
                    && uvIt->second < static_cast<int>(gltf.accessors.size())) {
                    const tinygltf::Accessor& a = gltf.accessors[static_cast<size_t>(uvIt->second)];
                    if (a.type == TINYGLTF_TYPE_VEC2
                        && a.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                        && getAccessorRawView(gltf, a, uvData, uvStride, uvComponents)
                        && uvComponents >= 2) {
                        hasUv = true;
                    }
                }

                const unsigned char* colorData = nullptr;
                size_t colorStride = 0;
                int colorComponents = 0;
                bool hasColor = false;
                auto colIt = prim.attributes.find("COLOR_0");
                if (colIt != prim.attributes.end()
                    && colIt->second >= 0
                    && colIt->second < static_cast<int>(gltf.accessors.size())) {
                    const tinygltf::Accessor& a = gltf.accessors[static_cast<size_t>(colIt->second)];
                    if ((a.type == TINYGLTF_TYPE_VEC3 || a.type == TINYGLTF_TYPE_VEC4)
                        && a.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                        && getAccessorRawView(gltf, a, colorData, colorStride, colorComponents)
                        && colorComponents >= 3) {
                        hasColor = true;
                    }
                }

                Mesh mesh{};
                mesh.materialIndex = prim.material;
                mesh.vertices.reserve(posAccessor.count);

                for (size_t v = 0; v < posAccessor.count; ++v) {
                    Vertex vert{};
                    const float* p = reinterpret_cast<const float*>(posData + v * posStride);
                    vert.pos = glm::vec3(p[0], p[1], p[2]);

                    if (hasNormal) {
                        const float* n = reinterpret_cast<const float*>(normalData + v * normalStride);
                        vert.normal = glm::normalize(glm::vec3(n[0], n[1], n[2]));
                    }
                    if (hasUv) {
                        const float* uv = reinterpret_cast<const float*>(uvData + v * uvStride);
                        vert.texCoord = glm::vec2(uv[0], uv[1]);
                    }
                    if (hasColor) {
                        const float* c = reinterpret_cast<const float*>(colorData + v * colorStride);
                        vert.color = glm::vec3(c[0], c[1], c[2]);
                    }

                    mesh.vertices.push_back(vert);
                }

                if (prim.indices >= 0 && prim.indices < static_cast<int>(gltf.accessors.size())) {
                    const tinygltf::Accessor& indexAccessor = gltf.accessors[static_cast<size_t>(prim.indices)];
                    const unsigned char* indexData = nullptr;
                    size_t indexStride = 0;
                    int indexComponents = 0;
                    if (!getAccessorRawView(gltf, indexAccessor, indexData, indexStride, indexComponents) || indexComponents != 1) {
                        return false;
                    }

                    mesh.indices.reserve(indexAccessor.count);
                    for (size_t ii = 0; ii < indexAccessor.count; ++ii) {
                        const unsigned char* ptr = indexData + ii * indexStride;
                        uint32_t idx = 0;
                        switch (indexAccessor.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            idx = static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(ptr));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            idx = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(ptr));
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            idx = *reinterpret_cast<const uint32_t*>(ptr);
                            break;
                        default:
                            return false;
                        }
                        mesh.indices.push_back(idx);
                    }
                } else {
                    mesh.indices.reserve(posAccessor.count);
                    for (size_t ii = 0; ii < posAccessor.count; ++ii) {
                        mesh.indices.push_back(static_cast<uint32_t>(ii));
                    }
                }

                const uint32_t meshIndex = static_cast<uint32_t>(outModel.meshes.size());
                outModel.meshes.push_back(std::move(mesh));
                dst.meshIndices.push_back(meshIndex);
            }
        }

        // Children links
        dst.children.reserve(srcNode.children.size());
        for (int c : srcNode.children) {
            if (c < 0 || c >= static_cast<int>(nodePtrs.size())) {
                continue;
            }
            Node* child = nodePtrs[static_cast<size_t>(c)];
            child->parent = &dst;
            dst.children.push_back(child);
        }
    }

    // Root nodes: prefer default scene if present; else all nodes without parent.
    if (!gltf.scenes.empty()) {
        int sceneIndex = gltf.defaultScene;
        if (sceneIndex < 0 || sceneIndex >= static_cast<int>(gltf.scenes.size())) {
            sceneIndex = 0;
        }
        const tinygltf::Scene& scene = gltf.scenes[static_cast<size_t>(sceneIndex)];
        outModel.nodes.reserve(scene.nodes.size());
        for (int n : scene.nodes) {
            if (n >= 0 && n < static_cast<int>(nodePtrs.size())) {
                outModel.nodes.push_back(nodePtrs[static_cast<size_t>(n)]);
            }
        }
    } else {
        outModel.nodes.reserve(nodePtrs.size());
        for (Node* n : nodePtrs) {
            if (n && n->parent == nullptr) {
                outModel.nodes.push_back(n);
            }
        }
    }

    outModel.rebuildLinearNodes();
    return !outModel.meshes.empty();
}

