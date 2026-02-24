#include "Resource/model/loaders/GltfModelLoader.h"

// Project
#include "Configs/AppConfig.h"

// System
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// Third-party
#include <glm/gtc/quaternion.hpp>

// Project
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Resource/core/ResourceManager.h"
#include "Resource/model/GltfTexture.h"
#include "Resource/model/Model.h"
#include "Resource/texture/KtxTextureLoader.h"

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

vk::raii::Sampler createSamplerFromGltf(VulkanResourceCreator& rc, const GltfSampler& gltfSampler, uint32_t mipLevels)
{
    auto toAddress = [](int w) {
        switch (w) {
        case 33071: return vk::SamplerAddressMode::eClampToEdge;
        case 33648: return vk::SamplerAddressMode::eMirroredRepeat;
        default: return vk::SamplerAddressMode::eRepeat;
        }
    };
    auto toMagFilter = [](int f, vk::Filter fb) {
        return (f == 9728) ? vk::Filter::eNearest : (f == 9729) ? vk::Filter::eLinear : fb;
    };
    auto toMinFilter = [](int f) {
        switch (f) {
        case 9728: case 9984: case 9986: return vk::Filter::eNearest;
        default: return vk::Filter::eLinear;
        }
    };
    auto toMipMode = [](int f) {
        return (f == 9986 || f == 9987) ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest;
    };

    vk::SamplerCreateInfo si{};
    si.magFilter = toMagFilter(gltfSampler.magFilter, vk::Filter::eLinear);
    si.minFilter = toMinFilter(gltfSampler.minFilter);
    si.mipmapMode = toMipMode(gltfSampler.minFilter);
    si.addressModeU = toAddress(gltfSampler.wrapS);
    si.addressModeV = toAddress(gltfSampler.wrapT);
    si.addressModeW = vk::SamplerAddressMode::eRepeat;
    si.mipLodBias = 0.0f;
    si.minLod = 0.0f;
    si.maxLod = (mipLevels > 0) ? static_cast<float>(mipLevels - 1) : 0.0f;
    si.borderColor = vk::BorderColor::eIntOpaqueBlack;
    si.unnormalizedCoordinates = VK_FALSE;
    const auto feats = rc.getPhysicalDevice().getFeatures();
    si.anisotropyEnable = feats.samplerAnisotropy ? VK_TRUE : VK_FALSE;
    si.maxAnisotropy = feats.samplerAnisotropy ? 16.0f : 1.0f;
    return vk::raii::Sampler(rc.getDevice(), si);
}

glm::vec4 readVec4FloatOrNormalized(const unsigned char* basePtr, size_t stride, size_t index,
                                    int componentType, bool normalized)
{
    const unsigned char* ptr = basePtr + index * stride;
    auto norm = [&](uint32_t v, uint32_t maxV) -> float {
        return normalized ? (static_cast<float>(v) / static_cast<float>(maxV)) : static_cast<float>(v);
    };

    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        const float* f = reinterpret_cast<const float*>(ptr);
        return glm::vec4(f[0], f[1], f[2], f[3]);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        const uint8_t* u = reinterpret_cast<const uint8_t*>(ptr);
        return glm::vec4(norm(u[0], 255u), norm(u[1], 255u), norm(u[2], 255u), norm(u[3], 255u));
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        const uint16_t* u = reinterpret_cast<const uint16_t*>(ptr);
        return glm::vec4(norm(u[0], 65535u), norm(u[1], 65535u), norm(u[2], 65535u), norm(u[3], 65535u));
    }
    default:
        return glm::vec4(0.0f);
    }
}

glm::vec3 readVec3FloatOrNormalized(const unsigned char* basePtr, size_t stride, size_t index,
                                    int componentType, bool normalized)
{
    const unsigned char* ptr = basePtr + index * stride;
    auto norm = [&](uint32_t v, uint32_t maxV) -> float {
        return normalized ? (static_cast<float>(v) / static_cast<float>(maxV)) : static_cast<float>(v);
    };

    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        const float* f = reinterpret_cast<const float*>(ptr);
        return glm::vec3(f[0], f[1], f[2]);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        const uint8_t* u = reinterpret_cast<const uint8_t*>(ptr);
        return glm::vec3(norm(u[0], 255u), norm(u[1], 255u), norm(u[2], 255u));
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        const uint16_t* u = reinterpret_cast<const uint16_t*>(ptr);
        return glm::vec3(norm(u[0], 65535u), norm(u[1], 65535u), norm(u[2], 65535u));
    }
    default:
        return glm::vec3(0.0f);
    }
}

glm::vec2 readVec2FloatOrNormalized(const unsigned char* basePtr, size_t stride, size_t index,
                                    int componentType, bool normalized)
{
    const unsigned char* ptr = basePtr + index * stride;
    auto norm = [&](uint32_t v, uint32_t maxV) -> float {
        return normalized ? (static_cast<float>(v) / static_cast<float>(maxV)) : static_cast<float>(v);
    };

    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_FLOAT: {
        const float* f = reinterpret_cast<const float*>(ptr);
        return glm::vec2(f[0], f[1]);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        const uint8_t* u = reinterpret_cast<const uint8_t*>(ptr);
        return glm::vec2(norm(u[0], 255u), norm(u[1], 255u));
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        const uint16_t* u = reinterpret_cast<const uint16_t*>(ptr);
        return glm::vec2(norm(u[0], 65535u), norm(u[1], 65535u));
    }
    default:
        return glm::vec2(0.0f);
    }
}

glm::u16vec4 readU16Vec4(const unsigned char* basePtr, size_t stride, size_t index, int componentType)
{
    const unsigned char* ptr = basePtr + index * stride;
    switch (componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
        const uint8_t* u = reinterpret_cast<const uint8_t*>(ptr);
        return glm::u16vec4(u[0], u[1], u[2], u[3]);
    }
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
        const uint16_t* u = reinterpret_cast<const uint16_t*>(ptr);
        return glm::u16vec4(u[0], u[1], u[2], u[3]);
    }
    default:
        return glm::u16vec4(0);
    }
}

// KTX2 file magic (first 12 bytes)
static const unsigned char KTX2_MAGIC[12] = {
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

static bool loadImageDataCallback(tinygltf::Image* image,
                                  const int image_idx,
                                  std::string* err,
                                  std::string* warn,
                                  int /*req_width*/,
                                  int /*req_height*/,
                                  const unsigned char* bytes,
                                  int size,
                                  void* /*user_data*/)
{
    (void)warn;
    if (!image || !bytes || size < 12) {
        if (err) *err += "loadImageData: invalid params for image[" + std::to_string(image_idx) + "].\n";
        return false;
    }
    // KTX2: pass through raw bytes (as_is) for later KtxTextureLoader
    if (size >= 12 && std::memcmp(bytes, KTX2_MAGIC, 12) == 0) {
        image->width = image->height = image->component = -1;
        image->bits = image->pixel_type = -1;
        image->mimeType = "image/ktx2";
        image->as_is = true;
        image->image.resize(static_cast<size_t>(size));
        std::memcpy(image->image.data(), bytes, static_cast<size_t>(size));
        return true;
    }
    if (err) {
        *err += "loadImageData: unsupported format for image[" + std::to_string(image_idx) +
                "] (only KTX2 supported when TINYGLTF_NO_STB_IMAGE is set).\n";
    }
    return false;
}

} // namespace

bool GltfModelLoader::loadFromFile(const std::string& filePath, Model& outModel)
{
    outModel.clear();

    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(loadImageDataCallback, nullptr);

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

    VulkanResourceCreator* resourceCreator = nullptr;
    if (outModel.GetResourceManager()) {
        resourceCreator = outModel.GetResourceManager()->getResourceCreator();
    }

    // Decide which glTF textures should be treated as sRGB (color) vs UNORM (linear).
    // We infer this from material usage (baseColor/emissive/specGloss) to avoid gamma issues.
    std::vector<bool> textureIsSrgb(gltf.textures.size(), false);
    auto markSrgb = [&](int textureIndex) {
        if (textureIndex >= 0 && textureIndex < static_cast<int>(textureIsSrgb.size())) {
            textureIsSrgb[static_cast<size_t>(textureIndex)] = true;
        }
    };
    for (const tinygltf::Material& mat : gltf.materials) {
        markSrgb(mat.pbrMetallicRoughness.baseColorTexture.index);
        markSrgb(mat.emissiveTexture.index);

        const auto extIt = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");
        if (extIt != mat.extensions.end()) {
            const tinygltf::Value& ext = extIt->second;
            if (ext.Has("diffuseTexture")) {
                const tinygltf::Value& v = ext.Get("diffuseTexture");
                if (v.Has("index")) markSrgb(v.Get("index").GetNumberAsInt());
            }
        }
    }

    // Textures (glTF textures -> embedded KTX2 via KtxTextureLoader)
    outModel.textures.clear();
    outModel.textures.resize(gltf.textures.size());
    for (size_t i = 0; i < gltf.textures.size(); ++i) {
        const tinygltf::Texture& srcTex = gltf.textures[i];
        GltfTexture t{};
        t.imageIndex = srcTex.source;
        t.samplerIndex = srcTex.sampler;

        if (srcTex.sampler >= 0 && srcTex.sampler < static_cast<int>(gltf.samplers.size())) {
            const tinygltf::Sampler& s = gltf.samplers[static_cast<size_t>(srcTex.sampler)];
            t.sampler.magFilter = s.magFilter;
            t.sampler.minFilter = s.minFilter;
            t.sampler.wrapS = s.wrapS;
            t.sampler.wrapT = s.wrapT;
        }

        const int imgIdx = srcTex.source;
        if (imgIdx >= 0 && imgIdx < static_cast<int>(gltf.images.size())) {
            const tinygltf::Image& img = gltf.images[static_cast<size_t>(imgIdx)];
            std::string name = !srcTex.name.empty() ? srcTex.name : (!img.name.empty() ? img.name : "texture_" + std::to_string(i));
            const unsigned char* ktxData = nullptr;
            size_t ktxSize = 0;

            if (img.mimeType == "image/ktx2") {
                if (img.bufferView >= 0 && img.bufferView < static_cast<int>(gltf.bufferViews.size())) {
                    const tinygltf::BufferView& bv = gltf.bufferViews[static_cast<size_t>(img.bufferView)];
                    if (bv.buffer >= 0 && bv.buffer < static_cast<int>(gltf.buffers.size())) {
                        const tinygltf::Buffer& buf = gltf.buffers[static_cast<size_t>(bv.buffer)];
                        const size_t off = static_cast<size_t>(bv.byteOffset);
                        const size_t sz = static_cast<size_t>(bv.byteLength);
                        if (off + sz <= buf.data.size()) {
                            ktxData = buf.data.data() + off;
                            ktxSize = sz;
                        }
                    }
                } else if (!img.image.empty() && img.as_is) {
                    ktxData = img.image.data();
                    ktxSize = img.image.size();
                }
            }

            if (ktxData && ktxSize > 0) {
                auto ktxResult = KtxTextureLoader::loadFromMemory(
                    ktxData, ktxSize,
                    resourceCreator,
                    nullptr,
                    name,
                    textureIsSrgb[i]);
                if (ktxResult) {
                    t.name = ktxResult->name;
                    t.vkFormat = ktxResult->format;
                    t.width = ktxResult->width;
                    t.height = ktxResult->height;
                    t.mipLevels = ktxResult->mipLevels;
                    t.isCompressed = ktxResult->isCompressed;
                    t.wasTranscoded = ktxResult->wasTranscoded;
                    if (ktxResult->image) t.image = std::move(*ktxResult->image);
                    if (ktxResult->memory) t.memory = std::move(*ktxResult->memory);
                    if (ktxResult->imageView) t.imageView = std::move(*ktxResult->imageView);
                    if (resourceCreator && t.image && !t.vkSampler) {
                        t.vkSampler = createSamplerFromGltf(*resourceCreator, t.sampler, t.mipLevels);
                    }
                }
            }
        }
        outModel.textures[i] = std::move(t);
    }

    // Materials
    outModel.materials.clear();
    if (gltf.materials.empty()) {
        outModel.materials.push_back(Material{});
    } else {
        outModel.materials.reserve(gltf.materials.size());
    }

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

        if (srcMaterial.alphaMode == "MASK") {
            m.alphaMode = AlphaMode::Mask;
        } else if (srcMaterial.alphaMode == "BLEND") {
            m.alphaMode = AlphaMode::Blend;
        } else {
            m.alphaMode = AlphaMode::Opaque;
        }

        m.baseColorTextureIndex = pbr.baseColorTexture.index;
        m.baseColorTexCoord = pbr.baseColorTexture.texCoord;
        m.metallicRoughnessTextureIndex = pbr.metallicRoughnessTexture.index;
        m.metallicRoughnessTexCoord = pbr.metallicRoughnessTexture.texCoord;

        // Bistro 等模型使用 KHR_materials_pbrSpecularGlossiness 扩展，pbrMetallicRoughness 为空
        if (m.baseColorTextureIndex < 0 || m.metallicRoughnessTextureIndex < 0) {
            const auto extIt = srcMaterial.extensions.find("KHR_materials_pbrSpecularGlossiness");
            if (extIt != srcMaterial.extensions.end()) {
                if (m.baseColorTextureIndex < 0 && extIt->second.Has("diffuseTexture")) {
                    const tinygltf::Value& diffuseTex = extIt->second.Get("diffuseTexture");
                    if (diffuseTex.Has("index")) {
                        m.baseColorTextureIndex = diffuseTex.Get("index").GetNumberAsInt();
                        m.baseColorTexCoord = diffuseTex.Has("texCoord") ? diffuseTex.Get("texCoord").GetNumberAsInt() : 0;
                    }
                }
                if (m.metallicRoughnessTextureIndex < 0 && extIt->second.Has("specularGlossinessTexture")) {
                    const tinygltf::Value& specGlossTex = extIt->second.Get("specularGlossinessTexture");
                    if (specGlossTex.Has("index")) {
                        m.metallicRoughnessTextureIndex = specGlossTex.Get("index").GetNumberAsInt();
                        m.metallicRoughnessTexCoord =
                            specGlossTex.Has("texCoord") ? specGlossTex.Get("texCoord").GetNumberAsInt() : 0;
                    }
                }
            }
        }

        m.normalTextureIndex = srcMaterial.normalTexture.index;
        m.normalTexCoord = srcMaterial.normalTexture.texCoord;
        m.normalScale = static_cast<float>(srcMaterial.normalTexture.scale);

        m.occlusionTextureIndex = srcMaterial.occlusionTexture.index;
        m.occlusionTexCoord = srcMaterial.occlusionTexture.texCoord;
        m.occlusionStrength = static_cast<float>(srcMaterial.occlusionTexture.strength);

        m.emissiveTextureIndex = srcMaterial.emissiveTexture.index;
        m.emissiveTexCoord = srcMaterial.emissiveTexture.texCoord;

        // 光追反射：支持 extras.reflective（教程 Task 11，支持 bool 或 int）
        if (srcMaterial.extras.Has("reflective")) {
            const tinygltf::Value& v = srcMaterial.extras.Get("reflective");
            if (v.IsBool()) {
                m.reflective = v.Get<bool>();
            } else if (v.IsInt() || v.IsNumber()) {
                m.reflective = (v.GetNumberAsInt() != 0);
            }
        }

        outModel.materials.push_back(m);
    }

    // 光追反射：配置强制启用（用于无 extras 的模型）
    for (uint32_t idx : AppConfig::REFLECTIVE_MATERIAL_INDICES) {
        if (idx < static_cast<uint32_t>(outModel.materials.size())) {
            outModel.materials[idx].reflective = true;
        }
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
        dst.hasMatrix = (srcNode.matrix.size() == 16);

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
                        && getAccessorRawView(gltf, a, colorData, colorStride, colorComponents)
                        && colorComponents >= 3) {
                        hasColor = true;
                    }
                }

                const unsigned char* tangentData = nullptr;
                size_t tangentStride = 0;
                int tangentComponents = 0;
                bool hasTangent = false;
                auto tanIt = prim.attributes.find("TANGENT");
                if (tanIt != prim.attributes.end()
                    && tanIt->second >= 0
                    && tanIt->second < static_cast<int>(gltf.accessors.size())) {
                    const tinygltf::Accessor& a = gltf.accessors[static_cast<size_t>(tanIt->second)];
                    if (a.type == TINYGLTF_TYPE_VEC4
                        && a.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                        && getAccessorRawView(gltf, a, tangentData, tangentStride, tangentComponents)
                        && tangentComponents >= 4) {
                        hasTangent = true;
                    }
                }

                const unsigned char* jointsData = nullptr;
                size_t jointsStride = 0;
                int jointsComponents = 0;
                bool hasJoints = false;
                tinygltf::Accessor jointsAccessor{};
                auto jointsIt = prim.attributes.find("JOINTS_0");
                if (jointsIt != prim.attributes.end()
                    && jointsIt->second >= 0
                    && jointsIt->second < static_cast<int>(gltf.accessors.size())) {
                    jointsAccessor = gltf.accessors[static_cast<size_t>(jointsIt->second)];
                    if (jointsAccessor.type == TINYGLTF_TYPE_VEC4
                        && (jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
                            || jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                        && getAccessorRawView(gltf, jointsAccessor, jointsData, jointsStride, jointsComponents)
                        && jointsComponents >= 4) {
                        hasJoints = true;
                    }
                }

                const unsigned char* weightsData = nullptr;
                size_t weightsStride = 0;
                int weightsComponents = 0;
                bool hasWeights = false;
                tinygltf::Accessor weightsAccessor{};
                auto weightsIt = prim.attributes.find("WEIGHTS_0");
                if (weightsIt != prim.attributes.end()
                    && weightsIt->second >= 0
                    && weightsIt->second < static_cast<int>(gltf.accessors.size())) {
                    weightsAccessor = gltf.accessors[static_cast<size_t>(weightsIt->second)];
                    if (weightsAccessor.type == TINYGLTF_TYPE_VEC4
                        && (weightsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                            || weightsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
                            || weightsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                        && getAccessorRawView(gltf, weightsAccessor, weightsData, weightsStride, weightsComponents)
                        && weightsComponents >= 4) {
                        hasWeights = true;
                    }
                }

                Mesh mesh{};
                mesh.materialIndex = prim.material >= 0 ? prim.material : 0;
                mesh.vertices.reserve(posAccessor.count);
                if (hasJoints) mesh.joints0.reserve(posAccessor.count);
                if (hasWeights) mesh.weights0.reserve(posAccessor.count);

                for (size_t v = 0; v < posAccessor.count; ++v) {
                    Vertex vert{};
                    const float* p = reinterpret_cast<const float*>(posData + v * posStride);
                    vert.pos = glm::vec3(p[0], p[1], p[2]);

                    if (hasNormal) {
                        const float* n = reinterpret_cast<const float*>(normalData + v * normalStride);
                        vert.normal = glm::normalize(glm::vec3(n[0], n[1], n[2]));
                    }
                    if (hasUv) {
                        const tinygltf::Accessor& uvAccessor = gltf.accessors[static_cast<size_t>(uvIt->second)];
                        vert.texCoord = readVec2FloatOrNormalized(uvData, uvStride, v, uvAccessor.componentType, uvAccessor.normalized);
                        // glTF 与 OpenGL 约定 V=0 在底部，Vulkan 约定 V=0 在顶部，需翻转 V 以修正纹理上下颠倒
                        vert.texCoord.y = 1.0f - vert.texCoord.y;
                    }
                    if (hasColor) {
                        const tinygltf::Accessor& colAccessor = gltf.accessors[static_cast<size_t>(colIt->second)];
                        if (colAccessor.type == TINYGLTF_TYPE_VEC4) {
                            const glm::vec4 c = readVec4FloatOrNormalized(colorData, colorStride, v, colAccessor.componentType, colAccessor.normalized);
                            vert.color = glm::vec3(c.x, c.y, c.z);
                        } else {
                            vert.color = readVec3FloatOrNormalized(colorData, colorStride, v, colAccessor.componentType, colAccessor.normalized);
                        }
                    }
                    if (hasTangent) {
                        const float* t = reinterpret_cast<const float*>(tangentData + v * tangentStride);
                        vert.tangent = glm::vec4(t[0], t[1], t[2], t[3]);
                    }

                    mesh.vertices.push_back(vert);
                    if (hasJoints) {
                        mesh.joints0.push_back(readU16Vec4(jointsData, jointsStride, v, jointsAccessor.componentType));
                    }
                    if (hasWeights) {
                        glm::vec4 w = readVec4FloatOrNormalized(weightsData, weightsStride, v, weightsAccessor.componentType, weightsAccessor.normalized);
                        const float sum = w.x + w.y + w.z + w.w;
                        if (sum > 0.0f) {
                            w /= sum;
                        }
                        mesh.weights0.push_back(w);
                    }
                }

                if (prim.indices >= 0 && prim.indices < static_cast<int>(gltf.accessors.size())) {
                    const tinygltf::Accessor& indexAccessor = gltf.accessors[static_cast<size_t>(prim.indices)];
                    const unsigned char* indexData = nullptr;
                    size_t indexStride = 0;
                    int indexComponents = 0;
                    if (!getAccessorRawView(gltf, indexAccessor, indexData, indexStride, indexComponents) || indexComponents != 1) {
                        continue;
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
                            continue;
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

    // Skins
    outModel.skins.clear();
    outModel.skins.reserve(gltf.skins.size());
    for (const tinygltf::Skin& srcSkin : gltf.skins) {
        Skin s{};
        s.name = srcSkin.name;
        s.skeletonRoot = srcSkin.skeleton;
        s.joints.assign(srcSkin.joints.begin(), srcSkin.joints.end());

        if (srcSkin.inverseBindMatrices >= 0 && srcSkin.inverseBindMatrices < static_cast<int>(gltf.accessors.size())) {
            const tinygltf::Accessor& acc = gltf.accessors[static_cast<size_t>(srcSkin.inverseBindMatrices)];
            if (acc.type == TINYGLTF_TYPE_MAT4 && acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                const unsigned char* data = nullptr;
                size_t stride = 0;
                int comps = 0;
                if (getAccessorRawView(gltf, acc, data, stride, comps) && comps == 16) {
                    s.inverseBindMatrices.resize(acc.count);
                    for (size_t i = 0; i < acc.count; ++i) {
                        const float* m = reinterpret_cast<const float*>(data + i * stride);
                        glm::mat4 mat(1.0f);
                        // glTF matrices are column-major arrays of 16 floats.
                        for (int c = 0; c < 4; ++c) {
                            for (int r = 0; r < 4; ++r) {
                                mat[c][r] = m[c * 4 + r];
                            }
                        }
                        s.inverseBindMatrices[i] = mat;
                    }
                }
            }
        }
        if (s.inverseBindMatrices.empty() && !s.joints.empty()) {
            s.inverseBindMatrices.assign(s.joints.size(), glm::mat4(1.0f));
        }

        outModel.skins.push_back(std::move(s));
    }

    // Animations
    outModel.animations.clear();
    outModel.animations.reserve(gltf.animations.size());
    for (const tinygltf::Animation& srcAnim : gltf.animations) {
        Animation a{};
        a.name = srcAnim.name;

        a.samplers.reserve(srcAnim.samplers.size());
        for (const tinygltf::AnimationSampler& srcSampler : srcAnim.samplers) {
            AnimationSampler s{};
            if (srcSampler.interpolation == "STEP") {
                s.interpolation = AnimationInterpolation::Step;
            } else if (srcSampler.interpolation == "CUBICSPLINE") {
                s.interpolation = AnimationInterpolation::CubicSpline;
            } else {
                s.interpolation = AnimationInterpolation::Linear;
            }

            // Input times
            if (srcSampler.input >= 0 && srcSampler.input < static_cast<int>(gltf.accessors.size())) {
                const tinygltf::Accessor& inAcc = gltf.accessors[static_cast<size_t>(srcSampler.input)];
                const unsigned char* data = nullptr;
                size_t stride = 0;
                int comps = 0;
                if (inAcc.type == TINYGLTF_TYPE_SCALAR
                    && inAcc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                    && getAccessorRawView(gltf, inAcc, data, stride, comps)
                    && comps == 1) {
                    s.inputs.resize(inAcc.count);
                    for (size_t i = 0; i < inAcc.count; ++i) {
                        const float* t = reinterpret_cast<const float*>(data + i * stride);
                        s.inputs[i] = *t;
                    }
                    if (!s.inputs.empty()) {
                        if (a.samplers.empty()) {
                            a.start = s.inputs.front();
                            a.end = s.inputs.back();
                        } else {
                            a.start = std::min(a.start, s.inputs.front());
                            a.end = std::max(a.end, s.inputs.back());
                        }
                    }
                }
            }

            // Output values (kept as packed floats; channels define interpretation)
            if (srcSampler.output >= 0 && srcSampler.output < static_cast<int>(gltf.accessors.size())) {
                const tinygltf::Accessor& outAcc = gltf.accessors[static_cast<size_t>(srcSampler.output)];
                const unsigned char* data = nullptr;
                size_t stride = 0;
                int comps = 0;
                if (outAcc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
                    && getAccessorRawView(gltf, outAcc, data, stride, comps)
                    && comps > 0) {
                    s.outputComponents = comps;
                    s.outputs.resize(outAcc.count * static_cast<size_t>(comps));
                    for (size_t i = 0; i < outAcc.count; ++i) {
                        const float* v = reinterpret_cast<const float*>(data + i * stride);
                        for (int c = 0; c < comps; ++c) {
                            s.outputs[i * static_cast<size_t>(comps) + static_cast<size_t>(c)] = v[c];
                        }
                    }
                }
            }

            a.samplers.push_back(std::move(s));
        }

        a.channels.reserve(srcAnim.channels.size());
        for (const tinygltf::AnimationChannel& srcChannel : srcAnim.channels) {
            AnimationChannel c{};
            c.samplerIndex = srcChannel.sampler;
            c.targetNode = srcChannel.target_node;
            if (srcChannel.target_path == "rotation") {
                c.path = AnimationPath::Rotation;
            } else if (srcChannel.target_path == "scale") {
                c.path = AnimationPath::Scale;
            } else if (srcChannel.target_path == "weights") {
                c.path = AnimationPath::Weights;
            } else {
                c.path = AnimationPath::Translation;
            }
            a.channels.push_back(c);
        }

        outModel.animations.push_back(std::move(a));
    }

    return !outModel.meshes.empty();
}

