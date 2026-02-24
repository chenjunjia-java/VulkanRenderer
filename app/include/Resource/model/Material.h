#pragma once

// Vulkan/Third-party
#include "Engine/Math/GlmConfig.h"

enum class AlphaMode {
    Opaque,
    Mask,
    Blend,
};

struct Material {
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    glm::vec3 emissiveFactor = glm::vec3(0.0f);

    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    bool doubleSided = false;

    /// 光追反射：若为 true，在 fragment 中沿镜面反射方向发射光线并采样命中三角形颜色
    bool reflective = false;

    int baseColorTextureIndex = -1;
    int baseColorTexCoord = 0;
    int metallicRoughnessTextureIndex = -1;
    int metallicRoughnessTexCoord = 0;
    int normalTextureIndex = -1;
    int normalTexCoord = 0;
    float normalScale = 1.0f;
    int occlusionTextureIndex = -1;
    int occlusionTexCoord = 0;
    float occlusionStrength = 1.0f;
    int emissiveTextureIndex = -1;
    int emissiveTexCoord = 0;
};

