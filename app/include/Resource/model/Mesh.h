#pragma once

// System
#include <cstdint>
#include <vector>

// Project
#include "Engine/Math/GlmConfig.h"
#include "Engine/Math/BoundingBox.h"
#include "Resource/model/Vertex.h"

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    int materialIndex = -1;

    // Local-space bounds (mesh space). Used for culling and occlusion proxy.
    BoundingBox bounds{};
    bool hasBounds = false;

    // Optional glTF attribute streams (loader-only for now; not consumed by current pipeline).
    std::vector<glm::vec4> tangents;      // TANGENT (xyz + sign)
    std::vector<glm::u16vec4> joints0;    // JOINTS_0
    std::vector<glm::vec4> weights0;      // WEIGHTS_0
};

