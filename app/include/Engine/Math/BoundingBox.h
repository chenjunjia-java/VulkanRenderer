#pragma once

#include "Engine/Math/GlmConfig.h"

// Axis-aligned bounding box. Stub implementation; extend with proper mesh bounds later.
struct BoundingBox {
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};

    BoundingBox() = default;
    BoundingBox(const glm::vec3& minVal, const glm::vec3& maxVal) : min(minVal), max(maxVal) {}

    // Transform AABB by matrix (approximates transformed bounds)
    void Transform(const glm::mat4& matrix) {
        glm::vec3 corners[8] = {
            glm::vec3(min.x, min.y, min.z),
            glm::vec3(max.x, min.y, min.z),
            glm::vec3(min.x, max.y, min.z),
            glm::vec3(max.x, max.y, min.z),
            glm::vec3(min.x, min.y, max.z),
            glm::vec3(max.x, min.y, max.z),
            glm::vec3(min.x, max.y, max.z),
            glm::vec3(max.x, max.y, max.z),
        };

        glm::vec3 newMin(1e30f);
        glm::vec3 newMax(-1e30f);

        for (int i = 0; i < 8; ++i) {
            glm::vec4 transformed = matrix * glm::vec4(corners[i], 1.0f);
            glm::vec3 p(transformed.x, transformed.y, transformed.z);

            newMin = glm::min(newMin, p);
            newMax = glm::max(newMax, p);
        }

        min = newMin;
        max = newMax;
    }
};

