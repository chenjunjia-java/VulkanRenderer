#pragma once

#include "Engine/Math/BoundingBox.h"
#include "Engine/Math/GlmConfig.h"

#include <array>

// Camera frustum (6 planes). Stub implementation; extend with proper plane extraction later.
class Frustum {
public:
    Frustum() = default;

    // Build frustum from view-projection matrix (extracts 6 planes)
    explicit Frustum(const glm::mat4& viewProj) {
        extractPlanes(viewProj);
    }

    bool Intersects(const BoundingBox& box) const {
        // AABB vs frustum plane test. Stub: return true if any plane has no intersection.
        for (const auto& plane : planes) {
            glm::vec3 positiveVertex = box.min;
            glm::vec3 negativeVertex = box.max;

            if (plane.x >= 0.0f) {
                positiveVertex.x = box.max.x;
                negativeVertex.x = box.min.x;
            }
            if (plane.y >= 0.0f) {
                positiveVertex.y = box.max.y;
                negativeVertex.y = box.min.y;
            }
            if (plane.z >= 0.0f) {
                positiveVertex.z = box.max.z;
                negativeVertex.z = box.min.z;
            }

            if (glm::dot(glm::vec3(plane), positiveVertex) + plane.w < 0.0f) {
                return false;
            }
        }
        return true;
    }

private:
    void extractPlanes(const glm::mat4& m) {
        // Left, Right, Bottom, Top, Near, Far
        planes[0] = glm::vec4(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]);
        planes[1] = glm::vec4(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]);
        planes[2] = glm::vec4(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]);
        planes[3] = glm::vec4(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]);
        planes[4] = glm::vec4(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]);
        planes[5] = glm::vec4(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]);

        for (auto& p : planes) {
            float len = glm::length(glm::vec3(p));
            if (len > 1e-6f) {
                p /= len;
            }
        }
    }

    std::array<glm::vec4, 6> planes{};
};

