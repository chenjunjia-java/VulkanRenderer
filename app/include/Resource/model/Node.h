#pragma once

// System
#include <cstdint>
#include <string>
#include <vector>

// Third-party
#include <glm/gtc/quaternion.hpp>

// Project
#include "Engine/Math/BoundingBox.h"
#include "Engine/Math/GlmConfig.h"

struct Node {
    std::string name;
    Node* parent = nullptr;
    std::vector<Node*> children;

    // A node can reference multiple meshes (e.g. glTF mesh primitives).
    std::vector<uint32_t> meshIndices;

    // Base transform in glTF: TRS + optional matrix.
    glm::vec3 translation = glm::vec3(0.0f);
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);
    glm::mat4 matrix = glm::mat4(1.0f);
    bool hasMatrix = false;

    // Local-space bounds of this node's subtree (node space). Built once after loading.
    BoundingBox subtreeBounds{};
    bool hasSubtreeBounds = false;

    // Stable index inside Model::getLinearNodes(). Used for culling & query indexing.
    uint32_t linearIndex = UINT32_MAX;

    glm::mat4 getLocalMatrix() const;
    glm::mat4 getGlobalMatrix() const;
};

