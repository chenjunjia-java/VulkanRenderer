#pragma once

// System
#include <string>
#include <vector>

// Vulkan/Third-party
#include "Engine/Math/GlmConfig.h"

struct Skin {
    std::string name;
    int skeletonRoot = -1;
    std::vector<int> joints;
    std::vector<glm::mat4> inverseBindMatrices;
};

