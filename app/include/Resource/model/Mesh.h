#pragma once

// System
#include <cstdint>
#include <vector>

// Project
#include "Resource/model/Vertex.h"

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    int materialIndex = -1;
};

