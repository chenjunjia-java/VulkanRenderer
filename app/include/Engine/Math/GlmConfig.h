#pragma once

// Centralize GLM configuration so all translation units see the same types.
// NOTE: Vulkan SDK's bundled GLM can fail to compile with clang-cl when
// GLM_FORCE_DEFAULT_ALIGNED_GENTYPES is enabled (missing compute_*::call).
// We rely on explicit alignas() in our UBO structs instead.

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

