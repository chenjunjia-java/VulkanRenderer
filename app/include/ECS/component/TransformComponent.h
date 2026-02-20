#pragma once

#include "ECS/component/Component.h"
#include "Engine/Math/GlmConfig.h"

// Transform component. Stub implementation; extend with full position/rotation/scale later.
class TransformComponent : public Component {
public:
    glm::mat4 GetTransformMatrix() const {
        return glm::mat4(1.0f);
    }
};
