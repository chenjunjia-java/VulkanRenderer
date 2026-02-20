#pragma once

#include "ECS/component/Component.h"
#include "Engine/Math/BoundingBox.h"

// Mesh component. Stub implementation; extend with actual mesh reference and bounds later.
class MeshComponent : public Component {
public:
    BoundingBox GetBoundingBox() const {
        return boundingBox;
    }

    void SetBoundingBox(const BoundingBox& box) {
        boundingBox = box;
    }

private:
    BoundingBox boundingBox;
};
