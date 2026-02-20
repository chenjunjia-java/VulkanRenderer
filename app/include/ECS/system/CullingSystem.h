#pragma once

#include "ECS/entity/Entity.h"
#include "ECS/component/MeshComponent.h"
#include "ECS/component/TransformComponent.h"
#include "Engine/Camera/Camera.h"
#include "Engine/Math/Frustum.h"

#include <vector>

class CullingSystem {
public:
    CullingSystem() : camera(nullptr) {}
    explicit CullingSystem(Camera* cam) : camera(cam) {}

    void SetCamera(Camera* cam) { camera = cam; }

    void CullScene(const std::vector<Entity*>& allEntities,
                   float aspectRatio = 16.0f / 9.0f,
                   float nearPlane = 0.1f,
                   float farPlane = 10.0f);

    const std::vector<Entity*>& GetVisibleEntities() const {
        return visibleEntities;
    }

private:
    Camera* camera = nullptr;
    std::vector<Entity*> visibleEntities;
};
