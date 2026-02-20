#include "ECS/system/CullingSystem.h"

void CullingSystem::CullScene(const std::vector<Entity*>& allEntities,
                              float aspectRatio,
                              float nearPlane,
                              float farPlane)
{
    visibleEntities.clear();

    if (!camera) return;

    Frustum frustum = camera->GetFrustum(aspectRatio, nearPlane, farPlane);

    for (Entity* entity : allEntities) {
        if (!entity->IsActive()) continue;

        auto* meshComponent = entity->GetComponent<MeshComponent>();
        if (!meshComponent) continue;

        auto* transformComponent = entity->GetComponent<TransformComponent>();
        if (!transformComponent) continue;

        BoundingBox boundingBox = meshComponent->GetBoundingBox();
        boundingBox.Transform(transformComponent->GetTransformMatrix());

        if (frustum.Intersects(boundingBox)) {
            visibleEntities.push_back(entity);
        }
    }
}
