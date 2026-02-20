#include "ECS/core/Scene.h"

Entity* Scene::AddEntity(const std::string& name)
{
    auto entity = std::make_unique<Entity>(name);
    Entity* entityPtr = entity.get();
    entities.push_back(std::move(entity));
    return entityPtr;
}

void Scene::Initialize()
{
    for (auto& entity : entities) {
        entity->Initialize();
    }
}

void Scene::Update(float deltaTime)
{
    for (auto& entity : entities) {
        entity->Update(deltaTime);
    }
}

void Scene::Render()
{
    for (auto& entity : entities) {
        entity->Render();
    }
}

std::vector<Entity*> Scene::GetEntities()
{
    std::vector<Entity*> result;
    result.reserve(entities.size());
    for (auto& entity : entities) {
        result.push_back(entity.get());
    }
    return result;
}
