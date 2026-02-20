#include "ECS/entity/Entity.h"

void Entity::Initialize()
{
    for (auto& component : components) {
        component->Initialize();
    }
}

void Entity::Update(float deltaTime)
{
    if (!active) return;

    for (auto& component : components) {
        component->Update(deltaTime);
    }
}

void Entity::Render()
{
    if (!active) return;

    for (auto& component : components) {
        component->Render();
    }
}
