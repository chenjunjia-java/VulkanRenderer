#pragma once

#include "ECS/entity/Entity.h"

#include <memory>
#include <string>
#include <vector>

class Scene {
public:
    Entity* AddEntity(const std::string& name);

    void Initialize();
    void Update(float deltaTime);
    void Render();

    std::vector<Entity*> GetEntities();

private:
    std::vector<std::unique_ptr<Entity>> entities;
};
