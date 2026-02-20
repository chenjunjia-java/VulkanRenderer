#include "Resource/core/ResourceManager.h"

#include "Resource/model/Model.h"
#include "Resource/shader/Shader.h"
#include "Resource/texture/Texture.h"

void ResourceManager::init(VulkanContext& context)
{
    vulkanResourceCreator.init(context);
}

void ResourceManager::cleanup()
{
    UnloadAll();
    vulkanResourceCreator.cleanup();
}

void ResourceManager::Release(const std::string& resourceId, std::type_index type)
{
    auto key = std::make_pair(type, resourceId);
    auto it = refCounts.find(key);
    if (it == refCounts.end()) return;

    it->second--;
    if (it->second <= 0) {
        auto typeIt = resources.find(type);
        if (typeIt != resources.end()) {
            auto resourceIt = typeIt->second.find(resourceId);
            if (resourceIt != typeIt->second.end()) {
                resourceIt->second->Unload();
                typeIt->second.erase(resourceIt);
            }
        }
        refCounts.erase(it);
    }
}

void ResourceManager::AddRef(const std::string& resourceId, std::type_index type)
{
    auto key = std::make_pair(type, resourceId);
    refCounts[key]++;
}

void ResourceManager::UnloadAll()
{
    for (auto& [type, typeResources] : resources) {
        for (auto& [id, resource] : typeResources) {
            resource->Unload();
        }
        typeResources.clear();
    }
    refCounts.clear();
}

