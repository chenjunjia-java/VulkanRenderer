#pragma once

#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Resource/core/Resource.h"

#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>

template <typename T>
class ResourceHandle;

class ResourceManager {
public:
    ResourceManager() = default;

    void init(VulkanContext& context);
    void cleanup();

    template <typename T>
    ResourceHandle<T> Load(const std::string& resourceId);

    template <typename T>
    T* GetResource(const std::string& resourceId);

    template <typename T>
    bool HasResource(const std::string& resourceId);

    void Release(const std::string& resourceId, std::type_index type);
    void AddRef(const std::string& resourceId, std::type_index type);
    void UnloadAll();

    VulkanResourceCreator* getResourceCreator() { return &vulkanResourceCreator; }
    vk::raii::Device& getDevice() { return vulkanResourceCreator.getDevice(); }
    const vk::raii::Device& getDevice() const { return vulkanResourceCreator.getDevice(); }

private:
    using ResourceMap = std::unordered_map<std::string, std::shared_ptr<Resource>>;
    using ResourceStorage = std::unordered_map<std::type_index, ResourceMap>;

    struct RefCountKeyHash {
        size_t operator()(const std::pair<std::type_index, std::string>& p) const {
            return std::hash<std::type_index>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::type_index, std::string>, int, RefCountKeyHash> refCounts;

    ResourceStorage resources;
    VulkanResourceCreator vulkanResourceCreator;
};

#include "Resource/core/ResourceHandle.h"

template <typename T>
ResourceHandle<T> ResourceManager::Load(const std::string& resourceId)
{
    static_assert(std::is_base_of_v<Resource, T>, "T must derive from Resource");

    std::type_index type(typeid(T));
    auto& typeResources = resources[type];
    auto it = typeResources.find(resourceId);

    if (it != typeResources.end()) {
        refCounts[{type, resourceId}]++;
        return ResourceHandle<T>(resourceId, this);
    }

    auto resource = std::make_shared<T>(resourceId);
    resource->SetResourceManager(this);
    if (!resource->Load()) {
        return ResourceHandle<T>();
    }

    typeResources[resourceId] = resource;
    refCounts[{type, resourceId}] = 1;

    return ResourceHandle<T>(resourceId, this);
}

template <typename T>
T* ResourceManager::GetResource(const std::string& resourceId)
{
    auto it = resources.find(std::type_index(typeid(T)));
    if (it == resources.end()) return nullptr;

    auto resourceIt = it->second.find(resourceId);
    if (resourceIt == it->second.end()) return nullptr;

    return static_cast<T*>(resourceIt->second.get());
}

template <typename T>
bool ResourceManager::HasResource(const std::string& resourceId)
{
    auto it = resources.find(std::type_index(typeid(T)));
    if (it == resources.end()) return false;
    return it->second.find(resourceId) != it->second.end();
}

