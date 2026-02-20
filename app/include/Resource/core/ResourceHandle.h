#pragma once

class ResourceManager;

#include <string>
#include <typeindex>

template <typename T>
class ResourceHandle {
public:
    ResourceHandle() : resourceManager(nullptr) {}

    ResourceHandle(const std::string& id, ResourceManager* manager)
        : resourceId(id), resourceManager(manager) {}

    ~ResourceHandle()
    {
        if (resourceManager) {
            resourceManager->Release(resourceId, std::type_index(typeid(T)));
        }
    }

    ResourceHandle(const ResourceHandle& other)
        : resourceId(other.resourceId), resourceManager(other.resourceManager)
    {
        if (resourceManager) {
            resourceManager->AddRef(resourceId, std::type_index(typeid(T)));
        }
    }

    ResourceHandle& operator=(const ResourceHandle& other)
    {
        if (this != &other) {
            if (resourceManager) {
                resourceManager->Release(resourceId, std::type_index(typeid(T)));
            }
            resourceId = other.resourceId;
            resourceManager = other.resourceManager;
            if (resourceManager) {
                resourceManager->AddRef(resourceId, std::type_index(typeid(T)));
            }
        }
        return *this;
    }

    ResourceHandle(ResourceHandle&& other) noexcept
        : resourceId(std::move(other.resourceId)), resourceManager(other.resourceManager)
    {
        other.resourceManager = nullptr;
    }

    ResourceHandle& operator=(ResourceHandle&& other) noexcept
    {
        if (this != &other) {
            if (resourceManager) {
                resourceManager->Release(resourceId, std::type_index(typeid(T)));
            }
            resourceId = std::move(other.resourceId);
            resourceManager = other.resourceManager;
            other.resourceManager = nullptr;
        }
        return *this;
    }

    T* Get() const
    {
        if (!resourceManager) return nullptr;
        return resourceManager->template GetResource<T>(resourceId);
    }

    bool IsValid() const
    {
        return resourceManager && resourceManager->template HasResource<T>(resourceId);
    }

    const std::string& GetId() const { return resourceId; }

    T* operator->() const { return Get(); }
    T& operator*() const { return *Get(); }
    explicit operator bool() const { return IsValid(); }

private:
    std::string resourceId;
    ResourceManager* resourceManager;
};

