#pragma once

#include <string>

class ResourceManager;

class Resource {
public:
    explicit Resource(const std::string& id);
    virtual ~Resource() = default;

    const std::string& GetId() const { return resourceId; }
    bool IsLoaded() const { return loaded; }

    void SetResourceManager(ResourceManager* rm) { resourceManager = rm; }
    ResourceManager* GetResourceManager() const { return resourceManager; }

    bool Load();
    void Unload();

protected:
    virtual bool doLoad() = 0;
    virtual void doUnload() = 0;

private:
    std::string resourceId;
    bool loaded = false;
    ResourceManager* resourceManager = nullptr;
};

