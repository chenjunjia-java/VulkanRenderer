#include "Resource/core/Resource.h"

bool Resource::Load()
{
    loaded = doLoad();
    return loaded;
}

void Resource::Unload()
{
    doUnload();
    loaded = false;
}

Resource::Resource(const std::string& id) : resourceId(id) {}

