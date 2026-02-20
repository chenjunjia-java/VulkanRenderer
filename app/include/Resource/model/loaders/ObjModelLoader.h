#pragma once

// Project
#include "Resource/model/loaders/IModelLoader.h"

class ObjModelLoader final : public IModelLoader {
public:
    bool loadFromFile(const std::string& filePath, Model& outModel) override;
};

