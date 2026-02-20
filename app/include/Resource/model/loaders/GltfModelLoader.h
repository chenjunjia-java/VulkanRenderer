#pragma once

// Project
#include "Resource/model/loaders/IModelLoader.h"

class GltfModelLoader final : public IModelLoader {
public:
    bool loadFromFile(const std::string& filePath, Model& outModel) override;
};

