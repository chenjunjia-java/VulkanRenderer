#pragma once

// System
#include <string>

class Model;

class IModelLoader {
public:
    virtual ~IModelLoader() = default;
    virtual bool loadFromFile(const std::string& filePath, Model& outModel) = 0;
};

