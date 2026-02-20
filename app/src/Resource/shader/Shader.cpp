#include "Resource/shader/Shader.h"

#include "Resource/core/ResourceManager.h"

#include <fstream>
#include <stdexcept>

static vk::ShaderStageFlagBits parseStageFromId(const std::string& id)
{
    if (id.size() >= 5 && id.substr(id.size() - 5) == "_vert") {
        return vk::ShaderStageFlagBits::eVertex;
    }
    if (id.size() >= 5 && id.substr(id.size() - 5) == "_frag") {
        return vk::ShaderStageFlagBits::eFragment;
    }
    if (id.size() >= 5 && id.substr(id.size() - 5) == "_comp") {
        return vk::ShaderStageFlagBits::eCompute;
    }
    return vk::ShaderStageFlagBits::eVertex;
}

Shader::Shader(const std::string& id) : Resource(id), stage(parseStageFromId(id)) {}

bool Shader::loadShaderCode(const std::string& filePath, std::vector<char>& buffer)
{
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    const size_t fileSize = static_cast<size_t>(file.tellg());
    buffer.resize(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    file.close();
    return true;
}

void Shader::createShaderModule(const std::vector<char>& code)
{
    vk::raii::Device& device = GetResourceManager()->getResourceCreator()->getDevice();
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    shaderModule = vk::raii::ShaderModule(device, createInfo);
}

bool Shader::doLoad()
{
    std::string extension;
    std::string subdir;
    switch (stage) {
        case vk::ShaderStageFlagBits::eVertex:
            extension = ".vert";
            subdir = "VertShaders/";
            break;
        case vk::ShaderStageFlagBits::eFragment:
            extension = ".frag";
            subdir = "FragShaders/";
            break;
        case vk::ShaderStageFlagBits::eCompute:
            extension = ".comp";
            subdir = "CompShaders/";
            break;
        default:
            return false;
    }

    std::string baseId = GetId();
    if (baseId.size() >= 5) {
        if (baseId.substr(baseId.size() - 5) == "_vert" || baseId.substr(baseId.size() - 5) == "_frag" ||
            baseId.substr(baseId.size() - 5) == "_comp") {
            baseId = baseId.substr(0, baseId.size() - 5);
        }
    }

    std::string filePath = ASSETS_PATH + "shaders/" + subdir + baseId + extension + ".spv";

    std::vector<char> shaderCode;
    if (!loadShaderCode(filePath, shaderCode)) {
        return false;
    }

    createShaderModule(shaderCode);
    return true;
}

void Shader::doUnload()
{
    shaderModule.reset();
}

