#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Resource/core/Resource.h"

#include <optional>

class Shader : public Resource {
public:
    explicit Shader(const std::string& id);

    vk::ShaderModule getShaderModule() const { return static_cast<vk::ShaderModule>(*shaderModule); }
    vk::ShaderStageFlagBits getStage() const { return stage; }

protected:
    bool doLoad() override;
    void doUnload() override;

private:
    bool loadShaderCode(const std::string& filePath, std::vector<char>& buffer);
    void createShaderModule(const std::vector<char>& code);

    std::optional<vk::raii::ShaderModule> shaderModule;
    vk::ShaderStageFlagBits stage;
};

