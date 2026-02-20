#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"

#include <memory>
#include <string>
#include <vector>

struct PassExecuteContext {
    vk::raii::CommandBuffer& commandBuffer;
    uint32_t imageIndex = 0;
};

class RenderPass {
public:
    RenderPass(std::string passName, std::vector<std::string> inputResources, std::vector<std::string> outputResources);
    virtual ~RenderPass() = default;

    const std::string& getName() const { return name; }
    const std::vector<std::string>& getInputs() const { return inputs; }
    const std::vector<std::string>& getOutputs() const { return outputs; }

    void execute(const PassExecuteContext& ctx);

protected:
    virtual void beginPass(const PassExecuteContext& ctx) = 0;
    virtual void render(const PassExecuteContext& ctx) = 0;
    virtual void endPass(const PassExecuteContext& ctx) = 0;

    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

