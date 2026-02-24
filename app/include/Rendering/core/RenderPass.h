#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <glm/mat4x4.hpp>

class Camera;

struct RenderStats {
    // Draw/管线统计
    uint64_t depthDrawCalls = 0;
    uint64_t forwardDrawCalls = 0;
    uint64_t opaqueItems = 0;
    uint64_t transparentItems = 0;
    uint64_t forwardPipelineBinds = 0;
    uint64_t forwardDescriptorBinds = 0;
    uint64_t forwardVertexBufferBinds = 0;
    uint64_t forwardIndexBufferBinds = 0;
    double forwardCollectMs = 0.0;
    double forwardSortMs = 0.0;
    double forwardIssueMs = 0.0;

    // Pass 级别 CPU 耗时（ms），由 Rendergraph 按 pass 名称写入
    double depthPrepassMs = 0.0;
    double rtaoMs = 0.0;
    double skyboxMs = 0.0;
    double forwardMs = 0.0;
    double bloomExtractMs = 0.0;
    double bloomBlurHMs = 0.0;
    double bloomBlurVMs = 0.0;
    double tonemapMs = 0.0;
    double occlusionMs = 0.0;
};

struct PassExecuteContext {
    vk::raii::CommandBuffer& commandBuffer;
    uint32_t imageIndex = 0;
    glm::mat4 modelMatrix{1.0f};
    const Camera* camera = nullptr;
    RenderStats* stats = nullptr;
};

class RenderPass {
public:
    RenderPass(std::string passName, std::vector<std::string> inputResources, std::vector<std::string> outputResources);
    virtual ~RenderPass() = default;

    const std::string& getName() const { return name; }
    const std::vector<std::string>& getInputs() const { return inputs; }
    const std::vector<std::string>& getOutputs() const { return outputs; }
    virtual std::optional<vk::ImageLayout> getRequiredInputLayout(const std::string& /*resource*/) const { return std::nullopt; }
    virtual std::optional<vk::ImageLayout> getRequiredOutputLayout(const std::string& /*resource*/) const { return std::nullopt; }

    void execute(const PassExecuteContext& ctx);

protected:
    virtual void beginPass(const PassExecuteContext& ctx) = 0;
    virtual void render(const PassExecuteContext& ctx) = 0;
    virtual void endPass(const PassExecuteContext& ctx) = 0;

    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

