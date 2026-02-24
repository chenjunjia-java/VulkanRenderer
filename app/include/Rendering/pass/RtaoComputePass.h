#pragma once

#include "Rendering/core/FrameManager.h"
#include "Rendering/core/RenderPass.h"
#include "Rendering/pipeline/RtaoComputePipeline.h"
#include "Rendering/RHI/Vulkan/RayTracingContext.h"

#include <optional>

class RtaoComputePass : public RenderPass {
public:
    RtaoComputePass(vk::raii::Device& device, RtaoComputePipeline& pipeline, FrameManager& frameManager, RayTracingContext& rayTracingContext);
    ~RtaoComputePass() override = default;

protected:
    void beginPass(const PassExecuteContext& ctx) override;
    void render(const PassExecuteContext& ctx) override;
    void endPass(const PassExecuteContext& ctx) override;

private:
    struct PushParams {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t step = 1;
        uint32_t iteration = 0;
    };

    void createDescriptorPool();
    void createDescriptorSets();
    void updateDescriptorsForFrame(uint32_t frameIndex);
    void dispatchTrace(vk::raii::CommandBuffer& cb, uint32_t frameIndex);
    void dispatchAtrous(vk::raii::CommandBuffer& cb, uint32_t frameIndex);
    void dispatchUpsample(vk::raii::CommandBuffer& cb, uint32_t frameIndex);

    vk::raii::Device* device = nullptr;
    RtaoComputePipeline* pipeline = nullptr;
    FrameManager* frameManager = nullptr;
    RayTracingContext* rayTracingContext = nullptr;
    std::optional<vk::raii::DescriptorPool> descriptorPool;
    std::optional<vk::raii::DescriptorSets> descriptorSets;
};

