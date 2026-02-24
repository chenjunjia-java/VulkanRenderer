#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Engine/Camera/Camera.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/RayTracingContext.h"
#include "ECS/system/CullingSystem.h"
#include "Rendering/core/FrameManager.h"
#include "Rendering/core/Rendergraph.h"
#include "Rendering/pipeline/GraphicsPipeline.h"
#include "Rendering/pipeline/DepthPrepassPipeline.h"
#include "Rendering/pipeline/SkyboxPipeline.h"
#include "Rendering/pipeline/RtaoComputePipeline.h"
#include "Rendering/pipeline/PostProcessPipeline.h"
#include "Rendering/pass/SkyboxPass.h"
#include "Rendering/ibl/EquirectToCubemap.h"
#include "Rendering/ibl/IblPrecompute.h"
#include "Rendering/mesh/GpuMesh.h"
#include "Rendering/mesh/GlobalMeshBuffer.h"
#include "Resource/core/ResourceHandle.h"
#include "Resource/core/ResourceManager.h"
#include "Resource/model/Model.h"
#include "Resource/shader/Shader.h"
#include "Rendering/animation/AnimationPlayer.h"
#include "ImGuiIntegration/ImGuiContext.h"

#include <glm/glm.hpp>
#include <optional>
#include <vector>

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    void init(GLFWwindow* window);
    void cleanup();

    void drawFrame();
    void update(float deltaTime);
    void waitIdle();

    void setFramebufferResized(bool resized) { frameManager.setFramebufferResized(resized); }
    void setCamera(const Camera* cam) { camera = cam; }
    bool getWantCaptureMouse() const { return imguiIntegration.getWantCaptureMouse(); }
    bool getWantCaptureKeyboard() const { return imguiIntegration.getWantCaptureKeyboard(); }
    bool getWantTextInput() const { return imguiIntegration.getWantTextInput(); }
    void setCullingSystem(CullingSystem* sys) { cullingSystem = sys; }
    /// Call when model transform changes (rotation, scale, etc.) so TLAS is rebuilt next frame.
    void invalidateTlas() { tlasNeedsUpdate = true; }

private:
    void recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex, const glm::mat4& modelMatrix);
    glm::mat4 computeSceneModelMatrix() const;
    void rebuildRayTracingInstances(const glm::mat4& modelMatrix);

    VulkanContext vulkanContext;
    SwapChain swapChain;
    ResourceManager resourceManager;
    ResourceHandle<Model> modelHandle;
    ResourceHandle<Shader> vertShaderHandle;
    ResourceHandle<Shader> fragShaderHandle;
    ResourceHandle<Shader> skyboxVertShaderHandle;
    ResourceHandle<Shader> skyboxFragShaderHandle;
    ResourceHandle<Shader> depthOnlyFragShaderHandle;
    ResourceHandle<Shader> depthPrepassVertShaderHandle;
    ResourceHandle<Shader> rtaoTraceCompShaderHandle;
    ResourceHandle<Shader> rtaoAtrousCompShaderHandle;
    ResourceHandle<Shader> rtaoUpsampleCompShaderHandle;
    ResourceHandle<Shader> fullscreenVertShaderHandle;
    ResourceHandle<Shader> bloomExtractFragShaderHandle;
    ResourceHandle<Shader> bloomBlurFragShaderHandle;
    ResourceHandle<Shader> tonemapBloomFragShaderHandle;
    GraphicsPipeline graphicsPipeline;
    DepthPrepassPipeline depthPrepassPipeline;
    RtaoComputePipeline rtaoComputePipeline;
    SkyboxPipeline skyboxPipeline;
    PostProcessPipeline postProcessPipeline;
    RayTracingContext rayTracingContext;
    std::optional<Rendergraph> rendergraph;
    FrameManager frameManager;
    std::vector<GpuMesh> modelMeshes;
    GlobalMeshBuffer globalMeshBuffer;
    uint32_t maxDraws = 0;
    CubemapResult envCubemapResult;
    IblResult iblResult;
    std::vector<RayTracingInstanceDesc> rayTracingInstances;
    AnimationPlayer animationPlayer;
    ImGuiIntegration imguiIntegration;

    bool tlasNeedsUpdate = true;  // true on init; set false after TLAS build; set true on animation/transform change
    std::optional<glm::mat4> cachedModelMatrixForTlas;  // detect scene-level rotation/scale change

    // Per-frame stats (for lightweight CPU-side validation / profiling output).
    RenderStats lastRenderStats{};
    struct CpuTimings {
        double acquireMs = 0.0;
        double recordMs = 0.0;
        double updateUboMs = 0.0;
        double submitMs = 0.0;
        double presentMs = 0.0;
        double totalMs = 0.0;
    };
    CpuTimings lastCpuTimings{};
    CpuTimings accumCpuTimings{};
    uint32_t accumFrames = 0;
    uint64_t swapchainRecreateCount = 0;
    uint64_t frameCounter = 0;

    GLFWwindow* window = nullptr;
    const Camera* camera = nullptr;
    CullingSystem* cullingSystem = nullptr;
};

