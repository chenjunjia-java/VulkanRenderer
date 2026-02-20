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
#include "Rendering/mesh/GpuMesh.h"
#include "Resource/core/ResourceHandle.h"
#include "Resource/core/ResourceManager.h"
#include "Resource/model/Model.h"
#include "Resource/shader/Shader.h"
#include "Resource/texture/Texture.h"

#include <optional>

class Renderer {
public:
    Renderer() = default;

    void init(GLFWwindow* window);
    void cleanup();

    void drawFrame();
    void waitIdle();

    void setFramebufferResized(bool resized) { frameManager.setFramebufferResized(resized); }
    void setCamera(const Camera* cam) { camera = cam; }
    void setCullingSystem(CullingSystem* sys) { cullingSystem = sys; }

private:
    void recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex, const glm::mat4& modelMatrix);
    glm::mat4 computeSceneModelMatrix() const;

    VulkanContext vulkanContext;
    SwapChain swapChain;
    ResourceManager resourceManager;
    ResourceHandle<Texture> textureHandle;
    ResourceHandle<Model> modelHandle;
    ResourceHandle<Shader> vertShaderHandle;
    ResourceHandle<Shader> fragShaderHandle;
    GraphicsPipeline graphicsPipeline;
    RayTracingContext rayTracingContext;
    std::optional<Rendergraph> rendergraph;
    FrameManager frameManager;
    GpuMesh sceneMesh;

    GLFWwindow* window = nullptr;
    const Camera* camera = nullptr;
    CullingSystem* cullingSystem = nullptr;
};

