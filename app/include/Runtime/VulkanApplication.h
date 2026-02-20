#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Engine/Events/EventBus.h"
#include "Engine/Events/Events.h"
#include "Rendering/renderer/Renderer.h"
#include "Engine/Camera/Camera.h"
#include "ECS/core/Scene.h"
#include "ECS/system/CullingSystem.h"

class VulkanApplication {
public:
    VulkanApplication() = default;

    void run();

private:
    void initWindow();
    void mainLoop();
    void cleanup();
    void processInput(float deltaTime);

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    GLFWwindow* window = nullptr;
    EventBus eventBus;
    EventBus::Subscription framebufferResizeSub;
    Renderer renderer;
    Camera camera;
    Scene scene;
    CullingSystem cullingSystem;
};

