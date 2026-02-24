#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Engine/Events/EventBus.h"
#include "Rendering/renderer/Renderer.h"
#include "Engine/Camera/Camera.h"
#include "ECS/core/Scene.h"
#include "ECS/system/CullingSystem.h"

class VulkanApplication {
public:
    VulkanApplication() = default;

    void run();

    enum class InputMode {
        Auto,
        CameraControl,
        UiInteraction
    };

private:
    void initWindow();
    void mainLoop();
    void cleanup();
    void processInput(float deltaTime);
    void setInputMode(InputMode mode);
    void toggleInputMode();
    bool canProcessCameraKeyboard() const;
    bool canProcessCameraMouse() const;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void charCallback(GLFWwindow* window, unsigned int c);

    GLFWwindow* window = nullptr;
    EventBus eventBus;
    EventBus::Subscription framebufferResizeSub;
    Renderer renderer;
    Camera camera;
    Scene scene;
    CullingSystem cullingSystem;

    bool frustumCullingEnabled = false;
    bool occlusionCullingEnabled = false;
    bool prevF1 = false;
    bool prevF2 = false;
    bool prevF3 = false;
    InputMode inputMode = InputMode::Auto;
};

