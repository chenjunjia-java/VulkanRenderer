#include "Runtime/VulkanApplication.h"
#include "Configs/AppConfig.h"
#include "Engine/Events/Events.h"

#include "imgui_impl_glfw.h"

#include <chrono>

void VulkanApplication::run()
{
    initWindow();
    glfwShowWindow(window);  // 确保窗口显示，避免 SwapChain 初始化时 framebuffer 尺寸为 0 导致无限阻塞

    try {
        renderer.init(window);
        renderer.setCamera(&camera);
        renderer.setCullingSystem(&cullingSystem);
        cullingSystem.SetCamera(&camera);

        framebufferResizeSub = eventBus.subscribe<FramebufferResizeEvent>([this](const FramebufferResizeEvent& e) {
            (void)e;
            renderer.setFramebufferResized(true);
        });

        mainLoop();
        cleanup();
    } catch (...) {
        // Ensure Vulkan resources are destroyed before VkDevice is destroyed.
        cleanup();
        throw;
    }
}

void VulkanApplication::initWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(AppConfig::WIDTH, AppConfig::HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCharCallback(window, charCallback);
    setInputMode(InputMode::Auto);
}

void VulkanApplication::mainLoop()
{
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        lastTime = currentTime;

        processInput(deltaTime);
        renderer.update(deltaTime);
        cullingSystem.CullScene(scene.GetEntities(), static_cast<float>(AppConfig::WIDTH) / AppConfig::HEIGHT, 0.1f, 10.0f);
        renderer.drawFrame();
        // End-of-frame: deliver queued events (e.g. window resize).
        eventBus.process();
    }

    renderer.waitIdle();
}

void VulkanApplication::cleanup()
{
    renderer.cleanup();

    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

void VulkanApplication::framebufferResizeCallback(GLFWwindow* win, int width, int height)
{
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (app && app->window) {
        app->eventBus.enqueue(FramebufferResizeEvent{ width, height });
    }
}

void VulkanApplication::processInput(float deltaTime)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    const bool f3Pressed = (glfwGetKey(window, GLFW_KEY_F3) == GLFW_PRESS);
    if (f3Pressed && !prevF3) {
        toggleInputMode();
    }
    prevF3 = f3Pressed;

    if (canProcessCameraKeyboard()) {
        camera.processInput(deltaTime, window);
    }
}

void VulkanApplication::mouseCallback(GLFWwindow* win, double xpos, double ypos)
{
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (!app || !app->window) return;
#ifndef IMGUI_DISABLE
    if (app->inputMode != InputMode::CameraControl) {
        ImGui_ImplGlfw_CursorPosCallback(win, xpos, ypos);
    }
#endif
    if (app->canProcessCameraMouse()) {
        bool rightPressed = (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        app->camera.processMousePosition(xpos, ypos, rightPressed);
    }
}

void VulkanApplication::scrollCallback(GLFWwindow* win, double xoffset, double yoffset)
{
    (void)xoffset;
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (!app || !app->window) return;
#ifndef IMGUI_DISABLE
    if (app->inputMode != InputMode::CameraControl) {
        ImGui_ImplGlfw_ScrollCallback(win, xoffset, yoffset);
    }
#endif
    if (app->canProcessCameraMouse()) {
        app->camera.processMouseScroll(static_cast<float>(yoffset));
    }
}

void VulkanApplication::mouseButtonCallback(GLFWwindow* win, int button, int action, int mods)
{
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (!app || !app->window) return;
    (void)mods;
#ifndef IMGUI_DISABLE
    if (app->inputMode != InputMode::CameraControl) {
        ImGui_ImplGlfw_MouseButtonCallback(win, button, action, mods);
    }
#endif
}

void VulkanApplication::keyCallback(GLFWwindow* win, int key, int scancode, int action, int mods)
{
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (!app || !app->window) return;
#ifndef IMGUI_DISABLE
    if (app->inputMode != InputMode::CameraControl) {
        ImGui_ImplGlfw_KeyCallback(win, key, scancode, action, mods);
    }
#endif
}

void VulkanApplication::charCallback(GLFWwindow* win, unsigned int c)
{
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (!app || !app->window) return;
#ifndef IMGUI_DISABLE
    if (app->inputMode != InputMode::CameraControl) {
        ImGui_ImplGlfw_CharCallback(win, c);
    }
#endif
}

void VulkanApplication::setInputMode(InputMode mode)
{
    inputMode = mode;
    if (!window) return;
    if (inputMode == InputMode::CameraControl) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void VulkanApplication::toggleInputMode()
{
    // Cycle: Auto -> CameraControl -> UiInteraction -> Auto
    if (inputMode == InputMode::Auto) {
        setInputMode(InputMode::CameraControl);
    } else if (inputMode == InputMode::CameraControl) {
        setInputMode(InputMode::UiInteraction);
    } else {
        setInputMode(InputMode::Auto);
    }
}

bool VulkanApplication::canProcessCameraKeyboard() const
{
    if (inputMode == InputMode::UiInteraction) return false;
    if (inputMode == InputMode::CameraControl) return true;
    // Auto: UI has priority when it wants the keyboard.
    // Only block camera movement when the UI is actively editing text.
    return !renderer.getWantTextInput();
}

bool VulkanApplication::canProcessCameraMouse() const
{
    if (inputMode == InputMode::UiInteraction) return false;
    if (inputMode == InputMode::CameraControl) return true;
    // Auto: UI has priority when it wants the mouse.
    return !renderer.getWantCaptureMouse();
}

