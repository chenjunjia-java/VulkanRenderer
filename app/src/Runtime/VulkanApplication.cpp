#include "Runtime/VulkanApplication.h"
#include "Rendering/RHI/Vulkan/VulkanTypes.h"

#include <chrono>

void VulkanApplication::run()
{
    initWindow();
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
}

void VulkanApplication::initWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
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
        cullingSystem.CullScene(scene.GetEntities(), static_cast<float>(WIDTH) / HEIGHT, 0.1f, 10.0f);
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

    camera.processInput(deltaTime, window);
}

void VulkanApplication::mouseCallback(GLFWwindow* win, double xpos, double ypos)
{
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (app && app->window) {
        bool rightPressed = (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        app->camera.processMousePosition(xpos, ypos, rightPressed);
    }
}

void VulkanApplication::scrollCallback(GLFWwindow* win, double xoffset, double yoffset)
{
    (void)xoffset;
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(win));
    if (app && app->window) {
        app->camera.processMouseScroll(static_cast<float>(yoffset));
    }
}

