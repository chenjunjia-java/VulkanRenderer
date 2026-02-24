#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

class VulkanContext;
class SwapChain;
class VulkanResourceCreator;

/// ImGui 集成模块（独立于主渲染管线）
/// 使用 Vulkan SDK 自带的 imgui，通过 imgui_impl_glfw + imgui_impl_vulkan 实现
/// 注意：类名避免与 ImGui 内部 ImGuiContext 冲突
class ImGuiIntegration {
public:
    struct UiStats {
        double acquireMs = 0.0;
        double recordMs = 0.0;
        double updateUboMs = 0.0;
        double submitMs = 0.0;
        double presentMs = 0.0;
        double totalMs = 0.0;
        uint64_t swapchainRecreateCount = 0;
        uint64_t frameCounter = 0;
    };

    ImGuiIntegration() = default;
    ~ImGuiIntegration();

    void init(VulkanContext& vulkanContext, VulkanResourceCreator& resourceCreator, SwapChain& swapChain, GLFWwindow* window);
    void cleanup();

    /// 每帧开始时调用
    void newFrame();
    void addPanel(const std::string& name, std::function<void()> drawFn);
    void clearPanels();

    /// 在 commandBuffer 中录制 ImGui 绘制命令
    /// 调用前需确保 commandBuffer 已 begin，且 rendergraph 已执行完毕（swapchain 图像已有内容）
    /// swapchainImage：当前帧的 swapchain 图像句柄
    /// swapchainImageView：对应的 image view
    void render(vk::raii::CommandBuffer& commandBuffer, vk::Image swapchainImage, vk::ImageView swapchainImageView,
                vk::Extent2D extent, vk::Format format);

    /// SwapChain 重建时调用，更新 MinImageCount
    void setMinImageCount(uint32_t minImageCount);
    void onSwapchainRecreated(SwapChain& swapChain, GLFWwindow* window);
    void setUiStats(const UiStats& stats) { uiStats = stats; }

    /// 是否启用 ImGui（用于按需开关）
    bool isEnabled() const { return enabled; }
    void setEnabled(bool e) { enabled = e; }

    /// 当 ImGui 需要捕获输入时返回 true，主循环可据此跳过相机/游戏输入
    bool getWantCaptureMouse() const;
    bool getWantCaptureKeyboard() const;
    bool getWantTextInput() const;

private:
    void createDescriptorPool(vk::raii::Device& device);
    void updateDisplayState(SwapChain& swapChain, GLFWwindow* window);
    void buildDefaultUi();

    vk::raii::Device* device = nullptr;
    std::optional<vk::raii::DescriptorPool> descriptorPool;
    uint32_t minImageCount = 2;
    uint32_t imageCount = 0;
    vk::Format swapchainFormat = vk::Format::eUndefined;
    bool initialized = false;
    bool enabled = true;
    UiStats uiStats{};
    std::vector<std::pair<std::string, std::function<void()>>> panels;
};
