#include "ImGuiIntegration/ImGuiContext.h"
#include "Rendering/RHI/Vulkan/VulkanContext.h"
#include "Rendering/RHI/Vulkan/SwapChain.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Configs/RuntimeConfig.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#ifndef IMGUI_DISABLE

static void checkVkResult(VkResult err)
{
    if (err == VK_SUCCESS) return;
    throw std::runtime_error("[ImGui Vulkan] VkResult = " + std::to_string(static_cast<int>(err)));
}

ImGuiIntegration::~ImGuiIntegration()
{
    if (initialized) {
        try {
            cleanup();
        } catch (...) {}
    }
}

void ImGuiIntegration::init(VulkanContext& vulkanContext, VulkanResourceCreator& resourceCreator, SwapChain& swapChain, GLFWwindow* window)
{
    if (initialized) return;

    device = &vulkanContext.getDevice();
    swapchainFormat = static_cast<vk::Format>(swapChain.getImageFormat());
    imageCount = static_cast<uint32_t>(swapChain.getImages().size());
    minImageCount = std::max(2u, imageCount > 1u ? imageCount - 1u : imageCount);

    createDescriptorPool(vulkanContext.getDevice());

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
#ifdef ImGuiConfigFlags_DockingEnable
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
#endif

    updateDisplayState(swapChain, window);

    ImGui::StyleColorsDark();

    // We manage GLFW callbacks ourselves (to route input between UI and 3D).
    // So do NOT let ImGui install/override callbacks here.
    ImGui_ImplGlfw_InitForVulkan(window, false);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = static_cast<VkInstance>(*vulkanContext.getInstance());
    initInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(*vulkanContext.getPhysicalDevice());
    initInfo.Device = static_cast<VkDevice>(*vulkanContext.getDevice());
    initInfo.QueueFamily = vulkanContext.findQueueFamilies(vulkanContext.getPhysicalDevice()).graphicsFamily.value();
    initInfo.Queue = static_cast<VkQueue>(static_cast<vk::Queue>(vulkanContext.getGraphicsQueue()));
    initInfo.DescriptorPool = static_cast<VkDescriptorPool>(static_cast<vk::DescriptorPool>(*descriptorPool));
    initInfo.MinImageCount = minImageCount;
    initInfo.ImageCount = imageCount;
    initInfo.MinAllocationSize = 1024 * 1024;
    initInfo.CheckVkResultFn = checkVkResult;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = reinterpret_cast<const VkFormat*>(&swapchainFormat);

    ImGui_ImplVulkan_Init(&initInfo);
    (void)resourceCreator;

    initialized = true;
}

void ImGuiIntegration::createDescriptorPool(vk::raii::Device& dev)
{
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 32 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 128 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 64 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 64 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 32 }
    };
    uint32_t maxSets = 0;
    for (const auto& ps : poolSizes) {
        maxSets += ps.descriptorCount;
    }
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(sizeof(poolSizes) / sizeof(poolSizes[0]));
    poolInfo.pPoolSizes = poolSizes;

    descriptorPool.emplace(dev, static_cast<vk::DescriptorPoolCreateInfo>(poolInfo));
}

void ImGuiIntegration::cleanup()
{
    if (!initialized) return;

    if (device) {
        device->waitIdle();
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    descriptorPool.reset();
    panels.clear();

    initialized = false;
}

void ImGuiIntegration::newFrame()
{
    if (!enabled || !initialized) return;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    buildDefaultUi();
    for (const auto& panel : panels) {
        if (panel.second) {
            panel.second();
        }
    }
}

void ImGuiIntegration::addPanel(const std::string& name, std::function<void()> drawFn)
{
    if (name.empty() || !drawFn) return;
    panels.emplace_back(name, std::move(drawFn));
}

void ImGuiIntegration::clearPanels()
{
    panels.clear();
}

void ImGuiIntegration::render(vk::raii::CommandBuffer& commandBuffer, vk::Image swapchainImage, vk::ImageView swapchainImageView,
                          vk::Extent2D extent, vk::Format format)
{
    (void)format;
    if (!enabled || !initialized) return;

    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || drawData->CmdListsCount == 0) return;

    int fbWidth = static_cast<int>(drawData->DisplaySize.x * drawData->FramebufferScale.x);
    int fbHeight = static_cast<int>(drawData->DisplaySize.y * drawData->FramebufferScale.y);
    if (fbWidth <= 0 || fbHeight <= 0) return;

    vk::CommandBuffer vkCb = static_cast<vk::CommandBuffer>(commandBuffer);

    vk::ImageMemoryBarrier barrierToColor{};
    barrierToColor.oldLayout = vk::ImageLayout::ePresentSrcKHR;
    barrierToColor.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
    barrierToColor.image = swapchainImage;
    barrierToColor.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    barrierToColor.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    barrierToColor.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {}, {}, {}, {barrierToColor});

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.imageView = swapchainImageView;
    colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.clearValue = vk::ClearValue{};

    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea = vk::Rect2D{{0, 0}, {static_cast<uint32_t>(extent.width), static_cast<uint32_t>(extent.height)}};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    commandBuffer.beginRendering(renderingInfo);

    ImGui_ImplVulkan_RenderDrawData(drawData, static_cast<VkCommandBuffer>(vkCb), VK_NULL_HANDLE);

    commandBuffer.endRendering();

    vk::ImageMemoryBarrier barrierToPresent{};
    barrierToPresent.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    barrierToPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
    barrierToPresent.image = swapchainImage;
    barrierToPresent.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    barrierToPresent.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    barrierToPresent.dstAccessMask = {};

    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        {}, {}, {}, {barrierToPresent});
}

void ImGuiIntegration::setMinImageCount(uint32_t minImageCount_)
{
    minImageCount = minImageCount_;
    if (initialized) {
        ImGui_ImplVulkan_SetMinImageCount(minImageCount);
    }
}

void ImGuiIntegration::onSwapchainRecreated(SwapChain& swapChain, GLFWwindow* window)
{
    imageCount = static_cast<uint32_t>(swapChain.getImages().size());
    swapchainFormat = swapChain.getImageFormat();
    setMinImageCount(std::max(2u, imageCount > 1u ? imageCount - 1u : imageCount));
    updateDisplayState(swapChain, window);
}

void ImGuiIntegration::updateDisplayState(SwapChain& swapChain, GLFWwindow* window)
{
    if (!initialized && !window) return;
    ImGuiIO& io = ImGui::GetIO();
    const vk::Extent2D extent = swapChain.getExtent();
    io.DisplaySize = ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));

    float scaleX = 1.0f;
    float scaleY = 1.0f;
    if (window) {
        glfwGetWindowContentScale(window, &scaleX, &scaleY);
    }
    io.DisplayFramebufferScale = ImVec2(scaleX, scaleY);
}

void ImGuiIntegration::buildDefaultUi()
{
    ImGui::Begin("Renderer Stats");
    ImGui::Text("Input: Auto (UI capture) | F3: Cycle Auto/Camera/UI");
    ImGui::Separator();

    if (ImGui::Button("Off")) RuntimeConfig::debugViewMode = 0;
    ImGui::SameLine();
    if (ImGui::Button("BaseColor")) RuntimeConfig::debugViewMode = 2;
    ImGui::SameLine();
    if (ImGui::Button("Ng")) RuntimeConfig::debugViewMode = 3;
    ImGui::SameLine();
    if (ImGui::Button("AO")) RuntimeConfig::debugViewMode = 4;

    // If you want to validate IBL contribution with those debug modes, allow toggling it here.
    ImGui::Checkbox("IBL Diffuse", &RuntimeConfig::enableDiffuseIbl);
    if (RuntimeConfig::enableDiffuseIbl) {
        ImGui::SliderFloat("DiffuseStrength", &RuntimeConfig::diffuseIblStrength, 0.0f, 2.0f);
    }
    ImGui::Checkbox("IBL Specular", &RuntimeConfig::enableSpecularIbl);
    if (RuntimeConfig::enableSpecularIbl) {
        ImGui::SliderFloat("SpecularStrength", &RuntimeConfig::specularIblStrength, 0.0f, 2.0f);
    }

    ImGui::Separator();
    ImGui::Text("PostProcess View");
    if (ImGui::Button("Final")) RuntimeConfig::postprocessDebugView = 0;
    ImGui::SameLine();
    if (ImGui::Button("SceneColor")) RuntimeConfig::postprocessDebugView = 1;
    ImGui::SameLine();
    if (ImGui::Button("Bloom")) RuntimeConfig::postprocessDebugView = 2;

    ImGui::Separator();
    ImGui::Text("Bloom");
    ImGui::SliderFloat("Threshold", &RuntimeConfig::bloomThreshold, 0.0f, 5.0f);
    ImGui::SliderFloat("SoftKnee", &RuntimeConfig::bloomSoftKnee, 0.0f, 1.0f);
    ImGui::SliderFloat("Intensity", &RuntimeConfig::bloomIntensity, 0.0f, 2.0f);
    ImGui::SliderFloat("BlurRadius", &RuntimeConfig::bloomBlurRadius, 0.25f, 64.0f);

    ImGui::Separator();
    ImGui::Text("Tonemap");
    ImGui::SliderFloat("Exposure", &RuntimeConfig::tonemapExposure, 0.1f, 4.0f);
    if (ImGui::Button("Reset Defaults")) {
        RuntimeConfig::resetToDefaults();
    }

    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(uiStats.frameCounter));
    ImGui::Text("Swapchain recreates: %llu", static_cast<unsigned long long>(uiStats.swapchainRecreateCount));
    ImGui::Separator();
    ImGui::Text("Acquire: %.3f ms", uiStats.acquireMs);
    ImGui::Text("Record: %.3f ms", uiStats.recordMs);
    ImGui::Text("Update UBO: %.3f ms", uiStats.updateUboMs);
    ImGui::Text("Submit: %.3f ms", uiStats.submitMs);
    ImGui::Text("Present: %.3f ms", uiStats.presentMs);
    ImGui::Text("Total: %.3f ms", uiStats.totalMs);
    ImGui::End();
}

bool ImGuiIntegration::getWantCaptureMouse() const
{
    if (!enabled || !initialized) return false;
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiIntegration::getWantCaptureKeyboard() const
{
    if (!enabled || !initialized) return false;
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImGuiIntegration::getWantTextInput() const
{
    if (!enabled || !initialized) return false;
    return ImGui::GetIO().WantTextInput;
}

#endif
