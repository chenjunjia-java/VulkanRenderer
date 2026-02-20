#include "Rendering/renderer/Renderer.h"
#include "Rendering/pass/ForwardPass.h"

#include <glm/gtc/matrix_transform.hpp>

#include <stdexcept>

glm::mat4 Renderer::computeSceneModelMatrix() const
{
    // Base conversion: imported OBJ is Z-up while scene uses Y-up.
    glm::mat4 zUpToYUp = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    // TLAS animation tutorial style: animate instance transform each frame.
    float time = static_cast<float>(glfwGetTime());
    glm::mat4 spin = glm::rotate(glm::mat4(1.0f), time * glm::radians(35.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    return spin * zUpToYUp;
}

void Renderer::init(GLFWwindow* inWindow)
{
    window = inWindow;

    vulkanContext.init(window);
    swapChain.init(vulkanContext, window);
    resourceManager.init(vulkanContext);

    textureHandle = resourceManager.Load<Texture>("viking_room");
    modelHandle = resourceManager.Load<Model>("viking_room");
    vertShaderHandle = resourceManager.Load<Shader>("pbr_vert");
    fragShaderHandle = resourceManager.Load<Shader>("pbr_frag");

    if (!textureHandle.IsValid() || !modelHandle.IsValid() || !vertShaderHandle.IsValid() || !fragShaderHandle.IsValid()) {
        throw std::runtime_error("failed to load texture, model or shader resource!");
    }
    if (modelHandle->getMeshes().empty()) {
        throw std::runtime_error("loaded model has no meshes!");
    }

    VulkanResourceCreator* resourceCreator = resourceManager.getResourceCreator();
    const Mesh& cpuMesh = modelHandle->getMeshes()[0];
    sceneMesh.upload(*resourceCreator, cpuMesh.vertices, cpuMesh.indices);
    graphicsPipeline.init(vulkanContext, swapChain, *resourceCreator, *vertShaderHandle.Get(), *fragShaderHandle.Get());
    rayTracingContext.init(vulkanContext, *resourceCreator, sceneMesh, computeSceneModelMatrix());

    rendergraph.emplace(vulkanContext.getDevice(), *resourceCreator);
    vk::Format colorFormat = swapChain.getImageFormat();
    vk::Format depthFormat = resourceCreator->findDepthFormat();
    vk::Extent2D extent = swapChain.getExtent();

    rendergraph->AddResource("color_msaa", colorFormat, extent,
                             vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                             vk::ImageAspectFlagBits::eColor, vulkanContext.getMsaaSamples());
    rendergraph->AddResource("depth", depthFormat, extent,
                             vk::ImageUsageFlagBits::eDepthStencilAttachment,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                             vk::ImageAspectFlagBits::eDepth, vulkanContext.getMsaaSamples());
    rendergraph->AddExternalResource("swapchain", colorFormat, extent,
                                     vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

    rendergraph->AddPass(std::make_unique<ForwardPass>(graphicsPipeline, frameManager, sceneMesh, *rendergraph, swapChain));
    rendergraph->Compile();

    frameManager.init(vulkanContext, swapChain, graphicsPipeline, *rendergraph, *resourceCreator,
                      sceneMesh, *textureHandle.Get(), rayTracingContext);
}

void Renderer::cleanup()
{
    if (vulkanContext.hasDevice()) {
        waitIdle();
    }

    frameManager.cleanup(vulkanContext.getDevice());
    if (rendergraph) {
        rendergraph->Cleanup();
    }
    rayTracingContext.cleanup();
    graphicsPipeline.cleanup();
    sceneMesh.reset();
    resourceManager.cleanup();
    swapChain.cleanup();
    vulkanContext.cleanup();
}

void Renderer::drawFrame()
{
    vk::raii::Device& device = vulkanContext.getDevice();

    vk::Fence inFlightFence = frameManager.getInFlightFence();
    (void)device.waitForFences(inFlightFence, VK_TRUE, UINT64_MAX);

    vk::Fence imageAvailableFence = frameManager.getImageAvailableFence();
    device.resetFences(imageAvailableFence);
    auto acquireResult = swapChain.acquireNextImage(UINT64_MAX, VK_NULL_HANDLE, imageAvailableFence);
    vk::Result result = acquireResult.result;
    uint32_t imageIndex = acquireResult.value;

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        swapChain.recreate(vulkanContext, window);
        graphicsPipeline.recreate(vulkanContext, swapChain, *resourceManager.getResourceCreator(),
                                  *vertShaderHandle.Get(), *fragShaderHandle.Get());
        rendergraph->Recompile(swapChain.getExtent());
        frameManager.recreate(vulkanContext, swapChain, graphicsPipeline, *rendergraph,
                              *resourceManager.getResourceCreator(), *textureHandle.Get(), rayTracingContext);
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    (void)device.waitForFences(imageAvailableFence, VK_TRUE, UINT64_MAX);
    device.resetFences(imageAvailableFence);
    device.resetFences(inFlightFence);

    vk::raii::CommandBuffer& commandBuffer = frameManager.getCommandBuffers()[frameManager.getCurrentFrame()];
    commandBuffer.reset();
    const glm::mat4 modelMatrix = computeSceneModelMatrix();
    recordCommandBuffer(commandBuffer, imageIndex, modelMatrix);

    if (camera) {
        frameManager.updateUniformBuffer(frameManager.getCurrentFrame(), frameManager.getSwapChainExtent(),
                                         *camera, modelMatrix);
    }

    vk::Semaphore signalSemaphores[] = {frameManager.getRenderFinishedSemaphore(imageIndex)};

    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.commandBufferCount = 1;
    vk::CommandBuffer cb = commandBuffer;
    submitInfo.pCommandBuffers = &cb;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vulkanContext.getGraphicsQueue().submit(submitInfo, inFlightFence);

    vk::SwapchainKHR swapChains[] = {swapChain.getSwapChain()};
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    vk::Result presentResult = vulkanContext.getPresentQueue().presentKHR(presentInfo);

    if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || frameManager.getFramebufferResized()) {
        frameManager.clearFramebufferResized();
        swapChain.recreate(vulkanContext, window);
        graphicsPipeline.recreate(vulkanContext, swapChain, *resourceManager.getResourceCreator(),
                                  *vertShaderHandle.Get(), *fragShaderHandle.Get());
        rendergraph->Recompile(swapChain.getExtent());
        frameManager.recreate(vulkanContext, swapChain, graphicsPipeline, *rendergraph,
                              *resourceManager.getResourceCreator(), *textureHandle.Get(), rayTracingContext);
    } else if (presentResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    frameManager.advanceFrame();
}

void Renderer::waitIdle()
{
    if (vulkanContext.hasDevice()) {
        vulkanContext.getDevice().waitIdle();
    }
}

void Renderer::recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex, const glm::mat4& modelMatrix)
{
    vk::CommandBufferBeginInfo beginInfo{};
    commandBuffer.begin(beginInfo);

    rayTracingContext.updateTopLevelAS(commandBuffer, modelMatrix);

    std::unordered_map<std::string, ExternalResourceView> externalViews;
    externalViews.emplace("swapchain", ExternalResourceView{
                                          swapChain.getImages()[imageIndex],
                                          swapChain.getImageView(imageIndex),
                                      });
    rendergraph->Execute(commandBuffer, imageIndex, externalViews);

    commandBuffer.end();
}

