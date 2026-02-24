#include "Rendering/renderer/Renderer.h"
#include "Rendering/pass/DepthPrepass.h"
#include "Rendering/pass/ForwardPass.h"
#include "Rendering/pass/BloomExtractPass.h"
#include "Rendering/pass/BloomBlurPass.h"
#include "Rendering/pass/RtaoComputePass.h"
#include "Rendering/pass/SkyboxPass.h"
#include "Rendering/pass/TonemapBloomPass.h"
#include "Configs/AppConfig.h"
#include "Resource/model/Node.h"
#include "Resource/texture/HdrTextureLoader.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

Renderer::~Renderer()
{
    // Best-effort cleanup to avoid leaving RAII objects alive after VkDevice destruction
    // when callers exit early (exceptions, debugger stop, etc.).
    try {
        if (vulkanContext.hasDevice()) {
            cleanup();
        }
    } catch (...) {
        // Destructors must not throw.
    }
}

glm::mat4 Renderer::computeSceneModelMatrix() const
{
    return glm::scale(glm::mat4(1.0f), glm::vec3(AppConfig::SCENE_MODEL_SCALE));
}

void Renderer::init(GLFWwindow* inWindow)
{
    window = inWindow;

    vulkanContext.init(window);
    swapChain.init(vulkanContext, window);
    resourceManager.init(vulkanContext);

    modelHandle = resourceManager.Load<Model>("bistro/bistro");
    vertShaderHandle = resourceManager.Load<Shader>("pbr_vert");
    fragShaderHandle = resourceManager.Load<Shader>("pbr_frag");
    depthPrepassVertShaderHandle = resourceManager.Load<Shader>("depth_prepass_vert");
    depthOnlyFragShaderHandle = resourceManager.Load<Shader>("depth_only_frag");
    rtaoTraceCompShaderHandle = resourceManager.Load<Shader>("rtao_trace_half_comp");
    rtaoAtrousCompShaderHandle = resourceManager.Load<Shader>("rtao_atrous_comp");
    rtaoUpsampleCompShaderHandle = resourceManager.Load<Shader>("rtao_upsample_comp");
    skyboxVertShaderHandle = resourceManager.Load<Shader>("skybox_vert");
    skyboxFragShaderHandle = resourceManager.Load<Shader>("skybox_frag");
    fullscreenVertShaderHandle = resourceManager.Load<Shader>("fullscreen_vert");
    bloomExtractFragShaderHandle = resourceManager.Load<Shader>("bloom_extract_frag");
    bloomBlurFragShaderHandle = resourceManager.Load<Shader>("bloom_blur_frag");
    tonemapBloomFragShaderHandle = resourceManager.Load<Shader>("tonemap_bloom_frag");

    if (!modelHandle.IsValid() || !vertShaderHandle.IsValid() || !fragShaderHandle.IsValid() ||
        !depthPrepassVertShaderHandle.IsValid() ||
        !depthOnlyFragShaderHandle.IsValid() ||
        !rtaoTraceCompShaderHandle.IsValid() || !rtaoAtrousCompShaderHandle.IsValid() || !rtaoUpsampleCompShaderHandle.IsValid() ||
        !skyboxVertShaderHandle.IsValid() || !skyboxFragShaderHandle.IsValid() ||
        !fullscreenVertShaderHandle.IsValid() || !bloomExtractFragShaderHandle.IsValid() ||
        !bloomBlurFragShaderHandle.IsValid() || !tonemapBloomFragShaderHandle.IsValid()) {
        throw std::runtime_error("failed to load model or shader resource!");
    }
    animationPlayer.setModel(modelHandle.Get());
    if (modelHandle->getMeshes().empty()) {
        throw std::runtime_error("loaded model has no meshes!");
    }

    VulkanResourceCreator* resourceCreator = resourceManager.getResourceCreator();
    const auto& cpuMeshes = modelHandle->getMeshes();
    const auto& materials = modelHandle->getMaterials();
    modelMeshes.clear();
    modelMeshes.resize(cpuMeshes.size());
    std::vector<uint8_t> meshOpaqueFlags(cpuMeshes.size(), 1u);
    for (size_t i = 0; i < cpuMeshes.size(); ++i) {
        modelMeshes[i].upload(*resourceCreator, cpuMeshes[i].vertices, cpuMeshes[i].indices);
        const int matIdx = cpuMeshes[i].materialIndex;
        const bool hasMat = (matIdx >= 0) && (matIdx < static_cast<int>(materials.size()));
        const bool opaque = !hasMat || (materials[static_cast<size_t>(matIdx)].alphaMode == AlphaMode::Opaque);
        meshOpaqueFlags[i] = opaque ? 1u : 0u;
    }

    globalMeshBuffer.init(*resourceCreator, modelMeshes);

    auto countMaxDraws = [](auto&& self, const std::vector<Node*>& nodes) -> uint32_t {
        uint32_t count = 0;
        for (Node* node : nodes) {
            if (!node) continue;
            count += static_cast<uint32_t>(node->meshIndices.size());
            if (!node->children.empty()) {
                count += self(self, node->children);
            }
        }
        return count;
    };
    maxDraws = countMaxDraws(countMaxDraws, modelHandle->getRootNodes());
    maxDraws = std::max(1u, maxDraws);

    const vk::Format hdrColorFormat = vk::Format::eR16G16B16A16Sfloat;
    graphicsPipeline.init(vulkanContext, swapChain, *resourceCreator, *vertShaderHandle.Get(), *fragShaderHandle.Get(), hdrColorFormat);
    depthPrepassPipeline.init(vulkanContext, swapChain, *resourceCreator, graphicsPipeline,
                              *depthPrepassVertShaderHandle.Get(), *depthOnlyFragShaderHandle.Get());
    rtaoComputePipeline.init(vulkanContext, *rtaoTraceCompShaderHandle.Get(), *rtaoAtrousCompShaderHandle.Get(), *rtaoUpsampleCompShaderHandle.Get());

    // Load HDR equirect and convert to cubemap for skybox
    std::string hdrPath = AppConfig::ENV_HDR_PATH;
    // NOTE: equirectangular map must wrap in U (longitude), otherwise seams will appear in the converted cubemap.
    auto equirectResult = HdrTextureLoader::loadFromFile(hdrPath, resourceCreator,
        vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eClampToEdge);
    if (equirectResult && equirectResult->imageView && equirectResult->sampler) {
        envCubemapResult = EquirectToCubemap::convert(*resourceCreator, *equirectResult->imageView, *equirectResult->sampler, 512);
    }

    vk::Format swapchainColorFormat = swapChain.getImageFormat();
    vk::Format depthFormat = resourceCreator->findDepthFormat();
    skyboxPipeline.init(vulkanContext.getDevice(), *resourceCreator, hdrColorFormat, depthFormat,
                        vulkanContext.getMsaaSamples(), *skyboxVertShaderHandle.Get(), *skyboxFragShaderHandle.Get());
    postProcessPipeline.init(vulkanContext, *resourceCreator, hdrColorFormat, swapchainColorFormat, *fullscreenVertShaderHandle.Get(),
                             *bloomExtractFragShaderHandle.Get(), *bloomBlurFragShaderHandle.Get(), *tonemapBloomFragShaderHandle.Get());

    const glm::mat4 sceneModelMatrix = computeSceneModelMatrix();
    rebuildRayTracingInstances(sceneModelMatrix);
    rayTracingContext.init(vulkanContext, *resourceCreator, modelMeshes, meshOpaqueFlags, rayTracingInstances);
    cachedModelMatrixForTlas = sceneModelMatrix;
    tlasNeedsUpdate = false;  // TLAS just built in init

    rendergraph.emplace(vulkanContext.getDevice(), *resourceCreator);
    vk::Extent2D extent = swapChain.getExtent();

    rendergraph->AddResource("color_msaa", hdrColorFormat, extent,
                             vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                             vk::ImageAspectFlagBits::eColor, vulkanContext.getMsaaSamples());
    rendergraph->AddResource("scene_color", hdrColorFormat, extent,
                             vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal,
                             vk::ImageAspectFlagBits::eColor, vk::SampleCountFlagBits::e1);
    rendergraph->AddResource("bloom_a", hdrColorFormat, extent,
                             vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal,
                             vk::ImageAspectFlagBits::eColor, vk::SampleCountFlagBits::e1, 2);
    rendergraph->AddResource("bloom_b", hdrColorFormat, extent,
                             vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal,
                             vk::ImageAspectFlagBits::eColor, vk::SampleCountFlagBits::e1, 2);
    rendergraph->AddResource("depth", depthFormat, extent,
                             vk::ImageUsageFlagBits::eDepthStencilAttachment,
                             vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                             vk::ImageAspectFlagBits::eDepth, vulkanContext.getMsaaSamples());
    rendergraph->AddExternalResource("swapchain", swapchainColorFormat, extent,
                                     vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

    bool hasEnvCubemap = envCubemapResult.cubeView && envCubemapResult.sampler;
    if (hasEnvCubemap) {
        rendergraph->AddPass(std::make_unique<SkyboxPass>(skyboxPipeline, frameManager, *rendergraph, swapChain));
    }
    const bool useSkyboxIblDebug = (AppConfig::SKYBOX_IBL_DEBUG_MODE > 0);
    const bool enableDepthResolve = (vulkanContext.getMsaaSamples() != vk::SampleCountFlagBits::e1);
    rendergraph->AddPass(std::make_unique<DepthPrepass>(
        depthPrepassPipeline, frameManager, *modelHandle.Get(), modelMeshes, globalMeshBuffer, maxDraws,
        *rendergraph, enableDepthResolve));
    rendergraph->AddPass(std::make_unique<RtaoComputePass>(vulkanContext.getDevice(), rtaoComputePipeline, frameManager, rayTracingContext));
    rendergraph->AddPass(std::make_unique<ForwardPass>(graphicsPipeline, frameManager, *modelHandle.Get(), modelMeshes,
                                                        globalMeshBuffer, maxDraws, *rendergraph, false, !hasEnvCubemap));
    if (AppConfig::ENABLE_BLOOM) {
        rendergraph->AddPass(std::make_unique<BloomExtractPass>(postProcessPipeline, frameManager, *rendergraph));
        rendergraph->AddPass(std::make_unique<BloomBlurPass>("BloomBlurPassH", "bloom_a", "bloom_b", true,
                                                             postProcessPipeline, frameManager, *rendergraph));
        rendergraph->AddPass(std::make_unique<BloomBlurPass>("BloomBlurPassV", "bloom_b", "bloom_a", false,
                                                             postProcessPipeline, frameManager, *rendergraph));
    }
    rendergraph->AddPass(std::make_unique<TonemapBloomPass>(postProcessPipeline, frameManager, *rendergraph, swapChain));
    rendergraph->Compile();

    frameManager.init(vulkanContext, swapChain, graphicsPipeline, *rendergraph, *resourceCreator,
                      *modelHandle.Get(), rayTracingContext, maxDraws);
    frameManager.createPostProcessResources(vulkanContext.getDevice(), postProcessPipeline.getDescriptorSetLayout());

    if (AppConfig::ENABLE_IMGUI) {
        imguiIntegration.init(vulkanContext, *resourceManager.getResourceCreator(), swapChain, window);
    }

    if (hasEnvCubemap) {
        // 若 SKYBOX_IBL_DEBUG_MODE > 0，需先计算 IBL，再用 irradiance/prefilter 作为天空盒纹理
        if (useSkyboxIblDebug) {
            iblResult = IblPrecompute::compute(*resourceCreator, *envCubemapResult.cubeView, *envCubemapResult.sampler);
        }
        vk::ImageView skyboxView = *envCubemapResult.cubeView;
        vk::Sampler skyboxSampler = *envCubemapResult.sampler;
        if (useSkyboxIblDebug && iblResult.sampler) {
            if (AppConfig::SKYBOX_IBL_DEBUG_MODE == 1 && iblResult.irradianceView) {
                skyboxView = *iblResult.irradianceView;
                skyboxSampler = *iblResult.sampler;
            } else if (AppConfig::SKYBOX_IBL_DEBUG_MODE == 2 && iblResult.prefilterView) {
                skyboxView = *iblResult.prefilterView;
                skyboxSampler = *iblResult.sampler;
            }
        }
        frameManager.createSkyboxResources(*resourceCreator, skyboxPipeline.getDescriptorSetLayout(),
                                           skyboxView, skyboxSampler);
        if (!useSkyboxIblDebug) {
            iblResult = IblPrecompute::compute(*resourceCreator, *envCubemapResult.cubeView, *envCubemapResult.sampler);
        }
        if (iblResult.irradianceView && iblResult.prefilterView && iblResult.brdfLutView && iblResult.sampler) {
            frameManager.setIblResources(vulkanContext.getDevice(), *iblResult.irradianceView, *iblResult.prefilterView,
                                         *iblResult.brdfLutView, *iblResult.sampler);
        }
    }
}

void Renderer::update(float deltaTime)
{
    if (animationPlayer.update(deltaTime)) {
        tlasNeedsUpdate = true;  // Animation modified node transforms
    }
}

void Renderer::cleanup()
{
    if (vulkanContext.hasDevice()) {
        waitIdle();
    }

    if (AppConfig::ENABLE_IMGUI) {
        imguiIntegration.cleanup();
    }
    frameManager.cleanup(vulkanContext.getDevice());
    globalMeshBuffer.cleanup();
    if (rendergraph) {
        rendergraph->Cleanup();
    }
    rayTracingContext.cleanup();
    rtaoComputePipeline.cleanup();
    depthPrepassPipeline.cleanup();
    skyboxPipeline.cleanup();
    postProcessPipeline.cleanup();
    graphicsPipeline.cleanup();
    modelMeshes.clear();
    resourceManager.cleanup();
    swapChain.cleanup();

    // 必须在销毁 Device 之前释放 env/IBL 资源（持有 ImageView/Sampler 等）
    envCubemapResult = {};
    iblResult = {};

    vulkanContext.cleanup();
}

void Renderer::drawFrame()
{
    vk::raii::Device& device = vulkanContext.getDevice();

    auto now = [] { return std::chrono::high_resolution_clock::now(); };
    auto toMs = [](auto dt) -> double { return std::chrono::duration<double, std::milli>(dt).count(); };
    const auto tFrame0 = now();

    frameCounter++;

    vk::Fence inFlightFence = frameManager.getInFlightFence();
    (void)device.waitForFences(inFlightFence, VK_TRUE, UINT64_MAX);

    vk::Fence imageAvailableFence = frameManager.getImageAvailableFence();
    device.resetFences(imageAvailableFence);
    vk::Result result = vk::Result::eSuccess;
    uint32_t imageIndex = 0;
    const auto tAcquire0 = now();
    try {
        auto acquireResult = swapChain.acquireNextImage(UINT64_MAX, VK_NULL_HANDLE, imageAvailableFence);
        result = acquireResult.result;
        imageIndex = acquireResult.value;
    } catch (const vk::OutOfDateKHRError&) {
        // vulkan-hpp may throw for OutOfDate instead of returning a VkResult.
        result = vk::Result::eErrorOutOfDateKHR;
    }

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        swapchainRecreateCount++;
        swapChain.recreate(vulkanContext, window);
        const vk::Format hdrColorFormat = vk::Format::eR16G16B16A16Sfloat;
        graphicsPipeline.recreate(vulkanContext, swapChain, *resourceManager.getResourceCreator(),
                                  *vertShaderHandle.Get(), *fragShaderHandle.Get(), hdrColorFormat);
        depthPrepassPipeline.recreate(vulkanContext, swapChain, *resourceManager.getResourceCreator(),
                                      graphicsPipeline, *depthPrepassVertShaderHandle.Get(), *depthOnlyFragShaderHandle.Get());
        postProcessPipeline.recreate(vulkanContext, *resourceManager.getResourceCreator(), hdrColorFormat, swapChain.getImageFormat(),
                                     *fullscreenVertShaderHandle.Get(), *bloomExtractFragShaderHandle.Get(),
                                     *bloomBlurFragShaderHandle.Get(), *tonemapBloomFragShaderHandle.Get());
        rtaoComputePipeline.recreate(vulkanContext, *rtaoTraceCompShaderHandle.Get(), *rtaoAtrousCompShaderHandle.Get(), *rtaoUpsampleCompShaderHandle.Get());
        rendergraph->Recompile(swapChain.getExtent());
        frameManager.recreate(vulkanContext, swapChain, graphicsPipeline, *rendergraph,
                              *resourceManager.getResourceCreator(), *modelHandle.Get(), rayTracingContext, maxDraws);
        frameManager.createPostProcessResources(vulkanContext.getDevice(), postProcessPipeline.getDescriptorSetLayout());
        if (iblResult.irradianceView && iblResult.prefilterView && iblResult.brdfLutView && iblResult.sampler) {
            frameManager.setIblResources(vulkanContext.getDevice(), *iblResult.irradianceView, *iblResult.prefilterView,
                                         *iblResult.brdfLutView, *iblResult.sampler);
        }
        if (AppConfig::ENABLE_IMGUI) {
            imguiIntegration.onSwapchainRecreated(swapChain, window);
        }
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    (void)device.waitForFences(imageAvailableFence, VK_TRUE, UINT64_MAX);
    lastCpuTimings.acquireMs = toMs(now() - tAcquire0);
    device.resetFences(imageAvailableFence);
    device.resetFences(inFlightFence);

    if (AppConfig::ENABLE_IMGUI) {
        ImGuiIntegration::UiStats stats{};
        stats.acquireMs = lastCpuTimings.acquireMs;
        stats.recordMs = lastCpuTimings.recordMs;
        stats.updateUboMs = lastCpuTimings.updateUboMs;
        stats.submitMs = lastCpuTimings.submitMs;
        stats.presentMs = lastCpuTimings.presentMs;
        stats.totalMs = lastCpuTimings.totalMs;
        stats.swapchainRecreateCount = swapchainRecreateCount;
        stats.frameCounter = frameCounter;
        imguiIntegration.setUiStats(stats);
        imguiIntegration.newFrame();
    }

    vk::raii::CommandBuffer& commandBuffer = frameManager.getCommandBuffers()[frameManager.getCurrentFrame()];
    commandBuffer.reset();
    const glm::mat4 modelMatrix = computeSceneModelMatrix();
    const auto tRecord0 = now();
    recordCommandBuffer(commandBuffer, imageIndex, modelMatrix);
    lastCpuTimings.recordMs = toMs(now() - tRecord0);

    if (camera) {
        const auto tUbo0 = now();
        frameManager.updateUniformBuffer(frameManager.getCurrentFrame(), frameManager.getSwapChainExtent(),
                                         *camera, modelMatrix);
        lastCpuTimings.updateUboMs = toMs(now() - tUbo0);
    } else {
        lastCpuTimings.updateUboMs = 0.0;
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

    const auto tSubmit0 = now();
    vulkanContext.getGraphicsQueue().submit(submitInfo, inFlightFence);
    lastCpuTimings.submitMs = toMs(now() - tSubmit0);

    vk::SwapchainKHR swapChains[] = {swapChain.getSwapChain()};
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    vk::Result presentResult = vk::Result::eSuccess;
    const auto tPresent0 = now();
    try {
        presentResult = vulkanContext.getPresentQueue().presentKHR(presentInfo);
    } catch (const vk::OutOfDateKHRError&) {
        // Avoid propagating the exception: treat as a normal OutOfDate result and recreate the swapchain.
        presentResult = vk::Result::eErrorOutOfDateKHR;
    }
    lastCpuTimings.presentMs = toMs(now() - tPresent0);

    if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || frameManager.getFramebufferResized()) {
        frameManager.clearFramebufferResized();
        swapchainRecreateCount++;
        swapChain.recreate(vulkanContext, window);
        const vk::Format hdrColorFormat = vk::Format::eR16G16B16A16Sfloat;
        graphicsPipeline.recreate(vulkanContext, swapChain, *resourceManager.getResourceCreator(),
                                  *vertShaderHandle.Get(), *fragShaderHandle.Get(), hdrColorFormat);
        depthPrepassPipeline.recreate(vulkanContext, swapChain, *resourceManager.getResourceCreator(),
                                      graphicsPipeline, *depthPrepassVertShaderHandle.Get(), *depthOnlyFragShaderHandle.Get());
        postProcessPipeline.recreate(vulkanContext, *resourceManager.getResourceCreator(), hdrColorFormat, swapChain.getImageFormat(),
                                     *fullscreenVertShaderHandle.Get(), *bloomExtractFragShaderHandle.Get(),
                                     *bloomBlurFragShaderHandle.Get(), *tonemapBloomFragShaderHandle.Get());
        rtaoComputePipeline.recreate(vulkanContext, *rtaoTraceCompShaderHandle.Get(), *rtaoAtrousCompShaderHandle.Get(), *rtaoUpsampleCompShaderHandle.Get());
        rendergraph->Recompile(swapChain.getExtent());
        frameManager.recreate(vulkanContext, swapChain, graphicsPipeline, *rendergraph,
                              *resourceManager.getResourceCreator(), *modelHandle.Get(), rayTracingContext, maxDraws);
        frameManager.createPostProcessResources(vulkanContext.getDevice(), postProcessPipeline.getDescriptorSetLayout());
        if (iblResult.irradianceView && iblResult.prefilterView && iblResult.brdfLutView && iblResult.sampler) {
            frameManager.setIblResources(vulkanContext.getDevice(), *iblResult.irradianceView, *iblResult.prefilterView,
                                         *iblResult.brdfLutView, *iblResult.sampler);
        }
        if (AppConfig::ENABLE_IMGUI) {
            imguiIntegration.onSwapchainRecreated(swapChain, window);
        }
    } else if (presentResult != vk::Result::eSuccess) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    lastCpuTimings.totalMs = toMs(now() - tFrame0);
    accumCpuTimings.acquireMs += lastCpuTimings.acquireMs;
    accumCpuTimings.recordMs += lastCpuTimings.recordMs;
    accumCpuTimings.updateUboMs += lastCpuTimings.updateUboMs;
    accumCpuTimings.submitMs += lastCpuTimings.submitMs;
    accumCpuTimings.presentMs += lastCpuTimings.presentMs;
    accumCpuTimings.totalMs += lastCpuTimings.totalMs;
    accumFrames++;

    if (frameCounter % AppConfig::PERF_PRINT_INTERVAL == 0u && accumFrames > 0) {
        if (AppConfig::ENABLE_PERF_DEBUG) {
            const double inv = 1.0 / static_cast<double>(accumFrames);
            std::ostream& out = std::cout;
            out << "[Perf]";
            if (AppConfig::PERF_PRINT_FRAME_STAGES) {
                out << " avg_ms acquire=" << (accumCpuTimings.acquireMs * inv)
                    << " record=" << (accumCpuTimings.recordMs * inv)
                    << " ubo=" << (accumCpuTimings.updateUboMs * inv)
                    << " submit=" << (accumCpuTimings.submitMs * inv)
                    << " present=" << (accumCpuTimings.presentMs * inv)
                    << " total=" << (accumCpuTimings.totalMs * inv);
            }
            if (AppConfig::PERF_PRINT_RTAO) {
                out << " | RTAO_ms=" << lastRenderStats.rtaoMs
                    << " depthPrepass_ms=" << lastRenderStats.depthPrepassMs;
            }
            if (AppConfig::PERF_PRINT_BLOOM) {
                out << " | bloom_extract=" << lastRenderStats.bloomExtractMs
                    << " blurH=" << lastRenderStats.bloomBlurHMs
                    << " blurV=" << lastRenderStats.bloomBlurVMs
                    << " tonemap=" << lastRenderStats.tonemapMs;
            }
            if (AppConfig::PERF_PRINT_FORWARD_DETAIL) {
                out << " | draws(depth/fwd)=" << lastRenderStats.depthDrawCalls << "/" << lastRenderStats.forwardDrawCalls
                    << " items(opaque/trans)=" << lastRenderStats.opaqueItems << "/" << lastRenderStats.transparentItems
                    << " fwd_ms(collect/sort/issue)=" << lastRenderStats.forwardCollectMs << "/"
                    << lastRenderStats.forwardSortMs << "/" << lastRenderStats.forwardIssueMs
                    << " fwd_binds(pipe/dset/vb/ib)=" << lastRenderStats.forwardPipelineBinds << "/"
                    << lastRenderStats.forwardDescriptorBinds << "/" << lastRenderStats.forwardVertexBufferBinds << "/"
                    << lastRenderStats.forwardIndexBufferBinds;
            }
            out << " swapchainRecreate=" << swapchainRecreateCount << std::endl;
        }
        accumCpuTimings = CpuTimings{};
        accumFrames = 0;
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

    // TLAS static caching: only rebuild when model transforms change (animation, rotation, scale, etc.)
    if (!cachedModelMatrixForTlas || *cachedModelMatrixForTlas != modelMatrix) {
        tlasNeedsUpdate = true;
        cachedModelMatrixForTlas = modelMatrix;
    }
    if (tlasNeedsUpdate) {
        rebuildRayTracingInstances(modelMatrix);
        rayTracingContext.updateTopLevelAS(commandBuffer, rayTracingInstances);
        tlasNeedsUpdate = false;
    }

    std::unordered_map<std::string, ExternalResourceView> externalViews;
    externalViews.emplace("swapchain", ExternalResourceView{
                                          swapChain.getImages()[imageIndex],
                                          swapChain.getImageView(imageIndex),
                                      });
    if (modelHandle.IsValid()) {
        frameManager.prepareSharedOpaqueIndirect(*modelHandle.Get(), globalMeshBuffer, modelMatrix);
    }
    lastRenderStats = RenderStats{};
    rendergraph->Execute(commandBuffer, imageIndex, modelMatrix, externalViews, camera, &lastRenderStats);

    if (AppConfig::ENABLE_IMGUI) {
        imguiIntegration.render(commandBuffer,
                            swapChain.getImages()[imageIndex],
                            swapChain.getImageView(imageIndex),
                            swapChain.getExtent(),
                            swapChain.getImageFormat());
    }

    commandBuffer.end();
}

void Renderer::rebuildRayTracingInstances(const glm::mat4& modelMatrix)
{
    rayTracingInstances.clear();
    if (!modelHandle.IsValid()) {
        return;
    }

    auto visit = [&](auto&& self, const std::vector<Node*>& nodes, const glm::mat4& parentWorld) -> void {
        for (Node* node : nodes) {
            if (!node) continue;
            const glm::mat4 worldFromNode = parentWorld * node->getLocalMatrix();
            for (uint32_t meshIndex : node->meshIndices) {
                RayTracingInstanceDesc inst{};
                inst.meshIndex = meshIndex;
                inst.transform = worldFromNode;
                rayTracingInstances.push_back(inst);
            }
            if (!node->children.empty()) {
                self(self, node->children, worldFromNode);
            }
        }
    };

    visit(visit, modelHandle->getRootNodes(), modelMatrix);
}

