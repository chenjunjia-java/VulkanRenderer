#include "Rendering/core/Rendergraph.h"
#include "Configs/AppConfig.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
struct BarrierParams {
    vk::PipelineStageFlags srcStage{};
    vk::PipelineStageFlags dstStage{};
    vk::AccessFlags srcAccess{};
    vk::AccessFlags dstAccess{};
};

BarrierParams inferBarrierParams(vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    // Minimal set for this project (dynamic rendering attachments + present).
    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        return {vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentWrite};
    }
    if (oldLayout == vk::ImageLayout::ePresentSrcKHR && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        return {vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, vk::AccessFlagBits::eColorAttachmentWrite};
    }
    if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::ePresentSrcKHR) {
        return {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe, vk::AccessFlagBits::eColorAttachmentWrite, {}};
    }
    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        return {vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eEarlyFragmentTests, {}, vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite};
    }
    if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        return {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
                vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead};
    }
    if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        return {vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eColorAttachmentWrite};
    }
    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        return {vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eFragmentShader, {},
                vk::AccessFlagBits::eShaderRead};
    }

    // Fallback: conservative synchronization (kept minimal for now).
    return {vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {}};
}

void transitionImageLayout(
    vk::raii::CommandBuffer& cb,
    vk::Image image,
    vk::ImageAspectFlags aspectFlags,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout)
{
    if (!image || oldLayout == newLayout) return;

    BarrierParams params = inferBarrierParams(oldLayout, newLayout);

    vk::ImageMemoryBarrier barrier{};
    barrier.setOldLayout(oldLayout)
        .setNewLayout(newLayout)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setSubresourceRange({aspectFlags, 0, 1, 0, 1})
        .setSrcAccessMask(params.srcAccess)
        .setDstAccessMask(params.dstAccess);

    cb.pipelineBarrier(params.srcStage, params.dstStage, vk::DependencyFlagBits::eByRegion, {}, {}, barrier);
}
}  // namespace

Rendergraph::Rendergraph(vk::raii::Device& dev, VulkanResourceCreator& creator)
    : device(dev)
    , resourceCreator(creator)
{
}

namespace {
vk::Extent2D applyExtentDivisor(vk::Extent2D baseExtent, uint32_t extentDivisor)
{
    const uint32_t d = std::max(1u, extentDivisor);
    return vk::Extent2D{
        std::max(1u, baseExtent.width / d),
        std::max(1u, baseExtent.height / d)};
}
}  // namespace

void Rendergraph::AddResource(const std::string& name, vk::Format format, vk::Extent2D ext,
                              vk::ImageUsageFlags usage, vk::ImageLayout initialLayout,
                              vk::ImageLayout finalLayout, vk::ImageAspectFlags aspectFlags,
                              vk::SampleCountFlagBits samples, uint32_t extentDivisor)
{
    if (compiled) {
        throw std::runtime_error("Rendergraph: cannot AddResource after Compile");
    }
    extent = ext;
    const vk::Extent2D resourceExtent = applyExtentDivisor(ext, extentDivisor);

    ImageResource resource;
    resource.name = name;
    resource.format = format;
    resource.extent = resourceExtent;
    resource.usage = usage;
    resource.initialLayout = initialLayout;
    resource.finalLayout = finalLayout;
    resource.aspectFlags = aspectFlags;
    resource.samples = samples;
    resource.extentDivisor = std::max(1u, extentDivisor);
    resource.isExternal = false;
    resource.currentLayout = initialLayout;

    resources[name] = std::move(resource);
}

void Rendergraph::AddExternalResource(const std::string& name, vk::Format format, vk::Extent2D ext,
                                      vk::ImageLayout initialLayout, vk::ImageLayout finalLayout)
{
    if (compiled) {
        throw std::runtime_error("Rendergraph: cannot AddExternalResource after Compile");
    }
    extent = ext;

    ImageResource resource;
    resource.name = name;
    resource.format = format;
    resource.extent = ext;
    resource.usage = vk::ImageUsageFlags{};
    resource.initialLayout = initialLayout;
    resource.finalLayout = finalLayout;
    resource.aspectFlags = vk::ImageAspectFlagBits::eColor;
    resource.samples = vk::SampleCountFlagBits::e1;
    resource.extentDivisor = 1;
    resource.isExternal = true;
    resource.currentLayout = initialLayout;

    resources[name] = std::move(resource);
}

void Rendergraph::AddPass(std::unique_ptr<RenderPass> pass)
{
    if (compiled) {
        throw std::runtime_error("Rendergraph: cannot AddPass after Compile");
    }
    if (pass) {
        passes.push_back(std::move(pass));
    }
}

void Rendergraph::Compile()
{
    if (compiled) {
        Cleanup();
    }

    std::unordered_map<std::string, size_t> resourceWriters;
    std::vector<std::vector<size_t>> dependencies(passes.size());
    std::vector<std::vector<size_t>> dependents(passes.size());

    for (size_t i = 0; i < passes.size(); ++i) {
        const RenderPass& pass = *passes[i];

        for (const auto& input : pass.getInputs()) {
            auto it = resourceWriters.find(input);
            if (it != resourceWriters.end()) {
                dependencies[i].push_back(it->second);
                dependents[it->second].push_back(i);
            }
        }

        for (const auto& output : pass.getOutputs()) {
            resourceWriters[output] = i;
        }
    }

    std::vector<bool> visited(passes.size(), false);
    std::vector<bool> inStack(passes.size(), false);
    executionOrder.clear();

    std::function<void(size_t)> visit = [&](size_t node) {
        if (inStack[node]) {
            throw std::runtime_error("Rendergraph: cycle detected in pass dependencies");
        }
        if (visited[node]) return;

        inStack[node] = true;
        // Ensure dependencies execute before this pass.
        for (auto dep : dependencies[node]) {
            visit(dep);
        }
        inStack[node] = false;
        visited[node] = true;
        executionOrder.push_back(node);
    };

    for (size_t i = 0; i < passes.size(); ++i) {
        if (!visited[i]) {
            visit(i);
        }
    }

    allocateInternalResources();
    compiled = true;
}

void Rendergraph::Recompile(vk::Extent2D newExtent)
{
    extent = newExtent;
    for (auto& [name, resource] : resources) {
        resource.extent = applyExtentDivisor(newExtent, resource.extentDivisor);
    }
    Cleanup();
    compiled = false;
    Compile();
}

void Rendergraph::Cleanup()
{
    for (auto& [name, resource] : resources) {
        if (!resource.isExternal) {
            resource.view.reset();
            resource.image.reset();
            resource.memory.reset();
        }
    }
    executionOrder.clear();
    externalImageLayouts.clear();
    compiled = false;
}

void Rendergraph::allocateInternalResources()
{
    for (auto& [name, resource] : resources) {
        if (resource.isExternal) continue;

        ImageAllocation alloc = resourceCreator.createImage(
            resource.extent.width, resource.extent.height, 1, resource.samples,
            resource.format, vk::ImageTiling::eOptimal, resource.usage,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        resource.image = std::move(alloc.image);
        resource.memory = std::move(alloc.memory);
        resource.view = resourceCreator.createImageView(
            static_cast<vk::Image>(*resource.image), resource.format,
            resource.aspectFlags, 1);

        // Newly created images start in undefined.
        resource.currentLayout = vk::ImageLayout::eUndefined;
    }
}

void Rendergraph::Execute(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex,
                          const glm::mat4& modelMatrix,
                          const std::unordered_map<std::string, ExternalResourceView>& externalViews,
                          const Camera* camera,
                          RenderStats* stats)
{
    if (!compiled) {
        throw std::runtime_error("Rendergraph: must Compile before Execute");
    }

    auto getExternalLayoutKey = [](vk::Image img) -> uint64_t {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkImage>(img)));
    };

    auto ensureLayout = [&](const std::string& resName, vk::Image image, vk::ImageAspectFlags aspect, vk::ImageLayout desiredLayout) {
        if (!image) return;

        auto& perImage = externalImageLayouts[resName];
        const uint64_t key = getExternalLayoutKey(image);
        vk::ImageLayout& tracked = perImage[key];  // default-inits to eUndefined
        transitionImageLayout(commandBuffer, image, aspect, tracked, desiredLayout);
        tracked = desiredLayout;
    };

    auto ensureResourceLayout = [&](const std::string& resName, vk::ImageLayout desiredLayout) {
        auto it = resources.find(resName);
        if (it == resources.end()) return;

        ImageResource& res = it->second;
        if (res.isExternal) {
            auto extIt = externalViews.find(resName);
            if (extIt == externalViews.end()) return;
            ensureLayout(resName, extIt->second.image, res.aspectFlags, desiredLayout);
            return;
        }

        if (!res.image) return;
        vk::Image img = static_cast<vk::Image>(*res.image);
        transitionImageLayout(commandBuffer, img, res.aspectFlags, res.currentLayout, desiredLayout);
        res.currentLayout = desiredLayout;
    };

    PassExecuteContext ctx{commandBuffer, imageIndex, modelMatrix, camera, stats};
    for (auto passIdx : executionOrder) {
        // Pre-pass: transition inputs/outputs to the layouts required for the pass.
        // Current minimal policy:
        // - Internal resources use their declared finalLayout as "working layout".
        // - External swapchain-like outputs (finalLayout == Present) are transitioned to color-attachment for rendering,
        //   then transitioned back to present after the pass.
        const RenderPass& pass = *passes[passIdx];
        for (const auto& input : pass.getInputs()) {
            auto rit = resources.find(input);
            if (rit != resources.end()) {
                const vk::ImageLayout inputLayout = pass.getRequiredInputLayout(input).value_or(rit->second.finalLayout);
                ensureResourceLayout(input, inputLayout);
            }
        }
        for (const auto& output : pass.getOutputs()) {
            auto rit = resources.find(output);
            if (rit == resources.end()) continue;
            const ImageResource& res = rit->second;
            const vk::ImageLayout outputLayout = pass.getRequiredOutputLayout(output).value_or(res.finalLayout);
            if (res.isExternal && res.finalLayout == vk::ImageLayout::ePresentSrcKHR) {
                ensureResourceLayout(output, vk::ImageLayout::eColorAttachmentOptimal);
            } else {
                ensureResourceLayout(output, outputLayout);
            }
        }

        const auto tPass0 = std::chrono::high_resolution_clock::now();
        passes[passIdx]->execute(ctx);
        const auto tPass1 = std::chrono::high_resolution_clock::now();
        if (AppConfig::ENABLE_PERF_DEBUG && stats) {
            const double passMs = std::chrono::duration<double, std::milli>(tPass1 - tPass0).count();
            const std::string& name = pass.getName();
            if (name == "DepthPrepass") stats->depthPrepassMs = passMs;
            else if (name == "RtaoComputePass") stats->rtaoMs = passMs;
            else if (name == "SkyboxPass") stats->skyboxMs = passMs;
            else if (name == "ScenePass") stats->forwardMs = passMs;
            else if (name == "BloomExtractPass") stats->bloomExtractMs = passMs;
            else if (name == "BloomBlurPassH") stats->bloomBlurHMs = passMs;
            else if (name == "BloomBlurPassV") stats->bloomBlurVMs = passMs;
            else if (name == "TonemapBloomPass") stats->tonemapMs = passMs;
            else if (name == "OcclusionPass") stats->occlusionMs = passMs;
        }

        // Post-pass: bring external presentable outputs back to their declared finalLayout.
        for (const auto& output : pass.getOutputs()) {
            auto rit = resources.find(output);
            if (rit == resources.end()) continue;
            const ImageResource& res = rit->second;
            if (res.isExternal && res.finalLayout == vk::ImageLayout::ePresentSrcKHR) {
                ensureResourceLayout(output, vk::ImageLayout::ePresentSrcKHR);
            }
        }
    }
}

vk::ImageView Rendergraph::GetImageView(const std::string& name) const
{
    auto it = resources.find(name);
    if (it == resources.end() || it->second.isExternal || !it->second.view) {
        return vk::ImageView{};
    }
    return static_cast<vk::ImageView>(*it->second.view);
}

vk::Extent2D Rendergraph::GetResourceExtent(const std::string& name) const
{
    auto it = resources.find(name);
    if (it == resources.end()) {
        return extent;
    }
    return it->second.extent;
}

