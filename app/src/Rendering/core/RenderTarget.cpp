#include "Rendering/core/RenderTarget.h"

void RenderTarget::create(VulkanResourceCreator& creator, const std::string& targetName, vk::Format fmt,
                         vk::Extent2D ext, vk::ImageUsageFlags use, vk::ImageLayout initLayout,
                         vk::ImageLayout finLayout, vk::ImageAspectFlags aspect, vk::SampleCountFlagBits samp)
{
    name = targetName;
    format = fmt;
    extent = ext;
    usage = use;
    initialLayout = initLayout;
    finalLayout = finLayout;
    aspectFlags = aspect;
    samples = samp;

    ImageAllocation alloc = creator.createImage(
        ext.width, ext.height, 1, samp, fmt, vk::ImageTiling::eOptimal, use,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    image = std::move(alloc.image);
    memory = std::move(alloc.memory);
    view = creator.createImageView(static_cast<vk::Image>(*image), format, aspectFlags, 1);
}

void RenderTarget::destroy()
{
    view.reset();
    image.reset();
    memory.reset();
}

vk::ImageView RenderTarget::getImageView() const
{
    return view ? static_cast<vk::ImageView>(*view) : vk::ImageView{};
}

vk::Image RenderTarget::getImage() const
{
    return image ? static_cast<vk::Image>(*image) : vk::Image{};
}

