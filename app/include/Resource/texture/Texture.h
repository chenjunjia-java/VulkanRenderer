#pragma once

#include "Rendering/RHI/Vulkan/VulkanTypes.h"
#include "Rendering/RHI/Vulkan/VulkanResourceCreator.h"
#include "Resource/core/Resource.h"

#include <optional>

class Texture : public Resource {
public:
    explicit Texture(const std::string& id) : Resource(id) {}

    vk::ImageView getImageView() const { return *textureImageView; }
    vk::Sampler getSampler() const { return *textureSampler; }

protected:
    bool doLoad() override;
    void doUnload() override;

private:
    unsigned char* loadImageData(const std::string& filePath, int* width, int* height, int* channels);
    void freeImageData(unsigned char* data);
    void createVulkanImage(unsigned char* data, int width, int height, int channels);

    std::optional<vk::raii::Image> textureImage;
    std::optional<vk::raii::DeviceMemory> textureImageMemory;
    std::optional<vk::raii::ImageView> textureImageView;
    std::optional<vk::raii::Sampler> textureSampler;

    int textureWidth = 0;
    int textureHeight = 0;
    int textureChannels = 0;
};

