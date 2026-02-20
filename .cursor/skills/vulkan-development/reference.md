# Vulkan 进阶参考

## 内存与性能

- **VK_MEMORY_PROPERTY_DEVICE_LOCAL**：GPU 专用，用于 vertex/index/texture
- **VK_MEMORY_PROPERTY_HOST_VISIBLE | HOST_COHERENT**：CPU 可映射，用于 uniform/staging
- **VK_MEMORY_PROPERTY_HOST_CACHED**：CPU 缓存，适合频繁读回场景
- 大 buffer 迁移：staging buffer 上传后 `vkCmdCopyBuffer`，再 `transitionImageLayout` 用于 texture

## 同步原语

| 类型 | 用途 |
|------|------|
| Fence | CPU 等待 GPU 完成 |
| Semaphore | GPU 内部同步（acquire → submit → present） |
| Barrier | 同一 command buffer 内 layout/access 同步 |

## 常用 Extension

- `VK_KHR_swapchain`：交换链
- `VK_EXT_debug_utils`：调试与 validation
- `VK_KHR_dynamic_rendering`（Vulkan 1.3+）：无 VkRenderPass 的现代渲染

## 前沿方向

- **Vulkan 1.3 / 1.4**：dynamic rendering、synchronization2
- **Mesh Shader**：`VK_EXT_mesh_shader`
- **Ray Tracing**：`VK_KHR_ray_tracing_pipeline`
