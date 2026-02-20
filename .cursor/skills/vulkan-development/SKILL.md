---
name: vulkan-development
description: Guides Vulkan graphics development with best practices for resource management, synchronization, pipeline setup, and debugging. Use when implementing Vulkan features, fixing rendering issues, adding pipelines, managing GPU resources, or debugging validation layer errors.
---

# Vulkan 开发指南

## 快速检查清单

新增或修改 Vulkan 功能时，按此顺序验证：

- [ ] 资源创建与销毁成对，且销毁顺序正确（依赖者先销毁）
- [ ] 所有 `vkCreate*` / `vkAllocate*` 检查返回值
- [ ] 同步正确：fence/semaphore 使用、帧并发控制
- [ ] Shader 的 `location` / `binding` 与 C++ 端一致
- [ ] Uniform/Vertex 结构体 `alignas(16)` 对齐

## 常见任务流程

### 添加新 Pipeline

1. 编写 `.vert` / `.frag`，用 `glslc` 编译为 `.spv`
2. `createShaderModule()` 加载 SPIR-V
3. `VkPipelineShaderStageCreateInfo` 配置 vertex/fragment stage
4. 创建/复用 `VkPipelineLayout`、`VkRenderPass`
5. `vkCreateGraphicsPipeline` 创建 pipeline
6. 在 `cleanup()` 中 `vkDestroyPipeline`、`vkDestroyPipelineLayout`

### 添加新 Descriptor 资源

1. 在 `VkDescriptorSetLayout` 中增加 `VkDescriptorSetLayoutBinding`
2. 更新 `VkDescriptorPool` 的 pool size（如 `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` 数量）
3. 分配/更新 `VkDescriptorSet`，`vkUpdateDescriptorSets` 绑定 buffer/image
4. Shader 中对应 `layout(binding = N)` 声明

### 处理 SwapChain 重建

1. `vkDeviceWaitIdle` 等待当前帧完成
2. `cleanupSwapChain()`：销毁 framebuffer、imageView、swapchain、renderPass 等
3. `createSwapChain()` 及依赖的 `createImageViews`、`createFramebuffers`、`createRenderPass`
4. 若 pipeline 依赖 extent/format，需一并重建

## 调试建议

- **Validation Layer 报错**：优先看 `pMessage`，常见为未同步、错误 layout、binding 不匹配
- **黑屏/花屏**：检查 view/projection 矩阵、depth 测试、framebuffer attachment 配置
- **崩溃**：检查句柄是否为 `VK_NULL_HANDLE`、销毁顺序、多线程访问

## 参考

- 项目规则：`.cursor/rules/vulkan-*.mdc`
- Vulkan Spec: https://registry.khronos.org/vulkan/
