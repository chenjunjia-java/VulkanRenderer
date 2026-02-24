# CPU 渲染性能分析报告

## 概述

当 GPU 未达瓶颈但帧率仍低于 30 FPS 时，通常是 **CPU 端成为瓶颈**。本报告分析了主循环和鼠标交互相关的 CPU 调用链，并给出优化建议。

## 主循环调用链（每帧执行）

```
mainLoop()
├── glfwPollEvents()          ← 鼠标回调在此被触发
├── processInput(deltaTime)
├── renderer.update(deltaTime)
├── cullingSystem.CullScene(...)
└── renderer.drawFrame()
```

---

## 1. 鼠标拖动期间的 CPU 调用

### 调用路径

```
glfwPollEvents()
  └── mouseCallback(xpos, ypos)     [GLFW 回调]
        └── camera.processMousePosition(xpos, ypos, rightPressed)
              └── processMouseMovement(xoffset, yoffset)
                    └── updateCameraVectors()   // 少量浮点运算
```

### 结论：**鼠标处理不是瓶颈**

- 每次移动仅做：差值计算、yaw/pitch 更新、`updateCameraVectors()`（若干三角函数和向量运算）
- 计算量极小，可以忽略

---

## 2. 已识别的 CPU 瓶颈（按影响程度排序）

### 2.1 【高】TLAS 每帧重建与更新（静态场景）

**位置**：`Renderer::recordCommandBuffer` → `rebuildRayTracingInstances` + `rayTracingContext.updateTopLevelAS`

**问题**：
- Bistro 场景是静态的，`modelMatrix` 每帧相同
- 每帧仍会：
  - 遍历整棵模型树，重建 `rayTracingInstances`
  - `writeInstances()` 写满 host 映射的 instance buffer
  - 在 command buffer 中录制 TLAS 更新
- 对于数百个 instance，这是每帧的明显开销

**优化**：对静态场景，在初始化时构建 TLAS 一次，后续帧不再更新。

---

### 2.2 【高】Occlusion Query 回读（若启用）

**位置**：`Renderer::drawFrame` → `frameManager.readOcclusionResults`

```cpp
// FrameManager.cpp:77 - 使用 VK_QUERY_RESULT_WAIT_BIT
const vk::QueryResultFlags flags = vk::QueryResultFlagBits::e64 
    | vk::QueryResultFlagBits::eWithAvailability 
    | vk::QueryResultFlagBits::eWait;  // ← CPU 会阻塞等待
```

**问题**：
- `eWait` 导致 CPU 阻塞直到 occlusion 结果可用
- Bistro 的 `occlusionQueryCount` = `linearNodes.size()`（数百个）
- 每次回读都要做一次 CPU-GPU 同步

**优化**：
- 若对帧率敏感，可暂时关闭 `enableOcclusionCulling`
- 或改用 `eWait` 以外的方式，避免每帧强制同步

---

### 2.3 【中】双重 Fence 等待

**位置**：`Renderer::drawFrame`

```cpp
device.waitForFences(inFlightFence, VK_TRUE, UINT64_MAX);   // 等待上一帧 GPU 完成
// ... acquire ...
device.waitForFences(imageAvailableFence, VK_TRUE, UINT64_MAX);  // 等待 acquire 完成
```

**说明**：逻辑正确，但对 CPU 来说存在两次阻塞点。若 GPU 很快完成，这些等待会相对短暂，主要瓶颈在其他 CPU 工作。

---

### 2.4 【中】Command Buffer 录制与大量 Draw Call

**位置**：`recordCommandBuffer` → `rendergraph->Execute` → `ForwardPass::render`

**问题**：
- Bistro 有大量 mesh，每帧都要：
  - 遍历节点树
  - 视锥/遮挡测试（若启用）
  - 对每个可见 mesh 调用 `vkCmdDrawIndexed`、`bindPipeline`、`bindDescriptorSets` 等
- 录制大量 Vulkan 命令本身有 CPU 成本

**优化**：
- 视情况使用 instancing 或 batch，减少 draw call 数量
- 使用更粗粒度的 frustum culling 降低遍历和录制成本

---

### 2.5 【低】CullScene 对未使用数据的计算

**位置**：`VulkanApplication::mainLoop` → `cullingSystem.CullScene(scene.GetEntities(), ...)`

**问题**：
- `scene.GetEntities()` 每帧构造新的 `std::vector<Entity*>`
- `CullScene` 的 `visibleEntities` **从未被 Renderer 使用**
- Renderer 直接使用 `Model` 的节点树，而非 ECS 的 `visibleEntities`
- 当前 `scene` 一般为空，开销较小，但逻辑上是无用计算

**优化**：若 ECS 不参与渲染，可移除 `CullScene` 调用；若将来使用，再接入并避免每帧 `GetEntities()` 分配。

---

### 2.6 【低】Scene::GetEntities 每帧分配

**位置**：`CullingSystem::CullScene` 的参数 `scene.GetEntities()`

```cpp
// Scene.cpp:32 - 每次调用都分配新 vector
std::vector<Entity*> Scene::GetEntities() {
    std::vector<Entity*> result;
    result.reserve(entities.size());
    for (auto& entity : entities) {
        result.push_back(entity.get());
    }
    return result;  // 按值返回，拷贝
}
```

**说明**：在 entities 不多时影响有限，但属于可避免的每帧分配。

---

## 3. 快速验证建议

### 3.1 禁用 Occlusion Culling 测试

`enableOcclusionCulling` 默认为 `false`，若你已手动开启，可先关闭并对比帧率。

### 3.2 缓存 TLAS（静态场景）

在静态场景下，TLAS 只需在初始化时构建一次，运行时不再更新，可显著减少每帧 CPU 工作。

### 3.3 添加简单计时

在 `drawFrame` 内对关键步骤加计时，定位最耗时的环节：

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
device.waitForFences(...);
auto t1 = std::chrono::high_resolution_clock::now();
// 记录 wait vs record vs submit 的时间
```

---

## 4. 总结

| 项目               | 是否瓶颈 | 备注                           |
|--------------------|----------|--------------------------------|
| 鼠标回调           | 否       | 计算量极小                     |
| TLAS 每帧更新       | 是       | 静态场景可仅初始化时构建        |
| Occlusion 回读     | 是（若开启） | 强制 CPU-GPU 同步          |
| 双重 Fence 等待    | 部分     | 正常同步，但增加等待点          |
| Command 录制       | 是       | 大量 draw call 有 CPU 开销     |
| CullScene          | 否（当前） | 对空 scene，且结果未使用    |

**优先建议**：先实现 TLAS 静态缓存，再考虑减少 occlusion 回读或优化 command 录制/批处理。
