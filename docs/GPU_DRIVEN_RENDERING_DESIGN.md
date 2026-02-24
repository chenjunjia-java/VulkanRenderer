# GPU 驱动渲染管线设计（Compute 剔除 + 间接绘制）

## 目标

将 CPU 端的视锥剔除、遮挡剔除、draw 下发逻辑迁移到 GPU，减轻 CPU 瓶颈（当前 collect ~8ms、issue ~25ms、共 ~33ms）。

## 当前架构瓶颈

- **Depth Prepass**: 2743 次 CPU 遍历 + drawIndexed
- **Forward Pass**: 2909 次 CPU 遍历 + drawIndexed
- 每次 draw：`bindPipeline`、`bindVertexBuffers`、`bindIndexBuffer`、`bindDescriptorSets`、`pushConstants`、`drawIndexed`
- 场景树遍历、视锥/遮挡剔除、排序全部在 CPU

## GPU 驱动方案总览

```
[CPU] 仅上传：draw 列表（meshId, transform, AABB）+ 相机矩阵
        ↓
[GPU] Compute Pass 1：视锥剔除（可选 + 遮挡）
        ↓
[GPU] 输出：VkDrawIndexedIndirectCommand 缓冲区（instanceCount=0/1）
        ↓
[GPU] vkCmdDrawIndexedIndirect / DrawIndirectCount
```

## 前置条件：数据结构改造

### 1. 合并 Vertex/Index Buffer（或 Bindless）

当前每个 mesh 独立 VB/IB，间接绘制要求**同一次调用共享 VB/IB**。

**方案 A：合并大 Buffer**
- 将所有 mesh 顶点/索引合并到一个大 buffer
- 每个 mesh 记录：`vertexOffset`、`firstIndex`、`indexCount`

```cpp
// DrawInstance 上传到 GPU
struct DrawInstance {
    glm::mat4 model;           // 世界变换
    uint32_t vertexOffset;     // 顶点偏移（合并 buffer）
    uint32_t firstIndex;       // 索引起始
    uint32_t indexCount;
    uint32_t materialIndex;
    // AABB 用于剔除（或由 model 变换得到）
    glm::vec3 aabbMin;
    glm::vec3 aabbMax;
};
```

**方案 B：Bindless 顶点/索引**
- 使用 `VK_EXT_descriptor_indexing`
- Descriptor 中为 `storage buffer array`，按 mesh 索引
- 需要更复杂的 shader 和 descriptor 管理

### 2. Bindless 材质

当前每材质一个 descriptor set，draw 时需要频繁切换。间接绘制无法在单个 `vkCmdDrawIndexedIndirect` 内切换 descriptor。

**改造**：
- 所有材质纹理放入 **texture array** / **bindless texture**
- 材质参数（baseColor、metallic、roughness 等）放入 **storage buffer**，按 `materialIndex` 索引
- 顶点/片元着色器用 `gl_DrawID` 或 instance 数据获取 `materialIndex`，再采样对应纹理

```glsl
// 伪代码
layout(binding = X) uniform sampler2D tex2DArray[256];
layout(binding = Y) readonly buffer MaterialBuffer { MaterialData materials[]; };
uint matIdx = drawData[gl_DrawID].materialIndex;
vec4 baseColor = texture(tex2DArray[materials[matIdx].baseColorTexIdx], uv) * materials[matIdx].baseColorFactor;
```

## Compute Shader 剔除

### 视锥剔除（优先实现）

```glsl
// culling.comp
layout(local_size_x = 256) in;

layout(binding = 0) uniform CullingUBO {
    mat4 viewProj;
    vec4 frustumPlanes[6];  // 或从 viewProj 推导
};

layout(binding = 1) readonly buffer DrawInput {
    DrawInstance draws[];
};

layout(binding = 2) buffer DrawOutput {
    VkDrawIndexedIndirectCommand commands[];
};

layout(binding = 3) buffer CountBuffer {
    uint visibleCount;  // 若用 DrawIndirectCount
};

bool frustumTest(vec3 aabbMin, vec3 aabbMax) {
    vec3 corners[8] = { ... };  // AABB 8 顶点
    for (int p = 0; p < 6; p++)
        if (all(greaterThan(frustumPlanes[p].xyz * corners[i] + frustumPlanes[p].w, vec3(0))))
            return false;  // 全部在平面外侧
    return true;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= draws.length()) return;

    DrawInstance d = draws[i];
    bool visible = frustumTest(d.aabbMin, d.aabbMax);
    uint instCount = visible ? 1u : 0u;

    commands[i] = VkDrawIndexedIndirectCommand(
        d.indexCount,
        instCount,
        d.firstIndex,
        d.vertexOffset,  // vertexOffset 需 Vulkan 1.2+ 或 VK_KHR_draw_indirect_count 扩展
        0
    );

    if (visible)
        atomicAdd(visibleCount, 1u);  // 用于 DrawIndirectCount
}
```

### 遮挡剔除（进阶）

**方案 1：Hierarchical Z-Buffer (HIZ)**
- Depth prepass 后生成 Mip 链，每级存 min depth
- Compute 中采样 HIZ，判断 AABB 投影是否被遮挡

**方案 2：软件光栅化**
- 在 compute 中光栅化 AABB 到低分辨率 buffer（如 64x64）
- 与 depth 比较，但实现复杂且可能有同步问题

**方案 3：先不做 GPU 遮挡**
- 只做视锥剔除即可显著减轻 CPU
- 遮挡可暂时保留 CPU 实现，或后续再加 HIZ

## 间接绘制调用

### 方式 1：固定数量 + instanceCount=0（推荐起步）

```cpp
// 所有 2909 个 command 都写入，剔除的 instanceCount=0
vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer, 0, 2909, sizeof(VkDrawIndexedIndirectCommand));
```

- 实现简单，无需 `VK_KHR_draw_indirect_count`
- 剔除的 draw 变成 no-op，GPU 仍会遍历 command，但几乎无几何体处理

### 方式 2：DrawIndirectCount（更优）

```cpp
// 需要 VK_KHR_draw_indirect_count
vkCmdDrawIndexedIndirectCount(commandBuffer, indirectBuffer, 0, countBuffer, 0, maxDraws, stride);
```

- Compute 输出 compact 后的 command 列表 + count
- 需额外一次 prefix sum / stream compact，或用 `atomicAdd` 写 count + 第二次 pass 做 compact

## 实施路线图

### Phase 1：合并 Buffer + 预计算 Draw 表（不改剔除）

1. 将所有 `GpuMesh` 合并为一个 vertex buffer、一个 index buffer
2. CPU 每帧构建 `DrawInstance[]` 上传到 GPU
3. **仍用 CPU 剔除**，但改为：只对可见项写 `DrawIndexedIndirectCommand`，用 `vkCmdDrawIndexedIndirectCount`
4. 验证渲染结果一致

### Phase 2：Compute 视锥剔除

1. 添加 culling compute shader
2. Draw 表常驻 GPU，每帧只更新 transform 和 AABB（若动画）
3. Compute 输出 indirect buffer（instanceCount 0/1）
4. 移除 CPU 视锥剔除和 collect 逻辑

### Phase 3：Bindless 材质

1. 改造 descriptor：纹理数组 + 材质 storage buffer
2. 顶点/片元 shader 用 `gl_DrawID` 取 materialIndex
3. 移除 per-draw `bindDescriptorSets`

### Phase 4（可选）：GPU 遮挡

1. 实现 HIZ 生成
2. Compute 中采样 HIZ 做遮挡测试

## 依赖与扩展

| 功能             | Vulkan 要求                               |
|------------------|-------------------------------------------|
| DrawIndexedIndirect | 标准支持                                |
| DrawIndirectCount   | `VK_KHR_draw_indirect_count`            |
| vertexOffset in DrawIndirect | Vulkan 1.2 或扩展                  |
| Bindless 纹理       | `VK_EXT_descriptor_indexing`            |

## 预期收益

- **CPU**：去除 ~8ms collect + ~25ms issue 中的大部分，保留少量 upload（仅矩阵等）
- **Draw call**：由 2909 次 `drawIndexed` 变为 1 次 `drawIndexedIndirect`（或 IndirectCount）
- **可扩展性**：物体数量增大时，CPU 成本几乎不增加

## 参考

- [NVIDIA - GPU-Based Occlusion Culling](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-6-hardware-occlusion-queries-made-useful)
- [Vulkan multi-draw indirect](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VkDrawIndexedIndirectCommand)
- [AMD - GPU-based culling](https://gpuopen.com/learn/conservative-rasterization/)
