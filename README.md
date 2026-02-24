## VulkanLearning（Vulkan 渲染器学习工程）

本仓库是一个以“把现代实时渲染关键模块落到 Vulkan 上”为目标的学习项目：在 **Vulkan 1.2 + Dynamic Rendering** 的基础上，逐步加入 **间接绘制（Multi-Draw Indirect）**、轻量 **Rendergraph**、基于 **RayQuery** 的 **RTAO** 与 **软阴影**、以及 **PBR IBL 运行时预计算** 等模块，并配套性能分析与调试 UI。

### 在线演示（GitHub Pages）

- **演示页面**：仓库根目录提供 `index.html`，GitHub Pages 工作流会上传整个仓库进行静态部署（包含 `videos/`）。
- **演示视频**：`videos/2026-02-24-15-20-17.mp4`（页面会直接引用该视频用于展示渲染效果）。

> 提示：Pages 的实际链接形如 `https://<用户名>.github.io/<仓库名>/`。如果你 fork / 改名了仓库，链接也会随之变化。

### 已实现特性概览（与代码对应）

- **渲染架构分层**
  - `Application`（窗口/输入/主循环）与 `Renderer`（Vulkan 渲染逻辑）解耦
  - Vulkan 资源以 RAII / init-cleanup 成对管理，SwapChain 可重建
- **Rendergraph（渲染图）**
  - `Rendergraph` 通过 pass 的输入/输出资源名构图，`Compile()` 时做依赖排序，`Execute()` 时按顺序执行
  - 对 internal resource 做 image layout 跟踪，对 external（swapchain）按 image handle 单独跟踪 layout 并在渲染前后切换
- **间接绘制（Multi-Draw Indirect）**
  - `GlobalMeshBuffer` 合并全局 VB/IB；CPU 每帧构建 `vk::DrawIndexedIndirectCommand`（`FrameManager::prepareSharedOpaqueIndirect`）
  - `DepthPrepass` / `ForwardPass` 使用 `vkCmdDrawIndexedIndirect` 批量绘制不透明物体；顶点着色器用 `gl_BaseInstance` 从 `DrawData` SSBO 取 per-draw model matrix
  - 当前按 **(doubleSided, materialIndex)** 分桶，桶内一条 MDI；透明物体仍走单个 `drawIndexed`（但复用同一套 DrawData）
- **RTAO（Ray-Traced Ambient Occlusion）**
  - Compute 端使用 `rayQueryEXT` 做 AO trace，并做基于 `prevViewProj` 的历史重投影与噪声感知的 temporal accumulation
  - 空间去噪：A-Trous 迭代滤波（结合深度/法线 edge-stopping + range 权重）
  - Trace → A-Trous → `rtao_upsample.comp` 写入 `rtao_full`（`R16F` storage image）。当前实现默认按屏幕分辨率运行，`upsample` 阶段更像“最终写回/选择输出”，后续可扩展为半分辨率上采样
  - 在 PBR 中与材质 AO 贴图融合
- **软阴影（RayQuery）**
  - 方向光：基于太阳角半径做“太阳盘面多射线采样”（泊松盘），得到软阴影过渡
  - 点光源：单射线 shadow ray 可见性测试（有距离相关 bias）
- **IBL（运行时预计算）**
  - 启动时从 HDR 等距柱状贴图生成 env cubemap，预计算 irradiance / GGX prefilter / BRDF LUT（详见 `docs/IBL_RUNTIME_PRECOMPUTE.md`）
  - PBR 中采样 irradiance（固定 LOD）+ prefilter（roughness→LOD）+ BRDF LUT，并做近似 specular occlusion
- **后处理**
  - Bloom（Extract + 双向 Blur）+ Tonemap（最终输出到 swapchain）
- **调试与统计**
  - ImGui 面板提供 IBL 强度、Bloom/Tonemap 参数调节与每帧 CPU 计时（Acquire/Record/Submit/Present/Total）
  - Rendergraph 记录每个 pass 的 CPU 侧执行耗时（可用于粗定位）

### 关键技术细节（简述）

#### 1) 间接绘制：从“每个 mesh 一个 draw”到“按材质桶批量 MDI”

项目当前走的是“**CPU 生成 indirect commands + GPU 执行 MDI**”的过渡形态：

- **为什么要合并 VB/IB**：`vkCmdDrawIndexedIndirect` 的一次调用只能绑定一套 VB/IB；所以将所有 mesh 的数据合并到 `GlobalMeshBuffer`，每个 mesh 记录 `vertexOffset / firstIndex / indexCount`。
- **为什么用 `firstInstance=drawId`**：顶点着色器里用 `gl_BaseInstance` 做索引，从 `DrawData` SSBO 取对应物体的 model matrix，实现“无 per-draw push model”。
- **仍然存在的 CPU 成本**：目前 `prepareSharedOpaqueIndirect` 每帧会遍历 linear nodes 计算 world matrix，并把 indirect commands 复制到 host-visible buffer；这也是后续 GPU-driven（compute culling + indirect count/compact）的切入点（见 `docs/GPU_DRIVEN_RENDERING_DESIGN.md`）。

#### 2) Rendergraph：最小可用的“依赖排序 + layout 管理”

- pass 通过 `getInputs()/getOutputs()` 声明资源依赖；Rendergraph 在 `Compile()` 做拓扑排序并检测环
- `Execute()` 在每个 pass 之前把输入/输出转到 pass 期望 layout（internal 用 `ImageResource::currentLayout` 跟踪，external 以 image handle 作为 key 跟踪）
- 当前 barrier 推断策略保持“项目所需的最小集合”，并有 conservative fallback（`eAllCommands`）

#### 3) RTAO：RayQuery + temporal + A-Trous

- **Trace**：compute 中构造 AO hemisphere 采样方向，对每个像素发射多条 ray query；对 MASK 材质做 alpha-test，BLEND 材质视为不遮挡
- **Temporal**：用 `prevViewProj` 从 worldPos 反投影到上一帧 UV，做深度差/噪声感知的 history clamp + reject
- **Denoise**：A-Trous 迭代（可配置迭代次数与 step），权重由 spatial/range/depth/normal 共同决定，避免跨几何边界糊掉

#### 4) 软阴影：方向光“太阳盘采样”

方向光阴影并非 shadow map，而是 RayQuery 直接查询 TLAS：

- `sunAngularRadius` 控制盘面大小
- `softShadowSampleCount` 控制采样数（<=1 时退化为硬阴影）

### 当前局限与待解决问题

- **CPU 瓶颈（帧率偏低）**
  - 现阶段 command buffer 录制 + 大量 draw 下发仍偏重；你观察到 `record` 可能占到 ~70% 的 CPU 时间
  - 静态场景下若 TLAS 每帧更新，会造成额外 CPU 负担（已在 `Renderer` 中做了“模型矩阵缓存，变化才更新 TLAS”的优化思路）
  - Occlusion Query 若启用且回读使用 `WAIT`，会产生明显 CPU-GPU 同步（详见 `docs/CPU_PERFORMANCE_ANALYSIS.md`）
- **画面问题**
  - **远处贴图闪烁 / shimmering**：常见原因是 mip 链不足、各向异性不足、alpha-mask 覆盖抖动等；项目已开启 anisotropy（若设备支持），但仍需检查 KTX2 资产是否包含完整 mip 链及合适的采样器参数
  - **锯齿**：当前主要依赖 MSAA + 一些 shader 稳定性处理（例如 specular AA），但没有 TAA；远景与高频细节仍可能明显
- **GPU-driven 渲染尚未完成**
  - 目前是 CPU 构建 indirect commands；下一步是 compute culling（视锥/遮挡）→ compact / indirect count → 真正把 collect/issue 压到 GPU（路线见 `docs/GPU_DRIVEN_RENDERING_DESIGN.md`）

---

## VulkanLearning (clangd / clang-cl / CMake)

### 依赖

- **Vulkan SDK**（需要环境变量 `VULKAN_SDK`）
- **LLVM/Clang**（包含 `clang-cl`）
- **CMake**
- **Ninja**
- **MSVC + Windows SDK**（不用 VS IDE 也行，但需要其头文件/库；脚本会自动加载环境）

### 生成 compile_commands.json（给 clangd 用）

在项目根目录执行：

```powershell
.\scripts\run_with_msvc_env.ps1 -Command "cmake --preset clang-cl-debug"
```

生成位置：

- `build\clang-cl-debug\compile_commands.json`

### 编译（Debug）

```powershell
.\scripts\run_with_msvc_env.ps1 -Command "cmake --build --preset clang-cl-debug"
```

生成的可执行文件：

- `build\clang-cl-debug\VulkanLearning.exe`
