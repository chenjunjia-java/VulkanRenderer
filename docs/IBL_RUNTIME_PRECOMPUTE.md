# IBL 运行时预计算

本项目的 PBR IBL（Image-Based Lighting）在**启动时**于 GPU 上一次性预计算，不依赖离线烘焙产物。

## 预计算流程

1. **环境 Cubemap**：从等距柱状 HDR (.hdr) 转为 `RGBA16F` cubemap，默认 512×512
2. **Irradiance Map**：对环境 cubemap 做漫反射卷积，输出 32×32 `RGBA16F` cubemap
3. **Prefilter Map**：GGX 重要性采样预滤波，128×128，含 mip（约 8 级），`RGBA16F`
4. **BRDF LUT**：2D 积分表，512×512，`RG16F`

## 默认参数

| 贴图            | 分辨率          | 格式        | 采样数/说明              |
|-----------------|-----------------|-------------|---------------------------|
| Env Cubemap     | 512×512         | RGBA16F     | 6 面                      |
| Irradiance      | 32×32           | RGBA16F     |  hemisphere ~5000 采样    |
| Prefilter       | 128×128 + mip   | RGBA16F     | 1024 重要性采样 / mip     |
| BRDF LUT        | 512×512         | RG16F       | 1024 积分采样             |

## 调节质量

在 `IblPrecompute::compute()` 中可调整：

- `irradianceSize`：默认 32，提高可增强漫反射精度，代价为显存与预计算时间
- `prefilterSize`：默认 128，提高可增强镜面反射精度
- `brdfLutSize`：默认 512，一般无需修改

在 shader 中：

- `irradiance_convolution.frag`：`sampleDelta` 控制半球采样密度（默认 0.025）
- `prefilter.frag`：`SAMPLE_COUNT` 控制重要性采样数（默认 1024）
- `brdf_integrate.frag`：`SAMPLE_COUNT` 控制积分采样数（默认 1024）

## 启动耗时预期

- Debug：约 1–3 秒（视设备而定）
- Release：通常 < 1 秒

预计算在 `Renderer::init()` 中、`EquirectToCubemap::convert()` 之后执行，完成后结果常驻 GPU，运行时仅做采样，无额外成本。

## 依赖

- HDR 等距柱状贴图：`assets/textures/hdr/qwantani_dusk_2_puresky_4k.hdr`
- 若加载失败，将使用占位 IBL（1×1 灰色 cubemap / 灰 LUT），PBR 仍可运行但环境光较弱
