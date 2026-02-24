#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace AppConfig {

// Default window width.
constexpr uint32_t WIDTH = 1200;
// Default window height.
constexpr uint32_t HEIGHT = 800;
// Frames in flight (CPU/GPU pacing).
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
// Asset root path (relative to executable working directory).
inline const std::string ASSETS_PATH = "assets/";
// HDR 环境贴图（等距柱状）路径：用于天空盒与 IBL 预计算输入
inline const std::string ENV_HDR_PATH = ASSETS_PATH + "textures/hdr/qwantani_dusk_2_puresky_4k.hdr";
// Global scene scale for large glTF scenes (e.g. bistro).
constexpr float SCENE_MODEL_SCALE = 1.0f;

// Camera initial position Z (distance from origin, Y-up).
constexpr float CAMERA_INITIAL_Z = 30.0f;
// Camera movement speed (WASD/QE units per second).
constexpr float CAMERA_MOVEMENT_SPEED = 15.0f;
// Camera mouse sensitivity (rotation speed).
constexpr float CAMERA_MOUSE_SENSITIVITY = 0.1f;

// ========== 性能调试 [Perf] ==========
// 总开关：是否打印性能统计
constexpr bool ENABLE_PERF_DEBUG = false;
// 打印间隔（每多少帧输出一次，减少日志刷屏）
constexpr uint32_t PERF_PRINT_INTERVAL = 120u;
// 细分 Pass 统计：打印 RTAO（trace/atrous/upsample）+ Bloom（extract/blurH/blurV/tonemap）的 CPU 耗时
constexpr bool PERF_PRINT_RTAO = true;
constexpr bool PERF_PRINT_BLOOM = true;
// 是否打印 forward pass 细分（collect/sort/issue、draws、binds）
constexpr bool PERF_PRINT_FORWARD_DETAIL = true;
// 是否打印帧管线主阶段（acquire/record/ubo/submit/present/total）
constexpr bool PERF_PRINT_FRAME_STAGES = true;

// 光追反射：开发时是否启用（关闭可加快调试迭代）
constexpr bool ENABLE_RAY_TRACED_REFLECTION = false;

// 光追反射：强制启用反射的材质索引（用于无 extras.reflective 的模型，如 Bistro）
// 留空则仅依赖 glTF extras。可设 {0} 等以快速测试。
inline const std::vector<uint32_t> REFLECTIVE_MATERIAL_INDICES = {};

// 光追反射 bindless 纹理数组最大材质数量（需与 descriptor layout 一致）
constexpr uint32_t MAX_REFLECTION_MATERIAL_COUNT = 256u;

// IBL 调试开关：用于快速定位环境光黑块来源
constexpr bool ENABLE_DIFFUSE_IBL = true;   // 漫反射 IBL（irradianceMap）
constexpr float DIFFUSE_IBL_STRENGTH = 0.3f;  // 漫反射强度，0~1，避免场景过白
constexpr bool ENABLE_SPECULAR_IBL = true;  // 镜面 IBL（prefilterMap + brdfLUT）
constexpr float SPECULAR_IBL_STRENGTH = 0.3f;  // 镜面强度，0~1

// 天空盒 IBL 调试：用于可视化 IBL 预计算结果，验证 irradiance/prefilter 是否正确
// 0=原始环境立方体贴图，1=irradiance（漫反射），2=prefilter（镜面，base mip）
constexpr int SKYBOX_IBL_DEBUG_MODE = 0;

// Bloom（稳定优先版本）：HDR 提亮阈值 + 软膝 + 强度
constexpr bool ENABLE_BLOOM = true;
// 通用推荐（大多数 HDR 场景比较稳）：threshold=1.0, softKnee=0.5, intensity=0.08, blurRadius=1.0
// - threshold: 越低越“泛”，越高越克制（建议 0.8~1.5）
// - softKnee : 0~1，越大过渡越柔（建议 0.3~0.7）
// - intensity: bloom 叠加强度（建议 0.04~0.12）
// - blurRadius: 模糊采样步长倍率（建议 0.75~1.5）
constexpr float BLOOM_THRESHOLD = 1.0f;
constexpr float BLOOM_SOFT_KNEE = 0.5f;
constexpr float BLOOM_INTENSITY = 0.3f;
constexpr float BLOOM_BLUR_RADIUS = 3.0f;

// Tonemap（后处理统一完成曝光+压缩动态范围）
// 通用推荐：0.6~1.2，越大越亮。该值用于 TonemapBloomPass，不再在 PBR shader 内二次 tone-map。
constexpr float TONEMAP_EXPOSURE = 0.8f;

// PostProcess 调试输出（用于排查 scene_color 是否写入/采样成功）
// 0=正常 Tonemap+Bloom
// 1=直接显示 scene_color（做简单曝光压缩：rgb/(1+rgb)）
// 2=直接显示 bloom_a（同上）
constexpr int POSTPROCESS_DEBUG_VIEW = 0;

// AO 调试开关：关闭后等效 ao=1（不再调制环境光/间接光）
constexpr bool ENABLE_AO = true;

// Ray Traced AO: rayQuery + TLAS 半球可见性估计（仅影响间接光）
constexpr bool ENABLE_RTAO = true;
constexpr uint32_t RTAO_RAY_COUNT = 8;         // 每像素射线数，建议 4~8
constexpr float RTAO_RADIUS = 1.2f;            // 世界空间最大追踪距离
constexpr float RTAO_BIAS = 0.0005f;             // 法线偏移，避免自相交
constexpr float RTAO_STRENGTH = 1.0f;          // 0~1，控制 RTAO 对最终 AO 的影响
constexpr float RTAO_TEMPORAL_ALPHA = 0.8f;    // history 权重，越高越稳但拖影更明显
constexpr float RTAO_DISOCCLUSION_THRESHOLD = 0.35f;  // 当前值与历史值差异过大时丢弃 history
constexpr bool ENABLE_RTAO_SPATIAL_DENOISE = true;
constexpr uint32_t RTAO_ATROUS_ITERATIONS = 5; // 推荐 3（速度优先）

// 点光源开关：关闭后 pointLightCount=0，仅方向光生效
constexpr bool ENABLE_POINT_LIGHTS = false;

// ImGui 调试 UI：启用后显示 Dear ImGui 叠加层（Demo 窗口等）
constexpr bool ENABLE_IMGUI = true;

// Debug view mode (iblParams.w).
// 0=Off (正常 PBR 渲染)
// 1=NdotL (max across lights)
// 2=baseColor
// 3=Ng (geometric normal)
// 4=AO (combined AO: SSAO * RTAO)
// 5~8=同 4
// 9=normal.y (法线 Y 分量，朝上面亮；RTAO upsample 输出)
// 10=bias (RTAO trace 的 bias 可视化)
// 11=footprint (RTAO trace 的 footprint 可视化)
// 12=current (RTAO 当前帧，无时域累积)
constexpr int DEBUG_VIEW_MODE = 0;

}  // namespace AppConfig
