#pragma once

#include "Configs/AppConfig.h"

namespace RuntimeConfig {

// --- Debug view (mapped to ubo.iblParams.w, see AppConfig comments) ---
// 0=Off, 2=baseColor, 3=Ng, 4=AO
inline int debugViewMode = AppConfig::DEBUG_VIEW_MODE;

// --- IBL toggles/strengths (mapped to ubo.iblParams.xyz) ---
inline bool enableDiffuseIbl = AppConfig::ENABLE_DIFFUSE_IBL;
inline float diffuseIblStrength = AppConfig::DIFFUSE_IBL_STRENGTH;
inline bool enableSpecularIbl = AppConfig::ENABLE_SPECULAR_IBL;
inline float specularIblStrength = AppConfig::SPECULAR_IBL_STRENGTH;

// --- AO toggle (mapped to ubo.iblParams.z) ---
inline bool enableAo = AppConfig::ENABLE_AO;

// --- Post-process debug (TonemapBloomPass push constants) ---
// 0=Final Tonemap+Bloom, 1=Show scene_color, 2=Show bloom_a
inline int postprocessDebugView = AppConfig::POSTPROCESS_DEBUG_VIEW;

// --- Bloom params (Tonemap/Bloom passes push constants) ---
inline float bloomThreshold = AppConfig::BLOOM_THRESHOLD;
inline float bloomSoftKnee = AppConfig::BLOOM_SOFT_KNEE;
inline float bloomIntensity = AppConfig::BLOOM_INTENSITY;
inline float bloomBlurRadius = AppConfig::BLOOM_BLUR_RADIUS;

// --- Tonemap params ---
inline float tonemapExposure = AppConfig::TONEMAP_EXPOSURE;

inline void resetToDefaults()
{
    debugViewMode = AppConfig::DEBUG_VIEW_MODE;
    enableDiffuseIbl = AppConfig::ENABLE_DIFFUSE_IBL;
    diffuseIblStrength = AppConfig::DIFFUSE_IBL_STRENGTH;
    enableSpecularIbl = AppConfig::ENABLE_SPECULAR_IBL;
    specularIblStrength = AppConfig::SPECULAR_IBL_STRENGTH;
    enableAo = AppConfig::ENABLE_AO;
    postprocessDebugView = AppConfig::POSTPROCESS_DEBUG_VIEW;
    bloomThreshold = AppConfig::BLOOM_THRESHOLD;
    bloomSoftKnee = AppConfig::BLOOM_SOFT_KNEE;
    bloomIntensity = AppConfig::BLOOM_INTENSITY;
    bloomBlurRadius = AppConfig::BLOOM_BLUR_RADIUS;
    tonemapExposure = AppConfig::TONEMAP_EXPOSURE;
}

}  // namespace RuntimeConfig

