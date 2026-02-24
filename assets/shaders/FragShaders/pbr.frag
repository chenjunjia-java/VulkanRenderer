#version 460
#extension GL_ARB_separate_shader_objects : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

layout(binding = 0) uniform PBRUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 directionalLightDir;   // xyz = direction (to light), w = enable (0/1)
    vec4 directionalLightColor;
    vec4 directionalLightParams; // x=sunAngularRadius(rad), y=softShadowSampleCount
    vec4 lightPositions[3];     // 3 point lights
    vec4 lightColors[3];
    vec4 camPos;
    vec4 params;                // x=exposure, y=gamma, z=ambientStrength, w=pointLightCount
    vec4 iblParams;             // x=enableDiffuseIBL, y=enableSpecularIBL, z=enableAO
    vec4 rtaoParams0;           // x=enableRTAO, y=rayCount, z=radius, w=bias
    vec4 rtaoParams1;           // x=strength, y=temporalAlpha, z=disocclusionThreshold, w=frameIndex
    mat4 prevViewProj;
} ubo;

layout(binding = 1) uniform sampler2D baseColorMap;
layout(binding = 2) uniform sampler2D metallicRoughnessMap;
layout(binding = 3) uniform sampler2D normalMap;
layout(binding = 4) uniform sampler2D occlusionMap;
layout(binding = 5) uniform sampler2D emissiveMap;

layout(binding = 6) uniform accelerationStructureEXT topLevelAS;

// 光追反射 bindless（教程 Task 9/10/11）
struct InstanceLUTEntry {
    uint materialID;
    uint indexBufferOffset;
};
layout(binding = 7) readonly buffer InstanceLUTBlock {
    InstanceLUTEntry entries[];
} instanceLUT;
layout(binding = 8) readonly buffer IndexBufferBlock { uint indices[]; } indexBuffer;
layout(binding = 9) readonly buffer UVBufferBlock { vec2 uvs[]; } uvBuffer;
layout(binding = 10) uniform sampler2D baseColorTextures[256];
layout(binding = 12) uniform samplerCube irradianceMap;
layout(binding = 13) uniform samplerCube prefilterMap;
layout(binding = 14) uniform sampler2D brdfLUT;
layout(binding = 15) uniform sampler2D rtaoFull;

layout(push_constant) uniform PBRPushConstants {
    mat4 model;
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 materialParams0; // x metallicFactor, y roughnessFactor, z alphaCutoff, w normalScale
    vec4 materialParams1; // x occlusionStrength, y alphaMode(0/1/2), z/w unused
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

vec3 safeNormalize(vec3 v)
{
    float len2 = dot(v, v);
    if (len2 > 1e-12) {
        return v * inversesqrt(len2);
    }
    // Fallback to a stable axis; avoids NaNs in BRDF math.
    return vec3(0.0, 1.0, 0.0);
}

// ACES filmic tone mapping (Krzysztof Narkowicz approximation).
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 Reinhard(vec3 x)
{
    return x / (x + vec3(1.0));
}

float DistributionGGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    float ggxV = GeometrySchlickGGX(NdotV, roughness);
    float ggxL = GeometrySchlickGGX(NdotL, roughness);
    return ggxV * ggxL;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float hash11(float p)
{
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float radicalInverseVdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint i, uint n)
{
    return vec2(float(i) / max(float(n), 1.0), radicalInverseVdC(i));
}

mat3 buildTbn(vec3 n)
{
    vec3 up = (abs(n.y) < 0.999) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);
    return mat3(t, b, n);
}

float TraceAoRay(vec3 origin, vec3 dir, float tMax)
{
    rayQueryEXT rq;
    rayQueryInitializeEXT(
        rq,
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF,
        origin,
        0.001,
        dir,
        tMax);
    while (rayQueryProceedEXT(rq)) { }
    return (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.0;
}

float ComputeShadowVisibility(vec3 worldPos, vec3 geomNormal, vec3 lightPos)
{
    vec3 toLight = lightPos - worldPos;
    float lightDistance = length(toLight);
    if (lightDistance <= 0.0001) {
        return 1.0;
    }

    vec3 rayDir = toLight / lightDistance;
    float bias = max(0.002, 0.0005 * lightDistance);
    vec3 rayOrigin = worldPos + geomNormal * bias;
    float tMin = 0.001;
    float tMax = max(lightDistance - bias, tMin);

    rayQueryEXT rq;
    rayQueryInitializeEXT(
        rq,
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF,
        rayOrigin,
        tMin,
        rayDir,
        tMax);

    while (rayQueryProceedEXT(rq)) { }
    return (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) ? 0.0 : 1.0;
}

// 单条射线：方向 L 是否被遮挡
float TraceShadowRayDirectional(vec3 rayOrigin, vec3 rayDir)
{
    float tMin = 0.001;
    float tMax = 500.0;
    rayQueryEXT rq;
    rayQueryInitializeEXT(
        rq,
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
        0xFF,
        rayOrigin,
        tMin,
        rayDir,
        tMax);
    while (rayQueryProceedEXT(rq)) { }
    return (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) ? 0.0 : 1.0;
}

float ComputeShadowVisibilityDirectional(vec3 worldPos, vec3 geomNormal, vec3 L)
{
    float dirLen2 = dot(L, L);
    if (dirLen2 < 1e-8) {
        return 1.0;
    }
    vec3 rayDir = normalize(L);
    float bias = 0.002;
    vec3 rayOrigin = worldPos + geomNormal * bias;

    float sunAngularRadius = ubo.directionalLightParams.x;
    int sampleCount = int(ubo.directionalLightParams.y + 0.5);
    sampleCount = clamp(sampleCount, 1, 16);

    // 硬阴影：单射线
    if (sunAngularRadius <= 0.0 || sampleCount <= 1) {
        float vis = TraceShadowRayDirectional(rayOrigin, rayDir);
        return vis > 0.5 ? 1.0 : 0.2;
    }

    // 软阴影：在太阳盘面内多采样
    vec3 T, B;
    vec3 up = (abs(rayDir.y) < 0.999) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 1.0);
    T = normalize(cross(up, rayDir));
    B = cross(rayDir, T);

    // 8点泊松盘 (在单位圆内)
    const vec2 poissonDisk[16] = vec2[](
        vec2(-0.613, -0.358), vec2( 0.582, -0.396), vec2(-0.451,  0.552), vec2( 0.208,  0.623),
        vec2(-0.263, -0.826), vec2(-0.883,  0.217), vec2( 0.759,  0.274), vec2( 0.352, -0.721),
        vec2(-0.713, -0.045), vec2( 0.039, -0.956), vec2(-0.189,  0.291), vec2( 0.595,  0.645),
        vec2(-0.515,  0.718), vec2( 0.941, -0.135), vec2(-0.137, -0.535), vec2( 0.169,  0.892)
    );

    float visSum = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        vec2 uv = poissonDisk[i];
        vec3 offset = sunAngularRadius * (uv.x * T + uv.y * B);
        vec3 sampleDir = normalize(rayDir + offset);
        visSum += TraceShadowRayDirectional(rayOrigin, sampleDir);
    }
    float vis = visSum / float(sampleCount);
    return mix(0.0, 1.0, vis);
}

const float RAY_REFLECTION_EPSILON = 0.001;
const float RAY_REFLECTION_TMAX = 1e4;
const uint RAY_REFLECTION_MAX_TEXTURES = 256u;

// 由 hit 的 instanceID、primIndex、重心坐标插值 UV（教程 Task 10）
vec2 intersection_uv(uint instanceID, uint primIndex, vec2 barycentrics) {
    uint indexOffset = instanceLUT.entries[nonuniformEXT(instanceID)].indexBufferOffset;
    uint i0 = indexBuffer.indices[indexOffset + primIndex * 3u + 0u];
    uint i1 = indexBuffer.indices[indexOffset + primIndex * 3u + 1u];
    uint i2 = indexBuffer.indices[indexOffset + primIndex * 3u + 2u];
    vec2 uv0 = uvBuffer.uvs[i0];
    vec2 uv1 = uvBuffer.uvs[i1];
    vec2 uv2 = uvBuffer.uvs[i2];
    float w0 = 1.0 - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;
    return w0 * uv0 + w1 * uv1 + w2 * uv2;
}

// 光追反射：沿镜面方向发射射线，采样命中三角形颜色（教程 Task 11 完整实现）
void apply_reflection(vec3 P, vec3 N, inout vec4 baseColor) {
    vec3 V = normalize(ubo.camPos.xyz - P);
    vec3 R = reflect(-V, N);
    vec3 rayOrigin = P + N * RAY_REFLECTION_EPSILON;

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT, 0xFF, rayOrigin, RAY_REFLECTION_EPSILON, R, RAY_REFLECTION_TMAX);

    // Proceed 循环：支持透明度，找到最近不透明命中（教程 Task 10）
    while (rayQueryProceedEXT(rq)) {
        uint instID = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false));  // meshIndex
        uint primIdx = uint(rayQueryGetIntersectionPrimitiveIndexEXT(rq, false));
        vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
        vec2 uv = intersection_uv(instID, primIdx, bary);
        uint matID = min(instanceLUT.entries[nonuniformEXT(instID)].materialID, RAY_REFLECTION_MAX_TEXTURES - 1u);
        vec4 hitColor = textureLod(baseColorTextures[nonuniformEXT(matID)], uv, 0.0);
        if (hitColor.a < 0.5) {
            // 透明：继续遍历
        } else {
            rayQueryConfirmIntersectionEXT(rq);
        }
    }

    bool hit = (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT);
    if (hit) {
        uint instID = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));  // meshIndex
        uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        vec2 uv = intersection_uv(instID, primIdx, bary);
        uint matID = min(instanceLUT.entries[nonuniformEXT(instID)].materialID, RAY_REFLECTION_MAX_TEXTURES - 1u);
        vec4 intersectionColor = textureLod(baseColorTextures[nonuniformEXT(matID)], uv, 0.0);
        baseColor.rgb = mix(baseColor.rgb, intersectionColor.rgb, 0.7);
    } else {
        vec3 skyColor = vec3(0.03, 0.05, 0.08);
        baseColor.rgb = mix(baseColor.rgb, skyColor, 0.5);
    }
}

void main()
{
    vec4 baseColorTex = texture(baseColorMap, inTexCoord);
    vec4 baseColor = baseColorTex * pc.baseColorFactor;

    int alphaMode = int(pc.materialParams1.y + 0.5);
    alphaMode = clamp(alphaMode, 0, 2);
    if (alphaMode == 1) { // Mask
        if (baseColor.a < pc.materialParams0.z) {
            discard;
        }
        // Mask is treated as fully opaque after the cutoff.
        baseColor.a = 1.0;
    }

    vec4 mrTex = texture(metallicRoughnessMap, inTexCoord);
    float metallic = clamp(mrTex.b * pc.materialParams0.x, 0.0, 1.0);
    float roughness = clamp(mrTex.g * pc.materialParams0.y, 0.04, 1.0);

    vec3 Ng = safeNormalize(inNormal);
    // Double-sided materials are rendered with cullMode=None. Flip the geometric normal (and tangent frame)
    // for backfaces, otherwise N·L clamps to 0 and the backface becomes unnaturally black.
    vec4 tangent = inTangent;
    if (!gl_FrontFacing) {
        Ng = -Ng;
        tangent.xyz = -tangent.xyz;
    }
    // 光追反射：在应用阴影前混合镜面反射采样（教程 Task 11）
    if (pc.materialParams1.z > 0.0) {
        apply_reflection(inWorldPos, Ng, baseColor);
    }
    // Start from geometric normal. Only apply normal map when tangent is valid.
    vec3 N = Ng;
    float tangentLen2 = dot(tangent.xyz, tangent.xyz);
    if (tangentLen2 > 1e-8) {
        vec3 T = normalize(tangent.xyz);
        vec3 B = cross(Ng, T) * tangent.w;
        float bitangentLen2 = dot(B, B);
        if (bitangentLen2 > 1e-8) {
            B = normalize(B);
            mat3 TBN = mat3(T, B, Ng);
            vec3 nTex = texture(normalMap, inTexCoord).xyz * 2.0 - 1.0;
            nTex.xy *= pc.materialParams0.w;
            // Specular AA (Toksvig-like): increase roughness when normal map gets mip-filtered.
            // This reduces sparkling on high-frequency surfaces (leaves, bark, gravel) during motion.
            float nLen = clamp(length(nTex), 0.0, 1.0);
            float normalVar = 1.0 - nLen;
            roughness = clamp(sqrt(roughness * roughness + normalVar * normalVar), 0.04, 1.0);
            N = normalize(TBN * nTex);
        }
    }
    vec3 V = safeNormalize(ubo.camPos.xyz - inWorldPos);

    vec3 F0 = mix(vec3(0.04), baseColor.rgb, metallic);

    vec3 Lo = vec3(0.0);

    // 1. Directional light (no attenuation, ray-traced shadow)
    if (ubo.directionalLightDir.w > 0.5) {
        vec3 lightDir = safeNormalize(ubo.directionalLightDir.xyz);
        vec3 L = -lightDir;  // direction points toward light, L is from surface to light
        vec3 lightColor = ubo.directionalLightColor.rgb;
        vec3 H = safeNormalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float D = DistributionGGX(NdotH, roughness);
        float G = GeometrySmith(NdotV, NdotL, roughness);
        vec3  F = FresnelSchlick(HdotV, F0);

        vec3 numerator = D * G * F;
        float denom = max(4.0 * NdotV * NdotL, 0.0001);
        vec3 specular = numerator / denom;

        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
        vec3 diffuse = kD * baseColor.rgb / PI;

        float visibility = ComputeShadowVisibilityDirectional(inWorldPos, Ng, L);
        Lo += (diffuse + specular) * lightColor * NdotL * visibility;
    }

    // 2. Point lights (3个，有衰减和阴影)
    int pointLightCount = int(ubo.params.w + 0.5);
    pointLightCount = clamp(pointLightCount, 0, 3);
    for (int i = 0; i < pointLightCount; ++i) {
        vec3 lightPos = ubo.lightPositions[i].xyz;
        vec3 lightColor = ubo.lightColors[i].rgb;

        vec3 L = safeNormalize(lightPos - inWorldPos);
        vec3 H = safeNormalize(V + L);

        float distance = length(lightPos - inWorldPos);
        const float pointLightRadiusScale = 2.5;  // 越大则衰减越快，影响范围越小
        float distScaled = distance * pointLightRadiusScale;
        float attenuation = 1.0 / max(distScaled * distScaled, 0.0001);
        vec3 radiance = lightColor * attenuation;

        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float D = DistributionGGX(NdotH, roughness);
        float G = GeometrySmith(NdotV, NdotL, roughness);
        vec3  F = FresnelSchlick(HdotV, F0);

        vec3 numerator = D * G * F;
        float denom = max(4.0 * NdotV * NdotL, 0.0001);
        vec3 specular = numerator / denom;

        vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
        vec3 diffuse = kD * baseColor.rgb / PI;

        float visibility = ComputeShadowVisibility(inWorldPos, Ng, lightPos);
        Lo += (diffuse + specular) * radiance * NdotL * visibility;
    }

    // Exposure is applied in postprocess tonemap.
    float aoTexture = (ubo.iblParams.z > 0.5) ? texture(occlusionMap, inTexCoord).r : 1.0;
    float occStrength = clamp(pc.materialParams1.x, 0.0, 1.0);
    aoTexture = mix(1.0, aoTexture, occStrength);

    float rtao = 1.0;
    if (ubo.rtaoParams0.x > 0.5 && alphaMode != 2) {
        // Use integer pixel fetch to avoid MSAA sample-position jitter causing edge halos/offset.
        rtao = texelFetch(rtaoFull, ivec2(gl_FragCoord.xy), 0).r;
    }
    float rtaoStrength = clamp(ubo.rtaoParams1.x, 0.0, 1.0);
    float ao = aoTexture * mix(1.0, rtao, rtaoStrength);

    float NdotV = max(dot(N, V), 0.0);
    vec3 R = reflect(-V, N);
    vec3 F = FresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    // Diffuse IBL: use geometric normal to avoid high-frequency normal-map speckling in indirect light.
    // Force base mip: irradiance cubemap is generated without a mip chain, and implicit LOD can
    // produce driver-dependent artifacts (stripes/seams) when derivatives cross cube face edges.
    vec3 irradiance = textureLod(irradianceMap, Ng, 0.0).rgb * ubo.iblParams.x;
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefilteredColor * (F * brdf.x + brdf.y) * ubo.iblParams.y;
    vec3 diffuseIBL = kD * irradiance * baseColor.rgb;

    // Approximate specular occlusion:
    // keeps reflections less visible in cavities while avoiding over-darkening on smooth surfaces.
    float specularOcclusion = clamp(pow(NdotV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
    vec3 ambient = diffuseIBL * ao + specularIBL * specularOcclusion;
    vec3 emissive = texture(emissiveMap, inTexCoord).rgb * pc.emissiveFactor.rgb;
    vec3 color = ambient + Lo + emissive;

    // Debug: output raw rtaoFull (used for compute debug outputs)
    if (ubo.iblParams.w > 8.5) {
        outColor = vec4(vec3(clamp(rtao, 0.0, 1.0)), 1.0);
        return;
    }
    // Debug: output AO (white=unoccluded, black=occluded)
    if (ubo.iblParams.w > 3.5) {
        outColor = vec4(vec3(ao), 1.0);
        return;
    }
    // Debug: output Ng (geometric normal, RGB=XYZ)
    if (ubo.iblParams.w > 2.5) {
        outColor = vec4(Ng * 0.5 + 0.5, 1.0);
        return;
    }
    // Debug: output baseColor (贴图 × baseColorFactor，排查黑块是否来自材质)
    if (ubo.iblParams.w > 1.5) {
        outColor = vec4(baseColor.rgb, 1.0);
        return;
    }
    // Debug: output max NdotL across all lights (white=lit, black=NdotL=0)
    if (ubo.iblParams.w > 0.5) {
        float maxNdotL = 0.0;
        if (ubo.directionalLightDir.w > 0.5) {
            vec3 Ld = -safeNormalize(ubo.directionalLightDir.xyz);
            maxNdotL = max(maxNdotL, max(dot(N, Ld), 0.0));
        }
        int plc = clamp(int(ubo.params.w + 0.5), 0, 3);
        for (int i = 0; i < plc; ++i) {
            vec3 Lp = safeNormalize(ubo.lightPositions[i].xyz - inWorldPos);
            maxNdotL = max(maxNdotL, max(dot(N, Lp), 0.0));
        }
        outColor = vec4(vec3(maxNdotL), 1.0);
        return;
    }

    // Output linear HDR here. Tone mapping/exposure is handled by TonemapBloomPass.
    // Keep debug views above unchanged.

    float outAlpha = baseColor.a;
    if (alphaMode == 0) { // Opaque
        outAlpha = 1.0;
    }
    // Premultiply only for BLEND (pipeline uses premultiplied alpha blending).
    if (alphaMode == 2) {
        color *= outAlpha;
    }
    outColor = vec4(color, outAlpha);
}

