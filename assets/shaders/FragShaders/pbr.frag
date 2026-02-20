#version 460
#extension GL_ARB_separate_shader_objects : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_ray_tracing : require

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout(binding = 0) uniform PBRUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightPositions[4];
    vec4 lightColors[4];
    vec4 camPos;
    vec4 params; // x=exposure, y=gamma, z=ambientStrength, w=lightCount
} ubo;

layout(binding = 1) uniform sampler2D baseColorMap;
layout(binding = 2) uniform accelerationStructureEXT topLevelAS;
layout(binding = 3) readonly buffer ShadowVertexUvBuffer {
    vec2 uv[];
} shadowVertexUvs;
layout(binding = 4) readonly buffer ShadowIndexBuffer {
    uint indexData[];
} shadowIndices;

layout(push_constant) uniform PBRMaterialPushConstants {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    float pad1;
} material;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

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

bool IsCandidateOpaqueForShadow(rayQueryEXT rq)
{
    const uint primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
    const uint i0 = shadowIndices.indexData[primitiveIndex * 3 + 0];
    const uint i1 = shadowIndices.indexData[primitiveIndex * 3 + 1];
    const uint i2 = shadowIndices.indexData[primitiveIndex * 3 + 2];

    vec2 uv0 = shadowVertexUvs.uv[i0];
    vec2 uv1 = shadowVertexUvs.uv[i1];
    vec2 uv2 = shadowVertexUvs.uv[i2];

    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
    float w = 1.0 - bary.x - bary.y;
    vec2 hitUv = uv0 * w + uv1 * bary.x + uv2 * bary.y;

    float alpha = texture(baseColorMap, hitUv).a * material.baseColorFactor.a;
    return alpha >= material.alphaCutoff;
}

vec2 GetCandidateHitUv(rayQueryEXT rq)
{
    const uint primitiveIndex = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
    const uint i0 = shadowIndices.indexData[primitiveIndex * 3 + 0];
    const uint i1 = shadowIndices.indexData[primitiveIndex * 3 + 1];
    const uint i2 = shadowIndices.indexData[primitiveIndex * 3 + 2];

    vec2 uv0 = shadowVertexUvs.uv[i0];
    vec2 uv1 = shadowVertexUvs.uv[i1];
    vec2 uv2 = shadowVertexUvs.uv[i2];

    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
    float w = 1.0 - bary.x - bary.y;
    return uv0 * w + uv1 * bary.x + uv2 * bary.y;
}

vec3 ComputeOneBounceReflection(vec3 worldPos, vec3 normal, vec3 viewDir, float roughness)
{
    vec3 rayDir = normalize(reflect(-viewDir, normal));
    vec3 rayOrigin = worldPos + normal * 0.02;
    float tMin = 0.001;
    float tMax = 100.0;

    rayQueryEXT rq;
    rayQueryInitializeEXT(
        rq,
        topLevelAS,
        gl_RayFlagsNoneEXT,
        0xFF,
        rayOrigin,
        tMin,
        rayDir,
        tMax);

    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) != gl_RayQueryCandidateIntersectionTriangleEXT) {
            continue;
        }

        vec2 hitUv = GetCandidateHitUv(rq);
        vec4 reflectedTex = texture(baseColorMap, hitUv) * material.baseColorFactor;
        if (reflectedTex.a >= material.alphaCutoff) {
            vec3 skyColor = vec3(0.03, 0.05, 0.08);
            float roughFade = 1.0 - roughness;
            return mix(skyColor, reflectedTex.rgb, roughFade);
        }
    }

    // Fallback environment term when reflection ray misses scene geometry.
    return vec3(0.03, 0.05, 0.08);
}

float ComputeShadowVisibility(vec3 worldPos, vec3 normal, vec3 lightPos)
{
    vec3 toLight = lightPos - worldPos;
    float lightDistance = length(toLight);
    if (lightDistance <= 0.0001) {
        return 1.0;
    }

    vec3 rayDir = toLight / lightDistance;
    vec3 rayOrigin = worldPos + normal * 0.01;
    float tMin = 0.001;
    float tMax = max(lightDistance - 0.01, tMin);

    rayQueryEXT rq;
    rayQueryInitializeEXT(
        rq,
        topLevelAS,
        gl_RayFlagsNoneEXT,
        0xFF,
        rayOrigin,
        tMin,
        rayDir,
        tMax);

    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) != gl_RayQueryCandidateIntersectionTriangleEXT) {
            continue;
        }

        if (IsCandidateOpaqueForShadow(rq)) {
            rayQueryTerminateEXT(rq);
            return 0.0;
        }
    }
    return 1.0;
}

void main()
{
    vec4 texColor = texture(baseColorMap, inTexCoord);
    vec4 baseColor = texColor * material.baseColorFactor;
    if (baseColor.a < material.alphaCutoff) {
        discard;
    }

    float metallic = clamp(material.metallicFactor, 0.0, 1.0);
    float roughness = clamp(material.roughnessFactor, 0.04, 1.0);

    vec3 N = normalize(inNormal);
    vec3 V = normalize(ubo.camPos.xyz - inWorldPos);

    vec3 F0 = mix(vec3(0.04), baseColor.rgb, metallic);

    int lightCount = int(ubo.params.w + 0.5);
    lightCount = clamp(lightCount, 0, 4);

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < lightCount; ++i) {
        vec3 lightPos = ubo.lightPositions[i].xyz;
        vec3 lightColor = ubo.lightColors[i].rgb;

        vec3 L = normalize(lightPos - inWorldPos);
        vec3 H = normalize(V + L);

        float distance = length(lightPos - inWorldPos);
        float attenuation = 1.0 / max(distance * distance, 0.0001);
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

        float visibility = ComputeShadowVisibility(inWorldPos, N, lightPos);
        Lo += (diffuse + specular) * radiance * NdotL * visibility;
    }

    float exposure = max(ubo.params.x, 0.0001);
    float gamma = max(ubo.params.y, 0.0001);
    float ambientStrength = clamp(ubo.params.z, 0.0, 1.0);

    vec3 ambient = ambientStrength * baseColor.rgb;
    float NdotV = max(dot(N, V), 0.0);
    vec3 fresnel = FresnelSchlick(NdotV, F0);
    vec3 reflectionColor = ComputeOneBounceReflection(inWorldPos, N, V, roughness);
    vec3 reflection = reflectionColor * fresnel * (1.0 - roughness);
    vec3 color = ambient + Lo + reflection;

    // Simple exposure-based tone mapping (LDR output).
    color = vec3(1.0) - exp(-color * exposure);
    color = pow(color, vec3(1.0 / gamma));

    outColor = vec4(color, baseColor.a);
}

