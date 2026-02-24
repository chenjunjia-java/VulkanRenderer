#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;      // kept for legacy vertex format (unused here)
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec4 inTangent;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;
layout(location = 3) out vec4 outTangent;

layout(binding = 0) uniform PBRUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 directionalLightDir;
    vec4 directionalLightColor;
    vec4 directionalLightParams;
    vec4 lightPositions[3];
    vec4 lightColors[3];
    vec4 camPos;
    vec4 params;
    vec4 iblParams; // x=enableDiffuseIBL, y=enableSpecularIBL, z=enableAO
} ubo;

layout(binding = 11, std430) readonly buffer DrawDataBuf {
    mat4 models[];
} drawData;

layout(push_constant) uniform PBRPushConstants {
    mat4 model;  // used only for transparent per-draw fallback when drawData not used
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 materialParams0; // x metallicFactor, y roughnessFactor, z alphaCutoff, w normalScale
    vec4 materialParams1; // x occlusionStrength, y alphaMode, z/w unused
} pc;

void main()
{
    // Indirect draw: firstInstance=drawId, use drawData.models[gl_BaseInstance]. Per-draw transparent: same.
    mat4 modelMat = drawData.models[gl_BaseInstance];
    vec4 worldPos = modelMat * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    outWorldPos = worldPos.xyz;

    // Correct normal transform even under node scaling (glTF often has scale).
    mat3 normalMat = transpose(inverse(mat3(modelMat)));
    outNormal = normalize(normalMat * inNormal);

    outTexCoord = inTexCoord;

    // Tangent: transform xyz, preserve handness (w).
    vec3 tangentRot = normalMat * inTangent.xyz;
    float tangentLen = length(tangentRot);
    outTangent = vec4(tangentLen > 1e-6 ? normalize(tangentRot) : vec3(0.0), inTangent.w);
}

