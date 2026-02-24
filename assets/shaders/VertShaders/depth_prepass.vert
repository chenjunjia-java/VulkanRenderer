#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in vec4 inTangent;
layout(location = 0) out vec3 outWorldNormal;
layout(location = 1) out float outLinearViewZ;
layout(location = 2) out vec2 outTexCoord;

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
    vec4 iblParams;
    vec4 rtaoParams0;
    vec4 rtaoParams1;
    mat4 prevViewProj;
} ubo;

layout(binding = 11, std430) readonly buffer DrawDataBuf {
    mat4 models[];
} drawData;

layout(push_constant) uniform PBRPushConstants {
    mat4 model;
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 materialParams0;
    vec4 materialParams1;
} pc;

void main()
{
    mat4 modelMat = drawData.models[gl_BaseInstance];
    vec4 worldPos = modelMat * vec4(inPosition, 1.0);
    vec4 viewPos = ubo.view * worldPos;
    mat3 normalMat = transpose(inverse(mat3(modelMat)));
    outWorldNormal = normalize(normalMat * inNormal);
    outLinearViewZ = -viewPos.z;
    outTexCoord = inTexCoord;
    gl_Position = ubo.proj * viewPos;
}

