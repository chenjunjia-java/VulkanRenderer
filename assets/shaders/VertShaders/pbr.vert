#version 450
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;      // kept for legacy vertex format (unused here)
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexCoord;

layout(binding = 0) uniform PBRUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 lightPositions[4];
    vec4 lightColors[4];
    vec4 camPos;
    vec4 params; // x=exposure, y=gamma, z=ambientStrength, w=lightCount
} ubo;

void main()
{
    vec4 worldPos = ubo.model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;

    outWorldPos = worldPos.xyz;

    // Model matrix is rotation-only in this project; mat3(model) is sufficient here.
    mat3 normalMat = mat3(ubo.model);
    outNormal = normalize(normalMat * inNormal);

    outTexCoord = inTexCoord;
}

