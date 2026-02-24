#version 460
#extension GL_ARB_separate_shader_objects : require

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
} ubo;

layout(push_constant) uniform PBRPushConstants {
    mat4 model;
} pc;

const vec3 kCubeTriangles[36] = vec3[](
    // -Z
    vec3(-1, -1, -1), vec3( 1, -1, -1), vec3( 1,  1, -1),
    vec3(-1, -1, -1), vec3( 1,  1, -1), vec3(-1,  1, -1),
    // +Z
    vec3(-1, -1,  1), vec3( 1,  1,  1), vec3( 1, -1,  1),
    vec3(-1, -1,  1), vec3(-1,  1,  1), vec3( 1,  1,  1),
    // -X
    vec3(-1, -1, -1), vec3(-1,  1, -1), vec3(-1,  1,  1),
    vec3(-1, -1, -1), vec3(-1,  1,  1), vec3(-1, -1,  1),
    // +X
    vec3( 1, -1, -1), vec3( 1, -1,  1), vec3( 1,  1,  1),
    vec3( 1, -1, -1), vec3( 1,  1,  1), vec3( 1,  1, -1),
    // -Y
    vec3(-1, -1, -1), vec3(-1, -1,  1), vec3( 1, -1,  1),
    vec3(-1, -1, -1), vec3( 1, -1,  1), vec3( 1, -1, -1),
    // +Y
    vec3(-1,  1, -1), vec3( 1,  1,  1), vec3(-1,  1,  1),
    vec3(-1,  1, -1), vec3( 1,  1, -1), vec3( 1,  1,  1)
);

void main()
{
    vec3 pos = kCubeTriangles[gl_VertexIndex];
    gl_Position = ubo.proj * ubo.view * pc.model * vec4(pos, 1.0);
}

