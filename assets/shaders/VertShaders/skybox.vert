#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outTexCoord;

// Same layout as PBR UBO at offset 64: view, proj (model skipped)
layout(binding = 0) uniform SkyboxUniforms {
    mat4 view;
    mat4 proj;
} ubo;

void main()
{
    outTexCoord = inPosition;
    mat4 rotView = mat4(mat3(ubo.view));
    vec4 clipPos = ubo.proj * rotView * vec4(inPosition, 1.0);
    gl_Position = clipPos.xyww;  // z=w so depth is always 1.0
}
