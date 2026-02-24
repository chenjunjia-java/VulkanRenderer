#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 aPos;

layout(location = 0) out vec3 localPos;

layout(binding = 0) uniform CaptureUniforms {
    mat4 projection;
    float roughness;
} ubo;

layout(push_constant) uniform PushView {
    mat4 view;
} pc;

void main()
{
    localPos = aPos;
    gl_Position = ubo.projection * pc.view * vec4(localPos, 1.0);
}
