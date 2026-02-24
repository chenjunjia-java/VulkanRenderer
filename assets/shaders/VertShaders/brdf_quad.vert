#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;

layout(location = 0) out vec2 outUV;

void main()
{
    outUV = inUV;
    gl_Position = vec4(inPos, 0.0, 1.0);
}
