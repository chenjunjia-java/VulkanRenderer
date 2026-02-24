#version 450

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneColorTex;

layout(push_constant) uniform PushConstants {
    vec4 params0; // x=threshold, y=softKnee, z=intensity
    vec4 params1; // unused
} pc;

float maxComponent(vec3 v)
{
    return max(v.x, max(v.y, v.z));
}

void main()
{
    vec3 hdr = texture(sceneColorTex, inUv).rgb;
    float brightness = maxComponent(hdr);
    float threshold = pc.params0.x;
    float softKnee = max(pc.params0.y, 1e-5);
    float knee = threshold * softKnee;
    float soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee + 1e-5);
    float contribution = max(brightness - threshold, soft) / max(brightness, 1e-5);
    outColor = vec4(hdr * contribution, 1.0);
}

