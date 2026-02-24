#version 450

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;

layout(push_constant) uniform PushConstants {
    vec4 params0; // x=threshold, y=softKnee, z=intensity, w=exposure
    vec4 params1; // x=debugView
} pc;

vec3 acesFitted(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 sceneHdr = texture(sceneColorTex, inUv).rgb;
    vec3 bloom = texture(bloomTex, inUv).rgb;

    float exposure = max(pc.params0.w, 0.0001);
    int debugView = int(pc.params1.x + 0.5);

    if (debugView == 1) {
        vec3 x = sceneHdr * exposure;
        outColor = vec4(x / (vec3(1.0) + x), 1.0);
        return;
    }
    if (debugView == 2) {
        vec3 x = bloom * exposure;
        outColor = vec4(x / (vec3(1.0) + x), 1.0);
        return;
    }

    vec3 hdr = (sceneHdr + bloom * pc.params0.z) * exposure;
    vec3 ldr = acesFitted(hdr);
    outColor = vec4(ldr, 1.0);
}

