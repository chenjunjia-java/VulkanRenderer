#version 450

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D bloomInputTex;

layout(push_constant) uniform PushConstants {
    vec4 params0; // w=blurRadius
    vec4 params1; // x=invWidth, y=invHeight, z=dirX, w=dirY
} pc;

void main()
{
    // Important: keep sampling step = 1 texel to avoid sparse-sampling stripes when radius increases.
    // Interpret blurRadius as a kernel radius (in texels) and derive a Gaussian sigma from it.
    float radius = max(pc.params0.w, 0.0);
    vec2 texel = vec2(pc.params1.x, pc.params1.y);

    vec2 dirRaw = vec2(pc.params1.z, pc.params1.w);
    vec2 dir = (dot(dirRaw, dirRaw) > 0.25) ? normalize(dirRaw) : vec2(1.0, 0.0);

    // Cap to keep shader cost bounded. Increase if you need very wide bloom.
    int r = int(clamp(floor(radius + 0.5), 0.0, 64.0));
    if (r <= 0) {
        outColor = vec4(texture(bloomInputTex, inUv).rgb, 1.0);
        return;
    }

    // Derive sigma from the effective radius (after clamping) so widening radius increases spread.
    float sigma = max(float(r) * 0.5, 0.001);
    float invTwoSigma2 = 0.5 / (sigma * sigma);

    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = -r; i <= r; ++i) {
        float x = float(i);
        float w = exp(-(x * x) * invTwoSigma2);
        sum += texture(bloomInputTex, inUv + dir * texel * x).rgb * w;
        wsum += w;
    }

    sum /= max(wsum, 1e-6);
    outColor = vec4(sum, 1.0);
}

