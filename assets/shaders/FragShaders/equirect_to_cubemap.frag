#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 localPos;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D equirectMap;

const vec2 invAtan = vec2(0.1591, 0.3183);

vec2 SampleSphericalMap(vec3 v)
{
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    // stb_image / Vulkan: image row 0 = top (sky). asin(+Y)=π/2 -> uv.y=1 maps to image bottom.
    // Flip V so sphere +Y (sky) samples image top.
    uv.y = 1.0 - uv.y;
    return uv;
}

void main()
{
    vec2 uv = SampleSphericalMap(normalize(localPos));
    vec3 color = texture(equirectMap, uv).rgb;
    outColor = vec4(color, 1.0);
}
