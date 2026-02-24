#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 outTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform samplerCube envMap;

void main()
{
    // Force LOD0: prevents driver-dependent artifacts when sampling cubemaps that have only 1 mip level
    // (e.g. irradiance/debug cubemaps) while the bound sampler uses mipmap filtering.
    vec3 envColor = textureLod(envMap, normalize(outTexCoord), 0.0).rgb;
    // Exposure tone mapping (more stable for very bright HDR skies)
    const float exposure = 0.35;
    envColor = vec3(1.0) - exp(-envColor * exposure);
    envColor = pow(envColor, vec3(1.0 / 2.2));
    outColor = vec4(envColor, 1.0);
}
