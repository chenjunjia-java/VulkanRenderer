#version 460
#extension GL_ARB_separate_shader_objects : require

layout(location = 0) in vec3 inWorldNormal;
layout(location = 1) in float inLinearViewZ;
layout(location = 2) in vec2 inTexCoord;
layout(location = 0) out vec4 outNormal;
layout(location = 1) out float outLinearDepth;

layout(binding = 1) uniform sampler2D baseColorMap;

layout(push_constant) uniform PBRPushConstants {
    mat4 model;
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 materialParams0; // z = alphaCutoff
    vec4 materialParams1; // y = alphaMode(0=Opaque,1=Mask,2=Blend)
} pc;

void main()
{
    int alphaMode = int(pc.materialParams1.y + 0.5);
        if (alphaMode == 1) { // Mask
            float alphaCutoff = pc.materialParams0.z;
            float a = texture(baseColorMap, inTexCoord).a * pc.baseColorFactor.a;
            if (a < alphaCutoff) {
                discard;
            }
        }

    vec3 n = normalize(inWorldNormal);
    n = n * 0.5 + 0.5;
    outNormal = vec4(n, 1.0);
    outLinearDepth = max(inLinearViewZ, 0.0);
}

