#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// Samples glTF material textures and writes a small PBR-ready GBuffer.

struct Material {
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 materialParams;
    uvec4 textureIndices0;
    uvec4 textureIndices1;
};

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inTangent;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) flat in uint inMaterialIndex;

layout(set = 0, binding = 0) uniform sampler2D sampledImages[];
layout(set = 0, binding = 2, std430) readonly buffer MaterialBuffer {
    Material materials[];
} materialBuffers[];

layout(push_constant) uniform GeometryPushConstants {
    mat4 viewProjection;
    uint materialBufferIndex;
} pc;

layout(location = 0) out vec4 outAlbedoAo;
layout(location = 1) out vec4 outNormalRoughness;
layout(location = 2) out vec4 outEmissiveMetallic;

const uint kInvalidResourceIndex = 0xFFFFFFFFu;

vec3 sampleNormalMap(Material material, vec3 baseNormal, vec4 tangent)
{
    if (material.textureIndices0.z == kInvalidResourceIndex) {
        return normalize(baseNormal);
    }

    vec3 tangentAxis = tangent.xyz;
    if (length(tangentAxis) < 1.0e-4) {
        return normalize(baseNormal);
    }

    tangentAxis = normalize(tangentAxis);
    vec3 normalAxis = normalize(baseNormal);
    vec3 bitangentAxis = normalize(cross(normalAxis, tangentAxis)) * tangent.w;
    vec3 tangentNormal = texture(sampledImages[nonuniformEXT(material.textureIndices0.z)], inTexCoord).xyz * 2.0 - 1.0;
    tangentNormal.xy *= material.materialParams.w;
    tangentNormal.z = sqrt(max(1.0 - dot(tangentNormal.xy, tangentNormal.xy), 0.0));
    return normalize(mat3(tangentAxis, bitangentAxis, normalAxis) * tangentNormal);
}

void main() {
    Material material = materialBuffers[nonuniformEXT(int(pc.materialBufferIndex))].materials[inMaterialIndex];

    vec4 baseColor = material.baseColorFactor * inColor;
    if (material.textureIndices0.x != kInvalidResourceIndex) {
        baseColor *= texture(sampledImages[nonuniformEXT(material.textureIndices0.x)], inTexCoord);
    }

    float metallic = clamp(material.materialParams.x, 0.0, 1.0);
    float roughness = clamp(material.materialParams.y, 0.045, 1.0);
    if (material.textureIndices0.y != kInvalidResourceIndex) {
        vec4 metallicRoughness = texture(sampledImages[nonuniformEXT(material.textureIndices0.y)], inTexCoord);
        metallic *= metallicRoughness.b;
        roughness *= metallicRoughness.g;
    }

    float ao = 1.0;
    if (material.textureIndices0.w != kInvalidResourceIndex) {
        float aoSample = texture(sampledImages[nonuniformEXT(material.textureIndices0.w)], inTexCoord).r;
        ao = mix(1.0, aoSample, clamp(material.materialParams.z, 0.0, 1.0));
    }

    vec3 emissive = material.emissiveFactor.rgb;
    if (material.textureIndices1.x != kInvalidResourceIndex) {
        emissive *= texture(sampledImages[nonuniformEXT(material.textureIndices1.x)], inTexCoord).rgb;
    }

    vec3 normal = sampleNormalMap(material, inNormal, inTangent);
    outAlbedoAo = vec4(baseColor.rgb, ao);
    outNormalRoughness = vec4(normal * 0.5 + 0.5, roughness);
    outEmissiveMetallic = vec4(emissive, metallic);
}
