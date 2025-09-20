#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

// Closest-hit shader only resolves hit material data. Raygen evaluates lighting
// and chooses the next bounce so the path loop is centralized.

struct Triangle {
    vec4 p0;
    vec4 p1;
    vec4 p2;
    vec4 n0;
    vec4 n1;
    vec4 n2;
    vec4 uv0;
    vec4 uv1;
    vec4 uv2;
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 materialParams;
    uvec4 textureIndices0;
    uvec4 textureIndices1;
};

struct PathPayload {
    vec4 radianceDistance;
    vec4 albedoMetallic;
    vec4 normalRoughness;
    vec4 emissiveHit;
};

layout(set = 0, binding = 0) uniform sampler2D sampledImages[];
layout(set = 0, binding = 2, std430) readonly buffer TriangleBuffer {
    Triangle triangles[];
} triangleBuffers[];

layout(location = 0) rayPayloadInEXT PathPayload payload;
hitAttributeEXT vec2 hitAttributes;

layout(push_constant) uniform PathTracePushConstants {
    mat4 inverseViewProjection;
    vec4 cameraPositionAndFrame;
    vec4 lightDirectionAndIntensity;
    vec4 environmentParams;
    uint triangleBufferIndex;
    uint triangleCount;
    uint frameIndex;
    uint emissiveTriangleBufferIndex;
    uint emissiveTriangleCount;
    uint reserved0;
    uint reserved1;
    uvec4 accumulationImageIndices0;
    uvec4 accumulationImageIndices1;
    uvec4 pathTraceParams;
    uvec4 guideImageIndices;
} pc;

const uint kInvalidResourceIndex = 0xFFFFFFFFu;

vec4 sampleOptional(uint textureIndex, vec2 uv, vec4 fallback)
{
    if (textureIndex == kInvalidResourceIndex) {
        return fallback;
    }
    return fallback * texture(sampledImages[nonuniformEXT(int(textureIndex))], uv);
}

void main() {
    Triangle tri = triangleBuffers[nonuniformEXT(int(pc.triangleBufferIndex))].triangles[gl_PrimitiveID];
    float u = hitAttributes.x;
    float v = hitAttributes.y;
    float w = 1.0 - u - v;

    vec2 uv = tri.uv0.xy * w + tri.uv1.xy * u + tri.uv2.xy * v;
    vec3 normal = normalize(tri.n0.xyz * w + tri.n1.xyz * u + tri.n2.xyz * v);
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0) {
        normal = -normal;
    }

    vec4 baseColorSample = sampleOptional(tri.textureIndices0.x, uv, tri.baseColorFactor);
    vec4 metallicRoughnessSample = tri.textureIndices0.y != kInvalidResourceIndex
        ? texture(sampledImages[nonuniformEXT(int(tri.textureIndices0.y))], uv)
        : vec4(1.0);
    vec4 emissiveSample = sampleOptional(tri.textureIndices1.x, uv, tri.emissiveFactor);

    float metallic = clamp(tri.materialParams.x * metallicRoughnessSample.b, 0.0, 1.0);
    float roughness = clamp(tri.materialParams.y * metallicRoughnessSample.g, 0.045, 1.0);

    payload.radianceDistance = vec4(0.0, 0.0, 0.0, gl_HitTEXT);
    payload.albedoMetallic = vec4(baseColorSample.rgb, metallic);
    payload.normalRoughness = vec4(normal, roughness);
    payload.emissiveHit = vec4(emissiveSample.rgb, 1.0);
}
