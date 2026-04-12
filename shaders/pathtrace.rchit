#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

// Closest-hit shader evaluates the same metallic-roughness data as the compute path.

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

layout(set = 0, binding = 0) uniform sampler2D sampledImages[];
layout(set = 0, binding = 2, std430) readonly buffer TriangleBuffer {
    Triangle triangles[];
} triangleBuffers[];

layout(location = 0) rayPayloadInEXT vec3 payloadColor;
hitAttributeEXT vec2 hitAttributes;

layout(push_constant) uniform PathTracePushConstants {
    mat4 inverseViewProjection;
    vec4 cameraPositionAndFrame;
    vec4 lightDirectionAndIntensity;
    uint triangleBufferIndex;
    uint triangleCount;
    uint frameIndex;
    uint reserved;
} pc;

const uint kInvalidResourceIndex = 0xFFFFFFFFu;
const float PI = 3.14159265359;

vec3 sampleSky(vec3 direction)
{
    float skyT = 0.5 * (direction.y + 1.0);
    return mix(vec3(0.08, 0.11, 0.16), vec3(0.40, 0.55, 0.82), skyT);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denominator = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denominator * denominator, 1.0e-4);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 1.0e-4);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) * GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

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
    vec3 N = normalize(tri.n0.xyz * w + tri.n1.xyz * u + tri.n2.xyz * v);
    vec3 V = normalize(-gl_WorldRayDirectionEXT);
    vec3 L = normalize(-pc.lightDirectionAndIntensity.xyz);
    vec3 H = normalize(V + L);

    vec4 baseColorSample = sampleOptional(tri.textureIndices0.x, uv, tri.baseColorFactor);
    vec4 metallicRoughnessSample = tri.textureIndices0.y != kInvalidResourceIndex
        ? texture(sampledImages[nonuniformEXT(int(tri.textureIndices0.y))], uv)
        : vec4(1.0);
    vec4 emissiveSample = sampleOptional(tri.textureIndices1.x, uv, tri.emissiveFactor);

    vec3 baseColor = baseColorSample.rgb;
    float metallic = clamp(tri.materialParams.x * metallicRoughnessSample.b, 0.0, 1.0);
    float roughness = clamp(tri.materialParams.y * metallicRoughnessSample.g, 0.045, 1.0);
    float ao = tri.textureIndices0.w != kInvalidResourceIndex
        ? mix(1.0, texture(sampledImages[nonuniformEXT(int(tri.textureIndices0.w))], uv).r, tri.materialParams.z)
        : 1.0;
    vec3 emissive = emissiveSample.rgb;

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    float D = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1.0e-4);

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 diffuse = kD * baseColor / PI;
    vec3 direct = (diffuse + specular) * NdotL * pc.lightDirectionAndIntensity.w;

    vec3 ambientDiffuse = sampleSky(N) * baseColor * (1.0 - metallic) * 0.20 * ao;
    vec3 ambientSpecular = sampleSky(reflect(-V, N)) * FresnelSchlick(max(dot(N, V), 0.0), F0)
        * mix(0.08, 0.45, metallic) * (1.0 - roughness * 0.65);

    payloadColor = ambientDiffuse + ambientSpecular + direct + emissive;
}
