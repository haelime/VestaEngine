#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

// Miss shader returns procedural sky and marks the payload as a miss.

struct PathPayload {
    vec4 radianceDistance;
    vec4 albedoMetallic;
    vec4 normalRoughness;
    vec4 emissiveHit;
};

layout(location = 0) rayPayloadInEXT PathPayload payload;

layout(set = 0, binding = 0) uniform sampler2D sampledImages[];

layout(push_constant) uniform PathTracePushConstants {
    mat4 inverseViewProjection;
    vec4 cameraPositionAndFrame;
    vec4 lightDirectionAndIntensity;
    vec4 environmentParams;
    vec4 cameraRightAperture;
    vec4 cameraUpFocalDistance;
    uint triangleBufferIndex;
    uint triangleCount;
    uint frameIndex;
    uint emissiveTriangleBufferIndex;
    uint emissiveTriangleCount;
    uint russianRouletteDepth;
    float fireflyClamp;
    uint pathTraceFlags;
    uint reserved0;
    uvec4 accumulationImageIndices0;
    uvec4 accumulationImageIndices1;
    uvec4 pathTraceParams;
    uvec4 guideImageIndices;
} pc;

const float PI = 3.14159265359;
const uint kInvalidResourceIndex = 0xFFFFFFFFu;

vec3 sampleEnvironmentMap(uint imageIndex, vec3 direction)
{
    float u = atan(direction.z, direction.x) / (2.0 * PI) + 0.5;
    float v = acos(clamp(direction.y, -1.0, 1.0)) / PI;
    return texture(sampledImages[nonuniformEXT(int(imageIndex))], vec2(u, v)).rgb * pc.environmentParams.x;
}

vec3 sampleSky(vec3 direction)
{
    float c = cos(pc.environmentParams.y);
    float s = sin(pc.environmentParams.y);
    direction = normalize(vec3(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z));
    if (pc.reserved0 != kInvalidResourceIndex) {
        return sampleEnvironmentMap(pc.reserved0, direction);
    }
    float skyT = 0.5 * (direction.y + 1.0);
    uint preset = uint(clamp(pc.environmentParams.z + 0.5, 0.0, 3.0));
    vec3 horizon = vec3(0.08, 0.11, 0.16);
    vec3 zenith = vec3(0.40, 0.55, 0.82);
    vec3 sunColor = vec3(1.35, 0.92, 0.45);
    if (preset == 1u) {
        horizon = vec3(0.20, 0.18, 0.16);
        zenith = vec3(0.78, 0.62, 0.38);
        sunColor = vec3(1.85, 0.88, 0.38);
    } else if (preset == 2u) {
        horizon = vec3(0.025, 0.032, 0.048);
        zenith = vec3(0.12, 0.17, 0.26);
        sunColor = vec3(0.48, 0.66, 1.15);
    } else if (preset == 3u) {
        horizon = vec3(0.12, 0.14, 0.13);
        zenith = vec3(0.36, 0.44, 0.38);
        sunColor = vec3(0.74, 0.86, 0.68);
    }
    vec3 sky = mix(horizon, zenith, skyT);
    float sunPower = preset == 2u ? 96.0 : 64.0;
    float sun = pow(max(dot(direction, normalize(vec3(0.55, 0.25, 0.80))), 0.0), sunPower);
    return (sky + sun * sunColor) * pc.environmentParams.x;
}

void main() {
    payload.radianceDistance = vec4(sampleSky(gl_WorldRayDirectionEXT), 0.0);
    payload.albedoMetallic = vec4(0.0);
    payload.normalRoughness = vec4(0.0, 1.0, 0.0, 1.0);
    payload.emissiveHit = vec4(0.0);
}
