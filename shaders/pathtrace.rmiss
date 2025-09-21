#version 460
#extension GL_EXT_ray_tracing : require

// Miss shader returns procedural sky and marks the payload as a miss.

struct PathPayload {
    vec4 radianceDistance;
    vec4 albedoMetallic;
    vec4 normalRoughness;
    vec4 emissiveHit;
};

layout(location = 0) rayPayloadInEXT PathPayload payload;

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
    uint reserved0;
    uint reserved1;
    uvec4 accumulationImageIndices0;
    uvec4 accumulationImageIndices1;
    uvec4 pathTraceParams;
    uvec4 guideImageIndices;
} pc;

vec3 sampleSky(vec3 direction)
{
    float c = cos(pc.environmentParams.y);
    float s = sin(pc.environmentParams.y);
    direction = normalize(vec3(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z));
    float skyT = 0.5 * (direction.y + 1.0);
    vec3 sky = mix(vec3(0.08, 0.11, 0.16), vec3(0.40, 0.55, 0.82), skyT);
    float sun = pow(max(dot(direction, normalize(vec3(0.55, 0.25, 0.80))), 0.0), 64.0);
    return (sky + sun * vec3(1.35, 0.92, 0.45)) * pc.environmentParams.x;
}

void main() {
    payload.radianceDistance = vec4(sampleSky(gl_WorldRayDirectionEXT), 0.0);
    payload.albedoMetallic = vec4(0.0);
    payload.normalRoughness = vec4(0.0, 1.0, 0.0, 1.0);
    payload.emissiveHit = vec4(0.0);
}
