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

vec3 sampleSky(vec3 direction)
{
    float skyT = 0.5 * (direction.y + 1.0);
    return mix(vec3(0.08, 0.11, 0.16), vec3(0.40, 0.55, 0.82), skyT);
}

void main() {
    payload.radianceDistance = vec4(sampleSky(gl_WorldRayDirectionEXT), 0.0);
    payload.albedoMetallic = vec4(0.0);
    payload.normalRoughness = vec4(0.0, 1.0, 0.0, 1.0);
    payload.emissiveHit = vec4(0.0);
}
