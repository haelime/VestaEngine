#version 460
#extension GL_EXT_ray_tracing : require

// Miss shader returns a simple sky gradient when no triangle was hit.

layout(location = 0) rayPayloadInEXT vec3 payloadColor;

void main() {
    float skyT = 0.5 * (gl_WorldRayDirectionEXT.y + 1.0);
    payloadColor = mix(vec3(0.08, 0.11, 0.16), vec3(0.40, 0.55, 0.82), skyT);
}
