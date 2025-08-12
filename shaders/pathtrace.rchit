#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

// Closest-hit shader shades the hit triangle using the same simple lighting
// idea as the compute fallback so both backends stay visually comparable.

struct Triangle {
    vec4 p0;
    vec4 p1;
    vec4 p2;
    vec4 albedo;
};

layout(set = 0, binding = 2, std430) readonly buffer TriangleBuffer {
    Triangle triangles[];
} triangleBuffers[];

layout(location = 0) rayPayloadInEXT vec3 payloadColor;

layout(push_constant) uniform PathTracePushConstants {
    mat4 inverseViewProjection;
    vec4 cameraPositionAndFrame;
    uint triangleBufferIndex;
    uint triangleCount;
    uint frameIndex;
    uint reserved;
} pc;

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 sampleHemisphere(vec3 normal, vec2 seed) {
    float z = rand(seed);
    float angle = 6.2831853 * rand(seed.yx + 0.37);
    float r = sqrt(max(0.0, 1.0 - z * z));
    vec3 tangent = normalize(abs(normal.y) > 0.99 ? cross(normal, vec3(1.0, 0.0, 0.0)) : cross(normal, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * cos(angle) * r + bitangent * sin(angle) * r + normal * z);
}

void main() {
    Triangle tri = triangleBuffers[nonuniformEXT(int(pc.triangleBufferIndex))].triangles[gl_PrimitiveID];
    vec3 normal = normalize(cross(tri.p1.xyz - tri.p0.xyz, tri.p2.xyz - tri.p0.xyz));
    vec3 hitAlbedo = tri.albedo.rgb;
    vec3 lightDir = normalize(vec3(-0.35, -1.0, -0.25));
    float direct = max(dot(normal, -lightDir), 0.0);

    vec2 seed = vec2(gl_PrimitiveID, pc.cameraPositionAndFrame.w + gl_HitTEXT);
    vec3 bounceDir = sampleHemisphere(normal, seed);
    float bounceSky = 0.5 * (bounceDir.y + 1.0);
    vec3 bounceColor = mix(vec3(0.04, 0.05, 0.07), vec3(0.28, 0.35, 0.46), bounceSky);

    payloadColor = hitAlbedo * (0.05 + direct * 0.95) + hitAlbedo * bounceColor * 0.25;
}
