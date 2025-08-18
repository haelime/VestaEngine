#version 460

// Expands each input vertex into a screen-space point sprite.

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inSplatParams;

layout(push_constant) uniform GaussianPushConstants {
    mat4 viewProjection;
    vec4 params;
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outOpacity;

void main() {
    gl_Position = pc.viewProjection * vec4(inPosition, 1.0);
    gl_PointSize = clamp(max(pc.params.x * max(inSplatParams.x, 0.01), 1.0), 1.0, 96.0);
    outColor = inColor;
    outOpacity = max(inSplatParams.y, 0.01);
}
