#version 460

layout(push_constant) uniform GaussianPushConstants {
    mat4 viewProjection;
    vec4 params;
} pc;

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 centered = gl_PointCoord * 2.0 - 1.0;
    float falloff = exp(-dot(centered, centered) * 2.8);
    outColor = vec4(inColor.rgb, pc.params.y * falloff);
}
