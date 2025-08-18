#version 460

// Shapes the point sprite into a soft radial falloff instead of a hard square.

layout(push_constant) uniform GaussianPushConstants {
    mat4 viewProjection;
    vec4 params;
} pc;

layout(location = 0) in vec4 inColor;
layout(location = 1) in float inOpacity;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 centered = gl_PointCoord * 2.0 - 1.0;
    float radial = dot(centered, centered);
    if (radial > 1.0) {
        discard;
    }

    float falloff = exp(-radial * 5.5);
    if (falloff < 0.035) {
        discard;
    }

    outColor = vec4(inColor.rgb, inOpacity * pc.params.y * falloff);
}
