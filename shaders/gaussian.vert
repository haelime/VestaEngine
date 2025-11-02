#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec4 inColor;

layout(push_constant) uniform GaussianPushConstants {
    mat4 viewProjection;
    vec4 params;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    gl_Position = pc.viewProjection * vec4(inPosition, 1.0);
    gl_PointSize = max(pc.params.x, 1.0);
    outColor = inColor;
}
