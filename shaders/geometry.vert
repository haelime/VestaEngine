#version 460

// Geometry pass vertex shader: forward world-space vertex data into the GBuffer.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(push_constant) uniform GeometryPushConstants {
    mat4 viewProjection;
} pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;

void main() {
    gl_Position = pc.viewProjection * vec4(inPosition, 1.0);
    outNormal = normalize(inNormal);
    outColor = inColor;
}
