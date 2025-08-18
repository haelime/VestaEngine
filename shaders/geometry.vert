#version 460

// Geometry pass vertex shader: forward world-space vertex data into the GBuffer.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec4 inColor;
layout(location = 4) in vec2 inTexCoord;
layout(location = 5) in uint inMaterialIndex;

layout(push_constant) uniform GeometryPushConstants {
    mat4 viewProjection;
    uint materialBufferIndex;
} pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outTangent;
layout(location = 2) out vec4 outColor;
layout(location = 3) out vec2 outTexCoord;
layout(location = 4) flat out uint outMaterialIndex;

void main() {
    gl_Position = pc.viewProjection * vec4(inPosition, 1.0);
    outNormal = normalize(inNormal);
    outTangent = inTangent;
    outColor = inColor;
    outTexCoord = inTexCoord;
    outMaterialIndex = inMaterialIndex;
}
