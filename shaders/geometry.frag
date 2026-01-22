#version 460

// Stores material color and a packed normal so later passes can light the scene
// without re-running vertex processing.

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormalDepth;

void main() {
    vec3 packedNormal = normalize(inNormal) * 0.5 + 0.5;
    outAlbedo = inColor;
    outNormalDepth = vec4(packedNormal, gl_FragCoord.z);
}
