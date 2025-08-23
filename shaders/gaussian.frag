#version 460

layout(push_constant) uniform GaussianPushConstants {
    mat4 viewMatrix;
    mat4 viewProjection;
    vec4 cameraPositionAndSceneType;
    vec4 params0;
    vec4 params1;
    uvec4 bufferIndices;
    uvec4 options;
} pc;

layout(location = 0) in vec4 inColor;
layout(location = 1) in float inOpacity;
layout(location = 2) in vec2 inLocalCoord;
layout(location = 3) in float inViewDepth;
layout(location = 0) out vec4 outAccum;
layout(location = 1) out vec4 outReveal;

void main()
{
    float radial = dot(inLocalCoord, inLocalCoord);
    if (radial > 1.0) {
        discard;
    }

    float sharpness = pc.cameraPositionAndSceneType.w > 0.5 ? 4.5 : 5.5;
    if (pc.options.x == 0u) {
        sharpness *= 1.15;
    }
    float falloff = exp(-radial * sharpness);
    float alpha = clamp(inOpacity * pc.params0.y * falloff, 0.0, 0.995);
    if (alpha < pc.params1.x) {
        discard;
    }

    float depthWeight = pc.cameraPositionAndSceneType.w > 0.5
        ? clamp(10.0 / (0.25 + inViewDepth), 0.35, 10.0)
        : 1.0;
    float weight = alpha * depthWeight * depthWeight;
    outAccum = vec4(inColor.rgb * weight, weight);
    outReveal = vec4(alpha);
}
