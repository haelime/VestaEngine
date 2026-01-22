#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// Picks which intermediate image to show, or blends several together for the
// portfolio view shown in Composite mode.

layout(rgba16f, set = 0, binding = 1) uniform readonly image2D storageImages[];

layout(push_constant) uniform CompositePushConstants {
    uint deferredImageIndex;
    uint pathTraceImageIndex;
    uint gaussianImageIndex;
    uint mode;
    vec4 params;
} pc;

layout(location = 0) out vec4 outColor;

vec3 tonemap(vec3 value) {
    return value / (value + vec3(1.0));
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 deferredSize = imageSize(storageImages[nonuniformEXT(int(pc.deferredImageIndex))]);
    ivec2 pathTraceSize = imageSize(storageImages[nonuniformEXT(int(pc.pathTraceImageIndex))]);
    ivec2 gaussianSize = imageSize(storageImages[nonuniformEXT(int(pc.gaussianImageIndex))]);
    vec2 uv = (vec2(pixel) + 0.5) / vec2(deferredSize);

    ivec2 pathTracePixel = clamp(ivec2(uv * vec2(pathTraceSize)), ivec2(0), pathTraceSize - ivec2(1));
    ivec2 gaussianPixel = clamp(ivec2(uv * vec2(gaussianSize)), ivec2(0), gaussianSize - ivec2(1));

    vec3 deferredColor = imageLoad(storageImages[nonuniformEXT(int(pc.deferredImageIndex))], clamp(pixel, ivec2(0), deferredSize - ivec2(1))).rgb;
    vec3 pathTraceColor = imageLoad(storageImages[nonuniformEXT(int(pc.pathTraceImageIndex))], pathTracePixel).rgb;
    vec4 gaussianColor = imageLoad(storageImages[nonuniformEXT(int(pc.gaussianImageIndex))], gaussianPixel);

    vec3 composite = deferredColor;
    if (pc.mode == 1u) {
        composite = deferredColor;
    } else if (pc.mode == 2u) {
        composite = gaussianColor.rgb;
    } else if (pc.mode == 3u) {
        composite = pathTraceColor;
    } else {
        composite = mix(deferredColor, pathTraceColor, 0.35) + gaussianColor.rgb * gaussianColor.a * pc.params.x;
    }

    composite = tonemap(composite);
    composite = pow(composite, vec3(1.0 / 2.2));
    outColor = vec4(composite, 1.0);
}
