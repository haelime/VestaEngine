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

const uint INVALID_IMAGE_INDEX = 0xffffffffu;

vec3 tonemap(vec3 value) {
    return value / (value + vec3(1.0));
}

bool hasImage(uint index) {
    return index != INVALID_IMAGE_INDEX;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 baseSize = ivec2(1);
    if (hasImage(pc.deferredImageIndex)) {
        baseSize = imageSize(storageImages[nonuniformEXT(int(pc.deferredImageIndex))]);
    } else if (hasImage(pc.pathTraceImageIndex)) {
        baseSize = imageSize(storageImages[nonuniformEXT(int(pc.pathTraceImageIndex))]);
    } else if (hasImage(pc.gaussianImageIndex)) {
        baseSize = imageSize(storageImages[nonuniformEXT(int(pc.gaussianImageIndex))]);
    }

    vec2 uv = (vec2(pixel) + 0.5) / vec2(baseSize);

    vec3 deferredColor = vec3(0.0);
    if (hasImage(pc.deferredImageIndex)) {
        ivec2 deferredSize = imageSize(storageImages[nonuniformEXT(int(pc.deferredImageIndex))]);
        ivec2 deferredPixel = clamp(pixel, ivec2(0), deferredSize - ivec2(1));
        deferredColor = imageLoad(storageImages[nonuniformEXT(int(pc.deferredImageIndex))], deferredPixel).rgb;
    }

    vec3 pathTraceColor = vec3(0.0);
    if (hasImage(pc.pathTraceImageIndex)) {
        ivec2 pathTraceSize = imageSize(storageImages[nonuniformEXT(int(pc.pathTraceImageIndex))]);
        ivec2 pathTracePixel = clamp(ivec2(uv * vec2(pathTraceSize)), ivec2(0), pathTraceSize - ivec2(1));
        pathTraceColor = imageLoad(storageImages[nonuniformEXT(int(pc.pathTraceImageIndex))], pathTracePixel).rgb;
    }

    vec4 gaussianColor = vec4(0.0);
    if (hasImage(pc.gaussianImageIndex)) {
        ivec2 gaussianSize = imageSize(storageImages[nonuniformEXT(int(pc.gaussianImageIndex))]);
        ivec2 gaussianPixel = clamp(ivec2(uv * vec2(gaussianSize)), ivec2(0), gaussianSize - ivec2(1));
        gaussianColor = imageLoad(storageImages[nonuniformEXT(int(pc.gaussianImageIndex))], gaussianPixel);
    }

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
