#version 460
#extension GL_EXT_nonuniform_qualifier : enable

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
    vec3 deferredColor = imageLoad(storageImages[nonuniformEXT(int(pc.deferredImageIndex))], pixel).rgb;
    vec3 pathTraceColor = imageLoad(storageImages[nonuniformEXT(int(pc.pathTraceImageIndex))], pixel).rgb;
    vec4 gaussianColor = imageLoad(storageImages[nonuniformEXT(int(pc.gaussianImageIndex))], pixel);

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
