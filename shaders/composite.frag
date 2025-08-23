#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// Picks which intermediate image to show, or blends several together for the
// portfolio view shown in Composite mode.

layout(rgba16f, set = 0, binding = 1) uniform readonly image2D storageImages[];

layout(push_constant) uniform CompositePushConstants {
    uvec4 imageIndices0;
    uvec4 imageIndices1;
    vec4 params;
} pc;

layout(location = 0) out vec4 outColor;

const uint INVALID_IMAGE_INDEX = 0xffffffffu;

vec3 srgb_to_linear(vec3 value)
{
    bvec3 low = lessThanEqual(value, vec3(0.04045));
    vec3 lowPart = value / 12.92;
    vec3 highPart = pow(max((value + 0.055) / 1.055, vec3(0.0)), vec3(2.4));
    return mix(highPart, lowPart, low);
}

vec3 tonemap(vec3 value) {
    const vec3 a = vec3(2.51);
    const vec3 b = vec3(0.03);
    const vec3 c = vec3(2.43);
    const vec3 d = vec3(0.59);
    const vec3 e = vec3(0.14);
    return clamp((value * (a * value + b)) / (value * (c * value + d) + e), 0.0, 1.0);
}

bool hasImage(uint index) {
    return index != INVALID_IMAGE_INDEX;
}

vec4 resolveGaussian(ivec2 pixel, vec2 uv)
{
    if (!hasImage(pc.imageIndices0.z) || !hasImage(pc.imageIndices0.w)) {
        return vec4(0.0);
    }

    ivec2 accumSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.z))]);
    ivec2 revealSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.w))]);
    ivec2 accumPixel = clamp(ivec2(uv * vec2(accumSize)), ivec2(0), accumSize - ivec2(1));
    ivec2 revealPixel = clamp(ivec2(uv * vec2(revealSize)), ivec2(0), revealSize - ivec2(1));

    vec4 accum = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.z))], accumPixel);
    vec4 reveal = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.w))], revealPixel);
    float alpha = clamp(1.0 - reveal.r, 0.0, 1.0);
    return vec4(accum.rgb, alpha);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 baseSize = ivec2(1);
    if (hasImage(pc.imageIndices0.x)) {
        baseSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.x))]);
    } else if (hasImage(pc.imageIndices0.y)) {
        baseSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.y))]);
    } else if (hasImage(pc.imageIndices0.z)) {
        baseSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.z))]);
    }

    vec2 uv = (vec2(pixel) + 0.5) / vec2(baseSize);

    vec3 deferredColor = vec3(0.0);
    if (hasImage(pc.imageIndices0.x)) {
        ivec2 deferredSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.x))]);
        ivec2 deferredPixel = clamp(pixel, ivec2(0), deferredSize - ivec2(1));
        deferredColor = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.x))], deferredPixel).rgb;
    }

    vec3 pathTraceColor = vec3(0.0);
    if (hasImage(pc.imageIndices0.y)) {
        ivec2 pathTraceSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.y))]);
        ivec2 pathTracePixel = clamp(ivec2(uv * vec2(pathTraceSize)), ivec2(0), pathTraceSize - ivec2(1));
        pathTraceColor = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.y))], pathTracePixel).rgb;
    }

    vec4 gaussianColor = resolveGaussian(pixel, uv);

    vec3 composite = deferredColor;
    if (pc.imageIndices1.x == 1u) {
        composite = deferredColor;
    } else if (pc.imageIndices1.x == 2u) {
        composite = gaussianColor.rgb;
    } else if (pc.imageIndices1.x == 3u) {
        composite = pathTraceColor;
    } else {
        vec3 base = mix(deferredColor, pathTraceColor, 0.35);
        vec3 gaussianLinear = srgb_to_linear(clamp(gaussianColor.rgb, vec3(0.0), vec3(1.0)));
        float gaussianWeight = gaussianColor.a * pc.params.x;
        composite = base * (1.0 - gaussianWeight) + gaussianLinear * pc.params.x;
    }

    if (pc.imageIndices1.x != 2u) {
        composite = tonemap(composite);
        composite = pow(composite, vec3(1.0 / 2.2));
    }
    outColor = vec4(composite, 1.0);
}
