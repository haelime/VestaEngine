#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// Picks which intermediate image to show, or blends several together for the
// portfolio view shown in Composite mode.

layout(rgba16f, set = 0, binding = 1) uniform readonly image2D storageImages[];
layout(set = 0, binding = 0) uniform sampler2D sampledImages[];
layout(set = 0, binding = 2, std430) readonly buffer StorageBufferUvec2 {
    uvec2 values[];
} storageBuffersUvec2[];

layout(push_constant) uniform CompositePushConstants {
    uvec4 imageIndices0;
    uvec4 imageIndices1;
    uvec4 imageIndices2;
    uvec4 imageIndices3;
    uvec4 imageIndices4;
    uvec4 gaussianDebug;
    vec4 params;
    vec4 compareParams;
    vec4 postParams;
    vec4 bloomParams;
    vec4 ssaoParams;
    vec4 motionBlurParams;
    mat4 inverseViewProjection;
} pc;

layout(location = 0) out vec4 outColor;

const uint INVALID_IMAGE_INDEX = 0xffffffffu;
const uint GAUSSIAN_TILE_SIZE = 8u;
const uint MODE_BLOOM_EXTRACT = 90u;
const uint MODE_BLOOM_DOWNSAMPLE = 91u;
const uint MODE_BLOOM_UPSAMPLE = 92u;

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

vec3 reinhardTonemap(vec3 value)
{
    return value / (vec3(1.0) + value);
}

bool hasImage(uint index) {
    return index != INVALID_IMAGE_INDEX;
}

ivec2 getOutputSize()
{
    return ivec2(max(pc.imageIndices4.z, 1u), max(pc.imageIndices4.w, 1u));
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

vec4 loadStorage(uint index, ivec2 pixel)
{
    if (!hasImage(index)) {
        return vec4(0.0);
    }
    ivec2 size = imageSize(storageImages[nonuniformEXT(int(index))]);
    ivec2 clampedPixel = clamp(pixel, ivec2(0), size - ivec2(1));
    return imageLoad(storageImages[nonuniformEXT(int(index))], clampedPixel);
}

vec3 reconstructWorldPosition(ivec2 pixel, ivec2 size, float depth)
{
    vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world = pc.inverseViewProjection * vec4(ndc, depth, 1.0);
    return world.xyz / max(world.w, 0.0001);
}

vec3 applyDisplayTransform(vec3 color);
vec3 applyPostProcess(vec3 color, vec2 uv);
vec3 computeBloom(vec2 uv);
vec3 applyFxaa(vec3 color, vec2 uv);
vec3 applyMotionBlur(vec3 color, vec2 uv);
vec3 heatmap(float value);

vec3 resolveShadowCascadeColor(ivec2 pixel)
{
    if (!hasImage(pc.imageIndices2.w)) {
        return vec3(-1.0);
    }

    ivec2 size = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
    ivec2 clampedPixel = clamp(pixel, ivec2(0), size - ivec2(1));
    float depth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], clampedPixel, 0).r;
    if (depth >= 0.99999) {
        return vec3(0.0);
    }

    float nearPlane = max(pc.params.z, 0.0001);
    float farPlane = max(pc.params.w, nearPlane + 0.0001);
    float linearDepth = (nearPlane * farPlane) / max(farPlane + depth * (nearPlane - farPlane), 0.0001);
    uint cascadeCount = uint(clamp(pc.ssaoParams.w, 1.0, 4.0));
    float lambda = clamp(pc.motionBlurParams.w, 0.0, 1.0);

    uint cascadeIndex = cascadeCount - 1u;
    for (uint i = 0u; i < 4u; ++i) {
        if (i >= cascadeCount) {
            break;
        }
        float p = float(i + 1u) / float(cascadeCount);
        float uniformSplit = nearPlane + (farPlane - nearPlane) * p;
        float logSplit = nearPlane * pow(farPlane / nearPlane, p);
        float splitDepth = mix(uniformSplit, logSplit, lambda);
        if (linearDepth <= splitDepth) {
            cascadeIndex = i;
            break;
        }
    }

    const vec3 cascadeColors[4] = vec3[](
        vec3(0.1, 0.85, 0.95),
        vec3(0.35, 0.95, 0.2),
        vec3(1.0, 0.82, 0.08),
        vec3(1.0, 0.18, 0.28));
    return cascadeColors[cascadeIndex];
}

float computeScreenSpaceAo(ivec2 pixel, ivec2 size, vec3 worldPosition, vec3 normal, float depth)
{
    if (pc.ssaoParams.x < 0.5 || depth >= 0.99999 || !hasImage(pc.imageIndices2.w)) {
        return 1.0;
    }

    const vec2 offsets[12] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(0.707, 0.707), vec2(-0.707, 0.707), vec2(0.707, -0.707), vec2(-0.707, -0.707),
        vec2(0.383, 0.924), vec2(-0.924, 0.383), vec2(0.924, -0.383), vec2(-0.383, -0.924));

    float viewDistance = max(length(worldPosition), 0.25);
    float radiusPixels = clamp((pc.ssaoParams.y * 95.0) / viewDistance, 2.0, 48.0);
    float occlusion = 0.0;
    float weightSum = 0.0;

    for (int sampleIndex = 0; sampleIndex < 12; ++sampleIndex) {
        vec2 scaledOffset = offsets[sampleIndex] * radiusPixels * (0.45 + 0.11 * float(sampleIndex % 4));
        ivec2 samplePixel = clamp(pixel + ivec2(round(scaledOffset)), ivec2(0), size - ivec2(1));
        float sampleDepth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], samplePixel, 0).r;
        if (sampleDepth >= 0.99999) {
            continue;
        }

        vec3 sampleWorld = reconstructWorldPosition(samplePixel, size, sampleDepth);
        vec3 delta = sampleWorld - worldPosition;
        float distanceToSample = length(delta);
        if (distanceToSample <= 0.0001 || distanceToSample > pc.ssaoParams.y) {
            continue;
        }

        float hemisphereWeight = max(dot(normal, delta / distanceToSample), 0.0);
        float rangeWeight = smoothstep(pc.ssaoParams.y, 0.0, distanceToSample);
        float isOccluder = sampleDepth < depth - 0.0008 ? 1.0 : 0.0;
        float weight = mix(0.35, 1.0, hemisphereWeight) * rangeWeight;
        occlusion += isOccluder * weight;
        weightSum += weight;
    }

    if (weightSum <= 0.0) {
        return 1.0;
    }

    return clamp(1.0 - clamp(occlusion / weightSum, 0.0, 1.0) * pc.ssaoParams.z, 0.0, 1.0);
}

float resolveGBufferEdgeMask(ivec2 pixel)
{
    if (!hasImage(pc.imageIndices2.y) || !hasImage(pc.imageIndices2.w) || !hasImage(pc.imageIndices3.x)) {
        return -1.0;
    }

    ivec2 size = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
    ivec2 centerPixel = clamp(pixel, ivec2(0), size - ivec2(1));
    float centerDepth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], centerPixel, 0).r;
    vec3 centerNormal = normalize(loadStorage(pc.imageIndices2.y, centerPixel).xyz * 2.0 - 1.0);
    vec2 centerIds = loadStorage(pc.imageIndices3.x, centerPixel).ba;

    const ivec2 offsets[4] = ivec2[](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
    float edge = 0.0;
    for (int offsetIndex = 0; offsetIndex < 4; ++offsetIndex) {
        ivec2 samplePixel = clamp(centerPixel + offsets[offsetIndex], ivec2(0), size - ivec2(1));
        float sampleDepth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], samplePixel, 0).r;
        vec3 sampleNormal = normalize(loadStorage(pc.imageIndices2.y, samplePixel).xyz * 2.0 - 1.0);
        vec2 sampleIds = loadStorage(pc.imageIndices3.x, samplePixel).ba;

        float depthEdge = abs(centerDepth - sampleDepth) * 160.0;
        float normalEdge = (1.0 - clamp(dot(centerNormal, sampleNormal), 0.0, 1.0)) * 3.5;
        float idEdge = any(greaterThan(abs(centerIds - sampleIds), vec2(0.001))) ? 1.0 : 0.0;
        edge = max(edge, max(max(depthEdge, normalEdge), idEdge));
    }
    return smoothstep(0.035, 0.22, edge);
}

vec3 resolveDebugView(ivec2 pixel)
{
    uint debugView = pc.imageIndices1.y;
    if (debugView == 1u) {
        if (!hasImage(pc.imageIndices2.x)) { return vec3(-1.0); }
        return loadStorage(pc.imageIndices2.x, pixel).rgb;
    }
    if (debugView == 2u) {
        if (!hasImage(pc.imageIndices2.y)) { return vec3(-1.0); }
        return loadStorage(pc.imageIndices2.y, pixel).rgb;
    }
    if (debugView == 3u) {
        if (!hasImage(pc.imageIndices2.w)) { return vec3(-1.0); }
        ivec2 size = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
        ivec2 clampedPixel = clamp(pixel, ivec2(0), size - ivec2(1));
        float depth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], clampedPixel, 0).r;
        vec3 worldPosition = reconstructWorldPosition(clampedPixel, size, depth);
        return clamp(worldPosition * 0.05 + 0.5, vec3(0.0), vec3(1.0));
    }
    if (debugView == 4u) {
        if (!hasImage(pc.imageIndices2.w)) { return vec3(-1.0); }
        ivec2 size = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
        ivec2 clampedPixel = clamp(pixel, ivec2(0), size - ivec2(1));
        float depth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], clampedPixel, 0).r;
        float nearPlane = max(pc.params.z, 0.0001);
        float farPlane = max(pc.params.w, nearPlane + 0.0001);
        float linearDepth = (nearPlane * farPlane) / max(farPlane + depth * (nearPlane - farPlane), 0.0001);
        return vec3(clamp(linearDepth / farPlane, 0.0, 1.0));
    }
    if (debugView == 5u) {
        if (!hasImage(pc.imageIndices3.x)) { return vec3(-1.0); }
        return vec3(loadStorage(pc.imageIndices3.x, pixel).rg, 0.0);
    }
    if (debugView == 6u) {
        if (!hasImage(pc.imageIndices3.x)) { return vec3(-1.0); }
        float idHash = loadStorage(pc.imageIndices3.x, pixel).b;
        return fract(vec3(0.97, 0.57, 0.23) + idHash * vec3(1.0, 2.17, 3.31));
    }
    if (debugView == 7u) {
        if (!hasImage(pc.imageIndices3.x)) { return vec3(-1.0); }
        float idHash = loadStorage(pc.imageIndices3.x, pixel).a;
        return fract(vec3(0.19, 0.83, 0.41) + idHash * vec3(3.73, 1.61, 2.29));
    }
    if (debugView == 8u) {
        if (!hasImage(pc.imageIndices2.y)) { return vec3(-1.0); }
        vec4 normalRoughness = loadStorage(pc.imageIndices2.y, pixel);
        return vec3(normalRoughness.a);
    }
    if (debugView == 9u) {
        if (!hasImage(pc.imageIndices2.z)) { return vec3(-1.0); }
        vec4 material = loadStorage(pc.imageIndices2.z, pixel);
        return vec3(material.a);
    }
    if (debugView == 10u) {
        if (!hasImage(pc.imageIndices2.z)) { return vec3(-1.0); }
        return loadStorage(pc.imageIndices2.z, pixel).rgb;
    }
    if (debugView == 11u) {
        if (!hasImage(pc.imageIndices2.x) || !hasImage(pc.imageIndices2.y) || !hasImage(pc.imageIndices2.w)) { return vec3(-1.0); }
        ivec2 size = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
        ivec2 clampedPixel = clamp(pixel, ivec2(0), size - ivec2(1));
        float depth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], clampedPixel, 0).r;
        vec4 normalRoughness = loadStorage(pc.imageIndices2.y, clampedPixel);
        vec3 normal = normalize(normalRoughness.xyz * 2.0 - 1.0);
        vec3 worldPosition = reconstructWorldPosition(clampedPixel, size, depth);
        float materialAo = clamp(loadStorage(pc.imageIndices2.x, clampedPixel).a, 0.0, 1.0);
        return vec3(materialAo * computeScreenSpaceAo(clampedPixel, size, worldPosition, normal, depth));
    }
    if (debugView == 12u) {
        if (!hasImage(pc.imageIndices3.y)) { return vec3(-1.0); }
        vec2 motion = loadStorage(pc.imageIndices3.y, pixel).xy;
        return vec3(clamp(motion * 24.0 + 0.5, vec2(0.0), vec2(1.0)), clamp(length(motion) * 48.0, 0.0, 1.0));
    }
    if (debugView == 13u || debugView == 14u || debugView == 15u || debugView == 27u || debugView == 29u) {
        if (!hasImage(pc.imageIndices3.z)) { return vec3(-1.0); }
        return applyDisplayTransform(loadStorage(pc.imageIndices3.z, pixel).rgb);
    }
    if (debugView == 16u) {
        if (!hasImage(pc.imageIndices0.y)) { return vec3(-1.0); }
        ivec2 baseSize = getOutputSize();
        vec2 uv = (vec2(pixel) + 0.5) / vec2(baseSize);
        ivec2 pathTraceSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.y))]);
        ivec2 pathTracePixel = clamp(ivec2(uv * vec2(pathTraceSize)), ivec2(0), pathTraceSize - ivec2(1));
        return applyDisplayTransform(imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.y))], pathTracePixel).rgb);
    }
    if (debugView == 17u) {
        if (!hasImage(pc.imageIndices0.x) || !hasImage(pc.imageIndices0.y)) { return vec3(-1.0); }
        ivec2 outputSize = getOutputSize();
        vec2 uv = (vec2(pixel) + 0.5) / vec2(outputSize);
        ivec2 deferredSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.x))]);
        ivec2 deferredPixel = clamp(ivec2(uv * vec2(deferredSize)), ivec2(0), deferredSize - ivec2(1));
        ivec2 pathTraceSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.y))]);
        ivec2 pathTracePixel = clamp(ivec2(uv * vec2(pathTraceSize)), ivec2(0), pathTraceSize - ivec2(1));
        vec3 rasterDisplay = applyDisplayTransform(imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.x))], deferredPixel).rgb);
        vec3 pathDisplay = applyDisplayTransform(imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.y))], pathTracePixel).rgb);
        return heatmap(length(rasterDisplay - pathDisplay) * pc.compareParams.z);
    }
    if (debugView == 18u) {
        float edge = resolveGBufferEdgeMask(pixel);
        if (edge < 0.0) { return vec3(-1.0); }
        return mix(vec3(0.015), vec3(1.0), edge);
    }
    if (debugView == 19u) {
        if (!hasImage(pc.imageIndices3.x)) { return vec3(-1.0); }
        vec2 uv = loadStorage(pc.imageIndices3.x, pixel).rg;
        vec2 dx = dFdx(uv);
        vec2 dy = dFdy(uv);
        float footprint = max(length(dx), length(dy));
        float estimatedMip = log2(max(footprint * 2048.0, 1.0));
        return heatmap(clamp(estimatedMip / 10.0, 0.0, 1.0));
    }
    if (debugView == 20u) {
        if (!hasImage(pc.imageIndices4.x)) { return vec3(-1.0); }
        ivec2 shadowSize = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices4.x))], 0);
        ivec2 baseSize = ivec2(1);
        if (hasImage(pc.imageIndices0.x)) {
            baseSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.x))]);
        }
        vec2 uv = (vec2(pixel) + 0.5) / vec2(baseSize);
        ivec2 shadowPixel = clamp(ivec2(uv * vec2(shadowSize)), ivec2(0), shadowSize - ivec2(1));
        float depth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices4.x))], shadowPixel, 0).r;
        return vec3(1.0 - clamp(depth, 0.0, 1.0));
    }
    if (debugView == 28u) {
        return resolveShadowCascadeColor(pixel);
    }
    if (debugView == 21u) {
        if (!hasImage(pc.imageIndices4.y)) { return vec3(-1.0); }
        vec3 overdrawValue = loadStorage(pc.imageIndices4.y, pixel).rgb;
        return heatmap(clamp(overdrawValue.r, 0.0, 1.0));
    }
    if (debugView >= 22u && debugView <= 26u) {
        if (!hasImage(pc.imageIndices0.x)) { return vec3(-1.0); }
        return loadStorage(pc.imageIndices0.x, pixel).rgb;
    }
    return vec3(-1.0);
}

vec3 heatmap(float value)
{
    value = clamp(value, 0.0, 1.0);
    vec3 blue = vec3(0.05, 0.12, 0.85);
    vec3 cyan = vec3(0.0, 0.85, 0.95);
    vec3 yellow = vec3(1.0, 0.9, 0.05);
    vec3 red = vec3(1.0, 0.08, 0.02);
    if (value < 0.33) {
        return mix(blue, cyan, value / 0.33);
    }
    if (value < 0.66) {
        return mix(cyan, yellow, (value - 0.33) / 0.33);
    }
    return mix(yellow, red, (value - 0.66) / 0.34);
}

vec3 applyDisplayTransform(vec3 color)
{
    color *= exp2(pc.params.y);
    uint toneMappingMode = uint(pc.compareParams.w + 0.5);
    if (toneMappingMode == 1u) {
        color = reinhardTonemap(color);
    } else if (toneMappingMode == 2u) {
        color = tonemap(color);
    } else {
        color = clamp(color, vec3(0.0), vec3(1.0));
    }
    return pow(color, vec3(1.0 / 2.2));
}

vec3 applyPostProcess(vec3 color, vec2 uv)
{
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, clamp(pc.postParams.x, 0.0, 2.0));
    color = (color - vec3(0.5)) * clamp(pc.postParams.y, 0.25, 2.0) + vec3(0.5);
    if (pc.postParams.z > 0.5 && pc.postParams.w > 0.0) {
        vec2 centered = uv * 2.0 - 1.0;
        float vignette = smoothstep(1.15, 0.25, dot(centered, centered));
        color *= mix(1.0, vignette, clamp(pc.postParams.w, 0.0, 1.0));
    }
    return clamp(color, vec3(0.0), vec3(1.0));
}

vec3 sampleBloomSource(uint imageIndex, vec2 uv)
{
    if (!hasImage(imageIndex)) {
        return vec3(0.0);
    }
    ivec2 imageSizeValue = imageSize(storageImages[nonuniformEXT(int(imageIndex))]);
    ivec2 samplePixel = clamp(ivec2(uv * vec2(imageSizeValue)), ivec2(0), imageSizeValue - ivec2(1));
    return applyDisplayTransform(imageLoad(storageImages[nonuniformEXT(int(imageIndex))], samplePixel).rgb);
}

vec3 bloomExtract(vec3 color)
{
    float threshold = max(pc.bloomParams.x, 0.0);
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float knee = smoothstep(threshold * 0.72, max(threshold, 0.0001), luminance);
    return color * knee;
}

vec3 sampleBloomPyramid(uint imageIndex, vec2 uv)
{
    if (!hasImage(imageIndex)) {
        return vec3(0.0);
    }
    ivec2 imageSizeValue = imageSize(storageImages[nonuniformEXT(int(imageIndex))]);
    ivec2 samplePixel = clamp(ivec2(uv * vec2(imageSizeValue)), ivec2(0), imageSizeValue - ivec2(1));
    return imageLoad(storageImages[nonuniformEXT(int(imageIndex))], samplePixel).rgb;
}

vec3 filterBloomPyramid(uint imageIndex, vec2 uv, float radius)
{
    if (!hasImage(imageIndex)) {
        return vec3(0.0);
    }
    ivec2 sourceSize = imageSize(storageImages[nonuniformEXT(int(imageIndex))]);
    vec2 texel = radius / vec2(sourceSize);
    vec3 center = sampleBloomPyramid(imageIndex, uv) * 0.30;
    vec3 cross = vec3(0.0);
    cross += sampleBloomPyramid(imageIndex, clamp(uv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0)));
    cross += sampleBloomPyramid(imageIndex, clamp(uv - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0)));
    cross += sampleBloomPyramid(imageIndex, clamp(uv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0)));
    cross += sampleBloomPyramid(imageIndex, clamp(uv - vec2(0.0, texel.y), vec2(0.0), vec2(1.0)));
    vec3 diagonal = vec3(0.0);
    diagonal += sampleBloomPyramid(imageIndex, clamp(uv + texel, vec2(0.0), vec2(1.0)));
    diagonal += sampleBloomPyramid(imageIndex, clamp(uv + vec2(-texel.x, texel.y), vec2(0.0), vec2(1.0)));
    diagonal += sampleBloomPyramid(imageIndex, clamp(uv + vec2(texel.x, -texel.y), vec2(0.0), vec2(1.0)));
    diagonal += sampleBloomPyramid(imageIndex, clamp(uv - texel, vec2(0.0), vec2(1.0)));
    return center + cross * 0.12 + diagonal * 0.055;
}

vec3 runBloomPipelineStage(vec2 uv)
{
    uint stage = pc.imageIndices1.x;
    uint sourceImage = pc.imageIndices0.x;
    if (stage == MODE_BLOOM_EXTRACT) {
        vec3 source = vec3(0.0);
        source += sampleBloomSource(sourceImage, uv) * 0.40;
        ivec2 sourceSize = hasImage(sourceImage) ? imageSize(storageImages[nonuniformEXT(int(sourceImage))]) : ivec2(1);
        vec2 texel = 1.0 / vec2(sourceSize);
        source += sampleBloomSource(sourceImage, clamp(uv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))) * 0.15;
        source += sampleBloomSource(sourceImage, clamp(uv - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0))) * 0.15;
        source += sampleBloomSource(sourceImage, clamp(uv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0))) * 0.15;
        source += sampleBloomSource(sourceImage, clamp(uv - vec2(0.0, texel.y), vec2(0.0), vec2(1.0))) * 0.15;
        return bloomExtract(source);
    }
    if (stage == MODE_BLOOM_DOWNSAMPLE) {
        return filterBloomPyramid(sourceImage, uv, 1.4);
    }
    if (stage == MODE_BLOOM_UPSAMPLE) {
        vec3 halfLevel = filterBloomPyramid(sourceImage, uv, 1.0);
        vec3 quarterLevel = filterBloomPyramid(pc.imageIndices0.y, uv, 1.8);
        return halfLevel * 0.62 + quarterLevel * 0.70;
    }
    return vec3(0.0);
}

vec3 computeBloom(vec2 uv)
{
    if (pc.bloomParams.z < 0.5 || pc.bloomParams.y <= 0.0) {
        return vec3(0.0);
    }

    if (hasImage(pc.imageIndices4.y) && pc.imageIndices1.y != 21u) {
        return filterBloomPyramid(pc.imageIndices4.y, uv, 0.75) * pc.bloomParams.y;
    }

    uint sourceImage = hasImage(pc.imageIndices0.x) ? pc.imageIndices0.x : pc.imageIndices0.y;
    if (!hasImage(sourceImage)) {
        return vec3(0.0);
    }

    ivec2 sourceSize = imageSize(storageImages[nonuniformEXT(int(sourceImage))]);
    vec2 texel = 1.0 / vec2(sourceSize);
    const vec2 offsets[13] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(-1.0, -1.0),
        vec2(2.0, 0.0), vec2(-2.0, 0.0), vec2(0.0, 2.0), vec2(0.0, -2.0));
    const float weights[13] = float[](
        0.18,
        0.11, 0.11, 0.11, 0.11,
        0.07, 0.07, 0.07, 0.07,
        0.04, 0.04, 0.04, 0.04);

    vec3 bloom = vec3(0.0);
    for (int sampleIndex = 0; sampleIndex < 13; ++sampleIndex) {
        vec2 sampleUv = clamp(uv + offsets[sampleIndex] * texel * 2.0, vec2(0.0), vec2(1.0));
        bloom += bloomExtract(sampleBloomSource(sourceImage, sampleUv)) * weights[sampleIndex];
    }
    return bloom * pc.bloomParams.y;
}

vec3 applyFxaa(vec3 color, vec2 uv)
{
    if (pc.bloomParams.w < 0.5) {
        return color;
    }

    uint sourceImage = hasImage(pc.imageIndices0.x) ? pc.imageIndices0.x : pc.imageIndices0.y;
    if (!hasImage(sourceImage)) {
        return color;
    }

    ivec2 sourceSize = imageSize(storageImages[nonuniformEXT(int(sourceImage))]);
    vec2 texel = 1.0 / vec2(sourceSize);
    vec3 north = sampleBloomSource(sourceImage, clamp(uv + vec2(0.0, -texel.y), vec2(0.0), vec2(1.0)));
    vec3 south = sampleBloomSource(sourceImage, clamp(uv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0)));
    vec3 east = sampleBloomSource(sourceImage, clamp(uv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0)));
    vec3 west = sampleBloomSource(sourceImage, clamp(uv + vec2(-texel.x, 0.0), vec2(0.0), vec2(1.0)));

    vec3 lumaWeights = vec3(0.299, 0.587, 0.114);
    float centerLuma = dot(color, lumaWeights);
    float minLuma = min(centerLuma, min(min(dot(north, lumaWeights), dot(south, lumaWeights)), min(dot(east, lumaWeights), dot(west, lumaWeights))));
    float maxLuma = max(centerLuma, max(max(dot(north, lumaWeights), dot(south, lumaWeights)), max(dot(east, lumaWeights), dot(west, lumaWeights))));
    float edgeAmount = smoothstep(0.04, 0.22, maxLuma - minLuma);
    vec3 average = (north + south + east + west) * 0.25;
    return mix(color, average, edgeAmount * 0.42);
}

vec3 applyMotionBlur(vec3 color, vec2 uv)
{
    if (pc.motionBlurParams.x < 0.5 || pc.motionBlurParams.y <= 0.0) {
        return color;
    }
    if (!hasImage(pc.imageIndices0.x) || !hasImage(pc.imageIndices3.y)) {
        return color;
    }

    ivec2 motionSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices3.y))]);
    ivec2 motionPixel = clamp(ivec2(uv * vec2(motionSize)), ivec2(0), motionSize - ivec2(1));
    vec2 motion = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices3.y))], motionPixel).xy;
    float motionMagnitude = length(motion);
    if (motionMagnitude < 0.0002) {
        return color;
    }

    int sampleCount = int(clamp(pc.motionBlurParams.z, 3.0, 9.0));
    vec3 accumulated = color;
    float weightSum = 1.0;
    float strength = clamp(pc.motionBlurParams.y, 0.0, 2.0);
    for (int sampleIndex = 1; sampleIndex < 9; ++sampleIndex) {
        if (sampleIndex >= sampleCount) {
            break;
        }
        float t = float(sampleIndex) / float(sampleCount - 1);
        float centered = t - 0.5;
        float weight = 1.0 - abs(centered) * 1.7;
        vec2 sampleUv = clamp(uv - motion * centered * strength, vec2(0.0), vec2(1.0));
        accumulated += sampleBloomSource(pc.imageIndices0.x, sampleUv) * weight;
        weightSum += weight;
    }
    return accumulated / max(weightSum, 0.0001);
}

vec3 resolveGaussianDebugView(vec4 gaussianColor, ivec2 pixel, vec2 uv)
{
    uint gaussianDebugView = pc.imageIndices1.z;
    if (gaussianDebugView == 0u) {
        return vec3(-1.0);
    }
    if (gaussianDebugView == 1u) {
        return vec3(gaussianColor.a);
    }
    if (gaussianDebugView == 2u) {
        if (!hasImage(pc.imageIndices0.w)) { return vec3(-1.0); }
        ivec2 revealSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.w))]);
        ivec2 revealPixel = clamp(ivec2(uv * vec2(revealSize)), ivec2(0), revealSize - ivec2(1));
        float reveal = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.w))], revealPixel).r;
        return vec3(reveal);
    }
    if (gaussianDebugView == 3u) {
        float density = 1.0 - clamp(gaussianColor.a, 0.0, 1.0);
        return heatmap(pow(1.0 - density, 0.35));
    }
    if (gaussianDebugView == 4u) {
        if (!hasImage(pc.imageIndices1.w)) { return vec3(-1.0); }
        ivec2 debugSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices1.w))]);
        ivec2 debugPixel = clamp(ivec2(uv * vec2(debugSize)), ivec2(0), debugSize - ivec2(1));
        vec4 debugValue = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices1.w))], debugPixel);
        if (debugValue.y <= 0.0) {
            return vec3(0.0);
        }
        return heatmap(1.0 - clamp(debugValue.x, 0.0, 1.0));
    }
    if (gaussianDebugView == 5u) {
        if (pc.gaussianDebug.x == INVALID_IMAGE_INDEX || pc.gaussianDebug.y == 0u || pc.gaussianDebug.z == 0u) {
            return vec3(-1.0);
        }
        uvec2 tileCoord = min(uvec2(pixel) / GAUSSIAN_TILE_SIZE, uvec2(pc.gaussianDebug.y - 1u, pc.gaussianDebug.z - 1u));
        uint tileIndex = tileCoord.y * pc.gaussianDebug.y + tileCoord.x;
        uvec2 range = storageBuffersUvec2[nonuniformEXT(int(pc.gaussianDebug.x))].values[tileIndex];
        if (range.x == INVALID_IMAGE_INDEX || range.y == INVALID_IMAGE_INDEX || range.y <= range.x) {
            return vec3(0.02);
        }
        float occupancy = clamp(float(range.y - range.x) / 256.0, 0.0, 1.0);
        return heatmap(occupancy);
    }
    if (gaussianDebugView == 6u) {
        if (!hasImage(pc.imageIndices1.w)) { return vec3(-1.0); }
        ivec2 debugSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices1.w))]);
        ivec2 debugPixel = clamp(ivec2(uv * vec2(debugSize)), ivec2(0), debugSize - ivec2(1));
        vec4 debugValue = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices1.w))], debugPixel);
        return heatmap(clamp(debugValue.z / 64.0, 0.0, 1.0));
    }
    if (gaussianDebugView == 7u) {
        if (!hasImage(pc.imageIndices1.w)) { return vec3(-1.0); }
        ivec2 debugSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices1.w))]);
        ivec2 debugPixel = clamp(ivec2(uv * vec2(debugSize)), ivec2(0), debugSize - ivec2(1));
        vec4 debugValue = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices1.w))], debugPixel);
        return heatmap(clamp(debugValue.w / 48.0, 0.0, 1.0));
    }
    if (gaussianDebugView == 8u) {
        if (!hasImage(pc.imageIndices1.w)) { return vec3(-1.0); }
        ivec2 debugSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices1.w))]);
        ivec2 debugPixel = clamp(ivec2(uv * vec2(debugSize)), ivec2(0), debugSize - ivec2(1));
        vec4 debugValue = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices1.w))], debugPixel);
        return debugValue.a > 0.0 ? debugValue.rgb : vec3(0.0);
    }
    if (gaussianDebugView == 9u) {
        if (!hasImage(pc.imageIndices1.w)) { return vec3(-1.0); }
        ivec2 debugSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices1.w))]);
        ivec2 debugPixel = clamp(ivec2(uv * vec2(debugSize)), ivec2(0), debugSize - ivec2(1));
        vec4 debugValue = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices1.w))], debugPixel);
        return debugValue.a > 0.0 ? debugValue.rgb : vec3(0.0);
    }
    if (gaussianDebugView == 10u) {
        if (!hasImage(pc.imageIndices1.w)) { return vec3(-1.0); }
        ivec2 debugSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices1.w))]);
        ivec2 debugPixel = clamp(ivec2(uv * vec2(debugSize)), ivec2(0), debugSize - ivec2(1));
        vec4 debugValue = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices1.w))], debugPixel);
        return debugValue.a > 0.0 ? debugValue.rgb : vec3(0.0);
    }
    if (gaussianDebugView == 11u) {
        if (!hasImage(pc.imageIndices2.w)) { return vec3(-1.0); }
        ivec2 depthSize = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
        ivec2 depthPixel = clamp(ivec2(uv * vec2(depthSize)), ivec2(0), depthSize - ivec2(1));
        float rasterDepth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], depthPixel, 0).r;
        return vec3(1.0 - clamp(rasterDepth, 0.0, 1.0));
    }
    if (gaussianDebugView == 12u) {
        bool rasterVisible = false;
        if (hasImage(pc.imageIndices2.w)) {
            ivec2 depthSize = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
            ivec2 depthPixel = clamp(ivec2(uv * vec2(depthSize)), ivec2(0), depthSize - ivec2(1));
            float rasterDepth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], depthPixel, 0).r;
            rasterVisible = rasterDepth < 0.99999;
        }
        bool gaussianVisible = gaussianColor.a > 0.02;
        if (!rasterVisible && !gaussianVisible) {
            return vec3(0.0);
        }
        return vec3(rasterVisible ? 0.16 : 0.0, gaussianVisible ? 0.95 : 0.0, rasterVisible && gaussianVisible ? 0.85 : 0.08);
    }
    if (gaussianDebugView == 13u) {
        if (!hasImage(pc.imageIndices2.w) || !hasImage(pc.imageIndices1.w)) { return vec3(-1.0); }
        ivec2 depthSize = textureSize(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], 0);
        ivec2 depthPixel = clamp(ivec2(uv * vec2(depthSize)), ivec2(0), depthSize - ivec2(1));
        float rasterDepth = texelFetch(sampledImages[nonuniformEXT(int(pc.imageIndices2.w))], depthPixel, 0).r;
        ivec2 debugSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices1.w))]);
        ivec2 debugPixel = clamp(ivec2(uv * vec2(debugSize)), ivec2(0), debugSize - ivec2(1));
        vec4 debugValue = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices1.w))], debugPixel);
        if (debugValue.a <= 0.0 || rasterDepth >= 0.99999) {
            return vec3(0.0);
        }
        return heatmap(clamp(abs(debugValue.x - rasterDepth) * 64.0, 0.0, 1.0));
    }
    return vec3(-1.0);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    ivec2 baseSize = getOutputSize();

    vec2 uv = (vec2(pixel) + 0.5) / vec2(baseSize);
    if (pc.imageIndices1.x == MODE_BLOOM_EXTRACT
        || pc.imageIndices1.x == MODE_BLOOM_DOWNSAMPLE
        || pc.imageIndices1.x == MODE_BLOOM_UPSAMPLE) {
        outColor = vec4(runBloomPipelineStage(uv), 1.0);
        return;
    }

    vec3 deferredColor = vec3(0.0);
    if (hasImage(pc.imageIndices0.x)) {
        ivec2 deferredSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.x))]);
        ivec2 deferredPixel = clamp(ivec2(uv * vec2(deferredSize)), ivec2(0), deferredSize - ivec2(1));
        deferredColor = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.x))], deferredPixel).rgb;
    }

    vec3 pathTraceColor = vec3(0.0);
    if (hasImage(pc.imageIndices0.y)) {
        ivec2 pathTraceSize = imageSize(storageImages[nonuniformEXT(int(pc.imageIndices0.y))]);
        ivec2 pathTracePixel = clamp(ivec2(uv * vec2(pathTraceSize)), ivec2(0), pathTraceSize - ivec2(1));
        pathTraceColor = imageLoad(storageImages[nonuniformEXT(int(pc.imageIndices0.y))], pathTracePixel).rgb;
    }

    vec4 gaussianColor = resolveGaussian(pixel, uv);
    vec3 debugColor = resolveDebugView(pixel);
    if (debugColor.x >= 0.0) {
        outColor = vec4(applyPostProcess(applyFxaa(debugColor, uv) + computeBloom(uv), uv), 1.0);
        return;
    }
    vec3 gaussianDebugColor = resolveGaussianDebugView(gaussianColor, pixel, uv);
    if (gaussianDebugColor.x >= 0.0) {
        outColor = vec4(applyPostProcess(applyFxaa(gaussianDebugColor, uv) + computeBloom(uv), uv), 1.0);
        return;
    }

    uint compareMode = uint(pc.compareParams.x + 0.5);
    if (pc.imageIndices1.x == 0u && compareMode != 0u && hasImage(pc.imageIndices0.x) && hasImage(pc.imageIndices0.y)) {
        vec3 rasterDisplay = applyDisplayTransform(deferredColor);
        vec3 pathDisplay = applyDisplayTransform(pathTraceColor);
        if (compareMode == 1u) {
            float splitPosition = clamp(pc.compareParams.y, 0.02, 0.98);
            vec3 splitColor = uv.x < splitPosition ? rasterDisplay : pathDisplay;
            float divider = 1.0 - smoothstep(0.0, 0.003, abs(uv.x - splitPosition));
            outColor = vec4(applyPostProcess(applyFxaa(mix(splitColor, vec3(1.0), divider), uv) + computeBloom(uv), uv), 1.0);
            return;
        }
        if (compareMode == 2u) {
            float diff = length(rasterDisplay - pathDisplay) * pc.compareParams.z;
            outColor = vec4(applyPostProcess(heatmap(diff), uv), 1.0);
            return;
        }
    }

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
        composite = applyDisplayTransform(composite);
    }
    if (pc.gaussianDebug.w != 0u) {
        vec3 cascadeColor = resolveShadowCascadeColor(pixel);
        if (cascadeColor.x >= 0.0) {
            composite = mix(composite, cascadeColor, 0.36);
        }
    }
    composite = applyMotionBlur(composite, uv);
    composite = applyFxaa(composite, uv);
    composite += computeBloom(uv);
    composite = applyPostProcess(composite, uv);
    outColor = vec4(composite, 1.0);
}
