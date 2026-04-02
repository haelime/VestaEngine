#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// Official Gaussian scenes are read from a dedicated SSBO so the shader can
// access rotation, anisotropic scale, and SH coefficients directly.

const float kShC0 = 0.28209479177387814;
const float kShC1 = 0.4886025119029199;
const float kShC2_0 = 1.0925484305920792;
const float kShC2_1 = -1.0925484305920792;
const float kShC2_2 = 0.31539156525252005;
const float kShC2_3 = -1.0925484305920792;
const float kShC2_4 = 0.5462742152960396;
const float kShC3_0 = -0.5900435899266435;
const float kShC3_1 = 2.890611442640554;
const float kShC3_2 = -0.4570457994644658;
const float kShC3_3 = 0.3731763325901154;
const float kShC3_4 = -0.4570457994644658;
const float kShC3_5 = 1.445305721320277;
const float kShC3_6 = -0.5900435899266435;

struct GaussianPrimitive {
    vec4 positionOpacity;
    vec4 scale;
    vec4 rotation;
    vec4 shCoefficients[16];
};

layout(set = 0, binding = 2, std430) readonly buffer GaussianBuffer {
    GaussianPrimitive gaussians[];
} gaussianBuffers[];

layout(push_constant) uniform GaussianPushConstants {
    mat4 viewMatrix;
    mat4 viewProjection;
    vec4 cameraPositionAndSceneType;
    vec4 params0;
    vec4 params1;
    uvec4 bufferIndices;
    uvec4 options;
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outOpacity;
layout(location = 2) out vec2 outLocalCoord;
layout(location = 3) out float outViewDepth;

vec3 quat_rotate(vec4 q, vec3 v)
{
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

vec3 sanitize_scale(vec3 scale)
{
    return max(scale, vec3(1.0e-6));
}

vec2 projection_scales(mat4 viewProjection, mat4 viewMatrix)
{
    mat3 projection3x3 = mat3(viewProjection) * transpose(mat3(viewMatrix));
    return vec2(length(projection3x3[0]), length(projection3x3[1]));
}

vec3 evaluate_sh_color(GaussianPrimitive gaussian, vec3 viewDir)
{
    vec3 result = kShC0 * gaussian.shCoefficients[0].rgb;
    uint degree = pc.bufferIndices.z;
    bool viewDependent = pc.bufferIndices.w != 0u && pc.cameraPositionAndSceneType.w > 0.5;
    if (!viewDependent || degree == 0u) {
        return max(result + 0.5, vec3(0.0));
    }

    float x = viewDir.x;
    float y = viewDir.y;
    float z = viewDir.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    result += (-kShC1 * y) * gaussian.shCoefficients[1].rgb;
    result += ( kShC1 * z) * gaussian.shCoefficients[2].rgb;
    result += (-kShC1 * x) * gaussian.shCoefficients[3].rgb;

    if (degree >= 2u) {
        result += (kShC2_0 * x * y) * gaussian.shCoefficients[4].rgb;
        result += (kShC2_1 * y * z) * gaussian.shCoefficients[5].rgb;
        result += (kShC2_2 * (2.0 * zz - xx - yy)) * gaussian.shCoefficients[6].rgb;
        result += (kShC2_3 * x * z) * gaussian.shCoefficients[7].rgb;
        result += (kShC2_4 * (xx - yy)) * gaussian.shCoefficients[8].rgb;
    }

    if (degree >= 3u) {
        result += (kShC3_0 * y * (3.0 * xx - yy)) * gaussian.shCoefficients[9].rgb;
        result += (kShC3_1 * x * y * z) * gaussian.shCoefficients[10].rgb;
        result += (kShC3_2 * y * (4.0 * zz - xx - yy)) * gaussian.shCoefficients[11].rgb;
        result += (kShC3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * gaussian.shCoefficients[12].rgb;
        result += (kShC3_4 * x * (4.0 * zz - xx - yy)) * gaussian.shCoefficients[13].rgb;
        result += (kShC3_5 * z * (xx - yy)) * gaussian.shCoefficients[14].rgb;
        result += (kShC3_6 * x * (xx - 3.0 * yy)) * gaussian.shCoefficients[15].rgb;
    }

    return max(result + 0.5, vec3(0.0));
}

void main()
{
    GaussianPrimitive gaussian = gaussianBuffers[nonuniformEXT(int(pc.bufferIndices.x))].gaussians[gl_InstanceIndex];

    vec2 corners[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    vec2 localCorner = corners[gl_VertexIndex & 3];
    vec3 centerPosition = gaussian.positionOpacity.xyz;
    vec4 centerClip = pc.viewProjection * vec4(centerPosition, 1.0);

    if (centerClip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        outColor = vec4(0.0);
        outOpacity = 0.0;
        outLocalCoord = localCorner;
        outViewDepth = 0.0;
        return;
    }

    vec2 clipOffset = vec2(0.0);
    float viewDepth = 0.0;
    if (pc.cameraPositionAndSceneType.w > 0.5) {
        vec4 rotation = gaussian.rotation;
        float rotationLength = length(rotation);
        rotation = rotationLength > 1.0e-4 ? rotation / rotationLength : vec4(0.0, 0.0, 0.0, 1.0);

        vec3 scale = sanitize_scale(gaussian.scale.xyz);
        vec3 axisU = quat_rotate(rotation, vec3(scale.x, 0.0, 0.0));
        vec3 axisV = quat_rotate(rotation, vec3(0.0, scale.y, 0.0));
        vec3 axisW = quat_rotate(rotation, vec3(0.0, 0.0, scale.z));

        vec3 centerView = (pc.viewMatrix * vec4(centerPosition, 1.0)).xyz;
        float cameraZ = min(centerView.z, -1.0e-4);
        viewDepth = -cameraZ;

        mat3 rotationScale = mat3(axisU, axisV, axisW);
        mat3 covarianceWorld = rotationScale * transpose(rotationScale);

        vec2 viewportSize = max(pc.params0.zw, vec2(1.0));
        vec2 focal = projection_scales(pc.viewProjection, pc.viewMatrix) * viewportSize * 0.5;
        float invCameraZ = 1.0 / cameraZ;
        float invCameraZ2 = invCameraZ * invCameraZ;
        mat3 J = mat3(
            focal.x * invCameraZ, 0.0, -(focal.x * centerView.x) * invCameraZ2,
            0.0, -focal.y * invCameraZ, (focal.y * centerView.y) * invCameraZ2,
            0.0, 0.0, 0.0);
        mat3 T = J * mat3(pc.viewMatrix);
        mat3 covarianceScreen = T * covarianceWorld * transpose(T);
        mat2 cov = mat2(
            covarianceScreen[0][0], covarianceScreen[0][1],
            covarianceScreen[1][0], covarianceScreen[1][1]);
        cov[0][0] += pc.options.x != 0u ? 0.45 : 0.7;
        cov[1][1] += pc.options.x != 0u ? 0.45 : 0.7;

        float trace = cov[0][0] + cov[1][1];
        float determinant = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
        float root = sqrt(max(trace * trace * 0.25 - determinant, 1.0e-8));
        float lambda0 = max(trace * 0.5 + root, 1.0e-8);
        float lambda1 = max(trace * 0.5 - root, 1.0e-8);

        vec2 eigenvector0 = abs(cov[0][1]) > 1.0e-6 ? normalize(vec2(lambda0 - cov[1][1], cov[0][1])) : vec2(1.0, 0.0);
        vec2 eigenvector1 = vec2(-eigenvector0.y, eigenvector0.x);
        float sigmaCoverage = pc.options.x != 0u ? 3.2 : 2.8;
        float majorLengthPixels = sqrt(lambda0) * sigmaCoverage;
        float minorLengthPixels = sqrt(lambda1) * sigmaCoverage;
        majorLengthPixels = min(majorLengthPixels, 96.0);
        minorLengthPixels = min(minorLengthPixels, 96.0);
        minorLengthPixels = max(minorLengthPixels, max(majorLengthPixels / 14.0, 0.65));
        vec2 majorAxisPixels = eigenvector0 * majorLengthPixels;
        vec2 minorAxisPixels = eigenvector1 * minorLengthPixels;

        if (pc.options.y != 0u && max(length(majorAxisPixels), length(minorAxisPixels)) < 0.35) {
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            outColor = vec4(0.0);
            outOpacity = 0.0;
            outLocalCoord = localCorner;
            outViewDepth = 0.0;
            return;
        }

        vec2 majorAxisNdc = majorAxisPixels * (2.0 / viewportSize);
        vec2 minorAxisNdc = minorAxisPixels * (2.0 / viewportSize);
        clipOffset = (majorAxisNdc * localCorner.x + minorAxisNdc * localCorner.y) * centerClip.w;
    } else {
        float radiusPixels = clamp(max(pc.params0.x * max(gaussian.scale.x, 0.01), 1.0), 1.0, 96.0);
        vec2 viewportSize = max(pc.params0.zw, vec2(1.0));
        clipOffset = localCorner * radiusPixels * (2.0 / viewportSize) * centerClip.w;
        viewDepth = centerClip.w;
    }

    gl_Position = centerClip + vec4(clipOffset, 0.0, 0.0);
    outColor = vec4(evaluate_sh_color(gaussian, normalize(pc.cameraPositionAndSceneType.xyz - centerPosition)), 1.0);
    outOpacity = gaussian.positionOpacity.w;
    outLocalCoord = localCorner;
    outViewDepth = viewDepth;
}
