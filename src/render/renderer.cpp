#include <vesta/render/renderer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <SDL.h>
#include <SDL_vulkan.h>
#include <fmt/format.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <stb_image.h>

#include <vesta/core/debug.h>
#include <vesta/render/passes/composite_pass.h>
#include <vesta/render/passes/deferred_lighting_pass.h>
#include <vesta/render/passes/ddgi_probe_update_pass.h>
#include <vesta/render/passes/gaussian_splat_pass.h>
#include <vesta/render/passes/official_gaussian_raster_pass.h>
#include <vesta/render/passes/overdraw_pass.h>
#include <vesta/render/passes/geometry_raster_pass.h>
#include <vesta/render/passes/path_denoise_pass.h>
#include <vesta/render/passes/path_tracer_pass.h>
#include <vesta/render/passes/ray_effects_pass.h>
#include <vesta/render/passes/restir_di_pass.h>
#include <vesta/render/passes/restir_di_resolve_pass.h>
#include <vesta/render/passes/shadow_map_pass.h>
#include <vesta/render/passes/temporal_aa_pass.h>
#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>

namespace vesta::render {
namespace {
// Presets are derived from the active GPU because the heaviest pass in this
// sample is path tracing, and its reasonable resolution scale changes a lot
// between low-end, non-RT, and modern RT-capable cards.
std::string ToUpper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

bool IsRtx5060Ti(const RenderDevice& device)
{
    return ToUpper(device.GetGpuName()).find("RTX 5060 TI") != std::string::npos;
}

uint32_t GaussianInteractivePreviewFrameBudget(const vesta::scene::Scene& scene)
{
    if (!scene.HasTrainedGaussians()) {
        return 0u;
    }
    const uint32_t gaussianCount = scene.GetGaussianCount();
    if (gaussianCount >= 4'000'000u) {
        return 18u;
    }
    if (gaussianCount >= 1'000'000u) {
        return 12u;
    }
    return 8u;
}

float RadicalInverseVdc(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

glm::vec2 Hammersley(uint32_t index, uint32_t count)
{
    return glm::vec2(static_cast<float>(index) / static_cast<float>(count), RadicalInverseVdc(index));
}

glm::vec3 ImportanceSampleGgx(glm::vec2 sample, float roughness)
{
    const float a = roughness * roughness;
    const float phi = 2.0f * glm::pi<float>() * sample.x;
    const float cosTheta = std::sqrt((1.0f - sample.y) / std::max(1.0f + (a * a - 1.0f) * sample.y, 1.0e-5f));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));
    return glm::normalize(glm::vec3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta));
}

float GeometrySchlickGgx(float nDotV, float roughness)
{
    const float r = roughness + 1.0f;
    const float k = (r * r) / 8.0f;
    return nDotV / std::max(nDotV * (1.0f - k) + k, 1.0e-5f);
}

glm::vec2 IntegrateBrdf(float nDotV, float roughness)
{
    const glm::vec3 view(std::sqrt(std::max(1.0f - nDotV * nDotV, 0.0f)), 0.0f, nDotV);
    constexpr uint32_t kSampleCount = 128u;
    float scale = 0.0f;
    float bias = 0.0f;
    for (uint32_t sampleIndex = 0; sampleIndex < kSampleCount; ++sampleIndex) {
        const glm::vec3 halfVector = ImportanceSampleGgx(Hammersley(sampleIndex, kSampleCount), roughness);
        const glm::vec3 light = glm::normalize(2.0f * glm::dot(view, halfVector) * halfVector - view);
        const float nDotL = std::max(light.z, 0.0f);
        const float nDotH = std::max(halfVector.z, 0.0f);
        const float vDotH = std::max(glm::dot(view, halfVector), 0.0f);
        if (nDotL <= 0.0f) {
            continue;
        }

        const float g = GeometrySchlickGgx(nDotL, roughness) * GeometrySchlickGgx(nDotV, roughness);
        const float gVis = (g * vDotH) / std::max(nDotH * nDotV, 1.0e-5f);
        const float fc = std::pow(1.0f - vDotH, 5.0f);
        scale += (1.0f - fc) * gVis;
        bias += fc * gVis;
    }
    return glm::vec2(scale, bias) / static_cast<float>(kSampleCount);
}

glm::vec3 DirectionFromEquirectUv(float u, float v)
{
    const float phi = (u - 0.5f) * 2.0f * glm::pi<float>();
    const float theta = v * glm::pi<float>();
    const float sinTheta = std::sin(theta);
    return glm::normalize(glm::vec3(std::cos(phi) * sinTheta, std::cos(theta), std::sin(phi) * sinTheta));
}

glm::vec3 DirectionFromCubeFace(uint32_t faceIndex, float u, float v)
{
    const float x = u * 2.0f - 1.0f;
    const float y = v * 2.0f - 1.0f;
    switch (faceIndex) {
    case 0: // +X
        return glm::normalize(glm::vec3(1.0f, -y, -x));
    case 1: // -X
        return glm::normalize(glm::vec3(-1.0f, -y, x));
    case 2: // +Y
        return glm::normalize(glm::vec3(x, 1.0f, y));
    case 3: // -Y
        return glm::normalize(glm::vec3(x, -1.0f, -y));
    case 4: // +Z
        return glm::normalize(glm::vec3(x, -y, 1.0f));
    case 5: // -Z
    default:
        return glm::normalize(glm::vec3(-x, -y, -1.0f));
    }
}

glm::vec3 SampleEquirect(std::span<const float> rgbaPixels, uint32_t width, uint32_t height, glm::vec3 direction)
{
    if (rgbaPixels.empty() || width == 0u || height == 0u) {
        return glm::vec3(0.0f);
    }
    direction = glm::normalize(direction);
    float u = std::atan2(direction.z, direction.x) / (2.0f * glm::pi<float>()) + 0.5f;
    const float v = std::acos(std::clamp(direction.y, -1.0f, 1.0f)) / glm::pi<float>();
    u = u - std::floor(u);
    const float x = u * static_cast<float>(width - 1u);
    const float y = std::clamp(v, 0.0f, 1.0f) * static_cast<float>(height - 1u);
    const uint32_t x0 = static_cast<uint32_t>(std::floor(x));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(y));
    const uint32_t x1 = (x0 + 1u) % width;
    const uint32_t y1 = std::min(y0 + 1u, height - 1u);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    auto sample = [&](uint32_t sx, uint32_t sy) {
        const size_t offset = (static_cast<size_t>(sy) * width + sx) * 4u;
        return glm::vec3(rgbaPixels[offset + 0u], rgbaPixels[offset + 1u], rgbaPixels[offset + 2u]);
    };
    const glm::vec3 a = glm::mix(sample(x0, y0), sample(x1, y0), tx);
    const glm::vec3 b = glm::mix(sample(x0, y1), sample(x1, y1), tx);
    return glm::mix(a, b, ty);
}

glm::vec3 CosineSampleHemisphere(uint32_t index, uint32_t count)
{
    const glm::vec2 xi = Hammersley(index, count);
    const float phi = 2.0f * glm::pi<float>() * xi.x;
    const float r = std::sqrt(xi.y);
    const float x = r * std::cos(phi);
    const float z = r * std::sin(phi);
    const float y = std::sqrt(std::max(0.0f, 1.0f - xi.y));
    return glm::vec3(x, y, z);
}

glm::vec3 ImportanceSampleGgxYUp(glm::vec2 sample, float roughness)
{
    const float a = roughness * roughness;
    const float phi = 2.0f * glm::pi<float>() * sample.x;
    const float cosTheta = std::sqrt((1.0f - sample.y) / std::max(1.0f + (a * a - 1.0f) * sample.y, 1.0e-5f));
    const float sinTheta = std::sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0f));
    return glm::normalize(glm::vec3(std::cos(phi) * sinTheta, cosTheta, std::sin(phi) * sinTheta));
}

glm::vec3 TangentToWorld(glm::vec3 tangentSample, glm::vec3 normal)
{
    const glm::vec3 up = std::abs(normal.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 tangent = glm::normalize(glm::cross(up, normal));
    const glm::vec3 bitangent = glm::cross(normal, tangent);
    return glm::normalize(tangent * tangentSample.x + normal * tangentSample.y + bitangent * tangentSample.z);
}

bool NeedsGeometryPass(const RendererSettings& settings)
{
    if (!settings.enableRaster) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode != RendererDisplayMode::PathTrace;
}

bool NeedsDeferredPass(const RendererSettings& settings)
{
    if (!settings.enableRaster) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode == RendererDisplayMode::Composite || settings.displayMode == RendererDisplayMode::DeferredLighting;
}

bool NeedsGaussianPass(const RendererSettings& settings)
{
    if (!settings.enableGaussian) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode == RendererDisplayMode::Composite || settings.displayMode == RendererDisplayMode::Gaussian;
}

bool NeedsPathTracePass(const RendererSettings& settings)
{
    if (!settings.enablePathTracing) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode == RendererDisplayMode::Composite || settings.displayMode == RendererDisplayMode::PathTrace;
}

bool NeedsPathDenoisePass(const RendererSettings& settings)
{
    return NeedsPathTracePass(settings) && settings.enablePathTraceDenoiser && settings.pathTraceDebugView == PathTraceDebugView::Final;
}

bool IsTemporalDebugView(RendererDebugView debugView)
{
    return debugView == RendererDebugView::TemporalHistoryColor
        || debugView == RendererDebugView::TemporalHistoryDepth
        || debugView == RendererDebugView::TemporalReprojection
        || debugView == RendererDebugView::TemporalDisocclusion
        || debugView == RendererDebugView::TemporalJitter;
}

bool NeedsTemporalAAPass(const RendererSettings& settings)
{
    return NeedsDeferredPass(settings)
        && (settings.enableTaa || settings.enableTemporalUpscaler || IsTemporalDebugView(settings.debugView));
}

bool IsRayEffectsRequested(const RendererSettings& settings)
{
    return settings.enableRtShadows || settings.enableRtAmbientOcclusion || settings.enableRtReflections
        || settings.enableRtGlobalIllumination;
}

bool IsRestirRequested(const RendererSettings& settings)
{
    return settings.enableRestirDi || settings.enableRestirGi || settings.enableRestirPt;
}

struct RuntimeShaderSource {
    std::string_view sourceName;
    bool requiresVulkan13{ false };
};

constexpr std::array<RuntimeShaderSource, 34> kRuntimeShaderSources{
    RuntimeShaderSource{ "gradient.comp" },
    RuntimeShaderSource{ "gradient_color.comp" },
    RuntimeShaderSource{ "hardcoded_triangle.frag" },
    RuntimeShaderSource{ "hardcoded_triangle.vert" },
    RuntimeShaderSource{ "sky.comp" },
    RuntimeShaderSource{ "composite.frag" },
    RuntimeShaderSource{ "composite.vert" },
    RuntimeShaderSource{ "deferred_lighting.comp" },
    RuntimeShaderSource{ "gaussian.frag" },
    RuntimeShaderSource{ "gaussian.vert" },
    RuntimeShaderSource{ "gaussian_bin.comp" },
    RuntimeShaderSource{ "gaussian_tile.comp" },
    RuntimeShaderSource{ "official_gaussian_raster.comp" },
    RuntimeShaderSource{ "official_gaussian_duplicate.comp" },
    RuntimeShaderSource{ "official_gaussian_preprocess.comp" },
    RuntimeShaderSource{ "official_gaussian_scan.comp" },
    RuntimeShaderSource{ "official_gaussian_sort.comp" },
    RuntimeShaderSource{ "official_gaussian_ranges.comp" },
    RuntimeShaderSource{ "geometry.frag" },
    RuntimeShaderSource{ "geometry.vert" },
    RuntimeShaderSource{ "shadow_depth.frag" },
    RuntimeShaderSource{ "shadow_depth.vert" },
    RuntimeShaderSource{ "overdraw.frag" },
    RuntimeShaderSource{ "overdraw.vert" },
    RuntimeShaderSource{ "ray_effects.comp", true },
    RuntimeShaderSource{ "restir_di.comp" },
    RuntimeShaderSource{ "restir_di_resolve.comp" },
    RuntimeShaderSource{ "ddgi_probe_update.comp", true },
    RuntimeShaderSource{ "pathtrace.comp" },
    RuntimeShaderSource{ "path_denoise.comp" },
    RuntimeShaderSource{ "temporal_aa.comp" },
    RuntimeShaderSource{ "pathtrace.rgen", true },
    RuntimeShaderSource{ "pathtrace.rmiss", true },
    RuntimeShaderSource{ "pathtrace.rchit", true },
};

bool UsesStreamingUpload(const RendererSettings& settings)
{
    return settings.sceneUploadMode == SceneUploadMode::Streaming && settings.useDeviceLocalSceneBuffers;
}

std::string QuoteCommandArgument(const std::filesystem::path& path)
{
    std::string value = path.string();
    size_t offset = 0;
    while ((offset = value.find('"', offset)) != std::string::npos) {
        value.insert(offset, "\\");
        offset += 2;
    }
    return "\"" + value + "\"";
}

std::filesystem::path ResolveShaderOutputDirectory()
{
    return vkutil::resolve_runtime_path("shaders/composite.frag.spv").parent_path();
}

std::filesystem::path ResolveShaderSourceDirectory()
{
    return vkutil::resolve_runtime_path("shaders/composite.frag").parent_path();
}

std::string ReadTextFileLimited(const std::filesystem::path& path, size_t maxLines)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return {};
    }

    std::ostringstream output;
    std::string line;
    size_t lines = 0;
    while (lines < maxLines && std::getline(input, line)) {
        output << line << '\n';
        ++lines;
    }
    if (input && lines == maxLines) {
        output << "... compiler output truncated ...\n";
    }
    return output.str();
}

std::string GetEnvironmentVariableString(const char* name)
{
    char* value = nullptr;
    size_t valueSize = 0;
    if (_dupenv_s(&value, &valueSize, name) != 0 || value == nullptr) {
        return {};
    }

    std::string result(value);
    std::free(value);
    return result;
}

bool CompileRuntimeShaders(std::string& message)
{
    const std::string vulkanSdk = GetEnvironmentVariableString("VULKAN_SDK");
    if (vulkanSdk.empty()) {
        message = "GLSL compile skipped: VULKAN_SDK is not set. Reloading existing SPIR-V.";
        return true;
    }

    const std::filesystem::path validatorPath = std::filesystem::path(vulkanSdk) / "Bin" / "glslangValidator.exe";
    if (!std::filesystem::exists(validatorPath)) {
        message = "GLSL compile skipped: glslangValidator.exe was not found at " + validatorPath.string()
            + ". Reloading existing SPIR-V.";
        return true;
    }

    std::filesystem::path sourceDirectory;
    std::filesystem::path outputDirectory;
    try {
        sourceDirectory = ResolveShaderSourceDirectory();
        outputDirectory = ResolveShaderOutputDirectory();
    } catch (const std::exception& error) {
        message = std::string("GLSL compile skipped: ") + error.what() + ". Reloading existing SPIR-V.";
        return true;
    }

    std::filesystem::create_directories(outputDirectory);
    const std::filesystem::path logPath = outputDirectory / ".shader_hot_reload.log";

    uint32_t compiledCount = 0u;
    for (const RuntimeShaderSource& shader : kRuntimeShaderSources) {
        const std::filesystem::path sourcePath = sourceDirectory / shader.sourceName;
        if (!std::filesystem::exists(sourcePath)) {
            message = "Shader source missing: " + sourcePath.string();
            return false;
        }

        const std::filesystem::path outputPath = outputDirectory / (std::string(shader.sourceName) + ".spv");
        std::ostringstream command;
        command << "call " << QuoteCommandArgument(validatorPath) << ' ';
        if (shader.requiresVulkan13) {
            command << "--target-env vulkan1.3 ";
        }
        command << "-V " << QuoteCommandArgument(sourcePath) << " -o " << QuoteCommandArgument(outputPath)
                << " > " << QuoteCommandArgument(logPath) << " 2>&1";

        const int result = std::system(command.str().c_str());
        const std::string compilerOutput = ReadTextFileLimited(logPath, 20);
        if (result != 0) {
            std::ostringstream failure;
            failure << "Shader compile failed: shaders/" << shader.sourceName << '\n';
            if (!compilerOutput.empty()) {
                failure << compilerOutput;
            } else {
                failure << "Compiler process exited with code " << result << " without output.\n";
            }
            message = failure.str();
            return false;
        }
        ++compiledCount;
    }

    message = "Compiled " + std::to_string(compiledCount) + " GLSL shaders to " + outputDirectory.string() + ".";
    return true;
}

void ValidateSceneLoadTransition(const SceneLoadStatus& status, SceneLoadState nextState, std::string_view context)
{
    VESTA_ASSERT_STATE(IsValidSceneLoadTransition(status.state, nextState),
        fmt::format("Invalid scene load transition {} -> {} in {} for '{}'",
            static_cast<uint32_t>(status.state),
            static_cast<uint32_t>(nextState),
            context,
            status.path.string()));
}

void ApplySceneLoadState(SceneLoadStatus& status, SceneLoadState nextState, std::string message, std::string_view context)
{
    ValidateSceneLoadTransition(status, nextState, context);
    status.state = nextState;
    status.message = std::move(message);
}

float ClampPathTraceScale(float scale)
{
    return std::clamp(scale, 0.25f, 1.0f);
}

VkExtent3D ScaleExtent(VkExtent3D extent, float scale)
{
    const float clampedScale = ClampPathTraceScale(scale);
    extent.width = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(extent.width) * clampedScale)));
    extent.height = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(extent.height) * clampedScale)));
    extent.depth = 1;
    return extent;
}

RendererPreset ChooseRecommendedPreset(const RenderDevice& device)
{
    const uint32_t dedicatedMemoryMiB = device.GetDedicatedVideoMemoryMiB();

    if (!device.IsRayTracingSupported()) {
        return dedicatedMemoryMiB >= 12u * 1024u ? RendererPreset::Balanced : RendererPreset::Performance;
    }

    if (IsRtx5060Ti(device)) {
        return dedicatedMemoryMiB >= 12u * 1024u ? RendererPreset::Quality : RendererPreset::Balanced;
    }

    if (dedicatedMemoryMiB >= 14u * 1024u) {
        return RendererPreset::Quality;
    }
    if (dedicatedMemoryMiB >= 8u * 1024u) {
        return RendererPreset::Balanced;
    }
    return RendererPreset::Performance;
}

void ApplyPresetSettings(RendererSettings& settings, const RenderDevice& device, RendererPreset preset)
{
    settings.displayMode = RendererDisplayMode::DeferredLighting;
    settings.enableRaster = true;
    settings.enableGaussian = true;
    settings.enablePathTracing = true;
    settings.gaussianOpacity = 1.0f;
    settings.gaussianShDegree = 0u;

    const bool hardwareRtPreferred = device.IsRayTracingSupported();
    settings.pathTraceBackend = hardwareRtPreferred ? PathTraceBackend::Auto : PathTraceBackend::Compute;

    switch (preset) {
    case RendererPreset::Performance:
        settings.gaussianMix = 0.18f;
        settings.pathTraceResolutionScale = hardwareRtPreferred ? 0.50f : 0.33f;
        break;
    case RendererPreset::Balanced:
        settings.gaussianMix = 0.24f;
        settings.pathTraceResolutionScale = hardwareRtPreferred ? 0.67f : 0.50f;
        break;
    case RendererPreset::Quality:
        settings.gaussianMix = 0.28f;
        settings.pathTraceResolutionScale = hardwareRtPreferred ? 1.0f : 0.67f;
        break;
    case RendererPreset::Recommended:
    default:
        ApplyPresetSettings(settings, device, ChooseRecommendedPreset(device));
        return;
    }

    settings.pathTraceResolutionScale = ClampPathTraceScale(settings.pathTraceResolutionScale);
}

std::array<glm::vec4, 6> ExtractFrustumPlanes(const glm::mat4& viewProjection)
{
    const glm::mat4 matrix = glm::transpose(viewProjection);
    std::array<glm::vec4, 6> planes{
        matrix[3] + matrix[0],
        matrix[3] - matrix[0],
        matrix[3] + matrix[1],
        matrix[3] - matrix[1],
        matrix[3] + matrix[2],
        matrix[3] - matrix[2],
    };

    for (glm::vec4& plane : planes) {
        const float length = glm::length(glm::vec3(plane));
        if (length > 0.0f) {
            plane /= length;
        }
    }
    return planes;
}

bool IsSurfaceVisible(const vesta::scene::SceneSurfaceBounds& bounds, const std::array<glm::vec4, 6>& planes)
{
    for (const glm::vec4& plane : planes) {
        const float distance = glm::dot(glm::vec3(plane), bounds.center) + plane.w;
        if (distance < -bounds.radius) {
            return false;
        }
    }
    return true;
}

bool IsSurfaceWithinDistance(const vesta::scene::SceneSurfaceBounds& bounds,
    const glm::vec3& cameraPosition,
    float sceneRadius,
    float distanceCullScale)
{
    const float distance = glm::distance(cameraPosition, bounds.center);
    const float allowedDistance = std::max(bounds.radius * 12.0f, sceneRadius * distanceCullScale);
    return distance <= allowedDistance;
}

float DefaultOrbitDistance(float currentDistance, float targetRadius)
{
    const float minimumDistance = std::max(targetRadius * 2.5f, 0.75f);
    if (currentDistance > 0.0f) {
        return std::max(currentDistance, minimumDistance);
    }
    return minimumDistance;
}

void EnsureParentDirectory(const std::filesystem::path& path)
{
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code errorCode;
        std::filesystem::create_directories(parent, errorCode);
    }
}

std::array<uint8_t, 3> ReadSwapchainRgb(const uint8_t* source, VkFormat format)
{
    if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
        return { source[2], source[1], source[0] };
    }
    return { source[0], source[1], source[2] };
}

void AppendU32Be(std::vector<uint8_t>& bytes, uint32_t value)
{
    bytes.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
}

void AppendU16Le(std::vector<uint8_t>& bytes, uint16_t value)
{
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
}

uint32_t Crc32(std::span<const uint8_t> bytes)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (uint8_t byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

uint32_t Adler32(std::span<const uint8_t> bytes)
{
    constexpr uint32_t kModAdler = 65521u;
    uint32_t a = 1u;
    uint32_t b = 0u;
    for (uint8_t byte : bytes) {
        a = (a + byte) % kModAdler;
        b = (b + a) % kModAdler;
    }
    return (b << 16) | a;
}

void AppendPngChunk(std::vector<uint8_t>& png, std::string_view type, std::span<const uint8_t> payload)
{
    AppendU32Be(png, static_cast<uint32_t>(payload.size()));
    const size_t chunkStart = png.size();
    png.insert(png.end(), type.begin(), type.end());
    png.insert(png.end(), payload.begin(), payload.end());
    AppendU32Be(png, Crc32(std::span<const uint8_t>(png.data() + chunkStart, png.size() - chunkStart)));
}

std::vector<uint8_t> DeflateStore(std::span<const uint8_t> bytes)
{
    std::vector<uint8_t> output;
    output.reserve(bytes.size() + bytes.size() / 65535u * 5u + 8u);
    output.push_back(0x78);
    output.push_back(0x01);

    size_t offset = 0;
    while (offset < bytes.size()) {
        const size_t chunkSize = std::min<size_t>(65535u, bytes.size() - offset);
        const bool finalBlock = offset + chunkSize == bytes.size();
        output.push_back(finalBlock ? 0x01 : 0x00);
        AppendU16Le(output, static_cast<uint16_t>(chunkSize));
        AppendU16Le(output, static_cast<uint16_t>(~static_cast<uint16_t>(chunkSize)));
        output.insert(output.end(), bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + chunkSize));
        offset += chunkSize;
    }

    AppendU32Be(output, Adler32(bytes));
    return output;
}

bool WriteSwapchainPng(const std::filesystem::path& path,
    const void* pixels,
    VkExtent2D extent,
    VkFormat format)
{
    if (pixels == nullptr || extent.width == 0 || extent.height == 0) {
        return false;
    }

    EnsureParentDirectory(path);

    std::vector<uint8_t> scanlines;
    scanlines.reserve(static_cast<size_t>(extent.height) * (1u + static_cast<size_t>(extent.width) * 3u));
    const auto* source = static_cast<const uint8_t*>(pixels);
    for (uint32_t y = 0; y < extent.height; ++y) {
        scanlines.push_back(0);
        for (uint32_t x = 0; x < extent.width; ++x) {
            const uint8_t* pixel = source + (static_cast<size_t>(y) * extent.width + x) * 4u;
            const std::array<uint8_t, 3> rgb = ReadSwapchainRgb(pixel, format);
            scanlines.insert(scanlines.end(), rgb.begin(), rgb.end());
        }
    }

    std::vector<uint8_t> png{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n' };
    std::vector<uint8_t> ihdr;
    ihdr.reserve(13);
    AppendU32Be(ihdr, extent.width);
    AppendU32Be(ihdr, extent.height);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(2); // truecolor RGB
    ihdr.push_back(0); // deflate
    ihdr.push_back(0); // adaptive filters
    ihdr.push_back(0); // no interlace
    AppendPngChunk(png, "IHDR", ihdr);
    const std::vector<uint8_t> idat = DeflateStore(scanlines);
    AppendPngChunk(png, "IDAT", idat);
    AppendPngChunk(png, "IEND", {});

    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        return false;
    }
    output.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return output.good();
}

bool WriteSwapchainPpm(const std::filesystem::path& path,
    const void* pixels,
    VkExtent2D extent,
    VkFormat format)
{
    if (pixels == nullptr || extent.width == 0 || extent.height == 0) {
        return false;
    }

    EnsureParentDirectory(path);

    std::ofstream output(path, std::ios::binary);
    if (!output.is_open()) {
        return false;
    }

    output << "P6\n" << extent.width << ' ' << extent.height << "\n255\n";
    const auto* source = static_cast<const uint8_t*>(pixels);
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            const uint8_t* pixel = source + (static_cast<size_t>(y) * extent.width + x) * 4u;
            const std::array<uint8_t, 3> rgb = ReadSwapchainRgb(pixel, format);
            output.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
        }
    }
    return output.good();
}

void ConfigureGeometryRasterPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& rasterPass = static_cast<GeometryRasterPass&>(pass);
    rasterPass.SetTargets(
        resources.gbufferAlbedo,
        resources.gbufferNormal,
        resources.gbufferMaterial,
        resources.gbufferDebug,
        resources.gbufferMotion,
        resources.gbufferReactive,
        resources.sceneDepth);
    rasterPass.SetScene(&renderer.GetScene());
    rasterPass.SetCamera(&renderer.GetCamera());
    const bool useVisibilitySet =
        (renderer.GetSettings().enableFrustumCulling || renderer.GetSettings().enableDistanceCulling)
        && renderer.HasValidVisibilitySet();
    rasterPass.SetVisibleSurfaceIndices(useVisibilitySet ? &renderer.GetVisibleSurfaceIndices() : nullptr);
    rasterPass.SetUseIndirectDraw(renderer.GetSettings().useIndirectDraw);
}

void ConfigureShadowMapPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& shadowPass = static_cast<ShadowMapPass&>(pass);
    shadowPass.SetOutput(resources.shadowMap);
    shadowPass.SetScene(&renderer.GetScene());
    shadowPass.SetCamera(&renderer.GetCamera());
    shadowPass.SetLight(renderer.GetSettings().lightDirectionAndIntensity);
    shadowPass.SetCascadeSettings(renderer.GetSettings().shadowCascadeCount, renderer.GetSettings().shadowCascadeLambda);
}

void ConfigureOverdrawPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& overdrawPass = static_cast<OverdrawPass&>(pass);
    overdrawPass.SetOutput(resources.overdraw);
    overdrawPass.SetScene(&renderer.GetScene());
    overdrawPass.SetCamera(&renderer.GetCamera());
}

void ConfigureDeferredLightingPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& lightingPass = static_cast<DeferredLightingPass&>(pass);
    const auto& settings = renderer.GetSettings();
    lightingPass.SetInputs(resources.gbufferAlbedo, resources.gbufferNormal, resources.gbufferMaterial, resources.sceneDepth);
    lightingPass.SetCamera(&renderer.GetCamera());
    lightingPass.SetLight(settings.lightDirectionAndIntensity);
    lightingPass.SetLightColors(settings.directionalLightColor, settings.pointLightColor, settings.spotLightColor, settings.areaLightColor);
    lightingPass.SetPointLight(settings.enablePointLight, settings.pointLightPositionAndIntensity);
    lightingPass.SetSpotLight(settings.enableSpotLight, settings.spotLightPositionAndIntensity, settings.spotLightDirectionAndAngle);
    lightingPass.SetAreaLight(settings.enableAreaLight, settings.areaLightPositionAndIntensity, settings.areaLightNormalAndSize);
    lightingPass.SetEnvironment(glm::vec4(settings.environmentIntensity,
        glm::radians(settings.environmentRotationDegrees),
        static_cast<float>(settings.environmentPreset),
        settings.environmentDiffuseStrength));
    lightingPass.SetEnvironmentImage(renderer.GetEnvironmentSampledImageIndex());
    lightingPass.SetIblDiffuseIrradianceImage(renderer.GetIblDiffuseIrradianceSampledImageIndex());
    lightingPass.SetIblBrdfLutImage(renderer.GetIblBrdfLutSampledImageIndex());
    lightingPass.SetIblSpecularPrefilterImage(renderer.GetIblSpecularPrefilterSampledImageIndex());
    lightingPass.SetEnvironmentSpecularStrength(settings.environmentSpecularStrength);
    lightingPass.SetRayEffects(resources.rayEffects,
        resources.rayReflection,
        resources.rayGlobalIllumination,
        settings.enableRtShadows,
        settings.enableRtAmbientOcclusion,
        settings.enableRtReflections,
        settings.rtDenoiser,
        settings.rtTemporalAccumulation);
    lightingPass.SetRestirDiResolve(resources.restirDirectLighting, settings.enableRestirDi);
    lightingPass.SetAmbientOcclusion(settings.enableSsao, settings.ssaoRadius, settings.ssaoIntensity);
    lightingPass.SetScreenSpaceReflections(
        settings.enableSsr, settings.ssrMaxDistance, settings.ssrThickness, settings.ssrIntensity);
    lightingPass.SetScreenSpaceGlobalIllumination(
        settings.enableSsgi, settings.ssgiRadius, settings.ssgiIntensity, settings.ssgiSampleCount);
    lightingPass.SetDdgi(settings.enableDdgi,
        settings.ddgiProbeCountX,
        settings.ddgiProbeCountY,
        settings.ddgiProbeCountZ,
        settings.ddgiProbeSpacing,
        settings.ddgiHysteresis,
        settings.ddgiIntensity);
    lightingPass.SetContactShadows(
        settings.enableContactShadows, settings.contactShadowLength, settings.contactShadowIntensity);
    if (resources.shadowMap && renderer.GetScene().HasRasterGeometry()) {
        const auto cascades = BuildDirectionalShadowCascades(renderer.GetScene().GetBounds(),
            renderer.GetCamera(),
            settings.lightDirectionAndIntensity,
            settings.shadowCascadeCount,
            settings.shadowCascadeLambda);
        lightingPass.SetShadowMap(resources.shadowMap,
            cascades,
            settings.shadowCascadeCount,
            settings.shadowCascadeLambda,
            settings.shadowBias,
            settings.shadowNormalBias,
            settings.shadowStrength,
            settings.enablePcssShadows,
            settings.shadowFilterRadius);
    } else {
        lightingPass.SetShadowMap({},
            {},
            0u,
            settings.shadowCascadeLambda,
            settings.shadowBias,
            settings.shadowNormalBias,
            0.0f,
            settings.enablePcssShadows,
            settings.shadowFilterRadius);
    }
    lightingPass.SetOutput(resources.deferredLighting);
    lightingPass.SetDebugOutput(resources.deferredLightingDebug, static_cast<uint32_t>(settings.debugView));
}

void ConfigureRayEffectsPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& rayEffectsPass = static_cast<RayEffectsPass&>(pass);
    const auto& settings = renderer.GetSettings();
    rayEffectsPass.SetInputs(resources.gbufferNormal, resources.sceneDepth);
    rayEffectsPass.SetOutputs(resources.rayEffects, resources.rayReflection, resources.rayGlobalIllumination);
    rayEffectsPass.SetScene(&renderer.GetScene());
    rayEffectsPass.SetCamera(&renderer.GetCamera());
    rayEffectsPass.SetFrameSlot(renderer.GetFrameSlot());
    rayEffectsPass.SetFrameIndex(renderer.GetPathTraceFrameIndex());
    rayEffectsPass.SetLight(settings.lightDirectionAndIntensity);
    rayEffectsPass.SetControls(settings.enableRtShadows,
        settings.enableRtAmbientOcclusion,
        settings.enableRtReflections,
        settings.enableRtGlobalIllumination,
        settings.rtShadowSamples,
        settings.rtAoSamples,
        settings.rtReflectionSamples,
        settings.rtGiSamples,
        settings.rtMaxRayDistance,
        settings.rtAoRadius,
        settings.rtReflectionRoughnessCutoff);
}

void ConfigureRestirDiPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources&)
{
    auto& restirPass = static_cast<RestirDiPass&>(pass);
    const auto& settings = renderer.GetSettings();
    const auto stats = renderer.GetRestirStats();
    const auto extent = renderer.GetRenderDevice().GetSwapchainExtent();
    restirPass.SetReservoirBuffers(renderer.GetRestirReservoirBuffer(), renderer.GetRestirHistoryReservoirBuffer());
    restirPass.SetControls(renderer.GetPathTraceFrameIndex(),
        extent.width,
        extent.height,
        stats.candidateLightCount,
        stats.reservoirCount,
        settings.restirSpatialSamples,
        stats.activeLightCount,
        stats.localLightCount,
        stats.emissiveTriangleCount,
        stats.temporalReuse,
        stats.spatialReuse);
}

void ConfigureRestirDiResolvePass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& resolvePass = static_cast<RestirDiResolvePass&>(pass);
    const auto& settings = renderer.GetSettings();
    const auto stats = renderer.GetRestirStats();
    resolvePass.SetInputs(resources.gbufferAlbedo, resources.gbufferNormal, resources.gbufferMaterial, resources.sceneDepth);
    resolvePass.SetOutput(resources.restirDirectLighting);
    resolvePass.SetReservoirBuffer(renderer.GetRestirReservoirBuffer());
    resolvePass.SetCamera(&renderer.GetCamera());
    resolvePass.SetLight(settings.lightDirectionAndIntensity);
    resolvePass.SetLightColors(
        settings.directionalLightColor, settings.pointLightColor, settings.spotLightColor, settings.areaLightColor);
    resolvePass.SetPointLight(settings.enablePointLight, settings.pointLightPositionAndIntensity);
    resolvePass.SetSpotLight(settings.enableSpotLight, settings.spotLightPositionAndIntensity, settings.spotLightDirectionAndAngle);
    resolvePass.SetAreaLight(settings.enableAreaLight, settings.areaLightPositionAndIntensity, settings.areaLightNormalAndSize);
    resolvePass.SetControls(renderer.GetPathTraceFrameIndex(),
        stats.reservoirCount,
        stats.candidateLightCount,
        stats.activeLightCount,
        stats.localLightCount,
        stats.emissiveTriangleCount,
        settings.restirSpatialSamples,
        settings.restirDirectLightingIntensity,
        settings.restirSpatialReuse,
        settings.restirShowReservoirs,
        settings.restirShowSelectedLight);
}

void ConfigureDdgiProbeUpdatePass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources&)
{
    auto& ddgiPass = static_cast<DdgiProbeUpdatePass&>(pass);
    const auto& settings = renderer.GetSettings();
    ddgiPass.SetProbeBuffers(renderer.GetDdgiIrradianceBuffer(), renderer.GetDdgiVisibilityBuffer());
    ddgiPass.SetScene(&renderer.GetScene());
    ddgiPass.SetFrameSlot(renderer.GetFrameSlot());
    ddgiPass.SetFrameIndex(renderer.GetPathTraceFrameIndex());
    ddgiPass.SetControls(settings.ddgiProbeCountX,
        settings.ddgiProbeCountY,
        settings.ddgiProbeCountZ,
        settings.ddgiRaysPerProbe,
        settings.ddgiProbeSpacing,
        settings.ddgiHysteresis,
        settings.lightDirectionAndIntensity,
        settings.directionalLightColor,
        glm::vec4(settings.environmentIntensity,
            glm::radians(settings.environmentRotationDegrees),
            static_cast<float>(settings.environmentPreset),
            settings.environmentDiffuseStrength));
}

void ConfigureGaussianPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& gaussianPass = static_cast<GaussianSplatPass&>(pass);
    gaussianPass.SetDepthInput(resources.sceneDepth);
    gaussianPass.SetOutputs(resources.gaussianAccum, resources.gaussianReveal);
    gaussianPass.SetScene(&renderer.GetScene());
    gaussianPass.SetCamera(&renderer.GetCamera());
    const bool interactivePreview = renderer.GetScene().HasTrainedGaussians() && renderer.IsGaussianInteractivePreviewActive();
    const uint32_t effectiveShDegree = interactivePreview
        ? 0u
        : std::min(renderer.GetSettings().gaussianShDegree, renderer.GetScene().GetGaussianShDegree());
    gaussianPass.SetParams(renderer.GetSettings().gaussianOpacity,
        renderer.GetSettings().enableGaussian,
        effectiveShDegree,
        interactivePreview ? false : renderer.GetSettings().gaussianViewDependentColor,
        interactivePreview ? false : renderer.GetSettings().gaussianAntialiasing,
        true);
}

void ConfigureOfficialGaussianPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& gaussianPass = static_cast<OfficialGaussianRasterPass&>(pass);
    const bool useDepthInput = renderer.GetSettings().enableRaster && renderer.GetScene().GetSceneKind() == vesta::scene::SceneKind::Mesh;
    gaussianPass.SetDepthInput(useDepthInput ? resources.sceneDepth : GraphTextureHandle{});
    gaussianPass.SetOutputs(resources.gaussianAccum, resources.gaussianReveal);
    gaussianPass.SetDebugOutput(resources.gaussianDebug);
    gaussianPass.SetScene(&renderer.GetScene());
    gaussianPass.SetCamera(&renderer.GetCamera());
    gaussianPass.SetJobSystem(&renderer.GetJobSystem());
    gaussianPass.SetFrameSlot(renderer.GetFrameSlot());
    const uint32_t effectiveShDegree =
        std::min(renderer.GetSettings().gaussianShDegree, renderer.GetScene().GetGaussianShDegree());
    gaussianPass.SetParams(renderer.GetSettings().gaussianOpacity,
        effectiveShDegree,
        renderer.GetSettings().gaussianViewDependentColor,
        renderer.GetSettings().gaussianAntialiasing,
        renderer.GetSettings().gaussianFastCulling);
    gaussianPass.SetDebugView(static_cast<uint32_t>(renderer.GetSettings().gaussianDebugView));
}

void ConfigurePathTracerPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& pathTracerPass = static_cast<PathTracerPass&>(pass);
    const auto& settings = renderer.GetSettings();
    pathTracerPass.SetOutput(resources.pathTraceOutput);
    pathTracerPass.SetDenoiserGuides(resources.pathTraceNormalGuide, resources.pathTraceDepthGuide);
    pathTracerPass.SetScene(&renderer.GetScene());
    pathTracerPass.SetCamera(&renderer.GetCamera());
    pathTracerPass.SetFrameIndex(renderer.GetPathTraceFrameIndex());
    pathTracerPass.SetFrameSlot(renderer.GetFrameSlot());
    pathTracerPass.SetEnabled(settings.enablePathTracing);
    pathTracerPass.SetBackendPreference(settings.pathTraceBackend);
    pathTracerPass.SetLight(settings.lightDirectionAndIntensity);
    pathTracerPass.SetEnvironment(glm::vec4(settings.environmentIntensity,
        glm::radians(settings.environmentRotationDegrees),
        static_cast<float>(settings.environmentPreset),
        settings.environmentDiffuseStrength));
    pathTracerPass.SetEnvironmentImage(renderer.GetEnvironmentSampledImageIndex());
    const glm::vec3 cameraRight = glm::normalize(glm::cross(renderer.GetCamera().GetForward(), renderer.GetCamera().GetUp()));
    pathTracerPass.SetLens(glm::vec4(cameraRight, settings.cameraApertureRadius),
        glm::vec4(renderer.GetCamera().GetUp(), settings.cameraFocalDistance));
    pathTracerPass.SetSamplesPerPixel(settings.pathTraceSamplesPerPixel);
    pathTracerPass.SetMaxBounces(settings.pathTraceMaxBounces);
    pathTracerPass.SetIntegratorControls(settings.pathTraceNextEventEstimation,
        settings.pathTraceRussianRoulette,
        settings.pathTraceRussianRouletteDepth,
        settings.pathTraceFireflyClamp);
    pathTracerPass.SetDebugView(settings.pathTraceDebugView);
}

void ConfigurePathDenoisePass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& denoisePass = static_cast<PathDenoisePass&>(pass);
    denoisePass.SetInput(resources.pathTraceOutput);
    denoisePass.SetGuides(resources.pathTraceNormalGuide, resources.pathTraceDepthGuide);
    denoisePass.SetOutput(resources.pathTraceDenoised);
    denoisePass.SetStrength(renderer.GetSettings().pathTraceDenoiserStrength);
    denoisePass.SetTemporalBlend(renderer.GetSettings().pathTraceDenoiserTemporalBlend);
    denoisePass.SetIterations(renderer.GetSettings().pathTraceDenoiserIterations);
    denoisePass.SetFrameIndex(renderer.GetPathTraceFrameIndex());
}

void ConfigureTemporalAAPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& temporalPass = static_cast<TemporalAAPass&>(pass);
    temporalPass.SetInputs(
        resources.deferredLighting,
        resources.gbufferNormal,
        resources.gbufferMaterial,
        resources.gbufferMotion,
        resources.gbufferReactive,
        resources.sceneDepth);
    temporalPass.SetOutput(resources.temporalLighting);
    temporalPass.SetEnabled(renderer.GetSettings().enableTaa || renderer.GetSettings().enableTemporalUpscaler);
    temporalPass.SetFeedback(renderer.GetSettings().taaFeedback);
    temporalPass.SetUpscalerSharpness(renderer.GetSettings().enableTemporalUpscaler ? renderer.GetSettings().temporalUpscalerSharpness : 0.0f);
    temporalPass.SetReactiveMask(
        renderer.GetSettings().temporalMaterialReactiveMask, renderer.GetSettings().temporalReactiveMaskStrength);
    temporalPass.SetFrameIndex(renderer.GetPathTraceFrameIndex());
    temporalPass.SetCameraMatrices(renderer.GetCamera().GetViewProjection(), renderer.GetCamera().GetInverseViewProjection());
    temporalPass.SetDebugView(renderer.GetSettings().debugView);
}

void ConfigureBloomPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& bloomPass = static_cast<BloomPass&>(pass);
    const GraphTextureHandle rasterInput = resources.temporalLighting ? resources.temporalLighting : resources.deferredLighting;
    const GraphTextureHandle pathTraceInput = resources.pathTraceDenoised ? resources.pathTraceDenoised : resources.pathTraceOutput;
    const GraphTextureHandle sourceInput = rasterInput ? rasterInput : pathTraceInput;

    if (bloomPass.Stage() == BloomPassStage::Extract) {
        bloomPass.SetInput(sourceInput);
        bloomPass.SetSecondaryInput({});
        bloomPass.SetOutput(resources.bloomHalf);
    } else if (bloomPass.Stage() == BloomPassStage::Downsample) {
        bloomPass.SetInput(resources.bloomHalf);
        bloomPass.SetSecondaryInput({});
        bloomPass.SetOutput(resources.bloomQuarter);
    } else {
        bloomPass.SetInput(resources.bloomHalf);
        bloomPass.SetSecondaryInput(resources.bloomQuarter);
        bloomPass.SetOutput(resources.bloomOutput);
    }
    bloomPass.SetParameters(renderer.GetSettings().bloomThreshold, renderer.GetSettings().bloomIntensity);
}

void ConfigureCompositePass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& compositePass = static_cast<CompositePass&>(pass);
    const GraphTextureHandle rasterInput = resources.temporalLighting ? resources.temporalLighting : resources.deferredLighting;
    const GraphTextureHandle pathTraceInput = resources.pathTraceDenoised ? resources.pathTraceDenoised : resources.pathTraceOutput;
    compositePass.SetInputs(rasterInput, pathTraceInput, resources.gaussianAccum, resources.gaussianReveal, resources.gaussianDebug);
    compositePass.SetGBufferInputs(
        resources.gbufferAlbedo,
        resources.gbufferNormal,
        resources.gbufferMaterial,
        resources.gbufferDebug,
        resources.gbufferMotion,
        resources.deferredLightingDebug,
        resources.sceneDepth);
    uint32_t gaussianTileRangeBufferIndex = kInvalidResourceIndex;
    if (const auto* officialGaussian = renderer.FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        gaussianTileRangeBufferIndex = officialGaussian->GetTileRangeBindlessStorageIndex();
    }
    const VkExtent2D extent = renderer.GetRenderDevice().GetSwapchainExtent();
    compositePass.SetGaussianDebugResources(
        gaussianTileRangeBufferIndex, (extent.width + 7u) / 8u, (extent.height + 7u) / 8u);
    compositePass.SetShadowMap(resources.shadowMap);
    compositePass.SetOverdraw(resources.overdraw);
    compositePass.SetBloomInput(resources.bloomOutput);
    compositePass.SetOutput(resources.swapchainTarget);
    compositePass.SetMode(static_cast<uint32_t>(renderer.GetSettings().displayMode),
        renderer.GetSettings().gaussianMix,
        static_cast<uint32_t>(renderer.GetSettings().debugView),
        static_cast<uint32_t>(renderer.GetSettings().gaussianDebugView));
    compositePass.SetCompare(static_cast<uint32_t>(renderer.GetSettings().compareMode),
        renderer.GetSettings().compareSplitPosition,
        renderer.GetSettings().compareDifferenceScale);
    compositePass.SetExposure(renderer.GetSettings().cameraExposureEv);
    compositePass.SetToneMapping(static_cast<uint32_t>(renderer.GetSettings().toneMappingMode));
    compositePass.SetPostProcess(renderer.GetSettings().colorGradingSaturation,
        renderer.GetSettings().colorGradingContrast,
        renderer.GetSettings().enableVignette,
        renderer.GetSettings().vignetteStrength,
        renderer.GetSettings().enableBloom,
        renderer.GetSettings().bloomThreshold,
        renderer.GetSettings().bloomIntensity,
        renderer.GetSettings().enableFxaa,
        renderer.GetSettings().enableMotionBlur,
        renderer.GetSettings().motionBlurStrength);
    compositePass.SetAmbientOcclusion(
        renderer.GetSettings().enableSsao, renderer.GetSettings().ssaoRadius, renderer.GetSettings().ssaoIntensity);
    compositePass.SetShadowCascadeDebug(renderer.GetSettings().shadowCascadeCount,
        renderer.GetSettings().shadowCascadeLambda,
        renderer.GetSettings().showShadowCascadeOverlay);
    compositePass.SetCameraMatrices(renderer.GetCamera().GetViewProjection(), renderer.GetCamera().GetInverseViewProjection());
    compositePass.SetDepthRange(renderer.GetCamera().GetNearPlane(), renderer.GetCamera().GetFarPlane());
}
} // namespace

TransientImageKey TransientImagePool::MakeKey(const ImageDesc& desc)
{
    return TransientImageKey{
        .extent = desc.extent,
        .format = desc.format,
        .usage = desc.usage,
        .aspectFlags = desc.aspectFlags,
        .initialLayout = desc.initialLayout,
        .memoryUsage = desc.memoryUsage,
        .mipLevels = desc.mipLevels,
        .arrayLayers = desc.arrayLayers,
    };
}

ImageHandle TransientImagePool::Acquire(RenderDevice& device, const ImageDesc& desc)
{
    const TransientImageKey key = MakeKey(desc);

    for (TransientImagePoolEntry& entry : _entries) {
        if (!entry.inUse && entry.key == key) {
            entry.inUse = true;
            return entry.handle;
        }
    }

    ImageHandle handle = device.CreateImage(desc);
    _entries.push_back(TransientImagePoolEntry{
        .handle = handle,
        .key = key,
        .inUse = true,
    });
    return handle;
}

void TransientImagePool::Release(ImageHandle handle)
{
    for (TransientImagePoolEntry& entry : _entries) {
        if (entry.handle == handle) {
            entry.inUse = false;
            return;
        }
    }
}

void TransientImagePool::Purge(RenderDevice& device)
{
    for (const TransientImagePoolEntry& entry : _entries) {
        if (entry.handle) {
            device.DestroyImage(entry.handle);
        }
    }
    _entries.clear();
}

bool Renderer::Initialize(SDL_Window* window, VkExtent2D initialExtent, bool enableValidation)
{
    _window = window;

    // RenderDevice owns Vulkan lifetime, while Renderer owns frame-level policy
    // such as presets, passes, transient resources, and camera/scene state.
    RenderDeviceDesc deviceDesc;
    deviceDesc.swapchainExtent = initialExtent;
    deviceDesc.enableValidation = enableValidation;
    deviceDesc.enableVSync = _settings.enableVSync;
    _device.Initialize(window, deviceDesc);
    CreateIblBrdfLut();
    _jobs.Initialize();
    ApplyPreset(RendererPreset::Recommended);

    _camera.SetViewport(initialExtent.width, initialExtent.height);
    InitializeCommands();
    InitializeSyncStructures();
    InitializeDefaultPasses();
    return true;
}

void Renderer::Shutdown()
{
    if (_sceneLoadFuture.valid()) {
        _sceneLoadFuture.wait();
        _sceneLoadFuture = {};
    }
    if (_visibilityFuture.valid()) {
        _visibilityFuture.wait();
        _visibilityFuture = {};
    }
    _sceneLoadInProgress = false;
    _visibilityCullInProgress = false;
    _sceneLoadStatus = {};

    _device.WaitIdle();
    ClearPassRegistry();
    DestroyFrameResources();
    _transientImagePool.Purge(_device);
    _pendingSceneUpload.scene.DestroyGpu(_device);
    _pendingSceneUpload = {};
    _scene.DestroyGpu(_device);
    DestroyIblResources();
    DestroyDdgiResources();
    DestroyRestirResources();
    for (RetiredSceneEntry& retiredScene : _retiredScenes) {
        retiredScene.scene.DestroyGpu(_device);
    }
    _retiredScenes.clear();
    _jobs.Shutdown();
    _device.Shutdown();
    _window = nullptr;
}

void Renderer::HandleEvent(const SDL_Event& event)
{
    _camera.HandleEvent(event);

    if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        _camera.SetViewport(static_cast<uint32_t>(event.window.data1), static_cast<uint32_t>(event.window.data2));
        _pathTraceFrameIndex = 0;
        return;
    }

    if (event.type != SDL_KEYDOWN || event.key.repeat != 0) {
        if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
            const bool selectedLight = _selection.kind == SelectionKind::DirectionalLight || _selection.kind == SelectionKind::PointLight
                || _selection.kind == SelectionKind::SpotLight || _selection.kind == SelectionKind::AreaLight;
            if (!selectedLight) {
                _selection = PickSelection(glm::vec2(static_cast<float>(event.button.x), static_cast<float>(event.button.y)));
            }
            _selectionDragging = _selection.kind != SelectionKind::None;
            _selectionEditedSinceDragStart = false;
            _lastDragMousePosition = glm::vec2(static_cast<float>(event.button.x), static_cast<float>(event.button.y));

            if (_selection.kind == SelectionKind::Object && _selection.objectIndex < _scene.GetObjects().size()) {
                const auto& object = _scene.GetObjects()[_selection.objectIndex];
                _dragPlaneOrigin = object.bounds.center;
                _dragPlaneNormal = _camera.GetForward();
                _dragGrabOffset = object.bounds.center - _dragPlaneOrigin;
            } else if (_selection.kind == SelectionKind::DirectionalLight) {
                _dragPlaneOrigin = _scene.GetBounds().center;
                _dragPlaneNormal = _camera.GetForward();
                _dragGrabOffset = glm::vec3(0.0f);
            } else if (_selection.kind == SelectionKind::PointLight) {
                _dragPlaneOrigin = glm::vec3(_settings.pointLightPositionAndIntensity);
                _dragPlaneNormal = _camera.GetForward();
                _dragGrabOffset = glm::vec3(0.0f);
            } else if (_selection.kind == SelectionKind::SpotLight) {
                _dragPlaneOrigin = glm::vec3(_settings.spotLightPositionAndIntensity);
                _dragPlaneNormal = _camera.GetForward();
                _dragGrabOffset = glm::vec3(0.0f);
            } else if (_selection.kind == SelectionKind::AreaLight) {
                _dragPlaneOrigin = glm::vec3(_settings.areaLightPositionAndIntensity);
                _dragPlaneNormal = _camera.GetForward();
                _dragGrabOffset = glm::vec3(0.0f);
            }
            return;
        }

        if (event.type == SDL_MOUSEMOTION && _selectionDragging) {
            UpdateSceneEditDrag(glm::vec2(static_cast<float>(event.motion.x), static_cast<float>(event.motion.y)));
            return;
        }

        if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
            EndSceneEditDrag();
            return;
        }
        return;
    }

    switch (event.key.keysym.sym) {
    case SDLK_1:
        _settings.displayMode = RendererDisplayMode::DeferredLighting;
        break;
    case SDLK_2:
        _settings.displayMode = RendererDisplayMode::Gaussian;
        break;
    case SDLK_3:
        _settings.displayMode = RendererDisplayMode::PathTrace;
        break;
    case SDLK_4:
        _settings.displayMode = RendererDisplayMode::Composite;
        break;
    case SDLK_g:
        _settings.enableGaussian = !_settings.enableGaussian;
        break;
    case SDLK_p:
        _settings.enablePathTracing = !_settings.enablePathTracing;
        break;
    case SDLK_r:
        _settings.enableRaster = !_settings.enableRaster;
        break;
    case SDLK_ESCAPE:
        ClearSelection();
        break;
    case SDLK_l:
        SelectDirectionalLight();
        break;
    default:
        break;
    }

    ResetAccumulation();
}

void Renderer::Update(float deltaSeconds)
{
    _sceneLoadStatus.pendingUploadBytes = static_cast<uint64_t>(_device.GetUploadBatchStats().pendingBytes);
    _sceneLoadStatus.pendingUploadCopies = _device.GetUploadBatchStats().pendingCopies;
    PumpSceneLoadRequests();
    PumpPendingSceneUpload();
    PumpVisibilityResults();

    _frameTimeMs = deltaSeconds * 1000.0f;
    _smoothedFrameTimeMs = _smoothedFrameTimeMs <= 0.0f ? _frameTimeMs : (_smoothedFrameTimeMs * 0.9f + _frameTimeMs * 0.1f);
    if (_settings.animationPlaying) {
        const float scaledDelta = deltaSeconds * std::clamp(_settings.animationTimeScale, 0.0f, 8.0f);
        _settings.animationTimeSeconds += scaledDelta;
        if (_settings.animateDirectionalLight) {
            const float t = _settings.animationTimeSeconds * 0.65f;
            const glm::vec3 direction = glm::normalize(glm::vec3(std::cos(t) * 0.65f, -1.0f, std::sin(t) * 0.65f));
            _settings.lightDirectionAndIntensity = glm::vec4(direction, _settings.lightDirectionAndIntensity.w);
            _pathTraceFrameIndex = 0;
        }
        if (_settings.animateEnvironment) {
            _settings.environmentRotationDegrees = std::fmod(_settings.environmentRotationDegrees + scaledDelta * 12.0f, 360.0f);
            _pathTraceFrameIndex = 0;
        }
    }
    if (_settings.frameTimingCapture || _settings.benchmarkOverlay) {
        _frameTimeHistoryMs[_frameTimeHistoryHead] = _frameTimeMs;
        _frameTimeHistoryHead = (_frameTimeHistoryHead + 1) % _frameTimeHistoryMs.size();
        _frameTimeHistoryCount = std::min(_frameTimeHistoryCount + 1, _frameTimeHistoryMs.size());
    } else {
        _frameTimeHistoryHead = 0;
        _frameTimeHistoryCount = 0;
        _frameTimeHistoryMs.fill(0.0f);
    }

    if (!_camera.IsOrbitEnabled()) {
        _trackSelectedObjectOrbit = false;
    }

    if (_trackSelectedObjectOrbit) {
        const auto& objects = _scene.GetObjects();
        if (_selection.kind == SelectionKind::Object && _selection.objectIndex < objects.size()) {
            _camera.SetOrbitTarget(objects[_selection.objectIndex].bounds.center);
        } else {
            _trackSelectedObjectOrbit = false;
        }
    }

    _camera.Update(deltaSeconds);
    // Progressive path tracing only makes sense while the viewpoint is stable.
    // As soon as the camera moves, old samples become history from the wrong camera.
    if (_camera.ConsumeMoved()) {
        _pathTraceFrameIndex = 0;
        _visibilityDirty = true;
        if (_scene.HasTrainedGaussians()) {
            _gaussianInteractivePreviewFramesRemaining = GaussianInteractivePreviewFrameBudget(_scene);
        }
        if (!_scene.HasTrainedGaussians() && _scene.SupportsRealtimeGaussianSorting()) {
            _scene.ResortGaussians(_device, _camera);
        }
    } else {
        ++_pathTraceFrameIndex;
        if (_gaussianInteractivePreviewFramesRemaining > 0) {
            --_gaussianInteractivePreviewFramesRemaining;
        }
    }

    DispatchVisibilityCullIfNeeded();
}

void Renderer::RenderFrame()
{
    RendererFrameContext& currentFrame = GetCurrentFrame();

    // Each overlapping frame owns its own fence. Waiting here guarantees the GPU
    // has finished using the command buffer and transient resources we are about to recycle.
    VK_CHECK(vkWaitForFences(_device.GetDevice(), 1, &currentFrame.renderFence, VK_TRUE, std::numeric_limits<uint64_t>::max()));
    ProcessCompletedFrameReadback(currentFrame);
    ReleaseRetiredScenes();
    ReleaseTransientResources(currentFrame);

    uint32_t swapchainImageIndex = 0;
    VkResult acquireResult = vkAcquireNextImageKHR(_device.GetDevice(),
        _device.GetSwapchain(),
        std::numeric_limits<uint64_t>::max(),
        currentFrame.acquireSemaphore,
        VK_NULL_HANDLE,
        &swapchainImageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(acquireResult);
    }

    VkSemaphore renderSemaphore = _swapchainImageRenderSemaphores.at(swapchainImageIndex);

    VK_CHECK(vkResetFences(_device.GetDevice(), 1, &currentFrame.renderFence));
    VK_CHECK(vkResetCommandBuffer(currentFrame.commandBuffer, 0));

    VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(currentFrame.commandBuffer, &beginInfo));

    // Build the logical frame graph first, then execute it. This keeps pass code
    // focused on "what it needs" instead of hand-written global barriers.
    EnsureDdgiResources();
    EnsureRestirResources();
    RenderGraph graph = BuildFrameGraph(swapchainImageIndex);
    RenderGraphExecutionContext executionContext{
        .device = _device,
        .frameContext = currentFrame,
        .transientImagePool = _transientImagePool,
        .commandBuffer = currentFrame.commandBuffer,
        .passTimings = &_lastRenderGraphTimings,
        .gpuTimestampsSupported = _renderGraphTimestampsSupported,
        .timestampPeriodNs = _timestampPeriodNs,
    };
    graph.Execute(executionContext);
    RecordOverlay(currentFrame.commandBuffer, swapchainImageIndex);
    RecordScreenshotReadback(currentFrame.commandBuffer, currentFrame, swapchainImageIndex);

    VK_CHECK(vkEndCommandBuffer(currentFrame.commandBuffer));

    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(currentFrame.commandBuffer);
    VkSemaphoreSubmitInfo waitInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, currentFrame.acquireSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, renderSemaphore);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(_device.GetGraphicsQueue(), 1, &submitInfo, currentFrame.renderFence));

    VkSwapchainKHR swapchain = _device.GetSwapchain();
    VkPresentInfoKHR presentInfo = vkinit::present_info();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderSemaphore;
    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_device.GetPresentQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        RecreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        VK_CHECK(presentResult);
    }

    ++_frameNumber;
}

void Renderer::SetOverlayCallbacks(OverlayDrawFn drawFn, OverlaySwapchainCallback swapchainCallback)
{
    _overlayDrawFn = std::move(drawFn);
    _overlaySwapchainCallback = std::move(swapchainCallback);
}

void Renderer::ClearOverlayCallbacks()
{
    _overlayDrawFn = {};
    _overlaySwapchainCallback = {};
}

PathTraceBackend Renderer::GetActivePathTraceBackend() const
{
    const auto* pathTracerPass = FindPass<PathTracerPass>("path-tracer");
    return pathTracerPass != nullptr ? pathTracerPass->GetActiveBackend() : PathTraceBackend::Compute;
}

RendererPreset Renderer::GetRecommendedPreset() const
{
    return ChooseRecommendedPreset(_device);
}

uint32_t Renderer::GetOfficialGaussianProjectedCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().projectedCount;
    }
    return 0;
}

uint32_t Renderer::GetOfficialGaussianDuplicateCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().duplicateCount;
    }
    return 0;
}

uint32_t Renderer::GetOfficialGaussianPaddedDuplicateCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().paddedDuplicateCount;
    }
    return 0;
}

uint32_t Renderer::GetOfficialGaussianTileCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().tileCount;
    }
    return 0;
}

float Renderer::GetOfficialGaussianAverageTilesTouched() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().averageTilesTouched;
    }
    return 0.0f;
}

uint64_t Renderer::GetOfficialGaussianRebuildCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().rebuildCount;
    }
    return 0;
}

float Renderer::GetOfficialGaussianPreprocessMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().preprocessMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianScanMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().scanMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianDuplicateMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().duplicateMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianSortMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().sortMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianRangeMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().rangeMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianRasterMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().rasterMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianTotalBuildMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().totalBuildMs;
    }
    return 0.0f;
}

vesta::scene::SceneKind Renderer::GetRecommendedSceneKind() const
{
    return _scene.GetSceneKind();
}

RendererDisplayMode Renderer::GetRecommendedDisplayModeForScene() const
{
    switch (_scene.GetSceneKind()) {
    case vesta::scene::SceneKind::Gaussian:
    case vesta::scene::SceneKind::PointCloud:
        return RendererDisplayMode::Gaussian;
    case vesta::scene::SceneKind::Mesh:
    case vesta::scene::SceneKind::Empty:
    default:
        return RendererDisplayMode::DeferredLighting;
    }
}

std::string Renderer::GetSelectionLabel() const
{
    switch (_selection.kind) {
    case SelectionKind::Object: {
        const auto& objects = _scene.GetObjects();
        if (_selection.objectIndex < objects.size()) {
            return objects[_selection.objectIndex].name;
        }
        return "Object";
    }
    case SelectionKind::DirectionalLight:
        return "Directional Light";
    case SelectionKind::PointLight:
        return "Point Light";
    case SelectionKind::SpotLight:
        return "Spot Light";
    case SelectionKind::AreaLight:
        return "Area Light";
    case SelectionKind::None:
    default:
        return "None";
    }
}

void Renderer::ApplyPreset(RendererPreset preset)
{
    ApplyPresetSettings(_settings, _device, preset);
    ResetAccumulation();
}

void Renderer::SelectDirectionalLight()
{
    _selection = EditorSelection{
        .kind = SelectionKind::DirectionalLight,
        .objectIndex = 0,
    };
    _trackSelectedObjectOrbit = false;
}

void Renderer::SelectPointLight()
{
    _selection = EditorSelection{
        .kind = SelectionKind::PointLight,
        .objectIndex = 0,
    };
    _trackSelectedObjectOrbit = false;
}

void Renderer::SelectSpotLight()
{
    _selection = EditorSelection{
        .kind = SelectionKind::SpotLight,
        .objectIndex = 0,
    };
    _trackSelectedObjectOrbit = false;
}

void Renderer::SelectAreaLight()
{
    _selection = EditorSelection{
        .kind = SelectionKind::AreaLight,
        .objectIndex = 0,
    };
    _trackSelectedObjectOrbit = false;
}

bool Renderer::SelectObject(uint32_t objectIndex)
{
    if (objectIndex >= _scene.GetObjects().size()) {
        return false;
    }

    _selection = EditorSelection{
        .kind = SelectionKind::Object,
        .objectIndex = objectIndex,
    };
    _selectionDragging = false;
    _selectionEditedSinceDragStart = false;
    return true;
}

bool Renderer::SetSelectedObjectPosition(glm::vec3 position)
{
    const auto& objects = _scene.GetObjects();
    if (_selection.kind != SelectionKind::Object || _selection.objectIndex >= objects.size()) {
        return false;
    }

    const glm::vec3 delta = position - objects[_selection.objectIndex].GetTranslation();
    if (!_scene.TranslateObject(_device, _selection.objectIndex, delta)) {
        return false;
    }

    const bool rebuildRayTracing = _scene.HasRayTracingScene() && _settings.enablePathTracing
        && GetActivePathTraceBackend() == PathTraceBackend::HardwareRT;
    OnSceneEdited(rebuildRayTracing);
    return true;
}

bool Renderer::RotateSelectedObject(glm::vec3 eulerDeltaDegrees)
{
    const auto& objects = _scene.GetObjects();
    if (_selection.kind != SelectionKind::Object || _selection.objectIndex >= objects.size()) {
        return false;
    }
    if (glm::dot(eulerDeltaDegrees, eulerDeltaDegrees) <= 1.0e-8f) {
        return true;
    }

    const glm::quat rotationDelta(glm::radians(eulerDeltaDegrees));
    if (!_scene.RotateObject(_device, _selection.objectIndex, rotationDelta)) {
        return false;
    }

    const bool rebuildRayTracing = _scene.HasRayTracingScene() && _settings.enablePathTracing
        && GetActivePathTraceBackend() == PathTraceBackend::HardwareRT;
    OnSceneEdited(rebuildRayTracing);
    return true;
}

bool Renderer::ScaleSelectedObject(float uniformScale)
{
    const auto& objects = _scene.GetObjects();
    if (_selection.kind != SelectionKind::Object || _selection.objectIndex >= objects.size()) {
        return false;
    }
    if (!_scene.ScaleObject(_device, _selection.objectIndex, uniformScale)) {
        return false;
    }

    const bool rebuildRayTracing = _scene.HasRayTracingScene() && _settings.enablePathTracing
        && GetActivePathTraceBackend() == PathTraceBackend::HardwareRT;
    OnSceneEdited(rebuildRayTracing);
    return true;
}

bool Renderer::UpdateMaterial(uint32_t materialIndex, const vesta::scene::SceneMaterial& material)
{
    if (!_scene.UpdateMaterial(_device, materialIndex, material)) {
        return false;
    }

    ResetAccumulation();
    return true;
}

void Renderer::ClearSelection()
{
    _selection = {};
    _selectionDragging = false;
    _selectionEditedSinceDragStart = false;
    _trackSelectedObjectOrbit = false;
}

bool Renderer::OrbitCameraAroundSelection()
{
    const auto& objects = _scene.GetObjects();
    if (_selection.kind != SelectionKind::Object || _selection.objectIndex >= objects.size()) {
        return false;
    }

    const auto& object = objects[_selection.objectIndex];
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), object.bounds.center), object.bounds.radius);
    _camera.EnableOrbit(object.bounds.center, distance);
    _trackSelectedObjectOrbit = true;
    ResetAccumulation();
    _visibilityDirty = true;
    return true;
}

void Renderer::OrbitCameraAroundScene()
{
    const auto& bounds = _scene.GetBounds();
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), bounds.center), bounds.radius);
    _camera.EnableOrbit(bounds.center, distance);
    _trackSelectedObjectOrbit = false;
    ResetAccumulation();
    _visibilityDirty = true;
}

bool Renderer::DollyCameraAroundSelection()
{
    const auto& objects = _scene.GetObjects();
    if (_selection.kind != SelectionKind::Object || _selection.objectIndex >= objects.size()) {
        return false;
    }

    const auto& object = objects[_selection.objectIndex];
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), object.bounds.center), object.bounds.radius);
    _camera.EnableDollyOrbit(object.bounds.center, distance, _camera.GetDollySpeedDegrees());
    _trackSelectedObjectOrbit = true;
    ResetAccumulation();
    _visibilityDirty = true;
    return true;
}

bool Renderer::EnsureRayTracingScene()
{
    if (!_device.IsRayTracingSupported() || _scene.HasRayTracingScene() || _scene.GetIndices().empty() || !_scene.GetVertexBuffer()) {
        return _scene.HasRayTracingScene();
    }

    _sceneLoadStatus.lastBlockingWait = "Scene::RebuildRayTracing";
    _device.SetDebugWaitContext("scene=" + _scene.GetSourcePath().string() + " stage=EnsureRayTracingScene");
    const bool built = _scene.RebuildRayTracing(_device);
    _sceneLoadStatus.blasMs = _scene.GetBottomLevelBuildMs();
    _sceneLoadStatus.tlasMs = _scene.GetTopLevelBuildMs();
    _sceneLoadStatus.lastBlockingWait.clear();
    ResetAccumulation();
    return built;
}

void Renderer::DollyCameraAroundScene()
{
    const auto& bounds = _scene.GetBounds();
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), bounds.center), bounds.radius);
    _camera.EnableDollyOrbit(bounds.center, distance, _camera.GetDollySpeedDegrees());
    _trackSelectedObjectOrbit = false;
    ResetAccumulation();
    _visibilityDirty = true;
}

void Renderer::DisableCameraOrbit()
{
    _camera.DisableOrbit();
    _trackSelectedObjectOrbit = false;
    ResetAccumulation();
}

std::pair<glm::vec3, glm::vec3> Renderer::ComputeMouseRay(glm::vec2 mousePosition) const
{
    const VkExtent2D extent = _device.GetSwapchainExtent();
    const glm::vec2 viewportSize(
        std::max(1.0f, static_cast<float>(extent.width)), std::max(1.0f, static_cast<float>(extent.height)));
    const glm::vec2 ndc(
        (mousePosition.x / viewportSize.x) * 2.0f - 1.0f,
        (mousePosition.y / viewportSize.y) * 2.0f - 1.0f);

    const glm::vec4 nearPoint = _camera.GetInverseViewProjection() * glm::vec4(ndc.x, ndc.y, 0.0f, 1.0f);
    const glm::vec4 farPoint = _camera.GetInverseViewProjection() * glm::vec4(ndc.x, ndc.y, 1.0f, 1.0f);
    const glm::vec3 worldNear = glm::vec3(nearPoint) / std::max(nearPoint.w, 1.0e-4f);
    const glm::vec3 worldFar = glm::vec3(farPoint) / std::max(farPoint.w, 1.0e-4f);
    return { _camera.GetPosition(), glm::normalize(worldFar - worldNear) };
}

EditorSelection Renderer::PickSelection(glm::vec2 mousePosition) const
{
    const auto [rayOrigin, rayDirection] = ComputeMouseRay(mousePosition);
    if (const std::optional<uint32_t> objectIndex = _scene.PickObject(rayOrigin, rayDirection); objectIndex.has_value()) {
        return EditorSelection{
            .kind = SelectionKind::Object,
            .objectIndex = *objectIndex,
        };
    }
    return {};
}

void Renderer::OnSceneEdited(bool rebuildRayTracing)
{
    ResetAccumulation();
    _visibilityDirty = true;
    _visibleSurfaceIndices.clear();
    _visibleSceneToken.reset();
    _frameSnapshot = {};
    if (_scene.HasTrainedGaussians()) {
        _gaussianInteractivePreviewFramesRemaining = GaussianInteractivePreviewFrameBudget(_scene);
    }
    if (!_scene.HasTrainedGaussians() && _scene.SupportsRealtimeGaussianSorting()) {
        _scene.ResortGaussians(_device, _camera);
    }

    if (rebuildRayTracing && _scene.HasRayTracingScene()) {
        _scene.RebuildRayTracing(_device);
    }
}

void Renderer::UpdateSceneEditDrag(const glm::vec2& mousePosition)
{
    if (!_selectionDragging || _selection.kind == SelectionKind::None) {
        return;
    }

    const auto [rayOrigin, rayDirection] = ComputeMouseRay(mousePosition);
    if (_selection.kind == SelectionKind::Object) {
        const auto& objects = _scene.GetObjects();
        if (_selection.objectIndex >= objects.size()) {
            return;
        }

        const vesta::scene::SceneObject& object = objects[_selection.objectIndex];
        const float denominator = glm::dot(rayDirection, _dragPlaneNormal);
        if (std::abs(denominator) < 1.0e-4f) {
            return;
        }

        const float t = glm::dot(object.bounds.center - rayOrigin, _dragPlaneNormal) / denominator;
        if (t <= 0.0f) {
            return;
        }

        const glm::vec3 hitPoint = rayOrigin + rayDirection * t;
        const glm::vec3 delta = hitPoint - object.bounds.center;
        if (_scene.TranslateObject(_device, _selection.objectIndex, delta)) {
            _selectionEditedSinceDragStart = true;
            OnSceneEdited(false);
        }
        return;
    }

    if (_selection.kind == SelectionKind::DirectionalLight) {
        const glm::vec2 delta = mousePosition - _lastDragMousePosition;
        glm::vec3 cameraRight = glm::cross(_camera.GetForward(), glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::length(cameraRight) < 1.0e-4f) {
            cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            cameraRight = glm::normalize(cameraRight);
        }
        glm::vec3 direction = glm::normalize(-glm::vec3(_settings.lightDirectionAndIntensity));
        direction = glm::normalize(
            direction + _camera.GetForward() * (-delta.y * 0.01f) + cameraRight * (-delta.x * 0.01f));
        _settings.lightDirectionAndIntensity = glm::vec4(-direction, _settings.lightDirectionAndIntensity.w);
        _selectionEditedSinceDragStart = true;
        OnSceneEdited(false);
    }

    if (_selection.kind == SelectionKind::PointLight || _selection.kind == SelectionKind::SpotLight
        || _selection.kind == SelectionKind::AreaLight) {
        const float denominator = glm::dot(rayDirection, _dragPlaneNormal);
        if (std::abs(denominator) < 1.0e-4f) {
            _lastDragMousePosition = mousePosition;
            return;
        }
        const float t = glm::dot(_dragPlaneOrigin - rayOrigin, _dragPlaneNormal) / denominator;
        if (t <= 0.0f) {
            _lastDragMousePosition = mousePosition;
            return;
        }
        const glm::vec3 hitPoint = rayOrigin + rayDirection * t;
        if (_selection.kind == SelectionKind::PointLight) {
            _settings.pointLightPositionAndIntensity = glm::vec4(hitPoint, _settings.pointLightPositionAndIntensity.w);
        } else if (_selection.kind == SelectionKind::SpotLight) {
            _settings.spotLightPositionAndIntensity = glm::vec4(hitPoint, _settings.spotLightPositionAndIntensity.w);
        } else {
            _settings.areaLightPositionAndIntensity = glm::vec4(hitPoint, _settings.areaLightPositionAndIntensity.w);
        }
        _dragPlaneOrigin = hitPoint;
        _selectionEditedSinceDragStart = true;
        OnSceneEdited(false);
    }

    _lastDragMousePosition = mousePosition;
}

void Renderer::EndSceneEditDrag()
{
    if (_selectionDragging && _selectionEditedSinceDragStart) {
        const bool rebuildRayTracing = _selection.kind == SelectionKind::Object && _scene.HasRayTracingScene()
            && _settings.enablePathTracing && GetActivePathTraceBackend() == PathTraceBackend::HardwareRT;
        OnSceneEdited(rebuildRayTracing);
    }
    _selectionDragging = false;
    _selectionEditedSinceDragStart = false;
}

SceneUploadOptions Renderer::GetSceneUploadOptions() const
{
    return SceneUploadOptions{
        .useDeviceLocalSceneBuffers = _settings.useDeviceLocalSceneBuffers,
        .buildRayTracingStructuresOnLoad = _settings.buildRayTracingStructuresOnLoad,
        .textureStreamingEnabled = _settings.textureStreamingEnabled,
        .useDeviceLocalTextures = _settings.useDeviceLocalTextures,
    };
}

bool Renderer::LoadScene(const std::filesystem::path& path)
{
    if (path.empty()) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene path is empty.";
        return false;
    }

    if (_sceneLoadInProgress) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene load already in progress.";
        return false;
    }

    const std::filesystem::path resolvedPath = vkutil::resolve_runtime_path(path);
    return LoadSceneResolved(resolvedPath);
}

bool Renderer::LoadSceneAsync(const std::filesystem::path& path)
{
    if (path.empty()) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene path is empty.";
        return false;
    }

    PumpSceneLoadRequests();
    if (_sceneLoadInProgress) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene load already in progress.";
        return false;
    }

    const std::filesystem::path resolvedPath = vkutil::resolve_runtime_path(path);
    _sceneLoadStatus = SceneLoadStatus{
        .state = SceneLoadState::Parsing,
        .path = resolvedPath,
        .message = "Parsing and preparing " + resolvedPath.filename().string() + "...",
    };
    _sceneLoadInProgress = true;
    _sceneLoadFuture = _jobs.Submit(vesta::core::JobPriority::Background, [resolvedPath]() {
        AsyncSceneLoadResult result;
        result.path = resolvedPath;
        const auto parseStart = std::chrono::steady_clock::now();

        try {
            vesta::scene::Scene loadedScene;
            result.success = loadedScene.ParseFromFile(resolvedPath);
            result.parseMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();
            if (result.success) {
                const auto prepareStart = std::chrono::steady_clock::now();
                result.success = loadedScene.PrepareParsedScene();
                result.prepareMs =
                    std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prepareStart).count();
            }
            if (result.success) {
                result.scene = std::move(loadedScene);
            } else {
                result.errorMessage = result.prepareMs > 0.0f ? "Failed to prepare scene file." : "Failed to parse scene file.";
            }
        } catch (const std::exception& exception) {
            result.errorMessage = exception.what();
        } catch (...) {
            result.errorMessage = "Unknown scene loading error.";
        }

        if (result.parseMs <= 0.0f) {
            result.parseMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();
        }
        return result;
    });
    return true;
}

bool Renderer::ReloadSceneAsync()
{
    const std::filesystem::path currentPath = _scene.GetSourcePath();
    if (currentPath.empty()) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "No scene to reload.";
        return false;
    }

    return LoadSceneAsync(currentPath);
}

void Renderer::CreateIblBrdfLut()
{
    if (_iblBrdfLutImage || _device.GetDevice() == VK_NULL_HANDLE) {
        return;
    }

    constexpr uint32_t kLutSize = 256u;
    std::vector<glm::vec2> lut(kLutSize * kLutSize);
    for (uint32_t y = 0; y < kLutSize; ++y) {
        const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(kLutSize);
        for (uint32_t x = 0; x < kLutSize; ++x) {
            const float nDotV = (static_cast<float>(x) + 0.5f) / static_cast<float>(kLutSize);
            lut[static_cast<size_t>(y) * kLutSize + x] = IntegrateBrdf(nDotV, roughness);
        }
    }

    _iblBrdfLutImage = _device.CreateImage(ImageDesc{
        .extent = VkExtent3D{ kLutSize, kLutSize, 1u },
        .format = VK_FORMAT_R32G32_SFLOAT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .registerBindlessSampled = true,
        .debugName = "IBL_BRDF_LUT",
    });
    _device.UploadImageData(_iblBrdfLutImage,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(lut.data()), lut.size() * sizeof(glm::vec2)));
    _iblBrdfLutSampledImageIndex = _device.GetImageResource(_iblBrdfLutImage).bindless.sampledImage;
}

void Renderer::CreateEnvironmentCubemapAtlas(std::span<const float> rgbaPixels, uint32_t width, uint32_t height)
{
    if (_device.GetDevice() == VK_NULL_HANDLE || rgbaPixels.empty() || width == 0u || height == 0u) {
        return;
    }
    if (_iblEnvironmentCubemapImage) {
        _device.DestroyImage(_iblEnvironmentCubemapImage);
        _iblEnvironmentCubemapImage = {};
        _iblEnvironmentCubemapSampledImageIndex = kInvalidResourceIndex;
    }

    constexpr uint32_t kFaceSize = 128u;
    constexpr uint32_t kAtlasColumns = 3u;
    constexpr uint32_t kAtlasRows = 2u;
    constexpr uint32_t kAtlasWidth = kFaceSize * kAtlasColumns;
    constexpr uint32_t kAtlasHeight = kFaceSize * kAtlasRows;
    std::vector<float> atlas(static_cast<size_t>(kAtlasWidth) * kAtlasHeight * 4u, 1.0f);
    for (uint32_t face = 0; face < 6u; ++face) {
        const uint32_t faceOffsetX = (face % kAtlasColumns) * kFaceSize;
        const uint32_t faceOffsetY = (face / kAtlasColumns) * kFaceSize;
        for (uint32_t y = 0; y < kFaceSize; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kFaceSize);
            for (uint32_t x = 0; x < kFaceSize; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kFaceSize);
                const glm::vec3 value = SampleEquirect(rgbaPixels, width, height, DirectionFromCubeFace(face, u, v));
                const size_t offset = (static_cast<size_t>(faceOffsetY + y) * kAtlasWidth + (faceOffsetX + x)) * 4u;
                atlas[offset + 0u] = value.r;
                atlas[offset + 1u] = value.g;
                atlas[offset + 2u] = value.b;
                atlas[offset + 3u] = 1.0f;
            }
        }
    }

    _iblEnvironmentCubemapImage = _device.CreateImage(ImageDesc{
        .extent = VkExtent3D{ kAtlasWidth, kAtlasHeight, 1u },
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .registerBindlessSampled = true,
        .debugName = "IblEnvironmentCubemapAtlas",
    });
    _device.UploadImageData(_iblEnvironmentCubemapImage,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(atlas.data()), atlas.size() * sizeof(float)));
    _iblEnvironmentCubemapSampledImageIndex = _device.GetImageResource(_iblEnvironmentCubemapImage).bindless.sampledImage;
}

void Renderer::CreateDiffuseIrradianceEquirect(std::span<const float> rgbaPixels, uint32_t width, uint32_t height)
{
    if (_device.GetDevice() == VK_NULL_HANDLE || rgbaPixels.empty() || width == 0u || height == 0u) {
        return;
    }
    if (_iblDiffuseIrradianceImage) {
        _device.DestroyImage(_iblDiffuseIrradianceImage);
        _iblDiffuseIrradianceImage = {};
        _iblDiffuseIrradianceSampledImageIndex = kInvalidResourceIndex;
    }

    constexpr uint32_t kIrradianceWidth = 64u;
    constexpr uint32_t kIrradianceHeight = 32u;
    constexpr uint32_t kSampleCount = 96u;
    std::vector<float> irradiance(static_cast<size_t>(kIrradianceWidth) * kIrradianceHeight * 4u, 1.0f);
    for (uint32_t y = 0; y < kIrradianceHeight; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kIrradianceHeight);
        for (uint32_t x = 0; x < kIrradianceWidth; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kIrradianceWidth);
            const glm::vec3 normal = DirectionFromEquirectUv(u, v);
            glm::vec3 sum(0.0f);
            for (uint32_t sampleIndex = 0; sampleIndex < kSampleCount; ++sampleIndex) {
                const glm::vec3 sampleDirection = TangentToWorld(CosineSampleHemisphere(sampleIndex, kSampleCount), normal);
                sum += SampleEquirect(rgbaPixels, width, height, sampleDirection);
            }
            const glm::vec3 value = sum / static_cast<float>(kSampleCount);
            const size_t offset = (static_cast<size_t>(y) * kIrradianceWidth + x) * 4u;
            irradiance[offset + 0u] = value.r;
            irradiance[offset + 1u] = value.g;
            irradiance[offset + 2u] = value.b;
            irradiance[offset + 3u] = 1.0f;
        }
    }

    _iblDiffuseIrradianceImage = _device.CreateImage(ImageDesc{
        .extent = VkExtent3D{ kIrradianceWidth, kIrradianceHeight, 1u },
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .registerBindlessSampled = true,
        .debugName = "IblDiffuseIrradianceEquirect",
    });
    _device.UploadImageData(_iblDiffuseIrradianceImage,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(irradiance.data()), irradiance.size() * sizeof(float)));
    _iblDiffuseIrradianceSampledImageIndex = _device.GetImageResource(_iblDiffuseIrradianceImage).bindless.sampledImage;
}

void Renderer::CreateSpecularPrefilterEquirectAtlas(std::span<const float> rgbaPixels, uint32_t width, uint32_t height)
{
    if (_device.GetDevice() == VK_NULL_HANDLE || rgbaPixels.empty() || width == 0u || height == 0u) {
        return;
    }
    if (_iblSpecularPrefilterImage) {
        _device.DestroyImage(_iblSpecularPrefilterImage);
        _iblSpecularPrefilterImage = {};
        _iblSpecularPrefilterSampledImageIndex = kInvalidResourceIndex;
    }

    constexpr uint32_t kPrefilterWidth = 128u;
    constexpr uint32_t kPrefilterHeight = 64u;
    constexpr uint32_t kRoughnessLevels = 5u;
    constexpr uint32_t kSampleCount = 64u;
    std::vector<float> prefilter(static_cast<size_t>(kPrefilterWidth) * kPrefilterHeight * kRoughnessLevels * 4u, 1.0f);
    for (uint32_t level = 0; level < kRoughnessLevels; ++level) {
        const float roughness = static_cast<float>(level) / static_cast<float>(kRoughnessLevels - 1u);
        for (uint32_t y = 0; y < kPrefilterHeight; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kPrefilterHeight);
            for (uint32_t x = 0; x < kPrefilterWidth; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kPrefilterWidth);
                const glm::vec3 reflection = DirectionFromEquirectUv(u, v);
                glm::vec3 sum(0.0f);
                float weightSum = 0.0f;
                for (uint32_t sampleIndex = 0; sampleIndex < kSampleCount; ++sampleIndex) {
                    const glm::vec3 halfVector = TangentToWorld(ImportanceSampleGgxYUp(Hammersley(sampleIndex, kSampleCount), roughness), reflection);
                    const glm::vec3 light = glm::normalize(2.0f * glm::dot(reflection, halfVector) * halfVector - reflection);
                    const float nDotL = std::max(glm::dot(reflection, light), 0.0f);
                    if (nDotL > 0.0f) {
                        sum += SampleEquirect(rgbaPixels, width, height, light) * nDotL;
                        weightSum += nDotL;
                    }
                }
                const glm::vec3 value = weightSum > 0.0f ? sum / weightSum : SampleEquirect(rgbaPixels, width, height, reflection);
                const size_t offset = ((static_cast<size_t>(level) * kPrefilterHeight + y) * kPrefilterWidth + x) * 4u;
                prefilter[offset + 0u] = value.r;
                prefilter[offset + 1u] = value.g;
                prefilter[offset + 2u] = value.b;
                prefilter[offset + 3u] = 1.0f;
            }
        }
    }

    _iblSpecularPrefilterImage = _device.CreateImage(ImageDesc{
        .extent = VkExtent3D{ kPrefilterWidth, kPrefilterHeight * kRoughnessLevels, 1u },
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .registerBindlessSampled = true,
        .debugName = "IblSpecularPrefilterEquirectAtlas",
    });
    _device.UploadImageData(_iblSpecularPrefilterImage,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(prefilter.data()), prefilter.size() * sizeof(float)));
    _iblSpecularPrefilterSampledImageIndex = _device.GetImageResource(_iblSpecularPrefilterImage).bindless.sampledImage;
}

void Renderer::DestroyIblResources()
{
    ClearExternalEnvironmentMap();
    if (_iblDiffuseIrradianceImage) {
        _device.DestroyImage(_iblDiffuseIrradianceImage);
        _iblDiffuseIrradianceImage = {};
    }
    _iblDiffuseIrradianceSampledImageIndex = kInvalidResourceIndex;
    if (_iblSpecularPrefilterImage) {
        _device.DestroyImage(_iblSpecularPrefilterImage);
        _iblSpecularPrefilterImage = {};
    }
    _iblSpecularPrefilterSampledImageIndex = kInvalidResourceIndex;
    if (_iblBrdfLutImage) {
        _device.DestroyImage(_iblBrdfLutImage);
        _iblBrdfLutImage = {};
    }
    _iblBrdfLutSampledImageIndex = kInvalidResourceIndex;
}

bool Renderer::LoadExternalEnvironmentMap(const std::filesystem::path& path)
{
    ClearExternalEnvironmentMap();

    int width = 0;
    int height = 0;
    int channels = 0;
    float* pixels = stbi_loadf(path.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (pixels == nullptr || width <= 0 || height <= 0) {
        if (pixels != nullptr) {
            stbi_image_free(pixels);
        }
        return false;
    }

    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t byteCount = texelCount * 4u * sizeof(float);
    _externalEnvironmentImage = _device.CreateImage(ImageDesc{
        .extent = VkExtent3D{ static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1u },
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .registerBindlessSampled = true,
        .debugName = "ExternalEnvironmentEquirect",
    });
    _device.UploadImageData(_externalEnvironmentImage,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(pixels), byteCount));
    _environmentSampledImageIndex = _device.GetImageResource(_externalEnvironmentImage).bindless.sampledImage;
    if (_environmentSampledImageIndex != kInvalidResourceIndex) {
        CreateEnvironmentCubemapAtlas(std::span<const float>(pixels, texelCount * 4u),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height));
        CreateDiffuseIrradianceEquirect(std::span<const float>(pixels, texelCount * 4u),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height));
        CreateSpecularPrefilterEquirectAtlas(std::span<const float>(pixels, texelCount * 4u),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height));
    }
    stbi_image_free(pixels);
    ResetAccumulation();
    return _environmentSampledImageIndex != kInvalidResourceIndex;
}

void Renderer::ClearExternalEnvironmentMap()
{
    if (_externalEnvironmentImage) {
        _device.DestroyImage(_externalEnvironmentImage);
        _externalEnvironmentImage = {};
    }
    _environmentSampledImageIndex = kInvalidResourceIndex;
    if (_iblEnvironmentCubemapImage) {
        _device.DestroyImage(_iblEnvironmentCubemapImage);
        _iblEnvironmentCubemapImage = {};
    }
    _iblEnvironmentCubemapSampledImageIndex = kInvalidResourceIndex;
    if (_iblDiffuseIrradianceImage) {
        _device.DestroyImage(_iblDiffuseIrradianceImage);
        _iblDiffuseIrradianceImage = {};
    }
    _iblDiffuseIrradianceSampledImageIndex = kInvalidResourceIndex;
    if (_iblSpecularPrefilterImage) {
        _device.DestroyImage(_iblSpecularPrefilterImage);
        _iblSpecularPrefilterImage = {};
    }
    _iblSpecularPrefilterSampledImageIndex = kInvalidResourceIndex;
    ResetAccumulation();
}

bool Renderer::RegisterPass(RenderPassRegistrationDesc desc)
{
    if (desc.id.empty() || !desc.pass || FindPassEntry(desc.id) != nullptr) {
        return false;
    }

    _passRegistry.push_back(RegisteredPassEntry{
        .id = std::move(desc.id),
        .pass = std::move(desc.pass),
        .configure = std::move(desc.configure),
        .order = desc.order,
        .enabled = desc.enabled,
    });

    RegisteredPassEntry& entry = _passRegistry.back();
    if (_device.GetDevice() != VK_NULL_HANDLE) {
        entry.pass->Initialize(_device);
    }

    _passExecutionPlanDirty = true;
    return true;
}

bool Renderer::UnregisterPass(std::string_view id)
{
    const auto it = std::find_if(_passRegistry.begin(), _passRegistry.end(), [id](const RegisteredPassEntry& entry) {
        return entry.id == id;
    });
    if (it == _passRegistry.end()) {
        return false;
    }

    if (_device.GetDevice() != VK_NULL_HANDLE) {
        it->pass->Shutdown(_device);
    }

    _passRegistry.erase(it);
    _passExecutionPlan.clear();
    _passExecutionPlanDirty = true;
    return true;
}

bool Renderer::SetPassEnabled(std::string_view id, bool enabled)
{
    RegisteredPassEntry* entry = FindPassEntry(id);
    if (entry == nullptr) {
        return false;
    }

    entry->enabled = enabled;
    _passExecutionPlanDirty = true;
    return true;
}

bool Renderer::SetPassOrder(std::string_view id, uint32_t order)
{
    RegisteredPassEntry* entry = FindPassEntry(id);
    if (entry == nullptr) {
        return false;
    }

    entry->order = order;
    _passExecutionPlanDirty = true;
    return true;
}

IRenderPass* Renderer::FindPass(std::string_view id)
{
    RegisteredPassEntry* entry = FindPassEntry(id);
    return entry != nullptr ? entry->pass.get() : nullptr;
}

const IRenderPass* Renderer::FindPass(std::string_view id) const
{
    const RegisteredPassEntry* entry = FindPassEntry(id);
    return entry != nullptr ? entry->pass.get() : nullptr;
}

GpuDrivenStats Renderer::GetGpuDrivenStats() const
{
    GpuDrivenStats stats{};
    stats.totalSurfaces = static_cast<uint32_t>(_scene.GetSurfaces().size());
    stats.visibilitySetValid = HasValidVisibilitySet();
    stats.visibleSurfaces = stats.visibilitySetValid ? static_cast<uint32_t>(_visibleSurfaceIndices.size()) : stats.totalSurfaces;
    stats.culledSurfaces = stats.totalSurfaces >= stats.visibleSurfaces ? stats.totalSurfaces - stats.visibleSurfaces : 0u;
    stats.indirectDrawEnabled = _settings.useIndirectDraw;
    stats.indirectDrawEstimate = _settings.useIndirectDraw && stats.visibleSurfaces > 0u ? 1u : stats.visibleSurfaces;
    stats.gpuDrivenBackend = false;
    return stats;
}

MeshletClusterStats Renderer::GetMeshletClusterStats() const
{
    constexpr uint32_t kTrianglesPerMeshlet = 64;

    MeshletClusterStats stats{};
    stats.trianglesPerMeshlet = kTrianglesPerMeshlet;
    stats.totalClusters = static_cast<uint32_t>(_scene.GetSurfaces().size());
    stats.boundsAvailable = static_cast<uint32_t>(_scene.GetSurfaceBounds().size());
    stats.visibilitySetValid = HasValidVisibilitySet();
    stats.coneCullingEnabled = false;
    stats.gpuDrivenBackend = false;

    const auto surfaceMeshletCount = [&](uint32_t surfaceIndex) -> uint32_t {
        if (surfaceIndex >= _scene.GetSurfaces().size()) {
            return 0u;
        }
        const uint32_t triangleCount = _scene.GetSurfaces()[surfaceIndex].indexCount / 3u;
        return std::max(1u, (triangleCount + kTrianglesPerMeshlet - 1u) / kTrianglesPerMeshlet);
    };

    for (uint32_t surfaceIndex = 0; surfaceIndex < static_cast<uint32_t>(_scene.GetSurfaces().size()); ++surfaceIndex) {
        stats.totalMeshlets += surfaceMeshletCount(surfaceIndex);
    }

    if (stats.visibilitySetValid) {
        stats.visibleClusters = static_cast<uint32_t>(_visibleSurfaceIndices.size());
        for (uint32_t surfaceIndex : _visibleSurfaceIndices) {
            stats.visibleMeshlets += surfaceMeshletCount(surfaceIndex);
        }
    } else {
        stats.visibleClusters = stats.totalClusters;
        stats.visibleMeshlets = stats.totalMeshlets;
    }

    stats.culledClusters = stats.totalClusters >= stats.visibleClusters ? stats.totalClusters - stats.visibleClusters : 0u;
    stats.culledMeshlets = stats.totalMeshlets >= stats.visibleMeshlets ? stats.totalMeshlets - stats.visibleMeshlets : 0u;
    return stats;
}

TemporalUpscalerStats Renderer::GetTemporalUpscalerStats() const
{
    const VkExtent2D outputExtent = _device.GetSwapchainExtent();
    const float scale = std::clamp(_settings.temporalUpscalerScale, 0.25f, 1.0f);
    TemporalUpscalerStats stats{};
    stats.outputWidth = outputExtent.width;
    stats.outputHeight = outputExtent.height;
    stats.inputWidth = std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(outputExtent.width) * scale)));
    stats.inputHeight = std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(outputExtent.height) * scale)));
    stats.scale = scale;
    stats.sharpness = std::clamp(_settings.temporalUpscalerSharpness, 0.0f, 1.0f);
    stats.requested = _settings.enableTemporalUpscaler;
    stats.backendAvailable = stats.requested && NeedsDeferredPass(_settings) && !NeedsGaussianPass(_settings);
    stats.taaHistoryAvailable = _settings.enableTaa || _settings.enableTemporalUpscaler || IsTemporalDebugView(_settings.debugView);
    stats.motionVectorsAvailable = true;
    stats.depthAvailable = true;
    stats.materialReactiveMaskAvailable = stats.backendAvailable && _settings.temporalMaterialReactiveMask;
    stats.authoredAlphaReactiveMaskAvailable = stats.backendAvailable && _settings.temporalMaterialReactiveMask;
    stats.reactiveMaskStrength = std::clamp(_settings.temporalReactiveMaskStrength, 0.0f, 1.0f);
    stats.reactiveMaskAvailable =
        stats.backendAvailable && stats.taaHistoryAvailable && stats.motionVectorsAvailable && stats.depthAvailable;
    return stats;
}

RestirStats Renderer::GetRestirStats() const
{
    constexpr uint64_t kEstimatedReservoirBytes = 32u;
    const VkExtent2D extent = _device.GetSwapchainExtent();
    RestirStats stats{};
    stats.requestedDi = _settings.enableRestirDi;
    stats.requestedGi = _settings.enableRestirGi;
    stats.requestedPt = _settings.enableRestirPt;
    stats.backendAvailable = false;
    stats.reservoirBuffersAvailable = false;
    stats.temporalReuse = _settings.restirTemporalReuse;
    stats.spatialReuse = _settings.restirSpatialReuse;
    stats.historyAvailable = false;
    stats.emissiveTriangleCount = static_cast<uint32_t>(std::min<size_t>(_scene.GetEmissiveTriangles().size(), std::numeric_limits<uint32_t>::max()));
    stats.localLightCount = 1u
        + (_settings.enablePointLight ? 1u : 0u)
        + (_settings.enableSpotLight ? 1u : 0u)
        + (_settings.enableAreaLight ? 1u : 0u);
    stats.activeLightCount = std::max(1u, stats.localLightCount + stats.emissiveTriangleCount);
    stats.candidateLightCount = std::clamp(_settings.restirCandidateLights, 1u, std::max(1u, stats.activeLightCount));
    stats.reservoirCount = std::clamp(_settings.restirReservoirCount, 1u, 8u);
    stats.reservoirPixels = static_cast<uint64_t>(extent.width) * static_cast<uint64_t>(extent.height) * stats.reservoirCount;
    const uint64_t reservoirBufferBytes = stats.reservoirPixels * kEstimatedReservoirBytes;
    const uint64_t temporalMultiplier = stats.temporalReuse ? 2u : 1u;
    stats.estimatedDiReservoirBytes = stats.requestedDi ? reservoirBufferBytes * temporalMultiplier : 0u;
    stats.estimatedGiReservoirBytes = stats.requestedGi ? reservoirBufferBytes * temporalMultiplier : 0u;
    stats.estimatedPtReservoirBytes = stats.requestedPt ? reservoirBufferBytes * temporalMultiplier : 0u;
    stats.estimatedReservoirBytes =
        stats.estimatedDiReservoirBytes + stats.estimatedGiReservoirBytes + stats.estimatedPtReservoirBytes;
    const bool requested = stats.requestedDi || stats.requestedGi || stats.requestedPt;
    stats.diReservoirBuffersAvailable = stats.requestedDi
        && (_restirReservoirBuffer && _restirReservoirBufferBytes >= reservoirBufferBytes
            && (!stats.temporalReuse
                || (_restirHistoryReservoirBuffer && _restirHistoryReservoirBufferBytes >= reservoirBufferBytes)));
    stats.giReservoirBuffersAvailable = stats.requestedGi
        && (_restirGiReservoirBuffer && _restirGiReservoirBufferBytes >= reservoirBufferBytes
            && (!stats.temporalReuse
                || (_restirGiHistoryReservoirBuffer && _restirGiHistoryReservoirBufferBytes >= reservoirBufferBytes)));
    stats.ptReservoirBuffersAvailable = stats.requestedPt
        && (_restirPtReservoirBuffer && _restirPtReservoirBufferBytes >= reservoirBufferBytes
            && (!stats.temporalReuse
                || (_restirPtHistoryReservoirBuffer && _restirPtHistoryReservoirBufferBytes >= reservoirBufferBytes)));
    stats.reservoirBuffersAvailable =
        requested
        && (!stats.requestedDi || stats.diReservoirBuffersAvailable)
        && (!stats.requestedGi || stats.giReservoirBuffersAvailable)
        && (!stats.requestedPt || stats.ptReservoirBuffersAvailable);
    stats.historyAvailable = stats.temporalReuse
        && ((!stats.requestedDi || (_restirHistoryReservoirBuffer && _restirHistoryReservoirBufferBytes >= reservoirBufferBytes))
            && (!stats.requestedGi || (_restirGiHistoryReservoirBuffer && _restirGiHistoryReservoirBufferBytes >= reservoirBufferBytes))
            && (!stats.requestedPt || (_restirPtHistoryReservoirBuffer && _restirPtHistoryReservoirBufferBytes >= reservoirBufferBytes)));
    const auto* restirPass = FindPass<RestirDiPass>("restir-di");
    const auto* restirResolvePass = FindPass<RestirDiResolvePass>("restir-di-resolve");
    stats.candidateSamplingAvailable = stats.requestedDi && stats.diReservoirBuffersAvailable
        && restirPass != nullptr && restirPass->IsBackendAvailable();
    stats.temporalReusePassAvailable = stats.candidateSamplingAvailable && stats.temporalReuse && stats.historyAvailable;
    stats.lightingResolveAvailable = stats.requestedDi && stats.candidateSamplingAvailable
        && NeedsDeferredPass(_settings) && restirResolvePass != nullptr && restirResolvePass->IsBackendAvailable();
    stats.spatialReusePassAvailable =
        stats.lightingResolveAvailable && stats.spatialReuse && _settings.restirSpatialSamples > 0u;
    stats.giReservoirBackendAvailable = stats.requestedGi && stats.giReservoirBuffersAvailable;
    stats.ptReservoirBackendAvailable = stats.requestedPt && stats.ptReservoirBuffersAvailable;
    stats.backendAvailable = requested && stats.reservoirBuffersAvailable
        && (stats.candidateSamplingAvailable || stats.giReservoirBackendAvailable || stats.ptReservoirBackendAvailable);
    return stats;
}

DdgiStats Renderer::GetDdgiStats() const
{
    constexpr uint64_t kIrradianceTexelsPerProbe = 6u * 6u;
    constexpr uint64_t kVisibilityTexelsPerProbe = 14u * 14u;
    constexpr uint64_t kRgba16fBytesPerTexel = 8u;
    constexpr uint64_t kRg16fBytesPerTexel = 4u;

    DdgiStats stats{};
    stats.probeCountX = std::clamp(_settings.ddgiProbeCountX, 1u, 32u);
    stats.probeCountY = std::clamp(_settings.ddgiProbeCountY, 1u, 16u);
    stats.probeCountZ = std::clamp(_settings.ddgiProbeCountZ, 1u, 32u);
    stats.totalProbeCount = stats.probeCountX * stats.probeCountY * stats.probeCountZ;
    stats.raysPerProbe = std::clamp(_settings.ddgiRaysPerProbe, 16u, 1024u);
    stats.raysPerUpdate = static_cast<uint64_t>(stats.totalProbeCount) * stats.raysPerProbe;
    stats.estimatedIrradianceBytes = static_cast<uint64_t>(stats.totalProbeCount) * kIrradianceTexelsPerProbe * kRgba16fBytesPerTexel;
    stats.estimatedVisibilityBytes = static_cast<uint64_t>(stats.totalProbeCount) * kVisibilityTexelsPerProbe * kRg16fBytesPerTexel;
    stats.probeSpacing = std::clamp(_settings.ddgiProbeSpacing, 0.25f, 10.0f);
    stats.hysteresis = std::clamp(_settings.ddgiHysteresis, 0.0f, 1.0f);
    stats.intensity = std::clamp(_settings.ddgiIntensity, 0.0f, 2.0f);
    stats.requested = _settings.enableDdgi;
    stats.probeStorageAvailable = _ddgiIrradianceBuffer && _ddgiVisibilityBuffer
        && _ddgiIrradianceBufferBytes >= stats.estimatedIrradianceBytes
        && _ddgiVisibilityBufferBytes >= stats.estimatedVisibilityBytes;
    stats.probeCompositeAvailable = stats.requested && NeedsDeferredPass(_settings);
    const auto* ddgiProbeUpdatePass = FindPass<DdgiProbeUpdatePass>("ddgi-probe-update");
    stats.rayUpdateAvailable = stats.requested && stats.probeStorageAvailable && _scene.HasRayTracingScene()
        && ddgiProbeUpdatePass != nullptr && ddgiProbeUpdatePass->IsBackendAvailable();
    stats.temporalBlendAvailable = stats.rayUpdateAvailable && stats.hysteresis > 0.0f;
    stats.backendAvailable =
        stats.requested && stats.probeStorageAvailable && (stats.probeCompositeAvailable || stats.rayUpdateAvailable);
    stats.overlayEnabled = _settings.showGiProbeOverlay;
    return stats;
}

void Renderer::EnsureDdgiResources()
{
    const DdgiStats stats = GetDdgiStats();
    if (!stats.requested || _device.GetDevice() == VK_NULL_HANDLE) {
        DestroyDdgiResources();
        return;
    }

    const auto recreateBuffer = [&](BufferHandle& handle, uint64_t& currentBytes, uint64_t requiredBytes, std::string_view name) {
        if (handle && currentBytes == requiredBytes) {
            return;
        }
        if (handle) {
            _device.DestroyBuffer(handle);
            handle = {};
            currentBytes = 0;
        }
        if (requiredBytes == 0u) {
            return;
        }
        handle = _device.CreateBuffer(BufferDesc{
            .size = static_cast<VkDeviceSize>(requiredBytes),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .registerBindlessStorage = true,
            .debugName = std::string(name),
        });
        currentBytes = requiredBytes;
    };

    recreateBuffer(_ddgiIrradianceBuffer, _ddgiIrradianceBufferBytes, stats.estimatedIrradianceBytes, "DDGI.IrradianceProbeStorage");
    recreateBuffer(_ddgiVisibilityBuffer, _ddgiVisibilityBufferBytes, stats.estimatedVisibilityBytes, "DDGI.VisibilityProbeStorage");
}

void Renderer::DestroyDdgiResources()
{
    if (_ddgiIrradianceBuffer) {
        _device.DestroyBuffer(_ddgiIrradianceBuffer);
        _ddgiIrradianceBuffer = {};
    }
    if (_ddgiVisibilityBuffer) {
        _device.DestroyBuffer(_ddgiVisibilityBuffer);
        _ddgiVisibilityBuffer = {};
    }
    _ddgiIrradianceBufferBytes = 0;
    _ddgiVisibilityBufferBytes = 0;
}

void Renderer::EnsureRestirResources()
{
    const RestirStats stats = GetRestirStats();
    const bool requested = stats.requestedDi || stats.requestedGi || stats.requestedPt;
    if (!requested || _device.GetDevice() == VK_NULL_HANDLE) {
        DestroyRestirResources();
        return;
    }

    constexpr uint64_t kEstimatedReservoirBytes = 32u;
    const uint64_t reservoirBufferBytes = stats.reservoirPixels * kEstimatedReservoirBytes;
    const auto recreateBuffer = [&](BufferHandle& handle, uint64_t& currentBytes, uint64_t requiredBytes, std::string_view name) {
        if (handle && currentBytes == requiredBytes) {
            return;
        }
        if (handle) {
            _device.DestroyBuffer(handle);
            handle = {};
            currentBytes = 0;
        }
        if (requiredBytes == 0u) {
            return;
        }
        handle = _device.CreateBuffer(BufferDesc{
            .size = static_cast<VkDeviceSize>(requiredBytes),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .registerBindlessStorage = true,
            .debugName = std::string(name),
        });
        currentBytes = requiredBytes;
    };

    const auto releaseBuffer = [&](BufferHandle& handle, uint64_t& currentBytes) {
        if (!handle) {
            currentBytes = 0;
            return;
        }
        _device.DestroyBuffer(handle);
        handle = {};
        currentBytes = 0;
    };
    const auto updateReservoirSet = [&](bool enabled,
                                        BufferHandle& current,
                                        uint64_t& currentBytes,
                                        BufferHandle& history,
                                        uint64_t& historyBytes,
                                        std::string_view currentName,
                                        std::string_view historyName) {
        if (!enabled) {
            releaseBuffer(current, currentBytes);
            releaseBuffer(history, historyBytes);
            return;
        }
        recreateBuffer(current, currentBytes, reservoirBufferBytes, currentName);
        if (stats.temporalReuse) {
            recreateBuffer(history, historyBytes, reservoirBufferBytes, historyName);
        } else {
            releaseBuffer(history, historyBytes);
        }
    };

    updateReservoirSet(stats.requestedDi,
        _restirReservoirBuffer,
        _restirReservoirBufferBytes,
        _restirHistoryReservoirBuffer,
        _restirHistoryReservoirBufferBytes,
        "ReSTIR.DI.CurrentReservoirStorage",
        "ReSTIR.DI.HistoryReservoirStorage");
    updateReservoirSet(stats.requestedGi,
        _restirGiReservoirBuffer,
        _restirGiReservoirBufferBytes,
        _restirGiHistoryReservoirBuffer,
        _restirGiHistoryReservoirBufferBytes,
        "ReSTIR.GI.CurrentReservoirStorage",
        "ReSTIR.GI.HistoryReservoirStorage");
    updateReservoirSet(stats.requestedPt,
        _restirPtReservoirBuffer,
        _restirPtReservoirBufferBytes,
        _restirPtHistoryReservoirBuffer,
        _restirPtHistoryReservoirBufferBytes,
        "ReSTIR.PT.CurrentReservoirStorage",
        "ReSTIR.PT.HistoryReservoirStorage");
}

void Renderer::DestroyRestirResources()
{
    if (_restirReservoirBuffer) {
        _device.DestroyBuffer(_restirReservoirBuffer);
        _restirReservoirBuffer = {};
    }
    if (_restirHistoryReservoirBuffer) {
        _device.DestroyBuffer(_restirHistoryReservoirBuffer);
        _restirHistoryReservoirBuffer = {};
    }
    if (_restirGiReservoirBuffer) {
        _device.DestroyBuffer(_restirGiReservoirBuffer);
        _restirGiReservoirBuffer = {};
    }
    if (_restirGiHistoryReservoirBuffer) {
        _device.DestroyBuffer(_restirGiHistoryReservoirBuffer);
        _restirGiHistoryReservoirBuffer = {};
    }
    if (_restirPtReservoirBuffer) {
        _device.DestroyBuffer(_restirPtReservoirBuffer);
        _restirPtReservoirBuffer = {};
    }
    if (_restirPtHistoryReservoirBuffer) {
        _device.DestroyBuffer(_restirPtHistoryReservoirBuffer);
        _restirPtHistoryReservoirBuffer = {};
    }
    _restirReservoirBufferBytes = 0;
    _restirHistoryReservoirBufferBytes = 0;
    _restirGiReservoirBufferBytes = 0;
    _restirGiHistoryReservoirBufferBytes = 0;
    _restirPtReservoirBufferBytes = 0;
    _restirPtHistoryReservoirBufferBytes = 0;
}

IblStats Renderer::GetIblStats() const
{
    constexpr uint64_t kCubeFaces = 6u;
    constexpr uint64_t kRgba16fBytesPerTexel = 8u;
    constexpr uint64_t kRgba32fBytesPerTexel = 16u;
    constexpr uint64_t kRg32fBytesPerTexel = 8u;

    IblStats stats{};
    stats.requested = _settings.environmentIntensity > 0.0f
        && (_settings.environmentDiffuseStrength > 0.0f || _settings.environmentSpecularStrength > 0.0f);
    stats.externalSourceAvailable = _settings.externalHdriAvailable;
    stats.environmentMapUploaded = _environmentSampledImageIndex != kInvalidResourceIndex;
    stats.environmentCubemapAvailable = _iblEnvironmentCubemapSampledImageIndex != kInvalidResourceIndex;
    stats.proceduralSource = !_settings.externalHdriAvailable;
    stats.sourceIsHdr = _settings.externalHdriIsHdr;
    stats.sourceWidth = _settings.externalHdriWidth;
    stats.sourceHeight = _settings.externalHdriHeight;
    stats.sourceChannels = _settings.externalHdriChannels;

    const uint64_t environmentCubePixels = kCubeFaces * stats.environmentCubemapResolution * stats.environmentCubemapResolution;
    stats.estimatedEnvironmentCubemapBytes = _iblEnvironmentCubemapImage
        ? static_cast<uint64_t>(_device.GetImageExtent(_iblEnvironmentCubemapImage).width)
            * _device.GetImageExtent(_iblEnvironmentCubemapImage).height * kRgba32fBytesPerTexel
        : environmentCubePixels * kRgba16fBytesPerTexel;

    const uint64_t diffusePixels = kCubeFaces * stats.diffuseCubemapResolution * stats.diffuseCubemapResolution;
    stats.estimatedDiffuseBytes = _iblDiffuseIrradianceImage
        ? static_cast<uint64_t>(_device.GetImageExtent(_iblDiffuseIrradianceImage).width)
            * _device.GetImageExtent(_iblDiffuseIrradianceImage).height * 4u * sizeof(float)
        : diffusePixels * kRgba16fBytesPerTexel;

    uint64_t specularPixels = 0;
    for (uint32_t mip = 0; mip < stats.specularMipCount; ++mip) {
        const uint32_t mipResolution = std::max(1u, stats.specularCubemapResolution >> mip);
        specularPixels += kCubeFaces * mipResolution * mipResolution;
    }
    stats.estimatedSpecularBytes = _iblSpecularPrefilterImage
        ? static_cast<uint64_t>(_device.GetImageExtent(_iblSpecularPrefilterImage).width)
            * _device.GetImageExtent(_iblSpecularPrefilterImage).height * 4u * sizeof(float)
        : specularPixels * kRgba16fBytesPerTexel;
    stats.estimatedBrdfLutBytes = static_cast<uint64_t>(stats.brdfLutResolution) * stats.brdfLutResolution * kRg32fBytesPerTexel;

    stats.diffuseBackendAvailable = true;
    stats.specularBackendAvailable = _iblSpecularPrefilterSampledImageIndex != kInvalidResourceIndex;
    stats.diffuseIrradianceAvailable = _iblDiffuseIrradianceSampledImageIndex != kInvalidResourceIndex;
    stats.specularPrefilterAvailable = _iblSpecularPrefilterSampledImageIndex != kInvalidResourceIndex;
    stats.brdfLutAvailable = _iblBrdfLutSampledImageIndex != kInvalidResourceIndex;
    return stats;
}

RayEffectsStats Renderer::GetRayEffectsStats() const
{
    const VkExtent2D extent = _device.GetSwapchainExtent();
    const uint32_t inputWidth = _settings.rtHalfResolution ? std::max(1u, (extent.width + 1u) / 2u) : extent.width;
    const uint32_t inputHeight = _settings.rtHalfResolution ? std::max(1u, (extent.height + 1u) / 2u) : extent.height;
    const uint64_t pixels = static_cast<uint64_t>(inputWidth) * static_cast<uint64_t>(inputHeight);

    RayEffectsStats stats{};
    stats.inputWidth = inputWidth;
    stats.inputHeight = inputHeight;
    stats.shadowSamples = std::clamp(_settings.rtShadowSamples, 1u, 8u);
    stats.aoSamples = std::clamp(_settings.rtAoSamples, 1u, 8u);
    stats.reflectionSamples = std::clamp(_settings.rtReflectionSamples, 1u, 8u);
    stats.giSamples = std::clamp(_settings.rtGiSamples, 1u, 8u);
    stats.shadowsRequested = _settings.enableRtShadows;
    stats.aoRequested = _settings.enableRtAmbientOcclusion;
    stats.reflectionsRequested = _settings.enableRtReflections;
    stats.giRequested = _settings.enableRtGlobalIllumination;
    stats.rayQueryAvailable = _device.GetRayTracingSupport().rayQueryFeatures.rayQuery == VK_TRUE;
    stats.rtPipelineAvailable = _device.GetRayTracingSupport().rayTracingPipelineFeatures.rayTracingPipeline == VK_TRUE;
    stats.tlasAvailable = _scene.HasRayTracingScene();
    const auto* rayEffectsPass = FindPass<RayEffectsPass>("ray-effects");
    stats.backendAvailable = stats.rayQueryAvailable && stats.tlasAvailable && IsRayEffectsRequested(_settings)
        && rayEffectsPass != nullptr && rayEffectsPass->IsBackendAvailable();
    stats.halfResolution = _settings.rtHalfResolution;
    stats.denoiserRequested = _settings.rtDenoiser;
    stats.giSpatialDenoiseAvailable = stats.backendAvailable && stats.giRequested && stats.denoiserRequested;
    stats.temporalAccumulation = _settings.rtTemporalAccumulation;
    stats.estimatedShadowRays = stats.shadowsRequested ? pixels * stats.shadowSamples : 0u;
    stats.estimatedAoRays = stats.aoRequested ? pixels * stats.aoSamples : 0u;
    stats.estimatedReflectionRays = stats.reflectionsRequested ? pixels * stats.reflectionSamples : 0u;
    stats.estimatedGiRays = stats.giRequested ? pixels * stats.giSamples : 0u;
    return stats;
}

std::vector<RenderPassDebugInfo> Renderer::GetRenderPassDebugInfo() const
{
    const auto estimateVisibleSurfaceCount = [&]() -> uint32_t {
        return HasValidVisibilitySet() ? static_cast<uint32_t>(_visibleSurfaceIndices.size()) : static_cast<uint32_t>(_scene.GetSurfaces().size());
    };
    const auto estimateVisibleTriangleCount = [&]() -> uint64_t {
        uint64_t triangles = 0;
        if (HasValidVisibilitySet()) {
            for (uint32_t surfaceIndex : _visibleSurfaceIndices) {
                if (surfaceIndex < _scene.GetSurfaces().size()) {
                    triangles += _scene.GetSurfaces()[surfaceIndex].indexCount / 3u;
                }
            }
            return triangles;
        }

        for (const auto& surface : _scene.GetSurfaces()) {
            triangles += surface.indexCount / 3u;
        }
        return triangles;
    };
    const auto computeDispatchGrid = [](VkExtent2D extent, uint32_t tileSize) -> uint32_t {
        return std::max(1u, (extent.width + tileSize - 1u) / tileSize) * std::max(1u, (extent.height + tileSize - 1u) / tileSize);
    };

    const VkExtent2D extent = _device.GetSwapchainExtent();
    const uint32_t fullResDispatchGrid = computeDispatchGrid(extent, 8u);
    const VkExtent2D pathTraceExtent{
        std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(extent.width) * _settings.pathTraceResolutionScale))),
        std::max(1u, static_cast<uint32_t>(std::ceil(static_cast<float>(extent.height) * _settings.pathTraceResolutionScale))),
    };
    const uint32_t pathTraceDispatchGrid = computeDispatchGrid(pathTraceExtent, 8u);
    const auto estimatePathRayWork = [&]() {
        struct PathRayWork {
            uint64_t primary{ 0 };
            uint64_t shadow{ 0 };
            uint64_t diffuse{ 0 };
            uint64_t specular{ 0 };

            [[nodiscard]] uint64_t total() const { return primary + shadow + diffuse + specular; }
        };

        const uint64_t sampleCount = static_cast<uint64_t>(pathTraceExtent.width) * pathTraceExtent.height
            * std::max(1u, _settings.pathTraceSamplesPerPixel);
        PathRayWork work{};
        work.primary = sampleCount;

        if (GetActivePathTraceBackend() != PathTraceBackend::HardwareRT) {
            return work;
        }

        const uint32_t bounceLimit = std::clamp(_settings.pathTraceMaxBounces, 1u, 12u);
        const uint64_t secondary = sampleCount * static_cast<uint64_t>(bounceLimit - 1u);
        const float averageMetallic = [&]() {
            const auto& materials = _scene.GetMaterials();
            if (materials.empty()) {
                return 0.0f;
            }
            float sum = 0.0f;
            for (const auto& material : materials) {
                sum += std::clamp(material.materialParams.x, 0.0f, 1.0f);
            }
            return sum / static_cast<float>(materials.size());
        }();
        const float specularShare = std::clamp(0.25f + averageMetallic * 0.5f, 0.1f, 0.75f);
        work.specular = static_cast<uint64_t>(std::round(static_cast<double>(secondary) * specularShare));
        work.diffuse = secondary - work.specular;

        if (_settings.pathTraceNextEventEstimation) {
            const uint64_t emissiveSamples = _scene.GetEmissiveTriangles().empty() ? 0ull : 4ull;
            work.shadow = sampleCount * static_cast<uint64_t>(bounceLimit) * (1ull + emissiveSamples);
        }
        return work;
    };

    std::vector<RenderPassDebugInfo> result;
    result.reserve(_passRegistry.size());
    for (const RegisteredPassEntry& entry : _passRegistry) {
        RenderPassDebugInfo info{
            .id = entry.id,
            .name = entry.pass ? std::string(entry.pass->Name()) : std::string{},
            .order = entry.order,
            .enabled = entry.enabled,
        };

        if (entry.id == "geometry-raster") {
            info.drawCount = _settings.useIndirectDraw && estimateVisibleSurfaceCount() > 0u ? 1u : estimateVisibleSurfaceCount();
            info.triangleCount = estimateVisibleTriangleCount();
            info.instanceCount = estimateVisibleSurfaceCount();
        } else if (entry.id == "deferred-lighting") {
            info.dispatchCount = 1u;
            info.rayCount = fullResDispatchGrid * 64ull;
        } else if (entry.id == "ray-effects") {
            info.dispatchCount = 1u;
            const uint32_t rayEffectsWidth =
                _settings.rtHalfResolution ? std::max(1u, (extent.width + 1u) / 2u) : extent.width;
            const uint32_t rayEffectsHeight =
                _settings.rtHalfResolution ? std::max(1u, (extent.height + 1u) / 2u) : extent.height;
            const uint64_t rayPixels = static_cast<uint64_t>(rayEffectsWidth) * rayEffectsHeight;
            info.shadowRayCount = _settings.enableRtShadows ? rayPixels * std::clamp(_settings.rtShadowSamples, 1u, 8u) : 0ull;
            info.diffuseRayCount =
                _settings.enableRtAmbientOcclusion ? rayPixels * std::clamp(_settings.rtAoSamples, 1u, 8u) : 0ull;
            info.specularRayCount =
                _settings.enableRtReflections ? rayPixels * std::clamp(_settings.rtReflectionSamples, 1u, 8u) : 0ull;
            info.rayCount = info.shadowRayCount + info.diffuseRayCount + info.specularRayCount;
        } else if (entry.id == "restir-di") {
            const RestirStats restirStats = GetRestirStats();
            info.dispatchCount = IsRestirRequested(_settings) ? 1u : 0u;
            info.rayCount = restirStats.reservoirPixels * restirStats.candidateLightCount;
        } else if (entry.id == "restir-di-resolve") {
            const RestirStats restirStats = GetRestirStats();
            info.dispatchCount = restirStats.lightingResolveAvailable ? 1u : 0u;
            info.rayCount = restirStats.reservoirPixels;
        } else if (entry.id == "ddgi-probe-update") {
            const DdgiStats ddgiStats = GetDdgiStats();
            info.dispatchCount = ddgiStats.rayUpdateAvailable ? 1u : 0u;
            info.rayCount = ddgiStats.raysPerUpdate;
            info.diffuseRayCount = ddgiStats.raysPerUpdate;
        } else if (entry.id == "gaussian-splat") {
            info.drawCount = _scene.HasGaussianSplats() ? 1u : 0u;
            info.splatCount = _scene.GetGaussianCount();
        } else if (entry.id == "official-gaussian-raster") {
            info.dispatchCount = _scene.HasTrainedGaussians() ? 10u : 0u;
            info.splatCount = GetOfficialGaussianDuplicateCount() > 0u ? GetOfficialGaussianDuplicateCount() : _scene.GetGaussianCount();
        } else if (entry.id == "path-tracer") {
            info.dispatchCount = 1u;
            info.triangleCount = _scene.GetTriangles().size();
            const auto rayWork = estimatePathRayWork();
            info.primaryRayCount = rayWork.primary;
            info.shadowRayCount = rayWork.shadow;
            info.diffuseRayCount = rayWork.diffuse;
            info.specularRayCount = rayWork.specular;
            info.rayCount = rayWork.total();
        } else if (entry.id == "path-denoise") {
            info.dispatchCount = 1u;
            info.rayCount = static_cast<uint64_t>(pathTraceDispatchGrid) * 64ull;
        } else if (entry.id == "temporal-aa") {
            info.dispatchCount = 1u;
            info.rayCount = static_cast<uint64_t>(fullResDispatchGrid) * 64ull;
        } else if (entry.id == "composite") {
            info.drawCount = 1u;
        }

        result.push_back(std::move(info));
    }
    std::sort(result.begin(), result.end(), [](const RenderPassDebugInfo& lhs, const RenderPassDebugInfo& rhs) {
        if (lhs.order != rhs.order) {
            return lhs.order < rhs.order;
        }
        return lhs.id < rhs.id;
    });
    return result;
}

bool Renderer::ReloadShaders()
{
    if (_device.GetDevice() == VK_NULL_HANDLE) {
        _lastShaderReloadMessage = "Renderer device is not initialized.";
        return false;
    }

    std::string compileMessage;
    if (!CompileRuntimeShaders(compileMessage)) {
        _lastShaderReloadMessage = compileMessage;
        return false;
    }

    _device.WaitIdle();
    for (RendererFrameContext& frame : _frames) {
        ReleaseTransientResources(frame);
    }
    _transientImagePool.Purge(_device);
    _lastRenderGraphTimings.clear();

    bool success = true;
    std::ostringstream message;
    for (RegisteredPassEntry& entry : _passRegistry) {
        if (!entry.pass) {
            continue;
        }
        const bool wasEnabled = entry.enabled;
        try {
            entry.pass->Shutdown(_device);
            entry.pass->Initialize(_device);
            entry.enabled = wasEnabled;
        } catch (const std::exception& error) {
            success = false;
            entry.enabled = false;
            message << entry.id << ": " << error.what() << '\n';
            try {
                entry.pass->Shutdown(_device);
            } catch (...) {
            }
        } catch (...) {
            success = false;
            entry.enabled = false;
            message << entry.id << ": unknown shader reload failure\n";
            try {
                entry.pass->Shutdown(_device);
            } catch (...) {
            }
        }
    }

    _passExecutionPlanDirty = true;
    ResetAccumulation();
    if (success) {
        _lastShaderReloadMessage = compileMessage + "\nAll render passes reloaded.";
    } else {
        _lastShaderReloadMessage = compileMessage + "\n" + message.str();
    }
    return success;
}

bool Renderer::RequestScreenshot(const std::filesystem::path& path)
{
    if (_device.GetDevice() == VK_NULL_HANDLE || path.empty()) {
        return false;
    }

    _pendingScreenshotPath = path;
    return true;
}

void Renderer::SetVSyncEnabled(bool enabled)
{
    if (_settings.enableVSync == enabled && _device.IsVSyncEnabled() == enabled) {
        return;
    }

    _settings.enableVSync = enabled;
    _device.SetVSyncEnabled(enabled);
    RecreateSwapchain();
    ResetAccumulation();
}

void Renderer::InitializeCommands()
{
    VkCommandPoolCreateInfo poolInfo =
        vkinit::command_pool_create_info(_device.GetGraphicsQueueFamily(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (RendererFrameContext& frame : _frames) {
        VK_CHECK(vkCreateCommandPool(_device.GetDevice(), &poolInfo, nullptr, &frame.commandPool));

        VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(frame.commandPool);
        VK_CHECK(vkAllocateCommandBuffers(_device.GetDevice(), &allocInfo, &frame.commandBuffer));
    }
}

void Renderer::InitializeSyncStructures()
{
    VkFenceCreateInfo fenceInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreInfo = vkinit::semaphore_create_info();

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(_device.GetPhysicalDevice(), &properties);
    _timestampPeriodNs = properties.limits.timestampPeriod;
    _renderGraphTimestampsSupported = properties.limits.timestampComputeAndGraphics == VK_TRUE && _timestampPeriodNs > 0.0f;

    for (RendererFrameContext& frame : _frames) {
        VK_CHECK(vkCreateFence(_device.GetDevice(), &fenceInfo, nullptr, &frame.renderFence));
        VK_CHECK(vkCreateSemaphore(_device.GetDevice(), &semaphoreInfo, nullptr, &frame.acquireSemaphore));
        if (_renderGraphTimestampsSupported) {
            VkQueryPoolCreateInfo queryPoolInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
            queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
            queryPoolInfo.queryCount = kMaxRenderGraphTimestampPasses * 2u;
            VK_CHECK(vkCreateQueryPool(_device.GetDevice(), &queryPoolInfo, nullptr, &frame.renderGraphTimestampPool));
        }
    }

    _swapchainImageRenderSemaphores.resize(_device.GetSwapchainImageHandles().size(), VK_NULL_HANDLE);
    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        VK_CHECK(vkCreateSemaphore(_device.GetDevice(), &semaphoreInfo, nullptr, &semaphore));
    }
}

void Renderer::InitializeDefaultPasses()
{
    ClearPassRegistry();

    // Pass order is explicit so the frame graph wiring stays easy to read.
    RegisterPass(RenderPassRegistrationDesc{
        .id = "geometry-raster",
        .pass = std::make_unique<GeometryRasterPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureGeometryRasterPass(*this, pass, resources);
        },
        .order = 10,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "deferred-lighting",
        .pass = std::make_unique<DeferredLightingPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureDeferredLightingPass(*this, pass, resources);
        },
        .order = 20,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "gaussian-splat",
        .pass = std::make_unique<GaussianSplatPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureGaussianPass(*this, pass, resources);
        },
        .order = 30,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "shadow-map",
        .pass = std::make_unique<ShadowMapPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureShadowMapPass(*this, pass, resources);
        },
        .order = 15,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "overdraw",
        .pass = std::make_unique<OverdrawPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureOverdrawPass(*this, pass, resources);
        },
        .order = 16,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "ray-effects",
        .pass = std::make_unique<RayEffectsPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureRayEffectsPass(*this, pass, resources);
        },
        .order = 18,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "restir-di",
        .pass = std::make_unique<RestirDiPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureRestirDiPass(*this, pass, resources);
        },
        .order = 19,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "restir-di-resolve",
        .pass = std::make_unique<RestirDiResolvePass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureRestirDiResolvePass(*this, pass, resources);
        },
        .order = 19,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "ddgi-probe-update",
        .pass = std::make_unique<DdgiProbeUpdatePass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureDdgiProbeUpdatePass(*this, pass, resources);
        },
        .order = 19,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "official-gaussian-raster",
        .pass = std::make_unique<OfficialGaussianRasterPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureOfficialGaussianPass(*this, pass, resources);
        },
        .order = 31,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "path-tracer",
        .pass = std::make_unique<PathTracerPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigurePathTracerPass(*this, pass, resources);
        },
        .order = 40,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "path-denoise",
        .pass = std::make_unique<PathDenoisePass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigurePathDenoisePass(*this, pass, resources);
        },
        .order = 45,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "temporal-aa",
        .pass = std::make_unique<TemporalAAPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureTemporalAAPass(*this, pass, resources);
        },
        .order = 46,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "bloom-extract",
        .pass = std::make_unique<BloomPass>(BloomPassStage::Extract),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureBloomPass(*this, pass, resources);
        },
        .order = 47,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "bloom-downsample",
        .pass = std::make_unique<BloomPass>(BloomPassStage::Downsample),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureBloomPass(*this, pass, resources);
        },
        .order = 48,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "bloom-upsample",
        .pass = std::make_unique<BloomPass>(BloomPassStage::Upsample),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureBloomPass(*this, pass, resources);
        },
        .order = 49,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "composite",
        .pass = std::make_unique<CompositePass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureCompositePass(*this, pass, resources);
        },
        .order = 50,
        .enabled = true,
    });
}

void Renderer::DestroyFrameResources()
{
    for (RendererFrameContext& frame : _frames) {
        if (frame.screenshotReadbackBuffer) {
            _device.DestroyBuffer(frame.screenshotReadbackBuffer);
            frame.screenshotReadbackBuffer = {};
            frame.screenshotPending = false;
        }
        ReleaseTransientResources(frame);

        if (frame.renderFence != VK_NULL_HANDLE) {
            vkDestroyFence(_device.GetDevice(), frame.renderFence, nullptr);
            frame.renderFence = VK_NULL_HANDLE;
        }
        if (frame.acquireSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(_device.GetDevice(), frame.acquireSemaphore, nullptr);
            frame.acquireSemaphore = VK_NULL_HANDLE;
        }
        if (frame.renderGraphTimestampPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(_device.GetDevice(), frame.renderGraphTimestampPool, nullptr);
            frame.renderGraphTimestampPool = VK_NULL_HANDLE;
            frame.renderGraphTimestampPending = false;
            frame.renderGraphTimestampPassCount = 0;
            frame.renderGraphTimestampPassNames.clear();
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(_device.GetDevice(), frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
            frame.commandBuffer = VK_NULL_HANDLE;
        }
    }

    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(_device.GetDevice(), semaphore, nullptr);
            semaphore = VK_NULL_HANDLE;
        }
    }
    _swapchainImageRenderSemaphores.clear();
}

void Renderer::ReleaseTransientResources(RendererFrameContext& frameContext)
{
    for (ImageHandle handle : frameContext.acquiredTransientImages) {
        _transientImagePool.Release(handle);
    }
    frameContext.acquiredTransientImages.clear();

    for (BufferHandle handle : frameContext.transientBuffers) {
        _device.DestroyBuffer(handle);
    }
    frameContext.transientBuffers.clear();
}

void Renderer::ProcessCompletedFrameReadback(RendererFrameContext& frameContext)
{
    if (!frameContext.screenshotPending || !frameContext.screenshotReadbackBuffer) {
        return;
    }

    const VkDeviceSize byteSize =
        static_cast<VkDeviceSize>(frameContext.screenshotExtent.width) * frameContext.screenshotExtent.height * 4u;
    _device.InvalidateBuffer(frameContext.screenshotReadbackBuffer, 0, byteSize);
    const AllocatedBuffer& buffer = _device.GetBufferResource(frameContext.screenshotReadbackBuffer);
    const std::string extension = [&]() {
        std::string value = frameContext.screenshotPath.extension().string();
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    }();
    const bool written = extension == ".png"
        ? WriteSwapchainPng(frameContext.screenshotPath,
            buffer.allocationInfo.pMappedData,
            frameContext.screenshotExtent,
            frameContext.screenshotFormat)
        : WriteSwapchainPpm(frameContext.screenshotPath,
            buffer.allocationInfo.pMappedData,
            frameContext.screenshotExtent,
            frameContext.screenshotFormat);
    if (!written) {
        fmt::println(stderr, "Failed to write screenshot '{}'", frameContext.screenshotPath.string());
    }

    _device.DestroyBuffer(frameContext.screenshotReadbackBuffer);
    frameContext.screenshotReadbackBuffer = {};
    frameContext.screenshotPath.clear();
    frameContext.screenshotExtent = {};
    frameContext.screenshotFormat = VK_FORMAT_UNDEFINED;
    frameContext.screenshotPending = false;
}

void Renderer::RecordScreenshotReadback(VkCommandBuffer commandBuffer, RendererFrameContext& frameContext, uint32_t swapchainImageIndex)
{
    if (_pendingScreenshotPath.empty()) {
        return;
    }

    if (frameContext.screenshotPending && frameContext.screenshotReadbackBuffer) {
        return;
    }

    const VkExtent2D extent = _device.GetSwapchainExtent();
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(extent.width) * extent.height * 4u;
    constexpr VmaAllocationCreateFlags kMappedHostFlags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    frameContext.screenshotReadbackBuffer = _device.CreateBuffer(BufferDesc{
        .size = byteSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .allocationFlags = kMappedHostFlags,
        .debugName = "ScreenshotReadback",
    });
    frameContext.screenshotPath = _pendingScreenshotPath;
    frameContext.screenshotExtent = extent;
    frameContext.screenshotFormat = _device.GetSwapchainFormat();
    frameContext.screenshotPending = true;
    _pendingScreenshotPath.clear();

    VkImage swapchainImage = _device.GetImage(_device.GetSwapchainImageHandle(swapchainImageIndex));
    const VkImageSubresourceRange colorRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    vkutil::transition_image(commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT,
        colorRange);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = VkExtent3D{ extent.width, extent.height, 1 };
    vkCmdCopyImageToBuffer(commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        _device.GetBuffer(frameContext.screenshotReadbackBuffer),
        1,
        &copyRegion);

    vkutil::transition_image(commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        colorRange);
}

void Renderer::RecordOverlay(VkCommandBuffer commandBuffer, uint32_t swapchainImageIndex)
{
    if (!_overlayDrawFn) {
        return;
    }

    VkImage swapchainImage = _device.GetImage(_device.GetSwapchainImageHandle(swapchainImageIndex));
    VkImageView swapchainView = _device.GetImageView(_device.GetSwapchainImageHandle(swapchainImageIndex));
    const VkExtent2D extent = _device.GetSwapchainExtent();
    const VkImageSubresourceRange colorRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);

    // The graph leaves the swapchain image ready for presentation. ImGui needs it
    // back in a color-attachment layout for one extra overlay draw, then we
    // transition it back to PRESENT before queue submission finishes.
    vkutil::transition_image(commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        colorRange);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchainView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, extent };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    _overlayDrawFn(commandBuffer);
    vkCmdEndRendering(commandBuffer);

    vkutil::transition_image(commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        colorRange);
}

void Renderer::RecreateSwapchain()
{
    int width = 0;
    int height = 0;
    SDL_Vulkan_GetDrawableSize(_window, &width, &height);
    if (width == 0 || height == 0) {
        return;
    }

    _device.WaitIdle();

    // Swapchain recreation invalidates images tied to the old extent. Purging the
    // transient pool here avoids reusing resources whose sizes no longer match.
    for (RendererFrameContext& frame : _frames) {
        ReleaseTransientResources(frame);
    }
    _transientImagePool.Purge(_device);

    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(_device.GetDevice(), semaphore, nullptr);
        }
    }
    _swapchainImageRenderSemaphores.clear();

    _device.RecreateSwapchain(VkExtent2D{ static_cast<uint32_t>(width), static_cast<uint32_t>(height) });
    _camera.SetViewport(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    _pathTraceFrameIndex = 0;
    if (_overlaySwapchainCallback) {
        _overlaySwapchainCallback(static_cast<uint32_t>(_device.GetSwapchainImageHandles().size()));
    }

    VkSemaphoreCreateInfo semaphoreInfo = vkinit::semaphore_create_info();
    _swapchainImageRenderSemaphores.resize(_device.GetSwapchainImageHandles().size(), VK_NULL_HANDLE);
    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        VK_CHECK(vkCreateSemaphore(_device.GetDevice(), &semaphoreInfo, nullptr, &semaphore));
    }
}

void Renderer::ClearPassRegistry()
{
    if (_device.GetDevice() != VK_NULL_HANDLE) {
        for (RegisteredPassEntry& entry : _passRegistry) {
            entry.pass->Shutdown(_device);
        }
    }

    _passExecutionPlan.clear();
    _passRegistry.clear();
    _passExecutionPlanDirty = true;
}

void Renderer::RebuildPassExecutionPlan()
{
    if (!_passExecutionPlanDirty) {
        return;
    }

    _passExecutionPlan.clear();
    _passExecutionPlan.reserve(_passRegistry.size());
    for (RegisteredPassEntry& entry : _passRegistry) {
        if (entry.enabled) {
            _passExecutionPlan.push_back(&entry);
        }
    }

    std::stable_sort(_passExecutionPlan.begin(), _passExecutionPlan.end(), [](const RegisteredPassEntry* lhs, const RegisteredPassEntry* rhs) {
        if (lhs->order != rhs->order) {
            return lhs->order < rhs->order;
        }
        return lhs->id < rhs->id;
    });

    _passExecutionPlanDirty = false;
}

void Renderer::PumpSceneLoadRequests()
{
    if (!_sceneLoadFuture.valid()) {
        return;
    }

    using namespace std::chrono_literals;
    if (_sceneLoadFuture.wait_for(0ms) != std::future_status::ready) {
        return;
    }

    AsyncSceneLoadResult result = _sceneLoadFuture.get();

    if (!result.success) {
        _sceneLoadInProgress = false;
        const std::string sceneName = result.path.empty() ? std::string("scene") : result.path.filename().string();
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.path = result.path;
        _sceneLoadStatus.parseMs = result.parseMs;
        _sceneLoadStatus.prepareMs = result.prepareMs;
        _sceneLoadStatus.geometryUploadMs = 0.0f;
        _sceneLoadStatus.textureUploadMs = 0.0f;
        _sceneLoadStatus.blasMs = 0.0f;
        _sceneLoadStatus.tlasMs = 0.0f;
        _sceneLoadStatus.message = "Failed to load " + sceneName;
        if (!result.errorMessage.empty()) {
            _sceneLoadStatus.message += ": " + result.errorMessage;
        }
        return;
    }

    ValidateSceneLoadTransition(_sceneLoadStatus, SceneLoadState::UploadingGeometry, "PumpSceneLoadRequests");
    _sceneLoadStatus.state = SceneLoadState::UploadingGeometry;
    _sceneLoadStatus.path = result.path;
    _sceneLoadStatus.parseMs = result.parseMs;
    _sceneLoadStatus.prepareMs = result.prepareMs;
    _sceneLoadStatus.message = "Uploading " + result.path.filename().string() + "...";
    if (UsesStreamingUpload(_settings)) {
        StartPendingSceneUpload(std::move(result.scene), result.parseMs, result.prepareMs);
    } else {
        VESTA_ASSERT(!_startupSafeModeActive,
            "Startup safe mode must not apply loaded scenes synchronously. Force streaming upload instead.");
        ApplySceneLoadState(_sceneLoadStatus,
            SceneLoadState::ReadyToSwap,
            "Finalizing " + result.path.filename().string() + "...",
            "PumpSceneLoadRequests");
        _sceneLoadInProgress = false;
        ApplyLoadedScene(std::move(result.scene));
    }
}

void Renderer::PumpPendingSceneUpload()
{
    if (!_pendingSceneUpload.active) {
        return;
    }

    using Stage = PendingSceneUploadStage;
    const auto stageLabel = [](Stage stage) {
        switch (stage) {
        case Stage::AllocateBuffers:
            return "AllocateBuffers";
        case Stage::UploadVertices:
            return "UploadVertices";
        case Stage::UploadGaussians:
            return "UploadGaussians";
        case Stage::UploadMaterials:
            return "UploadMaterials";
        case Stage::UploadIndices:
            return "UploadIndices";
        case Stage::UploadTriangles:
            return "UploadTriangles";
        case Stage::UploadTextures:
            return "UploadTextures";
        case Stage::BuildBLAS:
            return "BuildBLAS";
        case Stage::BuildTLAS:
            return "BuildTLAS";
        case Stage::SwapScene:
            return "SwapScene";
        case Stage::Idle:
        default:
            return "Idle";
        }
    };
    const auto uploadStart = std::chrono::steady_clock::now();
    const auto uploadOptions = GetSceneUploadOptions();
    const uint32_t uploadChunkBytes = std::max(64u * 1024u, _settings.maxUploadBytesPerFrame);
    const auto prepared = _pendingSceneUpload.scene.GetPreparedScene();
    if (!prepared) {
        _pendingSceneUpload = {};
        _sceneLoadInProgress = false;
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Pending scene upload lost its prepared scene.";
        return;
    }

    size_t remainingUploadBudget = uploadChunkBytes;
    while (_pendingSceneUpload.active) {
        _sceneLoadStatus.uploadStage = stageLabel(_pendingSceneUpload.stage);
        switch (_pendingSceneUpload.stage) {
        case Stage::AllocateBuffers:
            VESTA_ASSERT_STATE(prepared->IsLoaded(), "AllocateBuffers requires a prepared scene.");
            _pendingSceneUpload.scene.AllocateGpuResources(_device, uploadOptions);
            _pendingSceneUpload.stage = Stage::UploadVertices;
            _sceneLoadStatus.message = "Uploading vertices for " + _pendingSceneUpload.path.filename().string() + "...";
            continue;
        case Stage::UploadVertices: {
            const size_t totalBytes = sizeof(vesta::scene::SceneVertex) * prepared->vertices.size();
            VESTA_ASSERT_STATE(_pendingSceneUpload.vertexOffsetBytes <= totalBytes, "Vertex upload offset exceeded total vertex bytes.");
            if (_pendingSceneUpload.vertexOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadGaussians;
                _sceneLoadStatus.message = "Uploading gaussians for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes =
                std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.vertexOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Vertex, _pendingSceneUpload.vertexOffsetBytes, chunkBytes);
            _pendingSceneUpload.vertexOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadGaussians: {
            const size_t totalBytes = sizeof(vesta::scene::GaussianPrimitive) * prepared->gaussians.size();
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.gaussianOffsetBytes <= totalBytes, "Gaussian upload offset exceeded total gaussian bytes.");
            if (_pendingSceneUpload.gaussianOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadMaterials;
                _sceneLoadStatus.message = "Uploading materials for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes =
                std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.gaussianOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Gaussian, _pendingSceneUpload.gaussianOffsetBytes, chunkBytes);
            _pendingSceneUpload.gaussianOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadMaterials: {
            const size_t totalBytes = sizeof(vesta::scene::SceneMaterial) * prepared->materials.size();
            VESTA_ASSERT_STATE(_pendingSceneUpload.materialOffsetBytes <= totalBytes, "Material upload offset exceeded total material bytes.");
            if (_pendingSceneUpload.materialOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadIndices;
                _sceneLoadStatus.message = "Uploading indices for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes = std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.materialOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Material, _pendingSceneUpload.materialOffsetBytes, chunkBytes);
            _pendingSceneUpload.materialOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadIndices: {
            const size_t totalBytes = sizeof(uint32_t) * prepared->indices.size();
            VESTA_ASSERT_STATE(_pendingSceneUpload.indexOffsetBytes <= totalBytes, "Index upload offset exceeded total index bytes.");
            if (_pendingSceneUpload.indexOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadTriangles;
                _sceneLoadStatus.message = "Uploading triangles for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes = std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.indexOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Index, _pendingSceneUpload.indexOffsetBytes, chunkBytes);
            _pendingSceneUpload.indexOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadTriangles: {
            const size_t totalBytes = sizeof(vesta::scene::SceneTriangle) * prepared->triangles.size();
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.triangleOffsetBytes <= totalBytes, "Triangle upload offset exceeded total triangle bytes.");
            if (_pendingSceneUpload.triangleOffsetBytes >= totalBytes) {
                const SceneUploadContinuation continuation = DecideSceneUploadContinuation(
                    _settings.textureStreamingEnabled,
                    !prepared->textures.empty(),
                    _settings.buildRayTracingStructuresOnLoad,
                    _device.IsRayTracingSupported(),
                    !prepared->indices.empty());
                if (continuation == SceneUploadContinuation::UploadTextures) {
                    _pendingSceneUpload.stage = Stage::UploadTextures;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::UploadingTextures,
                        "Uploading textures for " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTriangles");
                } else if (continuation == SceneUploadContinuation::BuildBLAS) {
                    _pendingSceneUpload.stage = Stage::BuildBLAS;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::BuildingBLAS,
                        "Building BLAS for " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTriangles");
                } else {
                    _pendingSceneUpload.stage = Stage::SwapScene;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::ReadyToSwap,
                        "Finalizing " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTriangles");
                }
                continue;
            }

            const size_t chunkBytes =
                std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.triangleOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Triangle, _pendingSceneUpload.triangleOffsetBytes, chunkBytes);
            _pendingSceneUpload.triangleOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadTextures: {
            const uint32_t textureBudget = std::max(1u, _settings.maxTextureUploadBytesPerFrame);
            size_t uploadedBytes = 0;
            while (_pendingSceneUpload.textureIndex < prepared->textures.size()) {
                const auto& texture = prepared->textures[_pendingSceneUpload.textureIndex];
                if (texture.IsValid()) {
                    const size_t textureBytes = texture.rgba8Pixels.size();
                    if (uploadedBytes > 0 && uploadedBytes + textureBytes > textureBudget) {
                        break;
                    }
                    const auto textureUploadStart = std::chrono::steady_clock::now();
                    _pendingSceneUpload.scene.UploadGpuTexture(_device, _pendingSceneUpload.textureIndex);
                    _pendingSceneUpload.textureUploadMs +=
                        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - textureUploadStart).count();
                    uploadedBytes += textureBytes;
                }
                ++_pendingSceneUpload.textureIndex;
            }

            if (_pendingSceneUpload.textureIndex >= prepared->textures.size()) {
                const SceneUploadContinuation continuation = DecideSceneUploadContinuation(
                    false, false, _settings.buildRayTracingStructuresOnLoad, _device.IsRayTracingSupported(), !prepared->indices.empty());
                if (continuation == SceneUploadContinuation::BuildBLAS) {
                    _pendingSceneUpload.stage = Stage::BuildBLAS;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::BuildingBLAS,
                        "Building BLAS for " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTextures");
                } else {
                    _pendingSceneUpload.stage = Stage::SwapScene;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::ReadyToSwap,
                        "Finalizing " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTextures");
                }
            }
            break;
        }
        case Stage::BuildBLAS:
            _sceneLoadStatus.lastBlockingWait = "FlushUploadBatch before BLAS";
            _device.SetDebugWaitContext(
                "scene=" + _pendingSceneUpload.path.string() + " stage=BuildBLAS state=" + std::to_string(static_cast<uint32_t>(_sceneLoadStatus.state)));
            _device.FlushUploadBatch();
            _pendingSceneUpload.scene.BuildBottomLevelAccelerationStructure(_device);
            _pendingSceneUpload.stage = Stage::BuildTLAS;
            ApplySceneLoadState(_sceneLoadStatus,
                SceneLoadState::BuildingTLAS,
                "Building TLAS for " + _pendingSceneUpload.path.filename().string() + "...",
                "PumpPendingSceneUpload::BuildBLAS");
            break;
        case Stage::BuildTLAS:
            _sceneLoadStatus.lastBlockingWait = "ImmediateSubmit during TLAS";
            _device.SetDebugWaitContext(
                "scene=" + _pendingSceneUpload.path.string() + " stage=BuildTLAS state=" + std::to_string(static_cast<uint32_t>(_sceneLoadStatus.state)));
            _pendingSceneUpload.scene.BuildTopLevelAccelerationStructure(_device);
            _pendingSceneUpload.stage = Stage::SwapScene;
            ApplySceneLoadState(_sceneLoadStatus,
                SceneLoadState::ReadyToSwap,
                "Finalizing " + _pendingSceneUpload.path.filename().string() + "...",
                "PumpPendingSceneUpload::BuildTLAS");
            break;
        case Stage::SwapScene:
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.vertexOffsetBytes >= sizeof(vesta::scene::SceneVertex) * prepared->vertices.size()
                    || prepared->vertices.empty(),
                "SwapScene requires vertex upload completion.");
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.materialOffsetBytes >= sizeof(vesta::scene::SceneMaterial) * prepared->materials.size()
                    || prepared->materials.empty(),
                "SwapScene requires material upload completion.");
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.gaussianOffsetBytes >= sizeof(vesta::scene::GaussianPrimitive) * prepared->gaussians.size()
                    || prepared->gaussians.empty(),
                "SwapScene requires gaussian upload completion.");
            _sceneLoadStatus.lastBlockingWait = "FlushUploadBatch before SwapScene";
            _device.FlushUploadBatch();
            _pendingSceneUpload.uploadMs +=
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
            _sceneLoadStatus.geometryUploadMs = _pendingSceneUpload.uploadMs;
            _sceneLoadStatus.textureUploadMs = _pendingSceneUpload.textureUploadMs;
            _sceneLoadInProgress = false;
            ApplyLoadedScene(std::move(_pendingSceneUpload.scene));
            _pendingSceneUpload = {};
            return;
        case Stage::Idle:
        default:
            _device.FlushUploadBatch();
            _pendingSceneUpload.uploadMs +=
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
            _pendingSceneUpload = {};
            _sceneLoadInProgress = false;
            return;
        }
        break;
    }

    _device.FlushUploadBatch();

    _pendingSceneUpload.uploadMs +=
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
    _sceneLoadStatus.geometryUploadMs = _pendingSceneUpload.uploadMs;
    _sceneLoadStatus.textureUploadMs = _pendingSceneUpload.textureUploadMs;
    _sceneLoadStatus.blasMs = _pendingSceneUpload.scene.GetBottomLevelBuildMs();
    _sceneLoadStatus.tlasMs = _pendingSceneUpload.scene.GetTopLevelBuildMs();
}

void Renderer::PumpVisibilityResults()
{
    if (!_visibilityFuture.valid()) {
        return;
    }

    using namespace std::chrono_literals;
    if (_visibilityFuture.wait_for(0ms) != std::future_status::ready) {
        return;
    }

    VisibilityCullResult result = _visibilityFuture.get();
    _visibilityCullInProgress = false;

    if (_scene.GetPreparedScene() != result.scene) {
        return;
    }

    _visibleSceneToken = std::move(result.scene);
    _visibleSurfaceIndices = std::move(result.visibleSurfaceIndices);
    _frameSnapshot.visibleSet.scene = _visibleSceneToken;
    _frameSnapshot.visibleSet.surfaceIndices = _visibleSurfaceIndices;
}

void Renderer::DispatchVisibilityCullIfNeeded()
{
    if (!_settings.enableFrustumCulling && !_settings.enableDistanceCulling) {
        _visibleSurfaceIndices.clear();
        _visibleSceneToken.reset();
        _frameSnapshot = {};
        _visibilityDirty = false;
        return;
    }
    if (_visibilityCullInProgress || !_visibilityDirty) {
        return;
    }

    std::shared_ptr<const vesta::scene::PreparedScene> prepared = _scene.GetPreparedScene();
    if (!prepared || prepared->surfaces.empty()) {
        _visibleSurfaceIndices.clear();
        _visibleSceneToken = std::move(prepared);
        _frameSnapshot.visibleSet.scene = _visibleSceneToken;
        _frameSnapshot.visibleSet.surfaceIndices.clear();
        _visibilityDirty = false;
        return;
    }

    _visibilityCullInProgress = true;
    _visibilityDirty = false;
    const glm::mat4 viewProjection = _camera.GetViewProjection();
    const glm::vec3 cameraPosition = _camera.GetPosition();
    const bool useFrustumCulling = _settings.enableFrustumCulling;
    const bool useDistanceCulling = _settings.enableDistanceCulling;
    const float distanceCullScale = _settings.distanceCullScale;
    const float sceneRadius = prepared->bounds.radius;
    _visibilityFuture = _jobs.Submit(vesta::core::JobPriority::Normal,
        [this, prepared, viewProjection, cameraPosition, useFrustumCulling, useDistanceCulling, distanceCullScale, sceneRadius]() {
        VisibilityCullResult result;
        result.scene = prepared;

        if (prepared->surfaceBounds.empty()) {
            result.visibleSurfaceIndices.resize(prepared->surfaces.size());
            for (uint32_t surfaceIndex = 0; surfaceIndex < static_cast<uint32_t>(prepared->surfaces.size()); ++surfaceIndex) {
                result.visibleSurfaceIndices[surfaceIndex] = surfaceIndex;
            }
            return result;
        }

        const std::array<glm::vec4, 6> frustumPlanes = ExtractFrustumPlanes(viewProjection);
        if (_jobs.GetWorkerCount() <= 1 || prepared->surfaceBounds.size() < 64) {
            for (uint32_t surfaceIndex = 0; surfaceIndex < static_cast<uint32_t>(prepared->surfaceBounds.size()); ++surfaceIndex) {
                const auto& bounds = prepared->surfaceBounds[surfaceIndex];
                if ((!useFrustumCulling || IsSurfaceVisible(bounds, frustumPlanes))
                    && (!useDistanceCulling || IsSurfaceWithinDistance(bounds, cameraPosition, sceneRadius, distanceCullScale))) {
                    result.visibleSurfaceIndices.push_back(surfaceIndex);
                }
            }
            return result;
        }

        const size_t chunkSize = 64;
        const size_t chunkCount = (prepared->surfaceBounds.size() + chunkSize - 1) / chunkSize;
        std::vector<std::vector<uint32_t>> visibleChunks(chunkCount);
        std::future<void> cullFuture = _jobs.ParallelFor(chunkCount, 1, vesta::core::JobPriority::High, [&](size_t begin, size_t end) {
            for (size_t chunkIndex = begin; chunkIndex < end; ++chunkIndex) {
                const size_t surfaceBegin = chunkIndex * chunkSize;
                const size_t surfaceEnd = std::min(prepared->surfaceBounds.size(), surfaceBegin + chunkSize);
                std::vector<uint32_t>& chunk = visibleChunks[chunkIndex];
                chunk.reserve(surfaceEnd - surfaceBegin);
                for (size_t surfaceIndex = surfaceBegin; surfaceIndex < surfaceEnd; ++surfaceIndex) {
                    const auto& bounds = prepared->surfaceBounds[surfaceIndex];
                    if ((!useFrustumCulling || IsSurfaceVisible(bounds, frustumPlanes))
                        && (!useDistanceCulling || IsSurfaceWithinDistance(bounds, cameraPosition, sceneRadius, distanceCullScale))) {
                        chunk.push_back(static_cast<uint32_t>(surfaceIndex));
                    }
                }
            }
        });
        cullFuture.get();

        size_t totalVisible = 0;
        for (const std::vector<uint32_t>& chunk : visibleChunks) {
            totalVisible += chunk.size();
        }
        result.visibleSurfaceIndices.reserve(totalVisible);
        for (std::vector<uint32_t>& chunk : visibleChunks) {
            result.visibleSurfaceIndices.insert(result.visibleSurfaceIndices.end(), chunk.begin(), chunk.end());
        }
        return result;
    });
}

bool Renderer::LoadSceneResolved(const std::filesystem::path& resolvedPath)
{
    _sceneLoadStatus = SceneLoadStatus{
        .state = SceneLoadState::Parsing,
        .path = resolvedPath,
        .message = "Parsing " + resolvedPath.filename().string() + "...",
    };

    try {
        const auto parseStart = std::chrono::steady_clock::now();
        vesta::scene::Scene loadedScene;
        if (!loadedScene.ParseFromFile(resolvedPath)) {
            _sceneLoadStatus.state = SceneLoadState::Failed;
            _sceneLoadStatus.parseMs =
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();
            _sceneLoadStatus.message = "Failed to load " + resolvedPath.filename().string();
            return false;
        }
        _sceneLoadStatus.parseMs =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();

        ApplySceneLoadState(_sceneLoadStatus,
            SceneLoadState::Preparing,
            "Preparing " + resolvedPath.filename().string() + "...",
            "LoadSceneResolved");
        const auto prepareStart = std::chrono::steady_clock::now();
        if (!loadedScene.PrepareParsedScene()) {
            _sceneLoadStatus.state = SceneLoadState::Failed;
            _sceneLoadStatus.prepareMs =
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prepareStart).count();
            _sceneLoadStatus.message = "Failed to prepare " + resolvedPath.filename().string();
            return false;
        }
        _sceneLoadStatus.prepareMs =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prepareStart).count();

        ApplySceneLoadState(_sceneLoadStatus,
            SceneLoadState::UploadingGeometry,
            "Uploading " + resolvedPath.filename().string() + "...",
            "LoadSceneResolved");
        if (UsesStreamingUpload(_settings)) {
            StartPendingSceneUpload(std::move(loadedScene), _sceneLoadStatus.parseMs, _sceneLoadStatus.prepareMs);
        } else {
            VESTA_ASSERT(!_startupSafeModeActive,
                "Startup safe mode must not synchronously apply scenes from LoadSceneResolved.");
            ApplySceneLoadState(_sceneLoadStatus,
                SceneLoadState::ReadyToSwap,
                "Finalizing " + resolvedPath.filename().string() + "...",
                "LoadSceneResolved");
            ApplyLoadedScene(std::move(loadedScene));
        }
        return true;
    } catch (const std::exception& exception) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Failed to load " + resolvedPath.filename().string() + ": " + exception.what();
        return false;
    } catch (...) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Failed to load " + resolvedPath.filename().string();
        return false;
    }
}

void Renderer::StartPendingSceneUpload(vesta::scene::Scene&& scene, float parseMs, float prepareMs)
{
    _pendingSceneUpload = PendingSceneUpload{
        .scene = std::move(scene),
        .path = _sceneLoadStatus.path,
        .parseMs = parseMs,
        .prepareMs = prepareMs,
        .uploadMs = 0.0f,
        .textureUploadMs = 0.0f,
        .stage = PendingSceneUploadStage::AllocateBuffers,
        .active = true,
    };
    _sceneLoadInProgress = true;
    ValidateSceneLoadTransition(_sceneLoadStatus, SceneLoadState::UploadingGeometry, "StartPendingSceneUpload");
    _sceneLoadStatus.state = SceneLoadState::UploadingGeometry;
    _sceneLoadStatus.parseMs = parseMs;
    _sceneLoadStatus.prepareMs = prepareMs;
    _sceneLoadStatus.geometryUploadMs = 0.0f;
    _sceneLoadStatus.textureUploadMs = 0.0f;
    _sceneLoadStatus.blasMs = 0.0f;
    _sceneLoadStatus.tlasMs = 0.0f;
    _sceneLoadStatus.lastBlockingWait.clear();
    _sceneLoadStatus.message = "Allocating GPU buffers for " + _pendingSceneUpload.path.filename().string() + "...";
}

void Renderer::ApplyLoadedScene(vesta::scene::Scene&& scene)
{
    if (_startupSafeModeActive) {
        VESTA_ASSERT(UsesStreamingUpload(_settings), "Startup safe mode requires streaming upload before ApplyLoadedScene.");
    }
    vesta::scene::Scene previousScene = std::move(_scene);
    _scene = std::move(scene);
    if (!_scene.GetVertexBuffer()) {
        _sceneLoadStatus.lastBlockingWait = "Scene::UploadToGpu";
        _device.SetDebugWaitContext("scene=" + _scene.GetSourcePath().string() + " stage=ApplyLoadedScene::UploadToGpu");
        _scene.UploadToGpu(_device, GetSceneUploadOptions());
        _sceneLoadStatus.geometryUploadMs = _scene.GetGeometryUploadMs();
        _sceneLoadStatus.textureUploadMs = _scene.GetTextureUploadMs();
    }
    _sceneLoadStatus.blasMs = _scene.GetBottomLevelBuildMs();
    _sceneLoadStatus.tlasMs = _scene.GetTopLevelBuildMs();

    if (_settings.autoFocusSceneOnLoad) {
        _camera.Focus(_scene.GetBounds().center, _scene.GetBounds().radius);
    }
    if (!_scene.HasTrainedGaussians() && _scene.SupportsRealtimeGaussianSorting()) {
        _scene.ResortGaussians(_device, _camera);
    }
    ResetAccumulation();
    _visibilityDirty = true;
    _visibleSurfaceIndices.clear();
    _visibleSceneToken.reset();
    _frameSnapshot = {};
    ClearSelection();
    if (_scene.HasTrainedGaussians()) {
        _gaussianInteractivePreviewFramesRemaining = GaussianInteractivePreviewFrameBudget(_scene);
    }

    if (!previousScene.GetSourcePath().empty()) {
        if (_settings.deferOldSceneDestruction) {
            _retiredScenes.push_back(RetiredSceneEntry{
                .scene = std::move(previousScene),
                .safeFrameNumber = _frameNumber + kFrameOverlap,
            });
        } else {
            _device.WaitIdle();
            previousScene.DestroyGpu(_device);
        }
    }

    ApplySceneLoadState(
        _sceneLoadStatus, SceneLoadState::Ready, "Loaded " + _scene.GetSourcePath().filename().string(), "ApplyLoadedScene");
    _sceneLoadStatus.path = _scene.GetSourcePath();
    _sceneLoadStatus.uploadStage.clear();
    _sceneLoadStatus.lastBlockingWait.clear();
}

void Renderer::ReleaseRetiredScenes()
{
    while (!_retiredScenes.empty() && _retiredScenes.front().safeFrameNumber <= _frameNumber) {
        _retiredScenes.front().scene.DestroyGpu(_device);
        _retiredScenes.pop_front();
    }
}

RendererFrameContext& Renderer::GetCurrentFrame()
{
    return _frames[_frameNumber % kFrameOverlap];
}

RenderGraph Renderer::BuildFrameGraph(uint32_t swapchainImageIndex)
{
    RebuildPassExecutionPlan();

    RenderGraph graph;
    const bool useGeometryPass = NeedsGeometryPass(_settings);
    const bool useShadowMapPass = useGeometryPass && _settings.enableShadowMap && _scene.HasRasterGeometry();
    const bool useOverdrawPass = useGeometryPass && _settings.debugView == RendererDebugView::Overdraw && _scene.HasRasterGeometry();
    const bool useDeferredPass = NeedsDeferredPass(_settings);
    const bool useRayEffectsPass = useDeferredPass && IsRayEffectsRequested(_settings)
        && _device.GetRayTracingSupport().rayQueryFeatures.rayQuery == VK_TRUE && _scene.HasRayTracingScene();
    const bool useRestirPass = IsRestirRequested(_settings) && _restirReservoirBuffer;
    const bool useRestirResolvePass = useDeferredPass && _settings.enableRestirDi && useRestirPass;
    const bool useDdgiProbeUpdatePass = _settings.enableDdgi && _ddgiIrradianceBuffer && _ddgiVisibilityBuffer
        && _device.GetRayTracingSupport().rayQueryFeatures.rayQuery == VK_TRUE && _scene.HasRayTracingScene();
    const bool useGaussianPass = NeedsGaussianPass(_settings);
    const bool usePathTracePass = NeedsPathTracePass(_settings);
    const bool usePathDenoisePass = NeedsPathDenoisePass(_settings);
    const bool useTemporalAAPass = NeedsTemporalAAPass(_settings);
    const bool useBloomPass = _settings.enableBloom && _settings.bloomIntensity > 0.0f && (useDeferredPass || usePathTracePass);
    const bool useOfficialGaussianPass = useGaussianPass && _scene.HasTrainedGaussians() && !IsGaussianInteractivePreviewActive();
    const bool useLegacyGaussianPass = useGaussianPass && (!_scene.HasTrainedGaussians() || IsGaussianInteractivePreviewActive());

    const VkExtent2D swapchainExtent = _device.GetSwapchainExtent();
    const VkExtent3D renderExtent{ swapchainExtent.width, swapchainExtent.height, 1 };
    const bool useTemporalUpscaler = _settings.enableTemporalUpscaler && useDeferredPass && !useGaussianPass;
    const VkExtent3D rasterRenderExtent = useTemporalUpscaler
        ? ScaleExtent(renderExtent, _settings.temporalUpscalerScale)
        : renderExtent;

    // These logical resources describe the full frame. The graph decides which
    // concrete VkImage each handle resolves to for this frame execution.
    RendererGraphResources resources;
    resources.swapchainTarget =
        graph.ImportTexture("SwapchainTarget", _device.GetSwapchainImageHandle(swapchainImageIndex), ResourceUsage::Undefined);

    ImageDesc gbufferDesc{};
    gbufferDesc.extent = rasterRenderExtent;
    gbufferDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    gbufferDesc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    gbufferDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    gbufferDesc.registerBindlessStorage = true;

    ImageDesc depthDesc{};
    depthDesc.extent = rasterRenderExtent;
    depthDesc.format = VK_FORMAT_D32_SFLOAT;
    depthDesc.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    depthDesc.registerBindlessSampled = true;

    ImageDesc shadowDesc = depthDesc;
    shadowDesc.extent = VkExtent3D{ _settings.shadowMapSize, _settings.shadowMapSize, 1 };
    shadowDesc.debugName = "ShadowMap";

    ImageDesc storageDesc{};
    storageDesc.extent = renderExtent;
    storageDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    storageDesc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    storageDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        | VK_IMAGE_USAGE_SAMPLED_BIT;
    storageDesc.registerBindlessStorage = true;

    ImageDesc pathTraceDesc = storageDesc;
    pathTraceDesc.extent = ScaleExtent(renderExtent, _settings.pathTraceResolutionScale);
    ImageDesc rasterStorageDesc = storageDesc;
    rasterStorageDesc.extent = rasterRenderExtent;

    if (useGeometryPass) {
        resources.gbufferAlbedo = graph.CreateTexture("GBuffer.Albedo", gbufferDesc);
        resources.gbufferNormal = graph.CreateTexture("GBuffer.NormalRoughness", gbufferDesc);
        resources.gbufferMaterial = graph.CreateTexture("GBuffer.Material", gbufferDesc);
        resources.gbufferDebug = graph.CreateTexture("GBuffer.Debug", gbufferDesc);
        resources.gbufferMotion = graph.CreateTexture("GBuffer.Motion", gbufferDesc);
        resources.gbufferReactive = graph.CreateTexture("GBuffer.Reactive", gbufferDesc);
        resources.sceneDepth = graph.CreateTexture("SceneDepth", depthDesc);
    }
    if (useShadowMapPass) {
        resources.shadowMap = graph.CreateTexture("ShadowMap", shadowDesc);
    }
    if (useOverdrawPass) {
        resources.overdraw = graph.CreateTexture("RasterOverdraw", rasterStorageDesc);
    }
    if (useRayEffectsPass) {
        ImageDesc rayEffectsDesc = rasterStorageDesc;
        rayEffectsDesc.debugName = "RayEffects";
        if (_settings.rtHalfResolution) {
            rayEffectsDesc.extent.width = std::max(1u, (rasterRenderExtent.width + 1u) / 2u);
            rayEffectsDesc.extent.height = std::max(1u, (rasterRenderExtent.height + 1u) / 2u);
        }
        resources.rayEffects = graph.CreateTexture("RayEffects", rayEffectsDesc);
        if (_settings.enableRtReflections) {
            ImageDesc rayReflectionDesc = rayEffectsDesc;
            rayReflectionDesc.debugName = "RayEffects.Reflection";
            resources.rayReflection = graph.CreateTexture("RayEffects.Reflection", rayReflectionDesc);
        }
        if (_settings.enableRtGlobalIllumination) {
            ImageDesc rayGiDesc = rayEffectsDesc;
            rayGiDesc.debugName = "RayEffects.GI";
            resources.rayGlobalIllumination = graph.CreateTexture("RayEffects.GI", rayGiDesc);
        }
    }
    if (useRestirResolvePass) {
        ImageDesc restirResolveDesc = rasterStorageDesc;
        restirResolveDesc.debugName = "RestirDI.Resolve";
        resources.restirDirectLighting = graph.CreateTexture("RestirDI.Resolve", restirResolveDesc);
    }
    if (useDeferredPass) {
        resources.deferredLighting = graph.CreateTexture("DeferredLighting", rasterStorageDesc);
        resources.deferredLightingDebug = graph.CreateTexture("DeferredLighting.DebugAOV", rasterStorageDesc);
    }
    if (useTemporalAAPass) {
        resources.temporalLighting = graph.CreateTexture("TemporalLighting", storageDesc);
    }
    if (usePathTracePass) {
        resources.pathTraceOutput = graph.CreateTexture("PathTraceOutput", pathTraceDesc);
    }
    if (usePathDenoisePass) {
        resources.pathTraceNormalGuide = graph.CreateTexture("PathTraceGuide.Normal", pathTraceDesc);
        resources.pathTraceDepthGuide = graph.CreateTexture("PathTraceGuide.Depth", pathTraceDesc);
        resources.pathTraceDenoised = graph.CreateTexture("PathTraceDenoised", pathTraceDesc);
    }
    if (useBloomPass) {
        ImageDesc bloomHalfDesc = storageDesc;
        bloomHalfDesc.debugName = "Bloom.Half";
        bloomHalfDesc.extent = ScaleExtent(renderExtent, 0.5f);
        resources.bloomHalf = graph.CreateTexture("Bloom.Half", bloomHalfDesc);

        ImageDesc bloomQuarterDesc = storageDesc;
        bloomQuarterDesc.debugName = "Bloom.Quarter";
        bloomQuarterDesc.extent = ScaleExtent(renderExtent, 0.25f);
        resources.bloomQuarter = graph.CreateTexture("Bloom.Quarter", bloomQuarterDesc);

        ImageDesc bloomOutputDesc = bloomHalfDesc;
        bloomOutputDesc.debugName = "Bloom.Output";
        resources.bloomOutput = graph.CreateTexture("Bloom.Output", bloomOutputDesc);
    }
    if (useGaussianPass) {
        resources.gaussianAccum = graph.CreateTexture("GaussianAccum", storageDesc);
        resources.gaussianReveal = graph.CreateTexture("GaussianReveal", storageDesc);
    }
    if (useOfficialGaussianPass) {
        resources.gaussianDebug = graph.CreateTexture("GaussianDebug", storageDesc);
    }

    for (RegisteredPassEntry* entry : _passExecutionPlan) {
        const std::string_view id = entry->id;
        if (id == "geometry-raster" && !useGeometryPass) {
            continue;
        }
        if (id == "shadow-map" && !useShadowMapPass) {
            continue;
        }
        if (id == "overdraw" && !useOverdrawPass) {
            continue;
        }
        if (id == "ray-effects" && !useRayEffectsPass) {
            continue;
        }
        if (id == "restir-di" && !useRestirPass) {
            continue;
        }
        if (id == "restir-di-resolve" && !useRestirResolvePass) {
            continue;
        }
        if (id == "ddgi-probe-update" && !useDdgiProbeUpdatePass) {
            continue;
        }
        if (id == "deferred-lighting" && !useDeferredPass) {
            continue;
        }
        if (id == "gaussian-splat" && !useLegacyGaussianPass) {
            continue;
        }
        if (id == "official-gaussian-raster" && !useOfficialGaussianPass) {
            continue;
        }
        if (id == "path-tracer" && !usePathTracePass) {
            continue;
        }
        if (id == "path-denoise" && !usePathDenoisePass) {
            continue;
        }
        if (id == "temporal-aa" && !useTemporalAAPass) {
            continue;
        }
        if ((id == "bloom-extract" || id == "bloom-downsample" || id == "bloom-upsample") && !useBloomPass) {
            continue;
        }

        if (entry->configure) {
            entry->configure(*entry->pass, resources);
        }
        graph.AddPass(*entry->pass);
    }

    graph.SetFinalUsage(resources.swapchainTarget, ResourceUsage::Present);
    return graph;
}

Renderer::RegisteredPassEntry* Renderer::FindPassEntry(std::string_view id)
{
    const auto it = std::find_if(_passRegistry.begin(), _passRegistry.end(), [id](const RegisteredPassEntry& entry) {
        return entry.id == id;
    });
    return it != _passRegistry.end() ? &(*it) : nullptr;
}

const Renderer::RegisteredPassEntry* Renderer::FindPassEntry(std::string_view id) const
{
    const auto it = std::find_if(_passRegistry.begin(), _passRegistry.end(), [id](const RegisteredPassEntry& entry) {
        return entry.id == id;
    });
    return it != _passRegistry.end() ? &(*it) : nullptr;
}

} // namespace vesta::render
