//> includes
#include <vesta/render/vulkan/vk_engine.h>

#include <cassert>

#include <SDL.h>
#include <SDL_syswm.h>

#include <fmt/format.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

#include <vesta/core/debug.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <windows.h>
#pragma comment(lib, "Comdlg32.lib")
#endif

VestaEngine* loadedEngine = nullptr;

VestaEngine& VestaEngine::Get() { return *loadedEngine; }

namespace {
constexpr size_t kMaxRecentScenePaths = 5;
constexpr std::array<std::string_view, 17> kBenchmarkPassNames{
    "GeometryRasterPass",
    "ShadowMapPass",
    "OverdrawPass",
    "RayEffectsPass",
    "ReSTIR DI CandidatePass",
    "ReSTIR DI ResolvePass",
    "DDGI Probe UpdatePass",
    "DeferredLightingPass",
    "GaussianSplatPass",
    "OfficialGaussianRasterPass",
    "PathTracerPass",
    "PathDenoisePass",
    "TemporalAAPass",
    "Bloom ExtractPass",
    "Bloom DownsamplePass",
    "Bloom UpsamplePass",
    "CompositePass",
};

struct BenchmarkScenePreset {
    const char* label;
    const char* path;
    const char* purpose;
};

constexpr std::array<BenchmarkScenePreset, 10> kBenchmarkScenePresets{
    BenchmarkScenePreset{ "Sponza Atrium", "assets/benchmark_scenes/sponza/sponza.obj", "Large raster/PBR scene" },
    BenchmarkScenePreset{ "Amazon Bistro 5.2 Exterior", "assets/benchmark_scenes/Bistro_v5_2/BistroExterior.fbx", "Large outdoor stress scene" },
    BenchmarkScenePreset{ "Amazon Bistro 5.2 Interior", "assets/benchmark_scenes/Bistro_v5_2/BistroInterior.fbx", "Interior lighting stress scene" },
    BenchmarkScenePreset{ "Amazon Bistro 5.2 Interior Wine", "assets/benchmark_scenes/Bistro_v5_2/BistroInterior_Wine.fbx", "Interior glass and absorption stress scene" },
    BenchmarkScenePreset{ "San Miguel", "assets/benchmark_scenes/san_miguel/san-miguel.obj", "Large textured GI scene" },
    BenchmarkScenePreset{ "San Miguel Low Poly", "assets/benchmark_scenes/san_miguel/san-miguel-low-poly.obj", "Large textured GI scene" },
    BenchmarkScenePreset{ "Cornell Box", "assets/benchmark_scenes/cornell_box/cornell-box.obj", "Reference path-tracing scene" },
    BenchmarkScenePreset{ "Stanford Bunny", "assets/benchmark_scenes/stanford_bunny/bunny/reconstruction/bun_zipper.ply", "Classic mesh validation model" },
    BenchmarkScenePreset{ "Stanford Dragon", "assets/benchmark_scenes/stanford_dragon/dragon_recon/dragon_vrip_res2.ply", "High-detail mesh validation model" },
    BenchmarkScenePreset{ "Stanford Buddha", "assets/benchmark_scenes/stanford_buddha/happy_recon/happy_vrip_res2.ply", "High-detail mesh validation model" },
};

std::string NormalizedAssetPathKey(const std::filesystem::path& path)
{
    std::string key = path.lexically_normal().generic_string();
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return key;
}

void ApplyBenchmarkSceneLightingPreset(vesta::render::RendererSettings& settings,
    const std::filesystem::path& path,
    const vesta::scene::SceneBounds* bounds = nullptr)
{
    const std::string key = NormalizedAssetPathKey(path);
    const auto contains = [&](std::string_view token) {
        return key.find(token) != std::string::npos;
    };
    const auto boundsRadius = [&]() {
        return bounds != nullptr ? std::max(bounds->radius, 1.0f) : 1.0f;
    };
    const auto boundsCenter = [&]() {
        return bounds != nullptr ? bounds->center : glm::vec3(0.0f);
    };
    const auto scaledIntensity = [&](float base, float scale) {
        const float radius = boundsRadius();
        return bounds != nullptr ? std::max(base, radius * radius * scale) : base;
    };
    const auto directionToCenter = [&](glm::vec3 position, glm::vec3 fallback) {
        const glm::vec3 delta = boundsCenter() - position;
        return glm::length(delta) > 1.0e-4f ? glm::normalize(delta) : glm::normalize(fallback);
    };
    const bool isBenchmarkPreset = contains("cornell_box") || contains("cornell-box") || contains("bistro_v5_2")
        || contains("bistrointerior") || contains("bistroexterior") || contains("bistro_interior")
        || contains("interior.obj") || contains("bistro_exterior") || contains("exterior.obj") || contains("sponza")
        || contains("san_miguel") || contains("stanford_bunny") || contains("stanford_dragon") || contains("stanford_buddha")
        || contains("damagedhelmet") || contains("damaged_helmet");
    if (!isBenchmarkPreset) {
        return;
    }

    settings.animateDirectionalLight = false;
    settings.enablePointLight = false;
    settings.enableSpotLight = false;
    settings.enableAreaLight = false;
    settings.enableContactShadows = true;
    settings.environmentIntensity = 2.0f;
    settings.environmentDiffuseStrength = 1.0f;
    settings.environmentSpecularStrength = 1.0f;
    settings.lightDirectionAndIntensity.w = 1.0f;

    if (contains("cornell_box") || contains("cornell-box")) {
        settings.environmentPreset = 0u;
        settings.environmentIntensity = 0.015f;
        settings.environmentDiffuseStrength = 0.04f;
        settings.environmentSpecularStrength = 0.02f;
        settings.lightDirectionAndIntensity = glm::vec4(-0.15f, -1.0f, -0.10f, 0.0f);
        settings.directionalLightColor = glm::vec4(1.0f, 0.96f, 0.88f, 0.0f);
        settings.enableAreaLight = true;
        settings.areaLightPositionAndIntensity = glm::vec4(-0.234f, 5.319f, -3.043f, 18.0f);
        settings.areaLightNormalAndSize = glm::vec4(0.0f, -1.0f, 0.0f, 1.25f);
        settings.areaLightColor = glm::vec4(1.0f, 0.92f, 0.78f, 0.0f);
        settings.cameraExposureEv = 0.0f;
        settings.enableSsao = false;
        settings.enableSsgi = false;
        return;
    }

    if (contains("bistrointerior") || contains("bistro_interior") || contains("interior.obj")) {
        settings.environmentPreset = 0u;
        settings.lightDirectionAndIntensity = glm::vec4(-0.35f, -1.0f, -0.25f, 1.0f);
        settings.directionalLightColor = glm::vec4(1.0f, 0.9f, 0.78f, 0.0f);
        settings.enablePointLight = true;
        settings.pointLightPositionAndIntensity = glm::vec4(0.0f, 2.4f, 0.5f, 8.0f);
        settings.pointLightColor = glm::vec4(1.0f, 0.78f, 0.52f, 0.0f);
        settings.enableSpotLight = true;
        settings.spotLightPositionAndIntensity = glm::vec4(0.0f, 3.2f, 2.0f, 18.0f);
        settings.spotLightDirectionAndAngle = glm::vec4(0.0f, -0.85f, -0.45f, 34.0f);
        settings.spotLightColor = glm::vec4(1.0f, 0.86f, 0.68f, 0.0f);
        if (bounds != nullptr) {
            const glm::vec3 center = boundsCenter();
            const float radius = boundsRadius();
            settings.pointLightPositionAndIntensity = glm::vec4(center + glm::vec3(0.0f, radius * 0.12f, 0.0f),
                scaledIntensity(8.0f, 0.012f));
            const glm::vec3 spotPosition = center + glm::vec3(0.0f, radius * 0.34f, radius * 0.22f);
            settings.spotLightPositionAndIntensity = glm::vec4(spotPosition, scaledIntensity(18.0f, 0.020f));
            settings.spotLightDirectionAndAngle = glm::vec4(directionToCenter(spotPosition, glm::vec3(0.0f, -0.85f, -0.45f)), 38.0f);
        }
        return;
    }

    if (contains("bistroexterior") || contains("bistro_exterior") || contains("exterior.obj")) {
        settings.environmentPreset = 3u;
        settings.lightDirectionAndIntensity = glm::vec4(-0.55f, -1.0f, -0.25f, 1.0f);
        settings.directionalLightColor = glm::vec4(1.0f, 0.88f, 0.70f, 0.0f);
        settings.enableSpotLight = true;
        settings.spotLightPositionAndIntensity = glm::vec4(0.0f, 3.0f, 0.0f, 8.0f);
        settings.spotLightDirectionAndAngle = glm::vec4(0.0f, -1.0f, 0.0f, 42.0f);
        settings.spotLightColor = glm::vec4(1.0f, 0.80f, 0.58f, 0.0f);
        if (bounds != nullptr) {
            const glm::vec3 center = boundsCenter();
            const float radius = boundsRadius();
            const glm::vec3 spotPosition = center + glm::vec3(-radius * 0.18f, radius * 0.32f, radius * 0.18f);
            settings.spotLightPositionAndIntensity = glm::vec4(spotPosition, scaledIntensity(8.0f, 0.010f));
            settings.spotLightDirectionAndAngle = glm::vec4(directionToCenter(spotPosition, glm::vec3(0.0f, -1.0f, 0.0f)), 46.0f);
        }
        return;
    }

    if (contains("sponza")) {
        settings.environmentPreset = 1u;
        settings.lightDirectionAndIntensity = glm::vec4(-0.42f, -1.0f, -0.35f, 1.0f);
        settings.directionalLightColor = glm::vec4(1.0f, 0.90f, 0.76f, 0.0f);
        return;
    }

    if (contains("san_miguel")) {
        settings.environmentPreset = 1u;
        settings.lightDirectionAndIntensity = glm::vec4(-0.50f, -1.0f, -0.18f, 1.0f);
        settings.directionalLightColor = glm::vec4(1.0f, 0.86f, 0.64f, 0.0f);
        settings.enablePointLight = true;
        settings.pointLightPositionAndIntensity = glm::vec4(0.0f, 3.0f, 0.0f, 5.0f);
        settings.pointLightColor = glm::vec4(1.0f, 0.72f, 0.46f, 0.0f);
        if (bounds != nullptr) {
            const glm::vec3 center = boundsCenter();
            const float radius = boundsRadius();
            settings.pointLightPositionAndIntensity =
                glm::vec4(center + glm::vec3(0.0f, radius * 0.18f, 0.0f), scaledIntensity(5.0f, 0.008f));
        }
        return;
    }

    if (contains("stanford_bunny") || contains("stanford_dragon") || contains("stanford_buddha")) {
        settings.environmentPreset = 0u;
        settings.lightDirectionAndIntensity = glm::vec4(-0.35f, -1.0f, -0.25f, 1.0f);
        settings.directionalLightColor = glm::vec4(1.0f, 0.96f, 0.88f, 0.0f);
        settings.enablePointLight = true;
        settings.pointLightPositionAndIntensity = glm::vec4(1.8f, 2.5f, 2.2f, 4.0f);
        settings.pointLightColor = glm::vec4(0.70f, 0.84f, 1.0f, 0.0f);
        settings.enableAreaLight = true;
        settings.areaLightPositionAndIntensity = glm::vec4(-1.5f, 3.0f, 1.0f, 5.0f);
        settings.areaLightNormalAndSize = glm::vec4(0.35f, -0.9f, -0.25f, 1.4f);
        settings.areaLightColor = glm::vec4(1.0f, 0.86f, 0.66f, 0.0f);
    }
}

#if defined(NDEBUG)
constexpr bool bUseValidationLayers = false;
#else
// Debug builds default to validation on. Flip this if you want to profile without validation overhead.
constexpr bool bUseValidationLayers = true;
#endif

void CheckImGuiVkResult(VkResult err)
{
    if (err != VK_SUCCESS) {
        fmt::println(stderr, "ImGui Vulkan error: {}", string_VkResult(err));
        abort();
    }
}

const char* PresetLabel(vesta::render::RendererPreset preset)
{
    switch (preset) {
    case vesta::render::RendererPreset::Performance:
        return "Performance";
    case vesta::render::RendererPreset::Balanced:
        return "Balanced";
    case vesta::render::RendererPreset::Quality:
        return "Quality";
    case vesta::render::RendererPreset::Recommended:
    default:
        return "Recommended";
    }
}

const char* DisplayModeLabel(vesta::render::RendererDisplayMode mode)
{
    switch (mode) {
    case vesta::render::RendererDisplayMode::DeferredLighting:
        return "Raster";
    case vesta::render::RendererDisplayMode::RayTracing:
        return "RayTracing";
    case vesta::render::RendererDisplayMode::Gaussian:
        return "Gaussian";
    case vesta::render::RendererDisplayMode::PathTrace:
        return "PathTrace";
    case vesta::render::RendererDisplayMode::Composite:
    default:
        return "Composite";
    }
}

std::optional<std::filesystem::path> BenchmarkSceneHdri(const std::filesystem::path& scenePath)
{
    const std::string key = NormalizedAssetPathKey(scenePath);
    if (key.find("bistro_v5_2") != std::string::npos) {
        const std::filesystem::path hdri = scenePath.parent_path() / "san_giuseppe_bridge_4k.hdr";
        if (std::filesystem::exists(hdri)) {
            return hdri;
        }
    }

    if (key.find("damagedhelmet") != std::string::npos || key.find("damaged_helmet") != std::string::npos) {
        const std::filesystem::path hdri = std::filesystem::path("assets") / "benchmark_scenes" / "Bistro_v5_2"
            / "san_giuseppe_bridge_4k.hdr";
        if (std::filesystem::exists(hdri)) {
            return hdri;
        }
    }
    return std::nullopt;
}

const char* AntiAliasingModeLabel(vesta::render::AntiAliasingMode mode)
{
    switch (mode) {
    case vesta::render::AntiAliasingMode::None:
        return "None";
    case vesta::render::AntiAliasingMode::FXAA:
        return "FXAA";
    case vesta::render::AntiAliasingMode::TAA:
        return "TAA";
    case vesta::render::AntiAliasingMode::TAAU:
        return "TAAU";
    case vesta::render::AntiAliasingMode::MSAA:
        return "MSAA";
    case vesta::render::AntiAliasingMode::DLSS:
        return "DLSS";
    default:
        return "Unknown";
    }
}

void ApplyAntiAliasingMode(vesta::render::RendererSettings& settings, vesta::render::AntiAliasingMode mode)
{
    settings.antiAliasingMode = mode;
    settings.enableFxaa = mode == vesta::render::AntiAliasingMode::FXAA;
    settings.enableTaa = mode == vesta::render::AntiAliasingMode::TAA || mode == vesta::render::AntiAliasingMode::TAAU;
    settings.enableTemporalUpscaler = mode == vesta::render::AntiAliasingMode::TAAU;
    settings.enableMsaa = mode == vesta::render::AntiAliasingMode::MSAA;
    settings.enableDlss = mode == vesta::render::AntiAliasingMode::DLSS;
}

const char* SceneKindLabel(vesta::scene::SceneKind kind)
{
    switch (kind) {
    case vesta::scene::SceneKind::Mesh:
        return "Mesh";
    case vesta::scene::SceneKind::PointCloud:
        return "Point Cloud";
    case vesta::scene::SceneKind::Gaussian:
        return "Gaussian";
    case vesta::scene::SceneKind::Empty:
    default:
        return "Empty";
    }
}

const char* PresentModeLabel(VkPresentModeKHR mode)
{
    switch (mode) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        return "Immediate";
    case VK_PRESENT_MODE_MAILBOX_KHR:
        return "Mailbox";
    case VK_PRESENT_MODE_FIFO_KHR:
        return "FIFO";
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
        return "FIFO Relaxed";
    default:
        return "Other";
    }
}

const char* EnvironmentPresetLabel(uint32_t preset)
{
    switch (preset) {
    case 1u:
        return "Sunset";
    case 2u:
        return "Night";
    case 3u:
        return "Forest";
    case 0u:
    default:
        return "Studio";
    }
}

const char* PathTraceBackendLabel(vesta::render::PathTraceBackend backend)
{
    switch (backend) {
    case vesta::render::PathTraceBackend::HardwareRT:
        return "HardwareRT";
    case vesta::render::PathTraceBackend::Compute:
        return "Compute";
    case vesta::render::PathTraceBackend::Auto:
    default:
        return "Auto";
    }
}

const char* PathTraceDebugViewLabel(vesta::render::PathTraceDebugView view)
{
    switch (view) {
    case vesta::render::PathTraceDebugView::Albedo:
        return "Albedo";
    case vesta::render::PathTraceDebugView::Normal:
        return "Normal";
    case vesta::render::PathTraceDebugView::Depth:
        return "Depth";
    case vesta::render::PathTraceDebugView::Direct:
        return "Direct";
    case vesta::render::PathTraceDebugView::Indirect:
        return "Indirect";
    case vesta::render::PathTraceDebugView::RayCountHeatmap:
        return "Ray Count Heatmap";
    case vesta::render::PathTraceDebugView::DiffuseBounce:
        return "Diffuse Bounce";
    case vesta::render::PathTraceDebugView::SpecularBounce:
        return "Specular Bounce";
    case vesta::render::PathTraceDebugView::Throughput:
        return "Throughput";
    case vesta::render::PathTraceDebugView::Pdf:
        return "PDF";
    case vesta::render::PathTraceDebugView::Final:
    default:
        return "Final";
    }
}

const char* RendererDebugViewLabel(vesta::render::RendererDebugView view)
{
    switch (view) {
    case vesta::render::RendererDebugView::Albedo:
        return "Albedo";
    case vesta::render::RendererDebugView::Normal:
        return "Normal";
    case vesta::render::RendererDebugView::WorldPosition:
        return "World Position";
    case vesta::render::RendererDebugView::Depth:
        return "Linear Depth";
    case vesta::render::RendererDebugView::UV:
        return "UV";
    case vesta::render::RendererDebugView::MaterialId:
        return "Material ID";
    case vesta::render::RendererDebugView::ObjectId:
        return "Object ID";
    case vesta::render::RendererDebugView::Roughness:
        return "Roughness";
    case vesta::render::RendererDebugView::Metallic:
        return "Metallic";
    case vesta::render::RendererDebugView::Emissive:
        return "Emissive";
    case vesta::render::RendererDebugView::AmbientOcclusion:
        return "Ambient Occlusion";
    case vesta::render::RendererDebugView::MotionVector:
        return "Motion Vector";
    case vesta::render::RendererDebugView::DirectLighting:
        return "Direct Lighting";
    case vesta::render::RendererDebugView::IndirectLighting:
        return "Indirect Lighting";
    case vesta::render::RendererDebugView::Reflection:
        return "Reflection";
    case vesta::render::RendererDebugView::DenoisedResult:
        return "Denoised Result";
    case vesta::render::RendererDebugView::DifferenceFromReference:
        return "Difference from Reference";
    case vesta::render::RendererDebugView::Wireframe:
        return "Wireframe";
    case vesta::render::RendererDebugView::MipLevel:
        return "Mip Level";
    case vesta::render::RendererDebugView::ShadowMap:
        return "Shadow Map";
    case vesta::render::RendererDebugView::Overdraw:
        return "Overdraw";
    case vesta::render::RendererDebugView::TemporalHistoryColor:
        return "Temporal History Color";
    case vesta::render::RendererDebugView::TemporalHistoryDepth:
        return "Temporal History Depth";
    case vesta::render::RendererDebugView::TemporalReprojection:
        return "Temporal Reprojection";
    case vesta::render::RendererDebugView::TemporalDisocclusion:
        return "Temporal Disocclusion";
    case vesta::render::RendererDebugView::TemporalJitter:
        return "Temporal Jitter";
    case vesta::render::RendererDebugView::ContactShadow:
        return "Contact Shadow";
    case vesta::render::RendererDebugView::ShadowCascade:
        return "Shadow Cascade";
    case vesta::render::RendererDebugView::RayTracedGlobalIllumination:
        return "Ray-Traced GI";
    case vesta::render::RendererDebugView::FinalColor:
    default:
        return "Final Color";
    }
}

const char* GaussianDebugViewLabel(vesta::render::GaussianDebugView view)
{
    switch (view) {
    case vesta::render::GaussianDebugView::Alpha:
        return "Alpha";
    case vesta::render::GaussianDebugView::Revealage:
        return "Revealage";
    case vesta::render::GaussianDebugView::OverdrawHeatmap:
        return "Overdraw Heatmap";
    case vesta::render::GaussianDebugView::Depth:
        return "Depth";
    case vesta::render::GaussianDebugView::TileOccupancy:
        return "Tile Occupancy";
    case vesta::render::GaussianDebugView::SplatRadius:
        return "Splat Radius";
    case vesta::render::GaussianDebugView::ContributionCount:
        return "Contribution Count";
    case vesta::render::GaussianDebugView::SplatId:
        return "Splat ID";
    case vesta::render::GaussianDebugView::ShBand:
        return "SH Band";
    case vesta::render::GaussianDebugView::Covariance:
        return "Covariance";
    case vesta::render::GaussianDebugView::RasterDepth:
        return "Raster Depth";
    case vesta::render::GaussianDebugView::CompositionMask:
        return "Composition Mask";
    case vesta::render::GaussianDebugView::DepthDifference:
        return "Depth Difference";
    case vesta::render::GaussianDebugView::Final:
    default:
        return "Final";
    }
}

bool PathTracePassVisible(vesta::render::RendererDisplayMode mode)
{
    return mode == vesta::render::RendererDisplayMode::PathTrace || mode == vesta::render::RendererDisplayMode::Composite;
}

float PathTraceProgressFraction(uint32_t frameIndex, uint32_t targetFrames)
{
    const uint32_t target = std::max(targetFrames, 1u);
    return std::clamp(static_cast<float>(frameIndex) / static_cast<float>(target), 0.0f, 1.0f);
}

std::string PathTraceProgressLabel(uint32_t frameIndex, uint32_t targetFrames)
{
    const uint32_t target = std::max(targetFrames, 1u);
    const float percent = PathTraceProgressFraction(frameIndex, target) * 100.0f;
    return fmt::format("{} / {} ({:.0f}%)", std::min(frameIndex, target), target, percent);
}

void DrawPathTraceProgressBar(const vesta::render::RendererSettings& settings, uint32_t frameIndex, const ImVec2& size)
{
    const std::string label = PathTraceProgressLabel(frameIndex, settings.pathTraceTargetFrames);
    ImGui::ProgressBar(PathTraceProgressFraction(frameIndex, settings.pathTraceTargetFrames), size, label.c_str());
}

const char* CompareModeLabel(vesta::render::CompareMode mode)
{
    switch (mode) {
    case vesta::render::CompareMode::RasterPathSplit:
        return "Raster / Path Split";
    case vesta::render::CompareMode::DifferenceHeatmap:
        return "Difference Heatmap";
    case vesta::render::CompareMode::Off:
    default:
        return "Off";
    }
}

const char* RasterPipelineModeLabel(vesta::render::RasterPipelineMode mode)
{
    switch (mode) {
    case vesta::render::RasterPipelineMode::Forward:
        return "Forward";
    case vesta::render::RasterPipelineMode::Deferred:
    default:
        return "Deferred";
    }
}

const char* ToneMappingModeLabel(vesta::render::ToneMappingMode mode)
{
    switch (mode) {
    case vesta::render::ToneMappingMode::None:
        return "None";
    case vesta::render::ToneMappingMode::Reinhard:
        return "Reinhard";
    case vesta::render::ToneMappingMode::ACES:
    default:
        return "ACES";
    }
}

std::optional<size_t> BenchmarkPassIndex(std::string_view passName)
{
    for (size_t index = 0; index < kBenchmarkPassNames.size(); ++index) {
        if (kBenchmarkPassNames[index] == passName) {
            return index;
        }
    }
    return std::nullopt;
}

const char* SceneLoadStateLabel(vesta::render::SceneLoadState state)
{
    switch (state) {
    case vesta::render::SceneLoadState::Parsing:
        return "Parsing";
    case vesta::render::SceneLoadState::Preparing:
        return "Preparing";
    case vesta::render::SceneLoadState::UploadingGeometry:
        return "Uploading Geometry";
    case vesta::render::SceneLoadState::UploadingTextures:
        return "Uploading Textures";
    case vesta::render::SceneLoadState::BuildingBLAS:
        return "Building BLAS";
    case vesta::render::SceneLoadState::BuildingTLAS:
        return "Building TLAS";
    case vesta::render::SceneLoadState::ReadyToSwap:
        return "Ready To Swap";
    case vesta::render::SceneLoadState::Ready:
        return "Ready";
    case vesta::render::SceneLoadState::Failed:
        return "Failed";
    case vesta::render::SceneLoadState::Cancelled:
        return "Cancelled";
    case vesta::render::SceneLoadState::Idle:
    default:
        return "Idle";
    }
}

const char* ResourceUsageLabel(vesta::render::ResourceUsage usage)
{
    switch (usage) {
    case vesta::render::ResourceUsage::ColorAttachmentWrite:
        return "Color Write";
    case vesta::render::ResourceUsage::DepthAttachmentWrite:
        return "Depth Write";
    case vesta::render::ResourceUsage::DepthRead:
        return "Depth Read";
    case vesta::render::ResourceUsage::SampledRead:
        return "Sampled";
    case vesta::render::ResourceUsage::StorageRead:
        return "Storage Read";
    case vesta::render::ResourceUsage::StorageWrite:
        return "Storage Write";
    case vesta::render::ResourceUsage::TransferSrc:
        return "Transfer Src";
    case vesta::render::ResourceUsage::TransferDst:
        return "Transfer Dst";
    case vesta::render::ResourceUsage::Present:
        return "Present";
    case vesta::render::ResourceUsage::Undefined:
    default:
        return "Undefined";
    }
}

const char* VkFormatLabel(VkFormat format)
{
    switch (format) {
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return "RGBA16F";
    case VK_FORMAT_D32_SFLOAT:
        return "D32F";
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "BGRA8";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "RGBA8";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "RGBA8_sRGB";
    case VK_FORMAT_UNDEFINED:
    default:
        return "Unknown";
    }
}

VkImageLayout PreviewLayoutForResourceUsage(vesta::render::ResourceUsage usage)
{
    switch (usage) {
    case vesta::render::ResourceUsage::DepthRead:
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    case vesta::render::ResourceUsage::StorageRead:
    case vesta::render::ResourceUsage::StorageWrite:
        return VK_IMAGE_LAYOUT_GENERAL;
    case vesta::render::ResourceUsage::SampledRead:
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    default:
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }
}

bool IsPreviewableFrameTexture(const vesta::render::RenderDevice& device,
    const vesta::render::RenderGraphPassTiming::ResourceAccess& access)
{
    if (!access.image || access.imported) {
        return false;
    }
    const VkImageLayout layout = PreviewLayoutForResourceUsage(access.usage);
    if (layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        return false;
    }
    return (device.GetImageResource(access.image).desc.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
}

ImVec2 FitPreviewSize(VkExtent3D extent, float maxSize)
{
    const float width = static_cast<float>(std::max(extent.width, 1u));
    const float height = static_cast<float>(std::max(extent.height, 1u));
    ImVec2 previewSize(maxSize, maxSize);
    if (width > height) {
        previewSize.y = maxSize * (height / width);
    } else {
        previewSize.x = maxSize * (width / height);
    }
    return previewSize;
}

struct FrameTimingStats {
    float averageMs{ 0.0f };
    float minMs{ 0.0f };
    float maxMs{ 0.0f };
    float onePercentLowFps{ 0.0f };
};

FrameTimingStats CalculateFrameTimingStats(const std::array<float, 240>& history, size_t count)
{
    FrameTimingStats stats{};
    if (count == 0) {
        return stats;
    }

    std::vector<float> samples;
    samples.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (history[i] > 0.0f) {
            samples.push_back(history[i]);
        }
    }
    if (samples.empty()) {
        return stats;
    }

    const float total = std::accumulate(samples.begin(), samples.end(), 0.0f);
    stats.averageMs = total / static_cast<float>(samples.size());
    stats.minMs = *std::min_element(samples.begin(), samples.end());
    stats.maxMs = *std::max_element(samples.begin(), samples.end());

    std::sort(samples.begin(), samples.end(), std::greater<float>());
    const size_t worstCount = std::max<size_t>(1, samples.size() / 100);
    const float worstTotal = std::accumulate(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(worstCount), 0.0f);
    const float onePercentLowMs = worstTotal / static_cast<float>(worstCount);
    stats.onePercentLowFps = onePercentLowMs > 0.0f ? 1000.0f / onePercentLowMs : 0.0f;
    return stats;
}

float TotalGpuMs(const std::vector<vesta::render::RenderGraphPassTiming>& timings)
{
    float total = 0.0f;
    for (const auto& timing : timings) {
        if (timing.gpuTimingValid) {
            total += timing.gpuMs;
        }
    }
    return total;
}

const vesta::render::RenderGraphPassTiming* SlowestGpuPass(const std::vector<vesta::render::RenderGraphPassTiming>& timings)
{
    const vesta::render::RenderGraphPassTiming* slowest = nullptr;
    for (const auto& timing : timings) {
        if (!timing.gpuTimingValid) {
            continue;
        }
        if (slowest == nullptr || timing.gpuMs > slowest->gpuMs) {
            slowest = &timing;
        }
    }
    return slowest;
}

bool RuntimeWarningCooldownElapsed(int frameNumber, int lastWarningFrame, int cooldownFrames)
{
    return frameNumber - lastWarningFrame >= cooldownFrames;
}

double MiB(uint64_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

uint64_t BufferSizeBytes(const vesta::render::RenderDevice& device, vesta::render::BufferHandle handle)
{
    return handle ? static_cast<uint64_t>(device.GetBufferResource(handle).desc.size) : 0ull;
}

uint64_t TextureAssetBytes(const vesta::scene::SceneTextureAsset& texture)
{
    return static_cast<uint64_t>(texture.width) * texture.height * 4ull;
}

uint32_t FullMipCount(uint32_t width, uint32_t height)
{
    uint32_t levels = 1;
    uint32_t size = std::max(width, height);
    while (size > 1) {
        size >>= 1u;
        ++levels;
    }
    return levels;
}

std::string BufferUsageLabel(VkBufferUsageFlags usage)
{
    std::string label;
    auto append = [&](VkBufferUsageFlagBits flag, std::string_view name) {
        if ((usage & flag) == 0) {
            return;
        }
        if (!label.empty()) {
            label += " | ";
        }
        label += name;
    };
    append(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, "Vertex");
    append(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, "Index");
    append(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "Storage");
    append(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "Uniform");
    append(VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, "Indirect");
    append(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "CopySrc");
    append(VK_BUFFER_USAGE_TRANSFER_DST_BIT, "CopyDst");
    append(VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, "DeviceAddress");
    append(VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, "AS");
    append(VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, "ASInput");
    return label.empty() ? "Unknown" : label;
}

std::string BufferGroupLabel(const char* name, VkBufferUsageFlags usage)
{
    if ((usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) != 0) {
        return "Acceleration";
    }
    if ((usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0) {
        return "Uniform";
    }
    if ((usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0) {
        return "Vertex";
    }
    if ((usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0) {
        return "Index";
    }
    if (std::string_view(name).find("Readback") != std::string_view::npos ||
        ((usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0 && (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) == 0)) {
        return "Readback";
    }
    if ((usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        return "Storage";
    }
    return "Other";
}

std::string TextureSemanticLabel(const vesta::scene::Scene& scene, size_t textureIndex)
{
    std::string label;
    auto append = [&](std::string_view semantic) {
        if (label.find(semantic) != std::string::npos) {
            return;
        }
        if (!label.empty()) {
            label += " | ";
        }
        label += semantic;
    };
    auto match = [&](uint32_t index, std::string_view semantic) {
        if (index == textureIndex) {
            append(semantic);
        }
    };

    for (const auto& material : scene.GetMaterials()) {
        match(material.textureIndices0.x, "BaseColor");
        match(material.textureIndices0.y, "MetallicRoughness");
        match(material.textureIndices0.z, "Normal");
        match(material.textureIndices0.w, "Occlusion");
        match(material.textureIndices1.x, "Emissive");
    }
    return label.empty() ? "Sampled" : label;
}

void DrawRenderGraphResourceList(const char* label,
    const std::vector<vesta::render::RenderGraphPassTiming::ResourceAccess>& accesses)
{
    if (!ImGui::TreeNode(label)) {
        return;
    }

    if (ImGui::BeginTable(label, 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Usage");
        ImGui::TableSetupColumn("Format", ImGuiTableColumnFlags_WidthFixed, 82.0f);
        ImGui::TableSetupColumn("Resolution", ImGuiTableColumnFlags_WidthFixed, 96.0f);
        ImGui::TableSetupColumn("Scale", ImGuiTableColumnFlags_WidthFixed, 72.0f);
        ImGui::TableHeadersRow();
        for (const auto& access : accesses) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(access.name.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(ResourceUsageLabel(access.usage));
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(VkFormatLabel(access.format));
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%ux%u", access.extent.width, access.extent.height);
            ImGui::TableSetColumnIndex(4);
            ImGui::TextUnformatted("full-res");
        }
        ImGui::EndTable();
    }
    ImGui::TreePop();
}

void DrawRenderGraphBarrierList(const std::vector<vesta::render::RenderGraphPassTiming::BarrierInfo>& barriers)
{
    if (barriers.empty()) {
        ImGui::TextDisabled("Barriers: none");
        return;
    }

    if (ImGui::BeginTable("BarrierTransitions", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Resource");
        ImGui::TableSetupColumn("From");
        ImGui::TableSetupColumn("To");
        ImGui::TableHeadersRow();
        for (const auto& barrier : barriers) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(barrier.name.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(ResourceUsageLabel(barrier.fromUsage));
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(ResourceUsageLabel(barrier.toUsage));
        }
        ImGui::EndTable();
    }
}

void DrawBufferResourceRow(const char* name,
    const vesta::render::RenderDevice& device,
    vesta::render::BufferHandle handle,
    uint64_t logicalBytes = 0)
{
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(name);
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%s", handle ? "Resident" : "Missing");
    ImGui::TableSetColumnIndex(2);
    if (handle) {
        const auto& buffer = device.GetBufferResource(handle);
        ImGui::Text("%.2f MiB", MiB(static_cast<uint64_t>(buffer.desc.size)));
    } else {
        ImGui::TextUnformatted("-");
    }
    ImGui::TableSetColumnIndex(3);
    if (logicalBytes > 0) {
        ImGui::Text("%.2f MiB", MiB(logicalBytes));
    } else {
        ImGui::TextUnformatted("-");
    }
    ImGui::TableSetColumnIndex(4);
    if (handle) {
        const auto& buffer = device.GetBufferResource(handle);
        ImGui::TextUnformatted(BufferGroupLabel(name, buffer.desc.usage).c_str());
    } else {
        ImGui::TextUnformatted("-");
    }
    ImGui::TableSetColumnIndex(5);
    if (handle) {
        const auto& buffer = device.GetBufferResource(handle);
        ImGui::TextUnformatted(BufferUsageLabel(buffer.desc.usage).c_str());
    } else {
        ImGui::TextUnformatted("-");
    }
    ImGui::TableSetColumnIndex(6);
    if (handle) {
        const auto& buffer = device.GetBufferResource(handle);
        if (buffer.bindless.storageBuffer != vesta::render::kInvalidResourceIndex) {
            ImGui::Text("%u", buffer.bindless.storageBuffer);
        } else {
            ImGui::TextUnformatted("-");
        }
    } else {
        ImGui::TextUnformatted("-");
    }
    ImGui::TableSetColumnIndex(7);
    if (handle) {
        ImGui::Text("%u", handle.index);
    } else {
        ImGui::TextUnformatted("-");
    }
}

bool UseAsyncSceneLoading(const vesta::render::RendererSettings& settings)
{
    return settings.preferAsyncSceneLoading && settings.sceneUploadMode != vesta::render::SceneUploadMode::Synchronous;
}

const char* SceneUploadModeLabel(vesta::render::SceneUploadMode mode)
{
    switch (mode) {
    case vesta::render::SceneUploadMode::Streaming:
        return "Streaming";
    case vesta::render::SceneUploadMode::AsyncParseSyncUpload:
        return "Async Parse + Sync Upload";
    case vesta::render::SceneUploadMode::Synchronous:
    default:
        return "Synchronous";
    }
}

bool LooksLikeMeshPly(const std::filesystem::path& path)
{
    if (!std::filesystem::is_regular_file(path)) {
        return false;
    }

    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.starts_with("element face ")) {
            std::istringstream stream(line);
            std::string elementToken;
            std::string faceToken;
            size_t faceCount = 0;
            stream >> elementToken >> faceToken >> faceCount;
            return faceCount > 0;
        }
        if (line == "end_header" || line == "end_header\r") {
            break;
        }
    }
    return false;
}

void ApplySceneModeInference(vesta::render::RendererSettings& settings, const std::filesystem::path& path)
{
    if (std::filesystem::is_directory(path)) {
        settings.displayMode = vesta::render::RendererDisplayMode::Gaussian;
        settings.enableRaster = false;
        settings.enableGaussian = true;
        settings.enablePathTracing = false;
        return;
    }

    const std::filesystem::path extension = path.extension();
    if (extension == ".ply" || extension == ".PLY") {
        if (LooksLikeMeshPly(path)) {
            settings.displayMode = vesta::render::RendererDisplayMode::DeferredLighting;
            settings.enableRaster = true;
            settings.enableGaussian = true;
        } else {
            settings.displayMode = vesta::render::RendererDisplayMode::Gaussian;
            settings.enableRaster = false;
            settings.enableGaussian = true;
        }
        settings.enablePathTracing = false;
        return;
    }

    if (extension == ".glb" || extension == ".GLB" || extension == ".gltf" || extension == ".GLTF" || extension == ".fbx" ||
        extension == ".FBX" || extension == ".obj" || extension == ".OBJ") {
        settings.displayMode = vesta::render::RendererDisplayMode::DeferredLighting;
        settings.enableRaster = true;
        settings.enableGaussian = true;
    }
}

std::string CsvEscape(std::string value)
{
    const bool needsQuotes = value.find_first_of(",\"\n\r") != std::string::npos;
    if (!needsQuotes) {
        return value;
    }

    size_t quotePosition = 0;
    while ((quotePosition = value.find('"', quotePosition)) != std::string::npos) {
        value.insert(quotePosition, 1, '"');
        quotePosition += 2;
    }

    return "\"" + value + "\"";
}

std::string JsonEscape(std::string_view value)
{
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (char character : value) {
        switch (character) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += character;
            break;
        }
    }
    return escaped;
}

std::filesystem::path NormalizeScenePath(std::filesystem::path path)
{
    if (path.empty()) {
        return path;
    }

    std::wstring native = path.native();
    while (native.size() > 1 && (native.back() == L'\\' || native.back() == L'/')) {
        const bool keepDriveRoot = native.size() == 3 && native[1] == L':' && (native[2] == L'\\' || native[2] == L'/');
        if (keepDriveRoot) {
            break;
        }
        native.pop_back();
    }

    return std::filesystem::path(native).lexically_normal();
}

std::string ScenePathLabel(const std::filesystem::path& rawPath)
{
    const std::filesystem::path path = NormalizeScenePath(rawPath);
    if (path.empty()) {
        return "(unnamed scene)";
    }

    const std::string filename = path.filename().string();
    if (!filename.empty()) {
        return filename;
    }

    const std::string stem = path.stem().string();
    if (!stem.empty()) {
        return stem;
    }

    const std::string full = path.string();
    if (!full.empty()) {
        return full;
    }

    return "(unnamed scene)";
}

std::string MakeTimestampedLogLine(std::string_view message)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &nowTime);
#else
    localtime_r(&nowTime, &localTime);
#endif

    std::ostringstream stream;
    stream << '[' << std::put_time(&localTime, "%H:%M:%S") << "] " << message;
    return stream.str();
}

std::string TrimCopy(std::string_view value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

std::vector<std::string> SplitPhysicalLogLines(const std::string& line)
{
    std::vector<std::string> result;
    std::istringstream stream(line);
    std::string physicalLine;
    while (std::getline(stream, physicalLine)) {
        result.push_back(std::move(physicalLine));
    }
    if (result.empty()) {
        result.push_back(line);
    }
    return result;
}

struct ShaderCompilerDiagnostic {
    std::string severity;
    std::string file;
    int line{ 0 };
    std::string message;
};

std::filesystem::path ResolveShaderDiagnosticPath(const ShaderCompilerDiagnostic& diagnostic)
{
    if (diagnostic.file.empty() || diagnostic.file == "(compiler)") {
        return {};
    }

    std::filesystem::path filePath(diagnostic.file);
    std::vector<std::filesystem::path> candidates;
    if (filePath.is_absolute()) {
        candidates.push_back(filePath);
    } else {
        candidates.push_back(std::filesystem::current_path() / filePath);
        candidates.push_back(std::filesystem::current_path() / "shaders" / filePath.filename());
    }

    for (const auto& candidate : candidates) {
        std::error_code error;
        const std::filesystem::path normalized = std::filesystem::weakly_canonical(candidate, error);
        const std::filesystem::path resolved = error ? candidate.lexically_normal() : normalized;
        if (std::filesystem::exists(resolved)) {
            return resolved;
        }
    }
    return {};
}

bool OpenSourceFileAtLine(const std::filesystem::path& path, int line)
{
    if (path.empty() || !std::filesystem::exists(path)) {
        return false;
    }

#if defined(_WIN32)
    std::wstring normalized = path.wstring();
    for (wchar_t& character : normalized) {
        if (character == L'\\') {
            character = L'/';
        }
    }
    const std::wstring vscodeUri = L"vscode://file/" + normalized + L":" + std::to_wstring(std::max(line, 1));
    const HINSTANCE vscodeResult =
        ShellExecuteW(nullptr, L"open", vscodeUri.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if (reinterpret_cast<intptr_t>(vscodeResult) > 32) {
        return true;
    }

    const HINSTANCE fileResult =
        ShellExecuteW(nullptr, L"open", path.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<intptr_t>(fileResult) > 32;
#else
    (void)line;
    return false;
#endif
}

std::optional<ShaderCompilerDiagnostic> ParseShaderCompilerDiagnostic(std::string_view line)
{
    std::string_view severity = "ERROR";
    size_t marker = line.find("ERROR:");
    if (marker == std::string_view::npos) {
        marker = line.find("WARNING:");
        severity = "WARNING";
    }
    if (marker == std::string_view::npos) {
        return std::nullopt;
    }

    const std::string restStorage = TrimCopy(line.substr(marker + severity.size() + 1));
    const std::string_view rest{ restStorage };
    for (size_t cursor = 0; cursor < rest.size();) {
        const size_t colon = rest.find(':', cursor);
        if (colon == std::string_view::npos || colon + 1 >= rest.size()) {
            break;
        }

        size_t digitEnd = colon + 1;
        while (digitEnd < rest.size() && std::isdigit(static_cast<unsigned char>(rest[digitEnd]))) {
            ++digitEnd;
        }
        if (digitEnd > colon + 1 && digitEnd < rest.size() && rest[digitEnd] == ':') {
            ShaderCompilerDiagnostic diagnostic;
            diagnostic.severity = std::string(severity);
            diagnostic.file = TrimCopy(rest.substr(0, colon));
            diagnostic.line = std::max(0, std::stoi(std::string(rest.substr(colon + 1, digitEnd - colon - 1))));
            diagnostic.message = TrimCopy(rest.substr(digitEnd + 1));
            return diagnostic;
        }

        cursor = colon + 1;
    }

    ShaderCompilerDiagnostic diagnostic;
    diagnostic.severity = std::string(severity);
    diagnostic.file = "(compiler)";
    diagnostic.message = TrimCopy(rest);
    return diagnostic;
}

std::optional<ImVec2> ProjectWorldToViewport(const Camera& camera, glm::vec3 position, ImVec2 origin, ImVec2 size)
{
    if (size.x <= 1.0f || size.y <= 1.0f) {
        return std::nullopt;
    }

    const glm::vec4 clip = camera.GetViewProjection() * glm::vec4(position, 1.0f);
    if (clip.w <= 0.001f) {
        return std::nullopt;
    }

    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.x < -1.2f || ndc.x > 1.2f || ndc.y < -1.2f || ndc.y > 1.2f || ndc.z < -0.05f || ndc.z > 1.05f) {
        return std::nullopt;
    }

    return ImVec2{
        origin.x + (ndc.x * 0.5f + 0.5f) * size.x,
        origin.y + (ndc.y * 0.5f + 0.5f) * size.y,
    };
}

glm::vec3 RotateByGaussianQuaternion(const glm::vec4& quaternion, glm::vec3 vector)
{
    const glm::vec4 q = glm::length(quaternion) > 1.0e-4f ? quaternion / glm::length(quaternion) : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    const glm::vec3 t = 2.0f * glm::cross(glm::vec3(q), vector);
    return vector + q.w * t + glm::cross(glm::vec3(q), t);
}

glm::vec3 GaussianBaseColor(const vesta::scene::GaussianPrimitive& gaussian)
{
    constexpr float kShC0 = 0.28209479177387814f;
    const glm::vec3 color = glm::vec3(gaussian.shCoefficients[0]) * kShC0 + glm::vec3(0.5f);
    return glm::clamp(color, glm::vec3(0.0f), glm::vec3(1.0f));
}

std::filesystem::path MakeTimestampedCapturePath(std::string_view prefix, std::string_view extension)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &nowTime);
#else
    localtime_r(&localTime, &nowTime);
#endif

    std::ostringstream stream;
    stream << prefix << '_' << std::put_time(&localTime, "%Y%m%d_%H%M%S") << extension;
    return std::filesystem::path("out/captures") / stream.str();
}

bool WriteRenderGraphDot(
    const std::filesystem::path& path, const std::vector<vesta::render::RenderGraphPassTiming>& timings)
{
    std::filesystem::path outputPath = path;
    if (outputPath.is_relative()) {
        outputPath = std::filesystem::current_path() / outputPath;
    }
    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path());
    }

    std::ofstream output(outputPath, std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }

    output << "digraph VestaRenderGraph {\n"
           << "  rankdir=LR;\n"
           << "  node [shape=box, style=rounded];\n";

    for (size_t passIndex = 0; passIndex < timings.size(); ++passIndex) {
        const auto& timing = timings[passIndex];
        output << "  pass" << passIndex << " [label=\"" << JsonEscape(timing.name) << "\\nCPU "
               << timing.cpuMs << " ms\\nGPU ";
        if (timing.gpuTimingValid) {
            output << timing.gpuMs;
        } else {
            output << "n/a";
        }
        output << " ms\"];\n";
    }

    for (size_t readerIndex = 0; readerIndex < timings.size(); ++readerIndex) {
        const auto& reader = timings[readerIndex];
        for (const auto& input : reader.inputs) {
            std::optional<size_t> writerIndex;
            for (size_t candidateIndex = 0; candidateIndex < readerIndex; ++candidateIndex) {
                const auto& candidate = timings[candidateIndex];
                const bool writesResource = std::any_of(candidate.outputs.begin(), candidate.outputs.end(), [&](const auto& output) {
                    return output.name == input.name;
                });
                if (writesResource) {
                    writerIndex = candidateIndex;
                }
            }
            if (writerIndex.has_value()) {
                output << "  pass" << *writerIndex << " -> pass" << readerIndex << " [label=\""
                       << JsonEscape(input.name) << "\"];\n";
            }
        }
    }

    output << "}\n";
    return true;
}
}

void VestaEngine::init(const EngineLaunchOptions& options)
{
    _launchOptions = options;
    _showDebugUi = options.enableUi && options.showDebugUi;
    _startupState = {};

    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;
    log_startup_event("Engine init begin");

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    // Add more SDL window flags here if you want resizable, fullscreen, borderless, high-DPI behavior, etc.
    SDL_WindowFlags window_flags = static_cast<SDL_WindowFlags>(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow(
        // Window title is purely cosmetic and safe to customize.
        "Vesta Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);
    log_startup_event("SDL window created");

    // The renderer owns Vulkan. The engine layer only orchestrates windowing,
    // input, and debug UI around it.
    init_renderer();
    if (_launchOptions.enableUi) {
        init_imgui();
        log_startup_event("ImGui initialized");
    }

    // everything went fine
    _isInitialized = true;
    log_startup_event("Engine init complete");
}

void VestaEngine::init_renderer()
{
    _renderer.Initialize(_window, _windowExtent, bUseValidationLayers);
    log_startup_event("Renderer initialized");

    if (_launchOptions.startupPreset.has_value()) {
        _renderer.ApplyPreset(*_launchOptions.startupPreset);
    }

    auto& settings = _renderer.GetSettings();
    bool resetAccumulation = false;
    if (_launchOptions.startupDisplayMode.has_value()) {
        vesta::render::ApplyDisplayModePassSelection(settings, *_launchOptions.startupDisplayMode);
        resetAccumulation = true;
    }
    if (_launchOptions.startupCompareMode.has_value()) {
        settings.compareMode = *_launchOptions.startupCompareMode;
        settings.displayMode = vesta::render::RendererDisplayMode::Composite;
        resetAccumulation = true;
    }
    if (_launchOptions.startupDebugView.has_value()) {
        vesta::render::SelectRendererDebugView(settings, *_launchOptions.startupDebugView);
        resetAccumulation = true;
    }
    if (_launchOptions.startupPathTraceDebugView.has_value()) {
        vesta::render::SelectPathTraceDebugView(settings, *_launchOptions.startupPathTraceDebugView);
        resetAccumulation = true;
    }
    if (_launchOptions.startupGaussianDebugView.has_value()) {
        settings.gaussianDebugView = *_launchOptions.startupGaussianDebugView;
        resetAccumulation = true;
    }
    if (_launchOptions.startupCompareSplitPosition.has_value()) {
        settings.compareSplitPosition = std::clamp(*_launchOptions.startupCompareSplitPosition, 0.05f, 0.95f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupCompareDifferenceScale.has_value()) {
        settings.compareDifferenceScale = std::max(*_launchOptions.startupCompareDifferenceScale, 0.1f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupPathTraceBackend.has_value()) {
        settings.pathTraceBackend = *_launchOptions.startupPathTraceBackend;
        resetAccumulation = true;
    }
    if (_launchOptions.startupPathTraceResolutionScale.has_value()) {
        settings.pathTraceResolutionScale = std::clamp(*_launchOptions.startupPathTraceResolutionScale, 0.25f, 1.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupPathTraceNextEventEstimation.has_value()) {
        settings.pathTraceNextEventEstimation = *_launchOptions.startupPathTraceNextEventEstimation;
        resetAccumulation = true;
    }
    if (_launchOptions.startupPathTraceRussianRoulette.has_value()) {
        settings.pathTraceRussianRoulette = *_launchOptions.startupPathTraceRussianRoulette;
        resetAccumulation = true;
    }
    if (_launchOptions.startupPathTraceRussianRouletteDepth.has_value()) {
        settings.pathTraceRussianRouletteDepth = std::clamp(*_launchOptions.startupPathTraceRussianRouletteDepth, 1u, 12u);
        resetAccumulation = true;
    }
    if (_launchOptions.startupPathTraceFireflyClamp.has_value()) {
        settings.pathTraceFireflyClamp = std::clamp(*_launchOptions.startupPathTraceFireflyClamp, 0.0f, 64.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupGlobalIlluminationEnabled.has_value()) {
        settings.enableGlobalIllumination = *_launchOptions.startupGlobalIlluminationEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupAmbientOcclusionEnabled.has_value()) {
        settings.enableAmbientOcclusion = *_launchOptions.startupAmbientOcclusionEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupAntiAliasingMode.has_value()) {
        ApplyAntiAliasingMode(settings, *_launchOptions.startupAntiAliasingMode);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsaoEnabled.has_value()) {
        settings.enableSsao = *_launchOptions.startupSsaoEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsaoRadius.has_value()) {
        settings.ssaoRadius = std::clamp(*_launchOptions.startupSsaoRadius, 0.05f, 5.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsaoIntensity.has_value()) {
        settings.ssaoIntensity = std::clamp(*_launchOptions.startupSsaoIntensity, 0.0f, 4.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupTaaEnabled.has_value()) {
        settings.enableTaa = *_launchOptions.startupTaaEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupTaaFeedback.has_value()) {
        settings.taaFeedback = std::clamp(*_launchOptions.startupTaaFeedback, 0.0f, 0.98f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupTemporalUpscalerEnabled.has_value()) {
        settings.enableTemporalUpscaler = *_launchOptions.startupTemporalUpscalerEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupTemporalUpscalerScale.has_value()) {
        settings.temporalUpscalerScale = std::clamp(*_launchOptions.startupTemporalUpscalerScale, 0.25f, 1.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupTemporalUpscalerSharpness.has_value()) {
        settings.temporalUpscalerSharpness = std::clamp(*_launchOptions.startupTemporalUpscalerSharpness, 0.0f, 1.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsrEnabled.has_value()) {
        settings.enableSsr = *_launchOptions.startupSsrEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsrMaxDistance.has_value()) {
        settings.ssrMaxDistance = std::clamp(*_launchOptions.startupSsrMaxDistance, 0.5f, 100.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsrThickness.has_value()) {
        settings.ssrThickness = std::clamp(*_launchOptions.startupSsrThickness, 0.01f, 2.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsrIntensity.has_value()) {
        settings.ssrIntensity = std::clamp(*_launchOptions.startupSsrIntensity, 0.0f, 2.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsgiEnabled.has_value()) {
        settings.enableSsgi = *_launchOptions.startupSsgiEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsgiRadius.has_value()) {
        settings.ssgiRadius = std::clamp(*_launchOptions.startupSsgiRadius, 0.05f, 8.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsgiIntensity.has_value()) {
        settings.ssgiIntensity = std::clamp(*_launchOptions.startupSsgiIntensity, 0.0f, 2.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupSsgiSamples.has_value()) {
        settings.ssgiSampleCount = std::clamp(*_launchOptions.startupSsgiSamples, 4u, 16u);
        resetAccumulation = true;
    }
    if (_launchOptions.startupDdgiEnabled.has_value()) {
        settings.enableDdgi = *_launchOptions.startupDdgiEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupVoxelGiEnabled.has_value()) {
        settings.enableVoxelGi = *_launchOptions.startupVoxelGiEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRestirDiEnabled.has_value()) {
        settings.enableRestirDi = *_launchOptions.startupRestirDiEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRestirGiEnabled.has_value()) {
        settings.enableRestirGi = *_launchOptions.startupRestirGiEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRestirPtEnabled.has_value()) {
        settings.enableRestirPt = *_launchOptions.startupRestirPtEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupMeshletCullingEnabled.has_value()) {
        settings.enableMeshletCulling = *_launchOptions.startupMeshletCullingEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRtShadowsEnabled.has_value()) {
        settings.enableRtShadows = *_launchOptions.startupRtShadowsEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRtAoEnabled.has_value()) {
        settings.enableRtAmbientOcclusion = *_launchOptions.startupRtAoEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRtReflectionsEnabled.has_value()) {
        settings.enableRtReflections = *_launchOptions.startupRtReflectionsEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRtGiEnabled.has_value()) {
        settings.enableRtGlobalIllumination = *_launchOptions.startupRtGiEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRtHalfResolution.has_value()) {
        settings.rtHalfResolution = *_launchOptions.startupRtHalfResolution;
        resetAccumulation = true;
    }
    if (_launchOptions.startupRtMaxRayDistance.has_value()) {
        settings.rtMaxRayDistance = std::clamp(*_launchOptions.startupRtMaxRayDistance, 0.1f, 100000.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupRtAoRadius.has_value()) {
        settings.rtAoRadius = std::clamp(*_launchOptions.startupRtAoRadius, 0.05f, 32.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupMotionBlurEnabled.has_value()) {
        settings.enableMotionBlur = *_launchOptions.startupMotionBlurEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupMotionBlurStrength.has_value()) {
        settings.motionBlurStrength = std::clamp(*_launchOptions.startupMotionBlurStrength, 0.0f, 2.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupEnvironmentPreset.has_value()) {
        settings.environmentPreset = std::clamp(*_launchOptions.startupEnvironmentPreset, 0u, 3u);
        resetAccumulation = true;
    }
    if (_launchOptions.startupEnvironmentDiffuseStrength.has_value()) {
        settings.environmentDiffuseStrength = std::clamp(*_launchOptions.startupEnvironmentDiffuseStrength, 0.0f, 2.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupEmissiveIntensity.has_value()) {
        settings.emissiveIntensity = std::clamp(*_launchOptions.startupEmissiveIntensity, 0.0f, 64.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupEnvironmentSpecularStrength.has_value()) {
        settings.environmentSpecularStrength = std::clamp(*_launchOptions.startupEnvironmentSpecularStrength, 0.0f, 2.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupExternalHdriPath.has_value()) {
        apply_external_hdri_path(*_launchOptions.startupExternalHdriPath);
        resetAccumulation = true;
    }
    if (_launchOptions.startupPcssShadowsEnabled.has_value()) {
        settings.enablePcssShadows = *_launchOptions.startupPcssShadowsEnabled;
        resetAccumulation = true;
    }
    if (_launchOptions.startupShadowFilterRadius.has_value()) {
        settings.shadowFilterRadius = std::clamp(*_launchOptions.startupShadowFilterRadius, 0.5f, 4.0f);
        resetAccumulation = true;
    }
    if (_launchOptions.startupCameraPosition.has_value()) {
        _renderer.GetCamera().SetPosition(*_launchOptions.startupCameraPosition);
        resetAccumulation = true;
    }
    if (_launchOptions.startupCameraRotation.has_value()) {
        _renderer.GetCamera().SetRotationDegrees(*_launchOptions.startupCameraRotation);
        resetAccumulation = true;
    }
    if (_launchOptions.benchmark.has_value()) {
        settings.frameTimingCapture = true;
        settings.benchmarkOverlay = false;
    }
    if (resetAccumulation) {
        _renderer.ResetAccumulation();
    }

    if (_launchOptions.safeStartupMode) {
        _startupState.safeOverridesActive = true;
        _startupState.savedSettings = settings;
        settings = ApplyStartupSafeRendererSettings(settings, _launchOptions);
        _renderer.SetStartupSafeModeActive(true);
        VESTA_ASSERT(settings.sceneUploadMode == vesta::render::SceneUploadMode::Streaming,
            "Safe startup mode must force streaming upload.");
        VESTA_ASSERT(!settings.enablePathTracing && !settings.enableGaussian,
            "Safe startup mode must disable heavy startup rendering paths.");
        VESTA_ASSERT(!settings.buildRayTracingStructuresOnLoad,
            "Safe startup mode must defer RT structure build until after first present.");
        _renderer.ResetAccumulation();
        log_startup_event("Applied safe startup overrides");
    }

    if (_launchOptions.reloadShadersOnStartup) {
        const bool reloaded = _renderer.ReloadShaders();
        log_startup_event(reloaded ? "Startup shader reload complete: " + _renderer.GetLastShaderReloadMessage()
                                   : "Startup shader reload failed: " + _renderer.GetLastShaderReloadMessage());
    }

    auto requestSceneLoad = [&](const std::filesystem::path& path) {
        _startupState.startupSceneRequested = true;
        log_startup_event(std::string("Startup scene requested: ") + path.string());
        if (!_launchOptions.startupDisplayMode.has_value()) {
            ApplySceneModeInference(settings, path);
        }
        ApplyBenchmarkSceneLightingPreset(settings, path);
        if (!_launchOptions.startupExternalHdriPath.has_value()) {
            if (const std::optional<std::filesystem::path> hdri = BenchmarkSceneHdri(path)) {
                apply_external_hdri_path(*hdri);
            }
        }
        return _renderer.LoadSceneAsync(path);
    };

    bool loadedScene = false;
    std::filesystem::path acceptedScenePath;
    if (_launchOptions.startupScenePath.has_value()) {
        loadedScene = requestSceneLoad(*_launchOptions.startupScenePath);
        if (loadedScene) {
            acceptedScenePath = *_launchOptions.startupScenePath;
        }
    } else {
        const std::array<std::filesystem::path, 2> defaultScenes{
            std::filesystem::path("assets/basicmesh.glb"),
            std::filesystem::path("assets/structure.glb"),
        };
        for (const std::filesystem::path& candidate : defaultScenes) {
            if (requestSceneLoad(candidate)) {
                loadedScene = true;
                acceptedScenePath = candidate;
                break;
            }
        }
    }

    if (loadedScene) {
        remember_recent_scene(acceptedScenePath);
        log_startup_event("Startup scene load accepted");
    } else {
        if (_startupState.safeOverridesActive) {
            _renderer.GetSettings() = _startupState.savedSettings;
            _renderer.SetStartupSafeModeActive(false);
            _startupState.safeOverridesActive = false;
        }
        log_startup_event("Startup scene load was not accepted");
    }
}

void VestaEngine::cleanup()
{
    if (_isInitialized) {
        shutdown_imgui();
        _renderer.Shutdown();

        if (_window != nullptr) {
            SDL_DestroyWindow(_window);
            _window = nullptr;
        }

        SDL_Quit();
        _isInitialized = false;
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VestaEngine::draw(float deltaSeconds)
{
    // Build ImGui before rendering the frame so its draw data is ready when the
    // renderer records the overlay callback near the end of command recording.
    if (_startupState.safeOverridesActive) {
        const auto& settings = _renderer.GetSettings();
        VESTA_ASSERT(settings.sceneUploadMode == vesta::render::SceneUploadMode::Streaming,
            "Safe startup mode must keep scene uploads on the streaming path.");
    }
    _renderer.Update(deltaSeconds);
    if (!_startupState.firstFramePresented) {
        log_startup_event("First frame update complete");
    }
    begin_imgui_frame(deltaSeconds);
    if (!_startupState.firstFramePresented) {
        log_startup_event("Entering first RenderFrame");
    }
    _renderer.RenderFrame();
    if (_showGpuProfilerPanel) {
        const float gpuFrameMs = TotalGpuMs(_renderer.GetLastRenderGraphTimings());
        if (gpuFrameMs > 0.0f) {
            _gpuFrameTimeHistoryMs[_gpuFrameTimeHistoryHead] = gpuFrameMs;
            _gpuFrameTimeHistoryHead = (_gpuFrameTimeHistoryHead + 1) % _gpuFrameTimeHistoryMs.size();
            _gpuFrameTimeHistoryCount = std::min(_gpuFrameTimeHistoryCount + 1, _gpuFrameTimeHistoryMs.size());
        }
    } else {
        _gpuFrameTimeHistoryHead = 0;
        _gpuFrameTimeHistoryCount = 0;
        _gpuFrameTimeHistoryMs.fill(0.0f);
    }
    update_startup_state();
    update_runtime_warnings();
    _frameNumber++;
}

void VestaEngine::run()
{
    SDL_Event e;
    bool bQuit = false;
    auto previousTick = std::chrono::steady_clock::now();

    fmt::println("Entering main loop...");

    // main loop
    while (!bQuit) {
        const auto frameStart = std::chrono::steady_clock::now();

        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            if (_imguiInitialized) {
                ImGui::SetCurrentContext(_imguiContext);
                ImGui_ImplSDL2_ProcessEvent(&e);
            }

            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED && !_launchOptions.benchmark.has_value()) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }

            if (e.type == SDL_KEYDOWN && e.key.repeat == 0 && e.key.keysym.sym == SDLK_F1) {
                _showDebugUi = !_showDebugUi;
                continue;
            }
            if (e.type == SDL_KEYDOWN && e.key.repeat == 0 && e.key.keysym.sym == SDLK_F5) {
                const bool reloaded = _renderer.ReloadShaders();
                log_startup_event(reloaded ? "Shader hot reload complete" : "Shader hot reload failed: " + _renderer.GetLastShaderReloadMessage());
                continue;
            }
            if (e.type == SDL_KEYDOWN && e.key.repeat == 0 && e.key.keysym.sym == SDLK_F12) {
                const std::filesystem::path path = MakeTimestampedCapturePath("screenshot", ".png");
                log_startup_event(request_screenshot_with_metadata(path, "hotkey_screenshot") ? "Screenshot queued: " + path.string()
                                                                                               : "Screenshot failed");
                continue;
            }

            // ImGui gets first chance at the event. Only forward it to the
            // renderer when the UI is not actively capturing that input stream.
            if (should_forward_event_to_renderer(e)) {
                _renderer.HandleEvent(e);
            }
        }

        // do not draw if we are minimized
        if (stop_rendering && !_launchOptions.benchmark.has_value()) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            previousTick = std::chrono::steady_clock::now();
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - previousTick).count();
        previousTick = now;

        draw(deltaSeconds);
        update_benchmark(deltaSeconds);

        const auto& settings = _renderer.GetSettings();
        if (settings.enableFpsLimit && settings.fpsLimit > 0 && !_launchOptions.benchmark.has_value()) {
            const float targetSeconds = 1.0f / static_cast<float>(settings.fpsLimit);
            const auto frameEnd = std::chrono::steady_clock::now();
            const float elapsedSeconds = std::chrono::duration<float>(frameEnd - frameStart).count();
            if (elapsedSeconds < targetSeconds) {
                std::this_thread::sleep_for(std::chrono::duration<float>(targetSeconds - elapsedSeconds));
            }
        }
    }
}

void VestaEngine::log_startup_event(std::string_view message)
{
    const std::string line = MakeTimestampedLogLine(message);
    _logConsoleLines.push_back(line);
    if (_logConsoleLines.size() > 512) {
        _logConsoleLines.erase(_logConsoleLines.begin(), _logConsoleLines.begin() + static_cast<std::ptrdiff_t>(_logConsoleLines.size() - 512));
    }
    fmt::println("{}", line);
#if defined(_WIN32)
    OutputDebugStringA((line + "\n").c_str());
#endif

    const std::filesystem::path logPath = _launchOptions.startupLogPath;
    if (logPath.empty()) {
        return;
    }

    const std::filesystem::path parentPath = logPath.parent_path();
    if (!parentPath.empty()) {
        std::error_code errorCode;
        std::filesystem::create_directories(parentPath, errorCode);
    }

    std::ofstream output(logPath, std::ios::app);
    if (!output.is_open()) {
        return;
    }
    output << line << '\n';
}

void VestaEngine::update_runtime_warnings()
{
    if (_frameNumber < 30) {
        return;
    }

    const int warningCooldownFrames = 180;
    const auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();
    const auto& graphTimings = _renderer.GetLastRenderGraphTimings();
    const float cpuFrameMs = _renderer.GetSmoothedFrameTimeMs();
    const float gpuFrameMs = TotalGpuMs(graphTimings);

    for (const std::string& message : _renderer.GetRenderDevice().ConsumeValidationMessages()) {
        log_startup_event(message);
    }

    if (cpuFrameMs > 33.3f
        && RuntimeWarningCooldownElapsed(_frameNumber, _lastCpuFrameWarningFrame, warningCooldownFrames)) {
        log_startup_event(fmt::format("[PERF] CPU frame time high: {:.2f} ms", cpuFrameMs));
        _lastCpuFrameWarningFrame = _frameNumber;
    }

    if (gpuFrameMs > 20.0f
        && RuntimeWarningCooldownElapsed(_frameNumber, _lastGpuFrameWarningFrame, warningCooldownFrames)) {
        log_startup_event(fmt::format("[PERF] GPU frame time high: {:.2f} ms", gpuFrameMs));
        _lastGpuFrameWarningFrame = _frameNumber;
    }

    if (const auto* slowest = SlowestGpuPass(graphTimings);
        slowest != nullptr && slowest->gpuMs > 5.0f
        && RuntimeWarningCooldownElapsed(_frameNumber, _lastPassWarningFrame, warningCooldownFrames)) {
        log_startup_event(fmt::format("[PERF] Slow render pass: {} {:.2f} ms", slowest->name, slowest->gpuMs));
        _lastPassWarningFrame = _frameNumber;
    }

    if (settings.pathTraceBackend == vesta::render::PathTraceBackend::HardwareRT
        && settings.enablePathTracing
        && !scene.HasRayTracingScene()
        && RuntimeWarningCooldownElapsed(_frameNumber, _lastValidationWarningFrame, warningCooldownFrames)) {
        log_startup_event("[VALIDATION] Hardware RT path selected but TLAS is not resident");
        _lastValidationWarningFrame = _frameNumber;
    }

    if (!scene.GetTextures().empty()
        && scene.GetResidentTextureCount() < scene.GetTextures().size()
        && !_renderer.IsSceneLoadInProgress()
        && RuntimeWarningCooldownElapsed(_frameNumber, _lastResourceWarningFrame, warningCooldownFrames)) {
        log_startup_event(fmt::format("[RESOURCE] Texture residency incomplete: {}/{} resident",
            scene.GetResidentTextureCount(),
            scene.GetTextures().size()));
        _lastResourceWarningFrame = _frameNumber;
    }
}

void VestaEngine::update_startup_state()
{
    const auto& sceneLoadStatus = _renderer.GetSceneLoadStatus();
    const bool sceneBecameReady = _startupState.lastSceneLoadState != vesta::render::SceneLoadState::Ready
        && sceneLoadStatus.state == vesta::render::SceneLoadState::Ready;
    if (_startupState.lastSceneLoadState != sceneLoadStatus.state
        || _startupState.lastSceneLoadMessage != sceneLoadStatus.message) {
        std::string statusLine = "Scene state -> ";
        statusLine += SceneLoadStateLabel(sceneLoadStatus.state);
        if (!sceneLoadStatus.message.empty()) {
            statusLine += " | ";
            statusLine += sceneLoadStatus.message;
        }
        log_startup_event(statusLine);
        _startupState.lastSceneLoadState = sceneLoadStatus.state;
        _startupState.lastSceneLoadMessage = sceneLoadStatus.message;
        if (sceneLoadStatus.state == vesta::render::SceneLoadState::Ready
            || sceneLoadStatus.state == vesta::render::SceneLoadState::Failed) {
            _startupState.startupSceneResolved = true;
        }
    }
    if (_startupState.lastSceneLoadLogCount > sceneLoadStatus.logMessages.size()) {
        _startupState.lastSceneLoadLogCount = 0;
    }
    for (size_t logIndex = _startupState.lastSceneLoadLogCount; logIndex < sceneLoadStatus.logMessages.size(); ++logIndex) {
        log_startup_event("Scene load: " + sceneLoadStatus.logMessages[logIndex]);
    }
    _startupState.lastSceneLoadLogCount = sceneLoadStatus.logMessages.size();

    if (sceneBecameReady && !_renderer.GetScene().GetSourcePath().empty()) {
        ApplyBenchmarkSceneLightingPreset(_renderer.GetSettings(), _renderer.GetScene().GetSourcePath(), &_renderer.GetScene().GetBounds());
        _renderer.ResetAccumulation();
        log_startup_event("Scene bounds lighting preset applied");
    }

    if (!_startupState.firstFramePresented) {
        _startupState.firstFramePresented = true;
        log_startup_event("First frame presented");
    }

    if (_startupState.safeOverridesActive && _startupState.startupSceneResolved && _startupState.firstFramePresented) {
        _renderer.GetSettings() = _startupState.savedSettings;
        if (!_launchOptions.startupDisplayMode.has_value() && !_renderer.GetScene().GetSourcePath().empty()) {
            ApplySceneModeInference(_renderer.GetSettings(), _renderer.GetScene().GetSourcePath());
        }
        if (!_renderer.GetScene().GetSourcePath().empty()) {
            ApplyBenchmarkSceneLightingPreset(_renderer.GetSettings(), _renderer.GetScene().GetSourcePath(), &_renderer.GetScene().GetBounds());
        }
        _renderer.SetStartupSafeModeActive(false);
        if (_renderer.GetSettings().buildRayTracingStructuresOnLoad
            && _renderer.GetRenderDevice().IsRayTracingSupported()
            && !_renderer.GetScene().HasRayTracingScene()) {
            log_startup_event("Building deferred RT structures");
            if (_renderer.EnsureRayTracingScene()) {
                log_startup_event("Deferred RT structures ready");
            } else {
                log_startup_event("Deferred RT structures skipped");
            }
        }
        _renderer.ResetAccumulation();
        _startupState.safeOverridesActive = false;
        log_startup_event("Safe startup overrides restored");
    }

    if (_window != nullptr && _startupState.safeOverridesActive) {
        const std::string title = sceneLoadStatus.message.empty() ? "Vesta Engine - Loading..."
                                                                  : "Vesta Engine - " + sceneLoadStatus.message;
        SDL_SetWindowTitle(_window, title.c_str());
    } else if (_window != nullptr) {
        SDL_SetWindowTitle(_window, "Vesta Engine");
    }
}

void VestaEngine::update_benchmark(float deltaSeconds)
{
    if (!_launchOptions.benchmark.has_value() || _benchmarkState.completed) {
        return;
    }

    const auto& benchmark = *_launchOptions.benchmark;
    const auto& sceneLoadStatus = _renderer.GetSceneLoadStatus();
    if (sceneLoadStatus.state == vesta::render::SceneLoadState::Failed) {
        fmt::println(stderr, "Benchmark aborted: {}", sceneLoadStatus.message);
        _benchmarkState.completed = true;
        SDL_Event quitEvent{};
        quitEvent.type = SDL_QUIT;
        SDL_PushEvent(&quitEvent);
        return;
    }

    if (_renderer.IsSceneLoadInProgress()) {
        return;
    }

    if (!_benchmarkState.started) {
        _benchmarkState.started = true;
        _benchmarkState.lastGaussianRebuildCount = _renderer.GetOfficialGaussianRebuildCount();
        _benchmarkState.stableGaussianFrames = 0;
        fmt::println("Benchmark warmup for {:.1f}s", benchmark.warmupSeconds);
    }

    if (!_benchmarkState.capturing) {
        if (_renderer.GetScene().HasTrainedGaussians()) {
            if (_renderer.IsGaussianInteractivePreviewActive()) {
                _benchmarkState.stableGaussianFrames = 0;
                _benchmarkState.warmupElapsed = 0.0f;
                return;
            }
            const uint64_t rebuildCount = _renderer.GetOfficialGaussianRebuildCount();
            if (rebuildCount != _benchmarkState.lastGaussianRebuildCount) {
                _benchmarkState.lastGaussianRebuildCount = rebuildCount;
                _benchmarkState.stableGaussianFrames = 0;
                _benchmarkState.warmupElapsed = 0.0f;
            } else {
                ++_benchmarkState.stableGaussianFrames;
            }
        }
        _benchmarkState.warmupElapsed += deltaSeconds;
        if (_benchmarkState.warmupElapsed < benchmark.warmupSeconds) {
            return;
        }

        _benchmarkState.capturing = true;
        _benchmarkState.captureElapsed = 0.0f;
        _benchmarkState.frameTimesMs.clear();
        _benchmarkState.passGpuMsSums.fill(0.0f);
        _benchmarkState.passGpuSampleCounts.fill(0u);
        _benchmarkState.screenshotQueued = false;
        fmt::println(
            "Capturing benchmark for {:.1f}s -> {}", benchmark.captureSeconds, benchmark.csvOutputPath.string());
        return;
    }

    if (!_benchmarkState.screenshotQueued && !benchmark.screenshotOutputPath.empty()) {
        _benchmarkState.screenshotQueued = true;
        if (request_screenshot_with_metadata(benchmark.screenshotOutputPath, "benchmark")) {
            log_startup_event("Benchmark screenshot queued: " + benchmark.screenshotOutputPath.string());
        } else {
            log_startup_event("Benchmark screenshot failed");
        }
    }

    _benchmarkState.frameTimesMs.push_back(_renderer.GetFrameTimeMs());
    for (const auto& timing : _renderer.GetLastRenderGraphTimings()) {
        const std::optional<size_t> passIndex = BenchmarkPassIndex(timing.name);
        if (!passIndex.has_value() || !timing.gpuTimingValid) {
            continue;
        }
        _benchmarkState.passGpuMsSums[*passIndex] += timing.gpuMs;
        ++_benchmarkState.passGpuSampleCounts[*passIndex];
    }
    _benchmarkState.captureElapsed += deltaSeconds;
    if (_benchmarkState.captureElapsed < benchmark.captureSeconds) {
        return;
    }

    finish_benchmark();
    SDL_Event quitEvent{};
    quitEvent.type = SDL_QUIT;
    SDL_PushEvent(&quitEvent);
}

void VestaEngine::finish_benchmark()
{
    if (!_launchOptions.benchmark.has_value() || _benchmarkState.completed) {
        return;
    }

    _benchmarkState.completed = true;
    const auto& benchmark = *_launchOptions.benchmark;
    if (_benchmarkState.frameTimesMs.empty()) {
        fmt::println(stderr, "Benchmark finished without any captured frames.");
        return;
    }

    std::vector<float> sortedFrameTimes = _benchmarkState.frameTimesMs;
    std::sort(sortedFrameTimes.begin(), sortedFrameTimes.end());
    const float frameSum = std::accumulate(_benchmarkState.frameTimesMs.begin(), _benchmarkState.frameTimesMs.end(), 0.0f);
    const float averageFrameMs = frameSum / static_cast<float>(_benchmarkState.frameTimesMs.size());
    const float minFrameMs = sortedFrameTimes.front();
    const float maxFrameMs = sortedFrameTimes.back();
    const size_t p95Index =
        std::min(sortedFrameTimes.size() - 1, static_cast<size_t>(std::ceil(sortedFrameTimes.size() * 0.95f)) - 1);
    const float p95FrameMs = sortedFrameTimes[p95Index];
    const float averageFps = averageFrameMs > 0.0f ? 1000.0f / averageFrameMs : 0.0f;

    std::filesystem::path outputPath = benchmark.csvOutputPath;
    if (outputPath.is_relative()) {
        outputPath = std::filesystem::current_path() / outputPath;
    }

    const std::filesystem::path parentPath = outputPath.parent_path();
    if (!parentPath.empty()) {
        std::filesystem::create_directories(parentPath);
    }

    const bool writeHeader = !std::filesystem::exists(outputPath) || std::filesystem::file_size(outputPath) == 0;
    std::ofstream output(outputPath, std::ios::app);
    if (!output.is_open()) {
        fmt::println(stderr, "Failed to open benchmark output: {}", outputPath.string());
        return;
    }

    if (writeHeader) {
        output << "timestamp,scene,scene_kind,gpu,resolution,vsync,present_mode,fps_limit_enabled,fps_limit,display_mode,"
               << "debug_view,path_trace_debug_view,gaussian_debug_view,compare_mode,compare_split,compare_difference_scale,"
               << "ssao,ssao_radius,ssao_intensity,"
               << "taa,taa_feedback,temporal_upscaler,temporal_upscaler_backend,temporal_upscaler_input,temporal_upscaler_output,temporal_upscaler_scale,temporal_upscaler_sharpness,temporal_upscaler_history,temporal_upscaler_reactive_mask,temporal_upscaler_material_reactive_mask,temporal_upscaler_authored_alpha_reactive_mask,temporal_upscaler_reactive_strength,"
               << "restir_di,restir_gi,restir_pt,restir_backend,restir_reservoir_storage,restir_di_storage,restir_gi_storage,restir_pt_storage,restir_pt_path_state,restir_gi_backend,restir_pt_backend,restir_gi_candidate_pass,restir_pt_candidate_pass,restir_pt_path_state_reuse,restir_lights,restir_emissive_lights,restir_candidates,restir_reservoirs,restir_reservoir_mb,restir_di_reservoir_mb,restir_gi_reservoir_mb,restir_pt_reservoir_mb,restir_pt_path_state_mb,restir_temporal_reuse,restir_spatial_reuse,restir_history,restir_candidate_pass,restir_temporal_pass,restir_spatial_pass,restir_gi_resolve,restir_pt_resolve,restir_resolve,"
               << "rt_hybrid_backend,rt_hybrid_ray_query,rt_hybrid_tlas,rt_hybrid_resolution,rt_shadow_requested,rt_ao_requested,rt_reflection_requested,rt_gi_requested,rt_shadow_samples,rt_ao_samples,rt_reflection_samples,rt_gi_samples,rt_shadow_rays,rt_ao_rays,rt_reflection_rays,rt_gi_rays,rt_denoiser,rt_gi_spatial_denoise,rt_temporal,rt_gi_temporal,"
               << "ssr,ssr_max_distance,ssr_thickness,ssr_intensity,"
               << "ssgi,ssgi_radius,ssgi_intensity,ssgi_samples,"
               << "ddgi,ddgi_backend,ddgi_probes,ddgi_rays_per_update,ddgi_memory_mb,ddgi_probe_spacing,ddgi_hysteresis,ddgi_intensity,ddgi_overlay,ddgi_composite,ddgi_storage_composite,ddgi_moment_validation,ddgi_spatial_filtering,ddgi_ray_update,ddgi_temporal_blend,ddgi_probe_relocation,"
               << "voxel_gi,voxel_gi_backend,voxel_gi_resolution,voxel_gi_voxels,voxel_gi_extent,voxel_gi_voxel_size,voxel_gi_memory_mb,voxel_gi_radiance,voxel_gi_occupancy,voxel_gi_visualization,"
               << "shadow_map,shadow_map_size,shadow_cascades,shadow_cascade_lambda,shadow_bias,shadow_normal_bias,shadow_strength,shadow_pcss,shadow_filter_radius,contact_shadows,contact_shadow_length,contact_shadow_intensity,"
               << "pt_nee,pt_rr,pt_rr_depth,pt_firefly_clamp,"
               << "pt_denoiser,pt_denoiser_strength,pt_denoiser_temporal,pt_denoiser_iterations,"
               << "pt_primary_rays,pt_shadow_rays,pt_diffuse_rays,pt_specular_rays,pt_total_rays,"
               << "requested_backend,active_backend,scene_upload_mode,"
               << "gaussian,path_tracing,texture_streaming,indirect_draw,frustum_culling,distance_culling,async_compute,async_timeline,transfer_queue,"
               << "gpu_driven,gpu_driven_backend,visibility_set_valid,visible_surfaces,culled_surfaces,indirect_draw_estimate,"
               << "meshlet_culling,meshlet_backend,meshlet_visibility_storage,meshlet_visibility_mb,cluster_count,visible_clusters,culled_clusters,meshlet_count,visible_meshlets,culled_meshlets,cluster_bounds,meshlet_triangles,"
               << "gaussian_trained,gaussian_count,gaussian_sh_degree,gaussian_view_dependent_color,gaussian_antialiasing,"
               << "gaussian_fast_culling,gaussian_opacity,gaussian_mix,gaussian_interactive_preview,"
               << "pt_scale,environment_intensity,environment_rotation_deg,environment_preset,external_hdri,external_hdri_path,external_hdri_resolution,external_hdri_hdr,"
               << "ibl_source,ibl_backend,ibl_diffuse_cubemap,ibl_specular_cubemap,ibl_brdf_lut,ibl_estimated_mb,ibl_diffuse,ibl_specular,exposure_ev,"
               << "bloom,bloom_threshold,bloom_intensity,fxaa,motion_blur,motion_blur_strength,vignette,vignette_strength,saturation,contrast,"
               << "aperture_radius,focal_distance,"
               << "avg_frame_ms,p95_frame_ms,min_frame_ms,max_frame_ms,avg_fps,frame_count,"
               << "vertices,triangles,surfaces,textures_total,textures_resident,"
               << "bindless_sampled_images,bindless_sampled_image_capacity,bindless_sampled_cube_images,bindless_sampled_cube_image_capacity,bindless_storage_images,bindless_storage_image_capacity,"
               << "bindless_storage_buffers,bindless_storage_buffer_capacity,parse_ms,prepare_ms,"
               << "geometry_upload_ms,texture_upload_ms,blas_ms,tlas_ms,"
               << "gaussian_projected,gaussian_duplicates,gaussian_padded_duplicates,gaussian_tiles,gaussian_avg_tiles_touched,gaussian_rebuilds,"
               << "gaussian_preprocess_ms,gaussian_scan_ms,gaussian_duplicate_ms,gaussian_sort_ms,gaussian_range_ms,"
               << "gaussian_raster_ms,gaussian_total_build_ms,"
               << "geometry_pass_gpu_ms,shadow_pass_gpu_ms,overdraw_pass_gpu_ms,ray_effects_pass_gpu_ms,restir_candidate_pass_gpu_ms,restir_resolve_pass_gpu_ms,ddgi_probe_update_pass_gpu_ms,deferred_pass_gpu_ms,legacy_gaussian_pass_gpu_ms,official_gaussian_pass_gpu_ms,"
               << "path_trace_pass_gpu_ms,path_denoise_pass_gpu_ms,temporal_aa_pass_gpu_ms,bloom_extract_pass_gpu_ms,bloom_downsample_pass_gpu_ms,bloom_upsample_pass_gpu_ms,composite_pass_gpu_ms\n";
    }

    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &nowTime);
#else
    localtime_r(&nowTime, &localTime);
#endif
    std::ostringstream timestampStream;
    timestampStream << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");

    const auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();
    const auto& status = _renderer.GetSceneLoadStatus();
    const auto& device = _renderer.GetRenderDevice();
    const auto extent = device.GetSwapchainExtent();
    const auto bindlessStats = device.GetBindlessStats();
    const auto gpuDrivenStats = _renderer.GetGpuDrivenStats();
    const auto meshletStats = _renderer.GetMeshletClusterStats();
    const auto temporalUpscalerStats = _renderer.GetTemporalUpscalerStats();
    const auto restirStats = _renderer.GetRestirStats();
    const bool anyRestirResolve =
        restirStats.lightingResolveAvailable || restirStats.giResolveAvailable || restirStats.ptResolveAvailable;
    const auto ddgiStats = _renderer.GetDdgiStats();
    const auto voxelGiStats = _renderer.GetVoxelGiStats();
    const auto iblStats = _renderer.GetIblStats();
    const auto rayEffectsStats = _renderer.GetRayEffectsStats();
    std::string iblBackendLabel;
    if (iblStats.environmentMapUploaded) {
        iblBackendLabel = "EquirectSampling";
        if (iblStats.environmentCubemapAvailable) {
            iblBackendLabel += "+CubeImage";
        }
        if (iblStats.diffuseIrradianceAvailable) {
            iblBackendLabel += "+DiffuseIrradiance";
        }
        if (iblStats.specularPrefilterAvailable) {
            iblBackendLabel += "+SpecularPrefilterCube";
        }
        if (iblStats.brdfLutAvailable) {
            iblBackendLabel += "+BRDFLUT";
        }
    } else {
        iblBackendLabel = iblStats.brdfLutAvailable ? "Procedural+BRDFLUT" : "Staged";
    }
    const auto averagePassGpuMs = [&](size_t passIndex) {
        const uint32_t sampleCount = _benchmarkState.passGpuSampleCounts[passIndex];
        return sampleCount > 0u ? _benchmarkState.passGpuMsSums[passIndex] / static_cast<float>(sampleCount) : 0.0f;
    };
    vesta::render::RenderPassDebugInfo pathTracePassInfo{};
    for (const auto& pass : _renderer.GetRenderPassDebugInfo()) {
        if (pass.id == "path-tracer") {
            pathTracePassInfo = pass;
            break;
        }
    }

    output << CsvEscape(timestampStream.str()) << ','
           << CsvEscape(scene.GetSourcePath().string()) << ','
           << CsvEscape(SceneKindLabel(scene.GetSceneKind())) << ','
           << CsvEscape(device.GetGpuName()) << ','
           << CsvEscape(fmt::format("{}x{}", extent.width, extent.height)) << ','
           << (settings.enableVSync ? "true" : "false") << ','
           << PresentModeLabel(_renderer.GetRenderDevice().GetPresentMode()) << ','
           << (settings.enableFpsLimit ? "true" : "false") << ','
           << settings.fpsLimit << ','
           << DisplayModeLabel(settings.displayMode) << ','
           << CsvEscape(RendererDebugViewLabel(settings.debugView)) << ','
           << CsvEscape(PathTraceDebugViewLabel(settings.pathTraceDebugView)) << ','
           << CsvEscape(GaussianDebugViewLabel(settings.gaussianDebugView)) << ','
           << CsvEscape(CompareModeLabel(settings.compareMode)) << ','
           << settings.compareSplitPosition << ','
           << settings.compareDifferenceScale << ','
           << (settings.enableSsao ? "true" : "false") << ','
           << settings.ssaoRadius << ','
           << settings.ssaoIntensity << ','
           << (settings.enableTaa ? "true" : "false") << ','
           << settings.taaFeedback << ','
           << (settings.enableTemporalUpscaler ? "true" : "false") << ','
           << (temporalUpscalerStats.backendAvailable ? "TAAU" : "Staged") << ','
           << CsvEscape(fmt::format("{}x{}", temporalUpscalerStats.inputWidth, temporalUpscalerStats.inputHeight)) << ','
           << CsvEscape(fmt::format("{}x{}", temporalUpscalerStats.outputWidth, temporalUpscalerStats.outputHeight)) << ','
           << temporalUpscalerStats.scale << ','
           << temporalUpscalerStats.sharpness << ','
           << (temporalUpscalerStats.taaHistoryAvailable ? "true" : "false") << ','
           << (temporalUpscalerStats.reactiveMaskAvailable ? "true" : "false") << ','
           << (temporalUpscalerStats.materialReactiveMaskAvailable ? "true" : "false") << ','
           << (temporalUpscalerStats.authoredAlphaReactiveMaskAvailable ? "true" : "false") << ','
           << temporalUpscalerStats.reactiveMaskStrength << ','
           << (restirStats.requestedDi ? "true" : "false") << ','
           << (restirStats.requestedGi ? "true" : "false") << ','
           << (restirStats.requestedPt ? "true" : "false") << ','
           << (anyRestirResolve ? "CandidateReservoir+ShadingResolve"
                                : (restirStats.backendAvailable ? "ReservoirBackend" : "Staged"))
           << ','
           << (restirStats.reservoirBuffersAvailable ? "true" : "false") << ','
           << (restirStats.diReservoirBuffersAvailable ? "true" : "false") << ','
           << (restirStats.giReservoirBuffersAvailable ? "true" : "false") << ','
           << (restirStats.ptReservoirBuffersAvailable ? "true" : "false") << ','
           << (restirStats.ptPathStateAvailable ? "true" : "false") << ','
           << (restirStats.giReservoirBackendAvailable ? "true" : "false") << ','
           << (restirStats.ptReservoirBackendAvailable ? "true" : "false") << ','
           << (restirStats.giCandidatePassAvailable ? "true" : "false") << ','
           << (restirStats.ptCandidatePassAvailable ? "true" : "false") << ','
           << (restirStats.ptPathStateReuseAvailable ? "true" : "false") << ','
           << restirStats.activeLightCount << ','
           << restirStats.emissiveTriangleCount << ','
           << restirStats.candidateLightCount << ','
           << restirStats.reservoirCount << ','
           << MiB(restirStats.estimatedReservoirBytes) << ','
           << MiB(restirStats.estimatedDiReservoirBytes) << ','
           << MiB(restirStats.estimatedGiReservoirBytes) << ','
           << MiB(restirStats.estimatedPtReservoirBytes) << ','
           << MiB(restirStats.estimatedPtPathStateBytes) << ','
           << (restirStats.temporalReuse ? "true" : "false") << ','
           << (restirStats.spatialReuse ? "true" : "false") << ','
           << (restirStats.historyAvailable ? "true" : "false") << ','
           << (restirStats.candidateSamplingAvailable ? "true" : "false") << ','
           << (restirStats.temporalReusePassAvailable ? "true" : "false") << ','
           << (restirStats.spatialReusePassAvailable ? "true" : "false") << ','
           << (restirStats.giResolveAvailable ? "true" : "false") << ','
           << (restirStats.ptResolveAvailable ? "true" : "false") << ','
           << (anyRestirResolve ? "true" : "false") << ','
           << (rayEffectsStats.backendAvailable ? "RayQueryPass" : "Staged") << ','
           << (rayEffectsStats.rayQueryAvailable ? "true" : "false") << ','
           << (rayEffectsStats.tlasAvailable ? "true" : "false") << ','
           << CsvEscape(fmt::format("{}x{}", rayEffectsStats.inputWidth, rayEffectsStats.inputHeight)) << ','
           << (rayEffectsStats.shadowsRequested ? "true" : "false") << ','
           << (rayEffectsStats.aoRequested ? "true" : "false") << ','
           << (rayEffectsStats.reflectionsRequested ? "true" : "false") << ','
           << (rayEffectsStats.giRequested ? "true" : "false") << ','
           << rayEffectsStats.shadowSamples << ','
           << rayEffectsStats.aoSamples << ','
           << rayEffectsStats.reflectionSamples << ','
           << rayEffectsStats.giSamples << ','
           << rayEffectsStats.estimatedShadowRays << ','
           << rayEffectsStats.estimatedAoRays << ','
           << rayEffectsStats.estimatedReflectionRays << ','
           << rayEffectsStats.estimatedGiRays << ','
           << (rayEffectsStats.denoiserRequested ? "true" : "false") << ','
           << (rayEffectsStats.giSpatialDenoiseAvailable ? "true" : "false") << ','
           << (rayEffectsStats.temporalAccumulation ? "true" : "false") << ','
           << (rayEffectsStats.giTemporalAccumulationAvailable ? "true" : "false") << ','
           << (settings.enableSsr ? "true" : "false") << ','
           << settings.ssrMaxDistance << ','
           << settings.ssrThickness << ','
           << settings.ssrIntensity << ','
           << (settings.enableSsgi ? "true" : "false") << ','
           << settings.ssgiRadius << ','
           << settings.ssgiIntensity << ','
           << settings.ssgiSampleCount << ','
           << (ddgiStats.requested ? "true" : "false") << ','
           << CsvEscape(ddgiStats.storageCompositeAvailable && ddgiStats.rayUpdateAvailable ? "StorageComposite+RayUpdate"
                   : ddgiStats.storageCompositeAvailable                             ? "StorageComposite"
                   : ddgiStats.probeCompositeAvailable                               ? "ProbeComposite"
                   : ddgiStats.rayUpdateAvailable                                    ? "RayUpdate"
                                                                                     : "Staged")
           << ','
           << CsvEscape(fmt::format("{}x{}x{}", ddgiStats.probeCountX, ddgiStats.probeCountY, ddgiStats.probeCountZ)) << ','
           << ddgiStats.raysPerUpdate << ','
           << MiB(ddgiStats.estimatedIrradianceBytes + ddgiStats.estimatedVisibilityBytes + ddgiStats.estimatedRelocationBytes) << ','
           << ddgiStats.probeSpacing << ','
           << ddgiStats.hysteresis << ','
           << ddgiStats.intensity << ','
           << (ddgiStats.overlayEnabled ? "true" : "false") << ','
           << (ddgiStats.probeCompositeAvailable ? "true" : "false") << ','
           << (ddgiStats.storageCompositeAvailable ? "true" : "false") << ','
           << (ddgiStats.momentValidationAvailable ? "true" : "false") << ','
           << (ddgiStats.spatialFilteringAvailable ? "true" : "false") << ','
           << (ddgiStats.rayUpdateAvailable ? "true" : "false") << ','
           << (ddgiStats.temporalBlendAvailable ? "true" : "false") << ','
           << (ddgiStats.probeRelocationAvailable ? "true" : "false") << ','
           << (voxelGiStats.requested ? "true" : "false") << ','
           << CsvEscape(voxelGiStats.storageAvailable ? "VolumeStorage" : "Staged") << ','
           << voxelGiStats.resolution << ','
           << voxelGiStats.totalVoxels << ','
           << voxelGiStats.worldExtent << ','
           << voxelGiStats.voxelSize << ','
           << MiB(voxelGiStats.estimatedRadianceBytes + voxelGiStats.estimatedOccupancyBytes) << ','
           << (voxelGiStats.radianceAvailable ? "true" : "false") << ','
           << (voxelGiStats.occupancyAvailable ? "true" : "false") << ','
           << (voxelGiStats.visualizationAvailable ? "true" : "false") << ','
           << (settings.enableShadowMap ? "true" : "false") << ','
           << settings.shadowMapSize << ','
           << settings.shadowCascadeCount << ','
           << settings.shadowCascadeLambda << ','
           << settings.shadowBias << ','
           << settings.shadowNormalBias << ','
           << settings.shadowStrength << ','
           << (settings.enablePcssShadows ? "true" : "false") << ','
           << settings.shadowFilterRadius << ','
           << (settings.enableContactShadows ? "true" : "false") << ','
           << settings.contactShadowLength << ','
           << settings.contactShadowIntensity << ','
           << (settings.pathTraceNextEventEstimation ? "true" : "false") << ','
           << (settings.pathTraceRussianRoulette ? "true" : "false") << ','
           << settings.pathTraceRussianRouletteDepth << ','
           << settings.pathTraceFireflyClamp << ','
           << (settings.enablePathTraceDenoiser ? "true" : "false") << ','
           << settings.pathTraceDenoiserStrength << ','
           << settings.pathTraceDenoiserTemporalBlend << ','
           << settings.pathTraceDenoiserIterations << ','
           << pathTracePassInfo.primaryRayCount << ','
           << pathTracePassInfo.shadowRayCount << ','
           << pathTracePassInfo.diffuseRayCount << ','
           << pathTracePassInfo.specularRayCount << ','
           << pathTracePassInfo.rayCount << ','
           << PathTraceBackendLabel(settings.pathTraceBackend) << ','
           << PathTraceBackendLabel(_renderer.GetActivePathTraceBackend()) << ','
           << CsvEscape(SceneUploadModeLabel(settings.sceneUploadMode)) << ','
           << (settings.enableGaussian ? "true" : "false") << ','
           << (settings.enablePathTracing ? "true" : "false") << ','
           << (settings.textureStreamingEnabled ? "true" : "false") << ','
           << (settings.useIndirectDraw ? "true" : "false") << ','
           << (settings.enableFrustumCulling ? "true" : "false") << ','
           << (settings.enableDistanceCulling ? "true" : "false") << ','
           << (settings.enableAsyncCompute ? "true" : "false") << ','
           << (settings.showAsyncComputeTimeline ? "true" : "false") << ','
           << (device.HasTransferQueue() ? "true" : "false") << ','
           << (settings.enableGpuDrivenRendering ? "true" : "false") << ','
           << (gpuDrivenStats.gpuDrivenBackend ? "true" : "false") << ','
           << (gpuDrivenStats.visibilitySetValid ? "true" : "false") << ','
           << gpuDrivenStats.visibleSurfaces << ','
           << gpuDrivenStats.culledSurfaces << ','
           << gpuDrivenStats.indirectDrawEstimate << ','
           << (settings.enableMeshletCulling ? "true" : "false") << ','
           << CsvEscape(meshletStats.visibilityStorageAvailable ? "VisibilityStorage" : "Staged") << ','
           << (meshletStats.visibilityStorageAvailable ? "true" : "false") << ','
           << MiB(meshletStats.estimatedVisibilityBytes) << ','
           << meshletStats.totalClusters << ','
           << meshletStats.visibleClusters << ','
           << meshletStats.culledClusters << ','
           << meshletStats.totalMeshlets << ','
           << meshletStats.visibleMeshlets << ','
           << meshletStats.culledMeshlets << ','
           << meshletStats.boundsAvailable << ','
           << meshletStats.trianglesPerMeshlet << ','
           << (scene.HasTrainedGaussians() ? "true" : "false") << ','
           << scene.GetGaussianCount() << ','
           << scene.GetGaussianShDegree() << ','
           << (settings.gaussianViewDependentColor ? "true" : "false") << ','
           << (settings.gaussianAntialiasing ? "true" : "false") << ','
           << (settings.gaussianFastCulling ? "true" : "false") << ','
           << settings.gaussianOpacity << ','
           << settings.gaussianMix << ','
           << (_renderer.IsGaussianInteractivePreviewActive() ? "true" : "false") << ','
           << settings.pathTraceResolutionScale << ','
           << settings.environmentIntensity << ','
           << settings.environmentRotationDegrees << ','
           << CsvEscape(EnvironmentPresetLabel(settings.environmentPreset)) << ','
           << (settings.externalHdriAvailable ? "true" : "false") << ','
           << CsvEscape(settings.externalHdriPath.string()) << ','
           << CsvEscape(settings.externalHdriAvailable ? fmt::format("{}x{}", settings.externalHdriWidth, settings.externalHdriHeight) : "") << ','
           << (settings.externalHdriIsHdr ? "true" : "false") << ','
           << (iblStats.externalSourceAvailable ? "External" : "Procedural") << ','
           << CsvEscape(iblBackendLabel) << ','
           << CsvEscape(fmt::format("{}^2", iblStats.diffuseCubemapResolution)) << ','
           << CsvEscape(fmt::format("{}^2/{} mips", iblStats.specularCubemapResolution, iblStats.specularMipCount)) << ','
           << CsvEscape(fmt::format("{}^2", iblStats.brdfLutResolution)) << ','
           << MiB(iblStats.estimatedEnvironmentCubemapBytes + iblStats.estimatedDiffuseBytes + iblStats.estimatedSpecularBytes + iblStats.estimatedBrdfLutBytes) << ','
           << settings.environmentDiffuseStrength << ','
           << settings.environmentSpecularStrength << ','
           << settings.cameraExposureEv << ','
           << (settings.enableBloom ? "true" : "false") << ','
           << settings.bloomThreshold << ','
           << settings.bloomIntensity << ','
           << (settings.enableFxaa ? "true" : "false") << ','
           << (settings.enableMotionBlur ? "true" : "false") << ','
           << settings.motionBlurStrength << ','
           << (settings.enableVignette ? "true" : "false") << ','
           << settings.vignetteStrength << ','
           << settings.colorGradingSaturation << ','
           << settings.colorGradingContrast << ','
           << settings.cameraApertureRadius << ','
           << settings.cameraFocalDistance << ','
           << averageFrameMs << ','
           << p95FrameMs << ','
           << minFrameMs << ','
           << maxFrameMs << ','
           << averageFps << ','
           << _benchmarkState.frameTimesMs.size() << ','
           << scene.GetVertices().size() << ','
           << scene.GetTriangles().size() << ','
           << scene.GetSurfaces().size() << ','
           << scene.GetTextures().size() << ','
           << _renderer.GetResidentTextureCount() << ','
           << bindlessStats.sampledImagesUsed << ','
           << bindlessStats.sampledImagesCapacity << ','
           << bindlessStats.sampledCubeImagesUsed << ','
           << bindlessStats.sampledCubeImagesCapacity << ','
           << bindlessStats.storageImagesUsed << ','
           << bindlessStats.storageImagesCapacity << ','
           << bindlessStats.storageBuffersUsed << ','
           << bindlessStats.storageBuffersCapacity << ','
           << status.parseMs << ','
           << status.prepareMs << ','
           << status.geometryUploadMs << ','
           << status.textureUploadMs << ','
           << status.blasMs << ','
           << status.tlasMs << ','
           << _renderer.GetOfficialGaussianProjectedCount() << ','
           << _renderer.GetOfficialGaussianDuplicateCount() << ','
           << _renderer.GetOfficialGaussianPaddedDuplicateCount() << ','
           << _renderer.GetOfficialGaussianTileCount() << ','
           << _renderer.GetOfficialGaussianAverageTilesTouched() << ','
           << _renderer.GetOfficialGaussianRebuildCount() << ','
           << _renderer.GetOfficialGaussianPreprocessMs() << ','
           << _renderer.GetOfficialGaussianScanMs() << ','
           << _renderer.GetOfficialGaussianDuplicateMs() << ','
           << _renderer.GetOfficialGaussianSortMs() << ','
           << _renderer.GetOfficialGaussianRangeMs() << ','
           << _renderer.GetOfficialGaussianRasterMs() << ','
           << _renderer.GetOfficialGaussianTotalBuildMs() << ','
           << averagePassGpuMs(0) << ','
           << averagePassGpuMs(1) << ','
           << averagePassGpuMs(2) << ','
           << averagePassGpuMs(3) << ','
           << averagePassGpuMs(4) << ','
           << averagePassGpuMs(5) << ','
           << averagePassGpuMs(6) << ','
           << averagePassGpuMs(7) << ','
           << averagePassGpuMs(8) << ','
           << averagePassGpuMs(9) << ','
           << averagePassGpuMs(10) << ','
           << averagePassGpuMs(11) << ','
           << averagePassGpuMs(12) << ','
           << averagePassGpuMs(13) << ','
           << averagePassGpuMs(14) << ','
           << averagePassGpuMs(15) << ','
           << averagePassGpuMs(16) << '\n';

    fmt::println("Benchmark written to {}", outputPath.string());
}

bool VestaEngine::request_screenshot_with_metadata(const std::filesystem::path& path, std::string_view captureKind)
{
    if (!_renderer.RequestScreenshot(path)) {
        return false;
    }

    std::filesystem::path metadataPath = path;
    metadataPath.replace_extension(".json");
    if (metadataPath.is_relative()) {
        metadataPath = std::filesystem::current_path() / metadataPath;
    }
    if (!metadataPath.parent_path().empty()) {
        std::filesystem::create_directories(metadataPath.parent_path());
    }

    const auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();
    const auto extent = _renderer.GetRenderDevice().GetSwapchainExtent();
    const auto& camera = _renderer.GetCamera();
    const auto& graphTimings = _renderer.GetLastRenderGraphTimings();
    const std::vector<vesta::render::RenderPassDebugInfo> passDebugInfo = _renderer.GetRenderPassDebugInfo();
    const auto& device = _renderer.GetRenderDevice();
    const auto bindlessStats = device.GetBindlessStats();
    const auto gpuDrivenStats = _renderer.GetGpuDrivenStats();
    const auto meshletStats = _renderer.GetMeshletClusterStats();
    const auto temporalUpscalerStats = _renderer.GetTemporalUpscalerStats();
    const auto restirStats = _renderer.GetRestirStats();
    const auto ddgiStats = _renderer.GetDdgiStats();
    const auto voxelGiStats = _renderer.GetVoxelGiStats();
    const auto iblStats = _renderer.GetIblStats();
    const auto rayEffectsStats = _renderer.GetRayEffectsStats();
    const uint64_t sceneBufferBytes = BufferSizeBytes(device, scene.GetVertexBuffer())
        + BufferSizeBytes(device, scene.GetIndexBuffer())
        + BufferSizeBytes(device, scene.GetMaterialBuffer())
        + BufferSizeBytes(device, scene.GetTriangleBuffer())
        + BufferSizeBytes(device, scene.GetEmissiveTriangleBuffer())
        + BufferSizeBytes(device, scene.GetGaussianBuffer());
    const uint64_t accelerationBytes = BufferSizeBytes(device, scene.GetBottomLevelBuffer()) + BufferSizeBytes(device, scene.GetTopLevelBuffer());
    const float totalGpuMs = TotalGpuMs(graphTimings);

    std::ofstream output(metadataPath, std::ios::trunc);
    if (!output.is_open()) {
        log_startup_event("Capture metadata failed: " + metadataPath.string());
        return true;
    }

    output << "{\n"
           << "  \"capture_kind\": \"" << JsonEscape(captureKind) << "\",\n"
           << "  \"image\": \"" << JsonEscape(path.string()) << "\",\n"
           << "  \"scene\": \"" << JsonEscape(scene.GetSourcePath().string()) << "\",\n"
           << "  \"scene_kind\": \"" << JsonEscape(SceneKindLabel(scene.GetSceneKind())) << "\",\n"
           << "  \"gpu\": \"" << JsonEscape(_renderer.GetRenderDevice().GetGpuName()) << "\",\n"
           << "  \"api\": \"Vulkan\",\n"
           << "  \"resolution\": \"" << extent.width << "x" << extent.height << "\",\n"
           << "  \"vsync\": " << (settings.enableVSync ? "true" : "false") << ",\n"
           << "  \"present_mode\": \"" << PresentModeLabel(_renderer.GetRenderDevice().GetPresentMode()) << "\",\n"
           << "  \"fps_limit_enabled\": " << (settings.enableFpsLimit ? "true" : "false") << ",\n"
           << "  \"fps_limit\": " << settings.fpsLimit << ",\n"
           << "  \"display_mode\": \"" << DisplayModeLabel(settings.displayMode) << "\",\n"
           << "  \"debug_view\": \"" << RendererDebugViewLabel(settings.debugView) << "\",\n"
           << "  \"path_trace_debug_view\": \"" << PathTraceDebugViewLabel(settings.pathTraceDebugView) << "\",\n"
           << "  \"gaussian_debug_view\": \"" << GaussianDebugViewLabel(settings.gaussianDebugView) << "\",\n"
           << "  \"compare_mode\": \"" << CompareModeLabel(settings.compareMode) << "\",\n"
           << "  \"compare_split\": " << settings.compareSplitPosition << ",\n"
           << "  \"compare_difference_scale\": " << settings.compareDifferenceScale << ",\n"
           << "  \"ssao\": " << (settings.enableSsao ? "true" : "false") << ",\n"
           << "  \"ssao_radius\": " << settings.ssaoRadius << ",\n"
           << "  \"ssao_intensity\": " << settings.ssaoIntensity << ",\n"
           << "  \"taa\": " << (settings.enableTaa ? "true" : "false") << ",\n"
           << "  \"taa_feedback\": " << settings.taaFeedback << ",\n"
           << "  \"temporal_upscaler\": " << (settings.enableTemporalUpscaler ? "true" : "false") << ",\n"
           << "  \"temporal_upscaler_stats\": {\n"
           << "    \"backend_available\": " << (temporalUpscalerStats.backendAvailable ? "true" : "false") << ",\n"
           << "    \"input_width\": " << temporalUpscalerStats.inputWidth << ",\n"
           << "    \"input_height\": " << temporalUpscalerStats.inputHeight << ",\n"
           << "    \"output_width\": " << temporalUpscalerStats.outputWidth << ",\n"
           << "    \"output_height\": " << temporalUpscalerStats.outputHeight << ",\n"
           << "    \"scale\": " << temporalUpscalerStats.scale << ",\n"
           << "    \"sharpness\": " << temporalUpscalerStats.sharpness << ",\n"
           << "    \"taa_history_available\": " << (temporalUpscalerStats.taaHistoryAvailable ? "true" : "false") << ",\n"
           << "    \"motion_vectors_available\": " << (temporalUpscalerStats.motionVectorsAvailable ? "true" : "false") << ",\n"
           << "    \"depth_available\": " << (temporalUpscalerStats.depthAvailable ? "true" : "false") << ",\n"
           << "    \"reactive_mask_available\": " << (temporalUpscalerStats.reactiveMaskAvailable ? "true" : "false") << ",\n"
           << "    \"material_reactive_mask_available\": " << (temporalUpscalerStats.materialReactiveMaskAvailable ? "true" : "false") << ",\n"
           << "    \"authored_alpha_reactive_mask_available\": " << (temporalUpscalerStats.authoredAlphaReactiveMaskAvailable ? "true" : "false") << ",\n"
           << "    \"reactive_mask_strength\": " << temporalUpscalerStats.reactiveMaskStrength << "\n"
           << "  },\n"
           << "  \"restir_stats\": {\n"
           << "    \"di_requested\": " << (restirStats.requestedDi ? "true" : "false") << ",\n"
           << "    \"gi_requested\": " << (restirStats.requestedGi ? "true" : "false") << ",\n"
           << "    \"pt_requested\": " << (restirStats.requestedPt ? "true" : "false") << ",\n"
           << "    \"backend_available\": " << (restirStats.backendAvailable ? "true" : "false") << ",\n"
           << "    \"reservoir_buffers_available\": " << (restirStats.reservoirBuffersAvailable ? "true" : "false") << ",\n"
           << "    \"di_reservoir_buffers_available\": " << (restirStats.diReservoirBuffersAvailable ? "true" : "false") << ",\n"
           << "    \"gi_reservoir_buffers_available\": " << (restirStats.giReservoirBuffersAvailable ? "true" : "false") << ",\n"
           << "    \"pt_reservoir_buffers_available\": " << (restirStats.ptReservoirBuffersAvailable ? "true" : "false") << ",\n"
           << "    \"pt_path_state_available\": " << (restirStats.ptPathStateAvailable ? "true" : "false") << ",\n"
           << "    \"gi_reservoir_backend_available\": " << (restirStats.giReservoirBackendAvailable ? "true" : "false") << ",\n"
           << "    \"pt_reservoir_backend_available\": " << (restirStats.ptReservoirBackendAvailable ? "true" : "false") << ",\n"
           << "    \"gi_candidate_pass_available\": " << (restirStats.giCandidatePassAvailable ? "true" : "false") << ",\n"
           << "    \"pt_candidate_pass_available\": " << (restirStats.ptCandidatePassAvailable ? "true" : "false") << ",\n"
           << "    \"pt_path_state_reuse_available\": " << (restirStats.ptPathStateReuseAvailable ? "true" : "false") << ",\n"
           << "    \"active_light_count\": " << restirStats.activeLightCount << ",\n"
           << "    \"emissive_triangle_count\": " << restirStats.emissiveTriangleCount << ",\n"
           << "    \"candidate_light_count\": " << restirStats.candidateLightCount << ",\n"
           << "    \"reservoir_count\": " << restirStats.reservoirCount << ",\n"
           << "    \"reservoir_pixels\": " << restirStats.reservoirPixels << ",\n"
           << "    \"estimated_reservoir_bytes\": " << restirStats.estimatedReservoirBytes << ",\n"
           << "    \"estimated_di_reservoir_bytes\": " << restirStats.estimatedDiReservoirBytes << ",\n"
           << "    \"estimated_gi_reservoir_bytes\": " << restirStats.estimatedGiReservoirBytes << ",\n"
           << "    \"estimated_pt_reservoir_bytes\": " << restirStats.estimatedPtReservoirBytes << ",\n"
           << "    \"estimated_pt_path_state_bytes\": " << restirStats.estimatedPtPathStateBytes << ",\n"
           << "    \"temporal_reuse\": " << (restirStats.temporalReuse ? "true" : "false") << ",\n"
           << "    \"spatial_reuse\": " << (restirStats.spatialReuse ? "true" : "false") << ",\n"
           << "    \"history_available\": " << (restirStats.historyAvailable ? "true" : "false") << ",\n"
           << "    \"candidate_sampling_available\": " << (restirStats.candidateSamplingAvailable ? "true" : "false") << ",\n"
           << "    \"temporal_reuse_pass_available\": " << (restirStats.temporalReusePassAvailable ? "true" : "false") << ",\n"
           << "    \"spatial_reuse_pass_available\": " << (restirStats.spatialReusePassAvailable ? "true" : "false") << ",\n"
           << "    \"lighting_resolve_available\": " << (restirStats.lightingResolveAvailable ? "true" : "false") << ",\n"
           << "    \"gi_resolve_available\": " << (restirStats.giResolveAvailable ? "true" : "false") << ",\n"
           << "    \"pt_resolve_available\": " << (restirStats.ptResolveAvailable ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"ssr\": " << (settings.enableSsr ? "true" : "false") << ",\n"
           << "  \"ssr_max_distance\": " << settings.ssrMaxDistance << ",\n"
           << "  \"ssr_thickness\": " << settings.ssrThickness << ",\n"
           << "  \"ssr_intensity\": " << settings.ssrIntensity << ",\n"
           << "  \"ssgi\": " << (settings.enableSsgi ? "true" : "false") << ",\n"
           << "  \"ssgi_radius\": " << settings.ssgiRadius << ",\n"
           << "  \"ssgi_intensity\": " << settings.ssgiIntensity << ",\n"
           << "  \"ssgi_samples\": " << settings.ssgiSampleCount << ",\n"
           << "  \"ddgi_stats\": {\n"
           << "    \"requested\": " << (ddgiStats.requested ? "true" : "false") << ",\n"
           << "    \"backend_available\": " << (ddgiStats.backendAvailable ? "true" : "false") << ",\n"
           << "    \"probe_storage_available\": " << (ddgiStats.probeStorageAvailable ? "true" : "false") << ",\n"
           << "    \"probe_count_x\": " << ddgiStats.probeCountX << ",\n"
           << "    \"probe_count_y\": " << ddgiStats.probeCountY << ",\n"
           << "    \"probe_count_z\": " << ddgiStats.probeCountZ << ",\n"
           << "    \"total_probe_count\": " << ddgiStats.totalProbeCount << ",\n"
           << "    \"rays_per_probe\": " << ddgiStats.raysPerProbe << ",\n"
           << "    \"rays_per_update\": " << ddgiStats.raysPerUpdate << ",\n"
           << "    \"estimated_irradiance_bytes\": " << ddgiStats.estimatedIrradianceBytes << ",\n"
           << "    \"estimated_visibility_bytes\": " << ddgiStats.estimatedVisibilityBytes << ",\n"
           << "    \"estimated_relocation_bytes\": " << ddgiStats.estimatedRelocationBytes << ",\n"
           << "    \"probe_spacing\": " << ddgiStats.probeSpacing << ",\n"
           << "    \"hysteresis\": " << ddgiStats.hysteresis << ",\n"
           << "    \"intensity\": " << ddgiStats.intensity << ",\n"
           << "    \"overlay_enabled\": " << (ddgiStats.overlayEnabled ? "true" : "false") << ",\n"
           << "    \"probe_composite_available\": " << (ddgiStats.probeCompositeAvailable ? "true" : "false") << ",\n"
           << "    \"storage_composite_available\": " << (ddgiStats.storageCompositeAvailable ? "true" : "false") << ",\n"
           << "    \"moment_validation_available\": " << (ddgiStats.momentValidationAvailable ? "true" : "false") << ",\n"
           << "    \"spatial_filtering_available\": " << (ddgiStats.spatialFilteringAvailable ? "true" : "false") << ",\n"
           << "    \"ray_update_available\": " << (ddgiStats.rayUpdateAvailable ? "true" : "false") << ",\n"
           << "    \"temporal_blend_available\": " << (ddgiStats.temporalBlendAvailable ? "true" : "false") << ",\n"
           << "    \"probe_relocation_available\": " << (ddgiStats.probeRelocationAvailable ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"voxel_gi_stats\": {\n"
           << "    \"requested\": " << (voxelGiStats.requested ? "true" : "false") << ",\n"
           << "    \"storage_available\": " << (voxelGiStats.storageAvailable ? "true" : "false") << ",\n"
           << "    \"radiance_available\": " << (voxelGiStats.radianceAvailable ? "true" : "false") << ",\n"
           << "    \"occupancy_available\": " << (voxelGiStats.occupancyAvailable ? "true" : "false") << ",\n"
           << "    \"visualization_available\": " << (voxelGiStats.visualizationAvailable ? "true" : "false") << ",\n"
           << "    \"resolution\": " << voxelGiStats.resolution << ",\n"
           << "    \"total_voxels\": " << voxelGiStats.totalVoxels << ",\n"
           << "    \"world_extent\": " << voxelGiStats.worldExtent << ",\n"
           << "    \"voxel_size\": " << voxelGiStats.voxelSize << ",\n"
           << "    \"estimated_radiance_bytes\": " << voxelGiStats.estimatedRadianceBytes << ",\n"
           << "    \"estimated_occupancy_bytes\": " << voxelGiStats.estimatedOccupancyBytes << "\n"
           << "  },\n"
           << "  \"shadow_map\": " << (settings.enableShadowMap ? "true" : "false") << ",\n"
           << "  \"shadow_map_size\": " << settings.shadowMapSize << ",\n"
           << "  \"shadow_cascade_count\": " << settings.shadowCascadeCount << ",\n"
           << "  \"shadow_cascade_lambda\": " << settings.shadowCascadeLambda << ",\n"
           << "  \"shadow_bias\": " << settings.shadowBias << ",\n"
           << "  \"shadow_normal_bias\": " << settings.shadowNormalBias << ",\n"
           << "  \"shadow_strength\": " << settings.shadowStrength << ",\n"
           << "  \"shadow_pcss\": " << (settings.enablePcssShadows ? "true" : "false") << ",\n"
           << "  \"shadow_filter_radius\": " << settings.shadowFilterRadius << ",\n"
           << "  \"contact_shadows\": " << (settings.enableContactShadows ? "true" : "false") << ",\n"
           << "  \"contact_shadow_length\": " << settings.contactShadowLength << ",\n"
           << "  \"contact_shadow_intensity\": " << settings.contactShadowIntensity << ",\n"
           << "  \"async_compute\": " << (settings.enableAsyncCompute ? "true" : "false") << ",\n"
           << "  \"async_compute_timeline\": " << (settings.showAsyncComputeTimeline ? "true" : "false") << ",\n"
           << "  \"transfer_queue_available\": " << (device.HasTransferQueue() ? "true" : "false") << ",\n"
           << "  \"gpu_driven\": " << (settings.enableGpuDrivenRendering ? "true" : "false") << ",\n"
           << "  \"gpu_driven_stats\": {\n"
           << "    \"total_surfaces\": " << gpuDrivenStats.totalSurfaces << ",\n"
           << "    \"visible_surfaces\": " << gpuDrivenStats.visibleSurfaces << ",\n"
           << "    \"culled_surfaces\": " << gpuDrivenStats.culledSurfaces << ",\n"
           << "    \"indirect_draw_estimate\": " << gpuDrivenStats.indirectDrawEstimate << ",\n"
           << "    \"visibility_set_valid\": " << (gpuDrivenStats.visibilitySetValid ? "true" : "false") << ",\n"
           << "    \"indirect_draw_enabled\": " << (gpuDrivenStats.indirectDrawEnabled ? "true" : "false") << ",\n"
           << "    \"gpu_driven_backend\": " << (gpuDrivenStats.gpuDrivenBackend ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"meshlet_culling\": " << (settings.enableMeshletCulling ? "true" : "false") << ",\n"
           << "  \"meshlet_cluster_stats\": {\n"
           << "    \"backend\": \"" << (meshletStats.visibilityStorageAvailable ? "VisibilityStorage" : "Staged") << "\",\n"
           << "    \"visibility_storage_available\": " << (meshletStats.visibilityStorageAvailable ? "true" : "false") << ",\n"
           << "    \"estimated_visibility_bytes\": " << meshletStats.estimatedVisibilityBytes << ",\n"
           << "    \"cluster_count\": " << meshletStats.totalClusters << ",\n"
           << "    \"visible_clusters\": " << meshletStats.visibleClusters << ",\n"
           << "    \"culled_clusters\": " << meshletStats.culledClusters << ",\n"
           << "    \"meshlet_count\": " << meshletStats.totalMeshlets << ",\n"
           << "    \"visible_meshlets\": " << meshletStats.visibleMeshlets << ",\n"
           << "    \"culled_meshlets\": " << meshletStats.culledMeshlets << ",\n"
           << "    \"cluster_bounds\": " << meshletStats.boundsAvailable << ",\n"
           << "    \"triangles_per_meshlet\": " << meshletStats.trianglesPerMeshlet << ",\n"
           << "    \"visibility_set_valid\": " << (meshletStats.visibilitySetValid ? "true" : "false") << ",\n"
           << "    \"cone_culling\": " << (meshletStats.coneCullingEnabled ? "true" : "false") << ",\n"
           << "    \"gpu_driven_backend\": " << (meshletStats.gpuDrivenBackend ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"point_light\": " << (settings.enablePointLight ? "true" : "false") << ",\n"
           << "  \"spot_light\": " << (settings.enableSpotLight ? "true" : "false") << ",\n"
           << "  \"area_light\": " << (settings.enableAreaLight ? "true" : "false") << ",\n"
           << "  \"path_trace_next_event_estimation\": " << (settings.pathTraceNextEventEstimation ? "true" : "false") << ",\n"
           << "  \"path_trace_russian_roulette\": " << (settings.pathTraceRussianRoulette ? "true" : "false") << ",\n"
           << "  \"path_trace_russian_roulette_depth\": " << settings.pathTraceRussianRouletteDepth << ",\n"
           << "  \"path_trace_firefly_clamp\": " << settings.pathTraceFireflyClamp << ",\n"
           << "  \"path_trace_denoiser\": " << (settings.enablePathTraceDenoiser ? "true" : "false") << ",\n"
           << "  \"path_trace_denoiser_strength\": " << settings.pathTraceDenoiserStrength << ",\n"
           << "  \"path_trace_denoiser_temporal\": " << settings.pathTraceDenoiserTemporalBlend << ",\n"
           << "  \"path_trace_denoiser_iterations\": " << settings.pathTraceDenoiserIterations << ",\n"
           << "  \"path_trace_backend\": \"" << PathTraceBackendLabel(_renderer.GetActivePathTraceBackend()) << "\",\n"
           << "  \"ray_effects_stats\": {\n"
           << "    \"backend_available\": " << (rayEffectsStats.backendAvailable ? "true" : "false") << ",\n"
           << "    \"ray_query_available\": " << (rayEffectsStats.rayQueryAvailable ? "true" : "false") << ",\n"
           << "    \"rt_pipeline_available\": " << (rayEffectsStats.rtPipelineAvailable ? "true" : "false") << ",\n"
           << "    \"tlas_available\": " << (rayEffectsStats.tlasAvailable ? "true" : "false") << ",\n"
           << "    \"input_width\": " << rayEffectsStats.inputWidth << ",\n"
           << "    \"input_height\": " << rayEffectsStats.inputHeight << ",\n"
           << "    \"shadow_requested\": " << (rayEffectsStats.shadowsRequested ? "true" : "false") << ",\n"
           << "    \"ao_requested\": " << (rayEffectsStats.aoRequested ? "true" : "false") << ",\n"
           << "    \"reflection_requested\": " << (rayEffectsStats.reflectionsRequested ? "true" : "false") << ",\n"
           << "    \"gi_requested\": " << (rayEffectsStats.giRequested ? "true" : "false") << ",\n"
           << "    \"shadow_samples\": " << rayEffectsStats.shadowSamples << ",\n"
           << "    \"ao_samples\": " << rayEffectsStats.aoSamples << ",\n"
           << "    \"reflection_samples\": " << rayEffectsStats.reflectionSamples << ",\n"
           << "    \"gi_samples\": " << rayEffectsStats.giSamples << ",\n"
           << "    \"shadow_rays\": " << rayEffectsStats.estimatedShadowRays << ",\n"
           << "    \"ao_rays\": " << rayEffectsStats.estimatedAoRays << ",\n"
           << "    \"reflection_rays\": " << rayEffectsStats.estimatedReflectionRays << ",\n"
           << "    \"gi_rays\": " << rayEffectsStats.estimatedGiRays << ",\n"
           << "    \"half_resolution\": " << (rayEffectsStats.halfResolution ? "true" : "false") << ",\n"
           << "    \"denoiser_requested\": " << (rayEffectsStats.denoiserRequested ? "true" : "false") << ",\n"
           << "    \"gi_spatial_denoise_available\": " << (rayEffectsStats.giSpatialDenoiseAvailable ? "true" : "false") << ",\n"
           << "    \"temporal_accumulation\": " << (rayEffectsStats.temporalAccumulation ? "true" : "false") << ",\n"
           << "    \"gi_temporal_accumulation_available\": " << (rayEffectsStats.giTemporalAccumulationAvailable ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"frame_index\": " << _frameNumber << ",\n"
           << "  \"path_trace_frame_index\": " << _renderer.GetPathTraceFrameIndex() << ",\n"
           << "  \"emissive_intensity\": " << settings.emissiveIntensity << ",\n"
           << "  \"environment_intensity\": " << settings.environmentIntensity << ",\n"
           << "  \"environment_rotation_degrees\": " << settings.environmentRotationDegrees << ",\n"
           << "  \"environment_preset\": \"" << EnvironmentPresetLabel(settings.environmentPreset) << "\",\n"
           << "  \"external_hdri_available\": " << (settings.externalHdriAvailable ? "true" : "false") << ",\n"
           << "  \"external_hdri_path\": \"" << JsonEscape(settings.externalHdriPath.string()) << "\",\n"
           << "  \"external_hdri_width\": " << settings.externalHdriWidth << ",\n"
           << "  \"external_hdri_height\": " << settings.externalHdriHeight << ",\n"
           << "  \"external_hdri_channels\": " << settings.externalHdriChannels << ",\n"
           << "  \"external_hdri_is_hdr\": " << (settings.externalHdriIsHdr ? "true" : "false") << ",\n"
           << "  \"external_hdri_status\": \"" << JsonEscape(settings.externalHdriStatus) << "\",\n"
           << "  \"ibl_diffuse_strength\": " << settings.environmentDiffuseStrength << ",\n"
           << "  \"ibl_specular_strength\": " << settings.environmentSpecularStrength << ",\n"
           << "  \"ibl_stats\": {\n"
           << "    \"requested\": " << (iblStats.requested ? "true" : "false") << ",\n"
           << "    \"source\": \"" << (iblStats.externalSourceAvailable ? "External" : "Procedural") << "\",\n"
           << "    \"environment_map_uploaded\": " << (iblStats.environmentMapUploaded ? "true" : "false") << ",\n"
           << "    \"source_width\": " << iblStats.sourceWidth << ",\n"
           << "    \"source_height\": " << iblStats.sourceHeight << ",\n"
           << "    \"source_channels\": " << iblStats.sourceChannels << ",\n"
           << "    \"source_is_hdr\": " << (iblStats.sourceIsHdr ? "true" : "false") << ",\n"
           << "    \"environment_cubemap_resolution\": " << iblStats.environmentCubemapResolution << ",\n"
           << "    \"environment_cubemap_available\": " << (iblStats.environmentCubemapAvailable ? "true" : "false") << ",\n"
           << "    \"diffuse_cubemap_resolution\": " << iblStats.diffuseCubemapResolution << ",\n"
           << "    \"specular_cubemap_resolution\": " << iblStats.specularCubemapResolution << ",\n"
           << "    \"specular_mip_count\": " << iblStats.specularMipCount << ",\n"
           << "    \"brdf_lut_resolution\": " << iblStats.brdfLutResolution << ",\n"
           << "    \"estimated_environment_cubemap_bytes\": " << iblStats.estimatedEnvironmentCubemapBytes << ",\n"
           << "    \"estimated_diffuse_bytes\": " << iblStats.estimatedDiffuseBytes << ",\n"
           << "    \"estimated_specular_bytes\": " << iblStats.estimatedSpecularBytes << ",\n"
           << "    \"estimated_brdf_lut_bytes\": " << iblStats.estimatedBrdfLutBytes << ",\n"
           << "    \"diffuse_backend_available\": " << (iblStats.diffuseBackendAvailable ? "true" : "false") << ",\n"
           << "    \"specular_backend_available\": " << (iblStats.specularBackendAvailable ? "true" : "false") << ",\n"
           << "    \"diffuse_irradiance_available\": " << (iblStats.diffuseIrradianceAvailable ? "true" : "false") << ",\n"
           << "    \"specular_prefilter_available\": " << (iblStats.specularPrefilterAvailable ? "true" : "false") << ",\n"
           << "    \"brdf_lut_available\": " << (iblStats.brdfLutAvailable ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"exposure_ev\": " << settings.cameraExposureEv << ",\n"
           << "  \"bloom\": " << (settings.enableBloom ? "true" : "false") << ",\n"
           << "  \"bloom_threshold\": " << settings.bloomThreshold << ",\n"
           << "  \"bloom_intensity\": " << settings.bloomIntensity << ",\n"
           << "  \"fxaa\": " << (settings.enableFxaa ? "true" : "false") << ",\n"
           << "  \"motion_blur\": " << (settings.enableMotionBlur ? "true" : "false") << ",\n"
           << "  \"motion_blur_strength\": " << settings.motionBlurStrength << ",\n"
           << "  \"vignette\": " << (settings.enableVignette ? "true" : "false") << ",\n"
           << "  \"vignette_strength\": " << settings.vignetteStrength << ",\n"
           << "  \"saturation\": " << settings.colorGradingSaturation << ",\n"
           << "  \"contrast\": " << settings.colorGradingContrast << ",\n"
           << "  \"aperture_radius\": " << settings.cameraApertureRadius << ",\n"
           << "  \"focal_distance\": " << settings.cameraFocalDistance << ",\n"
           << "  \"camera_fov_degrees\": " << camera.GetFovDegrees() << ",\n"
           << "  \"camera_near\": " << camera.GetNearPlane() << ",\n"
           << "  \"camera_far\": " << camera.GetFarPlane() << ",\n"
           << "  \"frame_gpu_ms\": " << totalGpuMs << ",\n"
           << "  \"resource_summary\": {\n"
           << "    \"vertices\": " << scene.GetVertices().size() << ",\n"
           << "    \"triangles\": " << scene.GetTriangles().size() << ",\n"
           << "    \"surfaces\": " << scene.GetSurfaces().size() << ",\n"
           << "    \"textures_total\": " << scene.GetTextures().size() << ",\n"
           << "    \"textures_resident\": " << scene.GetResidentTextureCount() << ",\n"
           << "    \"bindless_sampled_images\": " << bindlessStats.sampledImagesUsed << ",\n"
           << "    \"bindless_sampled_image_capacity\": " << bindlessStats.sampledImagesCapacity << ",\n"
           << "    \"bindless_sampled_cube_images\": " << bindlessStats.sampledCubeImagesUsed << ",\n"
           << "    \"bindless_sampled_cube_image_capacity\": " << bindlessStats.sampledCubeImagesCapacity << ",\n"
           << "    \"bindless_storage_images\": " << bindlessStats.storageImagesUsed << ",\n"
           << "    \"bindless_storage_image_capacity\": " << bindlessStats.storageImagesCapacity << ",\n"
           << "    \"bindless_storage_buffers\": " << bindlessStats.storageBuffersUsed << ",\n"
           << "    \"bindless_storage_buffer_capacity\": " << bindlessStats.storageBuffersCapacity << ",\n"
           << "    \"gaussians\": " << scene.GetGaussianCount() << ",\n"
           << "    \"scene_buffer_bytes\": " << sceneBufferBytes << ",\n"
           << "    \"acceleration_structure_bytes\": " << accelerationBytes << ",\n"
           << "    \"tlas_resident\": " << (scene.HasRayTracingScene() ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"render_passes\": [\n";
    for (size_t passIndex = 0; passIndex < passDebugInfo.size(); ++passIndex) {
        const auto& pass = passDebugInfo[passIndex];
        output << "    {\n"
               << "      \"id\": \"" << JsonEscape(pass.id) << "\",\n"
               << "      \"name\": \"" << JsonEscape(pass.name) << "\",\n"
               << "      \"order\": " << pass.order << ",\n"
               << "      \"enabled\": " << (pass.enabled ? "true" : "false") << ",\n"
               << "      \"draw_count\": " << pass.drawCount << ",\n"
               << "      \"dispatch_count\": " << pass.dispatchCount << ",\n"
               << "      \"triangle_count\": " << pass.triangleCount << ",\n"
               << "      \"instance_count\": " << pass.instanceCount << ",\n"
               << "      \"ray_count\": " << pass.rayCount << ",\n"
               << "      \"primary_ray_count\": " << pass.primaryRayCount << ",\n"
               << "      \"shadow_ray_count\": " << pass.shadowRayCount << ",\n"
               << "      \"diffuse_ray_count\": " << pass.diffuseRayCount << ",\n"
               << "      \"specular_ray_count\": " << pass.specularRayCount << ",\n"
               << "      \"splat_count\": " << pass.splatCount << "\n"
               << "    }" << (passIndex + 1 < passDebugInfo.size() ? "," : "") << "\n";
    }
    output << "  ],\n"
           << "  \"render_graph_timings\": [\n";
    for (size_t timingIndex = 0; timingIndex < graphTimings.size(); ++timingIndex) {
        const auto& timing = graphTimings[timingIndex];
        output << "    {\n"
               << "      \"name\": \"" << JsonEscape(timing.name) << "\",\n"
               << "      \"cpu_ms\": " << timing.cpuMs << ",\n"
               << "      \"gpu_ms\": " << timing.gpuMs << ",\n"
               << "      \"gpu_timing_valid\": " << (timing.gpuTimingValid ? "true" : "false") << ",\n"
               << "      \"render_extent\": \"" << timing.renderExtent.width << "x" << timing.renderExtent.height << "\",\n"
               << "      \"read_count\": " << timing.readCount << ",\n"
               << "      \"write_count\": " << timing.writeCount << ",\n"
               << "      \"barrier_count\": " << timing.barrierCount << ",\n"
               << "      \"inputs\": [";
        for (size_t inputIndex = 0; inputIndex < timing.inputs.size(); ++inputIndex) {
            const auto& access = timing.inputs[inputIndex];
            output << (inputIndex == 0 ? "" : ", ")
                   << "{\"name\":\"" << JsonEscape(access.name) << "\",\"usage\":\"" << ResourceUsageLabel(access.usage)
                   << "\",\"format\":\"" << VkFormatLabel(access.format) << "\",\"resolution\":\""
                   << access.extent.width << "x" << access.extent.height << "\"}";
        }
        output << "],\n"
               << "      \"outputs\": [";
        for (size_t outputIndex = 0; outputIndex < timing.outputs.size(); ++outputIndex) {
            const auto& access = timing.outputs[outputIndex];
            output << (outputIndex == 0 ? "" : ", ")
                   << "{\"name\":\"" << JsonEscape(access.name) << "\",\"usage\":\"" << ResourceUsageLabel(access.usage)
                   << "\",\"format\":\"" << VkFormatLabel(access.format) << "\",\"resolution\":\""
                   << access.extent.width << "x" << access.extent.height << "\"}";
        }
        output << "]\n"
               << "    }" << (timingIndex + 1 < graphTimings.size() ? "," : "") << "\n";
    }
    output << "  ]\n"
           << "}\n";
    log_startup_event("Capture metadata written: " + metadataPath.string());
    return true;
}

void VestaEngine::init_imgui()
{
    auto& device = _renderer.GetRenderDevice();

    // ImGui needs its own descriptor pool because the backend allocates font and
    // UI resources independently from the renderer's bindless heap.
    constexpr std::array<VkDescriptorPoolSize, 1> poolSizes{
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 256;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device.GetDevice(), &poolInfo, nullptr, &_imguiDescriptorPool));

    IMGUI_CHECKVERSION();
    _imguiContext = ImGui::CreateContext();
    ImGui::SetCurrentContext(_imguiContext);

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
#if defined(IMGUI_HAS_DOCK)
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
#endif
    io.IniFilename = nullptr;

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.0f;
    style.FrameRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.PopupRounding = 8.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.08f, 0.10f, 0.92f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.14f, 0.18f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.14f, 0.22f, 0.28f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.33f, 0.39f, 0.80f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.46f, 0.53f, 0.90f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.34f, 0.56f, 0.63f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.72f, 0.34f, 0.18f, 0.82f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.86f, 0.44f, 0.22f, 0.92f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.94f, 0.54f, 0.28f, 1.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.17f, 0.21f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.17f, 0.24f, 0.29f, 1.0f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.31f, 0.37f, 1.0f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.86f, 0.44f, 0.22f, 0.95f);
    colors[ImGuiCol_SliderGrabActive] = ImVec4(0.96f, 0.54f, 0.28f, 1.0f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.95f, 0.64f, 0.29f, 1.0f);
    colors[ImGuiCol_Separator] = ImVec4(0.25f, 0.32f, 0.36f, 1.0f);

    ImGui_ImplSDL2_InitForVulkan(_window);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = device.GetInstance();
    initInfo.PhysicalDevice = device.GetPhysicalDevice();
    initInfo.Device = device.GetDevice();
    initInfo.QueueFamily = device.GetGraphicsQueueFamily();
    initInfo.Queue = device.GetGraphicsQueue();
    initInfo.DescriptorPool = _imguiDescriptorPool;
    initInfo.MinImageCount = std::max(2u, static_cast<uint32_t>(device.GetSwapchainImageHandles().size()));
    initInfo.ImageCount = static_cast<uint32_t>(device.GetSwapchainImageHandles().size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    const VkFormat swapchainFormat = device.GetSwapchainFormat();
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;
    initInfo.CheckVkResultFn = CheckImGuiVkResult;
    ImGui_ImplVulkan_Init(&initInfo);

    // The renderer exposes a late overlay hook so the engine can keep ImGui
    // ownership without making the renderer depend on ImGui types.
    _renderer.SetOverlayCallbacks(
        [this](VkCommandBuffer commandBuffer) {
            if (!_imguiInitialized) {
                return;
            }

            ImGui::SetCurrentContext(_imguiContext);
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
        },
        [this](uint32_t imageCount) {
            if (!_imguiInitialized) {
                return;
            }

            ImGui::SetCurrentContext(_imguiContext);
            ImGui_ImplVulkan_SetMinImageCount(std::max(2u, imageCount));
        });

    _imguiInitialized = true;
}

void VestaEngine::shutdown_imgui()
{
    if (!_imguiInitialized) {
        return;
    }

    _renderer.ClearOverlayCallbacks();
    ImGui::SetCurrentContext(_imguiContext);
    clear_texture_preview_descriptors();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext(_imguiContext);
    if (_imguiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(_renderer.GetRenderDevice().GetDevice(), _imguiDescriptorPool, nullptr);
        _imguiDescriptorPool = VK_NULL_HANDLE;
    }
    _imguiContext = nullptr;
    _imguiInitialized = false;
}

void VestaEngine::clear_texture_preview_descriptors()
{
    if (!_imguiInitialized) {
        _texturePreviewDescriptors.clear();
        _frameTexturePreviewDescriptors.clear();
        _frameTexturePreviewImages.clear();
        _engineTexturePreviewDescriptors.clear();
        _engineTexturePreviewImages.clear();
        _texturePreviewSceneVersion = 0;
        return;
    }

    ImGui::SetCurrentContext(_imguiContext);
    for (VkDescriptorSet descriptor : _texturePreviewDescriptors) {
        if (descriptor != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(descriptor);
        }
    }
    for (VkDescriptorSet descriptor : _frameTexturePreviewDescriptors) {
        if (descriptor != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(descriptor);
        }
    }
    for (VkDescriptorSet descriptor : _engineTexturePreviewDescriptors) {
        if (descriptor != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(descriptor);
        }
    }
    _texturePreviewDescriptors.clear();
    _frameTexturePreviewDescriptors.clear();
    _frameTexturePreviewImages.clear();
    _engineTexturePreviewDescriptors.clear();
    _engineTexturePreviewImages.clear();
    _texturePreviewSceneVersion = 0;
}

void VestaEngine::begin_imgui_frame(float deltaSeconds)
{
    if (!_imguiInitialized) {
        return;
    }

    ImGui::SetCurrentContext(_imguiContext);
    ImGuiIO& io = ImGui::GetIO();
    io.DeltaTime = std::max(deltaSeconds, 1.0f / 240.0f);

    ImGui_ImplSDL2_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    build_main_menu_bar();
    build_debug_dockspace();
    build_debug_ui();
    draw_light_gizmo_overlay();
    ImGui::Render();
}

void VestaEngine::build_debug_dockspace()
{
    if (!_imguiInitialized || (!_showDebugUi && !has_debug_window_open())) {
        return;
    }

    ImGui::SetCurrentContext(_imguiContext);
#if defined(IMGUI_HAS_DOCK)
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoBackground;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    if (ImGui::Begin("VestaEngine Debug DockSpace", nullptr, windowFlags)) {
        const ImGuiID dockspaceId = ImGui::GetID("VestaEngineDebugDockSpace");
        ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    }
    ImGui::End();
    ImGui::PopStyleVar(3);
#endif
}

bool VestaEngine::has_debug_window_open() const
{
    return _showDetailedStats || _showLegacyStatsPanel || _showLegacyRenderPanel || _showLegacyCameraPanel
        || _showFrameOverview || _showRenderGraphPanel || _showGpuProfilerPanel || _showDebugVisualizationPanel
        || _showRenderModeControlPanel || _showSceneInspectorPanel || _showResourceInspectorPanel || _showLogConsolePanel;
}

void VestaEngine::draw_light_gizmo_overlay()
{
    if (!_imguiInitialized || (!_showDebugUi && !has_debug_window_open())) {
        return;
    }

    const auto& selection = _renderer.GetSelection();
    const bool lightSelected = selection.kind == vesta::render::SelectionKind::PointLight
        || selection.kind == vesta::render::SelectionKind::SpotLight || selection.kind == vesta::render::SelectionKind::AreaLight;
    const auto& scene = _renderer.GetScene();
    const auto& settings = _renderer.GetSettings();
    const bool gaussianOverlay = settings.gaussianShowCovarianceEllipsoids && !scene.GetGaussians().empty();
    if (!lightSelected && !gaussianOverlay) {
        return;
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    if (viewport == nullptr) {
        return;
    }

    const Camera& camera = _renderer.GetCamera();
    ImDrawList* drawList = ImGui::GetForegroundDrawList();
    const ImVec2 origin = viewport->Pos;
    const ImVec2 size = viewport->Size;

    auto project = [&](glm::vec3 position) {
        return ProjectWorldToViewport(camera, position, origin, size);
    };
    auto normalized = [](glm::vec3 value, glm::vec3 fallback) {
        const float length = glm::length(value);
        return length > 1.0e-4f ? value / length : fallback;
    };
    auto drawLabel = [&](ImVec2 anchor, const char* label, ImU32 color) {
        const ImVec2 textPos(anchor.x + 12.0f, anchor.y - 18.0f);
        drawList->AddText(ImVec2(textPos.x + 1.0f, textPos.y + 1.0f), IM_COL32(0, 0, 0, 180), label);
        drawList->AddText(textPos, color, label);
    };

    if (gaussianOverlay) {
        const auto& gaussians = scene.GetGaussians();
        _selectedGaussianInspectorIndex =
            std::min(_selectedGaussianInspectorIndex, static_cast<uint32_t>(gaussians.size() - 1u));
        const auto& gaussian = gaussians[_selectedGaussianInspectorIndex];
        const glm::vec3 center(gaussian.positionOpacity);
        const auto centerScreen = project(center);
        if (centerScreen.has_value()) {
            const float sceneRadius = std::max(scene.GetBounds().radius, 0.001f);
            const glm::vec3 rawScale = glm::max(glm::abs(glm::vec3(gaussian.scale)), glm::vec3(1.0e-5f));
            const glm::vec3 axisLength = glm::clamp(rawScale * _gaussianInspectorOverlayScale,
                glm::vec3(sceneRadius * 0.002f),
                glm::vec3(sceneRadius * 0.14f));
            const glm::vec3 baseColor = GaussianBaseColor(gaussian);
            const ImU32 centerColor = IM_COL32(
                static_cast<int>(baseColor.r * 255.0f),
                static_cast<int>(baseColor.g * 255.0f),
                static_cast<int>(baseColor.b * 255.0f),
                255);
            drawList->AddCircleFilled(*centerScreen, 4.0f, centerColor, 20);
            drawList->AddCircle(*centerScreen, 12.0f, IM_COL32(255, 255, 255, 220), 28, 1.5f);
            if (_gaussianInspectorShowAxes) {
                const std::array<std::pair<glm::vec3, ImU32>, 3> axes{
                    std::pair<glm::vec3, ImU32>{ glm::vec3(axisLength.x, 0.0f, 0.0f), IM_COL32(255, 90, 90, 230) },
                    std::pair<glm::vec3, ImU32>{ glm::vec3(0.0f, axisLength.y, 0.0f), IM_COL32(90, 255, 130, 230) },
                    std::pair<glm::vec3, ImU32>{ glm::vec3(0.0f, 0.0f, axisLength.z), IM_COL32(100, 160, 255, 230) },
                };
                for (const auto& [axis, color] : axes) {
                    const auto positive = project(center + RotateByGaussianQuaternion(gaussian.rotation, axis));
                    const auto negative = project(center - RotateByGaussianQuaternion(gaussian.rotation, axis));
                    if (positive.has_value() && negative.has_value()) {
                        drawList->AddLine(*negative, *positive, color, 2.0f);
                    }
                }
            }
            const std::string label = fmt::format("Gaussian #{}  opacity {:.2f}", _selectedGaussianInspectorIndex, gaussian.positionOpacity.w);
            drawLabel(*centerScreen, label.c_str(), IM_COL32(255, 255, 255, 255));
        }
    }

    if (!lightSelected) {
        return;
    }

    if (selection.kind == vesta::render::SelectionKind::PointLight) {
        const glm::vec3 position(settings.pointLightPositionAndIntensity);
        const auto screen = project(position);
        if (!screen.has_value()) {
            return;
        }
        const ImU32 color = settings.enablePointLight ? IM_COL32(255, 203, 82, 255) : IM_COL32(150, 150, 150, 180);
        drawList->AddCircleFilled(*screen, 5.0f, color, 24);
        drawList->AddCircle(*screen, 15.0f, color, 32, 2.0f);
        drawList->AddLine(ImVec2(screen->x - 20.0f, screen->y), ImVec2(screen->x + 20.0f, screen->y), color, 1.5f);
        drawList->AddLine(ImVec2(screen->x, screen->y - 20.0f), ImVec2(screen->x, screen->y + 20.0f), color, 1.5f);
        drawLabel(*screen, "Point Light", color);
        return;
    }

    if (selection.kind == vesta::render::SelectionKind::SpotLight) {
        const glm::vec3 position(settings.spotLightPositionAndIntensity);
        const glm::vec3 direction = normalized(glm::vec3(settings.spotLightDirectionAndAngle), glm::vec3(0.0f, -1.0f, 0.0f));
        const auto screen = project(position);
        const auto tip = project(position + direction * 1.2f);
        if (!screen.has_value()) {
            return;
        }
        const ImU32 color = settings.enableSpotLight ? IM_COL32(255, 174, 88, 255) : IM_COL32(150, 150, 150, 180);
        drawList->AddCircleFilled(*screen, 5.0f, color, 20);
        drawList->AddCircle(*screen, 13.0f, color, 28, 2.0f);
        if (tip.has_value()) {
            drawList->AddLine(*screen, *tip, color, 2.0f);
            drawList->AddTriangleFilled(
                *tip, ImVec2(tip->x - 5.0f, tip->y + 10.0f), ImVec2(tip->x + 5.0f, tip->y + 10.0f), color);
        }
        drawLabel(*screen, "Spot Light", color);
        return;
    }

    if (selection.kind == vesta::render::SelectionKind::AreaLight) {
        const glm::vec3 position(settings.areaLightPositionAndIntensity);
        const glm::vec3 normal = normalized(glm::vec3(settings.areaLightNormalAndSize), glm::vec3(0.0f, -1.0f, 0.0f));
        const auto screen = project(position);
        const auto normalTip = project(position + normal * 1.0f);
        if (!screen.has_value()) {
            return;
        }
        const ImU32 color = settings.enableAreaLight ? IM_COL32(129, 188, 255, 255) : IM_COL32(150, 150, 150, 180);
        const float halfExtent = 13.0f;
        drawList->AddQuad(
            ImVec2(screen->x, screen->y - halfExtent),
            ImVec2(screen->x + halfExtent, screen->y),
            ImVec2(screen->x, screen->y + halfExtent),
            ImVec2(screen->x - halfExtent, screen->y),
            color,
            2.0f);
        drawList->AddCircleFilled(*screen, 4.0f, color, 16);
        if (normalTip.has_value()) {
            drawList->AddLine(*screen, *normalTip, color, 2.0f);
        }
        drawLabel(*screen, "Area Light", color);
    }
}

void VestaEngine::build_main_menu_bar()
{
    if (!_imguiInitialized) {
        return;
    }

    ImGui::SetCurrentContext(_imguiContext);

    auto& settings = _renderer.GetSettings();
    const bool sceneLoadInProgress = _renderer.IsSceneLoadInProgress();
    settings.preferAsyncSceneLoading = UseAsyncSceneLoading(settings);

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::BeginMenu("Open Scene...", !sceneLoadInProgress)) {
                if (ImGui::MenuItem("Scene File...")) {
                    if (const std::optional<std::filesystem::path> path = open_scene_with_system_dialog()) {
                        load_scene_path(*path);
                    }
                }
                if (ImGui::MenuItem("Gaussian Model Folder...")) {
                    if (const std::optional<std::filesystem::path> path = open_gaussian_model_with_system_dialog()) {
                        load_scene_path(*path);
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Open Recent", !_recentScenePaths.empty())) {
                for (const std::filesystem::path& recentPath : _recentScenePaths) {
                    const std::string label = ScenePathLabel(recentPath);
                    if (ImGui::MenuItem(label.c_str(), nullptr, false, !sceneLoadInProgress)) {
                        load_scene_path(recentPath);
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Load basicmesh.glb", nullptr, false, !sceneLoadInProgress)) {
                load_scene_path("assets/basicmesh.glb");
            }
            if (ImGui::MenuItem("Load structure.glb", nullptr, false, !sceneLoadInProgress)) {
                load_scene_path("assets/structure.glb");
            }
            if (ImGui::MenuItem("Load DamagedHelmet.glb", nullptr, false, !sceneLoadInProgress)) {
                load_scene_path("assets/demo/DamagedHelmet.glb");
            }
            if (ImGui::MenuItem("Load garden_input.ply", nullptr, false, !sceneLoadInProgress)) {
                load_scene_path("assets/demo/garden_input.ply");
            }
            if (ImGui::BeginMenu("Benchmark Scenes", !sceneLoadInProgress)) {
                for (const BenchmarkScenePreset& preset : kBenchmarkScenePresets) {
                    const bool exists = std::filesystem::exists(preset.path);
                    if (ImGui::MenuItem(preset.label, nullptr, false, exists)) {
                        load_scene_path(preset.path);
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem(
                    "Reload Current", nullptr, false, !sceneLoadInProgress && !_renderer.GetScene().GetSourcePath().empty())) {
                if (UseAsyncSceneLoading(settings)) {
                    _renderer.ReloadSceneAsync();
                } else {
                    _renderer.LoadScene(_renderer.GetScene().GetSourcePath());
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                SDL_Event quitEvent{};
                quitEvent.type = SDL_QUIT;
                SDL_PushEvent(&quitEvent);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Debug Overlay", nullptr, &_showDebugUi);
            ImGui::Separator();

            bool compositeSelected = settings.displayMode == vesta::render::RendererDisplayMode::Composite;
            if (ImGui::MenuItem("Composite", nullptr, compositeSelected)) {
                vesta::render::ApplyDisplayModePassSelection(settings, vesta::render::RendererDisplayMode::Composite);
                _renderer.ResetAccumulation();
            }

            bool deferredSelected = settings.displayMode == vesta::render::RendererDisplayMode::DeferredLighting;
            if (ImGui::MenuItem("Raster", nullptr, deferredSelected)) {
                vesta::render::ApplyDisplayModePassSelection(settings, vesta::render::RendererDisplayMode::DeferredLighting);
                _renderer.ResetAccumulation();
            }

            bool gaussianSelected = settings.displayMode == vesta::render::RendererDisplayMode::Gaussian;
            if (ImGui::MenuItem("Gaussian", nullptr, gaussianSelected)) {
                vesta::render::ApplyDisplayModePassSelection(settings, vesta::render::RendererDisplayMode::Gaussian);
                _renderer.ResetAccumulation();
            }

            bool rayTracingSelected = settings.displayMode == vesta::render::RendererDisplayMode::RayTracing;
            if (ImGui::MenuItem("Ray Tracing", nullptr, rayTracingSelected)) {
                vesta::render::ApplyDisplayModePassSelection(settings, vesta::render::RendererDisplayMode::RayTracing);
                _renderer.ResetAccumulation();
            }

            bool pathTraceSelected = settings.displayMode == vesta::render::RendererDisplayMode::PathTrace;
            if (ImGui::MenuItem("Path Trace", nullptr, pathTraceSelected)) {
                vesta::render::ApplyDisplayModePassSelection(settings, vesta::render::RendererDisplayMode::PathTrace);
                _renderer.ResetAccumulation();
            }
            ImGui::Separator();
            if (ImGui::BeginMenu("Debug Views")) {
                const auto debugViewItem = [&](const char* label, vesta::render::RendererDebugView view) {
                    const bool selected = settings.debugView == view;
                    if (ImGui::MenuItem(label, nullptr, selected)) {
                        vesta::render::SelectRendererDebugView(settings, view);
                        _showDebugUi = true;
                        _showDebugVisualizationPanel = true;
                        _renderer.ResetAccumulation();
                    }
                };
                debugViewItem("Final Color", vesta::render::RendererDebugView::FinalColor);
                ImGui::SeparatorText("Material");
                debugViewItem("Albedo / Base Color", vesta::render::RendererDebugView::Albedo);
                debugViewItem("Normal", vesta::render::RendererDebugView::Normal);
                debugViewItem("Roughness", vesta::render::RendererDebugView::Roughness);
                debugViewItem("Metallic", vesta::render::RendererDebugView::Metallic);
                debugViewItem("Emissive", vesta::render::RendererDebugView::Emissive);
                ImGui::SeparatorText("Raster");
                debugViewItem("Wireframe", vesta::render::RendererDebugView::Wireframe);
                debugViewItem("Overdraw", vesta::render::RendererDebugView::Overdraw);
                debugViewItem("Linear Depth", vesta::render::RendererDebugView::Depth);
                debugViewItem("Shadow Cascade", vesta::render::RendererDebugView::ShadowCascade);
                debugViewItem("Motion Vector", vesta::render::RendererDebugView::MotionVector);
                ImGui::SeparatorText("Lighting");
                debugViewItem("Direct Lighting", vesta::render::RendererDebugView::DirectLighting);
                debugViewItem("Indirect Lighting", vesta::render::RendererDebugView::IndirectLighting);
                debugViewItem("Reflection", vesta::render::RendererDebugView::Reflection);
                debugViewItem("Denoised Result", vesta::render::RendererDebugView::DenoisedResult);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Path Tracing AOV")) {
                const auto pathTraceViewItem = [&](const char* label, vesta::render::PathTraceDebugView view) {
                    const bool selected = settings.pathTraceDebugView == view;
                    if (ImGui::MenuItem(label, nullptr, selected)) {
                        vesta::render::SelectPathTraceDebugView(settings, view);
                        vesta::render::ApplyDisplayModePassSelection(settings, vesta::render::RendererDisplayMode::PathTrace);
                        _showDebugUi = true;
                        _showDebugVisualizationPanel = true;
                        _renderer.ResetAccumulation();
                    }
                };
                pathTraceViewItem("Final", vesta::render::PathTraceDebugView::Final);
                pathTraceViewItem("Albedo", vesta::render::PathTraceDebugView::Albedo);
                pathTraceViewItem("Normal", vesta::render::PathTraceDebugView::Normal);
                pathTraceViewItem("Depth", vesta::render::PathTraceDebugView::Depth);
                ImGui::SeparatorText("Lighting");
                pathTraceViewItem("Direct", vesta::render::PathTraceDebugView::Direct);
                pathTraceViewItem("Indirect", vesta::render::PathTraceDebugView::Indirect);
                pathTraceViewItem("Diffuse Bounce", vesta::render::PathTraceDebugView::DiffuseBounce);
                pathTraceViewItem("Specular Bounce", vesta::render::PathTraceDebugView::SpecularBounce);
                ImGui::SeparatorText("Integrator");
                pathTraceViewItem("Ray Count Heatmap", vesta::render::PathTraceDebugView::RayCountHeatmap);
                pathTraceViewItem("Throughput", vesta::render::PathTraceDebugView::Throughput);
                pathTraceViewItem("PDF", vesta::render::PathTraceDebugView::Pdf);
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Global Illumination", nullptr, settings.enableGlobalIllumination)) {
                settings.enableGlobalIllumination = !settings.enableGlobalIllumination;
                _renderer.ResetAccumulation();
            }
            if (ImGui::MenuItem("Ambient Occlusion", nullptr, settings.enableAmbientOcclusion)) {
                settings.enableAmbientOcclusion = !settings.enableAmbientOcclusion;
                _renderer.ResetAccumulation();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Options")) {
            if (ImGui::MenuItem("Optimize Inactive Passes", nullptr, settings.optimizeInactivePasses)) {
                settings.optimizeInactivePasses = !settings.optimizeInactivePasses;
                _renderer.ResetAccumulation();
            }
            if (ImGui::BeginMenu("Scene Upload Mode")) {
                const bool syncSelected = settings.sceneUploadMode == vesta::render::SceneUploadMode::Synchronous;
                if (ImGui::MenuItem("Synchronous", nullptr, syncSelected)) {
                    settings.sceneUploadMode = vesta::render::SceneUploadMode::Synchronous;
                    settings.preferAsyncSceneLoading = false;
                }

                const bool asyncSelected = settings.sceneUploadMode == vesta::render::SceneUploadMode::AsyncParseSyncUpload;
                if (ImGui::MenuItem("Async Parse + Sync Upload", nullptr, asyncSelected)) {
                    settings.sceneUploadMode = vesta::render::SceneUploadMode::AsyncParseSyncUpload;
                    settings.preferAsyncSceneLoading = true;
                }

                const bool streamingSelected = settings.sceneUploadMode == vesta::render::SceneUploadMode::Streaming;
                if (ImGui::MenuItem("Streaming", nullptr, streamingSelected, settings.useDeviceLocalSceneBuffers)) {
                    settings.sceneUploadMode = vesta::render::SceneUploadMode::Streaming;
                    settings.preferAsyncSceneLoading = true;
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Preset")) {
                if (ImGui::MenuItem("Recommended")) {
                    _renderer.ApplyPreset(vesta::render::RendererPreset::Recommended);
                }
                if (ImGui::MenuItem("Performance")) {
                    _renderer.ApplyPreset(vesta::render::RendererPreset::Performance);
                }
                if (ImGui::MenuItem("Balanced")) {
                    _renderer.ApplyPreset(vesta::render::RendererPreset::Balanced);
                }
                if (ImGui::MenuItem("Quality")) {
                    _renderer.ApplyPreset(vesta::render::RendererPreset::Quality);
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Engine Tuning")) {
                ImGui::MenuItem("Use Device Local Scene Buffers", nullptr, &settings.useDeviceLocalSceneBuffers);
                ImGui::MenuItem("Use Device Local Textures", nullptr, &settings.useDeviceLocalTextures);
                ImGui::MenuItem("Texture Streaming", nullptr, &settings.textureStreamingEnabled);
                ImGui::MenuItem("Build RT Structures On Load", nullptr, &settings.buildRayTracingStructuresOnLoad);
                ImGui::MenuItem("Defer Old Scene Destruction", nullptr, &settings.deferOldSceneDestruction);
                ImGui::MenuItem("Auto Focus Scene On Load", nullptr, &settings.autoFocusSceneOnLoad);
                ImGui::MenuItem("Frustum Culling", nullptr, &settings.enableFrustumCulling);
                ImGui::MenuItem("Distance Culling", nullptr, &settings.enableDistanceCulling);
                ImGui::MenuItem("Use Indirect Draw", nullptr, &settings.useIndirectDraw);
                ImGui::MenuItem("Frame Timing Capture", nullptr, &settings.frameTimingCapture);
                ImGui::MenuItem("Benchmark Overlay", nullptr, &settings.benchmarkOverlay);
                bool vsyncEnabled = settings.enableVSync;
                if (ImGui::MenuItem("VSync", nullptr, vsyncEnabled)) {
                    _renderer.SetVSyncEnabled(!vsyncEnabled);
                    log_startup_event(fmt::format("VSync {} ({})",
                        !vsyncEnabled ? "enabled" : "disabled",
                        PresentModeLabel(_renderer.GetRenderDevice().GetPresentMode())));
                }
                ImGui::MenuItem("FPS Limit", nullptr, &settings.enableFpsLimit);
                int fpsLimit = static_cast<int>(settings.fpsLimit);
                if (ImGui::SliderInt("FPS Limit Value", &fpsLimit, 15, 360)) {
                    settings.fpsLimit = static_cast<uint32_t>(fpsLimit);
                }
                int uploadBudgetMiB = static_cast<int>(settings.maxUploadBytesPerFrame / (1024u * 1024u));
                if (ImGui::SliderInt("Upload Budget (MiB)", &uploadBudgetMiB, 1, 256)) {
                    settings.maxUploadBytesPerFrame = static_cast<uint32_t>(uploadBudgetMiB) * 1024u * 1024u;
                }
                int textureUploadBudgetMiB = static_cast<int>(settings.maxTextureUploadBytesPerFrame / (1024u * 1024u));
                if (ImGui::SliderInt("Texture Budget (MiB)", &textureUploadBudgetMiB, 1, 512)) {
                    settings.maxTextureUploadBytesPerFrame = static_cast<uint32_t>(textureUploadBudgetMiB) * 1024u * 1024u;
                }
                ImGui::SliderFloat("Distance Cull Scale", &settings.distanceCullScale, 1.0f, 100.0f, "%.1f");
                ImGui::Separator();
                ImGui::TextDisabled("Validation: %s", bUseValidationLayers ? "Debug default" : "Off");
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("PT Backend")) {
                bool autoSelected = settings.pathTraceBackend == vesta::render::PathTraceBackend::Auto;
                if (ImGui::MenuItem("Auto", nullptr, autoSelected)) {
                    settings.pathTraceBackend = vesta::render::PathTraceBackend::Auto;
                    _renderer.ResetAccumulation();
                }

                bool computeSelected = settings.pathTraceBackend == vesta::render::PathTraceBackend::Compute;
                if (ImGui::MenuItem("Compute", nullptr, computeSelected)) {
                    settings.pathTraceBackend = vesta::render::PathTraceBackend::Compute;
                    _renderer.ResetAccumulation();
                }

                bool hardwareSelected = settings.pathTraceBackend == vesta::render::PathTraceBackend::HardwareRT;
                if (ImGui::MenuItem(
                        "Hardware RT", nullptr, hardwareSelected, _renderer.GetRenderDevice().IsRayTracingSupported())) {
                    settings.pathTraceBackend = vesta::render::PathTraceBackend::HardwareRT;
                    _renderer.ResetAccumulation();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Path Tracing")) {
                const uint32_t frameIndex = _renderer.GetPathTraceFrameIndex();
                ImGui::Text("Progress");
                DrawPathTraceProgressBar(settings, frameIndex, ImVec2(220.0f, 0.0f));
                int targetFrames = static_cast<int>(settings.pathTraceTargetFrames);
                if (ImGui::InputInt("Target Frames", &targetFrames, 16, 64)) {
                    settings.pathTraceTargetFrames = static_cast<uint32_t>(std::clamp(targetFrames, 1, 8192));
                }
                if (ImGui::SliderFloat("Resolution Scale", &settings.pathTraceResolutionScale, 0.25f, 1.0f, "%.2fx")) {
                    _renderer.ResetAccumulation();
                }
                int spp = static_cast<int>(settings.pathTraceSamplesPerPixel);
                if (ImGui::InputInt("Samples / Frame", &spp, 1, 4)) {
                    settings.pathTraceSamplesPerPixel = static_cast<uint32_t>(std::clamp(spp, 1, 64));
                    _renderer.ResetAccumulation();
                }
                int maxBounces = static_cast<int>(settings.pathTraceMaxBounces);
                if (ImGui::InputInt("Max Bounces", &maxBounces, 1, 2)) {
                    settings.pathTraceMaxBounces = static_cast<uint32_t>(std::clamp(maxBounces, 1, 16));
                    _renderer.ResetAccumulation();
                }
                if (ImGui::MenuItem("Denoiser", nullptr, settings.enablePathTraceDenoiser)) {
                    settings.enablePathTraceDenoiser = !settings.enablePathTraceDenoiser;
                    _renderer.ResetAccumulation();
                }
                if (settings.enablePathTraceDenoiser) {
                    if (ImGui::SliderFloat("Denoiser Strength", &settings.pathTraceDenoiserStrength, 0.0f, 1.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("Temporal Blend", &settings.pathTraceDenoiserTemporalBlend, 0.0f, 0.98f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                }
                if (ImGui::MenuItem("Next Event Estimation", nullptr, settings.pathTraceNextEventEstimation)) {
                    settings.pathTraceNextEventEstimation = !settings.pathTraceNextEventEstimation;
                    _renderer.ResetAccumulation();
                }
                if (ImGui::MenuItem("Russian Roulette", nullptr, settings.pathTraceRussianRoulette)) {
                    settings.pathTraceRussianRoulette = !settings.pathTraceRussianRoulette;
                    _renderer.ResetAccumulation();
                }
                if (ImGui::SliderFloat("Firefly Clamp", &settings.pathTraceFireflyClamp, 0.0f, 64.0f, "%.1f")) {
                    _renderer.ResetAccumulation();
                }
                if (ImGui::MenuItem("Reset Accumulation")) {
                    _renderer.ResetAccumulation();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Debug Aids")) {
                if (ImGui::MenuItem("G-Buffer Preview Strip", nullptr, settings.showGBufferPreview)) {
                    settings.showGBufferPreview = !settings.showGBufferPreview;
                }
                if (ImGui::MenuItem("Shadow Cascade Overlay", nullptr, settings.showShadowCascadeOverlay)) {
                    settings.showShadowCascadeOverlay = !settings.showShadowCascadeOverlay;
                    _renderer.ResetAccumulation();
                }
                if (ImGui::MenuItem("Wireframe View", nullptr, settings.debugView == vesta::render::RendererDebugView::Wireframe)) {
                    vesta::render::SelectRendererDebugView(settings,
                        settings.debugView == vesta::render::RendererDebugView::Wireframe
                            ? vesta::render::RendererDebugView::FinalColor
                            : vesta::render::RendererDebugView::Wireframe);
                    _showDebugUi = true;
                    _showDebugVisualizationPanel = true;
                    _renderer.ResetAccumulation();
                }
                if (ImGui::MenuItem("Overdraw Heatmap", nullptr, settings.debugView == vesta::render::RendererDebugView::Overdraw)) {
                    vesta::render::SelectRendererDebugView(settings,
                        settings.debugView == vesta::render::RendererDebugView::Overdraw
                            ? vesta::render::RendererDebugView::FinalColor
                            : vesta::render::RendererDebugView::Overdraw);
                    _showDebugUi = true;
                    _showDebugVisualizationPanel = true;
                    _renderer.ResetAccumulation();
                }
                if (ImGui::MenuItem("Temporal History", nullptr, settings.debugView == vesta::render::RendererDebugView::TemporalHistoryColor)) {
                    vesta::render::SelectRendererDebugView(settings,
                        settings.debugView == vesta::render::RendererDebugView::TemporalHistoryColor
                            ? vesta::render::RendererDebugView::FinalColor
                            : vesta::render::RendererDebugView::TemporalHistoryColor);
                    _showDebugUi = true;
                    _showDebugVisualizationPanel = true;
                    _renderer.ResetAccumulation();
                }
                ImGui::MenuItem("Show Upscaler Debug", nullptr, &settings.showTemporalUpscalerDebug);
                ImGui::MenuItem("Gaussian Tile Grid", nullptr, &settings.gaussianShowTileGrid);
                ImGui::MenuItem("Gaussian Spatial Bounds", nullptr, &settings.gaussianShowSpatialBounds);
                if (ImGui::MenuItem("Reset Debug Views")) {
                    vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::FinalColor);
                    settings.gaussianDebugView = vesta::render::GaussianDebugView::Final;
                    settings.compareMode = vesta::render::CompareMode::Off;
                    _renderer.ResetAccumulation();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Directional Light")) {
                float direction[3] = {
                    settings.lightDirectionAndIntensity.x,
                    settings.lightDirectionAndIntensity.y,
                    settings.lightDirectionAndIntensity.z,
                };
                if (ImGui::SliderFloat3("Direction", direction, -1.0f, 1.0f, "%.2f")) {
                    glm::vec3 normalized(direction[0], direction[1], direction[2]);
                    if (glm::length(normalized) > 1.0e-4f) {
                        normalized = glm::normalize(normalized);
                        settings.lightDirectionAndIntensity =
                            glm::vec4(normalized, settings.lightDirectionAndIntensity.w);
                        _renderer.ResetAccumulation();
                    }
                }
                if (ImGui::SliderFloat("Intensity", &settings.lightDirectionAndIntensity.w, 0.0f, 8.0f, "%.2f")) {
                    _renderer.ResetAccumulation();
                }
                if (ImGui::SliderFloat("Emission", &settings.emissiveIntensity, 0.0f, 64.0f, "%.2f")) {
                    _renderer.ResetAccumulation();
                }
                ImGui::SeparatorText("Environment");
                const char* environmentPresets[] = { "Studio", "Sunset", "Night", "Forest" };
                int environmentPreset = static_cast<int>(std::clamp(settings.environmentPreset, 0u, 3u));
                if (ImGui::Combo("Env Preset", &environmentPreset, environmentPresets, IM_ARRAYSIZE(environmentPresets))) {
                    settings.environmentPreset = static_cast<uint32_t>(environmentPreset);
                    _renderer.ResetAccumulation();
                }
                if (ImGui::SliderFloat("Env Intensity", &settings.environmentIntensity, 0.0f, 4.0f, "%.2f")) {
                    _renderer.ResetAccumulation();
                }
                if (ImGui::SliderFloat("Env Rotation", &settings.environmentRotationDegrees, 0.0f, 360.0f, "%.1f deg")) {
                    _renderer.ResetAccumulation();
                }
                if (ImGui::SliderFloat("IBL Diffuse", &settings.environmentDiffuseStrength, 0.0f, 2.0f, "%.2f")) {
                    _renderer.ResetAccumulation();
                }
                if (ImGui::SliderFloat("IBL Specular", &settings.environmentSpecularStrength, 0.0f, 2.0f, "%.2f")) {
                    _renderer.ResetAccumulation();
                }
                ImGui::Text("HDRI: %s", settings.externalHdriAvailable ? settings.externalHdriStatus.c_str() : "Procedural only");
                if (ImGui::MenuItem("Select For Drag")) {
                    _renderer.SelectDirectionalLight();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Debug")) {
            const auto panelMenuItem = [&](const char* label, bool& visible) {
                if (ImGui::MenuItem(label, nullptr, &visible) && visible) {
                    _showDebugUi = true;
                }
            };
            panelMenuItem("Frame / Engine Overview", _showFrameOverview);
            panelMenuItem("Render Graph", _showRenderGraphPanel);
            panelMenuItem("GPU Profiler", _showGpuProfilerPanel);
            panelMenuItem("Debug Visualization", _showDebugVisualizationPanel);
            panelMenuItem("Render Mode Control", _showRenderModeControlPanel);
            panelMenuItem("Scene Inspector", _showSceneInspectorPanel);
            panelMenuItem("Resource Inspector", _showResourceInspectorPanel);
            panelMenuItem("Log Console", _showLogConsolePanel);
            ImGui::Separator();
            panelMenuItem("Legacy Stats", _showLegacyStatsPanel);
            panelMenuItem("Legacy Render Controls", _showLegacyRenderPanel);
            panelMenuItem("Legacy Camera Controls", _showLegacyCameraPanel);
            ImGui::Separator();
            if (ImGui::MenuItem("Reset Accumulation")) {
                _renderer.ResetAccumulation();
                log_startup_event("Path tracing accumulation reset");
            }
            if (ImGui::MenuItem("Shader Hot Reload")) {
                const bool reloaded = _renderer.ReloadShaders();
                log_startup_event(reloaded ? "Shader hot reload complete" : "Shader hot reload failed: " + _renderer.GetLastShaderReloadMessage());
            }
            if (ImGui::MenuItem("Capture Frame")) {
                const std::filesystem::path path = MakeTimestampedCapturePath("frame", ".png");
                log_startup_event(request_screenshot_with_metadata(path, "menu_frame_capture") ? "Frame capture queued: " + path.string()
                                                                                                : "Frame capture failed");
            }
            ImGui::EndMenu();
        }

        const auto& sceneLoadStatus = _renderer.GetSceneLoadStatus();
        const std::string& sceneStatus = sceneLoadStatus.message;
        if (sceneLoadInProgress) {
            const float progress = std::clamp(sceneLoadStatus.progress, 0.0f, 1.0f);
            const std::string overlay = fmt::format("{:.0f}%", progress * 100.0f);
            const std::string sceneName = sceneLoadStatus.path.empty() ? std::string("scene") : sceneLoadStatus.path.filename().string();
            ImGui::Separator();
            ImGui::TextDisabled("Loading %s", sceneName.c_str());
            ImGui::SameLine();
            ImGui::ProgressBar(progress, ImVec2(150.0f, 0.0f), overlay.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton(sceneLoadStatus.cancelRequested ? "Cancelling" : "Cancel")) {
                _renderer.CancelSceneLoad();
            }
        } else if (!sceneStatus.empty()) {
            ImGui::Separator();
            ImGui::TextDisabled("%s", sceneStatus.c_str());
        }
        if (!sceneLoadInProgress && PathTracePassVisible(settings.displayMode)) {
            ImGui::Separator();
            ImGui::TextDisabled("PT");
            ImGui::SameLine();
            DrawPathTraceProgressBar(settings, _renderer.GetPathTraceFrameIndex(), ImVec2(150.0f, 0.0f));
        }

        ImGui::EndMainMenuBar();
    }
}

void VestaEngine::build_debug_ui()
{
    if (!_imguiInitialized || (!_showDebugUi && !has_debug_window_open())) {
        return;
    }

    ImGui::SetCurrentContext(_imguiContext);

    auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();
    auto& camera = _renderer.GetCamera();
    const auto& device = _renderer.GetRenderDevice();
    const float frameMs = _renderer.GetSmoothedFrameTimeMs();
    const float fps = frameMs > 0.0f ? 1000.0f / frameMs : 0.0f;
    const std::string sceneLabel = scene.GetSourcePath().empty() ? "No scene" : ScenePathLabel(scene.GetSourcePath());
    const auto& sceneLoadStatus = _renderer.GetSceneLoadStatus();
    const char* activeBackend = "Compute";
    switch (_renderer.GetActivePathTraceBackend()) {
    case vesta::render::PathTraceBackend::Auto:
        activeBackend = "Auto";
        break;
    case vesta::render::PathTraceBackend::HardwareRT:
        activeBackend = "Hardware RT";
        break;
    case vesta::render::PathTraceBackend::Compute:
    default:
        activeBackend = "Compute";
        break;
    }
    const std::string selectionLabel = _renderer.GetSelectionLabel();
    const auto& frameHistory = _renderer.GetFrameTimeHistoryMs();
    const size_t frameHistoryCount = _renderer.GetFrameTimeHistoryCount();
    const FrameTimingStats frameStats = CalculateFrameTimingStats(frameHistory, frameHistoryCount);
    const auto& graphTimings = _renderer.GetLastRenderGraphTimings();
    const float gpuFrameMs = TotalGpuMs(graphTimings);
    const VkExtent2D swapchainExtent = device.GetSwapchainExtent();

    if (_showFrameOverview) {
        ImGui::SetNextWindowPos(ImVec2(18.0f, 18.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(640.0f, 0.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Frame / Engine Overview", &_showFrameOverview, ImGuiWindowFlags_NoSavedSettings)) {
            if (ImGui::BeginTable("OverviewGrid", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Metric");
                ImGui::TableSetupColumn("Value");
                ImGui::TableSetupColumn("Metric");
                ImGui::TableSetupColumn("Value");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("FPS / Frame");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.1f / %.2f ms", fps, frameMs);
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted("Avg / Min / Max");
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%.2f / %.2f / %.2f ms", frameStats.averageMs, frameStats.minMs, frameStats.maxMs);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("CPU / GPU Frame");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.2f / %.2f ms", _renderer.GetFrameTimeMs(), gpuFrameMs);
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted("1%% Low");
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%.1f FPS", frameStats.onePercentLowFps);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Resolution / PT Scale");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%ux%u / %.2fx", swapchainExtent.width, swapchainExtent.height, settings.pathTraceResolutionScale);
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted("Current Render Pass");
                ImGui::TableSetColumnIndex(3);
                ImGui::TextUnformatted(DisplayModeLabel(settings.displayMode));

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("GPU / API");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("Vulkan / %s", device.GetGpuName().c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted("Frame Index");
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%d / PT %u", _frameNumber, _renderer.GetPathTraceFrameIndex());

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Environment");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.2fx / %.1f deg / Em %.2f",
                    settings.environmentIntensity,
                    settings.environmentRotationDegrees,
                    settings.emissiveIntensity);
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted("SSAO");
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%s / r %.2f / i %.2f", settings.enableSsao ? "On" : "Off", settings.ssaoRadius, settings.ssaoIntensity);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("Temporal AA");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%s / %.2f", settings.enableTaa ? "On" : "Off", settings.taaFeedback);
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted("SSR");
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%s / %.1f", settings.enableSsr ? "On" : "Off", settings.ssrMaxDistance);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("SSGI");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%s / r %.2f / %u", settings.enableSsgi ? "On" : "Off", settings.ssgiRadius, settings.ssgiSampleCount);
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted("Camera Look");
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%.2f EV / A %.3f / F %.2f",
                    settings.cameraExposureEv,
                    settings.cameraApertureRadius,
                    settings.cameraFocalDistance);
                ImGui::EndTable();
            }

            if (PathTracePassVisible(settings.displayMode)) {
                ImGui::SeparatorText("Path Tracing Progress");
                DrawPathTraceProgressBar(settings, _renderer.GetPathTraceFrameIndex(), ImVec2(-FLT_MIN, 0.0f));
                ImGui::Text("Backend %s  AOV %s  SPP %u  Bounces %u",
                    activeBackend,
                    PathTraceDebugViewLabel(settings.pathTraceDebugView),
                    settings.pathTraceSamplesPerPixel,
                    settings.pathTraceMaxBounces);
            }

            bool vsyncEnabled = settings.enableVSync;
            if (ImGui::Checkbox("VSync", &vsyncEnabled)) {
                _renderer.SetVSyncEnabled(vsyncEnabled);
                log_startup_event(fmt::format("VSync {} ({})",
                    vsyncEnabled ? "enabled" : "disabled",
                    PresentModeLabel(_renderer.GetRenderDevice().GetPresentMode())));
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Current present mode: %s", PresentModeLabel(_renderer.GetRenderDevice().GetPresentMode()));
            }
            ImGui::SameLine();
            ImGui::Checkbox("FPS Limit", &settings.enableFpsLimit);
            ImGui::SameLine();
            int fpsLimit = static_cast<int>(settings.fpsLimit);
            ImGui::SetNextItemWidth(110.0f);
            if (ImGui::InputInt("Limit", &fpsLimit, 5, 30)) {
                settings.fpsLimit = static_cast<uint32_t>(std::clamp(fpsLimit, 15, 360));
            }
            ImGui::SameLine();
            if (ImGui::Button("Capture")) {
                const std::filesystem::path path = MakeTimestampedCapturePath("frame", ".png");
                log_startup_event(request_screenshot_with_metadata(path, "frame_overview_capture") ? "Frame capture queued: " + path.string()
                                                                                                    : "Frame capture failed");
            }
            ImGui::SameLine();
            if (ImGui::Button("Screenshot")) {
                const std::filesystem::path path = MakeTimestampedCapturePath("screenshot", ".png");
                log_startup_event(request_screenshot_with_metadata(path, "frame_overview_screenshot") ? "Screenshot queued: " + path.string()
                                                                                                      : "Screenshot failed");
            }
            ImGui::SameLine();
            if (ImGui::Button("Shader Reload")) {
                const bool reloaded = _renderer.ReloadShaders();
                log_startup_event(reloaded ? "Shader hot reload complete" : "Shader hot reload failed: " + _renderer.GetLastShaderReloadMessage());
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset Accumulation")) {
                _renderer.ResetAccumulation();
                log_startup_event("Path tracing accumulation reset");
            }
        }
        ImGui::End();
    }

    if (_showRenderGraphPanel) {
        ImGui::SetNextWindowPos(ImVec2(18.0f, 238.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(560.0f, 360.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Render Pass / Render Graph", &_showRenderGraphPanel, ImGuiWindowFlags_NoSavedSettings)) {
            const char* displayModes[] = { "Hybrid: Raster + Gaussian + Path", "Rasterizer", "Gaussian Splatting", "Path Tracing", "Ray Tracing" };
            int displayMode = static_cast<int>(settings.displayMode);
            if (ImGui::Combo("Render Mode", &displayMode, displayModes, IM_ARRAYSIZE(displayModes))) {
                vesta::render::ApplyDisplayModePassSelection(settings, static_cast<vesta::render::RendererDisplayMode>(displayMode));
                _renderer.ResetAccumulation();
            }
            ImGui::SameLine();
            if (ImGui::Button("Export DOT")) {
                const std::filesystem::path path = MakeTimestampedCapturePath("render_graph", ".dot");
                log_startup_event(WriteRenderGraphDot(path, graphTimings) ? "Render graph DOT exported: " + path.string()
                                                                           : "Render graph DOT export failed");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Writes the current pass/resource dependency graph as a GraphViz DOT file.");
            }

            const std::vector<vesta::render::RenderPassDebugInfo> passInfo = _renderer.GetRenderPassDebugInfo();
            if (ImGui::BeginTable("PassRegistry", 8, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Pass");
                ImGui::TableSetupColumn("Order", ImGuiTableColumnFlags_WidthFixed, 52.0f);
                ImGui::TableSetupColumn("Move", ImGuiTableColumnFlags_WidthFixed, 74.0f);
                ImGui::TableSetupColumn("Enabled", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                ImGui::TableSetupColumn("Draw", ImGuiTableColumnFlags_WidthFixed, 54.0f);
                ImGui::TableSetupColumn("Dispatch", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                ImGui::TableSetupColumn("Work");
                ImGui::TableSetupColumn("Id");
                ImGui::TableHeadersRow();
                for (size_t passIndex = 0; passIndex < passInfo.size(); ++passIndex) {
                    const auto& pass = passInfo[passIndex];
                    bool enabled = pass.enabled;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(pass.name.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%u", pass.order);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::PushID(pass.id.c_str());
                    ImGui::BeginDisabled(passIndex == 0);
                    if (ImGui::SmallButton("Up")) {
                        const auto& previousPass = passInfo[passIndex - 1];
                        _renderer.SetPassOrder(pass.id, previousPass.order);
                        _renderer.SetPassOrder(previousPass.id, pass.order);
                        _renderer.ResetAccumulation();
                    }
                    ImGui::EndDisabled();
                    ImGui::SameLine();
                    ImGui::BeginDisabled(passIndex + 1 >= passInfo.size());
                    if (ImGui::SmallButton("Dn")) {
                        const auto& nextPass = passInfo[passIndex + 1];
                        _renderer.SetPassOrder(pass.id, nextPass.order);
                        _renderer.SetPassOrder(nextPass.id, pass.order);
                        _renderer.ResetAccumulation();
                    }
                    ImGui::EndDisabled();
                    ImGui::TableSetColumnIndex(3);
                    if (ImGui::Checkbox("##enabled", &enabled)) {
                        _renderer.SetPassEnabled(pass.id, enabled);
                        _renderer.ResetAccumulation();
                    }
                    ImGui::PopID();
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%u", pass.drawCount);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%u", pass.dispatchCount);
                    ImGui::TableSetColumnIndex(6);
                    if (pass.splatCount > 0u) {
                        ImGui::Text("%llu splats", static_cast<unsigned long long>(pass.splatCount));
                    } else if (pass.rayCount > 0u) {
                        ImGui::Text("%llu rays", static_cast<unsigned long long>(pass.rayCount));
                        if (pass.primaryRayCount > 0u && ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("primary %llu\nshadow %llu\ndiffuse %llu\nspecular %llu",
                                static_cast<unsigned long long>(pass.primaryRayCount),
                                static_cast<unsigned long long>(pass.shadowRayCount),
                                static_cast<unsigned long long>(pass.diffuseRayCount),
                                static_cast<unsigned long long>(pass.specularRayCount));
                        }
                    } else if (pass.triangleCount > 0u) {
                        ImGui::Text("%llu tris", static_cast<unsigned long long>(pass.triangleCount));
                    } else {
                        ImGui::TextUnformatted("-");
                    }
                    ImGui::TableSetColumnIndex(7);
                    ImGui::TextUnformatted(pass.id.c_str());
                }
                ImGui::EndTable();
            }

            if (ImGui::CollapsingHeader("Resource Dependency Edges", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::BeginTable("RenderGraphEdges", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                        ImVec2(0.0f, 160.0f))) {
                    ImGui::TableSetupColumn("Writer");
                    ImGui::TableSetupColumn("Resource");
                    ImGui::TableSetupColumn("Reader");
                    ImGui::TableSetupColumn("Usage", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                    ImGui::TableSetupColumn("Sync", ImGuiTableColumnFlags_WidthFixed, 74.0f);
                    ImGui::TableHeadersRow();
                    uint32_t edgeCount = 0;
                    for (size_t readerIndex = 0; readerIndex < graphTimings.size(); ++readerIndex) {
                        const auto& reader = graphTimings[readerIndex];
                        for (const auto& input : reader.inputs) {
                            std::optional<size_t> writerIndex;
                            for (size_t candidateIndex = 0; candidateIndex < readerIndex; ++candidateIndex) {
                                const auto& candidate = graphTimings[candidateIndex];
                                const bool writesResource = std::any_of(candidate.outputs.begin(), candidate.outputs.end(), [&](const auto& output) {
                                    return output.name == input.name;
                                });
                                if (writesResource) {
                                    writerIndex = candidateIndex;
                                }
                            }
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            if (writerIndex.has_value()) {
                                ImGui::TextUnformatted(graphTimings[*writerIndex].name.c_str());
                            } else {
                                ImGui::TextDisabled(input.imported ? "Imported" : "External");
                            }
                            ImGui::TableSetColumnIndex(1);
                            ImGui::TextUnformatted(input.name.c_str());
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TextUnformatted(reader.name.c_str());
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TextUnformatted(ResourceUsageLabel(input.usage));
                            ImGui::TableSetColumnIndex(4);
                            ImGui::TextUnformatted(writerIndex.has_value() ? "barrier" : "read");
                            ++edgeCount;
                        }
                    }
                    if (edgeCount == 0u) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::TextDisabled("No resource edges recorded yet.");
                    }
                    ImGui::EndTable();
                }
            }

            ImGui::SeparatorText("Current Frame Resources");
            for (const auto& timing : graphTimings) {
                if (ImGui::TreeNode(timing.name.c_str())) {
                    if (timing.gpuTimingValid) {
                        ImGui::Text("CPU %.3f ms, GPU %.3f ms, Render %ux%u, Barriers %u",
                            timing.cpuMs,
                            timing.gpuMs,
                            timing.renderExtent.width,
                            timing.renderExtent.height,
                            timing.barrierCount);
                    } else {
                        ImGui::Text("CPU %.3f ms, GPU -, Render %ux%u, Barriers %u",
                            timing.cpuMs,
                            timing.renderExtent.width,
                            timing.renderExtent.height,
                            timing.barrierCount);
                    }
                    DrawRenderGraphBarrierList(timing.barriers);
                    DrawRenderGraphResourceList("Inputs", timing.inputs);
                    DrawRenderGraphResourceList("Outputs", timing.outputs);
                    ImGui::TreePop();
                }
            }
        }
        ImGui::End();
    }

    if (_showGpuProfilerPanel) {
        ImGui::SetNextWindowPos(ImVec2(590.0f, 238.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(520.0f, 300.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("GPU Profiler", &_showGpuProfilerPanel, ImGuiWindowFlags_NoSavedSettings)) {
            const std::vector<vesta::render::RenderPassDebugInfo> passInfo = _renderer.GetRenderPassDebugInfo();
            uint32_t totalDraws = 0;
            uint32_t totalDispatches = 0;
            uint64_t totalRayWork = 0;
            uint64_t totalPrimaryRays = 0;
            uint64_t totalShadowRays = 0;
            uint64_t totalDiffuseRays = 0;
            uint64_t totalSpecularRays = 0;
            for (const auto& pass : passInfo) {
                if (!pass.enabled) {
                    continue;
                }
                totalDraws += pass.drawCount;
                totalDispatches += pass.dispatchCount;
                totalRayWork += pass.rayCount;
                totalPrimaryRays += pass.primaryRayCount;
                totalShadowRays += pass.shadowRayCount;
                totalDiffuseRays += pass.diffuseRayCount;
                totalSpecularRays += pass.specularRayCount;
            }
            ImGui::Text("CPU Frame %.3f ms", _renderer.GetFrameTimeMs());
            ImGui::Text("GPU Frame %.3f ms", gpuFrameMs);
            ImGui::Text("Draw / Dispatch %u / %u", totalDraws, totalDispatches);
            ImGui::Text("Triangles %zu", scene.GetTriangles().size());
            ImGui::Text("Estimated Ray Work %llu", static_cast<unsigned long long>(totalRayWork));
            if (totalPrimaryRays > 0u) {
                ImGui::Text("Ray Types P/S/D/Spec %llu / %llu / %llu / %llu",
                    static_cast<unsigned long long>(totalPrimaryRays),
                    static_cast<unsigned long long>(totalShadowRays),
                    static_cast<unsigned long long>(totalDiffuseRays),
                    static_cast<unsigned long long>(totalSpecularRays));
            }
            ImGui::Text("Visible Surfaces %u / %zu", _renderer.GetVisibleSurfaceCount(), scene.GetSurfaces().size());
            const auto meshletStats = _renderer.GetMeshletClusterStats();
            ImGui::Text("Meshlets %u visible / %u total", meshletStats.visibleMeshlets, meshletStats.totalMeshlets);
            ImGui::Text("Meshlet Visibility Storage %s  %.3f MiB",
                meshletStats.visibilityStorageAvailable ? "ready" : "staged",
                MiB(meshletStats.estimatedVisibilityBytes));
            const uint32_t totalGaussians = scene.GetGaussianCount();
            const uint32_t projectedGaussians = _renderer.GetOfficialGaussianProjectedCount();
            const uint32_t culledGaussians = projectedGaussians <= totalGaussians ? totalGaussians - projectedGaussians : 0u;
            ImGui::Text("Gaussians %u projected / %u total", projectedGaussians, totalGaussians);
            ImGui::Text("Culled Gaussians %u", culledGaussians);
            ImGui::Text("Splats rendered %u", _renderer.GetOfficialGaussianDuplicateCount());
            ImGui::Text("VRAM Dedicated %u MiB", device.GetDedicatedVideoMemoryMiB());
            const auto bindlessStats = device.GetBindlessStats();
            ImGui::Text("Bindless Srv/Cube/StorageImg/StorageBuf %u/%u  %u/%u  %u/%u  %u/%u",
                bindlessStats.sampledImagesUsed,
                bindlessStats.sampledImagesCapacity,
                bindlessStats.sampledCubeImagesUsed,
                bindlessStats.sampledCubeImagesCapacity,
                bindlessStats.storageImagesUsed,
                bindlessStats.storageImagesCapacity,
                bindlessStats.storageBuffersUsed,
                bindlessStats.storageBuffersCapacity);
            if (ImGui::BeginTable("GpuPassTiming", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Pass");
                ImGui::TableSetupColumn("CPU ms", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                ImGui::TableSetupColumn("GPU ms", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                ImGui::TableSetupColumn("Sync", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                ImGui::TableHeadersRow();
                for (const auto& timing : graphTimings) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(timing.name.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.3f", timing.cpuMs);
                    ImGui::TableSetColumnIndex(2);
                    timing.gpuTimingValid ? ImGui::Text("%.3f", timing.gpuMs) : ImGui::TextUnformatted("-");
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%u barriers", timing.barrierCount);
                }
                ImGui::EndTable();
            }
            if (ImGui::BeginTable("GpuPassWork", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Pass");
                ImGui::TableSetupColumn("Draw", ImGuiTableColumnFlags_WidthFixed, 54.0f);
                ImGui::TableSetupColumn("Dispatch", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                ImGui::TableSetupColumn("Triangles", ImGuiTableColumnFlags_WidthFixed, 82.0f);
                ImGui::TableSetupColumn("Rays / Splats");
                ImGui::TableHeadersRow();
                for (const auto& pass : passInfo) {
                    if (!pass.enabled) {
                        continue;
                    }
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(pass.name.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%u", pass.drawCount);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%u", pass.dispatchCount);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%llu", static_cast<unsigned long long>(pass.triangleCount));
                    ImGui::TableSetColumnIndex(4);
                    if (pass.splatCount > 0u) {
                        ImGui::Text("%llu splats", static_cast<unsigned long long>(pass.splatCount));
                    } else if (pass.rayCount > 0u) {
                        ImGui::Text("%llu rays", static_cast<unsigned long long>(pass.rayCount));
                        if (pass.primaryRayCount > 0u && ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("primary %llu\nshadow %llu\ndiffuse %llu\nspecular %llu",
                                static_cast<unsigned long long>(pass.primaryRayCount),
                                static_cast<unsigned long long>(pass.shadowRayCount),
                                static_cast<unsigned long long>(pass.diffuseRayCount),
                                static_cast<unsigned long long>(pass.specularRayCount));
                        }
                    } else {
                        ImGui::TextUnformatted("-");
                    }
                }
                ImGui::EndTable();
            }
            if (frameHistoryCount > 0) {
                ImGui::PlotLines("CPU Frame History", frameHistory.data(), static_cast<int>(frameHistoryCount), 0, nullptr, 0.0f,
                    std::max(33.0f, frameStats.maxMs * 1.1f), ImVec2(0.0f, 72.0f));
            }
            if (_gpuFrameTimeHistoryCount > 0) {
                const auto gpuHistoryEnd = _gpuFrameTimeHistoryMs.begin() + static_cast<std::ptrdiff_t>(_gpuFrameTimeHistoryCount);
                const float maxGpuHistoryMs = *std::max_element(_gpuFrameTimeHistoryMs.begin(), gpuHistoryEnd);
                ImGui::PlotLines("GPU Timestamp History",
                    _gpuFrameTimeHistoryMs.data(),
                    static_cast<int>(_gpuFrameTimeHistoryCount),
                    0,
                    nullptr,
                    0.0f,
                    std::max(33.0f, maxGpuHistoryMs * 1.1f),
                    ImVec2(0.0f, 72.0f));
            }
        }
        ImGui::End();
    }

    if (_showDebugVisualizationPanel) {
        ImGui::SetNextWindowPos(ImVec2(1124.0f, 238.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360.0f, 300.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Debug Visualization", &_showDebugVisualizationPanel, ImGuiWindowFlags_NoSavedSettings)) {
            const char* commonViews[] = {
                "Final Color",
                "Albedo / Base Color",
                "Normal",
                "World Position",
                "Linear Depth",
                "UV",
                "Material ID",
                "Object ID",
                "Roughness",
                "Metallic",
                "Emissive",
                "Ambient Occlusion",
                "Motion Vector",
                "Direct Lighting",
                "Indirect Lighting",
                "Reflection",
                "Denoised Result",
                "Difference from Reference",
                "Wireframe",
                "Mip Level",
                "Shadow Map",
                "Overdraw",
                "Temporal History Color",
                "Temporal History Depth",
                "Temporal Reprojection",
                "Temporal Disocclusion",
                "Temporal Jitter",
                "Contact Shadow",
                "Shadow Cascade",
                "Ray-Traced GI",
            };
            int commonView = static_cast<int>(settings.debugView);
            if (ImGui::Combo("Debug View", &commonView, commonViews, IM_ARRAYSIZE(commonViews))) {
                vesta::render::SelectRendererDebugView(settings, static_cast<vesta::render::RendererDebugView>(commonView));
                _renderer.ResetAccumulation();
            }
            ImGui::TextDisabled("Raster GBuffer views are live when the raster pass is active.");

            const char* pathTraceDebugViews[] = { "Final", "Albedo", "Normal", "Depth", "Direct", "Indirect", "Ray Count Heatmap", "Diffuse Bounce", "Specular Bounce", "Throughput", "PDF" };
            int pathTraceDebugView = static_cast<int>(settings.pathTraceDebugView);
            if (ImGui::Combo("Path Tracing AOV", &pathTraceDebugView, pathTraceDebugViews, IM_ARRAYSIZE(pathTraceDebugViews))) {
                vesta::render::SelectPathTraceDebugView(settings, static_cast<vesta::render::PathTraceDebugView>(pathTraceDebugView));
                _renderer.ResetAccumulation();
            }
            const char* gaussianViews[] = {
                "Final Splat Image",
                "Splat Alpha",
                "Revealage",
                "Overdraw Heatmap",
                "Splat Depth",
                "Tile Occupancy",
                "Splat Radius",
                "Contribution Count",
                "Splat ID",
                "SH Band",
                "Covariance",
                "Raster Depth",
                "Composition Mask",
                "Depth Difference",
            };
            int gaussianView = static_cast<int>(settings.gaussianDebugView);
            if (ImGui::Combo("Gaussian Debug View", &gaussianView, gaussianViews, IM_ARRAYSIZE(gaussianViews))) {
                settings.gaussianDebugView = static_cast<vesta::render::GaussianDebugView>(gaussianView);
                _renderer.ResetAccumulation();
            }
            bool wireframeView = settings.debugView == vesta::render::RendererDebugView::Wireframe;
            if (ImGui::Checkbox("Wireframe", &wireframeView)) {
                vesta::render::SelectRendererDebugView(settings,
                    wireframeView ? vesta::render::RendererDebugView::Wireframe : vesta::render::RendererDebugView::FinalColor);
                _renderer.ResetAccumulation();
            }
            ImGui::SameLine();
            bool overdrawView = settings.debugView == vesta::render::RendererDebugView::Overdraw;
            if (ImGui::Checkbox("Overdraw", &overdrawView)) {
                vesta::render::SelectRendererDebugView(settings,
                    overdrawView ? vesta::render::RendererDebugView::Overdraw : vesta::render::RendererDebugView::FinalColor);
                _renderer.ResetAccumulation();
            }
            ImGui::TextDisabled("Debug view shortcuts are mirrored in View and Options.");
            ImGui::SeparatorText("Reference Compare");
            const char* compareModes[] = { "Off", "Raster / Path Split", "Difference Heatmap" };
            int compareMode = static_cast<int>(settings.compareMode);
            if (ImGui::Combo("Compare Mode", &compareMode, compareModes, IM_ARRAYSIZE(compareModes))) {
                settings.compareMode = static_cast<vesta::render::CompareMode>(compareMode);
                settings.displayMode = vesta::render::RendererDisplayMode::Composite;
                _renderer.ResetAccumulation();
            }
            if (settings.compareMode == vesta::render::CompareMode::RasterPathSplit) {
                ImGui::SliderFloat("Split Position", &settings.compareSplitPosition, 0.05f, 0.95f, "%.2f");
            }
            if (settings.compareMode == vesta::render::CompareMode::DifferenceHeatmap) {
                ImGui::SliderFloat("Difference Scale", &settings.compareDifferenceScale, 0.5f, 12.0f, "%.1f");
            }
        }
        ImGui::End();
    }

    build_render_mode_control_panel();

    if (_showLegacyStatsPanel) {
        ImGui::SetNextWindowPos(ImVec2(18.0f, 18.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Stats", &_showLegacyStatsPanel, ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::Text("Frame %.2f ms", frameMs);
        ImGui::Text("FPS %.1f", fps);
        ImGui::Text("%s", device.GetGpuName().c_str());
        ImGui::Text("VRAM %u MiB", device.GetDedicatedVideoMemoryMiB());
        ImGui::SeparatorText("Scene");
        ImGui::TextWrapped("%s", sceneLabel.c_str());
        ImGui::Text("Type %s", SceneKindLabel(scene.GetSceneKind()));
        ImGui::Text("Display %s", DisplayModeLabel(settings.displayMode));
        ImGui::Text("Compare %s", CompareModeLabel(settings.compareMode));
        ImGui::Text("Load %s", SceneLoadStateLabel(sceneLoadStatus.state));
        if (_renderer.IsSceneLoadInProgress()) {
            const float progress = std::clamp(sceneLoadStatus.progress, 0.0f, 1.0f);
            ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0.0f));
            if (!sceneLoadStatus.message.empty()) {
                ImGui::TextWrapped("%s", sceneLoadStatus.message.c_str());
            }
            if (ImGui::Button("Cancel Scene Load")) {
                _renderer.CancelSceneLoad();
            }
            if (sceneLoadStatus.totalUploadBytes > 0u) {
                ImGui::Text("Upload %.1f / %.1f MiB",
                    static_cast<double>(sceneLoadStatus.completedUploadBytes) / (1024.0 * 1024.0),
                    static_cast<double>(sceneLoadStatus.totalUploadBytes) / (1024.0 * 1024.0));
            }
            if (sceneLoadStatus.totalTextures > 0u) {
                ImGui::Text("Textures %u / %u", sceneLoadStatus.uploadedTextures, sceneLoadStatus.totalTextures);
            }
        } else if (!sceneLoadStatus.message.empty()) {
            ImGui::TextWrapped("%s", sceneLoadStatus.message.c_str());
        }
        ImGui::Text("Selected %s", selectionLabel.c_str());
        if (scene.GetGaussianCount() > 0u) {
            ImGui::Text("Gaussians %u", scene.GetGaussianCount());
        }
        if (!scene.GetTriangles().empty()) {
            ImGui::Text("Triangles %zu", scene.GetTriangles().size());
        }
        ImGui::Text("Objects %zu", scene.GetObjects().size());
        if (!sceneLoadStatus.message.empty()) {
            ImGui::TextWrapped("%s", sceneLoadStatus.message.c_str());
        }

        ImGui::Checkbox("Detailed Info", &_showDetailedStats);
        if (_showDetailedStats) {
            ImGui::SeparatorText("Runtime");
            ImGui::Text("Recommended %s", PresetLabel(_renderer.GetRecommendedPreset()));
            ImGui::Text("Scene Upload %s", SceneUploadModeLabel(settings.sceneUploadMode));
            ImGui::Text("Skip Hidden Passes %s", settings.optimizeInactivePasses ? "On" : "Off");
            ImGui::Text("Device Local Buffers %s", settings.useDeviceLocalSceneBuffers ? "On" : "Off");
            ImGui::Text("Device Local Textures %s", settings.useDeviceLocalTextures ? "On" : "Off");
            ImGui::Text("Deferred Scene Free %s", settings.deferOldSceneDestruction ? "On" : "Off");
            ImGui::Text("Frustum Culling %s", settings.enableFrustumCulling ? "On" : "Off");
            ImGui::Text("Distance Culling %s", settings.enableDistanceCulling ? "On" : "Off");
            ImGui::Text("Indirect Draw %s", settings.useIndirectDraw ? "On" : "Off");
            ImGui::Text("VSync %s / %s", settings.enableVSync ? "On" : "Off", PresentModeLabel(device.GetPresentMode()));
            ImGui::Text("FPS Limit %s / %u", settings.enableFpsLimit ? "On" : "Off", settings.fpsLimit);
            ImGui::Text("Upload Budget %u MiB", settings.maxUploadBytesPerFrame / (1024u * 1024u));
            ImGui::Text("Texture Budget %u MiB", settings.maxTextureUploadBytesPerFrame / (1024u * 1024u));
            ImGui::Text("Upload Pending %.2f MiB",
                static_cast<float>(device.GetUploadBatchStats().pendingBytes) / (1024.0f * 1024.0f));
            ImGui::Text("Upload Staging %.2f MiB",
                static_cast<float>(device.GetUploadBatchStats().stagingCapacity) / (1024.0f * 1024.0f));
            ImGui::Text("Transfer Queue %s", device.HasTransferQueue() ? "Active" : "Graphics Fallback");
            ImGui::Text("Workers %u", _renderer.GetWorkerThreadCount());
            ImGui::Text("Queued Jobs %zu", _renderer.GetPendingJobCount());
            ImGui::Text("Retired Scenes %zu", _renderer.GetRetiredSceneCount());

            const auto& graphTimings = _renderer.GetLastRenderGraphTimings();
            if (!graphTimings.empty()) {
                ImGui::SeparatorText("Render Graph / Profiler");
                if (ImGui::BeginTable("RenderGraphProfiler", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Pass");
                    ImGui::TableSetupColumn("CPU ms", ImGuiTableColumnFlags_WidthFixed, 68.0f);
                    ImGui::TableSetupColumn("GPU ms", ImGuiTableColumnFlags_WidthFixed, 68.0f);
                    ImGui::TableSetupColumn("R", ImGuiTableColumnFlags_WidthFixed, 34.0f);
                    ImGui::TableSetupColumn("W", ImGuiTableColumnFlags_WidthFixed, 34.0f);
                    ImGui::TableSetupColumn("Barriers", ImGuiTableColumnFlags_WidthFixed, 58.0f);
                    ImGui::TableHeadersRow();
                    for (const auto& timing : graphTimings) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::TextUnformatted(timing.name.c_str());
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%.3f", timing.cpuMs);
                        ImGui::TableSetColumnIndex(2);
                        if (timing.gpuTimingValid) {
                            ImGui::Text("%.3f", timing.gpuMs);
                        } else {
                            ImGui::TextUnformatted("-");
                        }
                        ImGui::TableSetColumnIndex(3);
                        ImGui::Text("%u", timing.readCount);
                        ImGui::TableSetColumnIndex(4);
                        ImGui::Text("%u", timing.writeCount);
                        ImGui::TableSetColumnIndex(5);
                        ImGui::Text("%u", timing.barrierCount);
                    }
                    ImGui::EndTable();
                }
            }

            ImGui::SeparatorText("Scene Details");
            ImGui::Text("Recommended View %s", DisplayModeLabel(_renderer.GetRecommendedDisplayModeForScene()));
            ImGui::Text("Vertices %zu", scene.GetVertices().size());
            ImGui::Text("Gaussian SH Degree %u", scene.GetGaussianShDegree());
            ImGui::Text("Gaussian AA %s", settings.gaussianAntialiasing ? "On" : "Off");
            ImGui::Text("Gaussian Fast Culling %s", settings.gaussianFastCulling ? "On" : "Off");
            if (scene.HasTrainedGaussians()) {
                const uint32_t totalGaussians = scene.GetGaussianCount();
                const uint32_t projectedGaussians = _renderer.GetOfficialGaussianProjectedCount();
                const uint32_t culledGaussians = projectedGaussians <= totalGaussians ? totalGaussians - projectedGaussians : 0u;
                const float cullRatio = totalGaussians > 0u ? static_cast<float>(culledGaussians) / static_cast<float>(totalGaussians) : 0.0f;
                ImGui::Text("Projected %u / %u", projectedGaussians, totalGaussians);
                ImGui::Text("Culled %u (%.1f%%)", culledGaussians, cullRatio * 100.0f);
                ImGui::Text("Duplicates %u", _renderer.GetOfficialGaussianDuplicateCount());
                ImGui::Text("Padded Duplicates %u", _renderer.GetOfficialGaussianPaddedDuplicateCount());
                ImGui::Text("Tiles %u", _renderer.GetOfficialGaussianTileCount());
                ImGui::Text("Avg Tiles / Gaussian %.2f", _renderer.GetOfficialGaussianAverageTilesTouched());
                ImGui::Text("Gaussian Rebuilds %llu", static_cast<unsigned long long>(_renderer.GetOfficialGaussianRebuildCount()));
                ImGui::Text("Gaussian Preview %s", _renderer.IsGaussianInteractivePreviewActive() ? "Legacy Preview" : "Official");
                ImGui::Text("Preprocess %.3f ms", _renderer.GetOfficialGaussianPreprocessMs());
                ImGui::Text("Scan %.3f ms", _renderer.GetOfficialGaussianScanMs());
                ImGui::Text("Duplicate %.3f ms", _renderer.GetOfficialGaussianDuplicateMs());
                ImGui::Text("Sort %.3f ms", _renderer.GetOfficialGaussianSortMs());
                ImGui::Text("Range %.3f ms", _renderer.GetOfficialGaussianRangeMs());
                ImGui::Text("Raster %.3f ms", _renderer.GetOfficialGaussianRasterMs());
                ImGui::Text("Build Total %.3f ms", _renderer.GetOfficialGaussianTotalBuildMs());
            }
            ImGui::Text("Surfaces %zu", scene.GetSurfaces().size());
            ImGui::Text("Textures %zu / %u", scene.GetTextures().size(), _renderer.GetResidentTextureCount());
            ImGui::Text("Visible Surfaces %u", _renderer.GetVisibleSurfaceCount());
            if (!sceneLoadStatus.uploadStage.empty()) {
                ImGui::Text("Upload Stage %s", sceneLoadStatus.uploadStage.c_str());
            }
            if (!sceneLoadStatus.lastBlockingWait.empty()) {
                ImGui::TextWrapped("Last Wait %s", sceneLoadStatus.lastBlockingWait.c_str());
            }
            if (sceneLoadStatus.parseMs > 0.0f) {
                ImGui::Text("Parse %.2f ms", sceneLoadStatus.parseMs);
            }
            if (sceneLoadStatus.prepareMs > 0.0f) {
                ImGui::Text("Prepare %.2f ms", sceneLoadStatus.prepareMs);
            }
            if (sceneLoadStatus.geometryUploadMs > 0.0f) {
                ImGui::Text("Geometry Upload %.2f ms", sceneLoadStatus.geometryUploadMs);
            }
            if (sceneLoadStatus.textureUploadMs > 0.0f) {
                ImGui::Text("Texture Upload %.2f ms", sceneLoadStatus.textureUploadMs);
            }
            if (sceneLoadStatus.pendingUploadBytes > 0 || sceneLoadStatus.pendingUploadCopies > 0) {
                ImGui::Text("Pending Upload %llu bytes / %u copies",
                    static_cast<unsigned long long>(sceneLoadStatus.pendingUploadBytes),
                    sceneLoadStatus.pendingUploadCopies);
            }
            ImGui::Text("PT Frame %u", _renderer.GetPathTraceFrameIndex());
            ImGui::Text("PT SPP %u", settings.pathTraceSamplesPerPixel);
            ImGui::Text("PT Max Bounces %u", settings.pathTraceMaxBounces);
            ImGui::Text("PT Debug %s", PathTraceDebugViewLabel(settings.pathTraceDebugView));
            ImGui::Text("PT Denoiser %s", settings.enablePathTraceDenoiser ? "On" : "Off");
            ImGui::Text("RT Support %s", _renderer.GetRenderDevice().IsRayTracingSupported() ? "Yes" : "No");
            ImGui::Text("Active PT %s", activeBackend);
            if (scene.HasRayTracingScene()) {
                ImGui::Text("BLAS %.2f ms", scene.GetBottomLevelBuildMs());
                ImGui::Text("TLAS %.2f ms", scene.GetTopLevelBuildMs());
            }

            ImGui::SeparatorText("Selection");
            if (ImGui::Button("Select Directional")) {
                _renderer.SelectDirectionalLight();
            }
            ImGui::SameLine();
            if (ImGui::Button("Point")) {
                _renderer.SelectPointLight();
            }
            ImGui::SameLine();
            if (ImGui::Button("Spot")) {
                _renderer.SelectSpotLight();
            }
            ImGui::SameLine();
            if (ImGui::Button("Area")) {
                _renderer.SelectAreaLight();
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear Selection")) {
                _renderer.ClearSelection();
            }
            if (const auto& selection = _renderer.GetSelection();
                selection.kind == vesta::render::SelectionKind::Object && selection.objectIndex < scene.GetObjects().size()) {
                const auto& object = scene.GetObjects()[selection.objectIndex];
                const glm::vec3 translation = object.GetTranslation();
                ImGui::Text("Object %s", object.name.c_str());
                ImGui::Text("Translate %.2f %.2f %.2f", translation.x, translation.y, translation.z);
            } else if (_renderer.GetSelection().kind == vesta::render::SelectionKind::DirectionalLight) {
                ImGui::Text("Drag LMB in viewport to rotate the directional light");
            } else if (_renderer.GetSelection().kind == vesta::render::SelectionKind::PointLight) {
                ImGui::Text("Point Light %.2f %.2f %.2f",
                    settings.pointLightPositionAndIntensity.x,
                    settings.pointLightPositionAndIntensity.y,
                    settings.pointLightPositionAndIntensity.z);
                ImGui::Text("Drag LMB in viewport to move it on the camera plane");
            } else if (_renderer.GetSelection().kind == vesta::render::SelectionKind::SpotLight) {
                ImGui::Text("Spot Light %.2f %.2f %.2f",
                    settings.spotLightPositionAndIntensity.x,
                    settings.spotLightPositionAndIntensity.y,
                    settings.spotLightPositionAndIntensity.z);
                ImGui::Text("Drag LMB in viewport to move it on the camera plane");
            } else if (_renderer.GetSelection().kind == vesta::render::SelectionKind::AreaLight) {
                ImGui::Text("Area Light %.2f %.2f %.2f",
                    settings.areaLightPositionAndIntensity.x,
                    settings.areaLightPositionAndIntensity.y,
                    settings.areaLightPositionAndIntensity.z);
                ImGui::Text("Drag LMB in viewport to move it on the camera plane");
            }

            if (settings.benchmarkOverlay && _renderer.GetFrameTimeHistoryCount() > 0) {
                ImGui::SeparatorText("Benchmark");
                const auto& history = _renderer.GetFrameTimeHistoryMs();
                const int sampleCount = static_cast<int>(_renderer.GetFrameTimeHistoryCount());
                float averageMs = 0.0f;
                float peakMs = 0.0f;
                for (int i = 0; i < sampleCount; ++i) {
                    averageMs += history[static_cast<size_t>(i)];
                    peakMs = std::max(peakMs, history[static_cast<size_t>(i)]);
                }
                averageMs /= static_cast<float>(sampleCount);
                ImGui::Text("Avg %.2f ms", averageMs);
                ImGui::Text("Peak %.2f ms", peakMs);
                ImGui::PlotLines(
                    "Frame Times", history.data(), sampleCount, 0, nullptr, 0.0f, std::max(33.0f, peakMs * 1.1f), ImVec2(0.0f, 72.0f));
            }
        }
    }
        ImGui::End();
    }

    if (_showLegacyRenderPanel) {
        ImGui::SetNextWindowPos(ImVec2(458.0f, 18.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Render", &_showLegacyRenderPanel, ImGuiWindowFlags_NoSavedSettings)) {
        if (ImGui::Button("Apply Recommended")) {
            _renderer.ApplyPreset(vesta::render::RendererPreset::Recommended);
        }
        ImGui::SameLine();
        if (ImGui::Button("Performance")) {
            _renderer.ApplyPreset(vesta::render::RendererPreset::Performance);
        }
        ImGui::SameLine();
        if (ImGui::Button("Balanced")) {
            _renderer.ApplyPreset(vesta::render::RendererPreset::Balanced);
        }
        ImGui::SameLine();
        if (ImGui::Button("Quality")) {
            _renderer.ApplyPreset(vesta::render::RendererPreset::Quality);
        }

        const char* displayModes[] = { "Composite", "Raster", "Gaussian", "Path Trace", "Ray Trace" };
        int displayMode = static_cast<int>(settings.displayMode);
        if (ImGui::Combo("Display", &displayMode, displayModes, IM_ARRAYSIZE(displayModes))) {
            vesta::render::ApplyDisplayModePassSelection(settings, static_cast<vesta::render::RendererDisplayMode>(displayMode));
            _renderer.ResetAccumulation();
        }

        ImGui::TextDisabled("Pass selection follows Display; Composite keeps the hybrid stack.");
        if (ImGui::SliderFloat("Point Opacity", &settings.gaussianOpacity, 0.05f, 1.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }
        int gaussianShDegree = static_cast<int>(settings.gaussianShDegree);
        if (ImGui::SliderInt("Gaussian SH Degree", &gaussianShDegree, 0, 3)) {
            settings.gaussianShDegree = static_cast<uint32_t>(gaussianShDegree);
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("View-Dependent Gaussian Color", &settings.gaussianViewDependentColor)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("Gaussian Antialiasing", &settings.gaussianAntialiasing)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("Gaussian Fast Culling", &settings.gaussianFastCulling)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Gaussian Mix", &settings.gaussianMix, 0.0f, 1.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }
        ImGui::SeparatorText("Raster Lighting");
        if (ImGui::Checkbox("Global Illumination", &settings.enableGlobalIllumination)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("Ambient Occlusion", &settings.enableAmbientOcclusion)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("SSAO", &settings.enableSsao)) {
            _renderer.ResetAccumulation();
        }
        if (settings.enableSsao) {
            if (ImGui::SliderFloat("SSAO Radius", &settings.ssaoRadius, 0.05f, 5.0f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
            if (ImGui::SliderFloat("SSAO Intensity", &settings.ssaoIntensity, 0.0f, 4.0f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
        }
        if (ImGui::Checkbox("TAA", &settings.enableTaa)) {
            _renderer.ResetAccumulation();
        }
        if (settings.enableTaa) {
            if (ImGui::SliderFloat("TAA Feedback", &settings.taaFeedback, 0.0f, 0.98f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
        }
        if (ImGui::Checkbox("SSR", &settings.enableSsr)) {
            _renderer.ResetAccumulation();
        }
        if (settings.enableSsr) {
            if (ImGui::SliderFloat("SSR Distance", &settings.ssrMaxDistance, 0.5f, 100.0f, "%.1f")) {
                _renderer.ResetAccumulation();
            }
            if (ImGui::SliderFloat("SSR Thickness", &settings.ssrThickness, 0.01f, 2.0f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
            if (ImGui::SliderFloat("SSR Intensity", &settings.ssrIntensity, 0.0f, 2.0f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
        }
        if (ImGui::Checkbox("SSGI", &settings.enableSsgi)) {
            _renderer.ResetAccumulation();
        }
        if (settings.enableSsgi) {
            if (ImGui::SliderFloat("SSGI Radius", &settings.ssgiRadius, 0.05f, 8.0f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
            if (ImGui::SliderFloat("SSGI Intensity", &settings.ssgiIntensity, 0.0f, 2.0f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
            int ssgiSamples = static_cast<int>(settings.ssgiSampleCount);
            if (ImGui::SliderInt("SSGI Samples", &ssgiSamples, 4, 16)) {
                settings.ssgiSampleCount = static_cast<uint32_t>(std::clamp(ssgiSamples, 4, 16));
                _renderer.ResetAccumulation();
            }
        }
        if (ImGui::SliderFloat("PT Resolution", &settings.pathTraceResolutionScale, 0.25f, 1.0f, "%.2fx")) {
            _renderer.ResetAccumulation();
        }
        int pathTraceSpp = static_cast<int>(settings.pathTraceSamplesPerPixel);
        if (ImGui::SliderInt("PT Samples / Pixel", &pathTraceSpp, 1, 16)) {
            settings.pathTraceSamplesPerPixel = static_cast<uint32_t>(pathTraceSpp);
            _renderer.ResetAccumulation();
        }
        int pathTraceMaxBounces = static_cast<int>(settings.pathTraceMaxBounces);
        if (ImGui::SliderInt("PT Max Bounces", &pathTraceMaxBounces, 1, 12)) {
            settings.pathTraceMaxBounces = static_cast<uint32_t>(pathTraceMaxBounces);
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("PT Next Event Estimation", &settings.pathTraceNextEventEstimation)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("PT Russian Roulette", &settings.pathTraceRussianRoulette)) {
            _renderer.ResetAccumulation();
        }
        if (settings.pathTraceRussianRoulette) {
            int russianRouletteDepth = static_cast<int>(settings.pathTraceRussianRouletteDepth);
            if (ImGui::SliderInt("PT RR Depth", &russianRouletteDepth, 1, 12)) {
                settings.pathTraceRussianRouletteDepth = static_cast<uint32_t>(std::clamp(russianRouletteDepth, 1, 12));
                _renderer.ResetAccumulation();
            }
        }
        if (ImGui::SliderFloat("PT Firefly Clamp", &settings.pathTraceFireflyClamp, 0.0f, 64.0f, "%.1f")) {
            _renderer.ResetAccumulation();
        }

        const char* pathTraceDebugViews[] = { "Final", "Albedo", "Normal", "Depth", "Direct", "Indirect", "Ray Count Heatmap", "Diffuse Bounce", "Specular Bounce", "Throughput", "PDF" };
        int pathTraceDebugView = static_cast<int>(settings.pathTraceDebugView);
        if (ImGui::Combo("PT Debug View", &pathTraceDebugView, pathTraceDebugViews, IM_ARRAYSIZE(pathTraceDebugViews))) {
            vesta::render::SelectPathTraceDebugView(settings, static_cast<vesta::render::PathTraceDebugView>(pathTraceDebugView));
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("PT Denoiser", &settings.enablePathTraceDenoiser)) {
            _renderer.ResetAccumulation();
        }
        if (settings.enablePathTraceDenoiser) {
            if (settings.pathTraceDebugView != vesta::render::PathTraceDebugView::Final) {
                ImGui::TextDisabled("Denoiser applies to Final view only");
            }
            if (ImGui::SliderFloat("PT Denoiser Strength", &settings.pathTraceDenoiserStrength, 0.0f, 1.0f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
            if (ImGui::SliderFloat("PT Denoiser Temporal", &settings.pathTraceDenoiserTemporalBlend, 0.0f, 0.98f, "%.2f")) {
                _renderer.ResetAccumulation();
            }
            int denoiserIterations = static_cast<int>(settings.pathTraceDenoiserIterations);
            if (ImGui::SliderInt("PT Denoiser Iterations", &denoiserIterations, 1, 5)) {
                settings.pathTraceDenoiserIterations = static_cast<uint32_t>(std::clamp(denoiserIterations, 1, 5));
                _renderer.ResetAccumulation();
            }
        }
        if (ImGui::Button("Reset PT Accumulation")) {
            _renderer.ResetAccumulation();
        }

        if (ImGui::Checkbox("Shadow Map", &settings.enableShadowMap)) {
            _renderer.ResetAccumulation();
        }
        int shadowMapSize = static_cast<int>(settings.shadowMapSize);
        if (ImGui::SliderInt("Shadow Size", &shadowMapSize, 512, 4096)) {
            settings.shadowMapSize = static_cast<uint32_t>(std::clamp(shadowMapSize, 512, 4096));
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Shadow Bias", &settings.shadowBias, 0.0f, 0.01f, "%.4f")) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Normal Bias", &settings.shadowNormalBias, 0.0f, 0.1f, "%.3f")) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Shadow Strength", &settings.shadowStrength, 0.0f, 1.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("PCSS Soft Shadows", &settings.enablePcssShadows)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Shadow Filter Radius", &settings.shadowFilterRadius, 0.5f, 4.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::Checkbox("Contact Shadows", &settings.enableContactShadows)) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Contact Length", &settings.contactShadowLength, 0.05f, 8.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Contact Intensity", &settings.contactShadowIntensity, 0.0f, 1.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }

        const char* backendModes[] = { "Auto", "Compute", "Hardware RT" };
        int backendMode = static_cast<int>(settings.pathTraceBackend);
        if (ImGui::Combo("PT Backend", &backendMode, backendModes, IM_ARRAYSIZE(backendModes))) {
            settings.pathTraceBackend = static_cast<vesta::render::PathTraceBackend>(backendMode);
            _renderer.ResetAccumulation();
        }

        float lightDirection[3] = {
            settings.lightDirectionAndIntensity.x,
            settings.lightDirectionAndIntensity.y,
            settings.lightDirectionAndIntensity.z,
        };
        if (ImGui::SliderFloat3("Light Dir", lightDirection, -1.0f, 1.0f, "%.2f")) {
            glm::vec3 direction = glm::vec3(lightDirection[0], lightDirection[1], lightDirection[2]);
            if (glm::length(direction) > 1.0e-4f) {
                direction = glm::normalize(direction);
                settings.lightDirectionAndIntensity = glm::vec4(direction, settings.lightDirectionAndIntensity.w);
                _renderer.ResetAccumulation();
            }
        }
        if (ImGui::SliderFloat("Light Intensity", &settings.lightDirectionAndIntensity.w, 0.0f, 8.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }
        if (ImGui::SliderFloat("Emission", &settings.emissiveIntensity, 0.0f, 64.0f, "%.2f")) {
            _renderer.ResetAccumulation();
        }
    }
        ImGui::End();
    }

    if (_showLegacyCameraPanel) {
        ImGui::SetNextWindowPos(ImVec2(18.0f, 340.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Camera", &_showLegacyCameraPanel, ImGuiWindowFlags_NoSavedSettings)) {
        const bool canOrbitSelection =
            _renderer.GetSelection().kind == vesta::render::SelectionKind::Object
            && _renderer.GetSelection().objectIndex < scene.GetObjects().size();
        if (!canOrbitSelection) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Orbit Selected") && canOrbitSelection) {
            _renderer.OrbitCameraAroundSelection();
        }
        ImGui::SameLine();
        if (ImGui::Button("Dolly Selected") && canOrbitSelection) {
            _renderer.DollyCameraAroundSelection();
        }
        if (!canOrbitSelection) {
            ImGui::EndDisabled();
        }
        if (ImGui::Button("Orbit Scene")) {
            _renderer.OrbitCameraAroundScene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Dolly Scene")) {
            _renderer.DollyCameraAroundScene();
        }
        ImGui::SameLine();
        if (ImGui::Button("Fly Camera")) {
            _renderer.DisableCameraOrbit();
        }

        ImGui::Text("Mode %s", camera.IsDollyOrbitEnabled() ? "Dolly Orbit" : (camera.IsOrbitEnabled() ? "Orbit" : "Fly"));
        if (camera.IsOrbitEnabled()) {
            const glm::vec3 orbitTarget = camera.GetOrbitTarget();
            ImGui::Text("Target %.3f %.3f %.3f", orbitTarget.x, orbitTarget.y, orbitTarget.z);
            float orbitRadius = camera.GetOrbitRadius();
            if (ImGui::InputFloat("Dolly Radius", &orbitRadius, 0.1f, 1.0f, "%.3f")) {
                camera.SetOrbitRadius(orbitRadius);
                _renderer.ResetAccumulation();
            }
            float dollySpeed = camera.GetDollySpeedDegrees();
            if (ImGui::InputFloat("Dolly Speed", &dollySpeed, 1.0f, 10.0f, "%.2f deg/s")) {
                camera.SetDollySpeedDegrees(dollySpeed);
            }
            ImGui::Text("Track Selected %s", _renderer.IsTrackingSelectionOrbit() ? "On" : "Off");
        }

        float cameraPosition[3] = { camera.GetPosition().x, camera.GetPosition().y, camera.GetPosition().z };
        if (ImGui::InputFloat3("Position", cameraPosition, "%.3f")) {
            camera.SetPosition(glm::vec3(cameraPosition[0], cameraPosition[1], cameraPosition[2]));
            _renderer.ResetAccumulation();
        }

        const glm::vec3 rotation = camera.GetRotationDegrees();
        float cameraRotation[3] = { rotation.x, rotation.y, rotation.z };
        if (ImGui::InputFloat3("Rotation", cameraRotation, "%.2f")) {
            camera.SetRotationDegrees(glm::vec3(cameraRotation[0], cameraRotation[1], cameraRotation[2]));
            _renderer.ResetAccumulation();
        }
        float moveSpeed = camera.GetMoveSpeed();
        if (ImGui::DragFloat("Move Speed", &moveSpeed, 0.25f, 0.05f, 1000.0f, "%.2f")) {
            camera.SetMoveSpeed(moveSpeed);
        }
        ImGui::TextDisabled("Rotation order: Yaw Pitch Roll");
        ImGui::Text("Forward %.3f %.3f %.3f", camera.GetForward().x, camera.GetForward().y, camera.GetForward().z);
        ImGui::Text("Up %.3f %.3f %.3f", camera.GetUp().x, camera.GetUp().y, camera.GetUp().z);
        ImGui::SeparatorText("Controls");
        ImGui::Text("RMB + Mouse Look, RMB + Wheel Speed");
        ImGui::Text("Wheel Dolly, WASD / Q / E Move");
        ImGui::Text("LMB Pick/Drag Object");
        ImGui::Text("L Select Light, Esc Clear Selection");
        ImGui::Text("1 Raster, 2 Gaussian, 3 PT, 4 Composite");
        ImGui::Text("R/G/P toggles, F1 UI, F5 Reload, F12 Screenshot");
    }
        ImGui::End();
    }

    if (_showSceneInspectorPanel) {
        ImGui::SetNextWindowPos(ImVec2(1124.0f, 552.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420.0f, 360.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Scene / Camera / Light", &_showSceneInspectorPanel, ImGuiWindowFlags_NoSavedSettings)) {
            if (ImGui::BeginTabBar("SceneInspectorTabs")) {
                if (ImGui::BeginTabItem("Outliner")) {
                    if (ImGui::BeginTable("ObjectTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Object");
                        ImGui::TableSetupColumn("Prims", ImGuiTableColumnFlags_WidthFixed, 52.0f);
                        ImGui::TableSetupColumn("Tris", ImGuiTableColumnFlags_WidthFixed, 52.0f);
                        ImGui::TableSetupColumn("Position");
                        ImGui::TableHeadersRow();
                        const auto& objects = scene.GetObjects();
                        for (size_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex) {
                            const auto& object = objects[objectIndex];
                            const glm::vec3 position = object.GetTranslation();
                            const bool selected = _renderer.GetSelection().kind == vesta::render::SelectionKind::Object
                                && _renderer.GetSelection().objectIndex == static_cast<uint32_t>(objectIndex);
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(static_cast<int>(objectIndex));
                            const char* objectName = object.name.empty() ? "(unnamed)" : object.name.c_str();
                            if (ImGui::Selectable(objectName, selected, ImGuiSelectableFlags_SpanAllColumns)) {
                                _renderer.SelectObject(static_cast<uint32_t>(objectIndex));
                            }
                            ImGui::PopID();
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%u", object.primitiveCount);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%u", object.triangleCount);
                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%.2f %.2f %.2f", position.x, position.y, position.z);
                        }
                        auto drawLightRow = [&](const char* label,
                                                vesta::render::SelectionKind kind,
                                                bool enabled,
                                                glm::vec3 vectorValue,
                                                auto selectFn) {
                            const bool selected = _renderer.GetSelection().kind == kind;
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(label);
                            if (ImGui::Selectable(label, selected, ImGuiSelectableFlags_SpanAllColumns)) {
                                selectFn();
                            }
                            ImGui::PopID();
                            ImGui::TableSetColumnIndex(1);
                            ImGui::TextUnformatted("Light");
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TextUnformatted(enabled ? "On" : "Off");
                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%.2f %.2f %.2f", vectorValue.x, vectorValue.y, vectorValue.z);
                        };
                        drawLightRow("Directional Light",
                            vesta::render::SelectionKind::DirectionalLight,
                            true,
                            glm::vec3(settings.lightDirectionAndIntensity),
                            [&]() { _renderer.SelectDirectionalLight(); });
                        drawLightRow("Point Light",
                            vesta::render::SelectionKind::PointLight,
                            settings.enablePointLight,
                            glm::vec3(settings.pointLightPositionAndIntensity),
                            [&]() { _renderer.SelectPointLight(); });
                        drawLightRow("Spot Light",
                            vesta::render::SelectionKind::SpotLight,
                            settings.enableSpotLight,
                            glm::vec3(settings.spotLightPositionAndIntensity),
                            [&]() { _renderer.SelectSpotLight(); });
                        drawLightRow("Area Light",
                            vesta::render::SelectionKind::AreaLight,
                            settings.enableAreaLight,
                            glm::vec3(settings.areaLightPositionAndIntensity),
                            [&]() { _renderer.SelectAreaLight(); });
                        ImGui::EndTable();
                    }
                    if (_renderer.GetSelection().kind == vesta::render::SelectionKind::Object) {
                        if (ImGui::Button("Orbit Selected")) {
                            _renderer.OrbitCameraAroundSelection();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Dolly Selected")) {
                            _renderer.DollyCameraAroundSelection();
                        }
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Camera")) {
                    float fov = camera.GetFovDegrees();
                    float nearPlane = camera.GetNearPlane();
                    float farPlane = camera.GetFarPlane();
                    bool lensChanged = ImGui::SliderFloat("FOV", &fov, 20.0f, 120.0f, "%.1f");
                    lensChanged |= ImGui::InputFloat("Near", &nearPlane, 0.01f, 0.1f, "%.3f");
                    lensChanged |= ImGui::InputFloat("Far", &farPlane, 1.0f, 10.0f, "%.1f");
                    if (lensChanged) {
                        camera.SetLens(fov, nearPlane, farPlane);
                        _renderer.ResetAccumulation();
                    }
                    float moveSpeed = camera.GetMoveSpeed();
                    if (ImGui::DragFloat("Move Speed", &moveSpeed, 0.25f, 0.05f, 1000.0f, "%.2f")) {
                        camera.SetMoveSpeed(moveSpeed);
                    }
                    ImGui::SliderFloat("Exposure", &settings.cameraExposureEv, -6.0f, 6.0f, "%.2f EV");
                    bool dofChanged = ImGui::SliderFloat("Aperture Radius", &settings.cameraApertureRadius, 0.0f, 0.25f, "%.3f");
                    dofChanged |= ImGui::SliderFloat("Focal Distance", &settings.cameraFocalDistance, 0.05f, 100.0f, "%.2f");
                    if (dofChanged) {
                        _renderer.ResetAccumulation();
                    }
                    ImGui::Text("Position %.3f %.3f %.3f", camera.GetPosition().x, camera.GetPosition().y, camera.GetPosition().z);
                    ImGui::Text("Rotation %.2f %.2f %.2f",
                        camera.GetRotationDegrees().x,
                        camera.GetRotationDegrees().y,
                        camera.GetRotationDegrees().z);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Transform")) {
                    const auto& selection = _renderer.GetSelection();
                    const auto& objects = scene.GetObjects();
                    if (selection.kind == vesta::render::SelectionKind::Object && selection.objectIndex < objects.size()) {
                        const auto& object = objects[selection.objectIndex];
                        glm::vec3 position = object.GetTranslation();
                        ImGui::Text("Object %u", selection.objectIndex);
                        ImGui::TextUnformatted(object.name.empty() ? "(unnamed)" : object.name.c_str());
                        if (ImGui::DragFloat3("Position", &position.x, 0.01f, -10000.0f, 10000.0f, "%.3f")) {
                            _renderer.SetSelectedObjectPosition(position);
                        }
                        static float rotationDelta[3] = { 0.0f, 0.0f, 0.0f };
                        ImGui::InputFloat3("Rotate Delta", rotationDelta, "%.2f deg");
                        if (ImGui::Button("Apply Rotation")) {
                            if (_renderer.RotateSelectedObject(glm::vec3(rotationDelta[0], rotationDelta[1], rotationDelta[2]))) {
                                rotationDelta[0] = 0.0f;
                                rotationDelta[1] = 0.0f;
                                rotationDelta[2] = 0.0f;
                            }
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Reset Delta")) {
                            rotationDelta[0] = 0.0f;
                            rotationDelta[1] = 0.0f;
                            rotationDelta[2] = 0.0f;
                        }
                        static float uniformScale = 1.0f;
                        ImGui::InputFloat("Uniform Scale", &uniformScale, 0.05f, 0.25f, "%.3f");
                        ImGui::SameLine();
                        if (ImGui::Button("Apply Scale")) {
                            if (_renderer.ScaleSelectedObject(uniformScale)) {
                                uniformScale = 1.0f;
                            }
                        }
                        ImGui::Text("Bounds Center %.3f %.3f %.3f",
                            object.bounds.center.x,
                            object.bounds.center.y,
                            object.bounds.center.z);
                        ImGui::Text("Radius %.3f", object.bounds.radius);
                        ImGui::Text("Vertices %u  Triangles %u", object.vertexCount, object.triangleCount);
                    } else if (selection.kind == vesta::render::SelectionKind::PointLight) {
                        ImGui::TextUnformatted("Point Light");
                        if (ImGui::DragFloat3("Position", &settings.pointLightPositionAndIntensity.x, 0.05f, -100.0f, 100.0f, "%.2f")) {
                            _renderer.ResetAccumulation();
                        }
                        if (ImGui::SliderFloat("Intensity", &settings.pointLightPositionAndIntensity.w, 0.0f, 64.0f, "%.2f")) {
                            _renderer.ResetAccumulation();
                        }
                        if (ImGui::ColorEdit3("Color", &settings.pointLightColor.x, ImGuiColorEditFlags_Float)) {
                            _renderer.ResetAccumulation();
                        }
                    } else if (selection.kind == vesta::render::SelectionKind::SpotLight) {
                        ImGui::TextUnformatted("Spot Light");
                        if (ImGui::DragFloat3("Position", &settings.spotLightPositionAndIntensity.x, 0.05f, -100.0f, 100.0f, "%.2f")) {
                            _renderer.ResetAccumulation();
                        }
                        float spotDirection[3] = {
                            settings.spotLightDirectionAndAngle.x,
                            settings.spotLightDirectionAndAngle.y,
                            settings.spotLightDirectionAndAngle.z,
                        };
                        if (ImGui::SliderFloat3("Direction", spotDirection, -1.0f, 1.0f, "%.2f")) {
                            glm::vec3 direction(spotDirection[0], spotDirection[1], spotDirection[2]);
                            if (glm::length(direction) > 1.0e-4f) {
                                direction = glm::normalize(direction);
                                settings.spotLightDirectionAndAngle = glm::vec4(direction, settings.spotLightDirectionAndAngle.w);
                                _renderer.ResetAccumulation();
                            }
                        }
                        if (ImGui::SliderFloat("Intensity", &settings.spotLightPositionAndIntensity.w, 0.0f, 96.0f, "%.2f")) {
                            _renderer.ResetAccumulation();
                        }
                    } else if (selection.kind == vesta::render::SelectionKind::AreaLight) {
                        ImGui::TextUnformatted("Area Light");
                        if (ImGui::DragFloat3("Position", &settings.areaLightPositionAndIntensity.x, 0.05f, -100.0f, 100.0f, "%.2f")) {
                            _renderer.ResetAccumulation();
                        }
                        if (ImGui::SliderFloat("Size", &settings.areaLightNormalAndSize.w, 0.1f, 12.0f, "%.2f")) {
                            _renderer.ResetAccumulation();
                        }
                        if (ImGui::SliderFloat("Intensity", &settings.areaLightPositionAndIntensity.w, 0.0f, 48.0f, "%.2f")) {
                            _renderer.ResetAccumulation();
                        }
                    } else {
                        ImGui::TextUnformatted("No object or light selected");
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Light")) {
                    if (ImGui::Button("Select Directional")) {
                        _renderer.SelectDirectionalLight();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Select Point")) {
                        _renderer.SelectPointLight();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Select Spot")) {
                        _renderer.SelectSpotLight();
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Select Area")) {
                        _renderer.SelectAreaLight();
                    }
                    float lightDirection[3] = {
                        settings.lightDirectionAndIntensity.x,
                        settings.lightDirectionAndIntensity.y,
                        settings.lightDirectionAndIntensity.z,
                    };
                    if (ImGui::SliderFloat3("Directional", lightDirection, -1.0f, 1.0f, "%.2f")) {
                        glm::vec3 direction(lightDirection[0], lightDirection[1], lightDirection[2]);
                        if (glm::length(direction) > 1.0e-4f) {
                            direction = glm::normalize(direction);
                            settings.lightDirectionAndIntensity = glm::vec4(direction, settings.lightDirectionAndIntensity.w);
                            _renderer.ResetAccumulation();
                        }
                    }
                    if (ImGui::SliderFloat("Intensity", &settings.lightDirectionAndIntensity.w, 0.0f, 8.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::ColorEdit3("Directional Color", &settings.directionalLightColor.x, ImGuiColorEditFlags_Float)) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("Emission", &settings.emissiveIntensity, 0.0f, 64.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    ImGui::SeparatorText("Point Light");
                    if (ImGui::Checkbox("Point Enabled", &settings.enablePointLight)) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::DragFloat3("Point Position", &settings.pointLightPositionAndIntensity.x, 0.05f, -100.0f, 100.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("Point Intensity", &settings.pointLightPositionAndIntensity.w, 0.0f, 64.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::ColorEdit3("Point Color", &settings.pointLightColor.x, ImGuiColorEditFlags_Float)) {
                        _renderer.ResetAccumulation();
                    }
                    ImGui::Text("Radius %.1f", 8.0f);
                    ImGui::SeparatorText("Spot Light");
                    if (ImGui::Checkbox("Spot Enabled", &settings.enableSpotLight)) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::DragFloat3("Spot Position", &settings.spotLightPositionAndIntensity.x, 0.05f, -100.0f, 100.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    float spotDirection[3] = {
                        settings.spotLightDirectionAndAngle.x,
                        settings.spotLightDirectionAndAngle.y,
                        settings.spotLightDirectionAndAngle.z,
                    };
                    if (ImGui::SliderFloat3("Spot Direction", spotDirection, -1.0f, 1.0f, "%.2f")) {
                        glm::vec3 direction(spotDirection[0], spotDirection[1], spotDirection[2]);
                        if (glm::length(direction) > 1.0e-4f) {
                            direction = glm::normalize(direction);
                            settings.spotLightDirectionAndAngle = glm::vec4(direction, settings.spotLightDirectionAndAngle.w);
                            _renderer.ResetAccumulation();
                        }
                    }
                    if (ImGui::SliderFloat("Spot Angle", &settings.spotLightDirectionAndAngle.w, 5.0f, 80.0f, "%.1f deg")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("Spot Intensity", &settings.spotLightPositionAndIntensity.w, 0.0f, 96.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::ColorEdit3("Spot Color", &settings.spotLightColor.x, ImGuiColorEditFlags_Float)) {
                        _renderer.ResetAccumulation();
                    }
                    ImGui::SeparatorText("Area Light");
                    if (ImGui::Checkbox("Area Enabled", &settings.enableAreaLight)) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::DragFloat3("Area Position", &settings.areaLightPositionAndIntensity.x, 0.05f, -100.0f, 100.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    float areaNormal[3] = {
                        settings.areaLightNormalAndSize.x,
                        settings.areaLightNormalAndSize.y,
                        settings.areaLightNormalAndSize.z,
                    };
                    if (ImGui::SliderFloat3("Area Normal", areaNormal, -1.0f, 1.0f, "%.2f")) {
                        glm::vec3 normal(areaNormal[0], areaNormal[1], areaNormal[2]);
                        if (glm::length(normal) > 1.0e-4f) {
                            normal = glm::normalize(normal);
                            settings.areaLightNormalAndSize = glm::vec4(normal, settings.areaLightNormalAndSize.w);
                            _renderer.ResetAccumulation();
                        }
                    }
                    if (ImGui::SliderFloat("Area Size", &settings.areaLightNormalAndSize.w, 0.1f, 12.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("Area Intensity", &settings.areaLightPositionAndIntensity.w, 0.0f, 48.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::ColorEdit3("Area Color", &settings.areaLightColor.x, ImGuiColorEditFlags_Float)) {
                        _renderer.ResetAccumulation();
                    }
                    ImGui::SeparatorText("Environment");
                    const char* environmentPresets[] = { "Studio", "Sunset", "Night", "Forest" };
                    int environmentPreset = static_cast<int>(std::clamp(settings.environmentPreset, 0u, 3u));
                    if (ImGui::Combo("Preset", &environmentPreset, environmentPresets, IM_ARRAYSIZE(environmentPresets))) {
                        settings.environmentPreset = static_cast<uint32_t>(environmentPreset);
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("Env Intensity", &settings.environmentIntensity, 0.0f, 4.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("Env Rotation", &settings.environmentRotationDegrees, 0.0f, 360.0f, "%.1f deg")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("IBL Diffuse", &settings.environmentDiffuseStrength, 0.0f, 2.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    if (ImGui::SliderFloat("IBL Specular", &settings.environmentSpecularStrength, 0.0f, 2.0f, "%.2f")) {
                        _renderer.ResetAccumulation();
                    }
                    ImGui::Text("Source Procedural IBL: %s", EnvironmentPresetLabel(settings.environmentPreset));
                    ImGui::Text("External HDRI: %s", settings.externalHdriAvailable ? "Uploaded" : "Not active");
                    if (settings.externalHdriAvailable) {
                        ImGui::Text("%ux%u  %uch  %s",
                            settings.externalHdriWidth,
                            settings.externalHdriHeight,
                            settings.externalHdriChannels,
                            settings.externalHdriIsHdr ? "HDR" : "LDR");
                    }
                    ImGui::TextWrapped("%s", settings.externalHdriStatus.c_str());
                    const auto iblStats = _renderer.GetIblStats();
                    ImGui::SeparatorText("IBL Pipeline Readiness");
                    ImGui::Text("Source %s  Env Cube %u^2  Diffuse %u^2  Specular %u^2/%u mips  BRDF LUT %u^2",
                        iblStats.externalSourceAvailable ? "External HDRI" : "Procedural sky",
                        iblStats.environmentCubemapResolution,
                        iblStats.diffuseCubemapResolution,
                        iblStats.specularCubemapResolution,
                        iblStats.specularMipCount,
                        iblStats.brdfLutResolution);
                    ImGui::Text("Estimated GPU memory %.2f MiB  (env cube %.2f / diffuse %.2f / specular %.2f / LUT %.2f)",
                        MiB(iblStats.estimatedEnvironmentCubemapBytes + iblStats.estimatedDiffuseBytes + iblStats.estimatedSpecularBytes + iblStats.estimatedBrdfLutBytes),
                        MiB(iblStats.estimatedEnvironmentCubemapBytes),
                        MiB(iblStats.estimatedDiffuseBytes),
                        MiB(iblStats.estimatedSpecularBytes),
                        MiB(iblStats.estimatedBrdfLutBytes));
                    ImGui::Text("Cube %s  Diffuse %s  Prefilter %s  BRDF LUT %s",
                        iblStats.environmentCubemapAvailable ? "cube image" : (iblStats.environmentMapUploaded ? "staged" : "procedural fallback"),
                        iblStats.diffuseIrradianceAvailable ? "irradiance texture" : (iblStats.environmentMapUploaded ? "equirect sample" : (iblStats.diffuseBackendAvailable ? "procedural live" : "staged")),
                        iblStats.specularPrefilterAvailable ? "prefilter cube mips" : "staged",
                        iblStats.brdfLutAvailable ? "live" : "staged");
                    ImGui::TextDisabled(iblStats.environmentMapUploaded
                        ? "External HDRI is sampled directly, converted into Vulkan cube images, and convolved into diffuse irradiance plus specular prefilter cube mips."
                        : "Procedural sky is live; external HDRI irradiance/prefilter generation is available after an HDRI is loaded.");
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Animation")) {
                    ImGui::Checkbox("Play", &settings.animationPlaying);
                    ImGui::SameLine();
                    if (ImGui::Button("Stop")) {
                        settings.animationPlaying = false;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Reset Time")) {
                        settings.animationTimeSeconds = 0.0f;
                        _renderer.ResetAccumulation();
                    }
                    ImGui::SliderFloat("Time Scale", &settings.animationTimeScale, 0.0f, 4.0f, "%.2fx");
                    ImGui::InputFloat("Time", &settings.animationTimeSeconds, 0.1f, 1.0f, "%.2f s");
                    ImGui::Checkbox("Animate Directional Light", &settings.animateDirectionalLight);
                    ImGui::Checkbox("Animate Environment", &settings.animateEnvironment);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Materials")) {
                    if (ImGui::BeginTable("MaterialTable", 9, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 42.0f);
                        ImGui::TableSetupColumn("Base Color", ImGuiTableColumnFlags_WidthFixed, 112.0f);
                        ImGui::TableSetupColumn("Metallic", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("Roughness", ImGuiTableColumnFlags_WidthFixed, 76.0f);
                        ImGui::TableSetupColumn("Trans", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                        ImGui::TableSetupColumn("IOR", ImGuiTableColumnFlags_WidthFixed, 56.0f);
                        ImGui::TableSetupColumn("Emissive", ImGuiTableColumnFlags_WidthFixed, 112.0f);
                        ImGui::TableSetupColumn("Normal", ImGuiTableColumnFlags_WidthFixed, 58.0f);
                        ImGui::TableSetupColumn("Textures");
                        ImGui::TableHeadersRow();
                        const auto& materials = scene.GetMaterials();
                        for (size_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex) {
                            auto material = materials[materialIndex];
                            bool materialChanged = false;
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%zu", materialIndex);
                            ImGui::PushID(static_cast<int>(materialIndex));
                            ImGui::TableSetColumnIndex(1);
                            materialChanged |= ImGui::ColorEdit4("##base",
                                &material.baseColorFactor.x,
                                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
                            ImGui::TableSetColumnIndex(2);
                            materialChanged |= ImGui::SliderFloat("##metallic", &material.materialParams.x, 0.0f, 1.0f, "%.2f");
                            ImGui::TableSetColumnIndex(3);
                            materialChanged |= ImGui::SliderFloat("##roughness", &material.materialParams.y, 0.02f, 1.0f, "%.2f");
                            ImGui::TableSetColumnIndex(4);
                            materialChanged |= ImGui::SliderFloat("##trans", &material.opticalParams.x, 0.0f, 1.0f, "%.2f");
                            ImGui::TableSetColumnIndex(5);
                            materialChanged |= ImGui::SliderFloat("##ior", &material.opticalParams.y, 1.0f, 2.5f, "%.2f");
                            ImGui::TableSetColumnIndex(6);
                            materialChanged |= ImGui::ColorEdit3("##emissive", &material.emissiveFactor.x, ImGuiColorEditFlags_NoInputs);
                            ImGui::TableSetColumnIndex(7);
                            materialChanged |= ImGui::SliderFloat("##normal", &material.materialParams.w, 0.0f, 2.0f, "%.2f");
                            ImGui::TableSetColumnIndex(8);
                            ImGui::Text("BC %u MR %u N %u E %u",
                                material.textureIndices0.x,
                                material.textureIndices0.y,
                                material.textureIndices0.z,
                                material.textureIndices1.x);
                            if (materialChanged) {
                                _renderer.UpdateMaterial(static_cast<uint32_t>(materialIndex), material);
                            }
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
    }

    if (_showResourceInspectorPanel) {
        ImGui::SetNextWindowPos(ImVec2(590.0f, 552.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(520.0f, 360.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Resource Inspector", &_showResourceInspectorPanel, ImGuiWindowFlags_NoSavedSettings)) {
            const uint64_t vertexBytes = static_cast<uint64_t>(scene.GetVertices().size()) * sizeof(vesta::scene::SceneVertex);
            const uint64_t indexBytes = static_cast<uint64_t>(scene.GetIndices().size()) * sizeof(uint32_t);
            const uint64_t materialBytes = static_cast<uint64_t>(scene.GetMaterials().size()) * sizeof(vesta::scene::SceneMaterial);
            const uint64_t triangleBytes = static_cast<uint64_t>(scene.GetTriangles().size()) * sizeof(vesta::scene::SceneTriangle);
            const uint64_t emissiveBytes =
                static_cast<uint64_t>(scene.GetEmissiveTriangles().size()) * sizeof(vesta::scene::SceneEmissiveTriangle);
            const uint64_t gaussianBytes =
                static_cast<uint64_t>(scene.GetGaussians().size()) * sizeof(vesta::scene::GaussianPrimitive);
            uint64_t textureBytes = 0;
            uint64_t residentTextureBytes = 0;
            if (_texturePreviewSceneVersion != scene.GetContentVersion()
                || _texturePreviewDescriptors.size() != scene.GetTextures().size()) {
                clear_texture_preview_descriptors();
                _texturePreviewDescriptors.assign(scene.GetTextures().size(), VK_NULL_HANDLE);
                _texturePreviewSceneVersion = scene.GetContentVersion();
                _selectedTexturePreviewIndex = 0;
                _selectedBufferInspectorIndex = 0;
            }
            for (size_t textureIndex = 0; textureIndex < scene.GetTextures().size(); ++textureIndex) {
                const uint64_t bytes = TextureAssetBytes(scene.GetTextures()[textureIndex]);
                textureBytes += bytes;
                if (scene.HasResidentTexture(textureIndex)) {
                    residentTextureBytes += bytes;
                }
            }
            const uint64_t sceneBufferBytes = BufferSizeBytes(device, scene.GetVertexBuffer())
                + BufferSizeBytes(device, scene.GetIndexBuffer())
                + BufferSizeBytes(device, scene.GetMaterialBuffer())
                + BufferSizeBytes(device, scene.GetTriangleBuffer())
                + BufferSizeBytes(device, scene.GetEmissiveTriangleBuffer())
                + BufferSizeBytes(device, scene.GetGaussianBuffer());
            const uint64_t accelerationBytes =
                BufferSizeBytes(device, scene.GetBottomLevelBuffer()) + BufferSizeBytes(device, scene.GetTopLevelBuffer());
            struct FrameTextureRow {
                vesta::render::RenderGraphPassTiming::ResourceAccess access;
                uint32_t readers{ 0 };
                uint32_t writers{ 0 };
            };
            std::vector<FrameTextureRow> frameTextures;
            auto addFrameTexture = [&](const vesta::render::RenderGraphPassTiming::ResourceAccess& access, bool writer) {
                auto existing = std::find_if(frameTextures.begin(), frameTextures.end(), [&](const FrameTextureRow& row) {
                    return row.access.name == access.name;
                });
                if (existing == frameTextures.end()) {
                    FrameTextureRow row{};
                    row.access = access;
                    row.readers = writer ? 0u : 1u;
                    row.writers = writer ? 1u : 0u;
                    frameTextures.push_back(std::move(row));
                    return;
                }
                existing->access = access;
                existing->readers += writer ? 0u : 1u;
                existing->writers += writer ? 1u : 0u;
            };
            for (const auto& timing : graphTimings) {
                for (const auto& input : timing.inputs) {
                    addFrameTexture(input, false);
                }
                for (const auto& output : timing.outputs) {
                    addFrameTexture(output, true);
                }
            }
            std::sort(frameTextures.begin(), frameTextures.end(), [](const FrameTextureRow& lhs, const FrameTextureRow& rhs) {
                return lhs.access.name < rhs.access.name;
            });
            if (_frameTexturePreviewDescriptors.size() != frameTextures.size()) {
                for (VkDescriptorSet descriptor : _frameTexturePreviewDescriptors) {
                    if (descriptor != VK_NULL_HANDLE) {
                        ImGui_ImplVulkan_RemoveTexture(descriptor);
                    }
                }
                _frameTexturePreviewDescriptors.assign(frameTextures.size(), VK_NULL_HANDLE);
                _frameTexturePreviewImages.assign(frameTextures.size(), {});
                _selectedFrameTexturePreviewIndex = 0;
            }
            const auto iblStats = _renderer.GetIblStats();
            const uint64_t externalEnvironmentBytes =
                iblStats.environmentMapUploaded && iblStats.sourceWidth > 0u && iblStats.sourceHeight > 0u
                ? static_cast<uint64_t>(iblStats.sourceWidth) * iblStats.sourceHeight * 4u * sizeof(float)
                : 0u;
            const uint64_t environmentCubemapBytes =
                iblStats.environmentCubemapAvailable ? iblStats.estimatedEnvironmentCubemapBytes : 0u;
            const uint64_t liveIblTextureBytes =
                externalEnvironmentBytes
                + environmentCubemapBytes
                + (iblStats.diffuseIrradianceAvailable ? iblStats.estimatedDiffuseBytes : 0u)
                + (iblStats.specularPrefilterAvailable ? iblStats.estimatedSpecularBytes : 0u)
                + (iblStats.brdfLutAvailable ? iblStats.estimatedBrdfLutBytes : 0u);
            struct EngineTextureRow {
                std::string name;
                vesta::render::ImageHandle image;
                std::string resolution;
                std::string format;
                std::string usage;
                uint64_t memoryBytes{ 0 };
                uint32_t bindlessIndex{ vesta::render::kInvalidResourceIndex };
                std::string state;
                bool previewable{ false };
            };
            std::vector<EngineTextureRow> engineTextures;
            engineTextures.push_back(EngineTextureRow{
                .name = "External Environment Equirect",
                .image = _renderer.GetExternalEnvironmentImage(),
                .resolution = iblStats.environmentMapUploaded
                    ? fmt::format("{}x{}", iblStats.sourceWidth, iblStats.sourceHeight)
                    : "inactive",
                .format = "RGBA32F",
                .usage = "sampled environment",
                .memoryBytes = externalEnvironmentBytes,
                .bindlessIndex = _renderer.GetEnvironmentSampledImageIndex(),
                .state = iblStats.environmentMapUploaded ? "live" : "procedural fallback",
                .previewable = static_cast<bool>(_renderer.GetExternalEnvironmentImage()),
            });
            engineTextures.push_back(EngineTextureRow{
                .name = "Environment Cubemap",
                .image = _renderer.GetIblEnvironmentCubemapImage(),
                .resolution = iblStats.environmentCubemapAvailable
                    ? fmt::format("{}x{}x6 cube", iblStats.environmentCubemapResolution, iblStats.environmentCubemapResolution)
                    : fmt::format("{}x{}x6", iblStats.environmentCubemapResolution, iblStats.environmentCubemapResolution),
                .format = "RGBA32F",
                .usage = "sampled cubemap conversion image",
                .memoryBytes = iblStats.estimatedEnvironmentCubemapBytes,
                .bindlessIndex = _renderer.GetIblEnvironmentCubemapSampledImageIndex(),
                .state = iblStats.environmentCubemapAvailable ? "live" : "staged",
                .previewable = false,
            });
            engineTextures.push_back(EngineTextureRow{
                .name = "IBL BRDF LUT",
                .image = _renderer.GetIblBrdfLutImage(),
                .resolution = fmt::format("{}x{}", iblStats.brdfLutResolution, iblStats.brdfLutResolution),
                .format = "RG32F",
                .usage = "sampled specular IBL",
                .memoryBytes = iblStats.estimatedBrdfLutBytes,
                .bindlessIndex = _renderer.GetIblBrdfLutSampledImageIndex(),
                .state = iblStats.brdfLutAvailable ? "live" : "staged",
                .previewable = static_cast<bool>(_renderer.GetIblBrdfLutImage()),
            });
            engineTextures.push_back(EngineTextureRow{
                .name = iblStats.diffuseIrradianceAvailable ? "Diffuse Irradiance Equirect" : "Diffuse Irradiance Cubemap",
                .image = _renderer.GetIblDiffuseIrradianceImage(),
                .resolution = iblStats.diffuseIrradianceAvailable ? "64x32 equirect" : fmt::format("{}x{}x6", iblStats.diffuseCubemapResolution, iblStats.diffuseCubemapResolution),
                .format = iblStats.diffuseIrradianceAvailable ? "RGBA32F" : "RGBA16F",
                .usage = iblStats.diffuseIrradianceAvailable ? "sampled diffuse IBL" : "future irradiance convolution",
                .memoryBytes = iblStats.estimatedDiffuseBytes,
                .bindlessIndex = _renderer.GetIblDiffuseIrradianceSampledImageIndex(),
                .state = iblStats.diffuseIrradianceAvailable ? "live" : (iblStats.diffuseBackendAvailable ? "staged" : "backend required"),
                .previewable = static_cast<bool>(_renderer.GetIblDiffuseIrradianceImage()),
            });
            engineTextures.push_back(EngineTextureRow{
                .name = "Specular Prefilter Cubemap",
                .image = _renderer.GetIblSpecularPrefilterImage(),
                .resolution = fmt::format("{}x{}x6 mips {}", iblStats.specularCubemapResolution, iblStats.specularCubemapResolution, iblStats.specularMipCount),
                .format = iblStats.specularPrefilterAvailable ? "RGBA32F" : "RGBA16F",
                .usage = iblStats.specularPrefilterAvailable ? "sampled specular IBL cube" : "future prefiltered IBL",
                .memoryBytes = iblStats.estimatedSpecularBytes,
                .bindlessIndex = _renderer.GetIblSpecularPrefilterCubeSampledImageIndex(),
                .state = iblStats.specularPrefilterAvailable ? "live" : "staged",
                .previewable = false,
            });
            if (_engineTexturePreviewDescriptors.size() != engineTextures.size()) {
                for (VkDescriptorSet descriptor : _engineTexturePreviewDescriptors) {
                    if (descriptor != VK_NULL_HANDLE) {
                        ImGui_ImplVulkan_RemoveTexture(descriptor);
                    }
                }
                _engineTexturePreviewDescriptors.assign(engineTextures.size(), VK_NULL_HANDLE);
                _engineTexturePreviewImages.assign(engineTextures.size(), {});
                _selectedEngineTexturePreviewIndex = 0;
            }

            if (ImGui::BeginTabBar("ResourceTabs")) {
                if (ImGui::BeginTabItem("Summary")) {
                    if (ImGui::BeginTable("ResourceSummaryTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Category");
                        ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                        ImGui::TableHeadersRow();
                        auto row = [](const char* label, double mib) {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::TextUnformatted(label);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%.2f MiB", mib);
                        };
                        row("Scene Buffers", MiB(sceneBufferBytes));
                        row("Textures Resident", MiB(residentTextureBytes));
                        row("Textures CPU/Source", MiB(textureBytes));
                        row("Engine IBL Runtime", MiB(liveIblTextureBytes));
                        row("Acceleration Structures", MiB(accelerationBytes));
                        row("Total GPU Tracked", MiB(sceneBufferBytes + residentTextureBytes + accelerationBytes + liveIblTextureBytes));
                        ImGui::EndTable();
                    }
                    ImGui::Text("Textures %u / %zu resident", scene.GetResidentTextureCount(), scene.GetTextures().size());
                    ImGui::Text("IBL runtime %.2f MiB  staged %.2f MiB",
                        MiB(liveIblTextureBytes),
                        MiB(iblStats.estimatedEnvironmentCubemapBytes + iblStats.estimatedDiffuseBytes + iblStats.estimatedSpecularBytes));
                    ImGui::Text("Dedicated VRAM %u MiB", device.GetDedicatedVideoMemoryMiB());
                    ImGui::Text("Upload Last %.2f MiB  Pending %.2f MiB",
                        MiB(static_cast<uint64_t>(device.GetUploadBatchStats().lastSubmittedBytes)),
                        MiB(static_cast<uint64_t>(device.GetUploadBatchStats().pendingBytes)));
                    const auto bindlessStats = device.GetBindlessStats();
                    ImGui::SeparatorText("Bindless Heap");
                    auto bindlessUsageRow = [](const char* label, uint32_t used, uint32_t capacity) {
                        const float fraction = capacity > 0u ? static_cast<float>(used) / static_cast<float>(capacity) : 0.0f;
                        ImGui::Text("%s %u / %u", label, used, capacity);
                        ImGui::SameLine(190.0f);
                        ImGui::ProgressBar(fraction, ImVec2(-1.0f, 0.0f));
                    };
                    bindlessUsageRow("Sampled Images", bindlessStats.sampledImagesUsed, bindlessStats.sampledImagesCapacity);
                    bindlessUsageRow("Sampled Cubes", bindlessStats.sampledCubeImagesUsed, bindlessStats.sampledCubeImagesCapacity);
                    bindlessUsageRow("Storage Images", bindlessStats.storageImagesUsed, bindlessStats.storageImagesCapacity);
                    bindlessUsageRow("Storage Buffers", bindlessStats.storageBuffersUsed, bindlessStats.storageBuffersCapacity);
                    if (bindlessStats.sampledImagesUsed > bindlessStats.sampledImagesCapacity * 8u / 10u
                        || bindlessStats.sampledCubeImagesUsed > bindlessStats.sampledCubeImagesCapacity * 8u / 10u
                        || bindlessStats.storageImagesUsed > bindlessStats.storageImagesCapacity * 8u / 10u
                        || bindlessStats.storageBuffersUsed > bindlessStats.storageBuffersCapacity * 8u / 10u) {
                        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.18f, 1.0f), "Warning: bindless heap usage is above 80%%.");
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Frame Textures")) {
                    if (frameTextures.empty()) {
                        ImGui::TextDisabled("No render graph frame textures are available yet.");
                    } else {
                        _selectedFrameTexturePreviewIndex =
                            std::min(_selectedFrameTexturePreviewIndex, frameTextures.size() - 1u);
                    }
                    if (ImGui::BeginTable("FrameTextureTable",
                            8,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                            ImVec2(0.0f, 180.0f))) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                        ImGui::TableSetupColumn("Last Usage", ImGuiTableColumnFlags_WidthFixed, 108.0f);
                        ImGui::TableSetupColumn("Format", ImGuiTableColumnFlags_WidthFixed, 82.0f);
                        ImGui::TableSetupColumn("Resolution", ImGuiTableColumnFlags_WidthFixed, 96.0f);
                        ImGui::TableSetupColumn("Reads", ImGuiTableColumnFlags_WidthFixed, 44.0f);
                        ImGui::TableSetupColumn("Writes", ImGuiTableColumnFlags_WidthFixed, 48.0f);
                        ImGui::TableSetupColumn("Scale", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                        ImGui::TableHeadersRow();
                        for (size_t textureIndex = 0; textureIndex < frameTextures.size(); ++textureIndex) {
                            const FrameTextureRow& row = frameTextures[textureIndex];
                            ImGui::PushID(static_cast<int>(textureIndex));
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            if (ImGui::Selectable(row.access.name.c_str(),
                                    _selectedFrameTexturePreviewIndex == textureIndex,
                                    ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap)) {
                                _selectedFrameTexturePreviewIndex = textureIndex;
                            }
                            ImGui::TableSetColumnIndex(1);
                            const bool previewable = IsPreviewableFrameTexture(device, row.access);
                            if (textureIndex < _frameTexturePreviewImages.size()
                                && _frameTexturePreviewImages[textureIndex] != row.access.image) {
                                if (_frameTexturePreviewDescriptors[textureIndex] != VK_NULL_HANDLE) {
                                    ImGui_ImplVulkan_RemoveTexture(_frameTexturePreviewDescriptors[textureIndex]);
                                    _frameTexturePreviewDescriptors[textureIndex] = VK_NULL_HANDLE;
                                }
                                _frameTexturePreviewImages[textureIndex] = row.access.image;
                            }
                            if (previewable && _frameTexturePreviewDescriptors[textureIndex] == VK_NULL_HANDLE) {
                                _frameTexturePreviewDescriptors[textureIndex] = ImGui_ImplVulkan_AddTexture(device.GetDefaultSampler(),
                                    device.GetImageView(row.access.image),
                                    PreviewLayoutForResourceUsage(row.access.usage));
                                _frameTexturePreviewImages[textureIndex] = row.access.image;
                            }
                            if (previewable && _frameTexturePreviewDescriptors[textureIndex] != VK_NULL_HANDLE) {
                                ImGui::Image(reinterpret_cast<ImTextureID>(_frameTexturePreviewDescriptors[textureIndex]),
                                    ImVec2(48.0f, 48.0f));
                            } else if (row.access.imported) {
                                ImGui::TextDisabled("import");
                            } else {
                                ImGui::TextDisabled("-");
                            }
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TextUnformatted(ResourceUsageLabel(row.access.usage));
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TextUnformatted(VkFormatLabel(row.access.format));
                            ImGui::TableSetColumnIndex(4);
                            ImGui::Text("%ux%u", row.access.extent.width, row.access.extent.height);
                            ImGui::TableSetColumnIndex(5);
                            ImGui::Text("%u", row.readers);
                            ImGui::TableSetColumnIndex(6);
                            ImGui::Text("%u", row.writers);
                            ImGui::TableSetColumnIndex(7);
                            const VkExtent2D swapchainExtent = device.GetSwapchainExtent();
                            const bool fullRes = row.access.extent.width == swapchainExtent.width
                                && row.access.extent.height == swapchainExtent.height;
                            ImGui::TextUnformatted(fullRes ? "full-res" : "scaled");
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                    if (!frameTextures.empty()) {
                        const FrameTextureRow& selected = frameTextures[_selectedFrameTexturePreviewIndex];
                        ImGui::SeparatorText("Selected Frame Texture");
                        const bool previewable = IsPreviewableFrameTexture(device, selected.access);
                        if (previewable && _frameTexturePreviewDescriptors[_selectedFrameTexturePreviewIndex] != VK_NULL_HANDLE) {
                            ImGui::Image(reinterpret_cast<ImTextureID>(
                                             _frameTexturePreviewDescriptors[_selectedFrameTexturePreviewIndex]),
                                FitPreviewSize(selected.access.extent, 192.0f));
                        } else {
                            ImGui::BeginDisabled();
                            ImGui::Button("No Preview", ImVec2(160.0f, 96.0f));
                            ImGui::EndDisabled();
                        }
                        ImGui::SameLine();
                        ImGui::BeginGroup();
                        ImGui::Text("Name: %s", selected.access.name.c_str());
                        ImGui::Text("Usage: %s", ResourceUsageLabel(selected.access.usage));
                        ImGui::Text("Format: %s", VkFormatLabel(selected.access.format));
                        ImGui::Text("Resolution: %ux%u", selected.access.extent.width, selected.access.extent.height);
                        ImGui::Text("Reads/Writes: %u/%u", selected.readers, selected.writers);
                        ImGui::Text("Image Handle: %s",
                            selected.access.image ? std::to_string(selected.access.image.index).c_str() : "none");
                        if (!previewable) {
                            ImGui::TextDisabled("Preview requires a non-imported sampled image in a readable layout.");
                        }
                        ImGui::EndGroup();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Engine Textures")) {
                    if (engineTextures.empty()) {
                        ImGui::TextDisabled("No engine-owned textures are registered.");
                    } else {
                        _selectedEngineTexturePreviewIndex =
                            std::min(_selectedEngineTexturePreviewIndex, engineTextures.size() - 1u);
                    }
                    if (ImGui::BeginTable("EngineTextureTable",
                            8,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                            ImVec2(0.0f, 176.0f))) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                        ImGui::TableSetupColumn("Resolution", ImGuiTableColumnFlags_WidthFixed, 110.0f);
                        ImGui::TableSetupColumn("Format", ImGuiTableColumnFlags_WidthFixed, 76.0f);
                        ImGui::TableSetupColumn("Usage");
                        ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Bindless", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 112.0f);
                        ImGui::TableHeadersRow();
                        for (size_t textureIndex = 0; textureIndex < engineTextures.size(); ++textureIndex) {
                            const EngineTextureRow& row = engineTextures[textureIndex];
                            ImGui::PushID(static_cast<int>(textureIndex));
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            if (ImGui::Selectable(row.name.c_str(),
                                    _selectedEngineTexturePreviewIndex == textureIndex,
                                    ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap)) {
                                _selectedEngineTexturePreviewIndex = textureIndex;
                            }
                            ImGui::TableSetColumnIndex(1);
                            if (textureIndex < _engineTexturePreviewImages.size()
                                && _engineTexturePreviewImages[textureIndex] != row.image) {
                                if (_engineTexturePreviewDescriptors[textureIndex] != VK_NULL_HANDLE) {
                                    ImGui_ImplVulkan_RemoveTexture(_engineTexturePreviewDescriptors[textureIndex]);
                                    _engineTexturePreviewDescriptors[textureIndex] = VK_NULL_HANDLE;
                                }
                                _engineTexturePreviewImages[textureIndex] = row.image;
                            }
                            if (row.previewable && _engineTexturePreviewDescriptors[textureIndex] == VK_NULL_HANDLE) {
                                _engineTexturePreviewDescriptors[textureIndex] =
                                    ImGui_ImplVulkan_AddTexture(device.GetDefaultSampler(),
                                        device.GetImageView(row.image),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                                _engineTexturePreviewImages[textureIndex] = row.image;
                            }
                            if (row.previewable && _engineTexturePreviewDescriptors[textureIndex] != VK_NULL_HANDLE) {
                                ImGui::Image(reinterpret_cast<ImTextureID>(_engineTexturePreviewDescriptors[textureIndex]),
                                    ImVec2(48.0f, 48.0f));
                            } else {
                                ImGui::TextDisabled("-");
                            }
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TextUnformatted(row.resolution.c_str());
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TextUnformatted(row.format.c_str());
                            ImGui::TableSetColumnIndex(4);
                            ImGui::TextUnformatted(row.usage.c_str());
                            ImGui::TableSetColumnIndex(5);
                            ImGui::Text("%.2f MiB", MiB(row.memoryBytes));
                            ImGui::TableSetColumnIndex(6);
                            if (row.bindlessIndex != vesta::render::kInvalidResourceIndex) {
                                ImGui::Text("%u", row.bindlessIndex);
                            } else {
                                ImGui::TextUnformatted("-");
                            }
                            ImGui::TableSetColumnIndex(7);
                            ImGui::TextUnformatted(row.state.c_str());
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                    if (!engineTextures.empty()) {
                        const EngineTextureRow& selected = engineTextures[_selectedEngineTexturePreviewIndex];
                        ImGui::SeparatorText("Selected Engine Texture");
                        if (selected.previewable
                            && _engineTexturePreviewDescriptors[_selectedEngineTexturePreviewIndex] != VK_NULL_HANDLE) {
                            ImGui::Image(reinterpret_cast<ImTextureID>(
                                             _engineTexturePreviewDescriptors[_selectedEngineTexturePreviewIndex]),
                                ImVec2(192.0f, 192.0f));
                        } else {
                            ImGui::BeginDisabled();
                            ImGui::Button("No Preview", ImVec2(160.0f, 96.0f));
                            ImGui::EndDisabled();
                        }
                        ImGui::SameLine();
                        ImGui::BeginGroup();
                        ImGui::Text("Name: %s", selected.name.c_str());
                        ImGui::Text("Resolution: %s", selected.resolution.c_str());
                        ImGui::Text("Format: %s", selected.format.c_str());
                        ImGui::Text("Usage: %s", selected.usage.c_str());
                        ImGui::Text("Memory: %.2f MiB", MiB(selected.memoryBytes));
                        if (selected.bindlessIndex != vesta::render::kInvalidResourceIndex) {
                            ImGui::Text("Bindless: %u", selected.bindlessIndex);
                        } else {
                            ImGui::TextDisabled("Bindless: unavailable");
                        }
                        ImGui::Text("State: %s", selected.state.c_str());
                        if (selected.image) {
                            ImGui::Text("Image Handle: %u", selected.image.index);
                        } else {
                            ImGui::TextDisabled("Image Handle: none");
                        }
                        ImGui::EndGroup();
                    }
                    ImGui::SeparatorText("IBL Status");
                    ImGui::Text("Source: %s", iblStats.externalSourceAvailable ? "External HDRI/image" : "Procedural preset");
                    if (iblStats.externalSourceAvailable) {
                        ImGui::Text("%ux%u  channels %u  %s",
                            iblStats.sourceWidth,
                            iblStats.sourceHeight,
                            iblStats.sourceChannels,
                            iblStats.sourceIsHdr ? "HDR" : "LDR");
                    }
                    ImGui::Text("Diffuse irradiance: %s", iblStats.diffuseIrradianceAvailable ? "live equirect" : (iblStats.diffuseBackendAvailable ? "staged" : "backend required"));
                    ImGui::Text("Specular prefilter: %s", iblStats.specularPrefilterAvailable ? "live cube mips" : "staged");
                    ImGui::Text("BRDF LUT: %s", iblStats.brdfLutAvailable ? "live" : "staged");
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Textures")) {
                    if (ImGui::BeginTable("TextureTable", 9, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 82.0f);
                        ImGui::TableSetupColumn("Format", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("Mips", ImGuiTableColumnFlags_WidthFixed, 44.0f);
                        ImGui::TableSetupColumn("Usage", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                        ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Bindless", ImGuiTableColumnFlags_WidthFixed, 68.0f);
                        ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, 52.0f);
                        ImGui::TableHeadersRow();
                        const auto& textures = scene.GetTextures();
                        for (size_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
                            const auto& texture = textures[textureIndex];
                            ImGui::PushID(static_cast<int>(textureIndex));
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            const char* textureName = texture.name.empty() ? "(texture)" : texture.name.c_str();
                            if (ImGui::Selectable(textureName,
                                    _selectedTexturePreviewIndex == textureIndex,
                                    ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap)) {
                                _selectedTexturePreviewIndex = textureIndex;
                            }
                            ImGui::TableSetColumnIndex(1);
                            if (scene.HasResidentTexture(textureIndex)) {
                                if (_texturePreviewDescriptors[textureIndex] == VK_NULL_HANDLE) {
                                    const vesta::render::ImageHandle image = scene.GetTextureImage(textureIndex);
                                    if (image) {
                                        _texturePreviewDescriptors[textureIndex] = ImGui_ImplVulkan_AddTexture(device.GetDefaultSampler(),
                                            device.GetImageView(image),
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                                    }
                                }
                                if (_texturePreviewDescriptors[textureIndex] != VK_NULL_HANDLE) {
                                    ImGui::Image(reinterpret_cast<ImTextureID>(_texturePreviewDescriptors[textureIndex]),
                                        ImVec2(48.0f, 48.0f));
                                } else {
                                    ImGui::TextUnformatted("-");
                                }
                            } else {
                                ImGui::TextUnformatted("streaming");
                            }
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%ux%u", texture.width, texture.height);
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TextUnformatted(texture.srgb ? "RGBA8_sRGB" : "RGBA8");
                            ImGui::TableSetColumnIndex(4);
                            uint32_t residentMipCount = 0;
                            if (scene.HasResidentTexture(textureIndex)) {
                                const vesta::render::ImageHandle image = scene.GetTextureImage(textureIndex);
                                if (image) {
                                    residentMipCount = device.GetImageResource(image).desc.mipLevels;
                                }
                            }
                            ImGui::Text("%u/%u", std::max(residentMipCount, 1u), FullMipCount(texture.width, texture.height));
                            ImGui::TableSetColumnIndex(5);
                            const std::string usage = TextureSemanticLabel(scene, textureIndex);
                            ImGui::TextUnformatted(usage.c_str());
                            ImGui::TableSetColumnIndex(6);
                            ImGui::Text("%.2f MiB", MiB(TextureAssetBytes(texture)));
                            ImGui::TableSetColumnIndex(7);
                            if (scene.HasResidentTexture(textureIndex)) {
                                ImGui::Text("%u", scene.GetTextureBindlessIndex(textureIndex));
                            } else {
                                ImGui::TextUnformatted("-");
                            }
                            ImGui::TableSetColumnIndex(8);
                            ImGui::Text("%s", scene.HasResidentTexture(textureIndex) ? "Yes" : "No");
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                    const auto& textures = scene.GetTextures();
                    if (!textures.empty()) {
                        _selectedTexturePreviewIndex = std::min(_selectedTexturePreviewIndex, textures.size() - 1u);
                        const size_t selectedTexture = _selectedTexturePreviewIndex;
                        const auto& texture = textures[selectedTexture];
                        ImGui::SeparatorText("Selected Texture Preview");
                        if (scene.HasResidentTexture(selectedTexture)) {
                            if (_texturePreviewDescriptors[selectedTexture] == VK_NULL_HANDLE) {
                                const vesta::render::ImageHandle image = scene.GetTextureImage(selectedTexture);
                                if (image) {
                                    _texturePreviewDescriptors[selectedTexture] = ImGui_ImplVulkan_AddTexture(device.GetDefaultSampler(),
                                        device.GetImageView(image),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                                }
                            }
                            if (_texturePreviewDescriptors[selectedTexture] != VK_NULL_HANDLE) {
                                const float maxPreviewSize = 192.0f;
                                const float width = static_cast<float>(std::max(texture.width, 1u));
                                const float height = static_cast<float>(std::max(texture.height, 1u));
                                ImVec2 previewSize(maxPreviewSize, maxPreviewSize);
                                if (width > height) {
                                    previewSize.y = maxPreviewSize * (height / width);
                                } else {
                                    previewSize.x = maxPreviewSize * (width / height);
                                }
                                ImGui::Image(reinterpret_cast<ImTextureID>(_texturePreviewDescriptors[selectedTexture]), previewSize);
                            }
                        } else {
                            ImGui::TextDisabled("Texture is not resident yet.");
                        }
                        ImGui::SameLine();
                        ImGui::BeginGroup();
                        ImGui::Text("Name: %s", texture.name.empty() ? "(texture)" : texture.name.c_str());
                        ImGui::Text("Size: %ux%u", texture.width, texture.height);
                        ImGui::Text("Format: %s", texture.srgb ? "RGBA8_sRGB" : "RGBA8");
                        ImGui::Text("Usage: %s", TextureSemanticLabel(scene, selectedTexture).c_str());
                        ImGui::Text("Memory: %.2f MiB", MiB(TextureAssetBytes(texture)));
                        if (scene.HasResidentTexture(selectedTexture)) {
                            const vesta::render::ImageHandle image = scene.GetTextureImage(selectedTexture);
                            const uint32_t residentMipCount = image ? device.GetImageResource(image).desc.mipLevels : 1u;
                            ImGui::Text("Mips: %u/%u", residentMipCount, FullMipCount(texture.width, texture.height));
                            ImGui::Text("Bindless: %u", scene.GetTextureBindlessIndex(selectedTexture));
                        } else {
                            ImGui::TextDisabled("Mips: pending");
                            ImGui::TextDisabled("Bindless: pending");
                        }
                        ImGui::EndGroup();
                    } else {
                        ImGui::TextDisabled("No scene textures are loaded.");
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Buffers")) {
                    struct BufferInspectorEntry {
                        const char* name;
                        vesta::render::BufferHandle handle;
                        uint64_t logicalBytes;
                    };
                    const auto ddgiStats = _renderer.GetDdgiStats();
                    const auto voxelGiStats = _renderer.GetVoxelGiStats();
                    const auto restirStats = _renderer.GetRestirStats();
                    const auto meshletStats = _renderer.GetMeshletClusterStats();
                    const auto restirPerBufferBytes = [&](uint64_t totalBytes) {
                        return restirStats.temporalReuse ? totalBytes / 2u : totalBytes;
                    };
                    std::vector<BufferInspectorEntry> bufferEntries{
                        BufferInspectorEntry{ "Vertex Buffer", scene.GetVertexBuffer(), vertexBytes },
                        BufferInspectorEntry{ "Index Buffer", scene.GetIndexBuffer(), indexBytes },
                        BufferInspectorEntry{ "Material Buffer", scene.GetMaterialBuffer(), materialBytes },
                        BufferInspectorEntry{ "Triangle Buffer", scene.GetTriangleBuffer(), triangleBytes },
                        BufferInspectorEntry{ "Emissive Triangle Buffer", scene.GetEmissiveTriangleBuffer(), emissiveBytes },
                        BufferInspectorEntry{ "Gaussian Position/Covariance/SH", scene.GetGaussianBuffer(), gaussianBytes },
                        BufferInspectorEntry{ "Meshlet Visibility Cluster Storage", _renderer.GetMeshletVisibilityBuffer(), meshletStats.estimatedVisibilityBytes },
                    };
                    bufferEntries.push_back(BufferInspectorEntry{
                        "DDGI Irradiance Probe Storage",
                        _renderer.GetDdgiIrradianceBuffer(),
                        ddgiStats.estimatedIrradianceBytes,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "DDGI Visibility Probe Storage",
                        _renderer.GetDdgiVisibilityBuffer(),
                        ddgiStats.estimatedVisibilityBytes,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "DDGI Relocation Probe Storage",
                        _renderer.GetDdgiRelocationBuffer(),
                        ddgiStats.estimatedRelocationBytes,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "Voxel GI Radiance Volume Storage",
                        _renderer.GetVoxelGiRadianceBuffer(),
                        voxelGiStats.estimatedRadianceBytes,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "Voxel GI Occupancy Volume Storage",
                        _renderer.GetVoxelGiOccupancyBuffer(),
                        voxelGiStats.estimatedOccupancyBytes,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR DI Current Reservoir Storage",
                        _renderer.GetRestirReservoirBuffer(),
                        restirPerBufferBytes(restirStats.estimatedDiReservoirBytes),
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR DI History Reservoir Storage",
                        _renderer.GetRestirHistoryReservoirBuffer(),
                        restirStats.temporalReuse ? restirPerBufferBytes(restirStats.estimatedDiReservoirBytes) : 0u,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR GI Current Reservoir Storage",
                        _renderer.GetRestirGiReservoirBuffer(),
                        restirPerBufferBytes(restirStats.estimatedGiReservoirBytes),
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR GI History Reservoir Storage",
                        _renderer.GetRestirGiHistoryReservoirBuffer(),
                        restirStats.temporalReuse ? restirPerBufferBytes(restirStats.estimatedGiReservoirBytes) : 0u,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR PT Current Reservoir Storage",
                        _renderer.GetRestirPtReservoirBuffer(),
                        restirPerBufferBytes(restirStats.estimatedPtReservoirBytes),
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR PT History Reservoir Storage",
                        _renderer.GetRestirPtHistoryReservoirBuffer(),
                        restirStats.temporalReuse ? restirPerBufferBytes(restirStats.estimatedPtReservoirBytes) : 0u,
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR PT Current Path State Storage",
                        _renderer.GetRestirPtPathStateBuffer(),
                        restirPerBufferBytes(restirStats.estimatedPtPathStateBytes),
                    });
                    bufferEntries.push_back(BufferInspectorEntry{
                        "ReSTIR PT History Path State Storage",
                        _renderer.GetRestirPtHistoryPathStateBuffer(),
                        restirStats.temporalReuse ? restirPerBufferBytes(restirStats.estimatedPtPathStateBytes) : 0u,
                    });
                    if (ImGui::BeginTable("BufferTable", 8, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 74.0f);
                        ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Logical", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Group", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                        ImGui::TableSetupColumn("Usage");
                        ImGui::TableSetupColumn("Bindless", ImGuiTableColumnFlags_WidthFixed, 68.0f);
                        ImGui::TableSetupColumn("Handle", ImGuiTableColumnFlags_WidthFixed, 58.0f);
                        ImGui::TableHeadersRow();
                        for (size_t bufferIndex = 0; bufferIndex < bufferEntries.size(); ++bufferIndex) {
                            const BufferInspectorEntry& entry = bufferEntries[bufferIndex];
                            ImGui::PushID(static_cast<int>(bufferIndex));
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            if (ImGui::Selectable(entry.name,
                                    _selectedBufferInspectorIndex == static_cast<int>(bufferIndex),
                                    ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap)) {
                                _selectedBufferInspectorIndex = static_cast<int>(bufferIndex);
                            }
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%s", entry.handle ? "Resident" : "Missing");
                            ImGui::TableSetColumnIndex(2);
                            if (entry.handle) {
                                const auto& buffer = device.GetBufferResource(entry.handle);
                                ImGui::Text("%.2f MiB", MiB(static_cast<uint64_t>(buffer.desc.size)));
                            } else {
                                ImGui::TextUnformatted("-");
                            }
                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%.2f MiB", MiB(entry.logicalBytes));
                            ImGui::TableSetColumnIndex(4);
                            if (entry.handle) {
                                const auto& buffer = device.GetBufferResource(entry.handle);
                                ImGui::TextUnformatted(BufferGroupLabel(entry.name, buffer.desc.usage).c_str());
                            } else {
                                ImGui::TextUnformatted("-");
                            }
                            ImGui::TableSetColumnIndex(5);
                            if (entry.handle) {
                                const auto& buffer = device.GetBufferResource(entry.handle);
                                ImGui::TextUnformatted(BufferUsageLabel(buffer.desc.usage).c_str());
                            } else {
                                ImGui::TextUnformatted("-");
                            }
                            ImGui::TableSetColumnIndex(6);
                            if (entry.handle) {
                                const auto& buffer = device.GetBufferResource(entry.handle);
                                if (buffer.bindless.storageBuffer != vesta::render::kInvalidResourceIndex) {
                                    ImGui::Text("%u", buffer.bindless.storageBuffer);
                                } else {
                                    ImGui::TextUnformatted("-");
                                }
                            } else {
                                ImGui::TextUnformatted("-");
                            }
                            ImGui::TableSetColumnIndex(7);
                            ImGui::Text("%u", entry.handle ? entry.handle.index : 0u);
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                    _selectedBufferInspectorIndex = std::clamp(_selectedBufferInspectorIndex, 0, static_cast<int>(bufferEntries.size() - 1u));
                    const BufferInspectorEntry& selectedBuffer = bufferEntries[static_cast<size_t>(_selectedBufferInspectorIndex)];
                    ImGui::SeparatorText("Selected Buffer Inspector");
                    ImGui::Text("Name: %s", selectedBuffer.name);
                    ImGui::Text("State: %s", selectedBuffer.handle ? "Resident" : "Missing");
                    ImGui::Text("Logical: %.2f MiB", MiB(selectedBuffer.logicalBytes));
                    if (selectedBuffer.handle) {
                        const auto& buffer = device.GetBufferResource(selectedBuffer.handle);
                        ImGui::Text("GPU Allocation: %.2f MiB", MiB(static_cast<uint64_t>(buffer.desc.size)));
                        ImGui::Text("Group: %s", BufferGroupLabel(selectedBuffer.name, buffer.desc.usage).c_str());
                        ImGui::TextWrapped("Usage: %s", BufferUsageLabel(buffer.desc.usage).c_str());
                        ImGui::Text("Bindless Storage: %s",
                            buffer.bindless.storageBuffer != vesta::render::kInvalidResourceIndex ? "Resident" : "Not registered");
                        if (buffer.bindless.storageBuffer != vesta::render::kInvalidResourceIndex) {
                            ImGui::SameLine();
                            ImGui::Text("#%u", buffer.bindless.storageBuffer);
                        }
                        ImGui::Text("Handle Index: %u", selectedBuffer.handle.index);
                    } else {
                        ImGui::TextDisabled("No GPU allocation is available for this buffer.");
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Acceleration")) {
                    const uint64_t blasBytes = BufferSizeBytes(device, scene.GetBottomLevelBuffer());
                    const uint64_t tlasBytes = BufferSizeBytes(device, scene.GetTopLevelBuffer());
                    const bool rtSupported = device.IsRayTracingSupported();
                    const bool blasResident = scene.GetBottomLevelBuffer() && scene.GetBottomLevelBuildMs() > 0.0f;
                    const bool tlasResident = scene.HasRayTracingScene();
                    const uint32_t blasCount = blasResident ? 1u : 0u;
                    const uint32_t tlasCount = tlasResident ? 1u : 0u;
                    const uint32_t tlasInstanceCount = tlasResident ? 1u : 0u;
                    const auto& rtSupport = device.GetRayTracingSupport();
                    ImGui::Text("Ray Tracing Support %s", rtSupported ? "Available" : "Unavailable");
                    ImGui::Text("TLAS %s  BLAS %s", tlasResident ? "Resident" : "Missing", blasResident ? "Resident" : "Missing");
                    ImGui::Text("Build %.3f ms BLAS / %.3f ms TLAS", scene.GetBottomLevelBuildMs(), scene.GetTopLevelBuildMs());
                    ImGui::Text("Counts TLAS %u  BLAS %u  Instances %u", tlasCount, blasCount, tlasInstanceCount);
                    ImGui::Text("Primitives %zu triangles  Scene Objects %zu", scene.GetTriangles().size(), scene.GetObjects().size());
                    ImGui::Text("Memory %.2f MiB BLAS / %.2f MiB TLAS / %.2f MiB total",
                        MiB(blasBytes),
                        MiB(tlasBytes),
                        MiB(blasBytes + tlasBytes));
                    ImGui::Text("Build Mode Build-on-load  Update/Refit %s", settings.buildRayTracingStructuresOnLoad ? "manual rebuild" : "deferred");
                    ImGui::Text("Ray Query %s  Pipeline %s",
                        rtSupport.rayQueryFeatures.rayQuery == VK_TRUE ? "Yes" : "No",
                        rtSupport.rayTracingPipelineFeatures.rayTracingPipeline == VK_TRUE ? "Yes" : "No");
                    ImGui::Separator();
                    if (ImGui::BeginTable("AccelerationTable", 8, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 74.0f);
                        ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Logical", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Group", ImGuiTableColumnFlags_WidthFixed, 86.0f);
                        ImGui::TableSetupColumn("Usage");
                        ImGui::TableSetupColumn("Bindless", ImGuiTableColumnFlags_WidthFixed, 68.0f);
                        ImGui::TableSetupColumn("Handle", ImGuiTableColumnFlags_WidthFixed, 58.0f);
                        ImGui::TableHeadersRow();
                        DrawBufferResourceRow("BLAS Buffer", device, scene.GetBottomLevelBuffer());
                        DrawBufferResourceRow("TLAS Buffer", device, scene.GetTopLevelBuffer());
                        ImGui::EndTable();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Gaussian")) {
                    const uint64_t gaussianSourceBytes =
                        static_cast<uint64_t>(scene.GetGaussians().size()) * sizeof(vesta::scene::GaussianPrimitive);
                    const uint64_t gaussianGpuBytes = BufferSizeBytes(device, scene.GetGaussianBuffer());
                    const uint32_t totalGaussians = scene.GetGaussianCount();
                    const uint32_t projectedGaussians = _renderer.GetOfficialGaussianProjectedCount();
                    const uint32_t culledGaussians = projectedGaussians <= totalGaussians ? totalGaussians - projectedGaussians : 0u;
                    ImGui::Text("Position/Covariance/SH/Opacity buffer %s", scene.GetGaussianBuffer() ? "Resident" : "Missing");
                    ImGui::Text("Total %u  Projected %u  Culled %u", totalGaussians, projectedGaussians, culledGaussians);
                    ImGui::Text("Sort Keys %u duplicates  Padded %u",
                        _renderer.GetOfficialGaussianDuplicateCount(),
                        _renderer.GetOfficialGaussianPaddedDuplicateCount());
                    ImGui::Text("Tile/Bin Count %u  Avg Tiles %.2f",
                        _renderer.GetOfficialGaussianTileCount(),
                        _renderer.GetOfficialGaussianAverageTilesTouched());
                    ImGui::Text("Memory Source %.2f MiB  GPU %.2f MiB", MiB(gaussianSourceBytes), MiB(gaussianGpuBytes));
                    ImGui::Text("Stages %.3f preprocess  %.3f duplicate  %.3f sort  %.3f raster ms",
                        _renderer.GetOfficialGaussianPreprocessMs(),
                        _renderer.GetOfficialGaussianDuplicateMs(),
                        _renderer.GetOfficialGaussianSortMs(),
                        _renderer.GetOfficialGaussianRasterMs());
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
    }

    if (_showLogConsolePanel) {
        ImGui::SetNextWindowPos(ImVec2(18.0f, 612.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(560.0f, 300.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Log / Validation / Error Console", &_showLogConsolePanel, ImGuiWindowFlags_NoSavedSettings)) {
            auto classifyLogLine = [](const std::string& line) {
                struct Classification {
                    bool perf{ false };
                    bool validation{ false };
                    bool resource{ false };
                    bool shader{ false };
                    bool device{ false };
                    bool error{ false };
                };

                std::string lower = line;
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                return Classification{
                    line.find("[PERF]") != std::string::npos,
                    line.find("[VALIDATION]") != std::string::npos,
                    line.find("[RESOURCE]") != std::string::npos,
                    lower.find("shader") != std::string::npos || lower.find(".spv") != std::string::npos,
                    lower.find("device lost") != std::string::npos || lower.find("vk_error_device_lost") != std::string::npos ||
                        lower.find("crash") != std::string::npos,
                    lower.find("failed") != std::string::npos || line.find("[ERROR]") != std::string::npos,
                };
            };
            auto lineMatchesFilters = [&](const std::string& line) {
                const auto cls = classifyLogLine(line);
                const bool isInfo = !cls.perf && !cls.validation && !cls.resource && !cls.shader && !cls.device && !cls.error;
                if ((isInfo && !_logShowInfo) || (cls.perf && !_logShowPerformance) || (cls.validation && !_logShowValidation) ||
                    (cls.resource && !_logShowResources) || ((cls.shader || cls.device || cls.error) && !_logShowErrors)) {
                    return false;
                }
                const std::string_view filter{ _logFilterText.data() };
                return filter.empty() || line.find(filter) != std::string::npos;
            };

            int perfWarnings = 0;
            int validationWarnings = 0;
            int resourceWarnings = 0;
            int shaderMessages = 0;
            int deviceMessages = 0;
            int errors = 0;
            int visibleLines = 0;
            std::vector<ShaderCompilerDiagnostic> shaderDiagnostics;
            for (const std::string& line : _logConsoleLines) {
                const auto cls = classifyLogLine(line);
                perfWarnings += cls.perf ? 1 : 0;
                validationWarnings += cls.validation ? 1 : 0;
                resourceWarnings += cls.resource ? 1 : 0;
                shaderMessages += cls.shader ? 1 : 0;
                deviceMessages += cls.device ? 1 : 0;
                errors += cls.error ? 1 : 0;
                visibleLines += lineMatchesFilters(line) ? 1 : 0;
                for (const std::string& physicalLine : SplitPhysicalLogLines(line)) {
                    if (auto diagnostic = ParseShaderCompilerDiagnostic(physicalLine); diagnostic.has_value()) {
                        shaderDiagnostics.push_back(std::move(*diagnostic));
                    }
                }
            }
            ImGui::Text("Visible %d/%zu  Perf %d  Validation %d  Resource %d  Shader %d  Device %d  Errors %d",
                visibleLines,
                _logConsoleLines.size(),
                perfWarnings,
                validationWarnings,
                resourceWarnings,
                shaderMessages,
                deviceMessages,
                errors);
            ImGui::Separator();
            ImGui::Checkbox("Info", &_logShowInfo);
            ImGui::SameLine();
            ImGui::Checkbox("Perf", &_logShowPerformance);
            ImGui::SameLine();
            ImGui::Checkbox("Validation", &_logShowValidation);
            ImGui::SameLine();
            ImGui::Checkbox("Resource", &_logShowResources);
            ImGui::SameLine();
            ImGui::Checkbox("Errors", &_logShowErrors);
            ImGui::SetNextItemWidth(240.0f);
            ImGui::InputText("Filter", _logFilterText.data(), _logFilterText.size());
            ImGui::Separator();
            if (ImGui::Button("Clear")) {
                _logConsoleLines.clear();
            }
            ImGui::SameLine();
            if (ImGui::Button("Export Visible")) {
                const std::filesystem::path path = MakeTimestampedCapturePath("log", ".txt");
                std::filesystem::create_directories(path.parent_path());
                std::ofstream output(path, std::ios::binary);
                for (const std::string& line : _logConsoleLines) {
                    if (lineMatchesFilters(line)) {
                        output << line << '\n';
                    }
                }
                log_startup_event(output ? "Log export written: " + path.string() : "Log export failed: " + path.string());
            }
            ImGui::SameLine();
            if (ImGui::Button("Shader Reload")) {
                const bool reloaded = _renderer.ReloadShaders();
                log_startup_event(reloaded ? "Shader hot reload complete" : "Shader hot reload failed: " + _renderer.GetLastShaderReloadMessage());
            }
            ImGui::SameLine();
            if (ImGui::Button("Perf Snapshot")) {
                const auto& timings = _renderer.GetLastRenderGraphTimings();
                const float snapshotGpuMs = TotalGpuMs(timings);
                const auto* slowest = SlowestGpuPass(timings);
                log_startup_event(fmt::format("[PERF] Snapshot CPU {:.2f} ms GPU {:.2f} ms Passes {}{}",
                    _renderer.GetSmoothedFrameTimeMs(),
                    snapshotGpuMs,
                    timings.size(),
                    slowest != nullptr ? fmt::format(" Slowest {} {:.2f} ms", slowest->name, slowest->gpuMs) : std::string{}));
            }
            ImGui::Separator();
            if (!shaderDiagnostics.empty()) {
                ImGui::SeparatorText("Shader Diagnostics");
                if (ImGui::BeginTable("ShaderDiagnosticsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                    ImGui::TableSetupColumn("Severity", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                    ImGui::TableSetupColumn("File", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("Line", ImGuiTableColumnFlags_WidthFixed, 54.0f);
                    ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 112.0f);
                    ImGui::TableHeadersRow();
                    for (size_t index = 0; index < shaderDiagnostics.size(); ++index) {
                        const ShaderCompilerDiagnostic& diagnostic = shaderDiagnostics[index];
                        const std::filesystem::path sourcePath = ResolveShaderDiagnosticPath(diagnostic);
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        const ImVec4 severityColor = diagnostic.severity == "ERROR" ? ImVec4(1.0f, 0.36f, 0.28f, 1.0f)
                                                                                       : ImVec4(1.0f, 0.78f, 0.28f, 1.0f);
                        ImGui::TextColored(severityColor, "%s", diagnostic.severity.c_str());
                        ImGui::TableSetColumnIndex(1);
                        ImGui::TextUnformatted(diagnostic.file.c_str());
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%d", diagnostic.line);
                        ImGui::TableSetColumnIndex(3);
                        ImGui::TextWrapped("%s", diagnostic.message.c_str());
                        ImGui::TableSetColumnIndex(4);
                        ImGui::PushID(static_cast<int>(index));
                        if (sourcePath.empty()) {
                            ImGui::BeginDisabled();
                            ImGui::SmallButton("Open");
                            ImGui::EndDisabled();
                        } else if (ImGui::SmallButton("Open")) {
                            const bool opened = OpenSourceFileAtLine(sourcePath, diagnostic.line);
                            log_startup_event(opened ? "Opened shader source: " + sourcePath.string()
                                                     : "Failed to open shader source: " + sourcePath.string());
                        }
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Copy")) {
                            const std::string clipboard =
                                diagnostic.file + ":" + std::to_string(diagnostic.line) + " " + diagnostic.message;
                            ImGui::SetClipboardText(clipboard.c_str());
                        }
                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
                ImGui::Separator();
            }
            ImGui::BeginChild("LogScroll", ImVec2(0.0f, 0.0f), true);
            for (const std::string& line : _logConsoleLines) {
                if (!lineMatchesFilters(line)) {
                    continue;
                }
                const auto cls = classifyLogLine(line);
                if (cls.device) {
                    ImGui::TextColored(ImVec4(1.0f, 0.18f, 0.18f, 1.0f), "%s", line.c_str());
                } else if (cls.shader && cls.error) {
                    ImGui::TextColored(ImVec4(1.0f, 0.44f, 0.24f, 1.0f), "%s", line.c_str());
                } else if (cls.shader) {
                    ImGui::TextColored(ImVec4(0.72f, 0.58f, 1.0f, 1.0f), "%s", line.c_str());
                } else if (cls.perf) {
                    ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.28f, 1.0f), "%s", line.c_str());
                } else if (cls.validation || cls.resource) {
                    ImGui::TextColored(ImVec4(0.45f, 0.74f, 1.0f, 1.0f), "%s", line.c_str());
                } else if (cls.error) {
                    ImGui::TextColored(ImVec4(1.0f, 0.36f, 0.32f, 1.0f), "%s", line.c_str());
                } else {
                    ImGui::TextUnformatted(line.c_str());
                }
            }
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
                ImGui::SetScrollHereY(1.0f);
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }
}

void VestaEngine::build_render_mode_control_panel()
{
    if (!_imguiInitialized || !_showRenderModeControlPanel) {
        return;
    }

    ImGui::SetNextWindowPos(ImVec2(852.0f, 18.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(520.0f, 430.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Render Mode Control", &_showRenderModeControlPanel, ImGuiWindowFlags_NoSavedSettings)) {
        auto& settings = _renderer.GetSettings();
        ImGui::Text("Mode %s  Compare %s  Raster %s",
            DisplayModeLabel(settings.displayMode),
            CompareModeLabel(settings.compareMode),
            RasterPipelineModeLabel(settings.rasterPipelineMode));
        ImGui::Text("Debug %s  PT %s  Gaussian %s",
            RendererDebugViewLabel(settings.debugView),
            PathTraceDebugViewLabel(settings.pathTraceDebugView),
            GaussianDebugViewLabel(settings.gaussianDebugView));
        ImGui::Separator();

        if (ImGui::BeginTabBar("RenderModeControlTabs")) {
            if (ImGui::BeginTabItem("Demos")) {
                draw_killer_demo_panel();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Rasterizer")) {
                draw_rasterizer_debug_panel();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Path Tracing")) {
                draw_path_tracing_debug_panel();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Gaussian")) {
                draw_gaussian_splatting_debug_panel();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Ray Effects")) {
                draw_ray_tracing_debug_panel();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("GI")) {
                draw_global_illumination_panel();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Post")) {
                draw_post_process_panel();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Advanced")) {
                draw_advanced_portfolio_panel();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

void VestaEngine::draw_killer_demo_panel()
{
    auto& settings = _renderer.GetSettings();
    const bool sceneLoadInProgress = _renderer.IsSceneLoadInProgress();

    const auto armRasterPathSplitDemo = [&]() {
        settings.displayMode = vesta::render::RendererDisplayMode::Composite;
        settings.compareMode = vesta::render::CompareMode::RasterPathSplit;
        settings.compareSplitPosition = 0.5f;
        settings.enableRaster = true;
        settings.enablePathTracing = true;
        settings.enableGaussian = false;
        vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::FinalColor);
        settings.hybridDepthCompositeDebug = false;
        _showFrameOverview = true;
        _showRenderGraphPanel = true;
        _showGpuProfilerPanel = true;
        _showDebugVisualizationPanel = true;
        _renderer.ResetAccumulation();
        log_startup_event("Killer demo 1 armed: raster/path split comparison");
    };

    if (ImGui::CollapsingHeader("Benchmark Scene Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("BenchmarkScenePresetTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Scene");
            ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 72.0f);
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 96.0f);
            ImGui::TableSetupColumn("Use");
            ImGui::TableHeadersRow();
            for (const BenchmarkScenePreset& preset : kBenchmarkScenePresets) {
                const bool exists = std::filesystem::exists(preset.path);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(preset.label);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored(exists ? ImVec4(0.45f, 0.85f, 0.45f, 1.0f) : ImVec4(0.95f, 0.45f, 0.35f, 1.0f),
                    "%s",
                    exists ? "Ready" : "Missing");
                ImGui::TableSetColumnIndex(2);
                ImGui::BeginDisabled(!exists || sceneLoadInProgress);
                if (ImGui::Button((std::string("Load##") + preset.label).c_str())) {
                    load_scene_path(preset.path);
                }
                ImGui::EndDisabled();
                ImGui::TableSetColumnIndex(3);
                ImGui::TextDisabled("%s", preset.purpose);
            }
            ImGui::EndTable();
        }

        ImGui::BeginDisabled(!std::filesystem::exists("assets/benchmark_scenes/cornell_box/cornell-box.obj") || sceneLoadInProgress);
        if (ImGui::Button("Load Cornell + Raster / Path Split")) {
            load_scene_path("assets/benchmark_scenes/cornell_box/cornell-box.obj");
            armRasterPathSplitDemo();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!std::filesystem::exists("assets/benchmark_scenes/sponza/sponza.obj") || sceneLoadInProgress);
        if (ImGui::Button("Load Sponza + Raster PBR")) {
            load_scene_path("assets/benchmark_scenes/sponza/sponza.obj");
            settings.displayMode = vesta::render::RendererDisplayMode::DeferredLighting;
            settings.compareMode = vesta::render::CompareMode::Off;
            settings.enableRaster = true;
            settings.enablePathTracing = false;
            settings.enableGaussian = false;
            settings.enableSsao = true;
            settings.enableBloom = true;
            vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::FinalColor);
            _showFrameOverview = true;
            _showRenderGraphPanel = true;
            _showGpuProfilerPanel = true;
            _showDebugVisualizationPanel = true;
            _renderer.ResetAccumulation();
            log_startup_event("Benchmark scene preset armed: Sponza raster PBR");
        }
        ImGui::EndDisabled();
    }

    if (ImGui::Button("Demo 1: Raster / Path Split")) {
        armRasterPathSplitDemo();
    }
    ImGui::TextDisabled("Shows direct/indirect lighting and shadow quality differences through split-screen compare.");

    if (ImGui::Button("Demo 2: Real-time GI")) {
        settings.displayMode = vesta::render::RendererDisplayMode::DeferredLighting;
        settings.compareMode = vesta::render::CompareMode::Off;
        settings.enableRaster = true;
        settings.enableGaussian = false;
        settings.enablePathTracing = false;
        settings.enableSsgi = true;
        settings.ssgiIntensity = std::max(settings.ssgiIntensity, 0.55f);
        vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::IndirectLighting);
        settings.animateDirectionalLight = true;
        settings.showGiProbeOverlay = true;
        _showFrameOverview = true;
        _showSceneInspectorPanel = true;
        _showDebugVisualizationPanel = true;
        _renderer.ResetAccumulation();
        log_startup_event("Killer demo 2 armed: dynamic light GI response");
    }
    ImGui::TextDisabled("Enables SSGI, moving light, indirect-light AOV, and GI probe overlay state.");

    if (ImGui::Button("Demo 3: Gaussian Deep Debug")) {
        settings.displayMode = vesta::render::RendererDisplayMode::Gaussian;
        settings.enableRaster = false;
        settings.enableGaussian = true;
        settings.enablePathTracing = false;
        settings.gaussianDebugView = vesta::render::GaussianDebugView::TileOccupancy;
        settings.gaussianShowTileGrid = true;
        settings.gaussianShowSpatialBounds = true;
        settings.gaussianShowCovarianceEllipsoids = true;
        _showGpuProfilerPanel = true;
        _showDebugVisualizationPanel = true;
        _showResourceInspectorPanel = true;
        _renderer.ResetAccumulation();
        log_startup_event("Killer demo 3 armed: gaussian tile occupancy and spatial debug");
    }
    ImGui::TextDisabled("Focuses tile occupancy, splat buffers, covariance ellipsoid and spatial bounds controls.");

    if (ImGui::Button("Demo 4: Hybrid Depth Composite")) {
        settings.displayMode = vesta::render::RendererDisplayMode::Composite;
        settings.compareMode = vesta::render::CompareMode::Off;
        settings.enableRaster = true;
        settings.enableGaussian = true;
        settings.enablePathTracing = false;
        vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::FinalColor);
        settings.gaussianDebugView = vesta::render::GaussianDebugView::CompositionMask;
        settings.hybridDepthCompositeDebug = true;
        _showRenderGraphPanel = true;
        _showResourceInspectorPanel = true;
        _showDebugVisualizationPanel = true;
        _renderer.ResetAccumulation();
        log_startup_event("Killer demo 4 armed: gaussian/raster hybrid depth composite");
    }
    ImGui::TextDisabled("Highlights Gaussian depth and raster depth resources used by the composite pass.");
}

void VestaEngine::draw_rasterizer_debug_panel()
{
    auto& settings = _renderer.GetSettings();

    const char* pipelineModes[] = { "Forward", "Deferred" };
    int pipelineMode = static_cast<int>(settings.rasterPipelineMode);
    if (ImGui::Combo("Pipeline", &pipelineMode, pipelineModes, IM_ARRAYSIZE(pipelineModes))) {
        settings.rasterPipelineMode = static_cast<vesta::render::RasterPipelineMode>(pipelineMode);
        _renderer.ResetAccumulation();
    }

    ImGui::Checkbox("G-Buffer Preview Strip", &settings.showGBufferPreview);
    if (ImGui::Checkbox("Shadow Cascade Overlay", &settings.showShadowCascadeOverlay)) {
        _renderer.ResetAccumulation();
    }
    int cascadeCount = static_cast<int>(settings.shadowCascadeCount);
    if (ImGui::SliderInt("Shadow Cascades", &cascadeCount, 1, 4)) {
        settings.shadowCascadeCount = static_cast<uint32_t>(std::clamp(cascadeCount, 1, 4));
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Cascade Split Lambda", &settings.shadowCascadeLambda, 0.0f, 1.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Button("Shadow Cascade Debug")) {
        vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::ShadowCascade);
        _renderer.ResetAccumulation();
    }
    int shadowMapSize = static_cast<int>(settings.shadowMapSize);
    if (ImGui::SliderInt("Shadow Map Size", &shadowMapSize, 512, 4096)) {
        settings.shadowMapSize = static_cast<uint32_t>(std::clamp(shadowMapSize, 512, 4096));
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Shadow Bias", &settings.shadowBias, 0.0f, 0.01f, "%.4f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Normal Bias", &settings.shadowNormalBias, 0.0f, 0.1f, "%.3f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Shadow Strength", &settings.shadowStrength, 0.0f, 1.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("PCSS Soft Shadows", &settings.enablePcssShadows)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Shadow Filter Radius", &settings.shadowFilterRadius, 0.5f, 4.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Contact Shadows", &settings.enableContactShadows)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Contact Length", &settings.contactShadowLength, 0.05f, 8.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Contact Intensity", &settings.contactShadowIntensity, 0.0f, 1.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }

    ImGui::SeparatorText("Culling");
    ImGui::Checkbox("Frustum Culling", &settings.enableFrustumCulling);
    ImGui::Checkbox("Distance Culling", &settings.enableDistanceCulling);
    ImGui::Checkbox("Indirect Draw", &settings.useIndirectDraw);
    ImGui::SliderFloat("Distance Scale", &settings.distanceCullScale, 1.0f, 100.0f, "%.1f");

    ImGui::SeparatorText("G-Buffer / Raster AOV");
    if (ImGui::Button("Albedo")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::Albedo); }
    ImGui::SameLine();
    if (ImGui::Button("Normal")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::Normal); }
    ImGui::SameLine();
    if (ImGui::Button("Depth")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::Depth); }
    if (ImGui::Button("Roughness")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::Roughness); }
    ImGui::SameLine();
    if (ImGui::Button("Overdraw")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::Overdraw); }
    ImGui::SameLine();
    if (ImGui::Button("Wireframe")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::Wireframe); }
    ImGui::SameLine();
    if (ImGui::Button("Contact Shadow")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::ContactShadow); }

    ImGui::SeparatorText("Temporal Debug");
    if (ImGui::Button("History Color")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::TemporalHistoryColor); }
    ImGui::SameLine();
    if (ImGui::Button("History Depth")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::TemporalHistoryDepth); }
    if (ImGui::Button("Reprojection")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::TemporalReprojection); }
    ImGui::SameLine();
    if (ImGui::Button("Disocclusion")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::TemporalDisocclusion); }
    ImGui::SameLine();
    if (ImGui::Button("Jitter")) { vesta::render::SelectRendererDebugView(settings, vesta::render::RendererDebugView::TemporalJitter); }

    ImGui::SeparatorText("Temporal Upscaler");
    if (ImGui::Checkbox("Enable Temporal Upscaler", &settings.enableTemporalUpscaler)) {
        _renderer.ResetAccumulation();
    }
    ImGui::Checkbox("Show Upscaler Debug", &settings.showTemporalUpscalerDebug);
    if (ImGui::SliderFloat("Upscaler Input Scale", &settings.temporalUpscalerScale, 0.25f, 1.0f, "%.2fx")) {
        settings.temporalUpscalerScale = std::clamp(settings.temporalUpscalerScale, 0.25f, 1.0f);
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Upscaler Sharpness", &settings.temporalUpscalerSharpness, 0.0f, 1.0f, "%.2f")) {
        settings.temporalUpscalerSharpness = std::clamp(settings.temporalUpscalerSharpness, 0.0f, 1.0f);
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Material Reactive Mask", &settings.temporalMaterialReactiveMask)) {
        _renderer.ResetAccumulation();
    }
    if (settings.temporalMaterialReactiveMask
        && ImGui::SliderFloat("Reactive Mask Strength", &settings.temporalReactiveMaskStrength, 0.0f, 1.0f, "%.2f")) {
        settings.temporalReactiveMaskStrength = std::clamp(settings.temporalReactiveMaskStrength, 0.0f, 1.0f);
        _renderer.ResetAccumulation();
    }
    const auto temporalUpscalerStats = _renderer.GetTemporalUpscalerStats();
    ImGui::Text("Input %ux%u -> Output %ux%u",
        temporalUpscalerStats.inputWidth,
        temporalUpscalerStats.inputHeight,
        temporalUpscalerStats.outputWidth,
        temporalUpscalerStats.outputHeight);
    ImGui::Text("History %s  Motion Vectors %s  Depth %s  Reactive Mask %s",
        temporalUpscalerStats.taaHistoryAvailable ? "ready" : "disabled",
        temporalUpscalerStats.motionVectorsAvailable ? "ready" : "missing",
        temporalUpscalerStats.depthAvailable ? "ready" : "missing",
        temporalUpscalerStats.reactiveMaskAvailable ? "ready" : "staged");
    ImGui::Text("Material Reactive %s  Authored Alpha %s  Strength %.2f",
        temporalUpscalerStats.materialReactiveMaskAvailable ? "ready" : "disabled",
        temporalUpscalerStats.authoredAlphaReactiveMaskAvailable ? "ready" : "disabled",
        temporalUpscalerStats.reactiveMaskStrength);
    ImGui::Text("Backend %s", temporalUpscalerStats.backendAvailable ? "TAAU raster path" : "staged or blocked");
    if (!temporalUpscalerStats.backendAvailable) {
        ImGui::TextDisabled("TAAU is active for raster/deferred frames without Gaussian depth-composition inputs.");
    }
}

void VestaEngine::draw_path_tracing_debug_panel()
{
    auto& settings = _renderer.GetSettings();

    const char* backendModes[] = { "Auto", "Compute", "Hardware RT" };
    int backendMode = static_cast<int>(settings.pathTraceBackend);
    if (ImGui::Combo("Backend", &backendMode, backendModes, IM_ARRAYSIZE(backendModes))) {
        settings.pathTraceBackend = static_cast<vesta::render::PathTraceBackend>(backendMode);
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Resolution Scale", &settings.pathTraceResolutionScale, 0.25f, 1.0f, "%.2fx")) {
        _renderer.ResetAccumulation();
    }
    int spp = static_cast<int>(settings.pathTraceSamplesPerPixel);
    if (ImGui::SliderInt("Samples Per Pixel", &spp, 1, 64)) {
        settings.pathTraceSamplesPerPixel = static_cast<uint32_t>(std::clamp(spp, 1, 64));
        _renderer.ResetAccumulation();
    }
    int targetFrames = static_cast<int>(settings.pathTraceTargetFrames);
    if (ImGui::SliderInt("Target Frames", &targetFrames, 1, 1024)) {
        settings.pathTraceTargetFrames = static_cast<uint32_t>(std::clamp(targetFrames, 1, 8192));
    }
    int maxBounces = static_cast<int>(settings.pathTraceMaxBounces);
    if (ImGui::SliderInt("Max Bounces", &maxBounces, 1, 16)) {
        settings.pathTraceMaxBounces = static_cast<uint32_t>(std::clamp(maxBounces, 1, 16));
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Denoising", &settings.enablePathTraceDenoiser)) {
        _renderer.ResetAccumulation();
    }
    if (settings.enablePathTraceDenoiser) {
        ImGui::SliderFloat("Denoiser Strength", &settings.pathTraceDenoiserStrength, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Temporal Blend", &settings.pathTraceDenoiserTemporalBlend, 0.0f, 0.98f, "%.2f");
    }
    if (ImGui::Checkbox("Next Event Estimation", &settings.pathTraceNextEventEstimation)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Russian Roulette", &settings.pathTraceRussianRoulette)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Firefly Clamp", &settings.pathTraceFireflyClamp, 0.0f, 64.0f, "%.1f")) {
        _renderer.ResetAccumulation();
    }

    ImGui::SeparatorText("Debug Views");
    const char* debugViews[] = { "Final", "Albedo", "Normal", "Depth", "Direct", "Indirect", "Ray Cost Heatmap", "Diffuse Bounce", "Specular Bounce", "Throughput", "PDF" };
    int debugView = static_cast<int>(settings.pathTraceDebugView);
    if (ImGui::Combo("Path Trace AOV", &debugView, debugViews, IM_ARRAYSIZE(debugViews))) {
        vesta::render::SelectPathTraceDebugView(settings, static_cast<vesta::render::PathTraceDebugView>(debugView));
        _renderer.ResetAccumulation();
    }
    if (ImGui::Button("Ray Cost Heatmap")) {
        vesta::render::SelectPathTraceDebugView(settings, vesta::render::PathTraceDebugView::RayCountHeatmap);
        _renderer.ResetAccumulation();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Accumulation")) {
        _renderer.ResetAccumulation();
        log_startup_event("Path tracing accumulation reset from mode panel");
    }
    ImGui::SeparatorText("Progress");
    DrawPathTraceProgressBar(settings, _renderer.GetPathTraceFrameIndex(), ImVec2(-FLT_MIN, 0.0f));
    ImGui::Text("Accumulated Frames %u / Target %u", _renderer.GetPathTraceFrameIndex(), settings.pathTraceTargetFrames);
    for (const auto& pass : _renderer.GetRenderPassDebugInfo()) {
        if (pass.id != "path-tracer") {
            continue;
        }
        ImGui::SeparatorText("Ray Type Counters");
        ImGui::Text("Total Estimated Rays %llu", static_cast<unsigned long long>(pass.rayCount));
        ImGui::Text("Primary %llu", static_cast<unsigned long long>(pass.primaryRayCount));
        ImGui::Text("Shadow %llu", static_cast<unsigned long long>(pass.shadowRayCount));
        ImGui::Text("Diffuse %llu", static_cast<unsigned long long>(pass.diffuseRayCount));
        ImGui::Text("Specular %llu", static_cast<unsigned long long>(pass.specularRayCount));
        break;
    }
}

void VestaEngine::draw_gaussian_splatting_debug_panel()
{
    auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();

    if (ImGui::SliderFloat("Opacity", &settings.gaussianOpacity, 0.05f, 1.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Hybrid Mix", &settings.gaussianMix, 0.0f, 1.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    int shDegree = static_cast<int>(settings.gaussianShDegree);
    if (ImGui::SliderInt("SH Degree", &shDegree, 0, 3)) {
        settings.gaussianShDegree = static_cast<uint32_t>(std::clamp(shDegree, 0, 3));
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("View-dependent Color", &settings.gaussianViewDependentColor)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Antialiasing", &settings.gaussianAntialiasing)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Fast Culling", &settings.gaussianFastCulling)) {
        _renderer.ResetAccumulation();
    }

    ImGui::SeparatorText("Splat Counters");
    const uint32_t totalSplats = scene.GetGaussianCount();
    const uint32_t visibleSplats = _renderer.GetOfficialGaussianProjectedCount();
    const uint32_t culledSplats = visibleSplats <= totalSplats ? totalSplats - visibleSplats : 0u;
    const float sortCpuMs = std::max(0.0f, _renderer.GetOfficialGaussianTotalBuildMs() - _renderer.GetOfficialGaussianSortMs());
    ImGui::Text("Total / Visible / Culled %u / %u / %u", totalSplats, visibleSplats, culledSplats);
    ImGui::Text("Depth Sort CPU / GPU %.3f / %.3f ms", sortCpuMs, _renderer.GetOfficialGaussianSortMs());
    ImGui::Text("Preprocess %.3f  Duplicate %.3f  Range %.3f  Raster %.3f ms",
        _renderer.GetOfficialGaussianPreprocessMs(),
        _renderer.GetOfficialGaussianDuplicateMs(),
        _renderer.GetOfficialGaussianRangeMs(),
        _renderer.GetOfficialGaussianRasterMs());
    ImGui::Text("Tile Occupancy %u tiles  %.2f avg tiles/splat",
        _renderer.GetOfficialGaussianTileCount(),
        _renderer.GetOfficialGaussianAverageTilesTouched());
    ImGui::Text("Duplicated / Padded Splats %u / %u",
        _renderer.GetOfficialGaussianDuplicateCount(),
        _renderer.GetOfficialGaussianPaddedDuplicateCount());

    ImGui::SeparatorText("Debug Views");
    const char* gaussianViews[] = {
        "Final",
        "Alpha",
        "Revealage",
        "Overdraw Heatmap",
        "Depth",
        "Tile Occupancy",
        "Splat Radius",
        "Contribution Count",
        "Splat ID",
        "SH Band",
        "Covariance",
        "Raster Depth",
        "Composition Mask",
        "Depth Difference",
    };
    int gaussianView = static_cast<int>(settings.gaussianDebugView);
    if (ImGui::Combo("Gaussian AOV", &gaussianView, gaussianViews, IM_ARRAYSIZE(gaussianViews))) {
        settings.gaussianDebugView = static_cast<vesta::render::GaussianDebugView>(gaussianView);
        _renderer.ResetAccumulation();
    }
    if (ImGui::Button("Overdraw Heatmap")) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::OverdrawHeatmap;
    }
    ImGui::SameLine();
    if (ImGui::Button("Tile Occupancy")) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::TileOccupancy;
        settings.gaussianShowTileGrid = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Splat ID")) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::SplatId;
    }
    ImGui::SameLine();
    if (ImGui::Button("SH Band")) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::ShBand;
    }
    ImGui::SameLine();
    if (ImGui::Button("Covariance")) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::Covariance;
    }
    if (ImGui::Button("Composition Mask")) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::CompositionMask;
    }
    ImGui::SameLine();
    if (ImGui::Button("Depth Difference")) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::DepthDifference;
    }
    ImGui::Checkbox("Tile Grid Overlay", &settings.gaussianShowTileGrid);
    ImGui::Checkbox("Covariance Ellipsoids", &settings.gaussianShowCovarianceEllipsoids);
    ImGui::Checkbox("Spatial Bounds", &settings.gaussianShowSpatialBounds);

    ImGui::SeparatorText("Splat Inspector");
    const auto& gaussians = scene.GetGaussians();
    if (gaussians.empty()) {
        ImGui::TextDisabled("No Gaussian primitives are loaded.");
        return;
    }

    _selectedGaussianInspectorIndex = std::min(_selectedGaussianInspectorIndex, static_cast<uint32_t>(gaussians.size() - 1u));
    int selectedGaussian = static_cast<int>(_selectedGaussianInspectorIndex);
    if (ImGui::InputInt("Selected Splat", &selectedGaussian)) {
        selectedGaussian = std::clamp(selectedGaussian, 0, static_cast<int>(gaussians.size() - 1u));
        _selectedGaussianInspectorIndex = static_cast<uint32_t>(selectedGaussian);
    }
    if (ImGui::Button("Prev")) {
        _selectedGaussianInspectorIndex = _selectedGaussianInspectorIndex > 0u ? _selectedGaussianInspectorIndex - 1u : 0u;
    }
    ImGui::SameLine();
    if (ImGui::Button("Next")) {
        _selectedGaussianInspectorIndex =
            std::min<uint32_t>(_selectedGaussianInspectorIndex + 1u, static_cast<uint32_t>(gaussians.size() - 1u));
    }
    ImGui::SameLine();
    if (ImGui::Button("Nearest Camera")) {
        const glm::vec3 cameraPosition = _renderer.GetCamera().GetPosition();
        float bestDistanceSq = std::numeric_limits<float>::max();
        uint32_t bestIndex = _selectedGaussianInspectorIndex;
        for (uint32_t index = 0; index < gaussians.size(); ++index) {
            const glm::vec3 delta = glm::vec3(gaussians[index].positionOpacity) - cameraPosition;
            const float distanceSq = glm::dot(delta, delta);
            if (distanceSq < bestDistanceSq) {
                bestDistanceSq = distanceSq;
                bestIndex = index;
            }
        }
        _selectedGaussianInspectorIndex = bestIndex;
    }
    ImGui::Checkbox("Inspector Axis Overlay", &_gaussianInspectorShowAxes);
    ImGui::SameLine();
    if (ImGui::Checkbox("Show Selected Ellipsoid", &settings.gaussianShowCovarianceEllipsoids)
        && settings.gaussianShowCovarianceEllipsoids) {
        settings.gaussianDebugView = vesta::render::GaussianDebugView::Covariance;
    }
    ImGui::SliderFloat("Ellipsoid Overlay Scale", &_gaussianInspectorOverlayScale, 0.05f, 8.0f, "%.2fx");

    const auto& selected = gaussians[_selectedGaussianInspectorIndex];
    const glm::vec3 color = GaussianBaseColor(selected);
    ImGui::Text("Position %.3f %.3f %.3f  Opacity %.3f",
        selected.positionOpacity.x,
        selected.positionOpacity.y,
        selected.positionOpacity.z,
        selected.positionOpacity.w);
    ImGui::Text("Scale %.5f %.5f %.5f", selected.scale.x, selected.scale.y, selected.scale.z);
    ImGui::Text("Rotation %.3f %.3f %.3f %.3f", selected.rotation.x, selected.rotation.y, selected.rotation.z, selected.rotation.w);
    ImGui::ColorButton("DC Color", ImVec4(color.r, color.g, color.b, 1.0f), ImGuiColorEditFlags_NoTooltip, ImVec2(32.0f, 18.0f));
    ImGui::SameLine();
    ImGui::Text("SH degree %u  coefficients %u",
        scene.GetGaussianShDegree(),
        std::min<uint32_t>((scene.GetGaussianShDegree() + 1u) * (scene.GetGaussianShDegree() + 1u),
            vesta::scene::kGaussianMaxShCoefficients));
}

void VestaEngine::draw_ray_tracing_debug_panel()
{
    auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();
    const auto& device = _renderer.GetRenderDevice();
    const bool rayQuerySupported = device.GetRayTracingSupport().rayQueryFeatures.rayQuery == VK_TRUE;
    const bool pipelineSupported = device.GetRayTracingSupport().rayTracingPipelineFeatures.rayTracingPipeline == VK_TRUE;
    const bool hybridEffectsSupported = rayQuerySupported && scene.HasRayTracingScene();

    ImGui::Text("Hardware RT pipeline %s  Ray Query %s",
        pipelineSupported ? "available" : "unavailable",
        rayQuerySupported ? "available" : "unavailable");
    ImGui::Text("TLAS %s  BLAS %.3f ms  TLAS %.3f ms",
        scene.HasRayTracingScene() ? "resident" : "missing",
        scene.GetBottomLevelBuildMs(),
        scene.GetTopLevelBuildMs());
    ImGui::Text("Backend %s", PathTraceBackendLabel(_renderer.GetActivePathTraceBackend()));

    if (!hybridEffectsSupported) {
        ImGui::TextDisabled("Hybrid ray effects require ray query support and a resident TLAS.");
    }

    const auto rayEffectsStats = _renderer.GetRayEffectsStats();
    ImGui::SeparatorText("Hybrid Ray Effects Readiness");
    ImGui::Text("Input %ux%u  TLAS %s  RayQuery %s  Backend %s",
        rayEffectsStats.inputWidth,
        rayEffectsStats.inputHeight,
        rayEffectsStats.tlasAvailable ? "ready" : "missing",
        rayEffectsStats.rayQueryAvailable ? "ready" : "missing",
        rayEffectsStats.backendAvailable ? "live" : "staged");
    ImGui::Text("Estimated rays: Shadow %llu  AO %llu  Reflection %llu  GI %llu",
        static_cast<unsigned long long>(rayEffectsStats.estimatedShadowRays),
        static_cast<unsigned long long>(rayEffectsStats.estimatedAoRays),
        static_cast<unsigned long long>(rayEffectsStats.estimatedReflectionRays),
        static_cast<unsigned long long>(rayEffectsStats.estimatedGiRays));
    ImGui::Text("Denoiser %s  GI Spatial %s  GI Temporal %s  Resolution %s",
        rayEffectsStats.denoiserRequested ? "on" : "off",
        rayEffectsStats.giSpatialDenoiseAvailable ? "live" : "staged",
        rayEffectsStats.giTemporalAccumulationAvailable ? "live" : (rayEffectsStats.temporalAccumulation ? "armed" : "off"),
        rayEffectsStats.halfResolution ? "half" : "full");

    ImGui::BeginDisabled(!hybridEffectsSupported);
    bool resetHistory = false;
    resetHistory |= ImGui::Checkbox("RT Shadows", &settings.enableRtShadows);
    resetHistory |= ImGui::Checkbox("RT Ambient Occlusion", &settings.enableRtAmbientOcclusion);
    resetHistory |= ImGui::Checkbox("RT Reflections", &settings.enableRtReflections);
    resetHistory |= ImGui::Checkbox("RT Global Illumination", &settings.enableRtGlobalIllumination);
    int shadowSamples = static_cast<int>(settings.rtShadowSamples);
    if (ImGui::SliderInt("Shadow Samples", &shadowSamples, 1, 8)) {
        settings.rtShadowSamples = static_cast<uint32_t>(std::clamp(shadowSamples, 1, 8));
        resetHistory = true;
    }
    int aoSamples = static_cast<int>(settings.rtAoSamples);
    if (ImGui::SliderInt("AO Samples", &aoSamples, 1, 8)) {
        settings.rtAoSamples = static_cast<uint32_t>(std::clamp(aoSamples, 1, 8));
        resetHistory = true;
    }
    int reflectionSamples = static_cast<int>(settings.rtReflectionSamples);
    if (ImGui::SliderInt("Reflection Samples", &reflectionSamples, 1, 8)) {
        settings.rtReflectionSamples = static_cast<uint32_t>(std::clamp(reflectionSamples, 1, 8));
        resetHistory = true;
    }
    int giSamples = static_cast<int>(settings.rtGiSamples);
    if (ImGui::SliderInt("GI Samples", &giSamples, 1, 8)) {
        settings.rtGiSamples = static_cast<uint32_t>(std::clamp(giSamples, 1, 8));
        resetHistory = true;
    }
    resetHistory |= ImGui::SliderFloat("Max Ray Distance", &settings.rtMaxRayDistance, 0.5f, 100000.0f, "%.1f");
    resetHistory |= ImGui::SliderFloat("AO Radius", &settings.rtAoRadius, 0.05f, 20.0f, "%.2f");
    resetHistory |= ImGui::SliderFloat("Reflection Roughness Cutoff", &settings.rtReflectionRoughnessCutoff, 0.0f, 1.0f, "%.2f");
    resetHistory |= ImGui::Checkbox("Half Resolution", &settings.rtHalfResolution);
    resetHistory |= ImGui::Checkbox("Denoiser", &settings.rtDenoiser);
    resetHistory |= ImGui::Checkbox("Temporal Accumulation", &settings.rtTemporalAccumulation);
    ImGui::EndDisabled();
    if (resetHistory) {
        _renderer.ResetAccumulation();
    }

    ImGui::SeparatorText("Implemented vs Stub");
    ImGui::BulletText("Hardware path tracing uses RT pipeline when available.");
    ImGui::BulletText("Hybrid RT shadows, AO, and reflection visibility use a ray-query pass when Ray Query and TLAS are available.");
    ImGui::BulletText("RT reflection and RT GI resolve material-colored ray hits from the TLAS-backed scene triangle buffer.");
    ImGui::BulletText("RT GI spatial denoise and confidence-weighted temporal reuse are live when the denoiser/temporal toggles are enabled.");
    ImGui::BulletText("Acceleration structure residency and build timing are live in Resource Inspector.");
}

void VestaEngine::draw_global_illumination_panel()
{
    auto& settings = _renderer.GetSettings();

    bool resetHistory = false;
    if (ImGui::Checkbox("Path Traced GI", &settings.enablePathTracedGi)) {
        settings.enablePathTracing = settings.enablePathTracing || settings.enablePathTracedGi;
        resetHistory = true;
    }
    ImGui::TextDisabled("Path traced GI is represented by indirect bounces and the Indirect path AOV.");

    if (ImGui::Checkbox("SSGI", &settings.enableSsgi)) {
        resetHistory = true;
    }
    if (settings.enableSsgi) {
        resetHistory |= ImGui::SliderFloat("SSGI Radius", &settings.ssgiRadius, 0.05f, 8.0f, "%.2f");
        resetHistory |= ImGui::SliderFloat("SSGI Intensity", &settings.ssgiIntensity, 0.0f, 2.0f, "%.2f");
        int ssgiSamples = static_cast<int>(settings.ssgiSampleCount);
        if (ImGui::SliderInt("SSGI Samples", &ssgiSamples, 4, 16)) {
            settings.ssgiSampleCount = static_cast<uint32_t>(std::clamp(ssgiSamples, 4, 16));
            resetHistory = true;
        }
    }
    if (ImGui::Checkbox("Indirect Only Debug", &settings.showGiIndirectOnly)) {
        vesta::render::SelectRendererDebugView(settings,
            settings.showGiIndirectOnly ? vesta::render::RendererDebugView::IndirectLighting
                                        : vesta::render::RendererDebugView::FinalColor);
        resetHistory = true;
    }
    ImGui::Checkbox("GI Probe Overlay", &settings.showGiProbeOverlay);

    ImGui::SeparatorText("DDGI Probe Grid");
    if (ImGui::Checkbox("DDGI Probe Storage", &settings.enableDdgi)) {
        resetHistory = true;
    }
    int probesX = static_cast<int>(settings.ddgiProbeCountX);
    int probesY = static_cast<int>(settings.ddgiProbeCountY);
    int probesZ = static_cast<int>(settings.ddgiProbeCountZ);
    int raysPerProbe = static_cast<int>(settings.ddgiRaysPerProbe);
    if (ImGui::SliderInt("Probe Count X", &probesX, 1, 32)) {
        settings.ddgiProbeCountX = static_cast<uint32_t>(std::clamp(probesX, 1, 32));
        resetHistory = true;
    }
    if (ImGui::SliderInt("Probe Count Y", &probesY, 1, 16)) {
        settings.ddgiProbeCountY = static_cast<uint32_t>(std::clamp(probesY, 1, 16));
        resetHistory = true;
    }
    if (ImGui::SliderInt("Probe Count Z", &probesZ, 1, 32)) {
        settings.ddgiProbeCountZ = static_cast<uint32_t>(std::clamp(probesZ, 1, 32));
        resetHistory = true;
    }
    if (ImGui::SliderInt("Rays / Probe", &raysPerProbe, 16, 1024)) {
        settings.ddgiRaysPerProbe = static_cast<uint32_t>(std::clamp(raysPerProbe, 16, 1024));
        resetHistory = true;
    }
    resetHistory |= ImGui::SliderFloat("Probe Spacing", &settings.ddgiProbeSpacing, 0.25f, 10.0f, "%.2f");
    resetHistory |= ImGui::SliderFloat("Hysteresis", &settings.ddgiHysteresis, 0.0f, 1.0f, "%.2f");
    resetHistory |= ImGui::SliderFloat("Probe Composite Intensity", &settings.ddgiIntensity, 0.0f, 2.0f, "%.2f");
    const auto ddgiStats = _renderer.GetDdgiStats();
    ImGui::Text("Probes %u (%ux%ux%u)  Rays/update %llu",
        ddgiStats.totalProbeCount,
        ddgiStats.probeCountX,
        ddgiStats.probeCountY,
        ddgiStats.probeCountZ,
        static_cast<unsigned long long>(ddgiStats.raysPerUpdate));
    ImGui::Text("Estimated irradiance %.2f MiB  visibility %.2f MiB  relocation %.3f MiB",
        MiB(ddgiStats.estimatedIrradianceBytes),
        MiB(ddgiStats.estimatedVisibilityBytes),
        MiB(ddgiStats.estimatedRelocationBytes));
    ImGui::Text("Spacing %.2f  Hysteresis %.2f  Overlay %s",
        ddgiStats.probeSpacing,
        ddgiStats.hysteresis,
        ddgiStats.overlayEnabled ? "on" : "off");
    const char* ddgiBackendLabel = ddgiStats.storageCompositeAvailable && ddgiStats.rayUpdateAvailable ? "StorageComposite + RayUpdate"
        : ddgiStats.storageCompositeAvailable                                                         ? "StorageComposite"
        : ddgiStats.probeCompositeAvailable                                                           ? "ProbeComposite"
        : ddgiStats.rayUpdateAvailable                                                                ? "RayUpdate"
                                                                                                        : "Staged";
    ImGui::Text("Storage %s  Backend %s", ddgiStats.probeStorageAvailable ? "allocated" : "staged", ddgiBackendLabel);
    ImGui::Text("Composite %s  Storage Read %s  Moment Validation %s",
        ddgiStats.probeCompositeAvailable ? "live" : "staged",
        ddgiStats.storageCompositeAvailable ? "live" : "staged",
        ddgiStats.momentValidationAvailable ? "live" : "staged");
    ImGui::Text("Spatial Filter %s  Ray Update %s  Temporal Blend %s",
        ddgiStats.spatialFilteringAvailable ? "live" : "staged",
        ddgiStats.rayUpdateAvailable ? "live" : "staged",
        ddgiStats.temporalBlendAvailable ? "live" : "staged");
    ImGui::Text("Probe Relocation %s",
        ddgiStats.probeRelocationAvailable ? "live" : "staged");
    ImGui::TextDisabled(
        "DDGI probe storage, storage-backed irradiance composite, visibility moment validation, neighbor spatial filtering, ray-query probe update, temporal hysteresis blending, and probe relocation are live when TLAS/ray query are available.");

    ImGui::SeparatorText("Advanced GI");
    ImGui::Checkbox("Voxel GI", &settings.enableVoxelGi);
    int voxelResolution = static_cast<int>(settings.voxelGiResolution);
    if (ImGui::SliderInt("Voxel Resolution", &voxelResolution, 16, 128)) {
        settings.voxelGiResolution = static_cast<uint32_t>(std::clamp(voxelResolution, 16, 128));
        resetHistory = true;
    }
    resetHistory |= ImGui::SliderFloat("Voxel World Extent", &settings.voxelGiWorldExtent, 4.0f, 128.0f, "%.1f m");
    const auto voxelGiStats = _renderer.GetVoxelGiStats();
    ImGui::Text("Voxel GI %s  %u^3 (%u voxels)",
        voxelGiStats.storageAvailable ? "VolumeStorage" : "Staged",
        voxelGiStats.resolution,
        voxelGiStats.totalVoxels);
    ImGui::Text("Extent %.1f m  Voxel %.3f m  Memory %.2f MiB",
        voxelGiStats.worldExtent,
        voxelGiStats.voxelSize,
        MiB(voxelGiStats.estimatedRadianceBytes + voxelGiStats.estimatedOccupancyBytes));
    ImGui::Text("Radiance %s  Occupancy %s  Visualization %s",
        voxelGiStats.radianceAvailable ? "ready" : "staged",
        voxelGiStats.occupancyAvailable ? "ready" : "staged",
        voxelGiStats.visualizationAvailable ? "ready" : "staged");
    ImGui::BeginDisabled(true);
    ImGui::Checkbox("ReSTIR GI", &settings.enableRestirGi);
    ImGui::EndDisabled();
    ImGui::TextDisabled("Voxel GI volume storage is live when enabled; production voxelization and cone tracing remain staged.");

    if (resetHistory) {
        _renderer.ResetAccumulation();
    }
}

void VestaEngine::draw_post_process_panel()
{
    auto& settings = _renderer.GetSettings();

    const char* toneModes[] = { "None", "Reinhard", "ACES" };
    int toneMode = static_cast<int>(settings.toneMappingMode);
    if (ImGui::Combo("Tone Mapping", &toneMode, toneModes, IM_ARRAYSIZE(toneModes))) {
        settings.toneMappingMode = static_cast<vesta::render::ToneMappingMode>(toneMode);
        _renderer.ResetAccumulation();
    }
    ImGui::Text("Active Display Transform %s", ToneMappingModeLabel(settings.toneMappingMode));
    if (ImGui::SliderFloat("Exposure", &settings.cameraExposureEv, -6.0f, 6.0f, "%.2f EV")) {
        _renderer.ResetAccumulation();
    }
    ImGui::SliderFloat("Saturation", &settings.colorGradingSaturation, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Contrast", &settings.colorGradingContrast, 0.25f, 2.0f, "%.2f");

    ImGui::SeparatorText("Effects");
    ImGui::Checkbox("Bloom", &settings.enableBloom);
    if (settings.enableBloom) {
        ImGui::SliderFloat("Bloom Threshold", &settings.bloomThreshold, 0.0f, 8.0f, "%.2f");
        ImGui::SliderFloat("Bloom Intensity", &settings.bloomIntensity, 0.0f, 2.0f, "%.2f");
    }
    ImGui::Checkbox("Color Grading", &settings.enableColorGrading);
    ImGui::Checkbox("Vignette", &settings.enableVignette);
    if (settings.enableVignette) {
        ImGui::SliderFloat("Vignette Strength", &settings.vignetteStrength, 0.0f, 1.0f, "%.2f");
    }
    const char* aaModes[] = { "None", "FXAA", "TAA", "TAAU", "MSAA", "DLSS" };
    int aaMode = static_cast<int>(settings.antiAliasingMode);
    if (ImGui::Combo("Anti-Aliasing", &aaMode, aaModes, IM_ARRAYSIZE(aaModes))) {
        ApplyAntiAliasingMode(settings, static_cast<vesta::render::AntiAliasingMode>(aaMode));
        _renderer.ResetAccumulation();
    }
    if (settings.enableMsaa) {
        int msaaSamples = static_cast<int>(settings.msaaSampleCount);
        if (ImGui::SliderInt("MSAA Samples", &msaaSamples, 2, 8)) {
            settings.msaaSampleCount = static_cast<uint32_t>(std::clamp(msaaSamples, 2, 8));
            _renderer.ResetAccumulation();
        }
    }
    if (settings.enableDlss) {
        ImGui::TextDisabled("DLSS is exposed as a mode flag; backend integration is not active in this build.");
    }
    ImGui::Checkbox("Motion Blur", &settings.enableMotionBlur);
    if (settings.enableMotionBlur) {
        ImGui::SliderFloat("Motion Blur Strength", &settings.motionBlurStrength, 0.0f, 2.0f, "%.2f");
    }
    ImGui::SeparatorText("Depth of Field");
    if (ImGui::SliderFloat("Aperture Radius", &settings.cameraApertureRadius, 0.0f, 0.25f, "%.3f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Focal Distance", &settings.cameraFocalDistance, 0.05f, 100.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }

    ImGui::SeparatorText("Implemented vs Stub");
    ImGui::BulletText("Exposure, ACES-style display transform, multi-pass bloom, FXAA, motion blur, color controls, and vignette are live.");
    ImGui::BulletText("Depth-of-field parameters are live for path tracing camera settings.");
    ImGui::BulletText("Motion blur uses the GBuffer motion vector target for a lightweight screen-space blur.");
}

void VestaEngine::draw_advanced_portfolio_panel()
{
    auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();
    const auto& device = _renderer.GetRenderDevice();

    ImGui::SeparatorText("ReSTIR");
    ImGui::Checkbox("ReSTIR DI", &settings.enableRestirDi);
    ImGui::Checkbox("ReSTIR GI", &settings.enableRestirGi);
    ImGui::Checkbox("ReSTIR PT", &settings.enableRestirPt);
    int candidateLights = static_cast<int>(settings.restirCandidateLights);
    int reservoirs = static_cast<int>(settings.restirReservoirCount);
    int spatialSamples = static_cast<int>(settings.restirSpatialSamples);
    if (ImGui::SliderInt("Candidate Lights", &candidateLights, 1, 64)) {
        settings.restirCandidateLights = static_cast<uint32_t>(std::clamp(candidateLights, 1, 64));
    }
    if (ImGui::SliderInt("Reservoirs / Pixel", &reservoirs, 1, 8)) {
        settings.restirReservoirCount = static_cast<uint32_t>(std::clamp(reservoirs, 1, 8));
    }
    if (ImGui::SliderInt("Spatial Reuse Samples", &spatialSamples, 0, 16)) {
        settings.restirSpatialSamples = static_cast<uint32_t>(std::clamp(spatialSamples, 0, 16));
    }
    ImGui::SliderFloat("Resolve Intensity", &settings.restirDirectLightingIntensity, 0.0f, 2.0f, "%.2f");
    ImGui::Checkbox("Temporal Reuse", &settings.restirTemporalReuse);
    ImGui::Checkbox("Spatial Reuse", &settings.restirSpatialReuse);
    ImGui::Checkbox("Show Reservoir Debug", &settings.restirShowReservoirs);
    ImGui::Checkbox("Show Selected Light", &settings.restirShowSelectedLight);
    const auto restirStats = _renderer.GetRestirStats();
    ImGui::Text("Lights %u active (%u emissive tris, %u analytic/local)",
        restirStats.activeLightCount,
        restirStats.emissiveTriangleCount,
        restirStats.localLightCount);
    ImGui::Text("Reservoirs %u / pixel  Candidates %u  Estimated %.2f MiB",
        restirStats.reservoirCount,
        restirStats.candidateLightCount,
        MiB(restirStats.estimatedReservoirBytes));
    ImGui::Text("DI %.2f MiB  GI %.2f MiB  PT %.2f MiB  PT state %.2f MiB",
        MiB(restirStats.estimatedDiReservoirBytes),
        MiB(restirStats.estimatedGiReservoirBytes),
        MiB(restirStats.estimatedPtReservoirBytes),
        MiB(restirStats.estimatedPtPathStateBytes));
    ImGui::Text("History %s  Temporal %s  Spatial %s",
        restirStats.historyAvailable ? "ready" : "disabled",
        restirStats.temporalReuse ? "on" : "off",
        restirStats.spatialReuse ? "on" : "off");
    ImGui::Text("Storage %s  Backend %s",
        restirStats.reservoirBuffersAvailable ? "allocated" : "staged",
        (restirStats.lightingResolveAvailable || restirStats.giResolveAvailable || restirStats.ptResolveAvailable)
            ? "CandidateReservoir+ShadingResolve"
            : (restirStats.backendAvailable ? "ReservoirBackend" : "Staged"));
    ImGui::Text("Storage DI %s  GI %s  PT %s  PT State %s",
        restirStats.diReservoirBuffersAvailable ? "ready" : "staged",
        restirStats.giReservoirBuffersAvailable ? "ready" : "staged",
        restirStats.ptReservoirBuffersAvailable ? "ready" : "staged",
        restirStats.ptPathStateAvailable ? "ready" : "staged");
    ImGui::Text("Passes DI Candidate %s  GI Candidate %s  PT Candidate %s",
        restirStats.candidateSamplingAvailable ? "live" : "staged",
        restirStats.giCandidatePassAvailable ? "live" : "staged",
        restirStats.ptCandidatePassAvailable ? "live" : "staged");
    ImGui::Text("Resolve DI %s  GI %s  PT %s",
        restirStats.lightingResolveAvailable ? "live" : "staged",
        restirStats.giResolveAvailable ? "live" : "staged",
        restirStats.ptResolveAvailable ? "live" : "staged");
    ImGui::Text("Reuse Temporal %s  Spatial %s  PT Path State %s",
        restirStats.temporalReusePassAvailable ? "live" : "staged",
        restirStats.spatialReusePassAvailable ? "live" : "staged",
        restirStats.ptPathStateReuseAvailable ? "live" : "staged");
    ImGui::TextDisabled("DI/GI/PT candidate reservoir updates, PT path-state storage reuse, and screen-space shading resolve are live when requested.");

    ImGui::SeparatorText("GPU-driven Rendering");
    ImGui::Checkbox("Indirect Draw", &settings.useIndirectDraw);
    ImGui::BeginDisabled(true);
    ImGui::Checkbox("GPU-driven Culling", &settings.enableGpuDrivenRendering);
    ImGui::Checkbox("Meshlet / Cluster Culling", &settings.enableMeshletCulling);
    ImGui::EndDisabled();
    const auto gpuDrivenStats = _renderer.GetGpuDrivenStats();
    ImGui::Text("Visible Surfaces %u / %u  (%u culled)",
        gpuDrivenStats.visibleSurfaces,
        gpuDrivenStats.totalSurfaces,
        gpuDrivenStats.culledSurfaces);
    ImGui::Text("Indirect draw estimate %u  Mode %s",
        gpuDrivenStats.indirectDrawEstimate,
        gpuDrivenStats.indirectDrawEnabled ? "single indirect command" : "direct per-surface draw");
    ImGui::TextDisabled(gpuDrivenStats.visibilitySetValid ? "Visibility set is current; GPU-driven backend remains staged."
                                                          : "Visibility set is pending or disabled; stats assume full scene visible.");
    const auto meshletStats = _renderer.GetMeshletClusterStats();
    ImGui::Text("Cluster bounds %u / %u", meshletStats.boundsAvailable, meshletStats.totalClusters);
    ImGui::Text("Meshlets %u visible / %u total  (%u culled)",
        meshletStats.visibleMeshlets,
        meshletStats.totalMeshlets,
        meshletStats.culledMeshlets);
    ImGui::Text("Clusters %u visible / %u total  (%u culled)",
        meshletStats.visibleClusters,
        meshletStats.totalClusters,
        meshletStats.culledClusters);
    ImGui::Text("Visibility storage %s  Backend %s  Memory %.3f MiB",
        meshletStats.visibilityStorageAvailable ? "ready" : "staged",
        meshletStats.visibilityStorageAvailable ? "VisibilityStorage" : "Staged",
        MiB(meshletStats.estimatedVisibilityBytes));
    ImGui::Text("Meshlet grouping %u triangles / meshlet", meshletStats.trianglesPerMeshlet);
    ImGui::TextDisabled(meshletStats.visibilitySetValid ? "CPU visibility set is current; meshlet visibility storage is live when enabled."
                                                        : "No current visibility set; stats use full scene as visible.");
    if (ImGui::BeginTable("MeshletClusterStats", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Metric");
        ImGui::TableSetupColumn("Total", ImGuiTableColumnFlags_WidthFixed, 72.0f);
        ImGui::TableSetupColumn("Visible", ImGuiTableColumnFlags_WidthFixed, 72.0f);
        ImGui::TableSetupColumn("Culled", ImGuiTableColumnFlags_WidthFixed, 72.0f);
        ImGui::TableHeadersRow();
        auto clusterRow = [](const char* metric, uint32_t total, uint32_t visible, uint32_t culled) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(metric);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%u", total);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%u", visible);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%u", culled);
        };
        clusterRow("Clusters", meshletStats.totalClusters, meshletStats.visibleClusters, meshletStats.culledClusters);
        clusterRow("Meshlets", meshletStats.totalMeshlets, meshletStats.visibleMeshlets, meshletStats.culledMeshlets);
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Compute cone culling and indirect meshlet draws remain staged beyond the current visibility-storage backend.");

    ImGui::SeparatorText("Bindless");
    const auto bindlessStats = device.GetBindlessStats();
    ImGui::Text("Textures %zu  Resident %u", scene.GetTextures().size(), _renderer.GetResidentTextureCount());
    ImGui::Text("Sampled image descriptors %u / %u", bindlessStats.sampledImagesUsed, bindlessStats.sampledImagesCapacity);
    ImGui::Text("Sampled cube descriptors %u / %u", bindlessStats.sampledCubeImagesUsed, bindlessStats.sampledCubeImagesCapacity);
    ImGui::Text("Storage image descriptors %u / %u", bindlessStats.storageImagesUsed, bindlessStats.storageImagesCapacity);
    ImGui::Text("Storage buffer descriptors %u / %u", bindlessStats.storageBuffersUsed, bindlessStats.storageBuffersCapacity);
    ImGui::Text("Device local memory %u MiB", device.GetDedicatedVideoMemoryMiB());
    ImGui::Text("Path backend %s", PathTraceBackendLabel(_renderer.GetActivePathTraceBackend()));

    ImGui::SeparatorText("Async Compute");
    ImGui::BeginDisabled(true);
    ImGui::Checkbox("Async Compute", &settings.enableAsyncCompute);
    ImGui::Checkbox("Queue Timeline", &settings.showAsyncComputeTimeline);
    ImGui::EndDisabled();
    ImGui::Text("Graphics Queue Family %u", device.GetGraphicsQueueFamily());
    ImGui::Text("Transfer Queue %s", device.HasTransferQueue() ? "available" : "shared with graphics");
    if (device.HasTransferQueue()) {
        ImGui::SameLine();
        ImGui::Text("(family %u)", device.GetTransferQueueFamily());
    }
    if (ImGui::BeginTable("AsyncTimelineStub", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Pass");
        ImGui::TableSetupColumn("Queue", ImGuiTableColumnFlags_WidthFixed, 78.0f);
        ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 86.0f);
        ImGui::TableSetupColumn("Sync / Blocker");
        ImGui::TableHeadersRow();
        auto timelineRow = [](const char* pass, const char* queue, const char* state, const char* blocker) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(pass);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(queue);
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(state);
            ImGui::TableSetColumnIndex(3);
            ImGui::TextWrapped("%s", blocker);
        };
        timelineRow("Texture Upload", device.HasTransferQueue() ? "Transfer" : "Graphics", "Implemented", "Upload batch flushes before graphics consumption.");
        timelineRow("SSAO / SSGI", "Compute", "Staged", "Needs render graph async queue scheduling and history hazards.");
        timelineRow("Path Denoise", "Compute", "Staged", "Needs queue ownership barriers for path output and denoised target.");
        timelineRow("Gaussian Sort", "Compute", "Staged", "Needs overlap window between preprocessing and raster/composite.");
        timelineRow("Composite", "Graphics", "Serial", "Consumes all color/debug outputs before present.");
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Async compute is intentionally read-only until the render graph owns cross-queue scheduling.");
}

bool VestaEngine::should_forward_event_to_renderer(const SDL_Event& event) const
{
    if (!_imguiInitialized) {
        return true;
    }

    ImGui::SetCurrentContext(_imguiContext);
    const ImGuiIO& io = ImGui::GetIO();

    switch (event.type) {
    case SDL_MOUSEMOTION:
    case SDL_MOUSEWHEEL:
    case SDL_MOUSEBUTTONDOWN:
    case SDL_MOUSEBUTTONUP:
        return !io.WantCaptureMouse;
    case SDL_TEXTINPUT:
    case SDL_KEYDOWN:
    case SDL_KEYUP:
        return !io.WantCaptureKeyboard;
    default:
        return true;
    }
}

std::optional<std::filesystem::path> VestaEngine::open_scene_with_system_dialog() const
{
#if defined(_WIN32)
    SDL_SysWMinfo windowInfo{};
    SDL_VERSION(&windowInfo.version);
    if (!SDL_GetWindowWMInfo(_window, &windowInfo)) {
        return std::nullopt;
    }

    std::array<wchar_t, 4096> filePath{};
    std::wstring initialDirectory;
    const std::filesystem::path currentPath = _renderer.GetScene().GetSourcePath();
    if (!currentPath.empty()) {
        initialDirectory = currentPath.parent_path().wstring();
    }

    OPENFILENAMEW dialogInfo{};
    dialogInfo.lStructSize = sizeof(dialogInfo);
    dialogInfo.hwndOwner = windowInfo.info.win.window;
    dialogInfo.lpstrFile = filePath.data();
    dialogInfo.nMaxFile = static_cast<DWORD>(filePath.size());
    dialogInfo.lpstrFilter =
        L"Supported Scenes (*.glb;*.gltf;*.fbx;*.obj;*.ply)\0*.glb;*.gltf;*.fbx;*.obj;*.ply\0glTF Scenes (*.glb;*.gltf)\0*.glb;*.gltf\0OBJ Meshes (*.obj)\0*.obj\0FBX Meshes (*.fbx)\0*.fbx\0Mesh or Gaussian PLY (*.ply)\0*.ply\0All Files (*.*)\0*.*\0";
    dialogInfo.lpstrInitialDir = initialDirectory.empty() ? nullptr : initialDirectory.c_str();
    dialogInfo.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
    dialogInfo.lpstrDefExt = L"glb";

    if (!GetOpenFileNameW(&dialogInfo)) {
        return std::nullopt;
    }

    return std::filesystem::path(filePath.data());
#else
    return std::nullopt;
#endif
}

std::optional<std::filesystem::path> VestaEngine::open_gaussian_model_with_system_dialog() const
{
#if defined(_WIN32)
    SDL_SysWMinfo windowInfo{};
    SDL_VERSION(&windowInfo.version);
    if (!SDL_GetWindowWMInfo(_window, &windowInfo)) {
        return std::nullopt;
    }

    HRESULT initResult = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    const bool shouldUninitialize = SUCCEEDED(initResult);
    if (FAILED(initResult) && initResult != RPC_E_CHANGED_MODE) {
        return std::nullopt;
    }

    std::optional<std::filesystem::path> selectedPath;
    IFileOpenDialog* dialog = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))
        && dialog != nullptr) {
        DWORD options = 0;
        if (SUCCEEDED(dialog->GetOptions(&options))) {
            dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        }
        dialog->SetTitle(L"Select a trained Gaussian model directory");
        dialog->SetOkButtonLabel(L"Open Folder");

        const std::filesystem::path currentPath = NormalizeScenePath(_renderer.GetScene().GetSourcePath());
        std::filesystem::path initialDirectory = currentPath;
        if (!initialDirectory.empty() && std::filesystem::is_regular_file(initialDirectory)) {
            initialDirectory = initialDirectory.parent_path();
        }
        if (!initialDirectory.empty()) {
            IShellItem* initialFolder = nullptr;
            if (SUCCEEDED(SHCreateItemFromParsingName(initialDirectory.c_str(), nullptr, IID_PPV_ARGS(&initialFolder)))
                && initialFolder != nullptr) {
                dialog->SetFolder(initialFolder);
                initialFolder->Release();
            }
        }

        if (SUCCEEDED(dialog->Show(windowInfo.info.win.window))) {
            IShellItem* result = nullptr;
            if (SUCCEEDED(dialog->GetResult(&result)) && result != nullptr) {
                PWSTR path = nullptr;
                if (SUCCEEDED(result->GetDisplayName(SIGDN_FILESYSPATH, &path)) && path != nullptr) {
                    selectedPath = std::filesystem::path(path);
                    CoTaskMemFree(path);
                }
                result->Release();
            }
        }

        dialog->Release();
    }

    if (shouldUninitialize) {
        CoUninitialize();
    }

    return selectedPath;
#else
    return std::nullopt;
#endif
}

void VestaEngine::load_scene_path(const std::filesystem::path& path)
{
    const std::filesystem::path normalizedPath = NormalizeScenePath(path);
    if (normalizedPath.empty()) {
        return;
    }

    ApplySceneModeInference(_renderer.GetSettings(), normalizedPath);
    ApplyBenchmarkSceneLightingPreset(_renderer.GetSettings(), normalizedPath);
    if (const std::optional<std::filesystem::path> hdri = BenchmarkSceneHdri(normalizedPath)) {
        apply_external_hdri_path(*hdri);
    }
    _renderer.ResetAccumulation();

    const bool started = UseAsyncSceneLoading(_renderer.GetSettings())
        ? _renderer.LoadSceneAsync(normalizedPath)
        : _renderer.LoadScene(normalizedPath);
    if (started) {
        if (!UseAsyncSceneLoading(_renderer.GetSettings()) && _renderer.GetScene().IsLoaded()) {
            ApplyBenchmarkSceneLightingPreset(_renderer.GetSettings(), normalizedPath, &_renderer.GetScene().GetBounds());
            _renderer.ResetAccumulation();
        }
        remember_recent_scene(normalizedPath);
    }
}

void VestaEngine::apply_external_hdri_path(const std::filesystem::path& path)
{
    auto& settings = _renderer.GetSettings();
    settings.externalHdriPath = path;
    settings.externalHdriAvailable = false;
    settings.externalHdriIsHdr = false;
    settings.externalHdriWidth = 0;
    settings.externalHdriHeight = 0;
    settings.externalHdriChannels = 0;
    _renderer.ClearExternalEnvironmentMap();

    if (path.empty()) {
        settings.externalHdriStatus = "Procedural IBL";
        return;
    }

    std::filesystem::path normalizedPath = path;
    if (normalizedPath.is_relative()) {
        normalizedPath = std::filesystem::current_path() / normalizedPath;
    }
    settings.externalHdriPath = normalizedPath;

    if (!std::filesystem::exists(normalizedPath)) {
        settings.externalHdriStatus = "External HDRI missing: " + normalizedPath.string();
        log_startup_event(settings.externalHdriStatus);
        return;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_info(normalizedPath.string().c_str(), &width, &height, &channels) == 0 || width <= 0 || height <= 0) {
        settings.externalHdriStatus = "External HDRI probe failed: " + normalizedPath.string();
        log_startup_event(settings.externalHdriStatus);
        return;
    }

    if (!_renderer.LoadExternalEnvironmentMap(normalizedPath)) {
        settings.externalHdriStatus = "External HDRI upload failed: " + normalizedPath.string();
        log_startup_event(settings.externalHdriStatus);
        return;
    }

    settings.externalHdriAvailable = true;
    settings.externalHdriIsHdr = stbi_is_hdr(normalizedPath.string().c_str()) != 0;
    settings.externalHdriWidth = static_cast<uint32_t>(width);
    settings.externalHdriHeight = static_cast<uint32_t>(height);
    settings.externalHdriChannels = static_cast<uint32_t>(std::max(channels, 0));
    settings.externalHdriStatus = fmt::format("External environment uploaded: {}x{} {}ch {}",
        width,
        height,
        channels,
        settings.externalHdriIsHdr ? "HDR" : "LDR");
    log_startup_event(settings.externalHdriStatus);
}

void VestaEngine::remember_recent_scene(const std::filesystem::path& path)
{
    const std::filesystem::path normalizedPath = NormalizeScenePath(path);
    if (normalizedPath.empty()) {
        return;
    }

    const auto existing = std::find(_recentScenePaths.begin(), _recentScenePaths.end(), normalizedPath);
    if (existing != _recentScenePaths.end()) {
        _recentScenePaths.erase(existing);
    }

    _recentScenePaths.insert(_recentScenePaths.begin(), normalizedPath);
    if (_recentScenePaths.size() > kMaxRecentScenePaths) {
        _recentScenePaths.resize(kMaxRecentScenePaths);
    }
}
