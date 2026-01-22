#include <vesta/render/vulkan/vk_engine.h>

#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {
void PrintUsage()
{
    std::cout
        << "Usage: VestaEngine [options]\n"
        << "  --scene <path>                Load a scene at startup.\n"
        << "  --preset <recommended|performance|balanced|quality>\n"
        << "  --mode <composite|raster|deferred|gaussian|pathtrace>\n"
        << "  --compare <off|split|difference>\n"
        << "  --debug-view <final|albedo|normal|world-position|depth|uv|material-id|object-id|roughness|metallic|emissive|ao|motion-vector|direct|indirect|reflection|denoised|difference-reference|wireframe|mip-level|shadow-map|overdraw|history-color|history-depth|reprojection|disocclusion|jitter|contact-shadow>\n"
        << "  --pt-debug <final|albedo|normal|depth|direct|indirect|ray-count|diffuse-bounce|specular-bounce|throughput|pdf>\n"
        << "  --gaussian-debug <final|alpha|revealage|overdraw|depth|tile-occupancy|radius|contribution-count|splat-id|sh-band|covariance|raster-depth|composition-mask|depth-difference>\n"
        << "  --compare-split <0.05-0.95>\n"
        << "  --compare-scale <value>\n"
        << "  --pt-backend <auto|compute|hardwarert>\n"
        << "  --pt-scale <0.25-1.0>\n"
        << "  --pt-nee <on|off>             Toggle path tracing next-event estimation.\n"
        << "  --pt-rr <on|off>              Toggle path tracing Russian roulette termination.\n"
        << "  --pt-rr-depth <1-12>          Bounce depth where Russian roulette starts.\n"
        << "  --pt-firefly-clamp <value>    Clamp path throughput/fireflies; 0 disables.\n"
        << "  --ssao <on|off>               Toggle raster screen-space ambient occlusion.\n"
        << "  --ssao-radius <value>         SSAO world-space sample radius.\n"
        << "  --ssao-intensity <value>      SSAO darkening strength.\n"
        << "  --taa <on|off>                Toggle raster temporal anti-aliasing.\n"
        << "  --taa-feedback <0.0-0.98>     TAA history blend feedback.\n"
        << "  --ssr <on|off>                Toggle raster screen-space reflections.\n"
        << "  --ssr-distance <value>        SSR max ray distance.\n"
        << "  --ssr-thickness <value>       SSR depth thickness.\n"
        << "  --ssr-intensity <value>       SSR blend intensity.\n"
        << "  --ssgi <on|off>               Toggle screen-space global illumination.\n"
        << "  --ssgi-radius <value>         SSGI sample radius.\n"
        << "  --ssgi-intensity <value>      SSGI bounce strength.\n"
        << "  --ssgi-samples <4-16>         SSGI sample count.\n"
        << "  --shadow-pcss <on|off>        Toggle PCSS-style soft shadow filtering.\n"
        << "  --shadow-filter-radius <0.5-4> Shadow PCF/PCSS filter radius.\n"
        << "  --motion-blur <on|off>        Toggle screen-space motion blur.\n"
        << "  --motion-blur-strength <0-2>  Motion blur sample spread.\n"
        << "  --env-preset <studio|sunset|night|forest>\n"
        << "  --ibl-diffuse <0-2>           Diffuse environment lighting strength.\n"
        << "  --ibl-specular <0-2>          Specular environment reflection strength.\n"
        << "  --benchmark <csv-path>        Run a timed benchmark and exit.\n"
        << "  --screenshot <png-path>       Save a PNG capture during benchmark.\n"
        << "  --benchmark-seconds <value>   Benchmark capture duration.\n"
        << "  --warmup-seconds <value>      Benchmark warmup duration.\n"
        << "  --reload-shaders              Compile GLSL to runtime SPIR-V and reload passes on startup.\n"
        << "  --show-ui                     Force ImGui UI on.\n"
        << "  --no-ui                       Disable ImGui UI.\n"
        << "  --help                        Show this help.\n";
}

std::optional<vesta::render::RendererPreset> ParsePreset(std::string_view value)
{
    if (value == "recommended") {
        return vesta::render::RendererPreset::Recommended;
    }
    if (value == "performance") {
        return vesta::render::RendererPreset::Performance;
    }
    if (value == "balanced") {
        return vesta::render::RendererPreset::Balanced;
    }
    if (value == "quality") {
        return vesta::render::RendererPreset::Quality;
    }
    return std::nullopt;
}

std::optional<vesta::render::RendererDisplayMode> ParseDisplayMode(std::string_view value)
{
    if (value == "composite") {
        return vesta::render::RendererDisplayMode::Composite;
    }
    if (value == "deferred" || value == "raster") {
        return vesta::render::RendererDisplayMode::DeferredLighting;
    }
    if (value == "gaussian") {
        return vesta::render::RendererDisplayMode::Gaussian;
    }
    if (value == "pathtrace" || value == "path-trace") {
        return vesta::render::RendererDisplayMode::PathTrace;
    }
    return std::nullopt;
}

std::optional<vesta::render::PathTraceBackend> ParsePathTraceBackend(std::string_view value)
{
    if (value == "auto") {
        return vesta::render::PathTraceBackend::Auto;
    }
    if (value == "compute") {
        return vesta::render::PathTraceBackend::Compute;
    }
    if (value == "hardwarert" || value == "hardware-rt" || value == "rt") {
        return vesta::render::PathTraceBackend::HardwareRT;
    }
    return std::nullopt;
}

std::optional<uint32_t> ParseEnvironmentPreset(std::string_view value)
{
    if (value == "studio" || value == "default") {
        return 0u;
    }
    if (value == "sunset" || value == "warm") {
        return 1u;
    }
    if (value == "night" || value == "cool") {
        return 2u;
    }
    if (value == "forest" || value == "soft") {
        return 3u;
    }
    return std::nullopt;
}

std::optional<vesta::render::CompareMode> ParseCompareMode(std::string_view value)
{
    if (value == "off") {
        return vesta::render::CompareMode::Off;
    }
    if (value == "split" || value == "raster-path-split") {
        return vesta::render::CompareMode::RasterPathSplit;
    }
    if (value == "difference" || value == "diff" || value == "heatmap") {
        return vesta::render::CompareMode::DifferenceHeatmap;
    }
    return std::nullopt;
}

std::optional<vesta::render::RendererDebugView> ParseDebugView(std::string_view value)
{
    if (value == "final") { return vesta::render::RendererDebugView::FinalColor; }
    if (value == "albedo" || value == "base-color") { return vesta::render::RendererDebugView::Albedo; }
    if (value == "normal") { return vesta::render::RendererDebugView::Normal; }
    if (value == "world-position" || value == "position") { return vesta::render::RendererDebugView::WorldPosition; }
    if (value == "depth" || value == "linear-depth") { return vesta::render::RendererDebugView::Depth; }
    if (value == "uv") { return vesta::render::RendererDebugView::UV; }
    if (value == "material-id") { return vesta::render::RendererDebugView::MaterialId; }
    if (value == "object-id") { return vesta::render::RendererDebugView::ObjectId; }
    if (value == "roughness") { return vesta::render::RendererDebugView::Roughness; }
    if (value == "metallic") { return vesta::render::RendererDebugView::Metallic; }
    if (value == "emissive") { return vesta::render::RendererDebugView::Emissive; }
    if (value == "ao" || value == "ambient-occlusion" || value == "ssao") { return vesta::render::RendererDebugView::AmbientOcclusion; }
    if (value == "motion-vector" || value == "motion-vectors" || value == "velocity") { return vesta::render::RendererDebugView::MotionVector; }
    if (value == "direct" || value == "direct-lighting") { return vesta::render::RendererDebugView::DirectLighting; }
    if (value == "indirect" || value == "indirect-lighting" || value == "gi") { return vesta::render::RendererDebugView::IndirectLighting; }
    if (value == "reflection" || value == "reflections" || value == "ssr") { return vesta::render::RendererDebugView::Reflection; }
    if (value == "denoised" || value == "denoised-result" || value == "path-denoised") { return vesta::render::RendererDebugView::DenoisedResult; }
    if (value == "difference-reference" || value == "difference-from-reference" || value == "reference-difference" || value == "diff-reference") {
        return vesta::render::RendererDebugView::DifferenceFromReference;
    }
    if (value == "wireframe" || value == "edge" || value == "edge-overlay") { return vesta::render::RendererDebugView::Wireframe; }
    if (value == "mip-level" || value == "miplevel" || value == "mip") { return vesta::render::RendererDebugView::MipLevel; }
    if (value == "shadow-map" || value == "shadow" || value == "shadows") { return vesta::render::RendererDebugView::ShadowMap; }
    if (value == "overdraw" || value == "raster-overdraw") { return vesta::render::RendererDebugView::Overdraw; }
    if (value == "history-color" || value == "temporal-history" || value == "history") { return vesta::render::RendererDebugView::TemporalHistoryColor; }
    if (value == "history-depth" || value == "temporal-history-depth") { return vesta::render::RendererDebugView::TemporalHistoryDepth; }
    if (value == "reprojection" || value == "temporal-reprojection") { return vesta::render::RendererDebugView::TemporalReprojection; }
    if (value == "disocclusion" || value == "disocclusion-mask") { return vesta::render::RendererDebugView::TemporalDisocclusion; }
    if (value == "jitter" || value == "jitter-pattern" || value == "temporal-jitter") { return vesta::render::RendererDebugView::TemporalJitter; }
    if (value == "contact-shadow" || value == "contact-shadows") { return vesta::render::RendererDebugView::ContactShadow; }
    return std::nullopt;
}

std::optional<vesta::render::PathTraceDebugView> ParsePathTraceDebugView(std::string_view value)
{
    if (value == "final") { return vesta::render::PathTraceDebugView::Final; }
    if (value == "albedo") { return vesta::render::PathTraceDebugView::Albedo; }
    if (value == "normal") { return vesta::render::PathTraceDebugView::Normal; }
    if (value == "depth") { return vesta::render::PathTraceDebugView::Depth; }
    if (value == "direct") { return vesta::render::PathTraceDebugView::Direct; }
    if (value == "indirect") { return vesta::render::PathTraceDebugView::Indirect; }
    if (value == "ray-count" || value == "ray-heatmap" || value == "ray-count-heatmap") {
        return vesta::render::PathTraceDebugView::RayCountHeatmap;
    }
    if (value == "diffuse-bounce" || value == "diffuse") { return vesta::render::PathTraceDebugView::DiffuseBounce; }
    if (value == "specular-bounce" || value == "specular") { return vesta::render::PathTraceDebugView::SpecularBounce; }
    if (value == "throughput") { return vesta::render::PathTraceDebugView::Throughput; }
    if (value == "pdf") { return vesta::render::PathTraceDebugView::Pdf; }
    return std::nullopt;
}

std::optional<vesta::render::GaussianDebugView> ParseGaussianDebugView(std::string_view value)
{
    if (value == "final") { return vesta::render::GaussianDebugView::Final; }
    if (value == "alpha") { return vesta::render::GaussianDebugView::Alpha; }
    if (value == "revealage") { return vesta::render::GaussianDebugView::Revealage; }
    if (value == "overdraw" || value == "overdraw-heatmap") { return vesta::render::GaussianDebugView::OverdrawHeatmap; }
    if (value == "depth") { return vesta::render::GaussianDebugView::Depth; }
    if (value == "tile-occupancy" || value == "tiles") { return vesta::render::GaussianDebugView::TileOccupancy; }
    if (value == "radius" || value == "splat-radius") { return vesta::render::GaussianDebugView::SplatRadius; }
    if (value == "contribution-count" || value == "contributions" || value == "splat-count") { return vesta::render::GaussianDebugView::ContributionCount; }
    if (value == "splat-id" || value == "id" || value == "gaussian-id") { return vesta::render::GaussianDebugView::SplatId; }
    if (value == "sh-band" || value == "sh" || value == "spherical-harmonics") { return vesta::render::GaussianDebugView::ShBand; }
    if (value == "covariance" || value == "covariance-ellipsoid" || value == "ellipsoid") { return vesta::render::GaussianDebugView::Covariance; }
    if (value == "raster-depth" || value == "mesh-depth") { return vesta::render::GaussianDebugView::RasterDepth; }
    if (value == "composition-mask" || value == "hybrid-mask" || value == "compose-mask") { return vesta::render::GaussianDebugView::CompositionMask; }
    if (value == "depth-difference" || value == "depth-diff") { return vesta::render::GaussianDebugView::DepthDifference; }
    return std::nullopt;
}

bool TryParseFloat(const char* value, float& output)
{
    try {
        output = std::stof(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool TryParseUint(const char* value, uint32_t& output)
{
    try {
        output = static_cast<uint32_t>(std::stoul(value));
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<bool> ParseToggle(std::string_view value)
{
    if (value == "on" || value == "true" || value == "1" || value == "yes") {
        return true;
    }
    if (value == "off" || value == "false" || value == "0" || value == "no") {
        return false;
    }
    return std::nullopt;
}
} // namespace

int main(int argc, char* argv[])
{
    // Keep main intentionally small: all interesting lifetime management happens
    // inside VestaEngine so startup and shutdown order stays explicit.
    EngineLaunchOptions options;
    bool uiExplicit = false;

    for (int argIndex = 1; argIndex < argc; ++argIndex) {
        const std::string_view argument = argv[argIndex];
        auto requireValue = [&](std::string_view flag) -> const char* {
            if (argIndex + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++argIndex];
        };

        if (argument == "--help") {
            PrintUsage();
            return 0;
        }
        if (argument == "--scene") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupScenePath = value;
            continue;
        }
        if (argument == "--preset") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupPreset = ParsePreset(value);
            if (!options.startupPreset.has_value()) {
                std::cerr << "Unknown preset: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--mode") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupDisplayMode = ParseDisplayMode(value);
            if (!options.startupDisplayMode.has_value()) {
                std::cerr << "Unknown mode: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--pt-backend") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupPathTraceBackend = ParsePathTraceBackend(value);
            if (!options.startupPathTraceBackend.has_value()) {
                std::cerr << "Unknown PT backend: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--compare") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupCompareMode = ParseCompareMode(value);
            if (!options.startupCompareMode.has_value()) {
                std::cerr << "Unknown compare mode: " << value << "\n";
                return 1;
            }
            options.startupDisplayMode = vesta::render::RendererDisplayMode::Composite;
            continue;
        }
        if (argument == "--debug-view") {
            const char* value = requireValue(argument);
            if (value == nullptr) { return 1; }
            options.startupDebugView = ParseDebugView(value);
            if (!options.startupDebugView.has_value()) {
                std::cerr << "Unknown debug view: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--pt-debug") {
            const char* value = requireValue(argument);
            if (value == nullptr) { return 1; }
            options.startupPathTraceDebugView = ParsePathTraceDebugView(value);
            if (!options.startupPathTraceDebugView.has_value()) {
                std::cerr << "Unknown PT debug view: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--gaussian-debug") {
            const char* value = requireValue(argument);
            if (value == nullptr) { return 1; }
            options.startupGaussianDebugView = ParseGaussianDebugView(value);
            if (!options.startupGaussianDebugView.has_value()) {
                std::cerr << "Unknown Gaussian debug view: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--compare-split") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float split = 0.0f;
            if (!TryParseFloat(value, split) || split < 0.05f || split > 0.95f) {
                std::cerr << "Invalid compare split: " << value << "\n";
                return 1;
            }
            options.startupCompareSplitPosition = split;
            continue;
        }
        if (argument == "--compare-scale") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float scale = 0.0f;
            if (!TryParseFloat(value, scale) || scale <= 0.0f) {
                std::cerr << "Invalid compare scale: " << value << "\n";
                return 1;
            }
            options.startupCompareDifferenceScale = scale;
            continue;
        }
        if (argument == "--pt-scale") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float scale = 0.0f;
            if (!TryParseFloat(value, scale)) {
                std::cerr << "Invalid PT scale: " << value << "\n";
                return 1;
            }
            options.startupPathTraceResolutionScale = scale;
            continue;
        }
        if (argument == "--pt-nee") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupPathTraceNextEventEstimation = ParseToggle(value);
            if (!options.startupPathTraceNextEventEstimation.has_value()) {
                std::cerr << "Invalid PT NEE toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--pt-rr") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupPathTraceRussianRoulette = ParseToggle(value);
            if (!options.startupPathTraceRussianRoulette.has_value()) {
                std::cerr << "Invalid PT Russian roulette toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--pt-rr-depth") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            uint32_t depth = 0u;
            if (!TryParseUint(value, depth) || depth < 1u || depth > 12u) {
                std::cerr << "Invalid PT Russian roulette depth: " << value << "\n";
                return 1;
            }
            options.startupPathTraceRussianRouletteDepth = depth;
            continue;
        }
        if (argument == "--pt-firefly-clamp") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float clampValue = 0.0f;
            if (!TryParseFloat(value, clampValue) || clampValue < 0.0f) {
                std::cerr << "Invalid PT firefly clamp: " << value << "\n";
                return 1;
            }
            options.startupPathTraceFireflyClamp = clampValue;
            continue;
        }
        if (argument == "--ssao") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupSsaoEnabled = ParseToggle(value);
            if (!options.startupSsaoEnabled.has_value()) {
                std::cerr << "Invalid SSAO toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--ssao-radius") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float radius = 0.0f;
            if (!TryParseFloat(value, radius) || radius <= 0.0f) {
                std::cerr << "Invalid SSAO radius: " << value << "\n";
                return 1;
            }
            options.startupSsaoRadius = radius;
            continue;
        }
        if (argument == "--ssao-intensity") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float intensity = 0.0f;
            if (!TryParseFloat(value, intensity) || intensity < 0.0f) {
                std::cerr << "Invalid SSAO intensity: " << value << "\n";
                return 1;
            }
            options.startupSsaoIntensity = intensity;
            continue;
        }
        if (argument == "--taa") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupTaaEnabled = ParseToggle(value);
            if (!options.startupTaaEnabled.has_value()) {
                std::cerr << "Invalid TAA toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--taa-feedback") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float feedback = 0.0f;
            if (!TryParseFloat(value, feedback) || feedback < 0.0f || feedback > 0.98f) {
                std::cerr << "Invalid TAA feedback: " << value << "\n";
                return 1;
            }
            options.startupTaaFeedback = feedback;
            continue;
        }
        if (argument == "--ssr") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupSsrEnabled = ParseToggle(value);
            if (!options.startupSsrEnabled.has_value()) {
                std::cerr << "Invalid SSR toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--ssr-distance") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float distance = 0.0f;
            if (!TryParseFloat(value, distance) || distance <= 0.0f) {
                std::cerr << "Invalid SSR distance: " << value << "\n";
                return 1;
            }
            options.startupSsrMaxDistance = distance;
            continue;
        }
        if (argument == "--ssr-thickness") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float thickness = 0.0f;
            if (!TryParseFloat(value, thickness) || thickness <= 0.0f) {
                std::cerr << "Invalid SSR thickness: " << value << "\n";
                return 1;
            }
            options.startupSsrThickness = thickness;
            continue;
        }
        if (argument == "--ssr-intensity") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float intensity = 0.0f;
            if (!TryParseFloat(value, intensity) || intensity < 0.0f) {
                std::cerr << "Invalid SSR intensity: " << value << "\n";
                return 1;
            }
            options.startupSsrIntensity = intensity;
            continue;
        }
        if (argument == "--ssgi") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupSsgiEnabled = ParseToggle(value);
            if (!options.startupSsgiEnabled.has_value()) {
                std::cerr << "Invalid SSGI toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--ssgi-radius") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float radius = 0.0f;
            if (!TryParseFloat(value, radius) || radius <= 0.0f) {
                std::cerr << "Invalid SSGI radius: " << value << "\n";
                return 1;
            }
            options.startupSsgiRadius = radius;
            continue;
        }
        if (argument == "--ssgi-intensity") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float intensity = 0.0f;
            if (!TryParseFloat(value, intensity) || intensity < 0.0f) {
                std::cerr << "Invalid SSGI intensity: " << value << "\n";
                return 1;
            }
            options.startupSsgiIntensity = intensity;
            continue;
        }
        if (argument == "--ssgi-samples") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            uint32_t samples = 0u;
            if (!TryParseUint(value, samples) || samples < 4u || samples > 16u) {
                std::cerr << "Invalid SSGI sample count: " << value << "\n";
                return 1;
            }
            options.startupSsgiSamples = samples;
            continue;
        }
        if (argument == "--shadow-pcss") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupPcssShadowsEnabled = ParseToggle(value);
            if (!options.startupPcssShadowsEnabled.has_value()) {
                std::cerr << "Invalid shadow PCSS toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--shadow-filter-radius") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float radius = 0.0f;
            if (!TryParseFloat(value, radius) || radius < 0.5f || radius > 4.0f) {
                std::cerr << "Invalid shadow filter radius: " << value << "\n";
                return 1;
            }
            options.startupShadowFilterRadius = radius;
            continue;
        }
        if (argument == "--motion-blur") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupMotionBlurEnabled = ParseToggle(value);
            if (!options.startupMotionBlurEnabled.has_value()) {
                std::cerr << "Invalid motion blur toggle: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--motion-blur-strength") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float strength = 0.0f;
            if (!TryParseFloat(value, strength) || strength < 0.0f || strength > 2.0f) {
                std::cerr << "Invalid motion blur strength: " << value << "\n";
                return 1;
            }
            options.startupMotionBlurStrength = strength;
            continue;
        }
        if (argument == "--env-preset") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            options.startupEnvironmentPreset = ParseEnvironmentPreset(value);
            if (!options.startupEnvironmentPreset.has_value()) {
                std::cerr << "Unknown environment preset: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (argument == "--ibl-diffuse") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float strength = 0.0f;
            if (!TryParseFloat(value, strength) || strength < 0.0f || strength > 2.0f) {
                std::cerr << "Invalid IBL diffuse strength: " << value << "\n";
                return 1;
            }
            options.startupEnvironmentDiffuseStrength = strength;
            continue;
        }
        if (argument == "--ibl-specular") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float strength = 0.0f;
            if (!TryParseFloat(value, strength) || strength < 0.0f || strength > 2.0f) {
                std::cerr << "Invalid IBL specular strength: " << value << "\n";
                return 1;
            }
            options.startupEnvironmentSpecularStrength = strength;
            continue;
        }
        if (argument == "--benchmark") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            BenchmarkConfig benchmark;
            benchmark.csvOutputPath = value;
            options.benchmark = benchmark;
            continue;
        }
        if (argument == "--screenshot") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            if (!options.benchmark.has_value()) {
                options.benchmark = BenchmarkConfig{};
            }
            options.benchmark->screenshotOutputPath = value;
            continue;
        }
        if (argument == "--benchmark-seconds") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float seconds = 0.0f;
            if (!TryParseFloat(value, seconds) || seconds <= 0.0f) {
                std::cerr << "Invalid benchmark duration: " << value << "\n";
                return 1;
            }
            if (!options.benchmark.has_value()) {
                options.benchmark = BenchmarkConfig{};
            }
            options.benchmark->captureSeconds = seconds;
            continue;
        }
        if (argument == "--warmup-seconds") {
            const char* value = requireValue(argument);
            if (value == nullptr) {
                return 1;
            }
            float seconds = 0.0f;
            if (!TryParseFloat(value, seconds) || seconds < 0.0f) {
                std::cerr << "Invalid warmup duration: " << value << "\n";
                return 1;
            }
            if (!options.benchmark.has_value()) {
                options.benchmark = BenchmarkConfig{};
            }
            options.benchmark->warmupSeconds = seconds;
            continue;
        }
        if (argument == "--reload-shaders") {
            options.reloadShadersOnStartup = true;
            continue;
        }
        if (argument == "--show-ui") {
            options.enableUi = true;
            options.showDebugUi = true;
            uiExplicit = true;
            continue;
        }
        if (argument == "--no-ui") {
            options.enableUi = false;
            options.showDebugUi = false;
            uiExplicit = true;
            continue;
        }

        std::cerr << "Unknown argument: " << argument << "\n";
        PrintUsage();
        return 1;
    }

    if (options.benchmark.has_value() && options.benchmark->csvOutputPath.empty()) {
        options.benchmark->csvOutputPath = "out/benchmark.csv";
    }
    if (options.benchmark.has_value() && !uiExplicit) {
        options.enableUi = false;
        options.showDebugUi = false;
    }

    VestaEngine engine;
    engine.init(options);
    engine.run();
    engine.cleanup();
    return 0;
}
