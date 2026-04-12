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
        << "  --pt-backend <auto|compute|hardwarert>\n"
        << "  --pt-scale <0.25-1.0>\n"
        << "  --benchmark <csv-path>        Run a timed benchmark and exit.\n"
        << "  --benchmark-seconds <value>   Benchmark capture duration.\n"
        << "  --warmup-seconds <value>      Benchmark warmup duration.\n"
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

bool TryParseFloat(const char* value, float& output)
{
    try {
        output = std::stof(value);
        return true;
    } catch (...) {
        return false;
    }
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
