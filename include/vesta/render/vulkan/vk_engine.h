// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <vesta/render/renderer.h>
#include <vesta/render/vulkan/vk_types.h>

struct BenchmarkConfig {
    std::filesystem::path csvOutputPath;
    float warmupSeconds{ 2.0f };
    float captureSeconds{ 10.0f };
};

struct EngineLaunchOptions {
    std::optional<std::filesystem::path> startupScenePath;
    std::optional<vesta::render::RendererPreset> startupPreset;
    std::optional<vesta::render::RendererDisplayMode> startupDisplayMode;
    std::optional<vesta::render::PathTraceBackend> startupPathTraceBackend;
    std::optional<float> startupPathTraceResolutionScale;
    std::optional<BenchmarkConfig> benchmark;
    std::filesystem::path startupLogPath{ "out/startup.log" };
    bool safeStartupMode{ true };
    bool deferRayTracingBuildUntilAfterFirstPresent{ true };
    bool enableUi{ true };
    bool showDebugUi{ true };
};

[[nodiscard]] inline vesta::render::RendererSettings ApplyStartupSafeRendererSettings(
    vesta::render::RendererSettings settings, const EngineLaunchOptions& options)
{
    if (!options.safeStartupMode) {
        return settings;
    }

    settings.displayMode = vesta::render::RendererDisplayMode::DeferredLighting;
    settings.enableGaussian = false;
    settings.enablePathTracing = false;
    settings.buildRayTracingStructuresOnLoad =
        !options.deferRayTracingBuildUntilAfterFirstPresent ? settings.buildRayTracingStructuresOnLoad : false;
    settings.textureStreamingEnabled = false;
    settings.enableDistanceCulling = false;
    settings.useIndirectDraw = false;
    settings.sceneUploadMode = vesta::render::SceneUploadMode::Streaming;
    settings.preferAsyncSceneLoading = true;
    return settings;
}

class VestaEngine {
public:
    bool _isInitialized{ false };
    int _frameNumber{ 0 };
    bool stop_rendering{ false };
    // Window size is currently also the initial swapchain extent.
    VkExtent2D _windowExtent{ 1700, 900 };

    struct SDL_Window* _window{ nullptr };
    struct ImGuiContext* _imguiContext{ nullptr };
    bool _imguiInitialized{ false };
    bool _showDebugUi{ true };
    bool _showDetailedStats{ false };
    VkDescriptorPool _imguiDescriptorPool{ VK_NULL_HANDLE };
    std::vector<std::filesystem::path> _recentScenePaths;

    static VestaEngine& Get();

    // Owns the application loop around the renderer. The renderer itself stays
    // focused on Vulkan work; SDL event handling and ImGui live here.
    void init(const EngineLaunchOptions& options = {});
    void cleanup();
    void draw(float deltaSeconds);
    void run();

private:
    vesta::render::Renderer _renderer;

    void init_renderer();
    void init_imgui();
    void shutdown_imgui();
    void begin_imgui_frame(float deltaSeconds);
    void build_main_menu_bar();
    void build_debug_ui();
    [[nodiscard]] bool should_forward_event_to_renderer(const union SDL_Event& event) const;
    [[nodiscard]] std::optional<std::filesystem::path> open_scene_with_system_dialog() const;
    [[nodiscard]] std::optional<std::filesystem::path> open_gaussian_model_with_system_dialog() const;
    void log_startup_event(std::string_view message);
    void update_startup_state();
    void update_benchmark(float deltaSeconds);
    void finish_benchmark();
    void load_scene_path(const std::filesystem::path& path);
    void remember_recent_scene(const std::filesystem::path& path);

    EngineLaunchOptions _launchOptions{};
    struct BenchmarkState {
        bool started{ false };
        bool capturing{ false };
        bool completed{ false };
        float warmupElapsed{ 0.0f };
        float captureElapsed{ 0.0f };
        uint64_t lastGaussianRebuildCount{ 0 };
        uint32_t stableGaussianFrames{ 0 };
        std::vector<float> frameTimesMs;
    } _benchmarkState;
    struct StartupState {
        bool safeOverridesActive{ false };
        bool firstFramePresented{ false };
        bool startupSceneRequested{ false };
        bool startupSceneResolved{ false };
        vesta::render::RendererSettings savedSettings{};
        vesta::render::SceneLoadState lastSceneLoadState{ vesta::render::SceneLoadState::Idle };
        std::string lastSceneLoadMessage;
    } _startupState;
};
