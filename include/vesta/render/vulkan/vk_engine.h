// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <vesta/render/renderer.h>
#include <vesta/render/vulkan/vk_types.h>

struct BenchmarkConfig {
    std::filesystem::path csvOutputPath;
    std::filesystem::path screenshotOutputPath;
    float warmupSeconds{ 2.0f };
    float captureSeconds{ 10.0f };
};

struct EngineLaunchOptions {
    std::optional<std::filesystem::path> startupScenePath;
    std::optional<vesta::render::RendererPreset> startupPreset;
    std::optional<vesta::render::RendererDisplayMode> startupDisplayMode;
    std::optional<vesta::render::CompareMode> startupCompareMode;
    std::optional<vesta::render::RendererDebugView> startupDebugView;
    std::optional<vesta::render::PathTraceDebugView> startupPathTraceDebugView;
    std::optional<vesta::render::GaussianDebugView> startupGaussianDebugView;
    std::optional<float> startupCompareSplitPosition;
    std::optional<float> startupCompareDifferenceScale;
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
    bool _showFrameOverview{ true };
    bool _showRenderGraphPanel{ true };
    bool _showGpuProfilerPanel{ true };
    bool _showDebugVisualizationPanel{ true };
    bool _showSceneInspectorPanel{ true };
    bool _showResourceInspectorPanel{ true };
    bool _showLogConsolePanel{ true };
    bool _logShowInfo{ true };
    bool _logShowPerformance{ true };
    bool _logShowValidation{ true };
    bool _logShowResources{ true };
    bool _logShowErrors{ true };
    std::array<char, 128> _logFilterText{};
    bool _wireframeUiPlaceholder{ false };
    bool _overdrawUiPlaceholder{ false };
    int _lastCpuFrameWarningFrame{ -100000 };
    int _lastGpuFrameWarningFrame{ -100000 };
    int _lastPassWarningFrame{ -100000 };
    int _lastResourceWarningFrame{ -100000 };
    int _lastValidationWarningFrame{ -100000 };
    std::array<float, 240> _gpuFrameTimeHistoryMs{};
    size_t _gpuFrameTimeHistoryHead{ 0 };
    size_t _gpuFrameTimeHistoryCount{ 0 };
    VkDescriptorPool _imguiDescriptorPool{ VK_NULL_HANDLE };
    std::vector<std::filesystem::path> _recentScenePaths;
    std::vector<std::string> _logConsoleLines;
    std::vector<VkDescriptorSet> _texturePreviewDescriptors;
    uint64_t _texturePreviewSceneVersion{ 0 };

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
    void clear_texture_preview_descriptors();
    void begin_imgui_frame(float deltaSeconds);
    void build_main_menu_bar();
    void build_debug_ui();
    [[nodiscard]] bool should_forward_event_to_renderer(const union SDL_Event& event) const;
    [[nodiscard]] std::optional<std::filesystem::path> open_scene_with_system_dialog() const;
    [[nodiscard]] std::optional<std::filesystem::path> open_gaussian_model_with_system_dialog() const;
    void log_startup_event(std::string_view message);
    void update_runtime_warnings();
    void update_startup_state();
    void update_benchmark(float deltaSeconds);
    void finish_benchmark();
    bool request_screenshot_with_metadata(const std::filesystem::path& path, std::string_view captureKind);
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
        bool screenshotQueued{ false };
        std::vector<float> frameTimesMs;
        std::array<float, 7> passGpuMsSums{};
        std::array<uint32_t, 7> passGpuSampleCounts{};
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
