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
    std::optional<bool> startupPathTraceNextEventEstimation;
    std::optional<bool> startupPathTraceRussianRoulette;
    std::optional<uint32_t> startupPathTraceRussianRouletteDepth;
    std::optional<float> startupPathTraceFireflyClamp;
    std::optional<bool> startupSsaoEnabled;
    std::optional<float> startupSsaoRadius;
    std::optional<float> startupSsaoIntensity;
    std::optional<bool> startupTaaEnabled;
    std::optional<float> startupTaaFeedback;
    std::optional<bool> startupSsrEnabled;
    std::optional<float> startupSsrMaxDistance;
    std::optional<float> startupSsrThickness;
    std::optional<float> startupSsrIntensity;
    std::optional<bool> startupSsgiEnabled;
    std::optional<float> startupSsgiRadius;
    std::optional<float> startupSsgiIntensity;
    std::optional<uint32_t> startupSsgiSamples;
    std::optional<bool> startupDdgiEnabled;
    std::optional<bool> startupRtShadowsEnabled;
    std::optional<bool> startupRtAoEnabled;
    std::optional<bool> startupRtReflectionsEnabled;
    std::optional<bool> startupRtHalfResolution;
    std::optional<float> startupRtMaxRayDistance;
    std::optional<float> startupRtAoRadius;
    std::optional<bool> startupMotionBlurEnabled;
    std::optional<float> startupMotionBlurStrength;
    std::optional<uint32_t> startupEnvironmentPreset;
    std::optional<std::filesystem::path> startupExternalHdriPath;
    std::optional<float> startupEnvironmentDiffuseStrength;
    std::optional<float> startupEnvironmentSpecularStrength;
    std::optional<bool> startupPcssShadowsEnabled;
    std::optional<float> startupShadowFilterRadius;
    std::optional<BenchmarkConfig> benchmark;
    std::filesystem::path startupLogPath{ "out/startup.log" };
    bool safeStartupMode{ true };
    bool deferRayTracingBuildUntilAfterFirstPresent{ true };
    bool enableUi{ true };
    bool showDebugUi{ false };
    bool reloadShadersOnStartup{ false };
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
    bool _showDebugUi{ false };
    bool _showDetailedStats{ false };
    bool _showFrameOverview{ false };
    bool _showRenderGraphPanel{ false };
    bool _showGpuProfilerPanel{ false };
    bool _showDebugVisualizationPanel{ false };
    bool _showRenderModeControlPanel{ false };
    bool _showSceneInspectorPanel{ false };
    bool _showResourceInspectorPanel{ false };
    bool _showLogConsolePanel{ false };
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
    std::vector<VkDescriptorSet> _frameTexturePreviewDescriptors;
    std::vector<vesta::render::ImageHandle> _frameTexturePreviewImages;
    std::vector<VkDescriptorSet> _engineTexturePreviewDescriptors;
    std::vector<vesta::render::ImageHandle> _engineTexturePreviewImages;
    uint64_t _texturePreviewSceneVersion{ 0 };
    size_t _selectedTexturePreviewIndex{ 0 };
    size_t _selectedFrameTexturePreviewIndex{ 0 };
    size_t _selectedEngineTexturePreviewIndex{ 0 };
    uint32_t _selectedGaussianInspectorIndex{ 0 };
    float _gaussianInspectorOverlayScale{ 1.0f };
    bool _gaussianInspectorShowAxes{ true };
    int _selectedBufferInspectorIndex{ 0 };

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
    void build_debug_dockspace();
    void build_debug_ui();
    void draw_light_gizmo_overlay();
    void build_render_mode_control_panel();
    void draw_killer_demo_panel();
    void draw_rasterizer_debug_panel();
    void draw_path_tracing_debug_panel();
    void draw_gaussian_splatting_debug_panel();
    void draw_ray_tracing_debug_panel();
    void draw_global_illumination_panel();
    void draw_post_process_panel();
    void draw_advanced_portfolio_panel();
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
    void apply_external_hdri_path(const std::filesystem::path& path);
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
        std::array<float, 11> passGpuMsSums{};
        std::array<uint32_t, 11> passGpuSampleCounts{};
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
