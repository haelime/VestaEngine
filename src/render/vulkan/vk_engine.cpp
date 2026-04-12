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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string_view>
#include <thread>

#include <vesta/core/debug.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <commdlg.h>
#include <windows.h>
#pragma comment(lib, "Comdlg32.lib")
#endif

VestaEngine* loadedEngine = nullptr;

VestaEngine& VestaEngine::Get() { return *loadedEngine; }

namespace {
constexpr size_t kMaxRecentScenePaths = 5;

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
    case vesta::render::RendererDisplayMode::Gaussian:
        return "Gaussian";
    case vesta::render::RendererDisplayMode::PathTrace:
        return "PathTrace";
    case vesta::render::RendererDisplayMode::Composite:
    default:
        return "Composite";
    }
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
    case vesta::render::SceneLoadState::Idle:
    default:
        return "Idle";
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

void ApplySceneModeInference(vesta::render::RendererSettings& settings, const std::filesystem::path& path)
{
    const std::filesystem::path extension = path.extension();
    if (extension == ".ply" || extension == ".PLY") {
        settings.displayMode = vesta::render::RendererDisplayMode::Gaussian;
        settings.enableGaussian = true;
        settings.enablePathTracing = false;
        return;
    }

    if (extension == ".glb" || extension == ".GLB" || extension == ".gltf" || extension == ".GLTF") {
        settings.displayMode = vesta::render::RendererDisplayMode::DeferredLighting;
        settings.enableRaster = true;
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
        settings.displayMode = *_launchOptions.startupDisplayMode;
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

    auto requestSceneLoad = [&](const std::filesystem::path& path) {
        _startupState.startupSceneRequested = true;
        log_startup_event(std::string("Startup scene requested: ") + path.string());
        if (!_launchOptions.startupDisplayMode.has_value()) {
            ApplySceneModeInference(settings, path);
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
    update_startup_state();
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
    }
}

void VestaEngine::log_startup_event(std::string_view message)
{
    const std::string line = MakeTimestampedLogLine(message);
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

void VestaEngine::update_startup_state()
{
    const auto& sceneLoadStatus = _renderer.GetSceneLoadStatus();
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

    if (!_startupState.firstFramePresented) {
        _startupState.firstFramePresented = true;
        log_startup_event("First frame presented");
    }

    if (_startupState.safeOverridesActive && _startupState.startupSceneResolved && _startupState.firstFramePresented) {
        _renderer.GetSettings() = _startupState.savedSettings;
        if (!_launchOptions.startupDisplayMode.has_value() && !_renderer.GetScene().GetSourcePath().empty()) {
            ApplySceneModeInference(_renderer.GetSettings(), _renderer.GetScene().GetSourcePath());
        }
        _renderer.SetStartupSafeModeActive(false);
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
        fmt::println("Benchmark warmup for {:.1f}s", benchmark.warmupSeconds);
    }

    if (!_benchmarkState.capturing) {
        _benchmarkState.warmupElapsed += deltaSeconds;
        if (_benchmarkState.warmupElapsed < benchmark.warmupSeconds) {
            return;
        }

        _benchmarkState.capturing = true;
        _benchmarkState.captureElapsed = 0.0f;
        _benchmarkState.frameTimesMs.clear();
        fmt::println(
            "Capturing benchmark for {:.1f}s -> {}", benchmark.captureSeconds, benchmark.csvOutputPath.string());
        return;
    }

    _benchmarkState.frameTimesMs.push_back(_renderer.GetFrameTimeMs());
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
        output << "timestamp,scene,gpu,resolution,display_mode,requested_backend,active_backend,scene_upload_mode,"
               << "gaussian,path_tracing,texture_streaming,indirect_draw,frustum_culling,distance_culling,"
               << "pt_scale,avg_frame_ms,p95_frame_ms,min_frame_ms,max_frame_ms,avg_fps,frame_count,"
               << "vertices,triangles,surfaces,textures_total,textures_resident,parse_ms,prepare_ms,"
               << "geometry_upload_ms,texture_upload_ms,blas_ms,tlas_ms\n";
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
    const auto extent = _renderer.GetRenderDevice().GetSwapchainExtent();

    output << CsvEscape(timestampStream.str()) << ','
           << CsvEscape(scene.GetSourcePath().string()) << ','
           << CsvEscape(_renderer.GetRenderDevice().GetGpuName()) << ','
           << CsvEscape(fmt::format("{}x{}", extent.width, extent.height)) << ','
           << DisplayModeLabel(settings.displayMode) << ','
           << PathTraceBackendLabel(settings.pathTraceBackend) << ','
           << PathTraceBackendLabel(_renderer.GetActivePathTraceBackend()) << ','
           << CsvEscape(SceneUploadModeLabel(settings.sceneUploadMode)) << ','
           << (settings.enableGaussian ? "true" : "false") << ','
           << (settings.enablePathTracing ? "true" : "false") << ','
           << (settings.textureStreamingEnabled ? "true" : "false") << ','
           << (settings.useIndirectDraw ? "true" : "false") << ','
           << (settings.enableFrustumCulling ? "true" : "false") << ','
           << (settings.enableDistanceCulling ? "true" : "false") << ','
           << settings.pathTraceResolutionScale << ','
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
           << status.parseMs << ','
           << status.prepareMs << ','
           << status.geometryUploadMs << ','
           << status.textureUploadMs << ','
           << status.blasMs << ','
           << status.tlasMs << '\n';

    fmt::println("Benchmark written to {}", outputPath.string());
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
    build_debug_ui();
    ImGui::Render();
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
            if (ImGui::MenuItem("Open Scene...", nullptr, false, !sceneLoadInProgress)) {
                if (const std::optional<std::filesystem::path> path = open_scene_with_system_dialog()) {
                    load_scene_path(*path);
                }
            }
            if (ImGui::BeginMenu("Open Recent", !_recentScenePaths.empty())) {
                for (const std::filesystem::path& recentPath : _recentScenePaths) {
                    if (ImGui::MenuItem(recentPath.filename().string().c_str(), nullptr, false, !sceneLoadInProgress)) {
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
                settings.displayMode = vesta::render::RendererDisplayMode::Composite;
                _renderer.ResetAccumulation();
            }

            bool deferredSelected = settings.displayMode == vesta::render::RendererDisplayMode::DeferredLighting;
            if (ImGui::MenuItem("Raster", nullptr, deferredSelected)) {
                settings.displayMode = vesta::render::RendererDisplayMode::DeferredLighting;
                _renderer.ResetAccumulation();
            }

            bool gaussianSelected = settings.displayMode == vesta::render::RendererDisplayMode::Gaussian;
            if (ImGui::MenuItem("Gaussian", nullptr, gaussianSelected)) {
                settings.displayMode = vesta::render::RendererDisplayMode::Gaussian;
                _renderer.ResetAccumulation();
            }

            bool pathTraceSelected = settings.displayMode == vesta::render::RendererDisplayMode::PathTrace;
            if (ImGui::MenuItem("Path Trace", nullptr, pathTraceSelected)) {
                settings.displayMode = vesta::render::RendererDisplayMode::PathTrace;
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
                int uploadBudgetMiB = static_cast<int>(settings.maxUploadBytesPerFrame / (1024u * 1024u));
                if (ImGui::SliderInt("Upload Budget (MiB)", &uploadBudgetMiB, 1, 32)) {
                    settings.maxUploadBytesPerFrame = static_cast<uint32_t>(uploadBudgetMiB) * 1024u * 1024u;
                }
                int textureUploadBudgetMiB = static_cast<int>(settings.maxTextureUploadBytesPerFrame / (1024u * 1024u));
                if (ImGui::SliderInt("Texture Budget (MiB)", &textureUploadBudgetMiB, 1, 64)) {
                    settings.maxTextureUploadBytesPerFrame = static_cast<uint32_t>(textureUploadBudgetMiB) * 1024u * 1024u;
                }
                ImGui::SliderFloat("Distance Cull Scale", &settings.distanceCullScale, 1.0f, 12.0f, "%.1f");
                ImGui::Separator();
                ImGui::TextDisabled("Validation: %s", bUseValidationLayers ? "Debug default" : "Off");
                ImGui::EndMenu();
            }

            if (ImGui::MenuItem("Enable Raster", nullptr, settings.enableRaster)) {
                settings.enableRaster = !settings.enableRaster;
                _renderer.ResetAccumulation();
            }
            if (ImGui::MenuItem("Enable Gaussian", nullptr, settings.enableGaussian)) {
                settings.enableGaussian = !settings.enableGaussian;
                _renderer.ResetAccumulation();
            }
            if (ImGui::MenuItem("Enable Path Tracing", nullptr, settings.enablePathTracing)) {
                settings.enablePathTracing = !settings.enablePathTracing;
                _renderer.ResetAccumulation();
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
                if (ImGui::MenuItem("Select For Drag")) {
                    _renderer.SelectDirectionalLight();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        const std::string& sceneStatus = _renderer.GetSceneLoadStatusMessage();
        if (!sceneStatus.empty()) {
            ImGui::Separator();
            ImGui::TextDisabled("%s", sceneStatus.c_str());
        }

        ImGui::EndMainMenuBar();
    }
}

void VestaEngine::build_debug_ui()
{
    if (!_imguiInitialized || !_showDebugUi) {
        return;
    }

    ImGui::SetCurrentContext(_imguiContext);

    auto& settings = _renderer.GetSettings();
    const auto& scene = _renderer.GetScene();
    const auto& device = _renderer.GetRenderDevice();
    const float frameMs = _renderer.GetSmoothedFrameTimeMs();
    const float fps = frameMs > 0.0f ? 1000.0f / frameMs : 0.0f;

    ImGui::SetNextWindowPos(ImVec2(18.0f, 18.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Vesta Overlay", nullptr, ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::End();
        return;
    }

    ImGui::Text("Frame %.2f ms", frameMs);
    ImGui::Text("FPS %.1f", fps);
    ImGui::Text("%s", device.GetGpuName().c_str());
    ImGui::Text("VRAM %u MiB", device.GetDedicatedVideoMemoryMiB());
    ImGui::Text("Recommended %s", PresetLabel(_renderer.GetRecommendedPreset()));
    ImGui::Text("Scene Upload %s", SceneUploadModeLabel(settings.sceneUploadMode));
    ImGui::Text("Skip Hidden Passes %s", settings.optimizeInactivePasses ? "On" : "Off");
    ImGui::Text("Device Local Buffers %s", settings.useDeviceLocalSceneBuffers ? "On" : "Off");
    ImGui::Text("Device Local Textures %s", settings.useDeviceLocalTextures ? "On" : "Off");
    ImGui::Text("Deferred Scene Free %s", settings.deferOldSceneDestruction ? "On" : "Off");
    ImGui::Text("Frustum Culling %s", settings.enableFrustumCulling ? "On" : "Off");
    ImGui::Text("Distance Culling %s", settings.enableDistanceCulling ? "On" : "Off");
    ImGui::Text("Indirect Draw %s", settings.useIndirectDraw ? "On" : "Off");
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
    ImGui::Separator();

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
    ImGui::Separator();

    const char* displayModes[] = { "Composite", "Raster", "Gaussian", "Path Trace" };
    int displayMode = static_cast<int>(settings.displayMode);
    if (ImGui::Combo("Display", &displayMode, displayModes, IM_ARRAYSIZE(displayModes))) {
        settings.displayMode = static_cast<vesta::render::RendererDisplayMode>(displayMode);
        _renderer.ResetAccumulation();
    }

    if (ImGui::Checkbox("Raster", &settings.enableRaster)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Gaussian", &settings.enableGaussian)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::Checkbox("Path Tracing", &settings.enablePathTracing)) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Point Size", &settings.gaussianPointSize, 1.0f, 24.0f, "%.1f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Point Opacity", &settings.gaussianOpacity, 0.05f, 1.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("Gaussian Mix", &settings.gaussianMix, 0.0f, 1.0f, "%.2f")) {
        _renderer.ResetAccumulation();
    }
    if (ImGui::SliderFloat("PT Resolution", &settings.pathTraceResolutionScale, 0.25f, 1.0f, "%.2fx")) {
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

    ImGui::SeparatorText("Scene");
    ImGui::Text("%s", scene.GetSourcePath().empty() ? "No scene" : scene.GetSourcePath().filename().string().c_str());
    ImGui::Text("Scene Type %s", SceneKindLabel(scene.GetSceneKind()));
    ImGui::Text("Recommended View %s", DisplayModeLabel(_renderer.GetRecommendedDisplayModeForScene()));
    ImGui::Text("Vertices %zu", scene.GetVertices().size());
    ImGui::Text("Triangles %zu", scene.GetTriangles().size());
    ImGui::Text("Surfaces %zu", scene.GetSurfaces().size());
    ImGui::Text("Objects %zu", scene.GetObjects().size());
    ImGui::Text("Textures %zu / %u", scene.GetTextures().size(), _renderer.GetResidentTextureCount());
    ImGui::Text("Visible Surfaces %u", _renderer.GetVisibleSurfaceCount());
    const auto& sceneLoadStatus = _renderer.GetSceneLoadStatus();
    ImGui::Text("Load State %s", SceneLoadStateLabel(sceneLoadStatus.state));
    if (!sceneLoadStatus.message.empty()) {
        ImGui::TextWrapped("%s", sceneLoadStatus.message.c_str());
    }
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
    ImGui::Text("RT Support %s", _renderer.GetRenderDevice().IsRayTracingSupported() ? "Yes" : "No");
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
    ImGui::Text("Active PT %s", activeBackend);
    if (scene.HasRayTracingScene()) {
        ImGui::Text("BLAS %.2f ms", scene.GetBottomLevelBuildMs());
        ImGui::Text("TLAS %.2f ms", scene.GetTopLevelBuildMs());
    }

    ImGui::SeparatorText("Selection");
    ImGui::Text("Selected %s", _renderer.GetSelectionLabel().c_str());
    if (ImGui::Button("Select Light")) {
        _renderer.SelectDirectionalLight();
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
        ImGui::PlotLines("Frame Times", history.data(), sampleCount, 0, nullptr, 0.0f, std::max(33.0f, peakMs * 1.1f), ImVec2(0.0f, 72.0f));
    }

    ImGui::SeparatorText("Controls");
    ImGui::Text("RMB + Mouse Look");
    ImGui::Text("WASD / Q / E Move");
    ImGui::Text("LMB Pick/Drag Object");
    ImGui::Text("L Select Light, Esc Clear Selection");
    ImGui::Text("1 Raster, 2 Gaussian, 3 PT, 4 Composite");
    ImGui::Text("R/G/P toggles, F1 UI");
    ImGui::End();
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
        L"Supported Scenes (*.glb;*.gltf;*.ply)\0*.glb;*.gltf;*.ply\0glTF Scenes (*.glb;*.gltf)\0*.glb;*.gltf\0Gaussian PLY (*.ply)\0*.ply\0All Files (*.*)\0*.*\0";
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

void VestaEngine::load_scene_path(const std::filesystem::path& path)
{
    if (path.empty()) {
        return;
    }

    ApplySceneModeInference(_renderer.GetSettings(), path);
    _renderer.ResetAccumulation();

    const bool started = UseAsyncSceneLoading(_renderer.GetSettings()) ? _renderer.LoadSceneAsync(path) : _renderer.LoadScene(path);
    if (started) {
        remember_recent_scene(path);
    }
}

void VestaEngine::remember_recent_scene(const std::filesystem::path& path)
{
    if (path.empty()) {
        return;
    }

    const auto existing = std::find(_recentScenePaths.begin(), _recentScenePaths.end(), path);
    if (existing != _recentScenePaths.end()) {
        _recentScenePaths.erase(existing);
    }

    _recentScenePaths.insert(_recentScenePaths.begin(), path);
    if (_recentScenePaths.size() > kMaxRecentScenePaths) {
        _recentScenePaths.resize(kMaxRecentScenePaths);
    }
}
