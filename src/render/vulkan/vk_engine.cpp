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
#include <functional>
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
#include <shlobj.h>
#include <shobjidl.h>
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
    case vesta::render::PathTraceDebugView::Final:
    default:
        return "Final";
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
        ImGui::TextUnformatted(BufferUsageLabel(buffer.desc.usage).c_str());
    } else {
        ImGui::TextUnformatted("-");
    }
    ImGui::TableSetColumnIndex(5);
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
    ImGui::TableSetColumnIndex(6);
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
        settings.displayMode = vesta::render::RendererDisplayMode::Gaussian;
        settings.enableRaster = false;
        settings.enableGaussian = true;
        settings.enablePathTracing = false;
        return;
    }

    if (extension == ".glb" || extension == ".GLB" || extension == ".gltf" || extension == ".GLTF") {
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
                const std::filesystem::path path = MakeTimestampedCapturePath("screenshot", ".ppm");
                log_startup_event(_renderer.RequestScreenshot(path) ? "Screenshot queued: " + path.string() : "Screenshot failed");
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
        output << "timestamp,scene,scene_kind,gpu,resolution,display_mode,requested_backend,active_backend,scene_upload_mode,"
               << "gaussian,path_tracing,texture_streaming,indirect_draw,frustum_culling,distance_culling,"
               << "gaussian_trained,gaussian_count,gaussian_sh_degree,gaussian_view_dependent_color,gaussian_antialiasing,"
               << "gaussian_fast_culling,gaussian_opacity,gaussian_mix,gaussian_interactive_preview,"
               << "pt_scale,avg_frame_ms,p95_frame_ms,min_frame_ms,max_frame_ms,avg_fps,frame_count,"
               << "vertices,triangles,surfaces,textures_total,textures_resident,parse_ms,prepare_ms,"
               << "geometry_upload_ms,texture_upload_ms,blas_ms,tlas_ms,"
               << "gaussian_projected,gaussian_duplicates,gaussian_padded_duplicates,gaussian_tiles,gaussian_avg_tiles_touched,gaussian_rebuilds,"
               << "gaussian_preprocess_ms,gaussian_scan_ms,gaussian_duplicate_ms,gaussian_sort_ms,gaussian_range_ms,"
               << "gaussian_raster_ms,gaussian_total_build_ms\n";
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
           << CsvEscape(SceneKindLabel(scene.GetSceneKind())) << ','
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
           << _renderer.GetOfficialGaussianTotalBuildMs() << '\n';

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
                ImGui::MenuItem("FPS Limit", nullptr, &settings.enableFpsLimit);
                int fpsLimit = static_cast<int>(settings.fpsLimit);
                if (ImGui::SliderInt("FPS Limit Value", &fpsLimit, 15, 360)) {
                    settings.fpsLimit = static_cast<uint32_t>(fpsLimit);
                }
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

        if (ImGui::BeginMenu("Debug")) {
            ImGui::MenuItem("Frame / Engine Overview", nullptr, &_showFrameOverview);
            ImGui::MenuItem("Render Graph", nullptr, &_showRenderGraphPanel);
            ImGui::MenuItem("GPU Profiler", nullptr, &_showGpuProfilerPanel);
            ImGui::MenuItem("Debug Visualization", nullptr, &_showDebugVisualizationPanel);
            ImGui::MenuItem("Scene Inspector", nullptr, &_showSceneInspectorPanel);
            ImGui::MenuItem("Resource Inspector", nullptr, &_showResourceInspectorPanel);
            ImGui::MenuItem("Log Console", nullptr, &_showLogConsolePanel);
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
                const std::filesystem::path path = MakeTimestampedCapturePath("frame", ".ppm");
                log_startup_event(_renderer.RequestScreenshot(path) ? "Frame capture queued: " + path.string() : "Frame capture failed");
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
                ImGui::EndTable();
            }

            ImGui::BeginDisabled();
            ImGui::Checkbox("VSync", &_vsyncUiPlaceholder);
            ImGui::EndDisabled();
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
                const std::filesystem::path path = MakeTimestampedCapturePath("frame", ".ppm");
                log_startup_event(_renderer.RequestScreenshot(path) ? "Frame capture queued: " + path.string() : "Frame capture failed");
            }
            ImGui::SameLine();
            if (ImGui::Button("Screenshot")) {
                const std::filesystem::path path = MakeTimestampedCapturePath("screenshot", ".ppm");
                log_startup_event(_renderer.RequestScreenshot(path) ? "Screenshot queued: " + path.string() : "Screenshot failed");
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
            const char* displayModes[] = { "Hybrid: Raster + Gaussian + Path", "Rasterizer", "Gaussian Splatting", "Path Tracing" };
            int displayMode = static_cast<int>(settings.displayMode);
            if (ImGui::Combo("Render Mode", &displayMode, displayModes, IM_ARRAYSIZE(displayModes))) {
                settings.displayMode = static_cast<vesta::render::RendererDisplayMode>(displayMode);
                _renderer.ResetAccumulation();
            }

            const std::vector<vesta::render::RenderPassDebugInfo> passInfo = _renderer.GetRenderPassDebugInfo();
            if (ImGui::BeginTable("PassRegistry", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Pass");
                ImGui::TableSetupColumn("Order", ImGuiTableColumnFlags_WidthFixed, 52.0f);
                ImGui::TableSetupColumn("Enabled", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                ImGui::TableSetupColumn("Id");
                ImGui::TableHeadersRow();
                for (const auto& pass : passInfo) {
                    bool enabled = pass.enabled;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(pass.name.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%u", pass.order);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::PushID(pass.id.c_str());
                    if (ImGui::Checkbox("##enabled", &enabled)) {
                        _renderer.SetPassEnabled(pass.id, enabled);
                        _renderer.ResetAccumulation();
                    }
                    ImGui::PopID();
                    ImGui::TableSetColumnIndex(3);
                    ImGui::TextUnformatted(pass.id.c_str());
                }
                ImGui::EndTable();
            }

            ImGui::SeparatorText("Current Frame Resources");
            for (const auto& timing : graphTimings) {
                if (ImGui::TreeNode(timing.name.c_str())) {
                    if (timing.gpuTimingValid) {
                        ImGui::Text("CPU %.3f ms, GPU %.3f ms, Barriers %u", timing.cpuMs, timing.gpuMs, timing.barrierCount);
                    } else {
                        ImGui::Text("CPU %.3f ms, GPU -, Barriers %u", timing.cpuMs, timing.barrierCount);
                    }
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
            ImGui::Text("CPU Frame %.3f ms", _renderer.GetFrameTimeMs());
            ImGui::Text("GPU Frame %.3f ms", gpuFrameMs);
            ImGui::Text("Draw / Dispatch %s", "tracked per pass via timing table");
            ImGui::Text("Triangles %zu", scene.GetTriangles().size());
            ImGui::Text("Visible Surfaces %u / %zu", _renderer.GetVisibleSurfaceCount(), scene.GetSurfaces().size());
            ImGui::Text("Gaussians %u visible/projection %u", scene.GetGaussianCount(), _renderer.GetOfficialGaussianProjectedCount());
            ImGui::Text("Splats rendered %u", _renderer.GetOfficialGaussianDuplicateCount());
            ImGui::Text("VRAM Dedicated %u MiB", device.GetDedicatedVideoMemoryMiB());
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
            if (frameHistoryCount > 0) {
                ImGui::PlotLines("CPU Frame History", frameHistory.data(), static_cast<int>(frameHistoryCount), 0, nullptr, 0.0f,
                    std::max(33.0f, frameStats.maxMs * 1.1f), ImVec2(0.0f, 72.0f));
            }
        }
        ImGui::End();
    }

    if (_showDebugVisualizationPanel) {
        ImGui::SetNextWindowPos(ImVec2(1124.0f, 238.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(360.0f, 300.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Debug Visualization", &_showDebugVisualizationPanel, ImGuiWindowFlags_NoSavedSettings)) {
            const char* commonViews[] = {
                "Final Color", "Albedo / Base Color", "Normal", "Roughness", "Metallic", "Emissive"
            };
            int commonView = static_cast<int>(settings.debugView);
            if (ImGui::Combo("Debug View", &commonView, commonViews, IM_ARRAYSIZE(commonViews))) {
                settings.debugView = static_cast<vesta::render::RendererDebugView>(commonView);
                _renderer.ResetAccumulation();
            }
            ImGui::TextDisabled("Raster GBuffer views are live when the raster pass is active.");

            const char* pathTraceDebugViews[] = { "Final", "Albedo", "Normal", "Depth", "Direct", "Indirect" };
            int pathTraceDebugView = static_cast<int>(settings.pathTraceDebugView);
            if (ImGui::Combo("Path Tracing AOV", &pathTraceDebugView, pathTraceDebugViews, IM_ARRAYSIZE(pathTraceDebugViews))) {
                settings.pathTraceDebugView = static_cast<vesta::render::PathTraceDebugView>(pathTraceDebugView);
            }
            const char* gaussianViews[] = {
                "Final Splat Image", "Splat Alpha", "Revealage", "Overdraw Heatmap", "Splat Depth", "Tile Occupancy"
            };
            int gaussianView = static_cast<int>(settings.gaussianDebugView);
            if (ImGui::Combo("Gaussian Debug View", &gaussianView, gaussianViews, IM_ARRAYSIZE(gaussianViews))) {
                settings.gaussianDebugView = static_cast<vesta::render::GaussianDebugView>(gaussianView);
                _renderer.ResetAccumulation();
            }
            ImGui::Checkbox("Wireframe", &_wireframeUiPlaceholder);
            ImGui::Checkbox("Overdraw Heatmap", &_overdrawUiPlaceholder);
        }
        ImGui::End();
    }

    ImGui::SetNextWindowPos(ImVec2(18.0f, 18.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::Text("Frame %.2f ms", frameMs);
        ImGui::Text("FPS %.1f", fps);
        ImGui::Text("%s", device.GetGpuName().c_str());
        ImGui::Text("VRAM %u MiB", device.GetDedicatedVideoMemoryMiB());
        ImGui::SeparatorText("Scene");
        ImGui::TextWrapped("%s", sceneLabel.c_str());
        ImGui::Text("Type %s", SceneKindLabel(scene.GetSceneKind()));
        ImGui::Text("Display %s", DisplayModeLabel(settings.displayMode));
        ImGui::Text("Load %s", SceneLoadStateLabel(sceneLoadStatus.state));
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
                ImGui::Text("Projected %u", _renderer.GetOfficialGaussianProjectedCount());
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
                ImGui::PlotLines(
                    "Frame Times", history.data(), sampleCount, 0, nullptr, 0.0f, std::max(33.0f, peakMs * 1.1f), ImVec2(0.0f, 72.0f));
            }
        }
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(458.0f, 18.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Render", nullptr, ImGuiWindowFlags_NoSavedSettings)) {
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

        const char* pathTraceDebugViews[] = { "Final", "Albedo", "Normal", "Depth", "Direct", "Indirect" };
        int pathTraceDebugView = static_cast<int>(settings.pathTraceDebugView);
        if (ImGui::Combo("PT Debug View", &pathTraceDebugView, pathTraceDebugViews, IM_ARRAYSIZE(pathTraceDebugViews))) {
            settings.pathTraceDebugView = static_cast<vesta::render::PathTraceDebugView>(pathTraceDebugView);
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
        }
        if (ImGui::Button("Reset PT Accumulation")) {
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
    }
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(18.0f, 340.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420.0f, 0.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_NoSavedSettings)) {
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
        ImGui::TextDisabled("Rotation order: Yaw Pitch Roll");
        ImGui::Text("Forward %.3f %.3f %.3f", camera.GetForward().x, camera.GetForward().y, camera.GetForward().z);
        ImGui::Text("Up %.3f %.3f %.3f", camera.GetUp().x, camera.GetUp().y, camera.GetUp().z);
        ImGui::SeparatorText("Controls");
        ImGui::Text("%s", camera.IsOrbitEnabled() ? "RMB + Mouse Orbit / Dolly Adjust" : "RMB + Mouse Look");
        ImGui::Text("%s", camera.IsOrbitEnabled() ? "Wheel Zoom" : "WASD / Q / E Move");
        ImGui::Text("LMB Pick/Drag Object");
        ImGui::Text("L Select Light, Esc Clear Selection");
        ImGui::Text("1 Raster, 2 Gaussian, 3 PT, 4 Composite");
        ImGui::Text("R/G/P toggles, F1 UI, F5 Reload, F12 Screenshot");
    }
    ImGui::End();

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
                        ImGui::Text("Bounds Center %.3f %.3f %.3f",
                            object.bounds.center.x,
                            object.bounds.center.y,
                            object.bounds.center.z);
                        ImGui::Text("Radius %.3f", object.bounds.radius);
                        ImGui::Text("Vertices %u  Triangles %u", object.vertexCount, object.triangleCount);
                    } else {
                        ImGui::TextUnformatted("No object selected");
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Light")) {
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
                    ImGui::Text("Environment HDRI %s", "not loaded");
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Materials")) {
                    if (ImGui::BeginTable("MaterialTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 42.0f);
                        ImGui::TableSetupColumn("Base Color", ImGuiTableColumnFlags_WidthFixed, 112.0f);
                        ImGui::TableSetupColumn("Metallic", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("Roughness", ImGuiTableColumnFlags_WidthFixed, 76.0f);
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
                            materialChanged |= ImGui::ColorEdit3("##emissive", &material.emissiveFactor.x, ImGuiColorEditFlags_NoInputs);
                            ImGui::TableSetColumnIndex(5);
                            materialChanged |= ImGui::SliderFloat("##normal", &material.materialParams.w, 0.0f, 2.0f, "%.2f");
                            ImGui::TableSetColumnIndex(6);
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
                _texturePreviewDescriptors.assign(scene.GetTextures().size(), VK_NULL_HANDLE);
                _texturePreviewSceneVersion = scene.GetContentVersion();
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
                        row("Acceleration Structures", MiB(accelerationBytes));
                        row("Total GPU Tracked", MiB(sceneBufferBytes + residentTextureBytes + accelerationBytes));
                        ImGui::EndTable();
                    }
                    ImGui::Text("Textures %u / %zu resident", scene.GetResidentTextureCount(), scene.GetTextures().size());
                    ImGui::Text("Dedicated VRAM %u MiB", device.GetDedicatedVideoMemoryMiB());
                    ImGui::Text("Upload Last %.2f MiB  Pending %.2f MiB",
                        MiB(static_cast<uint64_t>(device.GetUploadBatchStats().lastSubmittedBytes)),
                        MiB(static_cast<uint64_t>(device.GetUploadBatchStats().pendingBytes)));
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
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::TextUnformatted(texture.name.empty() ? "(texture)" : texture.name.c_str());
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
                            ImGui::TextUnformatted("1");
                            ImGui::TableSetColumnIndex(5);
                            ImGui::TextUnformatted("Sampled");
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
                        }
                        ImGui::EndTable();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Buffers")) {
                    if (ImGui::BeginTable("BufferTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 74.0f);
                        ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Logical", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Usage");
                        ImGui::TableSetupColumn("Bindless", ImGuiTableColumnFlags_WidthFixed, 68.0f);
                        ImGui::TableSetupColumn("Handle", ImGuiTableColumnFlags_WidthFixed, 58.0f);
                        ImGui::TableHeadersRow();
                        DrawBufferResourceRow("Vertex Buffer", device, scene.GetVertexBuffer(), vertexBytes);
                        DrawBufferResourceRow("Index Buffer", device, scene.GetIndexBuffer(), indexBytes);
                        DrawBufferResourceRow("Material Buffer", device, scene.GetMaterialBuffer(), materialBytes);
                        DrawBufferResourceRow("Triangle Buffer", device, scene.GetTriangleBuffer(), triangleBytes);
                        DrawBufferResourceRow("Emissive Triangle Buffer", device, scene.GetEmissiveTriangleBuffer(), emissiveBytes);
                        DrawBufferResourceRow("Gaussian Position/Covariance/SH", device, scene.GetGaussianBuffer(), gaussianBytes);
                        ImGui::EndTable();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Acceleration")) {
                    ImGui::Text("TLAS %s", scene.HasRayTracingScene() ? "Resident" : "Missing");
                    ImGui::Text("BLAS Build %.3f ms", scene.GetBottomLevelBuildMs());
                    ImGui::Text("TLAS Build %.3f ms", scene.GetTopLevelBuildMs());
                    ImGui::Text("Instances %zu", scene.GetObjects().size());
                    ImGui::Separator();
                    if (ImGui::BeginTable("AccelerationTable", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Name");
                        ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 74.0f);
                        ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_WidthFixed, 78.0f);
                        ImGui::TableSetupColumn("Logical", ImGuiTableColumnFlags_WidthFixed, 78.0f);
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
                    ImGui::Text("Position/Covariance/SH/Opacity buffer %s", scene.GetGaussianBuffer() ? "Resident" : "Missing");
                    ImGui::Text("Sort Keys %u duplicates", _renderer.GetOfficialGaussianDuplicateCount());
                    ImGui::Text("Tile/Bin Count %u", _renderer.GetOfficialGaussianTileCount());
                    ImGui::Text("Memory Source %.2f MiB",
                        static_cast<double>(scene.GetGaussians().size() * sizeof(vesta::scene::GaussianPrimitive)) / (1024.0 * 1024.0));
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
            int perfWarnings = 0;
            int validationWarnings = 0;
            int resourceWarnings = 0;
            int errors = 0;
            for (const std::string& line : _logConsoleLines) {
                perfWarnings += line.find("[PERF]") != std::string::npos ? 1 : 0;
                validationWarnings += line.find("[VALIDATION]") != std::string::npos ? 1 : 0;
                resourceWarnings += line.find("[RESOURCE]") != std::string::npos ? 1 : 0;
                errors += line.find("failed") != std::string::npos || line.find("[ERROR]") != std::string::npos ? 1 : 0;
            }
            ImGui::Text("Perf %d  Validation %d  Resource %d  Errors %d",
                perfWarnings,
                validationWarnings,
                resourceWarnings,
                errors);
            ImGui::Separator();
            if (ImGui::Button("Clear")) {
                _logConsoleLines.clear();
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
            ImGui::BeginChild("LogScroll", ImVec2(0.0f, 0.0f), true);
            for (const std::string& line : _logConsoleLines) {
                if (line.find("[PERF]") != std::string::npos) {
                    ImGui::TextColored(ImVec4(1.0f, 0.78f, 0.28f, 1.0f), "%s", line.c_str());
                } else if (line.find("[VALIDATION]") != std::string::npos || line.find("[RESOURCE]") != std::string::npos) {
                    ImGui::TextColored(ImVec4(0.45f, 0.74f, 1.0f, 1.0f), "%s", line.c_str());
                } else if (line.find("failed") != std::string::npos || line.find("[ERROR]") != std::string::npos) {
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
        L"Supported Scenes (*.glb;*.gltf;*.fbx;*.ply)\0*.glb;*.gltf;*.fbx;*.ply\0glTF Scenes (*.glb;*.gltf)\0*.glb;*.gltf\0FBX Meshes (*.fbx)\0*.fbx\0Gaussian PLY (*.ply)\0*.ply\0All Files (*.*)\0*.*\0";
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
    _renderer.ResetAccumulation();

    const bool started = UseAsyncSceneLoading(_renderer.GetSettings())
        ? _renderer.LoadSceneAsync(normalizedPath)
        : _renderer.LoadScene(normalizedPath);
    if (started) {
        remember_recent_scene(normalizedPath);
    }
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
