//> includes
#include <vesta/render/vulkan/vk_engine.h>

#include <cassert>

#include <SDL.h>

#include <fmt/format.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>

VestaEngine* loadedEngine = nullptr;

VestaEngine& VestaEngine::Get() { return *loadedEngine; }

namespace {
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
}

void VestaEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

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

    init_renderer();
    init_imgui();

    // everything went fine
    _isInitialized = true;
}

void VestaEngine::init_renderer()
{
    _renderer.Initialize(_window, _windowExtent, bUseValidationLayers);
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
    _renderer.Update(deltaSeconds);
    begin_imgui_frame(deltaSeconds);
    _renderer.RenderFrame();
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
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
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

            if (should_forward_event_to_renderer(e)) {
                _renderer.HandleEvent(e);
            }
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            previousTick = std::chrono::steady_clock::now();
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        const float deltaSeconds = std::chrono::duration<float>(now - previousTick).count();
        previousTick = now;

        draw(deltaSeconds);
    }
}

void VestaEngine::init_imgui()
{
    auto& device = _renderer.GetRenderDevice();

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
    build_debug_ui();
    ImGui::Render();
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

    const char* displayModes[] = { "Composite", "Deferred", "Gaussian", "Path Trace" };
    int displayMode = static_cast<int>(settings.displayMode);
    if (ImGui::Combo("Display", &displayMode, displayModes, IM_ARRAYSIZE(displayModes))) {
        settings.displayMode = static_cast<vesta::render::RendererDisplayMode>(displayMode);
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

    ImGui::SeparatorText("Scene");
    ImGui::Text("%s", scene.GetSourcePath().empty() ? "No scene" : scene.GetSourcePath().filename().string().c_str());
    ImGui::Text("Vertices %zu", scene.GetVertices().size());
    ImGui::Text("Triangles %zu", scene.GetTriangles().size());
    ImGui::Text("Surfaces %zu", scene.GetSurfaces().size());
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

    ImGui::SeparatorText("Controls");
    ImGui::Text("RMB + Mouse Look");
    ImGui::Text("WASD / Q / E Move");
    ImGui::Text("1-4 Modes, G/P Toggles, F1 UI");
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
