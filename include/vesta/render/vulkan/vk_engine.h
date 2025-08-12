// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vesta/render/renderer.h>
#include <vesta/render/vulkan/vk_types.h>

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
    VkDescriptorPool _imguiDescriptorPool{ VK_NULL_HANDLE };

    static VestaEngine& Get();

    // Owns the application loop around the renderer. The renderer itself stays
    // focused on Vulkan work; SDL event handling and ImGui live here.
    void init();
    void cleanup();
    void draw(float deltaSeconds);
    void run();

private:
    vesta::render::Renderer _renderer;

    void init_renderer();
    void init_imgui();
    void shutdown_imgui();
    void begin_imgui_frame(float deltaSeconds);
    void build_debug_ui();
    [[nodiscard]] bool should_forward_event_to_renderer(const union SDL_Event& event) const;
};
