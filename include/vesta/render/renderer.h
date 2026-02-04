#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <vesta/core/job_system.h>
#include <vesta/render/path_trace_backend.h>
#include <vesta/render/graph/render_graph.h>
#include <vesta/render/rhi/render_device.h>
#include <vesta/scene/camera.h>
#include <vesta/scene/scene.h>

struct SDL_Window;
union SDL_Event;

namespace vesta::render {
// High-level view modes exposed to the app and debug UI.
enum class RendererDisplayMode : uint32_t {
    Composite = 0,
    DeferredLighting = 1,
    Gaussian = 2,
    PathTrace = 3,
};

enum class RendererPreset : uint32_t {
    Recommended = 0,
    Performance = 1,
    Balanced = 2,
    Quality = 3,
};

enum class SceneLoadState : uint32_t {
    Idle = 0,
    Parsing = 1,
    Uploading = 2,
    Ready = 3,
    Failed = 4,
};

enum class SceneUploadMode : uint32_t {
    Synchronous = 0,
    AsyncParseSyncUpload = 1,
};

struct SceneLoadStatus {
    SceneLoadState state{ SceneLoadState::Idle };
    std::filesystem::path path;
    std::string message;
    float parseMs{ 0.0f };
    float uploadMs{ 0.0f };
    float blasMs{ 0.0f };
    float tlasMs{ 0.0f };
};

struct SceneUploadOptions {
    bool useDeviceLocalSceneBuffers{ true };
    bool buildRayTracingStructuresOnLoad{ true };
};

// RendererSettings collects the knobs that can safely change at runtime.
// When one of these changes in a way that affects history, accumulation resets.
struct RendererSettings {
    RendererDisplayMode displayMode{ RendererDisplayMode::Composite };
    bool enableGaussian{ true };
    bool enablePathTracing{ true };
    bool optimizeInactivePasses{ true };
    bool preferAsyncSceneLoading{ true };
    bool useDeviceLocalSceneBuffers{ true };
    bool buildRayTracingStructuresOnLoad{ true };
    bool deferOldSceneDestruction{ true };
    bool autoFocusSceneOnLoad{ true };
    bool frameTimingCapture{ false };
    bool benchmarkOverlay{ false };
    bool enableFrustumCulling{ true };
    SceneUploadMode sceneUploadMode{ SceneUploadMode::AsyncParseSyncUpload };
    float gaussianPointSize{ 8.0f };
    float gaussianOpacity{ 0.35f };
    float gaussianMix{ 0.28f };
    float pathTraceResolutionScale{ 0.5f };
    PathTraceBackend pathTraceBackend{ PathTraceBackend::Auto };
};

// Each overlapping frame owns its own command buffer and sync objects so the CPU
// can prepare the next frame while the GPU is still finishing the previous one.
struct RendererFrameContext {
    VkCommandPool commandPool{ VK_NULL_HANDLE };
    VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
    VkSemaphore acquireSemaphore{ VK_NULL_HANDLE };
    VkFence renderFence{ VK_NULL_HANDLE };
    std::vector<ImageHandle> acquiredTransientImages;
    std::vector<BufferHandle> transientBuffers;
};

// These are the logical edges between passes in the frame graph.
struct RendererGraphResources {
    GraphTextureHandle swapchainTarget{};
    GraphTextureHandle gbufferAlbedo{};
    GraphTextureHandle gbufferNormal{};
    GraphTextureHandle sceneDepth{};
    GraphTextureHandle deferredLighting{};
    GraphTextureHandle pathTraceOutput{};
    GraphTextureHandle gaussianOutput{};
};

using RenderPassConfigureFn = std::function<void(IRenderPass&, const RendererGraphResources&)>;
using OverlayDrawFn = std::function<void(VkCommandBuffer)>;
using OverlaySwapchainCallback = std::function<void(uint32_t)>;

struct RenderPassRegistrationDesc {
    std::string id;
    std::unique_ptr<IRenderPass> pass;
    RenderPassConfigureFn configure;
    uint32_t order{ 0 };
    bool enabled{ true };
};

struct TransientImageKey {
    VkExtent3D extent{ 1, 1, 1 };
    VkFormat format{ VK_FORMAT_UNDEFINED };
    VkImageUsageFlags usage{ 0 };
    VkImageAspectFlags aspectFlags{ 0 };
    VkImageLayout initialLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
    VmaMemoryUsage memoryUsage{ VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    uint32_t mipLevels{ 1 };
    uint32_t arrayLayers{ 1 };

    [[nodiscard]] bool operator==(const TransientImageKey& other) const
    {
        return extent.width == other.extent.width && extent.height == other.extent.height && extent.depth == other.extent.depth
            && format == other.format && usage == other.usage && aspectFlags == other.aspectFlags
            && initialLayout == other.initialLayout && memoryUsage == other.memoryUsage && mipLevels == other.mipLevels
            && arrayLayers == other.arrayLayers;
    }
};

struct TransientImagePoolEntry {
    ImageHandle handle{};
    TransientImageKey key{};
    bool inUse{ false };
};

class TransientImagePool {
public:
    // Reuses images with identical descriptions to avoid creating temporary
    // Vulkan images every frame.
    [[nodiscard]] ImageHandle Acquire(RenderDevice& device, const ImageDesc& desc);
    void Release(ImageHandle handle);
    void Purge(RenderDevice& device);

private:
    [[nodiscard]] static TransientImageKey MakeKey(const ImageDesc& desc);

    std::vector<TransientImagePoolEntry> _entries;
};

class Renderer {
public:
    static constexpr uint32_t kFrameOverlap = 2;

    // Renderer drives the per-frame flow:
    // input -> camera update -> graph build -> pass execution -> presentation.
    bool Initialize(SDL_Window* window, VkExtent2D initialExtent, bool enableValidation);
    void Shutdown();
    void HandleEvent(const SDL_Event& event);
    void Update(float deltaSeconds);
    void RenderFrame();

    [[nodiscard]] const vesta::scene::Scene& GetScene() const { return _scene; }
    [[nodiscard]] const Camera& GetCamera() const { return _camera; }
    [[nodiscard]] Camera& GetCamera() { return _camera; }
    [[nodiscard]] const RendererSettings& GetSettings() const { return _settings; }
    [[nodiscard]] RendererSettings& GetSettings() { return _settings; }
    [[nodiscard]] const SceneLoadStatus& GetSceneLoadStatus() const { return _sceneLoadStatus; }
    [[nodiscard]] uint32_t GetPathTraceFrameIndex() const { return _pathTraceFrameIndex; }
    [[nodiscard]] uint32_t GetFrameSlot() const { return static_cast<uint32_t>(_frameNumber % kFrameOverlap); }
    [[nodiscard]] float GetFrameTimeMs() const { return _frameTimeMs; }
    [[nodiscard]] float GetSmoothedFrameTimeMs() const { return _smoothedFrameTimeMs; }
    [[nodiscard]] const std::array<float, 240>& GetFrameTimeHistoryMs() const { return _frameTimeHistoryMs; }
    [[nodiscard]] size_t GetFrameTimeHistoryCount() const { return _frameTimeHistoryCount; }
    [[nodiscard]] uint32_t GetVisibleSurfaceCount() const { return static_cast<uint32_t>(_visibleSurfaceIndices.size()); }
    [[nodiscard]] const std::vector<uint32_t>& GetVisibleSurfaceIndices() const { return _visibleSurfaceIndices; }
    [[nodiscard]] bool HasValidVisibilitySet() const { return _visibleSceneToken != nullptr && _visibleSceneToken == _scene.GetPreparedScene(); }
    [[nodiscard]] uint32_t GetWorkerThreadCount() const { return _jobs.GetWorkerCount(); }
    [[nodiscard]] size_t GetPendingJobCount() const { return _jobs.GetPendingJobCount(); }
    [[nodiscard]] RenderDevice& GetRenderDevice() { return _device; }
    [[nodiscard]] const RenderDevice& GetRenderDevice() const { return _device; }
    [[nodiscard]] PathTraceBackend GetActivePathTraceBackend() const;
    [[nodiscard]] RendererPreset GetRecommendedPreset() const;
    [[nodiscard]] bool IsSceneLoadInProgress() const { return _sceneLoadInProgress; }
    [[nodiscard]] const std::filesystem::path& GetPendingScenePath() const { return _sceneLoadStatus.path; }
    [[nodiscard]] const std::string& GetSceneLoadStatusMessage() const { return _sceneLoadStatus.message; }

    void ResetAccumulation() { _pathTraceFrameIndex = 0; }
    void ApplyPreset(RendererPreset preset);
    bool LoadScene(const std::filesystem::path& path);
    bool LoadSceneAsync(const std::filesystem::path& path);
    bool ReloadSceneAsync();
    void SetOverlayCallbacks(OverlayDrawFn drawFn, OverlaySwapchainCallback swapchainCallback = {});
    void ClearOverlayCallbacks();

    bool RegisterPass(RenderPassRegistrationDesc desc);
    bool UnregisterPass(std::string_view id);
    bool SetPassEnabled(std::string_view id, bool enabled);
    bool SetPassOrder(std::string_view id, uint32_t order);
    [[nodiscard]] IRenderPass* FindPass(std::string_view id);
    [[nodiscard]] const IRenderPass* FindPass(std::string_view id) const;

    template <typename TPass>
    [[nodiscard]] TPass* FindPass(std::string_view id)
    {
        return dynamic_cast<TPass*>(FindPass(id));
    }

    template <typename TPass>
    [[nodiscard]] const TPass* FindPass(std::string_view id) const
    {
        return dynamic_cast<const TPass*>(FindPass(id));
    }

private:
    struct RegisteredPassEntry {
        std::string id;
        std::unique_ptr<IRenderPass> pass;
        RenderPassConfigureFn configure;
        uint32_t order{ 0 };
        bool enabled{ true };
    };

    struct AsyncSceneLoadResult {
        std::filesystem::path path;
        vesta::scene::Scene scene;
        std::string errorMessage;
        float parseMs{ 0.0f };
        bool success{ false };
    };

    struct RetiredSceneEntry {
        vesta::scene::Scene scene;
        uint64_t safeFrameNumber{ 0 };
    };

    struct VisibilityCullResult {
        std::shared_ptr<const vesta::scene::PreparedScene> scene;
        std::vector<uint32_t> visibleSurfaceIndices;
    };

    void InitializeCommands();
    void InitializeSyncStructures();
    void InitializeDefaultPasses();
    void DestroyFrameResources();
    void ReleaseTransientResources(RendererFrameContext& frameContext);
    void RecordOverlay(VkCommandBuffer commandBuffer, uint32_t swapchainImageIndex);
    void RecreateSwapchain();
    void ClearPassRegistry();
    void RebuildPassExecutionPlan();
    void PumpSceneLoadRequests();
    void PumpVisibilityResults();
    void DispatchVisibilityCullIfNeeded();
    void ReleaseRetiredScenes();
    [[nodiscard]] SceneUploadOptions GetSceneUploadOptions() const;
    bool LoadSceneResolved(const std::filesystem::path& resolvedPath);
    void ApplyLoadedScene(vesta::scene::Scene&& scene);
    [[nodiscard]] RendererFrameContext& GetCurrentFrame();
    [[nodiscard]] RenderGraph BuildFrameGraph(uint32_t swapchainImageIndex);
    [[nodiscard]] RegisteredPassEntry* FindPassEntry(std::string_view id);
    [[nodiscard]] const RegisteredPassEntry* FindPassEntry(std::string_view id) const;

    RenderDevice _device;
    vesta::core::JobSystem _jobs;
    std::array<RendererFrameContext, kFrameOverlap> _frames{};
    std::vector<VkSemaphore> _swapchainImageRenderSemaphores;
    uint64_t _frameNumber{ 0 };
    std::vector<RegisteredPassEntry> _passRegistry;
    std::vector<RegisteredPassEntry*> _passExecutionPlan;
    bool _passExecutionPlanDirty{ true };
    TransientImagePool _transientImagePool;
    SDL_Window* _window{ nullptr };
    vesta::scene::Scene _scene;
    Camera _camera;
    RendererSettings _settings;
    uint32_t _pathTraceFrameIndex{ 0 };
    float _frameTimeMs{ 0.0f };
    float _smoothedFrameTimeMs{ 0.0f };
    std::array<float, 240> _frameTimeHistoryMs{};
    size_t _frameTimeHistoryHead{ 0 };
    size_t _frameTimeHistoryCount{ 0 };
    std::future<AsyncSceneLoadResult> _sceneLoadFuture;
    std::future<VisibilityCullResult> _visibilityFuture;
    SceneLoadStatus _sceneLoadStatus;
    std::deque<RetiredSceneEntry> _retiredScenes;
    std::vector<uint32_t> _visibleSurfaceIndices;
    std::shared_ptr<const vesta::scene::PreparedScene> _visibleSceneToken;
    bool _sceneLoadInProgress{ false };
    bool _visibilityCullInProgress{ false };
    bool _visibilityDirty{ true };
    OverlayDrawFn _overlayDrawFn;
    OverlaySwapchainCallback _overlaySwapchainCallback;

    void LoadDefaultScene();
};
} // namespace vesta::render
