#include <vesta/render/renderer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <SDL.h>
#include <SDL_vulkan.h>
#include <fmt/format.h>
#include <glm/glm.hpp>

#include <vesta/core/debug.h>
#include <vesta/render/passes/composite_pass.h>
#include <vesta/render/passes/deferred_lighting_pass.h>
#include <vesta/render/passes/gaussian_splat_pass.h>
#include <vesta/render/passes/official_gaussian_raster_pass.h>
#include <vesta/render/passes/geometry_raster_pass.h>
#include <vesta/render/passes/path_tracer_pass.h>
#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>

namespace vesta::render {
namespace {
// Presets are derived from the active GPU because the heaviest pass in this
// sample is path tracing, and its reasonable resolution scale changes a lot
// between low-end, non-RT, and modern RT-capable cards.
std::string ToUpper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

bool IsRtx5060Ti(const RenderDevice& device)
{
    return ToUpper(device.GetGpuName()).find("RTX 5060 TI") != std::string::npos;
}

uint32_t GaussianInteractivePreviewFrameBudget(const vesta::scene::Scene& scene)
{
    if (!scene.HasTrainedGaussians()) {
        return 0u;
    }
    const uint32_t gaussianCount = scene.GetGaussianCount();
    if (gaussianCount >= 4'000'000u) {
        return 360u;
    }
    if (gaussianCount >= 1'000'000u) {
        return 240u;
    }
    return 120u;
}

bool NeedsGeometryPass(const RendererSettings& settings)
{
    if (!settings.enableRaster) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode != RendererDisplayMode::PathTrace;
}

bool NeedsDeferredPass(const RendererSettings& settings)
{
    if (!settings.enableRaster) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode == RendererDisplayMode::Composite || settings.displayMode == RendererDisplayMode::DeferredLighting;
}

bool NeedsGaussianPass(const RendererSettings& settings)
{
    if (!settings.enableGaussian) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode == RendererDisplayMode::Composite || settings.displayMode == RendererDisplayMode::Gaussian;
}

bool NeedsPathTracePass(const RendererSettings& settings)
{
    if (!settings.enablePathTracing) {
        return false;
    }
    if (!settings.optimizeInactivePasses) {
        return true;
    }

    return settings.displayMode == RendererDisplayMode::Composite || settings.displayMode == RendererDisplayMode::PathTrace;
}

bool UsesStreamingUpload(const RendererSettings& settings)
{
    return settings.sceneUploadMode == SceneUploadMode::Streaming && settings.useDeviceLocalSceneBuffers;
}

void ValidateSceneLoadTransition(const SceneLoadStatus& status, SceneLoadState nextState, std::string_view context)
{
    VESTA_ASSERT_STATE(IsValidSceneLoadTransition(status.state, nextState),
        fmt::format("Invalid scene load transition {} -> {} in {} for '{}'",
            static_cast<uint32_t>(status.state),
            static_cast<uint32_t>(nextState),
            context,
            status.path.string()));
}

void ApplySceneLoadState(SceneLoadStatus& status, SceneLoadState nextState, std::string message, std::string_view context)
{
    ValidateSceneLoadTransition(status, nextState, context);
    status.state = nextState;
    status.message = std::move(message);
}

float ClampPathTraceScale(float scale)
{
    return std::clamp(scale, 0.25f, 1.0f);
}

VkExtent3D ScaleExtent(VkExtent3D extent, float scale)
{
    const float clampedScale = ClampPathTraceScale(scale);
    extent.width = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(extent.width) * clampedScale)));
    extent.height = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<float>(extent.height) * clampedScale)));
    extent.depth = 1;
    return extent;
}

RendererPreset ChooseRecommendedPreset(const RenderDevice& device)
{
    const uint32_t dedicatedMemoryMiB = device.GetDedicatedVideoMemoryMiB();

    if (!device.IsRayTracingSupported()) {
        return dedicatedMemoryMiB >= 12u * 1024u ? RendererPreset::Balanced : RendererPreset::Performance;
    }

    if (IsRtx5060Ti(device)) {
        return dedicatedMemoryMiB >= 12u * 1024u ? RendererPreset::Quality : RendererPreset::Balanced;
    }

    if (dedicatedMemoryMiB >= 14u * 1024u) {
        return RendererPreset::Quality;
    }
    if (dedicatedMemoryMiB >= 8u * 1024u) {
        return RendererPreset::Balanced;
    }
    return RendererPreset::Performance;
}

void ApplyPresetSettings(RendererSettings& settings, const RenderDevice& device, RendererPreset preset)
{
    settings.displayMode = RendererDisplayMode::DeferredLighting;
    settings.enableRaster = true;
    settings.enableGaussian = true;
    settings.enablePathTracing = true;
    settings.gaussianOpacity = 1.0f;
    settings.gaussianShDegree = 0u;

    const bool hardwareRtPreferred = device.IsRayTracingSupported();
    settings.pathTraceBackend = hardwareRtPreferred ? PathTraceBackend::Auto : PathTraceBackend::Compute;

    switch (preset) {
    case RendererPreset::Performance:
        settings.gaussianMix = 0.18f;
        settings.pathTraceResolutionScale = hardwareRtPreferred ? 0.50f : 0.33f;
        break;
    case RendererPreset::Balanced:
        settings.gaussianMix = 0.24f;
        settings.pathTraceResolutionScale = hardwareRtPreferred ? 0.67f : 0.50f;
        break;
    case RendererPreset::Quality:
        settings.gaussianMix = 0.28f;
        settings.pathTraceResolutionScale = hardwareRtPreferred ? 1.0f : 0.67f;
        break;
    case RendererPreset::Recommended:
    default:
        ApplyPresetSettings(settings, device, ChooseRecommendedPreset(device));
        return;
    }

    settings.pathTraceResolutionScale = ClampPathTraceScale(settings.pathTraceResolutionScale);
}

std::array<glm::vec4, 6> ExtractFrustumPlanes(const glm::mat4& viewProjection)
{
    const glm::mat4 matrix = glm::transpose(viewProjection);
    std::array<glm::vec4, 6> planes{
        matrix[3] + matrix[0],
        matrix[3] - matrix[0],
        matrix[3] + matrix[1],
        matrix[3] - matrix[1],
        matrix[3] + matrix[2],
        matrix[3] - matrix[2],
    };

    for (glm::vec4& plane : planes) {
        const float length = glm::length(glm::vec3(plane));
        if (length > 0.0f) {
            plane /= length;
        }
    }
    return planes;
}

bool IsSurfaceVisible(const vesta::scene::SceneSurfaceBounds& bounds, const std::array<glm::vec4, 6>& planes)
{
    for (const glm::vec4& plane : planes) {
        const float distance = glm::dot(glm::vec3(plane), bounds.center) + plane.w;
        if (distance < -bounds.radius) {
            return false;
        }
    }
    return true;
}

bool IsSurfaceWithinDistance(const vesta::scene::SceneSurfaceBounds& bounds,
    const glm::vec3& cameraPosition,
    float sceneRadius,
    float distanceCullScale)
{
    const float distance = glm::distance(cameraPosition, bounds.center);
    const float allowedDistance = std::max(bounds.radius * 12.0f, sceneRadius * distanceCullScale);
    return distance <= allowedDistance;
}

float DefaultOrbitDistance(float currentDistance, float targetRadius)
{
    const float minimumDistance = std::max(targetRadius * 2.5f, 0.75f);
    if (currentDistance > 0.0f) {
        return std::max(currentDistance, minimumDistance);
    }
    return minimumDistance;
}

void ConfigureGeometryRasterPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& rasterPass = static_cast<GeometryRasterPass&>(pass);
    rasterPass.SetTargets(resources.gbufferAlbedo, resources.gbufferNormal, resources.gbufferMaterial, resources.sceneDepth);
    rasterPass.SetScene(&renderer.GetScene());
    rasterPass.SetCamera(&renderer.GetCamera());
    const bool useVisibilitySet =
        (renderer.GetSettings().enableFrustumCulling || renderer.GetSettings().enableDistanceCulling)
        && renderer.HasValidVisibilitySet();
    rasterPass.SetVisibleSurfaceIndices(useVisibilitySet ? &renderer.GetVisibleSurfaceIndices() : nullptr);
    rasterPass.SetUseIndirectDraw(renderer.GetSettings().useIndirectDraw);
}

void ConfigureDeferredLightingPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& lightingPass = static_cast<DeferredLightingPass&>(pass);
    lightingPass.SetInputs(resources.gbufferAlbedo, resources.gbufferNormal, resources.gbufferMaterial, resources.sceneDepth);
    lightingPass.SetCamera(&renderer.GetCamera());
    lightingPass.SetLight(renderer.GetSettings().lightDirectionAndIntensity);
    lightingPass.SetOutput(resources.deferredLighting);
}

void ConfigureGaussianPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& gaussianPass = static_cast<GaussianSplatPass&>(pass);
    gaussianPass.SetDepthInput(resources.sceneDepth);
    gaussianPass.SetOutputs(resources.gaussianAccum, resources.gaussianReveal);
    gaussianPass.SetScene(&renderer.GetScene());
    gaussianPass.SetCamera(&renderer.GetCamera());
    const uint32_t effectiveShDegree =
        std::min(renderer.GetSettings().gaussianShDegree, renderer.GetScene().GetGaussianShDegree());
    gaussianPass.SetParams(renderer.GetSettings().gaussianOpacity,
        renderer.GetSettings().enableGaussian,
        effectiveShDegree,
        renderer.GetSettings().gaussianViewDependentColor,
        renderer.GetSettings().gaussianAntialiasing,
        renderer.GetSettings().gaussianFastCulling);
}

void ConfigureOfficialGaussianPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& gaussianPass = static_cast<OfficialGaussianRasterPass&>(pass);
    const bool useDepthInput = renderer.GetSettings().enableRaster && renderer.GetScene().GetSceneKind() == vesta::scene::SceneKind::Mesh;
    gaussianPass.SetDepthInput(useDepthInput ? resources.sceneDepth : GraphTextureHandle{});
    gaussianPass.SetOutputs(resources.gaussianAccum, resources.gaussianReveal);
    gaussianPass.SetScene(&renderer.GetScene());
    gaussianPass.SetCamera(&renderer.GetCamera());
    gaussianPass.SetJobSystem(&renderer.GetJobSystem());
    gaussianPass.SetFrameSlot(renderer.GetFrameSlot());
    const uint32_t effectiveShDegree =
        std::min(renderer.GetSettings().gaussianShDegree, renderer.GetScene().GetGaussianShDegree());
    gaussianPass.SetParams(renderer.GetSettings().gaussianOpacity,
        effectiveShDegree,
        renderer.GetSettings().gaussianViewDependentColor,
        renderer.GetSettings().gaussianAntialiasing,
        renderer.GetSettings().gaussianFastCulling);
}

void ConfigurePathTracerPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& pathTracerPass = static_cast<PathTracerPass&>(pass);
    pathTracerPass.SetOutput(resources.pathTraceOutput);
    pathTracerPass.SetScene(&renderer.GetScene());
    pathTracerPass.SetCamera(&renderer.GetCamera());
    pathTracerPass.SetFrameIndex(renderer.GetPathTraceFrameIndex());
    pathTracerPass.SetFrameSlot(renderer.GetFrameSlot());
    pathTracerPass.SetEnabled(renderer.GetSettings().enablePathTracing);
    pathTracerPass.SetBackendPreference(renderer.GetSettings().pathTraceBackend);
    pathTracerPass.SetLight(renderer.GetSettings().lightDirectionAndIntensity);
}

void ConfigureCompositePass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& compositePass = static_cast<CompositePass&>(pass);
    compositePass.SetInputs(resources.deferredLighting, resources.pathTraceOutput, resources.gaussianAccum, resources.gaussianReveal);
    compositePass.SetOutput(resources.swapchainTarget);
    compositePass.SetMode(static_cast<uint32_t>(renderer.GetSettings().displayMode), renderer.GetSettings().gaussianMix);
}
} // namespace

TransientImageKey TransientImagePool::MakeKey(const ImageDesc& desc)
{
    return TransientImageKey{
        .extent = desc.extent,
        .format = desc.format,
        .usage = desc.usage,
        .aspectFlags = desc.aspectFlags,
        .initialLayout = desc.initialLayout,
        .memoryUsage = desc.memoryUsage,
        .mipLevels = desc.mipLevels,
        .arrayLayers = desc.arrayLayers,
    };
}

ImageHandle TransientImagePool::Acquire(RenderDevice& device, const ImageDesc& desc)
{
    const TransientImageKey key = MakeKey(desc);

    for (TransientImagePoolEntry& entry : _entries) {
        if (!entry.inUse && entry.key == key) {
            entry.inUse = true;
            return entry.handle;
        }
    }

    ImageHandle handle = device.CreateImage(desc);
    _entries.push_back(TransientImagePoolEntry{
        .handle = handle,
        .key = key,
        .inUse = true,
    });
    return handle;
}

void TransientImagePool::Release(ImageHandle handle)
{
    for (TransientImagePoolEntry& entry : _entries) {
        if (entry.handle == handle) {
            entry.inUse = false;
            return;
        }
    }
}

void TransientImagePool::Purge(RenderDevice& device)
{
    for (const TransientImagePoolEntry& entry : _entries) {
        if (entry.handle) {
            device.DestroyImage(entry.handle);
        }
    }
    _entries.clear();
}

bool Renderer::Initialize(SDL_Window* window, VkExtent2D initialExtent, bool enableValidation)
{
    _window = window;

    // RenderDevice owns Vulkan lifetime, while Renderer owns frame-level policy
    // such as presets, passes, transient resources, and camera/scene state.
    RenderDeviceDesc deviceDesc;
    deviceDesc.swapchainExtent = initialExtent;
    deviceDesc.enableValidation = enableValidation;
    _device.Initialize(window, deviceDesc);
    _jobs.Initialize();
    ApplyPreset(RendererPreset::Recommended);

    _camera.SetViewport(initialExtent.width, initialExtent.height);
    InitializeCommands();
    InitializeSyncStructures();
    InitializeDefaultPasses();
    return true;
}

void Renderer::Shutdown()
{
    if (_sceneLoadFuture.valid()) {
        _sceneLoadFuture.wait();
        _sceneLoadFuture = {};
    }
    if (_visibilityFuture.valid()) {
        _visibilityFuture.wait();
        _visibilityFuture = {};
    }
    _sceneLoadInProgress = false;
    _visibilityCullInProgress = false;
    _sceneLoadStatus = {};

    _device.WaitIdle();
    ClearPassRegistry();
    DestroyFrameResources();
    _transientImagePool.Purge(_device);
    _pendingSceneUpload.scene.DestroyGpu(_device);
    _pendingSceneUpload = {};
    _scene.DestroyGpu(_device);
    for (RetiredSceneEntry& retiredScene : _retiredScenes) {
        retiredScene.scene.DestroyGpu(_device);
    }
    _retiredScenes.clear();
    _jobs.Shutdown();
    _device.Shutdown();
    _window = nullptr;
}

void Renderer::HandleEvent(const SDL_Event& event)
{
    _camera.HandleEvent(event);

    if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        _camera.SetViewport(static_cast<uint32_t>(event.window.data1), static_cast<uint32_t>(event.window.data2));
        _pathTraceFrameIndex = 0;
        return;
    }

    if (event.type != SDL_KEYDOWN || event.key.repeat != 0) {
        if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
            if (_selection.kind != SelectionKind::DirectionalLight) {
                _selection = PickSelection(glm::vec2(static_cast<float>(event.button.x), static_cast<float>(event.button.y)));
            }
            _selectionDragging = _selection.kind != SelectionKind::None;
            _selectionEditedSinceDragStart = false;
            _lastDragMousePosition = glm::vec2(static_cast<float>(event.button.x), static_cast<float>(event.button.y));

            if (_selection.kind == SelectionKind::Object && _selection.objectIndex < _scene.GetObjects().size()) {
                const auto& object = _scene.GetObjects()[_selection.objectIndex];
                _dragPlaneOrigin = object.bounds.center;
                _dragPlaneNormal = _camera.GetForward();
                _dragGrabOffset = object.bounds.center - _dragPlaneOrigin;
            } else if (_selection.kind == SelectionKind::DirectionalLight) {
                _dragPlaneOrigin = _scene.GetBounds().center;
                _dragPlaneNormal = _camera.GetForward();
                _dragGrabOffset = glm::vec3(0.0f);
            }
            return;
        }

        if (event.type == SDL_MOUSEMOTION && _selectionDragging) {
            UpdateSceneEditDrag(glm::vec2(static_cast<float>(event.motion.x), static_cast<float>(event.motion.y)));
            return;
        }

        if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
            EndSceneEditDrag();
            return;
        }
        return;
    }

    switch (event.key.keysym.sym) {
    case SDLK_1:
        _settings.displayMode = RendererDisplayMode::DeferredLighting;
        break;
    case SDLK_2:
        _settings.displayMode = RendererDisplayMode::Gaussian;
        break;
    case SDLK_3:
        _settings.displayMode = RendererDisplayMode::PathTrace;
        break;
    case SDLK_4:
        _settings.displayMode = RendererDisplayMode::Composite;
        break;
    case SDLK_g:
        _settings.enableGaussian = !_settings.enableGaussian;
        break;
    case SDLK_p:
        _settings.enablePathTracing = !_settings.enablePathTracing;
        break;
    case SDLK_r:
        _settings.enableRaster = !_settings.enableRaster;
        break;
    case SDLK_ESCAPE:
        ClearSelection();
        break;
    case SDLK_l:
        SelectDirectionalLight();
        break;
    default:
        break;
    }

    ResetAccumulation();
}

void Renderer::Update(float deltaSeconds)
{
    _sceneLoadStatus.pendingUploadBytes = static_cast<uint64_t>(_device.GetUploadBatchStats().pendingBytes);
    _sceneLoadStatus.pendingUploadCopies = _device.GetUploadBatchStats().pendingCopies;
    PumpSceneLoadRequests();
    PumpPendingSceneUpload();
    PumpVisibilityResults();

    _frameTimeMs = deltaSeconds * 1000.0f;
    _smoothedFrameTimeMs = _smoothedFrameTimeMs <= 0.0f ? _frameTimeMs : (_smoothedFrameTimeMs * 0.9f + _frameTimeMs * 0.1f);
    if (_settings.frameTimingCapture || _settings.benchmarkOverlay) {
        _frameTimeHistoryMs[_frameTimeHistoryHead] = _frameTimeMs;
        _frameTimeHistoryHead = (_frameTimeHistoryHead + 1) % _frameTimeHistoryMs.size();
        _frameTimeHistoryCount = std::min(_frameTimeHistoryCount + 1, _frameTimeHistoryMs.size());
    } else {
        _frameTimeHistoryHead = 0;
        _frameTimeHistoryCount = 0;
        _frameTimeHistoryMs.fill(0.0f);
    }

    if (!_camera.IsOrbitEnabled()) {
        _trackSelectedObjectOrbit = false;
    }

    if (_trackSelectedObjectOrbit) {
        const auto& objects = _scene.GetObjects();
        if (_selection.kind == SelectionKind::Object && _selection.objectIndex < objects.size()) {
            _camera.SetOrbitTarget(objects[_selection.objectIndex].bounds.center);
        } else {
            _trackSelectedObjectOrbit = false;
        }
    }

    _camera.Update(deltaSeconds);
    // Progressive path tracing only makes sense while the viewpoint is stable.
    // As soon as the camera moves, old samples become history from the wrong camera.
    if (_camera.ConsumeMoved()) {
        _pathTraceFrameIndex = 0;
        _visibilityDirty = true;
        if (_scene.HasTrainedGaussians()) {
            _gaussianInteractivePreviewFramesRemaining = GaussianInteractivePreviewFrameBudget(_scene);
        }
        if (!_scene.HasTrainedGaussians() && _scene.SupportsRealtimeGaussianSorting()) {
            _scene.ResortGaussians(_device, _camera);
        }
    } else {
        ++_pathTraceFrameIndex;
        if (_gaussianInteractivePreviewFramesRemaining > 0) {
            --_gaussianInteractivePreviewFramesRemaining;
        }
    }

    DispatchVisibilityCullIfNeeded();
}

void Renderer::RenderFrame()
{
    RendererFrameContext& currentFrame = GetCurrentFrame();

    // Each overlapping frame owns its own fence. Waiting here guarantees the GPU
    // has finished using the command buffer and transient resources we are about to recycle.
    VK_CHECK(vkWaitForFences(_device.GetDevice(), 1, &currentFrame.renderFence, VK_TRUE, std::numeric_limits<uint64_t>::max()));
    ReleaseRetiredScenes();
    ReleaseTransientResources(currentFrame);

    uint32_t swapchainImageIndex = 0;
    VkResult acquireResult = vkAcquireNextImageKHR(_device.GetDevice(),
        _device.GetSwapchain(),
        std::numeric_limits<uint64_t>::max(),
        currentFrame.acquireSemaphore,
        VK_NULL_HANDLE,
        &swapchainImageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(acquireResult);
    }

    VkSemaphore renderSemaphore = _swapchainImageRenderSemaphores.at(swapchainImageIndex);

    VK_CHECK(vkResetFences(_device.GetDevice(), 1, &currentFrame.renderFence));
    VK_CHECK(vkResetCommandBuffer(currentFrame.commandBuffer, 0));

    VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(currentFrame.commandBuffer, &beginInfo));

    // Build the logical frame graph first, then execute it. This keeps pass code
    // focused on "what it needs" instead of hand-written global barriers.
    RenderGraph graph = BuildFrameGraph(swapchainImageIndex);
    RenderGraphExecutionContext executionContext{
        .device = _device,
        .frameContext = currentFrame,
        .transientImagePool = _transientImagePool,
        .commandBuffer = currentFrame.commandBuffer,
    };
    graph.Execute(executionContext);
    RecordOverlay(currentFrame.commandBuffer, swapchainImageIndex);

    VK_CHECK(vkEndCommandBuffer(currentFrame.commandBuffer));

    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(currentFrame.commandBuffer);
    VkSemaphoreSubmitInfo waitInfo =
        vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, currentFrame.acquireSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, renderSemaphore);
    VkSubmitInfo2 submitInfo = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    VK_CHECK(vkQueueSubmit2(_device.GetGraphicsQueue(), 1, &submitInfo, currentFrame.renderFence));

    VkSwapchainKHR swapchain = _device.GetSwapchain();
    VkPresentInfoKHR presentInfo = vkinit::present_info();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderSemaphore;
    presentInfo.pImageIndices = &swapchainImageIndex;

    VkResult presentResult = vkQueuePresentKHR(_device.GetPresentQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        RecreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        VK_CHECK(presentResult);
    }

    ++_frameNumber;
}

void Renderer::SetOverlayCallbacks(OverlayDrawFn drawFn, OverlaySwapchainCallback swapchainCallback)
{
    _overlayDrawFn = std::move(drawFn);
    _overlaySwapchainCallback = std::move(swapchainCallback);
}

void Renderer::ClearOverlayCallbacks()
{
    _overlayDrawFn = {};
    _overlaySwapchainCallback = {};
}

PathTraceBackend Renderer::GetActivePathTraceBackend() const
{
    const auto* pathTracerPass = FindPass<PathTracerPass>("path-tracer");
    return pathTracerPass != nullptr ? pathTracerPass->GetActiveBackend() : PathTraceBackend::Compute;
}

RendererPreset Renderer::GetRecommendedPreset() const
{
    return ChooseRecommendedPreset(_device);
}

uint32_t Renderer::GetOfficialGaussianProjectedCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().projectedCount;
    }
    return 0;
}

uint32_t Renderer::GetOfficialGaussianDuplicateCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().duplicateCount;
    }
    return 0;
}

uint32_t Renderer::GetOfficialGaussianPaddedDuplicateCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().paddedDuplicateCount;
    }
    return 0;
}

uint32_t Renderer::GetOfficialGaussianTileCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().tileCount;
    }
    return 0;
}

float Renderer::GetOfficialGaussianAverageTilesTouched() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().averageTilesTouched;
    }
    return 0.0f;
}

uint64_t Renderer::GetOfficialGaussianRebuildCount() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().rebuildCount;
    }
    return 0;
}

float Renderer::GetOfficialGaussianPreprocessMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().preprocessMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianScanMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().scanMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianDuplicateMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().duplicateMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianSortMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().sortMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianRangeMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().rangeMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianRasterMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().rasterMs;
    }
    return 0.0f;
}

float Renderer::GetOfficialGaussianTotalBuildMs() const
{
    if (const auto* pass = FindPass<OfficialGaussianRasterPass>("official-gaussian-raster")) {
        return pass->GetStatistics().totalBuildMs;
    }
    return 0.0f;
}

vesta::scene::SceneKind Renderer::GetRecommendedSceneKind() const
{
    return _scene.GetSceneKind();
}

RendererDisplayMode Renderer::GetRecommendedDisplayModeForScene() const
{
    switch (_scene.GetSceneKind()) {
    case vesta::scene::SceneKind::Gaussian:
    case vesta::scene::SceneKind::PointCloud:
        return RendererDisplayMode::Gaussian;
    case vesta::scene::SceneKind::Mesh:
    case vesta::scene::SceneKind::Empty:
    default:
        return RendererDisplayMode::DeferredLighting;
    }
}

std::string Renderer::GetSelectionLabel() const
{
    switch (_selection.kind) {
    case SelectionKind::Object: {
        const auto& objects = _scene.GetObjects();
        if (_selection.objectIndex < objects.size()) {
            return objects[_selection.objectIndex].name;
        }
        return "Object";
    }
    case SelectionKind::DirectionalLight:
        return "Directional Light";
    case SelectionKind::None:
    default:
        return "None";
    }
}

void Renderer::ApplyPreset(RendererPreset preset)
{
    ApplyPresetSettings(_settings, _device, preset);
    ResetAccumulation();
}

void Renderer::SelectDirectionalLight()
{
    _selection = EditorSelection{
        .kind = SelectionKind::DirectionalLight,
        .objectIndex = 0,
    };
    _trackSelectedObjectOrbit = false;
}

void Renderer::ClearSelection()
{
    _selection = {};
    _selectionDragging = false;
    _selectionEditedSinceDragStart = false;
    _trackSelectedObjectOrbit = false;
}

bool Renderer::OrbitCameraAroundSelection()
{
    const auto& objects = _scene.GetObjects();
    if (_selection.kind != SelectionKind::Object || _selection.objectIndex >= objects.size()) {
        return false;
    }

    const auto& object = objects[_selection.objectIndex];
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), object.bounds.center), object.bounds.radius);
    _camera.EnableOrbit(object.bounds.center, distance);
    _trackSelectedObjectOrbit = true;
    ResetAccumulation();
    _visibilityDirty = true;
    return true;
}

void Renderer::OrbitCameraAroundScene()
{
    const auto& bounds = _scene.GetBounds();
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), bounds.center), bounds.radius);
    _camera.EnableOrbit(bounds.center, distance);
    _trackSelectedObjectOrbit = false;
    ResetAccumulation();
    _visibilityDirty = true;
}

bool Renderer::DollyCameraAroundSelection()
{
    const auto& objects = _scene.GetObjects();
    if (_selection.kind != SelectionKind::Object || _selection.objectIndex >= objects.size()) {
        return false;
    }

    const auto& object = objects[_selection.objectIndex];
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), object.bounds.center), object.bounds.radius);
    _camera.EnableDollyOrbit(object.bounds.center, distance, _camera.GetDollySpeedDegrees());
    _trackSelectedObjectOrbit = true;
    ResetAccumulation();
    _visibilityDirty = true;
    return true;
}

void Renderer::DollyCameraAroundScene()
{
    const auto& bounds = _scene.GetBounds();
    const float distance = DefaultOrbitDistance(glm::distance(_camera.GetPosition(), bounds.center), bounds.radius);
    _camera.EnableDollyOrbit(bounds.center, distance, _camera.GetDollySpeedDegrees());
    _trackSelectedObjectOrbit = false;
    ResetAccumulation();
    _visibilityDirty = true;
}

void Renderer::DisableCameraOrbit()
{
    _camera.DisableOrbit();
    _trackSelectedObjectOrbit = false;
    ResetAccumulation();
}

std::pair<glm::vec3, glm::vec3> Renderer::ComputeMouseRay(glm::vec2 mousePosition) const
{
    const VkExtent2D extent = _device.GetSwapchainExtent();
    const glm::vec2 viewportSize(
        std::max(1.0f, static_cast<float>(extent.width)), std::max(1.0f, static_cast<float>(extent.height)));
    const glm::vec2 ndc(
        (mousePosition.x / viewportSize.x) * 2.0f - 1.0f,
        (mousePosition.y / viewportSize.y) * 2.0f - 1.0f);

    const glm::vec4 nearPoint = _camera.GetInverseViewProjection() * glm::vec4(ndc.x, ndc.y, 0.0f, 1.0f);
    const glm::vec4 farPoint = _camera.GetInverseViewProjection() * glm::vec4(ndc.x, ndc.y, 1.0f, 1.0f);
    const glm::vec3 worldNear = glm::vec3(nearPoint) / std::max(nearPoint.w, 1.0e-4f);
    const glm::vec3 worldFar = glm::vec3(farPoint) / std::max(farPoint.w, 1.0e-4f);
    return { _camera.GetPosition(), glm::normalize(worldFar - worldNear) };
}

EditorSelection Renderer::PickSelection(glm::vec2 mousePosition) const
{
    const auto [rayOrigin, rayDirection] = ComputeMouseRay(mousePosition);
    if (const std::optional<uint32_t> objectIndex = _scene.PickObject(rayOrigin, rayDirection); objectIndex.has_value()) {
        return EditorSelection{
            .kind = SelectionKind::Object,
            .objectIndex = *objectIndex,
        };
    }
    return {};
}

void Renderer::OnSceneEdited(bool rebuildRayTracing)
{
    ResetAccumulation();
    _visibilityDirty = true;
    _visibleSurfaceIndices.clear();
    _visibleSceneToken.reset();
    _frameSnapshot = {};
    if (_scene.HasTrainedGaussians()) {
        _gaussianInteractivePreviewFramesRemaining = 8;
    }
    if (!_scene.HasTrainedGaussians() && _scene.SupportsRealtimeGaussianSorting()) {
        _scene.ResortGaussians(_device, _camera);
    }

    if (rebuildRayTracing && _scene.HasRayTracingScene()) {
        _scene.RebuildRayTracing(_device);
    }
}

void Renderer::UpdateSceneEditDrag(const glm::vec2& mousePosition)
{
    if (!_selectionDragging || _selection.kind == SelectionKind::None) {
        return;
    }

    const auto [rayOrigin, rayDirection] = ComputeMouseRay(mousePosition);
    if (_selection.kind == SelectionKind::Object) {
        const auto& objects = _scene.GetObjects();
        if (_selection.objectIndex >= objects.size()) {
            return;
        }

        const vesta::scene::SceneObject& object = objects[_selection.objectIndex];
        const float denominator = glm::dot(rayDirection, _dragPlaneNormal);
        if (std::abs(denominator) < 1.0e-4f) {
            return;
        }

        const float t = glm::dot(object.bounds.center - rayOrigin, _dragPlaneNormal) / denominator;
        if (t <= 0.0f) {
            return;
        }

        const glm::vec3 hitPoint = rayOrigin + rayDirection * t;
        const glm::vec3 delta = hitPoint - object.bounds.center;
        if (_scene.TranslateObject(_device, _selection.objectIndex, delta)) {
            _selectionEditedSinceDragStart = true;
            OnSceneEdited(false);
        }
        return;
    }

    if (_selection.kind == SelectionKind::DirectionalLight) {
        const glm::vec2 delta = mousePosition - _lastDragMousePosition;
        glm::vec3 cameraRight = glm::cross(_camera.GetForward(), glm::vec3(0.0f, 1.0f, 0.0f));
        if (glm::length(cameraRight) < 1.0e-4f) {
            cameraRight = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            cameraRight = glm::normalize(cameraRight);
        }
        glm::vec3 direction = glm::normalize(-glm::vec3(_settings.lightDirectionAndIntensity));
        direction = glm::normalize(
            direction + _camera.GetForward() * (-delta.y * 0.01f) + cameraRight * (-delta.x * 0.01f));
        _settings.lightDirectionAndIntensity = glm::vec4(-direction, _settings.lightDirectionAndIntensity.w);
        _selectionEditedSinceDragStart = true;
        OnSceneEdited(false);
    }

    _lastDragMousePosition = mousePosition;
}

void Renderer::EndSceneEditDrag()
{
    if (_selectionDragging && _selectionEditedSinceDragStart) {
        const bool rebuildRayTracing = _selection.kind == SelectionKind::Object && _scene.HasRayTracingScene()
            && _settings.enablePathTracing && GetActivePathTraceBackend() == PathTraceBackend::HardwareRT;
        OnSceneEdited(rebuildRayTracing);
    }
    _selectionDragging = false;
    _selectionEditedSinceDragStart = false;
}

SceneUploadOptions Renderer::GetSceneUploadOptions() const
{
    return SceneUploadOptions{
        .useDeviceLocalSceneBuffers = _settings.useDeviceLocalSceneBuffers,
        .buildRayTracingStructuresOnLoad = _settings.buildRayTracingStructuresOnLoad,
        .textureStreamingEnabled = _settings.textureStreamingEnabled,
        .useDeviceLocalTextures = _settings.useDeviceLocalTextures,
    };
}

bool Renderer::LoadScene(const std::filesystem::path& path)
{
    if (path.empty()) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene path is empty.";
        return false;
    }

    if (_sceneLoadInProgress) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene load already in progress.";
        return false;
    }

    const std::filesystem::path resolvedPath = vkutil::resolve_runtime_path(path);
    return LoadSceneResolved(resolvedPath);
}

bool Renderer::LoadSceneAsync(const std::filesystem::path& path)
{
    if (path.empty()) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene path is empty.";
        return false;
    }

    PumpSceneLoadRequests();
    if (_sceneLoadInProgress) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Scene load already in progress.";
        return false;
    }

    const std::filesystem::path resolvedPath = vkutil::resolve_runtime_path(path);
    _sceneLoadStatus = SceneLoadStatus{
        .state = SceneLoadState::Parsing,
        .path = resolvedPath,
        .message = "Parsing and preparing " + resolvedPath.filename().string() + "...",
    };
    _sceneLoadInProgress = true;
    _sceneLoadFuture = _jobs.Submit(vesta::core::JobPriority::Background, [resolvedPath]() {
        AsyncSceneLoadResult result;
        result.path = resolvedPath;
        const auto parseStart = std::chrono::steady_clock::now();

        try {
            vesta::scene::Scene loadedScene;
            result.success = loadedScene.ParseFromFile(resolvedPath);
            result.parseMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();
            if (result.success) {
                const auto prepareStart = std::chrono::steady_clock::now();
                result.success = loadedScene.PrepareParsedScene();
                result.prepareMs =
                    std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prepareStart).count();
            }
            if (result.success) {
                result.scene = std::move(loadedScene);
            } else {
                result.errorMessage = result.prepareMs > 0.0f ? "Failed to prepare scene file." : "Failed to parse scene file.";
            }
        } catch (const std::exception& exception) {
            result.errorMessage = exception.what();
        } catch (...) {
            result.errorMessage = "Unknown scene loading error.";
        }

        if (result.parseMs <= 0.0f) {
            result.parseMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();
        }
        return result;
    });
    return true;
}

bool Renderer::ReloadSceneAsync()
{
    const std::filesystem::path currentPath = _scene.GetSourcePath();
    if (currentPath.empty()) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "No scene to reload.";
        return false;
    }

    return LoadSceneAsync(currentPath);
}

bool Renderer::RegisterPass(RenderPassRegistrationDesc desc)
{
    if (desc.id.empty() || !desc.pass || FindPassEntry(desc.id) != nullptr) {
        return false;
    }

    _passRegistry.push_back(RegisteredPassEntry{
        .id = std::move(desc.id),
        .pass = std::move(desc.pass),
        .configure = std::move(desc.configure),
        .order = desc.order,
        .enabled = desc.enabled,
    });

    RegisteredPassEntry& entry = _passRegistry.back();
    if (_device.GetDevice() != VK_NULL_HANDLE) {
        entry.pass->Initialize(_device);
    }

    _passExecutionPlanDirty = true;
    return true;
}

bool Renderer::UnregisterPass(std::string_view id)
{
    const auto it = std::find_if(_passRegistry.begin(), _passRegistry.end(), [id](const RegisteredPassEntry& entry) {
        return entry.id == id;
    });
    if (it == _passRegistry.end()) {
        return false;
    }

    if (_device.GetDevice() != VK_NULL_HANDLE) {
        it->pass->Shutdown(_device);
    }

    _passRegistry.erase(it);
    _passExecutionPlan.clear();
    _passExecutionPlanDirty = true;
    return true;
}

bool Renderer::SetPassEnabled(std::string_view id, bool enabled)
{
    RegisteredPassEntry* entry = FindPassEntry(id);
    if (entry == nullptr) {
        return false;
    }

    entry->enabled = enabled;
    _passExecutionPlanDirty = true;
    return true;
}

bool Renderer::SetPassOrder(std::string_view id, uint32_t order)
{
    RegisteredPassEntry* entry = FindPassEntry(id);
    if (entry == nullptr) {
        return false;
    }

    entry->order = order;
    _passExecutionPlanDirty = true;
    return true;
}

IRenderPass* Renderer::FindPass(std::string_view id)
{
    RegisteredPassEntry* entry = FindPassEntry(id);
    return entry != nullptr ? entry->pass.get() : nullptr;
}

const IRenderPass* Renderer::FindPass(std::string_view id) const
{
    const RegisteredPassEntry* entry = FindPassEntry(id);
    return entry != nullptr ? entry->pass.get() : nullptr;
}

void Renderer::InitializeCommands()
{
    VkCommandPoolCreateInfo poolInfo =
        vkinit::command_pool_create_info(_device.GetGraphicsQueueFamily(), VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    for (RendererFrameContext& frame : _frames) {
        VK_CHECK(vkCreateCommandPool(_device.GetDevice(), &poolInfo, nullptr, &frame.commandPool));

        VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(frame.commandPool);
        VK_CHECK(vkAllocateCommandBuffers(_device.GetDevice(), &allocInfo, &frame.commandBuffer));
    }
}

void Renderer::InitializeSyncStructures()
{
    VkFenceCreateInfo fenceInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreInfo = vkinit::semaphore_create_info();

    for (RendererFrameContext& frame : _frames) {
        VK_CHECK(vkCreateFence(_device.GetDevice(), &fenceInfo, nullptr, &frame.renderFence));
        VK_CHECK(vkCreateSemaphore(_device.GetDevice(), &semaphoreInfo, nullptr, &frame.acquireSemaphore));
    }

    _swapchainImageRenderSemaphores.resize(_device.GetSwapchainImageHandles().size(), VK_NULL_HANDLE);
    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        VK_CHECK(vkCreateSemaphore(_device.GetDevice(), &semaphoreInfo, nullptr, &semaphore));
    }
}

void Renderer::InitializeDefaultPasses()
{
    ClearPassRegistry();

    // Pass order is explicit so the frame graph wiring stays easy to read.
    RegisterPass(RenderPassRegistrationDesc{
        .id = "geometry-raster",
        .pass = std::make_unique<GeometryRasterPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureGeometryRasterPass(*this, pass, resources);
        },
        .order = 10,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "deferred-lighting",
        .pass = std::make_unique<DeferredLightingPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureDeferredLightingPass(*this, pass, resources);
        },
        .order = 20,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "gaussian-splat",
        .pass = std::make_unique<GaussianSplatPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureGaussianPass(*this, pass, resources);
        },
        .order = 30,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "official-gaussian-raster",
        .pass = std::make_unique<OfficialGaussianRasterPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureOfficialGaussianPass(*this, pass, resources);
        },
        .order = 31,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "path-tracer",
        .pass = std::make_unique<PathTracerPass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigurePathTracerPass(*this, pass, resources);
        },
        .order = 40,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "composite",
        .pass = std::make_unique<CompositePass>(),
        .configure = [this](IRenderPass& pass, const RendererGraphResources& resources) {
            ConfigureCompositePass(*this, pass, resources);
        },
        .order = 50,
        .enabled = true,
    });
}

void Renderer::DestroyFrameResources()
{
    for (RendererFrameContext& frame : _frames) {
        ReleaseTransientResources(frame);

        if (frame.renderFence != VK_NULL_HANDLE) {
            vkDestroyFence(_device.GetDevice(), frame.renderFence, nullptr);
            frame.renderFence = VK_NULL_HANDLE;
        }
        if (frame.acquireSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(_device.GetDevice(), frame.acquireSemaphore, nullptr);
            frame.acquireSemaphore = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(_device.GetDevice(), frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
            frame.commandBuffer = VK_NULL_HANDLE;
        }
    }

    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(_device.GetDevice(), semaphore, nullptr);
            semaphore = VK_NULL_HANDLE;
        }
    }
    _swapchainImageRenderSemaphores.clear();
}

void Renderer::ReleaseTransientResources(RendererFrameContext& frameContext)
{
    for (ImageHandle handle : frameContext.acquiredTransientImages) {
        _transientImagePool.Release(handle);
    }
    frameContext.acquiredTransientImages.clear();

    for (BufferHandle handle : frameContext.transientBuffers) {
        _device.DestroyBuffer(handle);
    }
    frameContext.transientBuffers.clear();
}

void Renderer::RecordOverlay(VkCommandBuffer commandBuffer, uint32_t swapchainImageIndex)
{
    if (!_overlayDrawFn) {
        return;
    }

    VkImage swapchainImage = _device.GetImage(_device.GetSwapchainImageHandle(swapchainImageIndex));
    VkImageView swapchainView = _device.GetImageView(_device.GetSwapchainImageHandle(swapchainImageIndex));
    const VkExtent2D extent = _device.GetSwapchainExtent();
    const VkImageSubresourceRange colorRange = vkutil::make_image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);

    // The graph leaves the swapchain image ready for presentation. ImGui needs it
    // back in a color-attachment layout for one extra overlay draw, then we
    // transition it back to PRESENT before queue submission finishes.
    vkutil::transition_image(commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        colorRange);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchainView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = VkRect2D{ VkOffset2D{ 0, 0 }, extent };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    _overlayDrawFn(commandBuffer);
    vkCmdEndRendering(commandBuffer);

    vkutil::transition_image(commandBuffer,
        swapchainImage,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        colorRange);
}

void Renderer::RecreateSwapchain()
{
    int width = 0;
    int height = 0;
    SDL_Vulkan_GetDrawableSize(_window, &width, &height);
    if (width == 0 || height == 0) {
        return;
    }

    _device.WaitIdle();

    // Swapchain recreation invalidates images tied to the old extent. Purging the
    // transient pool here avoids reusing resources whose sizes no longer match.
    for (RendererFrameContext& frame : _frames) {
        ReleaseTransientResources(frame);
    }
    _transientImagePool.Purge(_device);

    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(_device.GetDevice(), semaphore, nullptr);
        }
    }
    _swapchainImageRenderSemaphores.clear();

    _device.RecreateSwapchain(VkExtent2D{ static_cast<uint32_t>(width), static_cast<uint32_t>(height) });
    _camera.SetViewport(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    _pathTraceFrameIndex = 0;
    if (_overlaySwapchainCallback) {
        _overlaySwapchainCallback(static_cast<uint32_t>(_device.GetSwapchainImageHandles().size()));
    }

    VkSemaphoreCreateInfo semaphoreInfo = vkinit::semaphore_create_info();
    _swapchainImageRenderSemaphores.resize(_device.GetSwapchainImageHandles().size(), VK_NULL_HANDLE);
    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        VK_CHECK(vkCreateSemaphore(_device.GetDevice(), &semaphoreInfo, nullptr, &semaphore));
    }
}

void Renderer::ClearPassRegistry()
{
    if (_device.GetDevice() != VK_NULL_HANDLE) {
        for (RegisteredPassEntry& entry : _passRegistry) {
            entry.pass->Shutdown(_device);
        }
    }

    _passExecutionPlan.clear();
    _passRegistry.clear();
    _passExecutionPlanDirty = true;
}

void Renderer::RebuildPassExecutionPlan()
{
    if (!_passExecutionPlanDirty) {
        return;
    }

    _passExecutionPlan.clear();
    _passExecutionPlan.reserve(_passRegistry.size());
    for (RegisteredPassEntry& entry : _passRegistry) {
        if (entry.enabled) {
            _passExecutionPlan.push_back(&entry);
        }
    }

    std::stable_sort(_passExecutionPlan.begin(), _passExecutionPlan.end(), [](const RegisteredPassEntry* lhs, const RegisteredPassEntry* rhs) {
        if (lhs->order != rhs->order) {
            return lhs->order < rhs->order;
        }
        return lhs->id < rhs->id;
    });

    _passExecutionPlanDirty = false;
}

void Renderer::PumpSceneLoadRequests()
{
    if (!_sceneLoadFuture.valid()) {
        return;
    }

    using namespace std::chrono_literals;
    if (_sceneLoadFuture.wait_for(0ms) != std::future_status::ready) {
        return;
    }

    AsyncSceneLoadResult result = _sceneLoadFuture.get();

    if (!result.success) {
        _sceneLoadInProgress = false;
        const std::string sceneName = result.path.empty() ? std::string("scene") : result.path.filename().string();
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.path = result.path;
        _sceneLoadStatus.parseMs = result.parseMs;
        _sceneLoadStatus.prepareMs = result.prepareMs;
        _sceneLoadStatus.geometryUploadMs = 0.0f;
        _sceneLoadStatus.textureUploadMs = 0.0f;
        _sceneLoadStatus.blasMs = 0.0f;
        _sceneLoadStatus.tlasMs = 0.0f;
        _sceneLoadStatus.message = "Failed to load " + sceneName;
        if (!result.errorMessage.empty()) {
            _sceneLoadStatus.message += ": " + result.errorMessage;
        }
        return;
    }

    ValidateSceneLoadTransition(_sceneLoadStatus, SceneLoadState::UploadingGeometry, "PumpSceneLoadRequests");
    _sceneLoadStatus.state = SceneLoadState::UploadingGeometry;
    _sceneLoadStatus.path = result.path;
    _sceneLoadStatus.parseMs = result.parseMs;
    _sceneLoadStatus.prepareMs = result.prepareMs;
    _sceneLoadStatus.message = "Uploading " + result.path.filename().string() + "...";
    if (UsesStreamingUpload(_settings)) {
        StartPendingSceneUpload(std::move(result.scene), result.parseMs, result.prepareMs);
    } else {
        VESTA_ASSERT(!_startupSafeModeActive,
            "Startup safe mode must not apply loaded scenes synchronously. Force streaming upload instead.");
        ApplySceneLoadState(_sceneLoadStatus,
            SceneLoadState::ReadyToSwap,
            "Finalizing " + result.path.filename().string() + "...",
            "PumpSceneLoadRequests");
        _sceneLoadInProgress = false;
        ApplyLoadedScene(std::move(result.scene));
    }
}

void Renderer::PumpPendingSceneUpload()
{
    if (!_pendingSceneUpload.active) {
        return;
    }

    using Stage = PendingSceneUploadStage;
    const auto stageLabel = [](Stage stage) {
        switch (stage) {
        case Stage::AllocateBuffers:
            return "AllocateBuffers";
        case Stage::UploadVertices:
            return "UploadVertices";
        case Stage::UploadGaussians:
            return "UploadGaussians";
        case Stage::UploadMaterials:
            return "UploadMaterials";
        case Stage::UploadIndices:
            return "UploadIndices";
        case Stage::UploadTriangles:
            return "UploadTriangles";
        case Stage::UploadTextures:
            return "UploadTextures";
        case Stage::BuildBLAS:
            return "BuildBLAS";
        case Stage::BuildTLAS:
            return "BuildTLAS";
        case Stage::SwapScene:
            return "SwapScene";
        case Stage::Idle:
        default:
            return "Idle";
        }
    };
    const auto uploadStart = std::chrono::steady_clock::now();
    const auto uploadOptions = GetSceneUploadOptions();
    const uint32_t uploadChunkBytes = std::max(64u * 1024u, _settings.maxUploadBytesPerFrame);
    const auto prepared = _pendingSceneUpload.scene.GetPreparedScene();
    if (!prepared) {
        _pendingSceneUpload = {};
        _sceneLoadInProgress = false;
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Pending scene upload lost its prepared scene.";
        return;
    }

    size_t remainingUploadBudget = uploadChunkBytes;
    while (_pendingSceneUpload.active) {
        _sceneLoadStatus.uploadStage = stageLabel(_pendingSceneUpload.stage);
        switch (_pendingSceneUpload.stage) {
        case Stage::AllocateBuffers:
            VESTA_ASSERT_STATE(prepared->IsLoaded(), "AllocateBuffers requires a prepared scene.");
            _pendingSceneUpload.scene.AllocateGpuResources(_device, uploadOptions);
            _pendingSceneUpload.stage = Stage::UploadVertices;
            _sceneLoadStatus.message = "Uploading vertices for " + _pendingSceneUpload.path.filename().string() + "...";
            continue;
        case Stage::UploadVertices: {
            const size_t totalBytes = sizeof(vesta::scene::SceneVertex) * prepared->vertices.size();
            VESTA_ASSERT_STATE(_pendingSceneUpload.vertexOffsetBytes <= totalBytes, "Vertex upload offset exceeded total vertex bytes.");
            if (_pendingSceneUpload.vertexOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadGaussians;
                _sceneLoadStatus.message = "Uploading gaussians for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes =
                std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.vertexOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Vertex, _pendingSceneUpload.vertexOffsetBytes, chunkBytes);
            _pendingSceneUpload.vertexOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadGaussians: {
            const size_t totalBytes = sizeof(vesta::scene::GaussianPrimitive) * prepared->gaussians.size();
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.gaussianOffsetBytes <= totalBytes, "Gaussian upload offset exceeded total gaussian bytes.");
            if (_pendingSceneUpload.gaussianOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadMaterials;
                _sceneLoadStatus.message = "Uploading materials for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes =
                std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.gaussianOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Gaussian, _pendingSceneUpload.gaussianOffsetBytes, chunkBytes);
            _pendingSceneUpload.gaussianOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadMaterials: {
            const size_t totalBytes = sizeof(vesta::scene::SceneMaterial) * prepared->materials.size();
            VESTA_ASSERT_STATE(_pendingSceneUpload.materialOffsetBytes <= totalBytes, "Material upload offset exceeded total material bytes.");
            if (_pendingSceneUpload.materialOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadIndices;
                _sceneLoadStatus.message = "Uploading indices for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes = std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.materialOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Material, _pendingSceneUpload.materialOffsetBytes, chunkBytes);
            _pendingSceneUpload.materialOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadIndices: {
            const size_t totalBytes = sizeof(uint32_t) * prepared->indices.size();
            VESTA_ASSERT_STATE(_pendingSceneUpload.indexOffsetBytes <= totalBytes, "Index upload offset exceeded total index bytes.");
            if (_pendingSceneUpload.indexOffsetBytes >= totalBytes) {
                _pendingSceneUpload.stage = Stage::UploadTriangles;
                _sceneLoadStatus.message = "Uploading triangles for " + _pendingSceneUpload.path.filename().string() + "...";
                continue;
            }

            const size_t chunkBytes = std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.indexOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Index, _pendingSceneUpload.indexOffsetBytes, chunkBytes);
            _pendingSceneUpload.indexOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadTriangles: {
            const size_t totalBytes = sizeof(vesta::scene::SceneTriangle) * prepared->triangles.size();
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.triangleOffsetBytes <= totalBytes, "Triangle upload offset exceeded total triangle bytes.");
            if (_pendingSceneUpload.triangleOffsetBytes >= totalBytes) {
                const SceneUploadContinuation continuation = DecideSceneUploadContinuation(
                    _settings.textureStreamingEnabled,
                    !prepared->textures.empty(),
                    _settings.buildRayTracingStructuresOnLoad,
                    _device.IsRayTracingSupported(),
                    !prepared->indices.empty());
                if (continuation == SceneUploadContinuation::UploadTextures) {
                    _pendingSceneUpload.stage = Stage::UploadTextures;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::UploadingTextures,
                        "Uploading textures for " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTriangles");
                } else if (continuation == SceneUploadContinuation::BuildBLAS) {
                    _pendingSceneUpload.stage = Stage::BuildBLAS;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::BuildingBLAS,
                        "Building BLAS for " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTriangles");
                } else {
                    _pendingSceneUpload.stage = Stage::SwapScene;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::ReadyToSwap,
                        "Finalizing " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTriangles");
                }
                continue;
            }

            const size_t chunkBytes =
                std::min<size_t>(remainingUploadBudget, totalBytes - _pendingSceneUpload.triangleOffsetBytes);
            _pendingSceneUpload.scene.UploadGpuResourceChunk(
                _device, vesta::scene::SceneUploadResource::Triangle, _pendingSceneUpload.triangleOffsetBytes, chunkBytes);
            _pendingSceneUpload.triangleOffsetBytes += chunkBytes;
            remainingUploadBudget -= chunkBytes;
            if (remainingUploadBudget == 0) {
                break;
            }
            continue;
        }
        case Stage::UploadTextures: {
            const uint32_t textureBudget = std::max(1u, _settings.maxTextureUploadBytesPerFrame);
            size_t uploadedBytes = 0;
            while (_pendingSceneUpload.textureIndex < prepared->textures.size()) {
                const auto& texture = prepared->textures[_pendingSceneUpload.textureIndex];
                if (texture.IsValid()) {
                    const size_t textureBytes = texture.rgba8Pixels.size();
                    if (uploadedBytes > 0 && uploadedBytes + textureBytes > textureBudget) {
                        break;
                    }
                    const auto textureUploadStart = std::chrono::steady_clock::now();
                    _pendingSceneUpload.scene.UploadGpuTexture(_device, _pendingSceneUpload.textureIndex);
                    _pendingSceneUpload.textureUploadMs +=
                        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - textureUploadStart).count();
                    uploadedBytes += textureBytes;
                }
                ++_pendingSceneUpload.textureIndex;
            }

            if (_pendingSceneUpload.textureIndex >= prepared->textures.size()) {
                const SceneUploadContinuation continuation = DecideSceneUploadContinuation(
                    false, false, _settings.buildRayTracingStructuresOnLoad, _device.IsRayTracingSupported(), !prepared->indices.empty());
                if (continuation == SceneUploadContinuation::BuildBLAS) {
                    _pendingSceneUpload.stage = Stage::BuildBLAS;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::BuildingBLAS,
                        "Building BLAS for " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTextures");
                } else {
                    _pendingSceneUpload.stage = Stage::SwapScene;
                    ApplySceneLoadState(_sceneLoadStatus,
                        SceneLoadState::ReadyToSwap,
                        "Finalizing " + _pendingSceneUpload.path.filename().string() + "...",
                        "PumpPendingSceneUpload::UploadTextures");
                }
            }
            break;
        }
        case Stage::BuildBLAS:
            _sceneLoadStatus.lastBlockingWait = "FlushUploadBatch before BLAS";
            _device.SetDebugWaitContext(
                "scene=" + _pendingSceneUpload.path.string() + " stage=BuildBLAS state=" + std::to_string(static_cast<uint32_t>(_sceneLoadStatus.state)));
            _device.FlushUploadBatch();
            _pendingSceneUpload.scene.BuildBottomLevelAccelerationStructure(_device);
            _pendingSceneUpload.stage = Stage::BuildTLAS;
            ApplySceneLoadState(_sceneLoadStatus,
                SceneLoadState::BuildingTLAS,
                "Building TLAS for " + _pendingSceneUpload.path.filename().string() + "...",
                "PumpPendingSceneUpload::BuildBLAS");
            break;
        case Stage::BuildTLAS:
            _sceneLoadStatus.lastBlockingWait = "ImmediateSubmit during TLAS";
            _device.SetDebugWaitContext(
                "scene=" + _pendingSceneUpload.path.string() + " stage=BuildTLAS state=" + std::to_string(static_cast<uint32_t>(_sceneLoadStatus.state)));
            _pendingSceneUpload.scene.BuildTopLevelAccelerationStructure(_device);
            _pendingSceneUpload.stage = Stage::SwapScene;
            ApplySceneLoadState(_sceneLoadStatus,
                SceneLoadState::ReadyToSwap,
                "Finalizing " + _pendingSceneUpload.path.filename().string() + "...",
                "PumpPendingSceneUpload::BuildTLAS");
            break;
        case Stage::SwapScene:
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.vertexOffsetBytes >= sizeof(vesta::scene::SceneVertex) * prepared->vertices.size()
                    || prepared->vertices.empty(),
                "SwapScene requires vertex upload completion.");
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.materialOffsetBytes >= sizeof(vesta::scene::SceneMaterial) * prepared->materials.size()
                    || prepared->materials.empty(),
                "SwapScene requires material upload completion.");
            VESTA_ASSERT_STATE(
                _pendingSceneUpload.gaussianOffsetBytes >= sizeof(vesta::scene::GaussianPrimitive) * prepared->gaussians.size()
                    || prepared->gaussians.empty(),
                "SwapScene requires gaussian upload completion.");
            _sceneLoadStatus.lastBlockingWait = "FlushUploadBatch before SwapScene";
            _device.FlushUploadBatch();
            _pendingSceneUpload.uploadMs +=
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
            _sceneLoadStatus.geometryUploadMs = _pendingSceneUpload.uploadMs;
            _sceneLoadStatus.textureUploadMs = _pendingSceneUpload.textureUploadMs;
            _sceneLoadInProgress = false;
            ApplyLoadedScene(std::move(_pendingSceneUpload.scene));
            _pendingSceneUpload = {};
            return;
        case Stage::Idle:
        default:
            _device.FlushUploadBatch();
            _pendingSceneUpload.uploadMs +=
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
            _pendingSceneUpload = {};
            _sceneLoadInProgress = false;
            return;
        }
        break;
    }

    _device.FlushUploadBatch();

    _pendingSceneUpload.uploadMs +=
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - uploadStart).count();
    _sceneLoadStatus.geometryUploadMs = _pendingSceneUpload.uploadMs;
    _sceneLoadStatus.textureUploadMs = _pendingSceneUpload.textureUploadMs;
    _sceneLoadStatus.blasMs = _pendingSceneUpload.scene.GetBottomLevelBuildMs();
    _sceneLoadStatus.tlasMs = _pendingSceneUpload.scene.GetTopLevelBuildMs();
}

void Renderer::PumpVisibilityResults()
{
    if (!_visibilityFuture.valid()) {
        return;
    }

    using namespace std::chrono_literals;
    if (_visibilityFuture.wait_for(0ms) != std::future_status::ready) {
        return;
    }

    VisibilityCullResult result = _visibilityFuture.get();
    _visibilityCullInProgress = false;

    if (_scene.GetPreparedScene() != result.scene) {
        return;
    }

    _visibleSceneToken = std::move(result.scene);
    _visibleSurfaceIndices = std::move(result.visibleSurfaceIndices);
    _frameSnapshot.visibleSet.scene = _visibleSceneToken;
    _frameSnapshot.visibleSet.surfaceIndices = _visibleSurfaceIndices;
}

void Renderer::DispatchVisibilityCullIfNeeded()
{
    if (!_settings.enableFrustumCulling && !_settings.enableDistanceCulling) {
        _visibleSurfaceIndices.clear();
        _visibleSceneToken.reset();
        _frameSnapshot = {};
        _visibilityDirty = false;
        return;
    }
    if (_visibilityCullInProgress || !_visibilityDirty) {
        return;
    }

    std::shared_ptr<const vesta::scene::PreparedScene> prepared = _scene.GetPreparedScene();
    if (!prepared || prepared->surfaces.empty()) {
        _visibleSurfaceIndices.clear();
        _visibleSceneToken = std::move(prepared);
        _frameSnapshot.visibleSet.scene = _visibleSceneToken;
        _frameSnapshot.visibleSet.surfaceIndices.clear();
        _visibilityDirty = false;
        return;
    }

    _visibilityCullInProgress = true;
    _visibilityDirty = false;
    const glm::mat4 viewProjection = _camera.GetViewProjection();
    const glm::vec3 cameraPosition = _camera.GetPosition();
    const bool useFrustumCulling = _settings.enableFrustumCulling;
    const bool useDistanceCulling = _settings.enableDistanceCulling;
    const float distanceCullScale = _settings.distanceCullScale;
    const float sceneRadius = prepared->bounds.radius;
    _visibilityFuture = _jobs.Submit(vesta::core::JobPriority::Normal,
        [this, prepared, viewProjection, cameraPosition, useFrustumCulling, useDistanceCulling, distanceCullScale, sceneRadius]() {
        VisibilityCullResult result;
        result.scene = prepared;

        if (prepared->surfaceBounds.empty()) {
            result.visibleSurfaceIndices.resize(prepared->surfaces.size());
            for (uint32_t surfaceIndex = 0; surfaceIndex < static_cast<uint32_t>(prepared->surfaces.size()); ++surfaceIndex) {
                result.visibleSurfaceIndices[surfaceIndex] = surfaceIndex;
            }
            return result;
        }

        const std::array<glm::vec4, 6> frustumPlanes = ExtractFrustumPlanes(viewProjection);
        if (_jobs.GetWorkerCount() <= 1 || prepared->surfaceBounds.size() < 64) {
            for (uint32_t surfaceIndex = 0; surfaceIndex < static_cast<uint32_t>(prepared->surfaceBounds.size()); ++surfaceIndex) {
                const auto& bounds = prepared->surfaceBounds[surfaceIndex];
                if ((!useFrustumCulling || IsSurfaceVisible(bounds, frustumPlanes))
                    && (!useDistanceCulling || IsSurfaceWithinDistance(bounds, cameraPosition, sceneRadius, distanceCullScale))) {
                    result.visibleSurfaceIndices.push_back(surfaceIndex);
                }
            }
            return result;
        }

        const size_t chunkSize = 64;
        const size_t chunkCount = (prepared->surfaceBounds.size() + chunkSize - 1) / chunkSize;
        std::vector<std::vector<uint32_t>> visibleChunks(chunkCount);
        std::future<void> cullFuture = _jobs.ParallelFor(chunkCount, 1, vesta::core::JobPriority::High, [&](size_t begin, size_t end) {
            for (size_t chunkIndex = begin; chunkIndex < end; ++chunkIndex) {
                const size_t surfaceBegin = chunkIndex * chunkSize;
                const size_t surfaceEnd = std::min(prepared->surfaceBounds.size(), surfaceBegin + chunkSize);
                std::vector<uint32_t>& chunk = visibleChunks[chunkIndex];
                chunk.reserve(surfaceEnd - surfaceBegin);
                for (size_t surfaceIndex = surfaceBegin; surfaceIndex < surfaceEnd; ++surfaceIndex) {
                    const auto& bounds = prepared->surfaceBounds[surfaceIndex];
                    if ((!useFrustumCulling || IsSurfaceVisible(bounds, frustumPlanes))
                        && (!useDistanceCulling || IsSurfaceWithinDistance(bounds, cameraPosition, sceneRadius, distanceCullScale))) {
                        chunk.push_back(static_cast<uint32_t>(surfaceIndex));
                    }
                }
            }
        });
        cullFuture.get();

        size_t totalVisible = 0;
        for (const std::vector<uint32_t>& chunk : visibleChunks) {
            totalVisible += chunk.size();
        }
        result.visibleSurfaceIndices.reserve(totalVisible);
        for (std::vector<uint32_t>& chunk : visibleChunks) {
            result.visibleSurfaceIndices.insert(result.visibleSurfaceIndices.end(), chunk.begin(), chunk.end());
        }
        return result;
    });
}

bool Renderer::LoadSceneResolved(const std::filesystem::path& resolvedPath)
{
    _sceneLoadStatus = SceneLoadStatus{
        .state = SceneLoadState::Parsing,
        .path = resolvedPath,
        .message = "Parsing " + resolvedPath.filename().string() + "...",
    };

    try {
        const auto parseStart = std::chrono::steady_clock::now();
        vesta::scene::Scene loadedScene;
        if (!loadedScene.ParseFromFile(resolvedPath)) {
            _sceneLoadStatus.state = SceneLoadState::Failed;
            _sceneLoadStatus.parseMs =
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();
            _sceneLoadStatus.message = "Failed to load " + resolvedPath.filename().string();
            return false;
        }
        _sceneLoadStatus.parseMs =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - parseStart).count();

        ApplySceneLoadState(_sceneLoadStatus,
            SceneLoadState::Preparing,
            "Preparing " + resolvedPath.filename().string() + "...",
            "LoadSceneResolved");
        const auto prepareStart = std::chrono::steady_clock::now();
        if (!loadedScene.PrepareParsedScene()) {
            _sceneLoadStatus.state = SceneLoadState::Failed;
            _sceneLoadStatus.prepareMs =
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prepareStart).count();
            _sceneLoadStatus.message = "Failed to prepare " + resolvedPath.filename().string();
            return false;
        }
        _sceneLoadStatus.prepareMs =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prepareStart).count();

        ApplySceneLoadState(_sceneLoadStatus,
            SceneLoadState::UploadingGeometry,
            "Uploading " + resolvedPath.filename().string() + "...",
            "LoadSceneResolved");
        if (UsesStreamingUpload(_settings)) {
            StartPendingSceneUpload(std::move(loadedScene), _sceneLoadStatus.parseMs, _sceneLoadStatus.prepareMs);
        } else {
            VESTA_ASSERT(!_startupSafeModeActive,
                "Startup safe mode must not synchronously apply scenes from LoadSceneResolved.");
            ApplySceneLoadState(_sceneLoadStatus,
                SceneLoadState::ReadyToSwap,
                "Finalizing " + resolvedPath.filename().string() + "...",
                "LoadSceneResolved");
            ApplyLoadedScene(std::move(loadedScene));
        }
        return true;
    } catch (const std::exception& exception) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Failed to load " + resolvedPath.filename().string() + ": " + exception.what();
        return false;
    } catch (...) {
        _sceneLoadStatus.state = SceneLoadState::Failed;
        _sceneLoadStatus.message = "Failed to load " + resolvedPath.filename().string();
        return false;
    }
}

void Renderer::StartPendingSceneUpload(vesta::scene::Scene&& scene, float parseMs, float prepareMs)
{
    _pendingSceneUpload = PendingSceneUpload{
        .scene = std::move(scene),
        .path = _sceneLoadStatus.path,
        .parseMs = parseMs,
        .prepareMs = prepareMs,
        .uploadMs = 0.0f,
        .textureUploadMs = 0.0f,
        .stage = PendingSceneUploadStage::AllocateBuffers,
        .active = true,
    };
    _sceneLoadInProgress = true;
    ValidateSceneLoadTransition(_sceneLoadStatus, SceneLoadState::UploadingGeometry, "StartPendingSceneUpload");
    _sceneLoadStatus.state = SceneLoadState::UploadingGeometry;
    _sceneLoadStatus.parseMs = parseMs;
    _sceneLoadStatus.prepareMs = prepareMs;
    _sceneLoadStatus.geometryUploadMs = 0.0f;
    _sceneLoadStatus.textureUploadMs = 0.0f;
    _sceneLoadStatus.blasMs = 0.0f;
    _sceneLoadStatus.tlasMs = 0.0f;
    _sceneLoadStatus.lastBlockingWait.clear();
    _sceneLoadStatus.message = "Allocating GPU buffers for " + _pendingSceneUpload.path.filename().string() + "...";
}

void Renderer::ApplyLoadedScene(vesta::scene::Scene&& scene)
{
    if (_startupSafeModeActive) {
        VESTA_ASSERT(UsesStreamingUpload(_settings), "Startup safe mode requires streaming upload before ApplyLoadedScene.");
    }
    vesta::scene::Scene previousScene = std::move(_scene);
    _scene = std::move(scene);
    if (!_scene.GetVertexBuffer()) {
        _sceneLoadStatus.lastBlockingWait = "Scene::UploadToGpu";
        _device.SetDebugWaitContext("scene=" + _scene.GetSourcePath().string() + " stage=ApplyLoadedScene::UploadToGpu");
        _scene.UploadToGpu(_device, GetSceneUploadOptions());
        _sceneLoadStatus.geometryUploadMs = _scene.GetGeometryUploadMs();
        _sceneLoadStatus.textureUploadMs = _scene.GetTextureUploadMs();
    }
    _sceneLoadStatus.blasMs = _scene.GetBottomLevelBuildMs();
    _sceneLoadStatus.tlasMs = _scene.GetTopLevelBuildMs();

    if (_settings.autoFocusSceneOnLoad) {
        _camera.Focus(_scene.GetBounds().center, _scene.GetBounds().radius);
    }
    if (!_scene.HasTrainedGaussians() && _scene.SupportsRealtimeGaussianSorting()) {
        _scene.ResortGaussians(_device, _camera);
    }
    ResetAccumulation();
    _visibilityDirty = true;
    _visibleSurfaceIndices.clear();
    _visibleSceneToken.reset();
    _frameSnapshot = {};
    ClearSelection();
    if (_scene.HasTrainedGaussians()) {
        _gaussianInteractivePreviewFramesRemaining = GaussianInteractivePreviewFrameBudget(_scene);
    }

    if (!previousScene.GetSourcePath().empty()) {
        if (_settings.deferOldSceneDestruction) {
            _retiredScenes.push_back(RetiredSceneEntry{
                .scene = std::move(previousScene),
                .safeFrameNumber = _frameNumber + kFrameOverlap,
            });
        } else {
            _device.WaitIdle();
            previousScene.DestroyGpu(_device);
        }
    }

    ApplySceneLoadState(
        _sceneLoadStatus, SceneLoadState::Ready, "Loaded " + _scene.GetSourcePath().filename().string(), "ApplyLoadedScene");
    _sceneLoadStatus.path = _scene.GetSourcePath();
    _sceneLoadStatus.uploadStage.clear();
    _sceneLoadStatus.lastBlockingWait.clear();
}

void Renderer::ReleaseRetiredScenes()
{
    while (!_retiredScenes.empty() && _retiredScenes.front().safeFrameNumber <= _frameNumber) {
        _retiredScenes.front().scene.DestroyGpu(_device);
        _retiredScenes.pop_front();
    }
}

RendererFrameContext& Renderer::GetCurrentFrame()
{
    return _frames[_frameNumber % kFrameOverlap];
}

RenderGraph Renderer::BuildFrameGraph(uint32_t swapchainImageIndex)
{
    RebuildPassExecutionPlan();

    RenderGraph graph;
    const bool useGeometryPass = NeedsGeometryPass(_settings);
    const bool useDeferredPass = NeedsDeferredPass(_settings);
    const bool useGaussianPass = NeedsGaussianPass(_settings);
    const bool usePathTracePass = NeedsPathTracePass(_settings);
    const bool useOfficialGaussianPass = useGaussianPass && _scene.HasTrainedGaussians() && !IsGaussianInteractivePreviewActive();
    const bool useLegacyGaussianPass = useGaussianPass && (!_scene.HasTrainedGaussians() || IsGaussianInteractivePreviewActive());

    const VkExtent2D swapchainExtent = _device.GetSwapchainExtent();
    const VkExtent3D renderExtent{ swapchainExtent.width, swapchainExtent.height, 1 };

    // These logical resources describe the full frame. The graph decides which
    // concrete VkImage each handle resolves to for this frame execution.
    RendererGraphResources resources;
    resources.swapchainTarget =
        graph.ImportTexture("SwapchainTarget", _device.GetSwapchainImageHandle(swapchainImageIndex), ResourceUsage::Undefined);

    ImageDesc gbufferDesc{};
    gbufferDesc.extent = renderExtent;
    gbufferDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    gbufferDesc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    gbufferDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    gbufferDesc.registerBindlessStorage = true;

    ImageDesc depthDesc{};
    depthDesc.extent = renderExtent;
    depthDesc.format = VK_FORMAT_D32_SFLOAT;
    depthDesc.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    depthDesc.registerBindlessSampled = true;

    ImageDesc storageDesc{};
    storageDesc.extent = renderExtent;
    storageDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    storageDesc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    storageDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    storageDesc.registerBindlessStorage = true;

    ImageDesc pathTraceDesc = storageDesc;
    pathTraceDesc.extent = ScaleExtent(renderExtent, _settings.pathTraceResolutionScale);

    if (useGeometryPass) {
        resources.gbufferAlbedo = graph.CreateTexture("GBuffer.Albedo", gbufferDesc);
        resources.gbufferNormal = graph.CreateTexture("GBuffer.NormalRoughness", gbufferDesc);
        resources.gbufferMaterial = graph.CreateTexture("GBuffer.Material", gbufferDesc);
        resources.sceneDepth = graph.CreateTexture("SceneDepth", depthDesc);
    }
    if (useDeferredPass) {
        resources.deferredLighting = graph.CreateTexture("DeferredLighting", storageDesc);
    }
    if (usePathTracePass) {
        resources.pathTraceOutput = graph.CreateTexture("PathTraceOutput", pathTraceDesc);
    }
    if (useGaussianPass) {
        resources.gaussianAccum = graph.CreateTexture("GaussianAccum", storageDesc);
        resources.gaussianReveal = graph.CreateTexture("GaussianReveal", storageDesc);
    }

    for (RegisteredPassEntry* entry : _passExecutionPlan) {
        const std::string_view id = entry->id;
        if (id == "geometry-raster" && !useGeometryPass) {
            continue;
        }
        if (id == "deferred-lighting" && !useDeferredPass) {
            continue;
        }
        if (id == "gaussian-splat" && !useLegacyGaussianPass) {
            continue;
        }
        if (id == "official-gaussian-raster" && !useOfficialGaussianPass) {
            continue;
        }
        if (id == "path-tracer" && !usePathTracePass) {
            continue;
        }

        if (entry->configure) {
            entry->configure(*entry->pass, resources);
        }
        graph.AddPass(*entry->pass);
    }

    graph.SetFinalUsage(resources.swapchainTarget, ResourceUsage::Present);
    return graph;
}

Renderer::RegisteredPassEntry* Renderer::FindPassEntry(std::string_view id)
{
    const auto it = std::find_if(_passRegistry.begin(), _passRegistry.end(), [id](const RegisteredPassEntry& entry) {
        return entry.id == id;
    });
    return it != _passRegistry.end() ? &(*it) : nullptr;
}

const Renderer::RegisteredPassEntry* Renderer::FindPassEntry(std::string_view id) const
{
    const auto it = std::find_if(_passRegistry.begin(), _passRegistry.end(), [id](const RegisteredPassEntry& entry) {
        return entry.id == id;
    });
    return it != _passRegistry.end() ? &(*it) : nullptr;
}

} // namespace vesta::render
