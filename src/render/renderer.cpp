#include <vesta/render/renderer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <memory>
#include <utility>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vesta/render/passes/composite_pass.h>
#include <vesta/render/passes/deferred_lighting_pass.h>
#include <vesta/render/passes/gaussian_splat_pass.h>
#include <vesta/render/passes/geometry_raster_pass.h>
#include <vesta/render/passes/path_tracer_pass.h>
#include <vesta/render/vulkan/vk_images.h>
#include <vesta/render/vulkan/vk_initializers.h>
#include <vesta/render/vulkan/vk_loader.h>

namespace vesta::render {
namespace {
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
    settings.displayMode = RendererDisplayMode::Composite;
    settings.enableGaussian = true;
    settings.enablePathTracing = true;
    settings.gaussianPointSize = 8.0f;
    settings.gaussianOpacity = 0.35f;

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

void ConfigureGeometryRasterPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& rasterPass = static_cast<GeometryRasterPass&>(pass);
    rasterPass.SetTargets(resources.gbufferAlbedo, resources.gbufferNormal, resources.sceneDepth);
    rasterPass.SetScene(&renderer.GetScene());
    rasterPass.SetCamera(&renderer.GetCamera());
}

void ConfigureDeferredLightingPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& lightingPass = static_cast<DeferredLightingPass&>(pass);
    lightingPass.SetInputs(resources.gbufferAlbedo, resources.gbufferNormal);
    lightingPass.SetOutput(resources.deferredLighting);
}

void ConfigureGaussianPass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& gaussianPass = static_cast<GaussianSplatPass&>(pass);
    gaussianPass.SetDepthInput(resources.sceneDepth);
    gaussianPass.SetOutput(resources.gaussianOutput);
    gaussianPass.SetScene(&renderer.GetScene());
    gaussianPass.SetCamera(&renderer.GetCamera());
    gaussianPass.SetParams(renderer.GetSettings().gaussianPointSize,
        renderer.GetSettings().gaussianOpacity,
        renderer.GetSettings().enableGaussian);
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
}

void ConfigureCompositePass(Renderer& renderer, IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& compositePass = static_cast<CompositePass&>(pass);
    compositePass.SetInputs(resources.deferredLighting, resources.pathTraceOutput, resources.gaussianOutput);
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

    RenderDeviceDesc deviceDesc;
    deviceDesc.swapchainExtent = initialExtent;
    deviceDesc.enableValidation = enableValidation;
    _device.Initialize(window, deviceDesc);
    ApplyPreset(RendererPreset::Recommended);

    _camera.SetViewport(initialExtent.width, initialExtent.height);
    LoadDefaultScene();
    InitializeCommands();
    InitializeSyncStructures();
    InitializeDefaultPasses();
    return true;
}

void Renderer::Shutdown()
{
    _device.WaitIdle();
    ClearPassRegistry();
    DestroyFrameResources();
    _transientImagePool.Purge(_device);
    _scene.DestroyGpu(_device);
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
        return;
    }

    switch (event.key.keysym.sym) {
    case SDLK_1:
        _settings.displayMode = RendererDisplayMode::Composite;
        break;
    case SDLK_2:
        _settings.displayMode = RendererDisplayMode::DeferredLighting;
        break;
    case SDLK_3:
        _settings.displayMode = RendererDisplayMode::Gaussian;
        break;
    case SDLK_4:
        _settings.displayMode = RendererDisplayMode::PathTrace;
        break;
    case SDLK_g:
        _settings.enableGaussian = !_settings.enableGaussian;
        break;
    case SDLK_p:
        _settings.enablePathTracing = !_settings.enablePathTracing;
        break;
    default:
        break;
    }
}

void Renderer::Update(float deltaSeconds)
{
    _frameTimeMs = deltaSeconds * 1000.0f;
    _smoothedFrameTimeMs = _smoothedFrameTimeMs <= 0.0f ? _frameTimeMs : (_smoothedFrameTimeMs * 0.9f + _frameTimeMs * 0.1f);

    _camera.Update(deltaSeconds);
    if (_camera.ConsumeMoved()) {
        _pathTraceFrameIndex = 0;
    } else {
        ++_pathTraceFrameIndex;
    }
}

void Renderer::RenderFrame()
{
    RendererFrameContext& currentFrame = GetCurrentFrame();

    VK_CHECK(vkWaitForFences(_device.GetDevice(), 1, &currentFrame.renderFence, VK_TRUE, std::numeric_limits<uint64_t>::max()));
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

void Renderer::ApplyPreset(RendererPreset preset)
{
    ApplyPresetSettings(_settings, _device, preset);
    ResetAccumulation();
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

RendererFrameContext& Renderer::GetCurrentFrame()
{
    return _frames[_frameNumber % kFrameOverlap];
}

RenderGraph Renderer::BuildFrameGraph(uint32_t swapchainImageIndex)
{
    RebuildPassExecutionPlan();

    RenderGraph graph;

    const VkExtent2D swapchainExtent = _device.GetSwapchainExtent();
    const VkExtent3D renderExtent{ swapchainExtent.width, swapchainExtent.height, 1 };

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
    depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    ImageDesc storageDesc{};
    storageDesc.extent = renderExtent;
    storageDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    storageDesc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    storageDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    storageDesc.registerBindlessStorage = true;

    ImageDesc pathTraceDesc = storageDesc;
    pathTraceDesc.extent = ScaleExtent(renderExtent, _settings.pathTraceResolutionScale);

    resources.gbufferAlbedo = graph.CreateTexture("GBuffer.Albedo", gbufferDesc);
    resources.gbufferNormal = graph.CreateTexture("GBuffer.NormalDepth", gbufferDesc);
    resources.sceneDepth = graph.CreateTexture("SceneDepth", depthDesc);
    resources.deferredLighting = graph.CreateTexture("DeferredLighting", storageDesc);
    resources.pathTraceOutput = graph.CreateTexture("PathTraceOutput", pathTraceDesc);
    resources.gaussianOutput = graph.CreateTexture("GaussianOutput", storageDesc);

    for (RegisteredPassEntry* entry : _passExecutionPlan) {
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

void Renderer::LoadDefaultScene()
{
    try {
        const std::array<std::filesystem::path, 2> candidates{
            std::filesystem::path("assets/basicmesh.glb"),
            std::filesystem::path("assets/structure.glb"),
        };

        for (const std::filesystem::path& candidate : candidates) {
            const std::filesystem::path resolved = vkutil::resolve_runtime_path(candidate);
            if (_scene.LoadFromFile(resolved)) {
                _scene.UploadToGpu(_device);
                _camera.Focus(_scene.GetBounds().center, _scene.GetBounds().radius);
                break;
            }
        }
    } catch (...) {
    }
}
} // namespace vesta::render
