#include <vesta/render/renderer.h>

#include <algorithm>
#include <limits>
#include <utility>

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vesta/render/passes/composite_pass.h>
#include <vesta/render/passes/deferred_raster_pass.h>
#include <vesta/render/passes/gaussian_splat_pass.h>
#include <vesta/render/passes/path_tracer_pass.h>
#include <vesta/render/vulkan/vk_initializers.h>

namespace vesta::render {
namespace {
void ConfigureDeferredRasterPass(IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& deferredPass = static_cast<DeferredRasterPass&>(pass);
    deferredPass.SetGBufferTargets(resources.gbufferAlbedo, resources.gbufferNormal, resources.sceneDepth);
    deferredPass.SetLightingTarget(resources.deferredLighting);
}

void ConfigurePathTracerPass(IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& pathTracerPass = static_cast<PathTracerPass&>(pass);
    pathTracerPass.SetDepthInput(resources.sceneDepth);
    pathTracerPass.SetOutput(resources.pathTraceOutput);
}

void ConfigureGaussianSplatPass(IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& gaussianPass = static_cast<GaussianSplatPass&>(pass);
    gaussianPass.SetDepthInput(resources.sceneDepth);
    gaussianPass.SetOutput(resources.gaussianOutput);
}

void ConfigureCompositePass(IRenderPass& pass, const RendererGraphResources& resources)
{
    auto& compositePass = static_cast<CompositePass&>(pass);
    compositePass.SetInputs(resources.deferredLighting, resources.pathTraceOutput, resources.gaussianOutput);
    compositePass.SetOutput(resources.swapchainTarget);
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

    InitializeCommands();
    InitializeSyncStructures();
    InitializeDefaultPasses();
    return true;
}

void Renderer::Shutdown()
{
    _device.WaitIdle();
    DestroyFrameResources();
    _transientImagePool.Purge(_device);
    ClearPassRegistry();
    _device.Shutdown();
    _window = nullptr;
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
        .id = "deferred-raster",
        .pass = std::make_unique<DeferredRasterPass>(),
        .configure = ConfigureDeferredRasterPass,
        .order = 100,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "path-tracer",
        .pass = std::make_unique<PathTracerPass>(),
        .configure = ConfigurePathTracerPass,
        .order = 200,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "gaussian-splat",
        .pass = std::make_unique<GaussianSplatPass>(),
        .configure = ConfigureGaussianSplatPass,
        .order = 300,
        .enabled = true,
    });
    RegisterPass(RenderPassRegistrationDesc{
        .id = "composite",
        .pass = std::make_unique<CompositePass>(),
        .configure = ConfigureCompositePass,
        .order = 400,
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

    VkSemaphoreCreateInfo semaphoreInfo = vkinit::semaphore_create_info();
    _swapchainImageRenderSemaphores.resize(_device.GetSwapchainImageHandles().size(), VK_NULL_HANDLE);
    for (VkSemaphore& semaphore : _swapchainImageRenderSemaphores) {
        VK_CHECK(vkCreateSemaphore(_device.GetDevice(), &semaphoreInfo, nullptr, &semaphore));
    }
}

void Renderer::ClearPassRegistry()
{
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
    gbufferDesc.format = _device.GetSwapchainFormat();
    gbufferDesc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    gbufferDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    ImageDesc depthDesc{};
    depthDesc.extent = renderExtent;
    depthDesc.format = VK_FORMAT_D32_SFLOAT;
    depthDesc.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    ImageDesc storageDesc{};
    storageDesc.extent = renderExtent;
    storageDesc.format = _device.GetSwapchainFormat();
    storageDesc.aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    storageDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    resources.gbufferAlbedo = graph.CreateTexture("GBuffer.Albedo", gbufferDesc);
    resources.gbufferNormal = graph.CreateTexture("GBuffer.Normal", gbufferDesc);
    resources.sceneDepth = graph.CreateTexture("SceneDepth", depthDesc);
    resources.deferredLighting = graph.CreateTexture("DeferredLighting", gbufferDesc);
    resources.pathTraceOutput = graph.CreateTexture("PathTraceOutput", storageDesc);
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
} // namespace vesta::render
