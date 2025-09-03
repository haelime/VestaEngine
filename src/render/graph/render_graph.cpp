#include <vesta/render/graph/render_graph.h>

#include <algorithm>
#include <chrono>
#include <limits>

#include <vesta/render/renderer.h>
#include <vesta/render/vulkan/vk_initializers.h>

namespace vesta::render {
// RenderGraphBuilder is intentionally tiny. Passes only say "I read this" or
// "I write that", and the graph handles the synchronization details later.
void RenderGraphBuilder::Read(GraphTextureHandle texture, ResourceUsage usage)
{
    _passNode->reads.push_back(RenderGraphTextureAccess{ texture, usage });
}

void RenderGraphBuilder::Write(GraphTextureHandle texture, ResourceUsage usage)
{
    _passNode->writes.push_back(RenderGraphTextureAccess{ texture, usage });
}

RenderGraphContext::RenderGraphContext(RenderDevice& device,
    const std::vector<ImageHandle>& resolvedImages,
    VkCommandBuffer commandBuffer,
    VkExtent2D renderExtent)
    : _device(device), _resolvedImages(resolvedImages), _commandBuffer(commandBuffer), _renderExtent(renderExtent)
{
}

VkImage RenderGraphContext::GetTexture(GraphTextureHandle texture) const
{
    return _device.GetImage(_resolvedImages.at(texture.index));
}

ImageHandle RenderGraphContext::GetTextureHandle(GraphTextureHandle texture) const
{
    return _resolvedImages.at(texture.index);
}

VkImageView RenderGraphContext::GetTextureView(GraphTextureHandle texture) const
{
    return _device.GetImageView(_resolvedImages.at(texture.index));
}

VkFormat RenderGraphContext::GetTextureFormat(GraphTextureHandle texture) const
{
    return _device.GetImageFormat(_resolvedImages.at(texture.index));
}

VkExtent3D RenderGraphContext::GetTextureExtent(GraphTextureHandle texture) const
{
    return _device.GetImageExtent(_resolvedImages.at(texture.index));
}

GraphTextureHandle RenderGraph::CreateTexture(std::string_view name, const ImageDesc& desc)
{
    _compiled = false;
    _textures.push_back(TextureResource{ std::string(name), desc, {}, ResourceUsage::Undefined, false });
    return GraphTextureHandle{ static_cast<uint32_t>(_textures.size() - 1) };
}

GraphTextureHandle RenderGraph::ImportTexture(std::string_view name, ImageHandle image, ResourceUsage initialUsage)
{
    _compiled = false;
    _textures.push_back(TextureResource{ std::string(name), {}, image, initialUsage, true });
    return GraphTextureHandle{ static_cast<uint32_t>(_textures.size() - 1) };
}

void RenderGraph::SetFinalUsage(GraphTextureHandle texture, ResourceUsage usage)
{
    _compiled = false;
    _finalUsages.push_back(FinalUsageRequest{ texture, usage });
}

void RenderGraph::AddPass(IRenderPass& pass)
{
    _compiled = false;
    _passes.push_back(RenderGraphPassNode{ .pass = &pass });

    RenderGraphBuilder builder;
    builder._passNode = &_passes.back();
    pass.Setup(builder);
}

RenderGraph::TextureState RenderGraph::ResolveUsage(const TextureResource& resource, ResourceUsage usage)
{
    const bool depth = (resource.desc.aspectFlags & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;

    switch (usage) {
    case ResourceUsage::Undefined:
        return {};
    case ResourceUsage::ColorAttachmentWrite:
        return TextureState{ true,
            usage,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    case ResourceUsage::DepthAttachmentWrite:
        return TextureState{ true,
            usage,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL };
    case ResourceUsage::DepthRead:
        return TextureState{ true,
            usage,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    case ResourceUsage::SampledRead:
        return TextureState{ true,
            usage,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    case ResourceUsage::StorageRead:
        return TextureState{ true,
            usage,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL };
    case ResourceUsage::StorageWrite:
        return TextureState{ true,
            usage,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL };
    case ResourceUsage::TransferSrc:
        return TextureState{ true, usage, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
    case ResourceUsage::TransferDst:
        return TextureState{ true, usage, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL };
    case ResourceUsage::Present:
        return TextureState{ true, usage, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR };
    }

    return {};
}

bool RenderGraph::IsWriteUsage(ResourceUsage usage)
{
    switch (usage) {
    case ResourceUsage::ColorAttachmentWrite:
    case ResourceUsage::DepthAttachmentWrite:
    case ResourceUsage::StorageWrite:
    case ResourceUsage::TransferDst:
        return true;
    default:
        return false;
    }
}

void RenderGraph::Compile(RenderDevice& device)
{
    // Compile walks the pass list once and simulates texture state over time.
    // That lets us emit the exact image barriers needed before each pass.
    std::vector<TextureState> states(_textures.size());
    _compiledResources.clear();
    _compiledResources.resize(_textures.size());
    _compiledPasses.clear();
    _finalBarriers.clear();
    _compiledPasses.reserve(_passes.size());

    for (size_t textureIndex = 0; textureIndex < _textures.size(); ++textureIndex) {
        TextureResource& texture = _textures[textureIndex];
        if (texture.imported) {
            const AllocatedImage& image = device.GetImageResource(texture.importedImage);
            texture.desc = image.desc;
        }

        _compiledResources[textureIndex] = CompiledTextureResource{
            .desc = texture.desc,
            .importedImage = texture.importedImage,
            .imported = texture.imported,
        };
        states[textureIndex] = ResolveUsage(texture, texture.initialUsage);
    }

    for (const RenderGraphPassNode& pass : _passes) {
        CompiledPass compiledPass;
        compiledPass.pass = pass.pass;
        compiledPass.readCount = static_cast<uint32_t>(pass.reads.size());
        compiledPass.writeCount = static_cast<uint32_t>(pass.writes.size());

        auto makeAccessSummary = [&](const RenderGraphTextureAccess& access) {
            const TextureResource& resource = _textures[access.texture.index];
            return RenderGraphPassTiming::ResourceAccess{
                .name = resource.name,
                .usage = access.usage,
                .format = resource.desc.format,
                .extent = resource.desc.extent,
            };
        };

        auto processAccess = [&](const RenderGraphTextureAccess& access) {
            TextureResource& resource = _textures[access.texture.index];
            TextureState& previous = states[access.texture.index];
            const TextureState next = ResolveUsage(resource, access.usage);
            const bool needsBarrier =
                !previous.initialized || previous.layout != next.layout || IsWriteUsage(previous.usage) || IsWriteUsage(access.usage);
            if (needsBarrier) {
                CompiledBarrier barrier;
                barrier.resource = access.texture;
                barrier.fromUsage = previous.usage;
                barrier.toUsage = access.usage;
                barrier.imageBarrier = vkinit::image_barrier(VK_NULL_HANDLE,
                    previous.initialized ? previous.stageMask : VK_PIPELINE_STAGE_2_NONE,
                    previous.initialized ? previous.accessMask : VK_ACCESS_2_NONE,
                    next.stageMask,
                    next.accessMask,
                    previous.initialized ? previous.layout : VK_IMAGE_LAYOUT_UNDEFINED,
                    next.layout,
                    resource.desc.aspectFlags);
                compiledPass.barriers.push_back(barrier);
            }

            previous = next;
        };

        for (const RenderGraphTextureAccess& read : pass.reads) {
            compiledPass.inputs.push_back(makeAccessSummary(read));
            processAccess(read);
        }
        for (const RenderGraphTextureAccess& write : pass.writes) {
            compiledPass.outputs.push_back(makeAccessSummary(write));
            processAccess(write);
        }

        _compiledPasses.push_back(std::move(compiledPass));
    }

    for (const FinalUsageRequest& finalUsage : _finalUsages) {
        TextureResource& resource = _textures[finalUsage.texture.index];
        TextureState& previous = states[finalUsage.texture.index];
        const TextureState next = ResolveUsage(resource, finalUsage.usage);
        const bool needsBarrier =
            !previous.initialized || previous.layout != next.layout || IsWriteUsage(previous.usage) || IsWriteUsage(finalUsage.usage);
        if (!needsBarrier) {
            continue;
        }

        CompiledBarrier barrier;
        barrier.resource = finalUsage.texture;
        barrier.fromUsage = previous.usage;
        barrier.toUsage = finalUsage.usage;
        barrier.imageBarrier = vkinit::image_barrier(VK_NULL_HANDLE,
            previous.initialized ? previous.stageMask : VK_PIPELINE_STAGE_2_NONE,
            previous.initialized ? previous.accessMask : VK_ACCESS_2_NONE,
            next.stageMask,
            next.accessMask,
            previous.initialized ? previous.layout : VK_IMAGE_LAYOUT_UNDEFINED,
            next.layout,
            resource.desc.aspectFlags);
        _finalBarriers.push_back(barrier);
        previous = next;
    }

    _compiled = true;
}

void RenderGraph::Execute(RenderGraphExecutionContext& executionContext)
{
    if (!_compiled) {
        Compile(executionContext.device);
    }
    if (executionContext.passTimings != nullptr) {
        executionContext.passTimings->clear();
        executionContext.passTimings->reserve(_compiledPasses.size());
    }

    std::vector<float> previousGpuMs;
    RendererFrameContext& frameContext = executionContext.frameContext;
    const bool useGpuTimestamps = executionContext.gpuTimestampsSupported
        && frameContext.renderGraphTimestampPool != VK_NULL_HANDLE
        && executionContext.timestampPeriodNs > 0.0f;
    if (useGpuTimestamps && frameContext.renderGraphTimestampPending && frameContext.renderGraphTimestampPassCount > 0u) {
        const uint32_t queryCount = frameContext.renderGraphTimestampPassCount * 2u;
        std::vector<uint64_t> queryResults(static_cast<size_t>(queryCount) * 2u, 0u);
        const VkResult result = vkGetQueryPoolResults(executionContext.device.GetDevice(),
            frameContext.renderGraphTimestampPool,
            0,
            queryCount,
            queryResults.size() * sizeof(uint64_t),
            queryResults.data(),
            sizeof(uint64_t) * 2u,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_SUCCESS) {
            previousGpuMs.resize(frameContext.renderGraphTimestampPassCount, -1.0f);
            for (uint32_t passIndex = 0; passIndex < frameContext.renderGraphTimestampPassCount; ++passIndex) {
                const uint32_t beginQuery = passIndex * 2u;
                const uint32_t endQuery = beginQuery + 1u;
                const bool beginAvailable = queryResults[static_cast<size_t>(beginQuery) * 2u + 1u] != 0u;
                const bool endAvailable = queryResults[static_cast<size_t>(endQuery) * 2u + 1u] != 0u;
                if (!beginAvailable || !endAvailable) {
                    continue;
                }
                const uint64_t beginTimestamp = queryResults[static_cast<size_t>(beginQuery) * 2u];
                const uint64_t endTimestamp = queryResults[static_cast<size_t>(endQuery) * 2u];
                if (endTimestamp >= beginTimestamp) {
                    const double elapsedNs = static_cast<double>(endTimestamp - beginTimestamp) * executionContext.timestampPeriodNs;
                    previousGpuMs[passIndex] = static_cast<float>(elapsedNs / 1.0e6);
                }
            }
        }
        frameContext.renderGraphTimestampPending = false;
    }

    const uint32_t timestampPassCount = useGpuTimestamps
        ? std::min<uint32_t>(static_cast<uint32_t>(_compiledPasses.size()), kMaxRenderGraphTimestampPasses)
        : 0u;
    if (timestampPassCount > 0u) {
        vkCmdResetQueryPool(executionContext.commandBuffer, frameContext.renderGraphTimestampPool, 0, timestampPassCount * 2u);
        frameContext.renderGraphTimestampPassNames.clear();
        frameContext.renderGraphTimestampPassNames.reserve(timestampPassCount);
        for (uint32_t passIndex = 0; passIndex < timestampPassCount; ++passIndex) {
            frameContext.renderGraphTimestampPassNames.push_back(std::string(_compiledPasses[passIndex].pass->Name()));
        }
        frameContext.renderGraphTimestampPassCount = timestampPassCount;
        frameContext.renderGraphTimestampPending = true;
    } else {
        frameContext.renderGraphTimestampPassCount = 0;
        frameContext.renderGraphTimestampPassNames.clear();
        frameContext.renderGraphTimestampPending = false;
    }

    // Transient graph resources become real images here. Imported resources such
    // as the swapchain keep their existing image handles.
    std::vector<ImageHandle> resolvedImages(_compiledResources.size());
    for (size_t textureIndex = 0; textureIndex < _compiledResources.size(); ++textureIndex) {
        const CompiledTextureResource& resource = _compiledResources[textureIndex];
        if (resource.imported) {
            resolvedImages[textureIndex] = resource.importedImage;
            continue;
        }

        ImageHandle handle = executionContext.transientImagePool.Acquire(executionContext.device, resource.desc);
        resolvedImages[textureIndex] = handle;
        executionContext.frameContext.acquiredTransientImages.push_back(handle);
    }

    for (size_t passIndex = 0; passIndex < _compiledPasses.size(); ++passIndex) {
        const CompiledPass& compiledPass = _compiledPasses[passIndex];
        std::vector<VkImageMemoryBarrier2> imageBarriers;
        imageBarriers.reserve(compiledPass.barriers.size());

        for (const CompiledBarrier& barrier : compiledPass.barriers) {
            VkImageMemoryBarrier2 imageBarrier = barrier.imageBarrier;
            imageBarrier.image = executionContext.device.GetImage(resolvedImages[barrier.resource.index]);
            imageBarriers.push_back(imageBarrier);
        }

        if (!imageBarriers.empty()) {
            VkDependencyInfo dependencyInfo{};
            dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size());
            dependencyInfo.pImageMemoryBarriers = imageBarriers.data();
            vkCmdPipelineBarrier2(executionContext.commandBuffer, &dependencyInfo);
        }

        RenderGraphContext context(
            executionContext.device, resolvedImages, executionContext.commandBuffer, executionContext.device.GetSwapchainExtent());
        if (passIndex < timestampPassCount) {
            vkCmdWriteTimestamp2(executionContext.commandBuffer,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                frameContext.renderGraphTimestampPool,
                static_cast<uint32_t>(passIndex) * 2u);
        }
        const auto cpuBegin = std::chrono::steady_clock::now();
        compiledPass.pass->Execute(context);
        const auto cpuEnd = std::chrono::steady_clock::now();
        if (passIndex < timestampPassCount) {
            vkCmdWriteTimestamp2(executionContext.commandBuffer,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                frameContext.renderGraphTimestampPool,
                static_cast<uint32_t>(passIndex) * 2u + 1u);
        }

        if (executionContext.passTimings != nullptr) {
            const float cpuMs = std::chrono::duration<float, std::milli>(cpuEnd - cpuBegin).count();
            float gpuMs = 0.0f;
            bool gpuTimingValid = false;
            if (passIndex < previousGpuMs.size() && previousGpuMs[passIndex] >= 0.0f) {
                gpuMs = previousGpuMs[passIndex];
                gpuTimingValid = true;
            }
            executionContext.passTimings->push_back(RenderGraphPassTiming{
                .name = std::string(compiledPass.pass->Name()),
                .readCount = compiledPass.readCount,
                .writeCount = compiledPass.writeCount,
                .barrierCount = static_cast<uint32_t>(compiledPass.barriers.size()),
                .cpuMs = cpuMs,
                .gpuMs = gpuMs,
                .gpuTimingValid = gpuTimingValid,
                .inputs = compiledPass.inputs,
                .outputs = compiledPass.outputs,
            });
        }
    }

    std::vector<VkImageMemoryBarrier2> finalImageBarriers;
    finalImageBarriers.reserve(_finalBarriers.size());
    for (const CompiledBarrier& barrier : _finalBarriers) {
        VkImageMemoryBarrier2 imageBarrier = barrier.imageBarrier;
        imageBarrier.image = executionContext.device.GetImage(resolvedImages[barrier.resource.index]);
        finalImageBarriers.push_back(imageBarrier);
    }

    if (!finalImageBarriers.empty()) {
        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(finalImageBarriers.size());
        dependencyInfo.pImageMemoryBarriers = finalImageBarriers.data();
        vkCmdPipelineBarrier2(executionContext.commandBuffer, &dependencyInfo);
    }
}
} // namespace vesta::render
