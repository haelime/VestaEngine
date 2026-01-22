#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <vesta/render/passes/irender_pass.h>
#include <vesta/render/resources/resource_handles.h>
#include <vesta/render/rhi/render_device.h>

namespace vesta::render {
class TransientImagePool;
struct RendererFrameContext;

// GraphTextureHandle points at a logical resource, not necessarily a unique VkImage.
// The graph can recycle transient images between frames when descriptions match.
struct GraphTextureHandle {
    uint32_t index{ kInvalidResourceIndex };

    [[nodiscard]] explicit operator bool() const { return index != kInvalidResourceIndex; }
    [[nodiscard]] bool operator==(const GraphTextureHandle&) const = default;
};

enum class ResourceUsage {
    Undefined,
    ColorAttachmentWrite,
    DepthAttachmentWrite,
    DepthRead,
    SampledRead,
    StorageRead,
    StorageWrite,
    TransferSrc,
    TransferDst,
    Present,
};

// A pass declares intent ("read as sampled", "write as storage"), and the graph
// derives the Vulkan barrier and layout transition needed between passes.
struct RenderGraphTextureAccess {
    GraphTextureHandle texture{};
    ResourceUsage usage{ ResourceUsage::Undefined };
};

struct RenderGraphPassNode {
    IRenderPass* pass{ nullptr };
    std::vector<RenderGraphTextureAccess> reads;
    std::vector<RenderGraphTextureAccess> writes;
};

struct CompiledBarrier {
    GraphTextureHandle resource{};
    ResourceUsage fromUsage{ ResourceUsage::Undefined };
    ResourceUsage toUsage{ ResourceUsage::Undefined };
    VkImageMemoryBarrier2 imageBarrier{};
};

class RenderGraphBuilder {
public:
    void Read(GraphTextureHandle texture, ResourceUsage usage);
    void Write(GraphTextureHandle texture, ResourceUsage usage);

private:
    friend class RenderGraph;
    RenderGraphPassNode* _passNode{ nullptr };
};

struct RenderGraphExecutionContext {
    RenderDevice& device;
    RendererFrameContext& frameContext;
    TransientImagePool& transientImagePool;
    VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
};

class RenderGraphContext {
public:
    RenderGraphContext(RenderDevice& device,
        const std::vector<ImageHandle>& resolvedImages,
        VkCommandBuffer commandBuffer,
        VkExtent2D renderExtent);

    [[nodiscard]] RenderDevice& GetDevice() const { return _device; }
    [[nodiscard]] VkCommandBuffer GetCommandBuffer() const { return _commandBuffer; }
    [[nodiscard]] VkImage GetTexture(GraphTextureHandle texture) const;
    [[nodiscard]] ImageHandle GetTextureHandle(GraphTextureHandle texture) const;
    [[nodiscard]] VkImageView GetTextureView(GraphTextureHandle texture) const;
    [[nodiscard]] VkFormat GetTextureFormat(GraphTextureHandle texture) const;
    [[nodiscard]] VkExtent3D GetTextureExtent(GraphTextureHandle texture) const;
    [[nodiscard]] VkExtent2D GetRenderExtent() const { return _renderExtent; }

private:
    RenderDevice& _device;
    const std::vector<ImageHandle>& _resolvedImages;
    VkCommandBuffer _commandBuffer{ VK_NULL_HANDLE };
    VkExtent2D _renderExtent{};
};

class RenderGraph {
public:
    // CreateTexture() declares a transient graph resource. ImportTexture() wraps
    // an externally owned image such as a swapchain image.
    GraphTextureHandle CreateTexture(std::string_view name, const ImageDesc& desc);
    GraphTextureHandle ImportTexture(std::string_view name, ImageHandle image, ResourceUsage initialUsage);
    void SetFinalUsage(GraphTextureHandle texture, ResourceUsage usage);

    void AddPass(IRenderPass& pass);
    void Compile(RenderDevice& device);
    void Execute(RenderGraphExecutionContext& executionContext);

private:
    struct TextureResource {
        std::string name;
        ImageDesc desc{};
        ImageHandle importedImage{};
        ResourceUsage initialUsage{ ResourceUsage::Undefined };
        bool imported{ false };
    };

    struct CompiledTextureResource {
        ImageDesc desc{};
        ImageHandle importedImage{};
        bool imported{ false };
    };

    struct CompiledPass {
        IRenderPass* pass{ nullptr };
        std::vector<CompiledBarrier> barriers;
    };

    struct FinalUsageRequest {
        GraphTextureHandle texture{};
        ResourceUsage usage{ ResourceUsage::Undefined };
    };

    struct TextureState {
        bool initialized{ false };
        ResourceUsage usage{ ResourceUsage::Undefined };
        VkPipelineStageFlags2 stageMask{ VK_PIPELINE_STAGE_2_NONE };
        VkAccessFlags2 accessMask{ VK_ACCESS_2_NONE };
        VkImageLayout layout{ VK_IMAGE_LAYOUT_UNDEFINED };
    };

    // Compile() converts high-level resource usage declarations into the exact
    // per-pass barriers that Vulkan expects before command recording.
    [[nodiscard]] static TextureState ResolveUsage(const TextureResource& resource, ResourceUsage usage);
    [[nodiscard]] static bool IsWriteUsage(ResourceUsage usage);

    std::vector<TextureResource> _textures;
    std::vector<CompiledTextureResource> _compiledResources;
    std::vector<RenderGraphPassNode> _passes;
    std::vector<CompiledPass> _compiledPasses;
    std::vector<FinalUsageRequest> _finalUsages;
    std::vector<CompiledBarrier> _finalBarriers;
    bool _compiled{ false };
};
} // namespace vesta::render
