#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <vesta/render/graph/render_graph.h>
#include <vesta/render/rhi/render_device.h>

struct SDL_Window;

namespace vesta::render {
struct RendererFrameContext {
    VkCommandPool commandPool{ VK_NULL_HANDLE };
    VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
    VkSemaphore acquireSemaphore{ VK_NULL_HANDLE };
    VkFence renderFence{ VK_NULL_HANDLE };
    std::vector<ImageHandle> acquiredTransientImages;
    std::vector<BufferHandle> transientBuffers;
};

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

    bool Initialize(SDL_Window* window, VkExtent2D initialExtent, bool enableValidation);
    void Shutdown();
    void RenderFrame();

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

    void InitializeCommands();
    void InitializeSyncStructures();
    void InitializeDefaultPasses();
    void DestroyFrameResources();
    void ReleaseTransientResources(RendererFrameContext& frameContext);
    void RecreateSwapchain();
    void ClearPassRegistry();
    void RebuildPassExecutionPlan();
    [[nodiscard]] RendererFrameContext& GetCurrentFrame();
    [[nodiscard]] RenderGraph BuildFrameGraph(uint32_t swapchainImageIndex);
    [[nodiscard]] RegisteredPassEntry* FindPassEntry(std::string_view id);
    [[nodiscard]] const RegisteredPassEntry* FindPassEntry(std::string_view id) const;

    RenderDevice _device;
    std::array<RendererFrameContext, kFrameOverlap> _frames{};
    std::vector<VkSemaphore> _swapchainImageRenderSemaphores;
    uint64_t _frameNumber{ 0 };
    std::vector<RegisteredPassEntry> _passRegistry;
    std::vector<RegisteredPassEntry*> _passExecutionPlan;
    bool _passExecutionPlanDirty{ true };
    TransientImagePool _transientImagePool;
    SDL_Window* _window{ nullptr };
};
} // namespace vesta::render
