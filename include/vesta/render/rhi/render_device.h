#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include <SDL_video.h>

#include <vesta/render/resources/resource_handles.h>
#include <vesta/render/vulkan/vk_types.h>

namespace vesta::render {
// Central bindless heap used by the sample shaders. Instead of passing one
// descriptor set per texture/buffer, shaders index into large arrays.
class BindlessDescriptorManager {
public:
    static constexpr uint32_t kMaxSampledImages = 1024;
    static constexpr uint32_t kMaxStorageImages = 1024;
    static constexpr uint32_t kMaxStorageBuffers = 1024;

    void Initialize(VkDevice device, VkSampler defaultSampler);
    void Shutdown(VkDevice device);

    [[nodiscard]] uint32_t RegisterSampledImage(VkDevice device, VkImageView view, VkImageLayout layout);
    [[nodiscard]] uint32_t RegisterStorageImage(VkDevice device, VkImageView view, VkImageLayout layout);
    [[nodiscard]] uint32_t RegisterStorageBuffer(VkDevice device, VkBuffer buffer, VkDeviceSize range);

    [[nodiscard]] VkDescriptorSetLayout GetLayout() const { return _layout; }
    [[nodiscard]] VkDescriptorSet GetSet() const { return _set; }

private:
    VkDescriptorPool _pool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout _layout{ VK_NULL_HANDLE };
    VkDescriptorSet _set{ VK_NULL_HANDLE };
    VkSampler _defaultSampler{ VK_NULL_HANDLE };
    uint32_t _nextSampledImage{ 0 };
    uint32_t _nextStorageImage{ 0 };
    uint32_t _nextStorageBuffer{ 0 };
};

struct RenderDeviceDesc {
    const char* appName{ "VestaEngine" };
    const char* engineName{ "VestaEngine" };
    VkExtent2D swapchainExtent{ 1700, 900 };
    bool enableValidation{ false };
};

// BufferDesc and ImageDesc are small "creation recipes" that the renderer and
// graph use to request GPU resources without touching Vulkan structs directly.
struct BufferDesc {
    VkDeviceSize size{ 0 };
    VkBufferUsageFlags usage{ 0 };
    VmaMemoryUsage memoryUsage{ VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    VmaAllocationCreateFlags allocationFlags{ 0 };
    bool registerBindlessStorage{ false };
    std::string debugName;
};

struct ImageDesc {
    VkExtent3D extent{ 1, 1, 1 };
    VkFormat format{ VK_FORMAT_UNDEFINED };
    VkImageUsageFlags usage{ 0 };
    VkImageAspectFlags aspectFlags{ VK_IMAGE_ASPECT_COLOR_BIT };
    VkImageLayout initialLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
    VmaMemoryUsage memoryUsage{ VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    uint32_t mipLevels{ 1 };
    uint32_t arrayLayers{ 1 };
    bool registerBindlessSampled{ false };
    bool registerBindlessStorage{ false };
    std::string debugName;
};

struct BindlessResourceIndices {
    uint32_t sampledImage{ kInvalidResourceIndex };
    uint32_t storageImage{ kInvalidResourceIndex };
    uint32_t storageBuffer{ kInvalidResourceIndex };
};

// AllocatedBuffer / AllocatedImage store both the raw Vulkan object and the VMA
// allocation that owns its memory.
struct AllocatedBuffer {
    VkBuffer buffer{ VK_NULL_HANDLE };
    VmaAllocation allocation{ VK_NULL_HANDLE };
    VmaAllocationInfo allocationInfo{};
    BufferDesc desc{};
    BindlessResourceIndices bindless{};
};

struct AllocatedImage {
    VkImage image{ VK_NULL_HANDLE };
    VkImageView defaultView{ VK_NULL_HANDLE };
    VmaAllocation allocation{ VK_NULL_HANDLE };
    VmaAllocationInfo allocationInfo{};
    ImageDesc desc{};
    BindlessResourceIndices bindless{};
    bool ownedBySwapchain{ false };
};

struct RayTracingSupport {
    bool supported{ false };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR
    };
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
    };
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR
    };
};

// Function pointers are loaded only when ray tracing is available. Keeping them
// together makes it obvious which Vulkan entry points are optional extensions.
struct RayTracingFunctions {
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR{ nullptr };
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR{ nullptr };
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR{ nullptr };
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR{ nullptr };
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR{ nullptr };
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR{ nullptr };
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR{ nullptr };
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR{ nullptr };
};

struct UploadBatchStats {
    VkDeviceSize stagingCapacity{ 0 };
    VkDeviceSize pendingBytes{ 0 };
    VkDeviceSize lastSubmittedBytes{ 0 };
    VkDeviceSize totalSubmittedBytes{ 0 };
    uint32_t pendingCopies{ 0 };
};

class RenderDevice {
public:
    // RenderDevice owns the long-lived Vulkan objects: instance, device,
    // allocator, swapchain, and small resource registries for buffers/images.
    bool Initialize(SDL_Window* window, const RenderDeviceDesc& desc);
    void Shutdown();
    void WaitIdle() const;

    void RecreateSwapchain(VkExtent2D extent);

    [[nodiscard]] BufferHandle CreateBuffer(const BufferDesc& desc);
    [[nodiscard]] ImageHandle CreateImage(const ImageDesc& desc);
    void DestroyBuffer(BufferHandle handle);
    void DestroyImage(ImageHandle handle);
    void ImmediateSubmit(const std::function<void(VkCommandBuffer)>& recorder);
    void UploadBufferData(BufferHandle destination, VkDeviceSize destinationOffset, std::span<const std::byte> data);
    void UploadImageData(ImageHandle destination, std::span<const std::byte> data);
    void FlushUploadBatch();
    void SetDebugWaitContext(std::string_view context);

    [[nodiscard]] VkBuffer GetBuffer(BufferHandle handle) const;
    [[nodiscard]] VkImage GetImage(ImageHandle handle) const;
    [[nodiscard]] VkImageView GetImageView(ImageHandle handle) const;
    [[nodiscard]] VkFormat GetImageFormat(ImageHandle handle) const;
    [[nodiscard]] VkExtent3D GetImageExtent(ImageHandle handle) const;
    [[nodiscard]] VkImageAspectFlags GetImageAspectFlags(ImageHandle handle) const;
    [[nodiscard]] VkDeviceAddress GetBufferDeviceAddress(BufferHandle handle) const;

    [[nodiscard]] const AllocatedImage& GetImageResource(ImageHandle handle) const;
    [[nodiscard]] const AllocatedBuffer& GetBufferResource(BufferHandle handle) const;
    [[nodiscard]] bool IsRayTracingSupported() const { return _rayTracingSupport.supported; }
    [[nodiscard]] const RayTracingSupport& GetRayTracingSupport() const { return _rayTracingSupport; }
    [[nodiscard]] const RayTracingFunctions& GetRayTracingFunctions() const { return _rayTracingFunctions; }
    [[nodiscard]] const std::string& GetGpuName() const { return _gpuName; }
    [[nodiscard]] uint32_t GetDedicatedVideoMemoryMiB() const { return _dedicatedVideoMemoryMiB; }
    [[nodiscard]] const UploadBatchStats& GetUploadBatchStats() const { return _uploadBatchStats; }
    [[nodiscard]] bool HasTransferQueue() const { return _transferQueue != VK_NULL_HANDLE; }

    [[nodiscard]] VkInstance GetInstance() const { return _instance; }
    [[nodiscard]] VkPhysicalDevice GetPhysicalDevice() const { return _physicalDevice; }
    [[nodiscard]] VkDevice GetDevice() const { return _device; }
    [[nodiscard]] VkSurfaceKHR GetSurface() const { return _surface; }
    [[nodiscard]] VkQueue GetGraphicsQueue() const { return _graphicsQueue; }
    [[nodiscard]] VkQueue GetPresentQueue() const { return _presentQueue; }
    [[nodiscard]] uint32_t GetGraphicsQueueFamily() const { return _graphicsQueueFamily; }
    [[nodiscard]] uint32_t GetPresentQueueFamily() const { return _presentQueueFamily; }
    [[nodiscard]] VkQueue GetTransferQueue() const { return _transferQueue != VK_NULL_HANDLE ? _transferQueue : _graphicsQueue; }
    [[nodiscard]] uint32_t GetTransferQueueFamily() const
    {
        return _transferQueue != VK_NULL_HANDLE ? _transferQueueFamily : _graphicsQueueFamily;
    }
    [[nodiscard]] VkSwapchainKHR GetSwapchain() const { return _swapchain; }
    [[nodiscard]] VkFormat GetSwapchainFormat() const { return _swapchainImageFormat; }
    [[nodiscard]] VkExtent2D GetSwapchainExtent() const { return _swapchainExtent; }
    [[nodiscard]] ImageHandle GetSwapchainImageHandle(uint32_t imageIndex) const;
    [[nodiscard]] const std::vector<ImageHandle>& GetSwapchainImageHandles() const { return _swapchainImageHandles; }
    [[nodiscard]] BindlessDescriptorManager& GetBindless() { return _bindless; }

private:
    void CreateInstanceAndDevice(const RenderDeviceDesc& desc);
    void CreateAllocator();
    void CreateSwapchain(VkExtent2D extent);
    void DestroySwapchain();
    void CleanupResourceStorage();
    void InitializeImmediateContext();
    void ShutdownImmediateContext();
    void EnsureUploadCapacity(VkDeviceSize requiredBytes);
    void BeginUploadBatchRecording();
    void WaitForFenceOrAssert(VkFence fence, std::string_view waitLabel);

    [[nodiscard]] BufferHandle AllocateBufferSlot();
    [[nodiscard]] ImageHandle AllocateImageSlot();

    struct ImmediateContext {
        VkCommandPool commandPool{ VK_NULL_HANDLE };
        VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
        VkFence fence{ VK_NULL_HANDLE };
    };

    struct UploadContext {
        VkCommandPool commandPool{ VK_NULL_HANDLE };
        VkCommandBuffer commandBuffer{ VK_NULL_HANDLE };
        VkFence fence{ VK_NULL_HANDLE };
        BufferHandle stagingBuffer{};
        void* mappedData{ nullptr };
        VkDeviceSize capacity{ 0 };
        VkDeviceSize offset{ 0 };
        bool recording{ false };
        uint32_t pendingCopies{ 0 };
    };

    SDL_Window* _window{ nullptr };

    VkInstance _instance{ VK_NULL_HANDLE };
    VkDebugUtilsMessengerEXT _debugMessenger{ VK_NULL_HANDLE };
    VkPhysicalDevice _physicalDevice{ VK_NULL_HANDLE };
    VkDevice _device{ VK_NULL_HANDLE };
    VkSurfaceKHR _surface{ VK_NULL_HANDLE };

    VkQueue _graphicsQueue{ VK_NULL_HANDLE };
    uint32_t _graphicsQueueFamily{ 0 };
    VkQueue _presentQueue{ VK_NULL_HANDLE };
    uint32_t _presentQueueFamily{ 0 };
    VkQueue _transferQueue{ VK_NULL_HANDLE };
    uint32_t _transferQueueFamily{ 0 };

    VmaAllocator _allocator{ VK_NULL_HANDLE };
    VkSampler _defaultSampler{ VK_NULL_HANDLE };

    VkSwapchainKHR _swapchain{ VK_NULL_HANDLE };
    VkFormat _swapchainImageFormat{ VK_FORMAT_UNDEFINED };
    VkExtent2D _swapchainExtent{};
    std::vector<ImageHandle> _swapchainImageHandles;

    std::vector<AllocatedBuffer> _buffers;
    std::vector<AllocatedImage> _images;
    std::vector<uint32_t> _freeBufferSlots;
    std::vector<uint32_t> _freeImageSlots;

    BindlessDescriptorManager _bindless;
    RayTracingSupport _rayTracingSupport;
    RayTracingFunctions _rayTracingFunctions;
    mutable ImmediateContext _immediateContext;
    UploadContext _uploadContext;
    UploadBatchStats _uploadBatchStats;
    std::string _debugWaitContext;
    std::string _gpuName;
    uint32_t _dedicatedVideoMemoryMiB{ 0 };
};
} // namespace vesta::render
