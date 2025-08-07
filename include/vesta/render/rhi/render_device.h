#pragma once

#include <string>

#include <SDL_video.h>

#include <vesta/render/resources/resource_handles.h>
#include <vesta/render/vulkan/vk_types.h>

namespace vesta::render {
// 글로벌 bindless descriptor set을 관리한다.
// 현재는 sampled image / storage image / storage buffer용 슬롯만 가진다.
class BindlessDescriptorManager {
public:
    static constexpr uint32_t kMaxSampledImages = 1024;
    static constexpr uint32_t kMaxStorageImages = 1024;
    static constexpr uint32_t kMaxStorageBuffers = 1024;

    void Initialize(VkDevice device);
    void Shutdown(VkDevice device);

    [[nodiscard]] uint32_t RegisterSampledImage(VkDevice device, VkImageView view, VkImageLayout layout);
    [[nodiscard]] uint32_t RegisterStorageImage(VkDevice device, VkImageView view, VkImageLayout layout);
    [[nodiscard]] uint32_t RegisterStorageBuffer(VkDevice device, VkBuffer buffer, VkDeviceSize range);

    [[nodiscard]] VkDescriptorSetLayout GetLayout() const { return _layout; }
    [[nodiscard]] VkDescriptorSet GetSet() const { return _set; }

private:
    // bindless 전용 descriptor heap 상태다.
    VkDescriptorPool _pool{ VK_NULL_HANDLE };
    VkDescriptorSetLayout _layout{ VK_NULL_HANDLE };
    VkDescriptorSet _set{ VK_NULL_HANDLE };
    // 다음 등록 시 사용할 빈 슬롯 인덱스다.
    uint32_t _nextSampledImage{ 0 };
    uint32_t _nextStorageImage{ 0 };
    uint32_t _nextStorageBuffer{ 0 };
};

// RenderDevice 생성 시 필요한 최상위 설정이다.
struct RenderDeviceDesc {
    // 디버그 레이어와 GPU 툴에서 보이는 앱/엔진 이름이다.
    const char* appName{ "VestaEngine" };
    const char* engineName{ "VestaEngine" };
    // 초기 swapchain 크기다. 보통 창 크기와 맞춘다.
    VkExtent2D swapchainExtent{ 1700, 900 };
    // validation layer 활성화 여부다.
    bool enableValidation{ false };
};

// 버퍼 생성 요청을 설명하는 POD 설정이다.
struct BufferDesc {
    // 바이트 단위 크기다.
    VkDeviceSize size{ 0 };
    // vertex/index/storage/device-address 등 Vulkan usage 플래그다.
    VkBufferUsageFlags usage{ 0 };
    // VMA가 어떤 종류의 메모리를 선호할지 결정한다.
    VmaMemoryUsage memoryUsage{ VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    // mapped, host access, dedicated allocation 같은 세부 옵션이다.
    VmaAllocationCreateFlags allocationFlags{ 0 };
    // 생성 직후 bindless storage buffer 슬롯에 등록할지 여부다.
    bool registerBindlessStorage{ false };
    // RenderDoc, 로그, 디버깅용 이름이다.
    std::string debugName;
};

// 이미지 생성 요청을 설명하는 POD 설정이다.
struct ImageDesc {
    // 2D/3D 포함 Vulkan 이미지 실제 크기다.
    VkExtent3D extent{ 1, 1, 1 };
    // 색/깊이/중간 버퍼 포맷이다.
    VkFormat format{ VK_FORMAT_UNDEFINED };
    // color attachment, storage, sampled 같은 사용 목적이다.
    VkImageUsageFlags usage{ 0 };
    // color/depth/stencil 중 어떤 subresource를 view로 볼지 정의한다.
    VkImageAspectFlags aspectFlags{ VK_IMAGE_ASPECT_COLOR_BIT };
    // 생성 직후 그래프가 가정할 초기 레이아웃이다.
    VkImageLayout initialLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
    // VMA 메모리 선호 정책이다.
    VmaMemoryUsage memoryUsage{ VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE };
    // 후속 mip generation이나 array texture 확장을 위해 남겨 둔 값이다.
    uint32_t mipLevels{ 1 };
    uint32_t arrayLayers{ 1 };
    // 생성 직후 bindless sampled/storage 슬롯에 등록할지 여부다.
    bool registerBindlessSampled{ false };
    bool registerBindlessStorage{ false };
    // 디버깅용 이름이다.
    std::string debugName;
};

// 한 리소스가 bindless set 안에서 어떤 슬롯을 점유하는지 기록한다.
struct BindlessResourceIndices {
    uint32_t sampledImage{ kInvalidResourceIndex };
    uint32_t storageImage{ kInvalidResourceIndex };
    uint32_t storageBuffer{ kInvalidResourceIndex };
};

// 실제 VkBuffer와 VMA allocation을 함께 보관하는 내부 레코드다.
struct AllocatedBuffer {
    VkBuffer buffer{ VK_NULL_HANDLE };
    VmaAllocation allocation{ VK_NULL_HANDLE };
    VmaAllocationInfo allocationInfo{};
    // 원본 생성 설정을 같이 들고 있어 디버깅과 재구성에 쓴다.
    BufferDesc desc{};
    // bindless에 올라간 경우 슬롯 번호를 기억한다.
    BindlessResourceIndices bindless{};
};

// 실제 VkImage와 view, VMA allocation을 함께 보관하는 내부 레코드다.
struct AllocatedImage {
    VkImage image{ VK_NULL_HANDLE };
    // 기본 2D view다. 대부분 패스는 이 view를 바로 사용한다.
    VkImageView defaultView{ VK_NULL_HANDLE };
    VmaAllocation allocation{ VK_NULL_HANDLE };
    VmaAllocationInfo allocationInfo{};
    ImageDesc desc{};
    BindlessResourceIndices bindless{};
    // swapchain이 소유한 이미지면 VMA로 파괴하면 안 되므로 구분 플래그가 필요하다.
    bool ownedBySwapchain{ false };
};

// Vulkan instance/device/swapchain/VMA/bindless를 한 곳에서 관리하는 RHI 계층이다.
class RenderDevice {
public:
    bool Initialize(SDL_Window* window, const RenderDeviceDesc& desc);
    void Shutdown();
    void WaitIdle() const;

    void RecreateSwapchain(VkExtent2D extent);

    [[nodiscard]] BufferHandle CreateBuffer(const BufferDesc& desc);
    [[nodiscard]] ImageHandle CreateImage(const ImageDesc& desc);
    void DestroyBuffer(BufferHandle handle);
    void DestroyImage(ImageHandle handle);

    [[nodiscard]] VkBuffer GetBuffer(BufferHandle handle) const;
    [[nodiscard]] VkImage GetImage(ImageHandle handle) const;
    [[nodiscard]] VkImageView GetImageView(ImageHandle handle) const;
    [[nodiscard]] VkFormat GetImageFormat(ImageHandle handle) const;
    [[nodiscard]] VkExtent3D GetImageExtent(ImageHandle handle) const;
    [[nodiscard]] VkImageAspectFlags GetImageAspectFlags(ImageHandle handle) const;
    [[nodiscard]] VkDeviceAddress GetBufferDeviceAddress(BufferHandle handle) const;

    [[nodiscard]] const AllocatedImage& GetImageResource(ImageHandle handle) const;
    [[nodiscard]] const AllocatedBuffer& GetBufferResource(BufferHandle handle) const;

    [[nodiscard]] VkInstance GetInstance() const { return _instance; }
    [[nodiscard]] VkPhysicalDevice GetPhysicalDevice() const { return _physicalDevice; }
    [[nodiscard]] VkDevice GetDevice() const { return _device; }
    [[nodiscard]] VkSurfaceKHR GetSurface() const { return _surface; }
    [[nodiscard]] VkQueue GetGraphicsQueue() const { return _graphicsQueue; }
    [[nodiscard]] VkQueue GetPresentQueue() const { return _presentQueue; }
    [[nodiscard]] uint32_t GetGraphicsQueueFamily() const { return _graphicsQueueFamily; }
    [[nodiscard]] uint32_t GetPresentQueueFamily() const { return _presentQueueFamily; }
    [[nodiscard]] VkSwapchainKHR GetSwapchain() const { return _swapchain; }
    [[nodiscard]] VkFormat GetSwapchainFormat() const { return _swapchainImageFormat; }
    [[nodiscard]] VkExtent2D GetSwapchainExtent() const { return _swapchainExtent; }
    [[nodiscard]] ImageHandle GetSwapchainImageHandle(uint32_t imageIndex) const;
    [[nodiscard]] const std::vector<ImageHandle>& GetSwapchainImageHandles() const { return _swapchainImageHandles; }
    [[nodiscard]] BindlessDescriptorManager& GetBindless() { return _bindless; }

private:
    // Initialize 단계에서 호출되는 세부 생성 루틴이다.
    void CreateInstanceAndDevice(const RenderDeviceDesc& desc);
    void CreateAllocator();
    void CreateSwapchain(VkExtent2D extent);
    void DestroySwapchain();
    // 종료 시 남아 있는 일반 GPU 리소스를 정리한다.
    void CleanupResourceStorage();

    // 내부 벡터 기반 resource registry에서 빈 슬롯을 재사용한다.
    [[nodiscard]] BufferHandle AllocateBufferSlot();
    [[nodiscard]] ImageHandle AllocateImageSlot();

    // SDL 창은 surface 재생성과 drawable size 조회에 필요하다.
    SDL_Window* _window{ nullptr };

    // Vulkan 최상위 객체들이다.
    VkInstance _instance{ VK_NULL_HANDLE };
    VkDebugUtilsMessengerEXT _debugMessenger{ VK_NULL_HANDLE };
    VkPhysicalDevice _physicalDevice{ VK_NULL_HANDLE };
    VkDevice _device{ VK_NULL_HANDLE };
    VkSurfaceKHR _surface{ VK_NULL_HANDLE };

    // 현재는 graphics/present 큐만 관리한다.
    VkQueue _graphicsQueue{ VK_NULL_HANDLE };
    uint32_t _graphicsQueueFamily{ 0 };
    VkQueue _presentQueue{ VK_NULL_HANDLE };
    uint32_t _presentQueueFamily{ 0 };

    // 모든 버퍼/이미지 메모리 할당이 이 allocator를 지난다.
    VmaAllocator _allocator{ VK_NULL_HANDLE };

    // swapchain과 그 메타데이터다.
    VkSwapchainKHR _swapchain{ VK_NULL_HANDLE };
    VkFormat _swapchainImageFormat{ VK_FORMAT_UNDEFINED };
    VkExtent2D _swapchainExtent{};
    std::vector<ImageHandle> _swapchainImageHandles;

    // handle index로 접근하는 실제 자원 저장소다.
    std::vector<AllocatedBuffer> _buffers;
    std::vector<AllocatedImage> _images;
    // 파괴 후 재사용 가능한 빈 슬롯 목록이다.
    std::vector<uint32_t> _freeBufferSlots;
    std::vector<uint32_t> _freeImageSlots;

    BindlessDescriptorManager _bindless;
};
} // namespace vesta::render
